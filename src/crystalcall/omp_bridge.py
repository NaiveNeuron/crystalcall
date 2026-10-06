"""OMP-Lite headless bridge.

OMP is the authoritative programmatic tool-calling runtime.  This module drives
the pinned ``omp`` binary in JSON-RPC/stdio mode and owns the *authoritative*
tool-dispatch path:

* benchmark tools are registered with ``set_host_tools``, so the model reaches
  them from inside OMP's Python PTC kernel via OMP's own ``tool.<name>(args)``
  loopback bridge;
* every resulting ``host_tool_call`` arrives here, is canonicalised with the
  one shared canonicaliser, is offered an exact-key claim against the
  quarantined future table, and otherwise executes normally;
* the ``host_tool_result`` returned to OMP is byte-identical in either case.

Nothing speculative is ever written into a frame sent back to OMP.
"""

from __future__ import annotations

import base64
import json
import os
import platform
import queue
import shutil
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonicalize import CanonicalCall, CanonicalizationError, canonicalize
from .effects import EffectRegistry
from .futures import FutureKey, FutureTable

__all__ = [
    "OmpRpcClient",
    "OmpRpcError",
    "HostToolRouter",
    "AuthoritativeCall",
    "TurnResult",
    "OmpLiteProfile",
]


class OmpRpcError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# profile / launch
# --------------------------------------------------------------------------- #

def pinned_omp_version(root: Path | None = None) -> str | None:
    """The `omp` build the corpus was produced with (`reports/doctor.json`)."""
    base = root or Path(__file__).resolve().parents[2]
    doctor = base / "reports/doctor.json"
    if not doctor.exists():
        return None
    try:
        pins = json.loads(doctor.read_text()).get("pins") or {}
    except json.JSONDecodeError:
        return None
    version = (pins.get("omp") or {}).get("version")
    return str(version).split("/")[-1] if version else None


def resolve_omp_binary(binary: str = "omp", root: Path | None = None) -> str:
    """Prefer the *vendored pinned* build over whatever is on `PATH`.

    The host `omp` can be upgraded out from under a frozen corpus, and the
    stateful arm's native tool-call path is not comparable across builds.
    Resolution order:

    1. `CRYSTALCALL_OMP_BIN` -- explicit override, always wins;
    2. `experiments/vendor/omp/<pinned>/omp-<platform>` -- the checksum-verified
       build named by `reports/doctor.json`;
    3. `PATH` -- whatever the host has, which may have drifted.

    Only the bare name `omp` is redirected; an explicit path is respected.
    """
    override = os.environ.get("CRYSTALCALL_OMP_BIN")
    if override:
        return override
    if binary != "omp":
        return shutil.which(binary) or binary
    base = root or Path(__file__).resolve().parents[2]
    version = pinned_omp_version(base)
    if version:
        machine = {"aarch64": "arm64", "arm64": "arm64", "x86_64": "x64"}.get(
            platform.machine(), platform.machine()
        )
        cand = base / f"experiments/vendor/omp/{version}/omp-linux-{machine}"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return shutil.which(binary) or binary



@dataclass(frozen=True, slots=True)
class OmpLiteProfile:
    """Everything needed to launch the frozen OMP-Lite configuration."""

    binary: str = "omp"
    profile: str = "crystalcall"
    config: Path | None = None
    system_prompt: Path | None = None
    cwd: Path | None = None
    model: str = "vllm/diffusiongemma-26b-a4b-it"
    thinking: str = "off"
    tools: tuple[str, ...] = ("eval",)
    extra_env: Mapping[str, str] = field(default_factory=dict)

    def argv(self) -> list[str]:
        exe = resolve_omp_binary(self.binary)
        argv = [
            exe,
            "--mode",
            "rpc",
            "--profile",
            self.profile,
            "--model",
            self.model,
            "--thinking",
            self.thinking,
            "--tools",
            ",".join(self.tools),
            "--no-extensions",
            "--no-skills",
            "--no-rules",
            "--no-lsp",
            "--no-pty",
            "--no-session",
            "--no-title",
            "--auto-approve",
        ]
        if self.config is not None:
            argv += ["--config", str(self.config)]
        if self.system_prompt is not None:
            argv += ["--system-prompt", str(self.system_prompt)]
        if self.cwd is not None:
            argv += ["--cwd", str(self.cwd)]
        return argv

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        # Remove every credential and dialect override so the harness cannot
        # accidentally reach a hosted provider or flip the tool wire format.
        for key in list(env):
            if key.endswith("_API_KEY") or key.endswith("_OAUTH_TOKEN"):
                env.pop(key, None)
        env.pop("PI_DIALECT", None)
        env.update(
            {
                "PI_NO_TITLE": "1",
                "PI_PY": "1",
                "PI_JS": "0",
                "PI_RB": "0",
                "PI_JL": "0",
                "OMP_PROFILE": self.profile,
            }
        )
        env.update(dict(self.extra_env))
        return env

    def fingerprint(self) -> str:
        import hashlib

        payload = {
            "argv": [a for a in self.argv() if not a.startswith("/")],
            "profile": self.profile,
            "model": self.model,
            "thinking": self.thinking,
            "tools": list(self.tools),
            "config_sha256": _sha256_file(self.config) if self.config else None,
            "system_prompt_sha256": (
                _sha256_file(self.system_prompt) if self.system_prompt else None
            ),
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()


def _sha256_file(path: Path | None) -> str | None:
    import hashlib

    if path is None or not Path(path).exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# authoritative tool calls
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class AuthoritativeCall:
    """One authoritative tool dispatch performed by OMP."""

    tool_call_id: str
    host_call_id: str
    tool_name: str
    raw_arguments: dict[str, Any]
    call: CanonicalCall | None
    request_ns: int
    finish_ns: int = 0
    claimed_speculative: bool = False
    error: str | None = None
    canonicalization_error: str | None = None
    result_digest: str = ""
    wall_ms: float = 0.0
    #: Set only when `HostToolRouter(verify_claims=True)` re-executed a claimed
    #: call normally: "match" | "divergent" | "error" | None (not verified).
    verification: str | None = None
    verification_detail: str = ""
    verification_ms: float = 0.0
    reference_digest: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "tool_call_id": self.tool_call_id,
            "host_call_id": self.host_call_id,
            "tool": self.call.tool if self.call else self.tool_name,
            "args": self.call.args_dict if self.call else self.raw_arguments,
            "snapshot_id": self.call.snapshot_id if self.call else None,
            "identity": self.call.identity if self.call else None,
            "request_ns": self.request_ns,
            "finish_ns": self.finish_ns,
            "wall_ms": self.wall_ms,
            "claimed_speculative": self.claimed_speculative,
            "error": self.error,
            "canonicalization_error": self.canonicalization_error,
            "result_digest": self.result_digest,
            "verification": self.verification,
            "verification_detail": self.verification_detail,
            "verification_ms": self.verification_ms,
            "reference_digest": self.reference_digest,
        }


ToolImpl = Callable[[Mapping[str, Any], threading.Event], Any]


class HostToolRouter:
    """Canonicalise, claim-or-execute, and log every authoritative call."""

    def __init__(
        self,
        registry: EffectRegistry,
        impls: Mapping[str, ToolImpl],
        *,
        futures: FutureTable | None = None,
        request_id_provider: Callable[[], str] | None = None,
        claim_timeout: float | None = None,
        #: After a successful claim, re-execute the same canonical call
        #: normally and compare result digests.  This is the direct, per-call
        #: measurement behind the "zero observed semantic divergence" gate: it
        #: holds the model fixed by construction, so a mismatch can only come
        #: from scheduling.  The verification runs *after* the result is
        #: returned to OMP and is timed separately, so it never enters the
        #: measured latency.
        verify_claims: bool = False,
    ) -> None:
        missing = sorted(set(registry.names()) - set(impls))
        if missing:
            raise ValueError(f"no implementation registered for tools: {missing}")
        self.registry = registry
        self.impls = dict(impls)
        self.futures = futures
        self.request_id_provider = request_id_provider or (lambda: "")
        self.claim_timeout = claim_timeout
        self.verify_claims = verify_claims
        self.log: list[AuthoritativeCall] = []
        #: Claimed calls awaiting verification.  Drained by
        #: `flush_verifications()` after the turn, never inside it.
        self._pending: list[tuple[AuthoritativeCall, CanonicalCall]] = []
        self._lock = threading.Lock()

    def handle(self, tool_name: str, arguments: Mapping[str, Any], *, host_call_id: str,
               tool_call_id: str) -> tuple[Any, str | None]:
        """Returns ``(value, error)``.  Never returns a quarantined object."""
        t0 = time.monotonic_ns()
        # OMP's Python bridge adds an intent field; it is not a tool parameter.
        args = {k: v for k, v in arguments.items() if k != "i"}
        rec = AuthoritativeCall(
            tool_call_id=tool_call_id,
            host_call_id=host_call_id,
            tool_name=tool_name,
            raw_arguments=dict(args),
            call=None,
            request_ns=t0,
        )
        try:
            call = canonicalize(self.registry, tool_name, args)
        except CanonicalizationError as exc:
            rec.canonicalization_error = f"{exc.reason}:{exc.detail}"
            rec.finish_ns = time.monotonic_ns()
            rec.wall_ms = (rec.finish_ns - t0) / 1e6
            with self._lock:
                self.log.append(rec)
            return (None, f"invalid call: {exc.reason} ({exc.detail})")
        rec.call = call

        value: Any = None
        error: str | None = None

        request_id = self.request_id_provider()
        if self.futures is not None and request_id and call.effect.speculation_eligible:
            key = FutureKey.of(request_id, call)
            claimed, claimed_value = self.futures.claim(key, timeout=self.claim_timeout)
            if claimed:
                rec.claimed_speculative = True
                value = claimed_value

        if not rec.claimed_speculative:
            impl = self.impls[call.tool]
            cancel = threading.Event()
            try:
                value = impl(call.args_dict, cancel)
            except Exception as exc:  # noqa: BLE001 - surfaced as a tool error
                error = f"{type(exc).__name__}: {exc}"

        rec.finish_ns = time.monotonic_ns()
        rec.wall_ms = (rec.finish_ns - t0) / 1e6
        rec.error = error
        if error is None:
            rec.result_digest = _digest_value(value)
        if rec.claimed_speculative and self.verify_claims and error is None:
            # Deferred on purpose.  Re-executing the tool here would run inside
            # the live OMP turn and would delay the model's next request for the
            # CrystalCall arm only, contaminating exactly the treatment whose
            # latency is being measured.  The check happens after the turn, via
            # `flush_verifications()`.
            with self._lock:
                self._pending.append((rec, call))
        with self._lock:
            self.log.append(rec)
        return (value, error)

    def flush_verifications(self) -> dict[str, int]:
        """Run every deferred claim verification.  Call AFTER the turn ends.

        Re-executes each claimed call normally and compares result digests.  The
        model is held fixed by construction, so a mismatch can only come from
        scheduling.  Returns the tally; the per-call verdicts land on the
        recorded :class:`AuthoritativeCall` objects.
        """
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        tally = {"verified": 0, "match": 0, "divergent": 0, "error": 0}
        for rec, call in pending:
            self._verify(rec, call)
            tally["verified"] += 1
            if rec.verification in tally:
                tally[rec.verification] += 1
        return tally

    @property
    def pending_verifications(self) -> int:
        with self._lock:
            return len(self._pending)

    def _verify(self, rec: AuthoritativeCall, call: CanonicalCall) -> None:
        """Re-execute a claimed call normally and compare result digests."""
        v0 = time.monotonic_ns()
        try:
            reference = self.impls[call.tool](call.args_dict, threading.Event())
        except Exception as exc:  # noqa: BLE001 - recorded, never raised out
            rec.verification = "error"
            rec.verification_detail = f"{type(exc).__name__}: {exc}"
        else:
            ref_digest = _digest_value(reference)
            rec.reference_digest = ref_digest
            if ref_digest == rec.result_digest:
                rec.verification = "match"
            else:
                rec.verification = "divergent"
                rec.verification_detail = f"{rec.result_digest[:16]} != {ref_digest[:16]}"
        rec.verification_ms = (time.monotonic_ns() - v0) / 1e6

    def calls(self) -> tuple[AuthoritativeCall, ...]:
        with self._lock:
            return tuple(self.log)

    def reset(self) -> None:
        with self._lock:
            self.log.clear()
            self._pending.clear()

    def host_tool_definitions(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for schema in self.registry.json_schemas():
            out.append(
                {
                    "name": schema["name"],
                    "label": schema["name"],
                    "description": schema["description"],
                    "loadMode": "essential",
                    "parameters": schema["parameters"],
                }
            )
        return out


def _digest_value(value: Any) -> str:
    import hashlib

    try:
        blob = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        blob = repr(value)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# RPC client
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class TurnResult:
    """One completed OMP turn."""

    prompt: str
    assistant_text: str
    started_ns: int
    ended_ns: int
    events: list[dict[str, Any]]
    tool_calls: list[AuthoritativeCall]
    eval_programs: list[str]
    error: str | None = None
    aborted: bool = False

    @property
    def wall_ms(self) -> float:
        return (self.ended_ns - self.started_ns) / 1e6


class OmpRpcClient:
    """Minimal, strict client for ``omp --mode rpc``."""

    def __init__(
        self,
        profile: OmpLiteProfile,
        router: HostToolRouter | None = None,
        *,
        event_hook: Callable[[Mapping[str, Any]], None] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self.profile = profile
        self.router = router
        self.event_hook = event_hook
        self.log_path = log_path
        self._proc: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._responses: dict[str, dict[str, Any]] = {}
        self._resp_lock = threading.Condition()
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._chunks: dict[str, list[bytes | None]] = {}
        self._seq = 0
        self._write_lock = threading.Lock()
        self._log_fh: Any = None
        self.protocol_version = 1
        self.ready: dict[str, Any] = {}

    # ---- lifecycle ------------------------------------------------------ #

    def start(self, timeout: float = 60.0) -> dict[str, Any]:
        argv = self.profile.argv()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = open(self.log_path, "a", encoding="utf-8")  # noqa: SIM115
        self._proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self.profile.env(),
            cwd=str(self.profile.cwd) if self.profile.cwd else None,
        )
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()
        deadline = time.monotonic() + timeout
        while not self.ready and time.monotonic() < deadline:
            time.sleep(0.01)
            if self._proc.poll() is not None:
                err = self._proc.stderr.read() if self._proc.stderr else ""
                raise OmpRpcError(f"omp exited during startup: {err[-2000:]}")
        if not self.ready:
            raise OmpRpcError("omp did not emit a ready frame")
        if 2 in self.ready.get("supportedProtocolVersions", []):
            resp = self.request("negotiate_protocol", protocolVersion=2)
            self.protocol_version = int(resp.get("data", {}).get("protocolVersion", 1))
        if self.router is not None:
            self.request("set_host_tools", tools=self.router.host_tool_definitions())
        return self.ready

    def close(self, timeout: float = 20.0) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin and not self._proc.stdin.closed:
                self._proc.stdin.close()
        except OSError:
            pass
        try:
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._proc.kill()
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None

    def __enter__(self) -> OmpRpcClient:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- framing -------------------------------------------------------- #

    def _send(self, frame: Mapping[str, Any]) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise OmpRpcError("client is not started")
        line = json.dumps(frame, ensure_ascii=False, separators=(",", ":"))
        with self._write_lock:
            if self._proc.stdin.closed:
                # The session is shutting down; dropping a frame here is
                # correct, and raising would only noise up the reader thread.
                return
            try:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            except (ValueError, BrokenPipeError, OSError):
                return
        self._log("send", frame)

    def _log(self, direction: str, frame: Mapping[str, Any]) -> None:
        if self._log_fh is None:
            return
        self._log_fh.write(
            json.dumps(
                {"t": time.monotonic_ns(), "dir": direction, "frame": frame},
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )

    def _next_id(self, prefix: str) -> str:
        self._seq += 1
        return f"{prefix}-{self._seq}"

    def request(self, command: str, *, timeout: float = 600.0, **payload: Any) -> dict[str, Any]:
        rid = self._next_id(command)
        self._send({"id": rid, "type": command, **payload})
        deadline = time.monotonic() + timeout
        with self._resp_lock:
            while rid not in self._responses:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise OmpRpcError(f"timeout waiting for response to {command}")
                self._resp_lock.wait(min(0.25, remaining))
            resp = self._responses.pop(rid)
        if not resp.get("success", False):
            raise OmpRpcError(f"{command} failed: {resp.get('error')} ({resp.get('code')})")
        return resp

    def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            try:
                frame = json.loads(line)
            except json.JSONDecodeError:
                continue
            self._log("recv", frame)
            self._dispatch(frame)

    def _dispatch(self, frame: dict[str, Any]) -> None:
        kind = frame.get("type")
        if kind == "rpc_chunk":
            merged = self._reassemble(frame)
            if merged is not None:
                self._dispatch(merged)
            return
        if kind == "ready":
            self.ready = frame
            return
        if kind == "response":
            rid = frame.get("id")
            if isinstance(rid, str):
                with self._resp_lock:
                    self._responses[rid] = frame
                    self._resp_lock.notify_all()
            return
        if kind == "host_tool_call":
            threading.Thread(target=self._run_host_tool, args=(frame,), daemon=True).start()
            return
        if kind in ("extension_ui_request", "host_uri_request"):
            # OMP-Lite loads no extensions; answer defensively so nothing stalls.
            rid = frame.get("id")
            if isinstance(rid, str):
                key = "extension_ui_response" if kind == "extension_ui_request" else "host_uri_result"
                self._send({"type": key, "id": rid, "error": "unsupported in OMP-Lite"})
            return
        frame.setdefault("_recv_ns", time.monotonic_ns())
        self._events.put(frame)
        if self.event_hook is not None:
            try:
                self.event_hook(frame)
            except Exception:  # noqa: BLE001 - hooks must never break the loop
                pass

    def _reassemble(self, frame: dict[str, Any]) -> dict[str, Any] | None:
        cid = str(frame.get("chunkId"))
        count = int(frame.get("count", 0))
        index = int(frame.get("index", 0))
        buf = self._chunks.setdefault(cid, [None] * count)
        if len(buf) != count:
            self._chunks.pop(cid, None)
            return None
        buf[index] = base64.b64decode(frame.get("data", ""))
        if any(b is None for b in buf):
            return None
        self._chunks.pop(cid, None)
        blob = b"".join(b for b in buf if b is not None)
        expect = frame.get("byteLength")
        if expect is not None and len(blob) != int(expect):
            return None
        try:
            return json.loads(blob.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    # ---- host tools ----------------------------------------------------- #

    def _run_host_tool(self, frame: dict[str, Any]) -> None:
        host_id = str(frame.get("id"))
        name = str(frame.get("toolName"))
        args = frame.get("arguments") or {}
        tool_call_id = str(frame.get("toolCallId", ""))
        if self.router is None:
            self._send(
                {
                    "type": "host_tool_result",
                    "id": host_id,
                    "isError": True,
                    "result": {"content": [{"type": "text", "text": "no host tool router"}]},
                }
            )
            return
        value, error = self.router.handle(
            name, args, host_call_id=host_id, tool_call_id=tool_call_id
        )
        if error is not None:
            self._send(
                {
                    "type": "host_tool_result",
                    "id": host_id,
                    "isError": True,
                    "result": {"content": [{"type": "text", "text": error}]},
                }
            )
            return
        text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
        self._send(
            {
                "type": "host_tool_result",
                "id": host_id,
                "result": {"content": [{"type": "text", "text": text}]},
            }
        )

    # ---- prompting ------------------------------------------------------ #

    def rebind(self, router: HostToolRouter) -> None:
        """Start a fresh session and re-register the host tool set.

        Each benchmark prompt gets its own tool surface and its own empty
        conversation, so no context or tool identity leaks between prompts.
        """
        self.router = router
        self.request("new_session")
        self.request("set_host_tools", tools=router.host_tool_definitions())
        self._drain_events()

    def get_state(self) -> dict[str, Any]:
        return self.request("get_state").get("data", {})

    def prompt(self, message: str, *, timeout: float = 3600.0) -> TurnResult:
        if self.router is not None:
            self.router.reset()
        self._drain_events()
        started = time.monotonic_ns()
        resp = self.request("prompt", message=message, timeout=120.0)
        data = resp.get("data") or {}
        events: list[dict[str, Any]] = []
        text_parts: list[str] = []
        eval_programs: list[str] = []
        error: str | None = None
        aborted = False

        if data.get("agentInvoked") is False:
            ended = time.monotonic_ns()
            return TurnResult(
                prompt=message,
                assistant_text="",
                started_ns=started,
                ended_ns=ended,
                events=events,
                tool_calls=list(self.router.calls()) if self.router else [],
                eval_programs=eval_programs,
                error="agent_not_invoked",
            )

        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                error = "turn_timeout"
                self.request("abort", timeout=30.0)
                aborted = True
                break
            try:
                ev = self._events.get(timeout=min(1.0, remaining))
            except queue.Empty:
                if self._proc is not None and self._proc.poll() is not None:
                    error = "omp_exited"
                    break
                continue
            events.append(ev)
            et = ev.get("type")
            if et == "message_update":
                ame = ev.get("assistantMessageEvent") or {}
                if ame.get("type") == "text_delta":
                    text_parts.append(str(ame.get("delta", "")))
            elif et == "tool_execution_start":
                prog = _extract_eval_program(ev)
                if prog:
                    eval_programs.append(prog)
            elif et == "agent_end":
                if ev.get("isTerminal") is not False:
                    break
        ended = time.monotonic_ns()
        assistant = "".join(text_parts)
        if not assistant:
            assistant = _assistant_text_from_agent_end(events)
        if not eval_programs:
            eval_programs = _eval_programs_from_events(events)
        return TurnResult(
            prompt=message,
            assistant_text=assistant,
            started_ns=started,
            ended_ns=ended,
            events=events,
            tool_calls=list(self.router.calls()) if self.router else [],
            eval_programs=eval_programs,
            error=error,
            aborted=aborted,
        )

    def _drain_events(self) -> None:
        while True:
            try:
                self._events.get_nowait()
            except queue.Empty:
                return


def _walk(obj: Any) -> Any:
    """Yield every mapping nested anywhere inside ``obj``."""
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, Mapping):
            yield cur
            stack.extend(cur.values())
        elif isinstance(cur, (list, tuple)):
            stack.extend(cur)


def _extract_eval_program(event: Mapping[str, Any]) -> str | None:
    for node in _walk(event):
        if node.get("language") in ("py", "python") and isinstance(node.get("code"), str):
            return node["code"]
        name = node.get("toolName") or node.get("name")
        if name == "eval":
            args = node.get("arguments") or node.get("args") or node.get("input")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = None
            if isinstance(args, Mapping) and isinstance(args.get("code"), str):
                return args["code"]
    return None


def _eval_programs_from_events(events: Sequence[Mapping[str, Any]]) -> list[str]:
    out: list[str] = []
    for ev in events:
        prog = _extract_eval_program(ev)
        if prog and prog not in out:
            out.append(prog)
    return out


def _assistant_text_from_agent_end(events: Sequence[Mapping[str, Any]]) -> str:
    for ev in reversed(events):
        if ev.get("type") != "agent_end":
            continue
        msgs = ev.get("messages") or []
        parts: list[str] = []
        for m in msgs:
            if not isinstance(m, Mapping) or m.get("role") != "assistant":
                continue
            content = m.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") == "text":
                        parts.append(str(block.get("text", "")))
        if parts:
            return "".join(parts)
    return ""
