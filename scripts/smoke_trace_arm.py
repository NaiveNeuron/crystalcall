# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "tokenizers>=0.20", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""GPU-free end-to-end exercise of the *production* arm path, `run_trace_arm`.

This is the highest-risk surface in the project: the held-out matrix pushes
thousands of expensive model turns through it. Here it is driven by a stub that
impersonates the instrumented vLLM server on both of its channels at once:

* the **authoritative** channel - an OpenAI-compatible SSE stream that emits a
  native `eval` tool call carrying a Python PTC program, then a closing prose
  turn, as DiffusionGemma does under `--tool-call-parser gemma4`;
* the **private revision channel** - NDJSON `crystalcall-revision/2` events
  pushed to the harness's own `RevisionSink` while the request is still open,
  carrying real DiffusionGemma token ids, so the decoder, the parser, the
  canonicaliser, the launch gate and the future table all run for real.

Proved without a GPU: arm registry resolution, the frozen-threshold guard,
OMP-Lite launch, sink binding on the agreed port, revision-driven parse + gate +
launch, the exact-key claim from the authoritative `host_tool_call`, deferred
claim verification, the append-only hash-chained shard, the trajectory join
across differing server/parent request ids, and recovery of the crystallization
milestones by `load_trajectories` -> `analyze_trajectory`.

The provisional canvases deliberately carry a wrong argument for the first few
revisions, so mutation, discard and wasted work are exercised too.

Run:  uv run scripts/smoke_trace_arm.py [--arm crystalcall] [--prompts 4]
Exit: 0 on success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crystalcall.harness import (  # noqa: E402
    ARMS,
    TraceRunConfig,
    load_tokenizer,
    parse_sink_endpoint,
    resolve_model_dir,
    run_trace_arm,
)
from crystalcall.manifest import load_manifest, registry_for  # noqa: E402
from crystalcall.parse_ptc import PtcExtractor  # noqa: E402
from crystalcall.replay import analyze_trajectory, build_views, load_trajectories  # noqa: E402
from crystalcall.tokens import CanvasDecoder  # noqa: E402
from crystalcall.trace import TraceReader  # noqa: E402

CANVAS = 256


class _Tok:
    """The real DiffusionGemma tokenizer, encode side."""

    def __init__(self) -> None:
        from tokenizers import Tokenizer

        d = resolve_model_dir("google/diffusiongemma-26B-A4B-it")
        self._t = Tokenizer.from_file(str(d / "tokenizer.json"))

    def encode_ids(self, text: str) -> list[int]:
        return list(self._t.encode(text, add_special_tokens=False).ids)


def encode_canvas(tok: _Tok, text: str, width: int = CANVAS) -> tuple[int, ...]:
    ids = tok.encode_ids(text)
    pad = ids[-1] if ids else 0
    return tuple(ids[:width] + [pad] * max(0, width - len(ids)))


class RevisionEmitter:
    """Host-side stand-in for `patches/crystalcall_trace.py`."""

    def __init__(self, endpoint: str) -> None:
        self.host, self.port = parse_sink_endpoint(endpoint)
        self.sent = 0
        self.errors = 0

    def emit_request(
        self,
        request_id: str,
        canvases: list[tuple[str, bool]],
        *,
        step_ms: float,
        tok: _Tok,
        prompt_tokens: int,
    ) -> None:
        try:
            sock = socket.create_connection((self.host, self.port), timeout=5)
        except OSError:
            self.errors += 1
            return

        def send(obj: dict[str, Any]) -> None:
            sock.sendall((json.dumps(obj) + "\n").encode())
            self.sent += 1

        try:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            send(
                {
                    "kind": "request_begin",
                    "request_id": request_id,
                    "monotonic_ns": time.monotonic_ns(),
                    "prompt_tokens": prompt_tokens,
                    "sampler": {
                        "canvas_length": CANVAS,
                        "max_denoising_steps": 48,
                        "entropy_bound": 0.1,
                        "stub": True,
                    },
                }
            )
            n = len(canvases)
            for step, (text, settled) in enumerate(canvases):
                accepted = step >= n - 2
                send(
                    {
                        "kind": "revision",
                        "request_id": request_id,
                        "canvas_id": 0,
                        "step": step,
                        "monotonic_ns": time.monotonic_ns(),
                        "predicted_token_ids": list(encode_canvas(tok, text)),
                        "accepted_mask": [1 if accepted else 0] * CANVAS,
                        "token_confidence": [0.995 if settled else 0.55] * CANVAS,
                        "mean_entropy": 0.04 if settled else 0.5,
                        "converged": step == n - 1,
                        "phase": "commit" if step == n - 1 else "denoise",
                        "prefix_len": 0,
                    }
                )
                time.sleep(step_ms / 1000.0)
            now = time.monotonic_ns()
            send(
                {
                    "kind": "canvas_done",
                    "request_id": request_id,
                    "canvas_id": 0,
                    "monotonic_ns": now,
                }
            )
            send(
                {
                    "kind": "request_end",
                    "request_id": request_id,
                    "monotonic_ns": now,
                    "output_tokens": CANVAS,
                    "finish_reason": "stop",
                }
            )
        finally:
            sock.close()


class StubServer:
    """Serves both channels. Revisions are pushed before the answer, as the
    sampler does, so the gate sees the canvases while the request is open."""

    def __init__(
        self,
        *,
        port: int,
        emitter: RevisionEmitter,
        tok: _Tok,
        steps: int,
        settle_at: int,
        step_ms: float,
        model_id: str,
        programs: list[tuple[str, str]],
    ) -> None:
        self.port = port
        self.emitter = emitter
        self.tok = tok
        self.steps = steps
        self.settle_at = settle_at
        self.step_ms = step_ms
        self.model_id = model_id
        self.programs = programs
        self.turns = 0
        self.program_turns = 0
        self.lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None

    def _canvases(self, final: str, mutated: str) -> list[tuple[str, bool]]:
        # The canvas must carry DiffusionGemma's NATIVE tool-call wire format,
        # not a fenced block. A fenced stub would let this GPU-free smoke pass
        # even if the extractor found zero candidates in the real format.
        def envelope(code: str) -> str:
            return (
                "<|channel>thought\n<channel|><|tool_call>call:eval{"
                'language:<|"|>py<|"|>, code:<|"|>' + code + '<|"|>}<tool_call|>'
            )

        fin = envelope(final)
        early = envelope(mutated)
        return [
            (fin if i >= self.settle_at else early, i >= self.settle_at)
            for i in range(self.steps)
        ]

    def start(self) -> None:
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_a: Any) -> None:
                return

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    body = json.dumps(
                        {
                            "object": "list",
                            "data": [
                                {"id": outer.model_id, "object": "model", "max_model_len": 32768}
                            ],
                        }
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                n = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(n))
                except json.JSONDecodeError:
                    self.send_error(400)
                    return
                msgs = req.get("messages") or []
                # A request whose history already contains a tool result (or an
                # assistant tool-call turn) is the closing-prose turn.
                is_prose = any(
                    isinstance(m, dict)
                    and (m.get("role") == "tool" or m.get("tool_calls"))
                    for m in msgs
                )
                with outer.lock:
                    outer.turns += 1
                    rid = f"chatcmpl-stub-{outer.turns:04d}"
                    if is_prose:
                        final = mutated = ""
                    else:
                        idx = min(outer.program_turns, len(outer.programs) - 1)
                        final, mutated = outer.programs[idx]
                        outer.program_turns += 1

                if not is_prose:
                    outer.emitter.emit_request(
                        rid,
                        outer._canvases(final, mutated),
                        step_ms=outer.step_ms,
                        tok=outer.tok,
                        prompt_tokens=len(msgs) * 64,
                    )
                    args = json.dumps({"language": "py", "code": final})
                    pieces: list[dict[str, Any]] = [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_stub_0",
                                    "type": "function",
                                    "function": {"name": "eval", "arguments": args},
                                }
                            ],
                        }
                    ]
                    finish = "tool_calls"
                else:
                    pieces = [{"content": "Ran "}, {"content": "the program."}]
                    finish = "stop"

                base = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": outer.model_id,
                }
                chunks = [
                    {**base, "choices": [{"index": 0, "delta": d, "finish_reason": None}]}
                    for d in pieces
                ]
                chunks.append(
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]}
                )
                chunks.append(
                    {
                        **base,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 100,
                            "completion_tokens": CANVAS,
                            "total_tokens": 100 + CANVAS,
                        },
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for c in chunks:
                    payload = b"data: " + json.dumps(c).encode() + b"\n\n"
                    self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                    self.wfile.flush()
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), Handler)
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()


def programs_for(record: Any) -> tuple[str, str]:
    """The authoritative program, plus a plausible earlier (wrong) revision."""
    lines: list[str] = []
    for i, call in enumerate(record.expected_calls):
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(call["args"].items()))
        lines.append(f"r{i} = tool.{call['tool']}({args})")
    tail = "display({" + ", ".join(f'"r{i}": r{i}' for i in range(len(lines))) + "})"
    final = "\n".join([*lines, tail]) + "\n"

    mutated = final
    for k, v in sorted(record.expected_calls[0]["args"].items()):
        if isinstance(v, str) and v:
            mutated = final.replace(f"{k}={v!r}", f"{k}={(v + 'ZZ')!r}", 1)
            break
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            mutated = final.replace(f"{k}={v!r}", f"{k}={type(v)(v + 1)!r}", 1)
            break
    return final, mutated


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="crystalcall", choices=sorted(ARMS))
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--settle-at", type=int, default=4)
    ap.add_argument("--step-ms", type=float, default=12.0)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--sink", default="127.0.0.1:9801")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    spec = ARMS[args.arm]
    print(f"arm         {args.arm}  model={spec.model}  instr={spec.instrumentation}")

    work = Path(tempfile.mkdtemp(prefix="crystalcall-smoke-arm-"))
    failures: list[str] = []
    tok = _Tok()
    emitter = RevisionEmitter(args.sink)
    records = load_manifest(ROOT / "experiments/manifests/dev.jsonl")[: args.prompts]
    programs = [programs_for(r) for r in records]
    server = StubServer(
        port=args.port,
        emitter=emitter,
        tok=tok,
        steps=args.steps,
        settle_at=args.settle_at,
        step_ms=args.step_ms,
        model_id=spec.model.split("/", 1)[1],
        programs=programs,
    )

    # A frozen policy is mandatory for a speculating arm; supply a provisional
    # one through the env override so nothing is written into configs/.
    frozen = work / "frozen_policy.json"
    frozen.write_text(
        json.dumps(
            {
                "schema_version": "crystalcall-policy/1",
                "selected_on": "PROVISIONAL-SMOKE-ONLY",
                "gate": {"k": 2, "tau": 0.9, "min_overlap_ms": 0.0, "confidence_mode": "min"},
            },
            indent=2,
        )
    )
    os.environ["CRYSTALCALL_FROZEN_POLICY"] = str(frozen)

    server.start()
    try:
        cfg = TraceRunConfig(
            root=ROOT,
            manifest=ROOT / "experiments/manifests/dev.jsonl",
            arm=args.arm,
            seeds=(0,),
            out_dir=work / "raw",
            limit=args.prompts,
            server_url=f"http://127.0.0.1:{args.port}/v1",
            trace_sink_endpoint=args.sink,
        )
        summary = run_trace_arm(cfg)
    finally:
        server.stop()
        os.environ.pop("CRYSTALCALL_FROZEN_POLICY", None)

    print(
        f"run         prompts={summary['prompts']} turns={summary['turns']} "
        f"failures={summary['failures']} claimed={summary.get('claimed')} "
        f"verified={summary.get('verified_match')} "
        f"divergence={summary.get('semantic_divergence')} "
        f"verr={summary.get('verification_error')}"
    )
    print(
        f"policy      {summary['policy']['name']} k={summary['policy']['k']} "
        f"tau={summary['policy']['tau']} frozen={summary['frozen_policy']['status']}"
    )
    print(
        f"stub        model_requests={server.turns} program_turns={server.program_turns} "
        f"revision_events={emitter.sent} sink_errors={emitter.errors}"
    )

    shard = Path(summary["shards"][0]["path"])
    verify = TraceReader(shard).verify()
    print(f"shard       {shard.name} records={verify['records']} chain={verify['final_chain'][:16]}")

    trajectories = load_trajectories([shard], verify=True)
    decoder = CanvasDecoder(
        tokenizer=load_tokenizer(resolve_model_dir("google/diffusiongemma-26B-A4B-it"))
    )
    by_prompt = {r.prompt_id: r for r in records}
    total_calls = 0
    with_lead = 0
    for traj in trajectories:
        rec = by_prompt.get(traj.prompt_id)
        if rec is None:
            failures.append(f"{traj.prompt_id}: not in the manifest slice")
            continue
        reg = registry_for(rec)
        views = build_views(traj, PtcExtractor(registry=reg), decoder)
        an = analyze_trajectory(traj, views, reg)
        total_calls += len(an.final_calls)
        leads = [c.lead_ms() for c in an.final_calls if c.lead_ms() is not None]
        with_lead += len(leads)
        print(
            f"  {traj.prompt_id:<22} revs={len(traj.revisions):>3} "
            f"calls={len(an.final_calls)} correct={traj.benchmark_correct} "
            f"claims={traj.claims.get('claimed')} obs={traj.observed.get('launched')} "
            f"lead_ms={[round(x, 1) for x in leads]}"
        )

    # The stub must emit DiffusionGemma's NATIVE envelope. A fenced block is
    # also parsed by the extractor, so this GPU-free path would report healthy
    # leads even if the real format yielded zero candidates. Assert it on the
    # recorded shard, so a future stub edit cannot silently revert.
    from crystalcall.parse_ptc import locate_code_regions

    envelopes = 0
    gemma_regions = 0
    for traj in trajectories:
        for rev in traj.revisions:
            text = decoder.decode(rev.predicted_token_ids).text
            if 'call:eval{' in text and 'code:<|"|>' in text:
                envelopes += 1
                if any(r.locator == "gemma_tool_call" for r in locate_code_regions(text)):
                    gemma_regions += 1
    print(f"envelope    revisions with the native payload={envelopes} parsed as gemma={gemma_regions}")
    if envelopes == 0:
        failures.append(
            "no recorded revision carries the native `call:eval{code:<|\"|>` payload: "
            "the stub is emitting a canvas easier to parse than production"
        )
    elif gemma_regions != envelopes:
        failures.append(
            f"{envelopes - gemma_regions} native payloads produced no gemma_tool_call region"
        )

    if emitter.errors:
        failures.append(f"{emitter.errors} revision batches could not reach the sink")
    if summary["failures"]:
        failures.append(f"{summary['failures']} turns failed")
    if server.program_turns < args.prompts:
        failures.append(f"only {server.program_turns} program turns served")
    if not trajectories:
        failures.append("no trajectory recovered from the shard")
    if any(len(t.revisions) == 0 for t in trajectories):
        failures.append("a trajectory recovered zero revisions: the id join is broken")
    if total_calls == 0:
        failures.append("no authoritative final call was analysed")
    if with_lead == 0:
        failures.append("no pre-commit lead computed: the milestone logic is not firing")
    if spec.instrumentation == "full":
        if not summary.get("claimed"):
            failures.append("the speculating arm claimed nothing")
        if summary.get("semantic_divergence"):
            failures.append(f"{summary['semantic_divergence']} semantic divergences")
        if summary.get("verification_error"):
            failures.append(f"{summary['verification_error']} verification errors")
        if summary.get("claimed") != summary.get("verified_match"):
            failures.append(
                f"claims not fully verified: {summary.get('claimed')} claimed vs "
                f"{summary.get('verified_match')} verified"
            )

    report = {
        "schema_version": "crystalcall-smoke-arm/1",
        "arm": args.arm,
        "summary": {k: v for k, v in summary.items() if k != "shards"},
        "shard": str(shard),
        "shard_verify": verify,
        "model_requests": server.turns,
        "program_turns": server.program_turns,
        "revision_events_sent": emitter.sent,
        "trajectories": len(trajectories),
        "final_calls": total_calls,
        "calls_with_lead": with_lead,
        "failures": failures,
        "ok": not failures,
    }
    out = ROOT / "reports/smoke_trace_arm.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report      {out}")

    if args.keep:
        print(f"workspace   kept at {work}")
    else:
        shutil.rmtree(work, ignore_errors=True)

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: production arm path verified end to end without a GPU")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
