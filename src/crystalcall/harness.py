"""Experiment harness: run one arm over a frozen manifest, or replay offline.

One model request per prompt turn, one in flight at a time, one fresh OMP
session per prompt.  Everything the analysis needs lands in an append-only
hash-chained trace shard.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import httpx

from .canonicalize import CanonicalizationError, canonicalize
from .doctor import collect as collect_doctor
from .effects import EffectRegistry, ResourceClass
from .futures import FutureTable
from .omp_bridge import HostToolRouter, OmpLiteProfile, OmpRpcClient, OmpRpcError, TurnResult
from .parse_ptc import PtcExtractor
from .ar import AR_POLICIES, assert_monotonic, build_ar_revisions, deltas_from_events
from .replay import (
    LATENCY_CONDITIONS_MS,
    BASELINE_POLICIES,
    PREACT_ARM,
    PREACT_POLICIES,
    Trajectory,
    analyze_trajectory,
    build_views,
    load_trajectories,
    simulate_policy,
    write_jsonl,
)
from .runtime import DenoisePtcRuntime, Instrumentation
from .scheduler import (
    POLICY_LIBRARY,
    ConfidenceMode,
    CostModel,
    GatePolicy,
    RemainingDecodeModel,
    policy_by_name,
)
from .sidechannel import RequestStream, RevisionSink
from .tokens import CanvasDecoder, OffsetTokenizerAdapter
from .trace import TraceWriter

__all__ = [
    "TraceRunConfig",
    "run_trace_arm",
    "run_replay",
    "run_overhead",
    "load_tokenizer",
    "score_bfcl",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def load_tokenizer(model_path: str | Path) -> Any:
    """Load the checkpoint's tokenizer.json without pulling in transformers."""
    from tokenizers import Tokenizer  # type: ignore[import-not-found]

    p = Path(model_path)
    tok_file = p / "tokenizer.json" if p.is_dir() else p
    tk = Tokenizer.from_file(str(tok_file))

    class _Adapter:
        def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
            return tk.decode(list(ids), skip_special_tokens=False)

        def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
            return [tk.id_to_token(int(i)) or "" for i in ids]

    return _Adapter()


def resolve_model_dir(repo_id: str) -> Path:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    snaps = sorted((folder / "snapshots").glob("*")) if (folder / "snapshots").exists() else []
    if not snaps:
        raise FileNotFoundError(f"no local snapshot for {repo_id} under {folder}")
    return snaps[-1]


def score_bfcl(
    registry: EffectRegistry,
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """AST-style BFCL correctness on canonical identities.

    A prediction is correct when the multiset of canonical identities issued by
    the authoritative program equals the multiset of expected identities.
    """

    def ids(rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
        good: list[str] = []
        bad: list[str] = []
        for r in rows:
            tool = r.get("tool")
            args = r.get("args") or {}
            if not isinstance(tool, str):
                bad.append("no_tool")
                continue
            try:
                good.append(canonicalize(registry, tool, args).identity)
            except CanonicalizationError as exc:
                bad.append(f"{tool}:{exc.reason}")
        return good, bad

    exp_ids, exp_bad = ids(expected)
    act_ids, act_bad = ids(actual)
    correct = sorted(exp_ids) == sorted(act_ids)
    return {
        "correct": correct,
        "expected_ids": sorted(exp_ids),
        "actual_ids": sorted(act_ids),
        "expected_invalid": exp_bad,
        "actual_invalid": act_bad,
        "n_expected": len(exp_ids),
        "n_actual": len(act_ids),
    }



# --------------------------------------------------------------------------- #
# side channel endpoint
# --------------------------------------------------------------------------- #


#: Fixed loopback endpoint for the private revision side channel.  The server
#: is launched with ``CRYSTALCALL_TRACE_SINK`` pointing here, so the harness
#: must bind exactly this port; ``scripts/overhead_ladder.sh`` and
#: ``scripts/run_all.sh`` use the same constant.
DEFAULT_TRACE_SINK = "127.0.0.1:9801"


def parse_sink_endpoint(endpoint: str | None) -> tuple[str, int]:
    """Resolve the side-channel endpoint: argument, env, then the default."""
    raw = endpoint or os.environ.get("CRYSTALCALL_TRACE_SINK") or DEFAULT_TRACE_SINK
    host, _, port = raw.rpartition(":")
    if not host or not port.isdigit():
        raise ValueError(f"bad trace sink endpoint {raw!r}; expected host:port")
    return host, int(port)


# --------------------------------------------------------------------------- #
# online arm
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ArmSpec:
    """Everything that distinguishes one online experimental arm.

    Only these fields may differ between arms.  The OMP-Lite settings overlay,
    the tool schemas, the canonicaliser, the tool implementations and the retry
    policy are identical by construction.
    """

    name: str
    model: str
    system_prompt: str
    instrumentation: str
    #: Autoregressive arms project a monotonic token stream into revision form
    #: instead of consuming the diffusion side channel.
    autoregressive: bool = False
    #: Server-side sampler environment this arm requires.  The P-ReAct-style
    #: baseline is a *decoding* change, not a prompt change: the prompt, the
    #: tool schemas and the executor are identical to every other arm, and the
    #: tool-call span is prioritised inside the sampler instead.  The harness
    #: does not start servers; it asserts the running server matches.
    sampler_env: tuple[tuple[str, str], ...] = ()
    description: str = ""


ARMS: dict[str, ArmSpec] = {
    "diffusion_standard": ArmSpec(
        name="diffusion_standard",
        model="vllm/diffusiongemma-26b-a4b-it",
        system_prompt="configs/system_prompt.md",
        instrumentation="parse",
        description="Unmodified decoding, trace + parse only. Source of every derived policy.",
    ),
    "crystalcall": ArmSpec(
        name="crystalcall",
        model="vllm/diffusiongemma-26b-a4b-it",
        system_prompt="configs/system_prompt.md",
        instrumentation="full",
        description="Unmodified decoding plus the frozen pre-commit launch gate.",
    ),
    "preact_style": ArmSpec(
        name="preact_style",
        model="vllm/diffusiongemma-26b-a4b-it",
        system_prompt="configs/system_prompt.md",
        instrumentation="parse",
        sampler_env=(("CRYSTALCALL_PREACT", "1"), ("CRYSTALCALL_PREACT_MODE", "both")),
        description=(
            "P-ReAct-style tool-priority decoding: identical prompt, model, tool "
            "schema and executor; the sampler prioritises the tool-call span so it "
            "commits early."
        ),
    ),
    "gemma4_ar": ArmSpec(
        name="gemma4_ar",
        model="vllm/gemma-4-26b-a4b-it",
        system_prompt="configs/system_prompt.md",
        instrumentation="off",
        autoregressive=True,
        description="Matched autoregressive control; feeds both ar_streaming and ar_sptc.",
    ),
}


@dataclass(slots=True)
class TraceRunConfig:
    root: Path
    manifest: Path
    arm: str
    seeds: tuple[int, ...] = (0,)
    #: ``None`` means "take it from :data:`ARMS`".  An explicit value overrides
    #: the arm registry and is logged; the two must never silently disagree.
    model: str | None = None
    out_dir: Path | None = None
    instrumentation: str | None = None
    policy: str = "full_gate"
    limit: int | None = None
    tool_sleep_ms: float = 0.0
    resume: bool = False
    server_url: str = "http://127.0.0.1:8000/v1"
    trace_sink_endpoint: str | None = None
    keep_parse_results: bool = False


def _server_ready(url: str, timeout: float = 5.0) -> bool:
    try:
        r = httpx.get(url.rstrip("/") + "/models", timeout=timeout)
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def load_frozen_gate(root: Path, base: GatePolicy) -> tuple[GatePolicy, dict[str, Any]]:
    """Apply the development-selected thresholds from `configs/frozen_policy.json`.

    `k`, `tau` and the minimum useful overlap are the only tuned quantities and
    are chosen by `crystalcall pilot` on the 100-prompt development set.  The
    library defaults exist so the gate is runnable before the pilot; a held-out
    arm that speculates MUST use the frozen values, or the experiment would be
    reporting an untuned policy under the frozen policy's name.
    """
    override = os.environ.get("CRYSTALCALL_FROZEN_POLICY", "").strip()
    path = Path(override) if override else root / "configs/frozen_policy.json"
    if not path.exists():
        return base, {"status": "absent", "path": str(path)}
    doc = json.loads(path.read_text())
    gate = doc.get("gate") or {}
    mode = str(gate.get("confidence_mode", base.confidence_mode.value))
    frozen = dc_replace(
        base,
        k=int(gate.get("k", base.k)),
        tau=float(gate.get("tau", base.tau)),
        min_overlap_ms=float(gate.get("min_overlap_ms", base.min_overlap_ms)),
        confidence_mode=ConfidenceMode(mode),
    )
    meta = {
        "status": "applied",
        "path": str(path),
        "sha256": _sha256(path),
        "selected_on": doc.get("selected_on"),
        "gate": gate,
    }
    return frozen, meta

def run_trace_arm(cfg: TraceRunConfig) -> dict[str, Any]:
    """Run one experimental arm over a manifest, writing an append-only shard."""
    from .manifest import load_manifest, registry_for
    from .tools import build_impls

    root = Path(cfg.root)

    # Pre-flight BEFORE the server is touched. Inside the generation loop, a
    # single malformed declaration -- one BFCL prompt whose tool the rules
    # classify as non-Pure -- would abort the run after the model had already
    # loaded and leave a truncated shard behind. Manifest and tool-surface
    # problems must cost seconds, not a model load.
    records = load_manifest(cfg.manifest)
    if cfg.limit:
        records = records[: cfg.limit]
    for rec in records:
        try:
            registry_for(rec)
            build_impls(rec.tools)
        except Exception as exc:  # noqa: BLE001 - re-raised with the prompt id
            raise ValueError(
                f"pre-flight failed for prompt {rec.prompt_id!r}: {type(exc).__name__}: {exc}"
            ) from exc

    if not _server_ready(cfg.server_url):
        raise RuntimeError(
            f"model server not ready at {cfg.server_url}; start it with `crystalcall serve`"
        )

    spec = ARMS.get(cfg.arm)
    if spec is None:
        raise KeyError(f"unknown arm {cfg.arm!r}; have {sorted(ARMS)}")
    # The arm registry is authoritative for model, prompt and instrumentation;
    # explicit CLI values may only override, never silently disagree.
    model = cfg.model or spec.model
    if cfg.model and cfg.model != spec.model:
        print(f"note: overriding arm model {spec.model} -> {cfg.model}", flush=True)
    instrumentation = Instrumentation(cfg.instrumentation or spec.instrumentation)
    system_prompt = root / spec.system_prompt
    if not system_prompt.exists():
        raise FileNotFoundError(f"arm {cfg.arm}: missing {system_prompt}")
    policy: GatePolicy = policy_by_name(cfg.policy)
    policy, frozen_meta = load_frozen_gate(root, policy)
    if instrumentation.launches and frozen_meta["status"] != "applied":
        raise RuntimeError(
            "arm "
            f"{cfg.arm!r} speculates but configs/frozen_policy.json is absent: "
            "run `crystalcall pilot` on the development manifest first so k, tau "
            "and the minimum useful overlap are selected on development data"
        )

    model_repo = (
        "google/diffusiongemma-26B-A4B-it"
        if "diffusion" in model
        else "google/gemma-4-26B-A4B-it"
    )
    tokenizer = load_tokenizer(resolve_model_dir(model_repo))
    decoder = CanvasDecoder(tokenizer=tokenizer)

    # The server is started BEFORE this process and was told where to push
    # revisions, so the sink must bind that exact port.  An ephemeral port
    # would silently receive nothing.
    sink_host, sink_port = parse_sink_endpoint(cfg.trace_sink_endpoint)
    sink = RevisionSink(host=sink_host, port=sink_port).start()
    if instrumentation is not Instrumentation.OFF and sink.port != sink_port:
        raise RuntimeError(
            f"revision sink bound {sink.port}, expected {sink_port}; the server pushes to "
            f"{sink_host}:{sink_port}"
        )
    futures = FutureTable(concurrency={rc: 2 for rc in ResourceClass}) if instrumentation.launches else None
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=decoder,
        policy=policy,
        costs=_load_cost_model(root),
        remaining_model=_load_remaining_model(root),
        instrumentation=instrumentation,
        futures=futures,
        # Judge overlap against the duration the injected tools really take;
        # `None` when no sleep is injected keeps the manifest's declaration.
        tool_duration_override_ms=(float(cfg.tool_sleep_ms) if cfg.tool_sleep_ms else None),
    )

    out_dir = Path(cfg.out_dir or root / "experiments/raw" / cfg.arm)
    out_dir.mkdir(parents=True, exist_ok=True)

    doctor_digest = _cached_doctor_digest(root)
    manifest_sha = _sha256(cfg.manifest)

    profile = OmpLiteProfile(
        profile="crystalcall",
        config=root / "configs/omp-lite.yml",
        system_prompt=system_prompt,
        model=model,
        thinking="off",
        tools=("eval",),
        cwd=root / "experiments/workspace",
        extra_env={"CRYSTALCALL_TOOL_SLEEP_MS": str(cfg.tool_sleep_ms)},
    )
    (root / "experiments/workspace").mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "arm": cfg.arm,
        "manifest": str(cfg.manifest),
        "manifest_sha256": manifest_sha,
        "policy": policy.to_json(),
        "policy_fingerprint": policy.fingerprint(),
        "frozen_policy": frozen_meta,
        "instrumentation": instrumentation.value,
        "seeds": list(cfg.seeds),
        "prompts": len(records),
        "doctor_digest": doctor_digest,
        "profile_fingerprint": profile.fingerprint(),
        "shards": [],
        "turns": 0,
        "failures": 0,
        "claimed": 0,
        "verified_match": 0,
        "semantic_divergence": 0,
        "verification_error": 0,
    }

    for seed in cfg.seeds:
        shard = out_dir / f"{cfg.arm}-seed{seed}.jsonl"
        # A shard is committed ATOMICALLY: written under `.partial` and renamed
        # only after `writer.close()` seals the footer, digest and sidecar.
        # Otherwise an aborted run leaves a truncated shard that `--resume`
        # counts as complete -- which is how a 100-prompt dataset silently
        # becomes a 72-record one.
        partial = shard.with_name(shard.name + ".partial")
        if shard.exists():
            if cfg.resume:
                summary["shards"].append({"path": str(shard), "status": "skipped_existing"})
                continue
            raise FileExistsError(f"{shard} exists; pass --resume to skip completed shards")
        if partial.exists():
            stale = sum(1 for _ in partial.open(encoding="utf-8", errors="replace"))
            summary.setdefault("discarded_partials", []).append(
                {"path": str(partial), "records": stale}
            )
            print(
                f"discarding partial shard {partial.name} ({stale} records) from an "
                f"aborted run; it was never committed",
                flush=True,
            )
            for victim in (partial, partial.with_name(partial.name + ".meta.json")):
                try:
                    victim.chmod(0o644)
                except OSError:
                    pass
                victim.unlink(missing_ok=True)
        writer = TraceWriter(
            partial,
            meta={
                "arm": cfg.arm,
                "seed": seed,
                "model": cfg.model,
                "manifest": str(cfg.manifest),
                "manifest_sha256": manifest_sha,
                "policy": policy.to_json(),
                "instrumentation": instrumentation.value,
                "frozen_policy": frozen_meta,
                "doctor_digest": doctor_digest,
                "profile_fingerprint": profile.fingerprint(),
                "tool_sleep_ms": cfg.tool_sleep_ms,
            },
        )
        client: OmpRpcClient | None = None
        try:
            for i, rec in enumerate(records):
                registry = registry_for(rec)
                impls = build_impls(rec.tools)
                request_id = f"{cfg.arm}/{rec.prompt_id}/s{seed}"
                router = HostToolRouter(
                    registry,
                    impls,
                    futures=futures,
                    request_id_provider=lambda rid=request_id: rid,
                    claim_timeout=120.0,
                    # Only an arm that actually speculates has claims to
                    # verify; this is the direct per-call measurement behind
                    # the zero-semantic-divergence gate.
                    verify_claims=instrumentation.launches,
                )
                if client is None:
                    client = OmpRpcClient(profile, router)
                    client.start(timeout=180)
                else:
                    client.rebind(router)

                marker = sink.marker()
                rt_holder: dict[str, Any] = {}

                def _bind(stream, _reg=registry, _impls=impls, _rid=request_id, _h=rt_holder):
                    if "rt" in _h:
                        return
                    _h["rt"] = runtime.begin(
                        _rid, _reg, _impls, stream, keep_parse_results=cfg.keep_parse_results
                    )
                    _h["stream"] = stream

                sink.set_new_stream_hook(_bind)
                t0 = time.monotonic_ns()
                try:
                    turn: TurnResult = client.prompt(rec.user_message, timeout=1800.0)
                    error = turn.error
                except OmpRpcError as exc:
                    turn = TurnResult(
                        prompt=rec.user_message,
                        assistant_text="",
                        started_ns=t0,
                        ended_ns=time.monotonic_ns(),
                        events=[],
                        tool_calls=[],
                        eval_programs=[],
                        error=f"rpc_error: {exc}",
                    )
                    error = turn.error
                finally:
                    sink.set_new_stream_hook(None)

                # The turn is over: only now may claimed results be
                # re-executed for verification, so the check cannot delay the
                # model's next request in the speculating arm.
                verification = router.flush_verifications()
                # The HTTP response and the private side channel are
                # asynchronous: snapshotting here without waiting persists only a
                # PREFIX of each request's revision stream, and the longer the
                # decode the worse the loss. Wait for `request_end` on every
                # stream this turn opened, then re-read so late-opening streams
                # are included.
                drain_timeout = float(
                    os.environ.get("CRYSTALCALL_TRACE_DRAIN_TIMEOUT_S", "30")
                )
                undrained: list[str] = []
                for _ in range(2):
                    for s in sink.streams_since(marker):
                        if not s.wait_closed(drain_timeout):
                            undrained.append(s.request_id)
                streams = sink.streams_since(marker)
                if undrained:
                    error = (error or "") + ";trace_stream_not_drained"
                rt = rt_holder.get("rt")
                report = runtime.end(rt, rt_holder.get("stream")) if rt is not None else {}

                ar_revisions = None
                ar_check = None
                if spec.autoregressive:
                    # No diffusion side channel exists for the AR control, so the
                    # monotonic token stream is projected into revision form and
                    # its prefix property is verified rather than assumed.  Only
                    # the program-producing model request is projected: the
                    # closing-prose request is a separate turn and is not part
                    # of the trajectory a launch policy observes.
                    cutoff = program_turn_end_ns(turn)
                    deltas = [
                        d
                        for d in deltas_from_events(turn.events)
                        if not d.monotonic_ns or d.monotonic_ns <= cutoff
                    ]
                    ar_revisions = build_ar_revisions(
                        deltas, request_id=request_id, tokenizer=tokenizer
                    )
                    ar_check = assert_monotonic(ar_revisions)
                    if not ar_check["monotonic"]:
                        error = (error or "") + ";ar_prefix_not_monotonic"

                _write_request(
                    writer,
                    rec=rec,
                    arm=cfg.arm,
                    seed=seed,
                    request_id=request_id,
                    streams=streams,
                    program_stream=rt_holder.get("stream"),
                    turn=turn,
                    router=router,
                    registry=registry,
                    runtime_report=report,
                    error=error,
                    ar_revisions=ar_revisions,
                    ar_check=ar_check,
                    verification=verification,
                )
                for s in streams:
                    sink.drop(s.request_id)
                summary["turns"] += max(1, len(streams))
                for c in router.calls():
                    if c.claimed_speculative:
                        summary["claimed"] += 1
                    if c.verification == "divergent":
                        summary["semantic_divergence"] += 1
                    elif c.verification == "match":
                        summary["verified_match"] += 1
                    elif c.verification == "error":
                        summary["verification_error"] += 1
                if error:
                    summary["failures"] += 1
                if (i + 1) % 10 == 0:
                    writer.flush()
                    print(
                        f"[{cfg.arm} seed{seed}] {i + 1}/{len(records)} "
                        f"turns={summary['turns']} failures={summary['failures']} "
                        f"claimed={summary['claimed']} "
                        f"divergence={summary['semantic_divergence']}",
                        flush=True,
                    )
            completed = True
        except BaseException:
            completed = False
            raise
        finally:
            if client is not None:
                client.close()
            meta = writer.close()
            if completed:
                # Commit: rename the sealed shard and its sidecar into place.
                # Only now can `--resume` see it, and only now is it a shard.
                meta_src = partial.with_name(partial.name + ".meta.json")
                partial.replace(shard)
                if meta_src.exists():
                    meta_src.replace(shard.with_name(shard.name + ".meta.json"))
                summary["shards"].append({"path": str(shard), **dict(meta)})
            else:
                # Left on disk under `.partial` as failure evidence; never
                # counted as a shard and never skipped by `--resume`.
                summary["shards"].append(
                    {"path": str(partial), "status": "uncommitted_partial", **dict(meta)}
                )

    if futures is not None:
        futures.close()
    runtime.close()
    sink.close()
    summary["runtime_costs"] = runtime.measured_costs().to_json()
    _write_json(out_dir / f"{cfg.arm}-run.json", summary)
    return summary


def program_turn_end_ns(turn: TurnResult) -> int:
    """When the program-producing model request finished.

    An OMP turn issues several model requests: one that emits the `eval` tool
    call carrying the program, then one more that writes the closing prose after
    the tool results come back.  The authoritative program executes at the end
    of the *first* one, so that -- not the end of the whole turn -- is the point
    a non-speculative baseline could start tool work.
    """
    for ev in turn.events:
        if ev.get("type") == "tool_execution_start":
            ns = int(ev.get("_recv_ns") or 0)
            if ns:
                return ns
    return turn.ended_ns


def _write_request(
    writer: TraceWriter,
    *,
    rec: Any,
    arm: str,
    seed: int,
    request_id: str,
    streams: Sequence[Any],
    program_stream: Any | None,
    turn: TurnResult,
    router: HostToolRouter,
    registry: EffectRegistry,
    runtime_report: Mapping[str, Any],
    error: str | None,
    ar_revisions: Sequence[Any] | None = None,
    ar_check: Mapping[str, Any] | None = None,
    verification: Mapping[str, int] | None = None,
) -> None:
    # The analysed trajectory is exactly the program-producing model request.
    # Later requests in the same OMP turn restart `canvas_id` and `step` at 0,
    # so merging them would collide canvases and let closing-prose revisions
    # perturb the program trajectory.  Their timing and identity are still
    # recorded, so nothing is hidden.
    prog = program_stream if program_stream is not None else (streams[0] if streams else None)
    start_ns = prog.begin_ns if prog and prog.begin_ns else turn.started_ns
    response_done_ns = (prog.end_ns if prog else 0) or program_turn_end_ns(turn)
    if ar_revisions:
        start_ns = ar_revisions[0].monotonic_ns or turn.started_ns
        response_done_ns = ar_revisions[-1].monotonic_ns
    if not response_done_ns:
        response_done_ns = turn.ended_ns

    writer.append(
        "request",
        {
            "request_id": request_id,
            "prompt_id": rec.prompt_id,
            "category": rec.category,
            "seed": seed,
            "arm": arm,
            "request_start_ns": start_ns,
            "meta": {
                "source": rec.source,
                "source_id": rec.source_id,
                "user_message": rec.user_message,
                "tools": [t.name for t in rec.tools],
                "registry_fingerprint": registry.fingerprint(),
                "program_server_request_id": prog.request_id if prog else None,
                "server_request_ids": [s.request_id for s in streams],
                "model_requests": [
                    {
                        "server_request_id": s.request_id,
                        "begin_ns": s.begin_ns,
                        "end_ns": s.end_ns,
                        "revisions": s.steps,
                        "prompt_tokens": s.prompt_tokens,
                        "output_tokens": s.output_tokens,
                        "finish_reason": s.finish_reason,
                        "is_program_turn": prog is not None and s.request_id == prog.request_id,
                    }
                    for s in streams
                ],
                "turn_end_ns": turn.ended_ns,
                "sampler": prog.sampler if prog else {},
            },
        },
    )
    # Sampler revisions arrive tagged with the *server's* request id, while the
    # shard's `request`/`final` records are keyed by the parent
    # `arm/prompt/seed` identity.  `load_trajectories` joins on that key, so the
    # revision must be re-tagged here or every revision would be dropped on
    # load.  The original server ids stay in `meta.model_requests`.
    if prog is not None:
        for rev in prog.snapshot():
            writer.revision(
                rev if rev.request_id == request_id else dc_replace(rev, request_id=request_id)
            )
        for canvas_id, ns in sorted(prog.canvas_done_ns.items()):
            writer.append(
                "canvas_done",
                {"request_id": request_id, "canvas_id": canvas_id, "monotonic_ns": ns},
            )
        # Readiness signals from a priority-decoding server, re-tagged onto the
        # same parent key as the revisions so `preact_ready` can join them.
        for ev in prog.ready:
            writer.append(
                "tool_call_ready",
                {
                    "request_id": request_id,
                    "canvas_id": int(ev.get("canvas_id", 0) or 0),
                    "step": int(ev.get("step", -1)),
                    "monotonic_ns": int(ev.get("monotonic_ns", 0) or 0),
                    "span_lo": int(ev.get("span_lo", -1)),
                    "span_hi": int(ev.get("span_hi", -1)),
                },
            )
    for rev in ar_revisions or ():
        writer.revision(rev)
    if ar_revisions:
        writer.append(
            "canvas_done",
            {
                "request_id": request_id,
                "canvas_id": 0,
                "monotonic_ns": ar_revisions[-1].monotonic_ns,
            },
        )

    auth = [c.to_json() for c in router.calls()]
    programs = list(turn.eval_programs)
    parse = PtcExtractor(registry=registry).parse_final_program(programs[0]) if programs else None
    score = score_bfcl(
        registry,
        rec.expected_calls,
        [{"tool": a["tool"], "args": a["args"]} for a in auth if a.get("tool")],
    )
    writer.append(
        "final",
        {
            "request_id": request_id,
            "final_text": turn.assistant_text,
            "final_program": programs[0] if programs else None,
            "all_programs": programs,
            "response_done_ns": response_done_ns,
            "turn_wall_ms": turn.wall_ms,
            "authoritative_calls": [
                {
                    "tool": a["tool"],
                    "args": a["args"],
                    "snapshot_id": a.get("snapshot_id"),
                    "identity": a.get("identity"),
                    "claimed_speculative": a.get("claimed_speculative"),
                    "wall_ms": a.get("wall_ms"),
                    "error": a.get("error"),
                    "canonicalization_error": a.get("canonicalization_error"),
                    "result_digest": a.get("result_digest"),
                    "request_ns": a.get("request_ns"),
                    "finish_ns": a.get("finish_ns"),
                    "verification": a.get("verification"),
                    "verification_detail": a.get("verification_detail"),
                    "verification_ms": a.get("verification_ms"),
                    "reference_digest": a.get("reference_digest"),
                }
                for a in auth
            ],
            "claims": {
                "claimed": sum(1 for a in auth if a.get("claimed_speculative")),
                "verified_match": sum(1 for a in auth if a.get("verification") == "match"),
                "verified_divergent": sum(
                    1 for a in auth if a.get("verification") == "divergent"
                ),
                "verification_error": sum(1 for a in auth if a.get("verification") == "error"),
                "deferred_flush": dict(verification) if verification else None,
            },
            "valid_final_parse": bool(parse and parse.ok and parse.candidates),
            "final_parse": parse.to_json() if parse else None,
            "benchmark_correct": score["correct"],
            "score": score,
            "runtime": runtime_report,
            "ar_prefix_check": dict(ar_check) if ar_check else None,
            "exclusion": error,
        },
    )


# --------------------------------------------------------------------------- #
# offline replay
# --------------------------------------------------------------------------- #


def run_replay(
    *,
    root: Path,
    shards: Sequence[Path],
    policies: Sequence[str] = ("all",),
    latencies: Sequence[float] = LATENCY_CONDITIONS_MS,
    out_dir: Path | None = None,
) -> dict[str, Any]:
    from .manifest import load_manifest, registry_for

    root = Path(root)
    out_dir = Path(out_dir or root / "experiments/derived")
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(shards, verify=True)
    if not trajectories:
        return {"status": "no_trajectories", "shards": [str(s) for s in shards]}

    # Registry per prompt comes from the frozen manifests.
    registries: dict[str, EffectRegistry] = {}
    for name in ("dev.jsonl", "test.jsonl"):
        p = root / "experiments/manifests" / name
        if not p.exists():
            continue
        for rec in load_manifest(p):
            registries[rec.prompt_id] = registry_for(rec)

    model_dir = resolve_model_dir("google/diffusiongemma-26B-A4B-it")
    decoder = CanvasDecoder(tokenizer=load_tokenizer(model_dir), cache_size=64)

    if list(policies) == ["all"]:
        gate_names = list(POLICY_LIBRARY)
        policy_list: list[str | GatePolicy] = [
            *BASELINE_POLICIES,
            *PREACT_POLICIES,
            *AR_POLICIES,
            *[POLICY_LIBRARY[n] for n in gate_names],
        ]
    else:
        policy_list = []
        for name in policies:
            if name in POLICY_LIBRARY:
                policy_list.append(POLICY_LIBRARY[name])
            else:
                policy_list.append(name)

    # Duplicate policy names would emit two identical rows per (trajectory,
    # latency) and silently inflate every sample count and confidence interval.
    names = [p if isinstance(p, str) else p.name for p in policy_list]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"duplicate policy names in the replay set: {dupes}")

    costs = _load_cost_model(root)
    remaining = _load_remaining_model(root)

    analyses: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for traj in trajectories:
        reg = registries.get(traj.prompt_id)
        if reg is None:
            skipped.append({"trajectory_id": traj.request_id, "reason": "no_registry"})
            continue
        views = build_views(traj, PtcExtractor(registry=reg), decoder)
        analysis = analyze_trajectory(traj, views, reg)
        analyses.append(analysis.to_json())
        # An arm only gets the policies that are meaningful for its model
        # family and its sampler: autoregressive policies presuppose a
        # monotonic prefix, canvas/gate policies presuppose revisable canvases,
        # and `preact_ready` presupposes a priority-decoding server.
        is_ar = "gemma4" in traj.arm or traj.arm.startswith("ar_")
        is_preact = traj.arm == PREACT_ARM
        for pol in policy_list:
            pname = pol if isinstance(pol, str) else pol.name
            if is_ar != (pname in AR_POLICIES):
                continue
            if (pname in PREACT_POLICIES) != is_preact:
                continue
            for lat in latencies:
                out = simulate_policy(
                    traj,
                    views,
                    analysis,
                    reg,
                    pol,
                    lat,
                    remaining_model=remaining,
                    costs=costs,
                )
                outcomes.append(out.to_json())

    n1 = write_jsonl(out_dir / "trajectory_analyses.jsonl", analyses)
    n2 = write_jsonl(out_dir / "policy_outcomes.jsonl", outcomes)
    # The gate ablation table is the declared ablation policies on the declared
    # ablation arm.  It is a view of the same outcomes, not a second experiment.
    spec_path = root / "configs/experiments/main_table.yaml"
    n_abl = 0
    if spec_path.exists():
        import yaml

        spec = yaml.safe_load(spec_path.read_text()) or {}
        abl_arm = spec.get("ablation_arm")
        abl_policies = set(spec.get("ablation_policies") or ())
        if abl_arm and abl_policies:
            n_abl = write_jsonl(
                out_dir / "gate_ablations.jsonl",
                [
                    o
                    for o in outcomes
                    if o.get("arm") == abl_arm and o.get("policy") in abl_policies
                ],
            )
    if skipped:
        write_jsonl(out_dir / "replay_skipped.jsonl", skipped)
    report = {
        "status": "ok",
        "shards": [str(s) for s in shards],
        "trajectories": len(trajectories),
        "analyses": n1,
        "outcomes": n2,
        "gate_ablation_rows": n_abl,
        "skipped": len(skipped),
        "policies": [p if isinstance(p, str) else p.name for p in policy_list],
        "latencies_ms": list(latencies),
        "decoder_fallbacks": decoder.fallbacks,
        "out_dir": str(out_dir),
    }
    _write_json(out_dir / "replay_run.json", report)
    return report


# --------------------------------------------------------------------------- #
# Phase 0 overhead ladder
# --------------------------------------------------------------------------- #


OVERHEAD_PROMPTS = [
    "Call get_weather for Paris.",
    "Look up the symbol handle_retry.",
    "Search the frozen repository for retry timeout.",
    "Add 17 and 25.",
    "Query the frozen ticket database for open tickets.",
]


OVERHEAD_LEVELS: tuple[str, ...] = ("off", "nopayload", "full", "parse")


def _start_server(
    root: Path, script: str, *, level: str, sink: str, timeout: float = 1200.0
) -> None:
    env = dict(os.environ)
    env["CRYSTALCALL_TRACE"] = "0" if level == "off" else "1"
    env["CRYSTALCALL_TRACE_LEVEL"] = "full" if level in ("full", "parse") else level
    env["CRYSTALCALL_TRACE_SINK"] = sink
    p = subprocess.run(  # noqa: S603
        ["bash", str(root / script)], cwd=root, env=env, timeout=timeout, check=False
    )
    if p.returncode != 0:
        raise RuntimeError(f"{script} failed with exit {p.returncode}")


def _stop_server(root: Path, script: str) -> None:
    subprocess.run(  # noqa: S603
        ["bash", str(root / script), "--stop"], cwd=root, check=False, timeout=300
    )


def foreign_gpu_holders(own_container: str = "cc-diffusiongemma") -> dict[str, Any]:
    """GPU compute processes that are not ours.

    An overhead rung is a latency measurement, so a foreign process on the same
    accelerator does not add noise -- it invents cost: an unrelated container
    reclaiming the GPU mid-ladder shows up as a large "instrumentation" cost.
    Detected and reported, never silently recorded.
    """
    out: dict[str, Any] = {"queried": False, "processes": [], "foreign_mib": 0}
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return out
    if proc.returncode != 0:
        return out
    out["queried"] = True

    # EVERY pid in our container, not just its init pid: vLLM runs EngineCore
    # and worker processes as separate host pids, and they are the ones holding
    # the ~40 GiB. Keying on `.State.Pid` alone would have classified our own
    # server as foreign on every rung and failed the whole ladder.
    ours: set[str] = set()
    try:
        top = subprocess.run(
            ["docker", "top", own_container, "-o", "pid"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        if top.returncode == 0:
            for line in (top.stdout or "").splitlines():
                tok = line.strip()
                if tok.isdigit():
                    ours.add(tok)
    except (OSError, subprocess.TimeoutExpired):
        pass
    if not ours:
        try:
            insp = subprocess.run(
                ["docker", "inspect", "--format", "{{.State.Pid}}", own_container],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            )
            if insp.returncode == 0 and insp.stdout.strip().isdigit():
                ours.add(insp.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            pass
    out["own_pids"] = sorted(ours, key=int)

    for line in (proc.stdout or "").splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2 or not parts[0].isdigit():
            continue
        pid, mib = parts[0], parts[1]
        name = parts[2] if len(parts) > 2 else ""
        mib_i = int(mib) if mib.isdigit() else 0
        # Small allocations are display/compositor processes, not workloads.
        entry = {"pid": pid, "used_mib": mib_i, "process_name": name}
        out["processes"].append(entry)
        if pid not in ours and mib_i >= 1024:
            out["foreign_mib"] += mib_i
    out["foreign"] = out["foreign_mib"] > 0
    return out


def wait_for_gpu_headroom(
    *,
    need_gib: float = 98.0,
    total_gib: float = 121.69,
    timeout_s: float = 0.0,
    interval_s: float = 30.0,
    own_container: str = "cc-diffusiongemma",
) -> dict[str, Any]:
    """Block until the accelerator can actually host the server.

    vLLM refuses to start unless free memory covers
    ``gpu_memory_utilization x total`` -- 0.8 x 121.69 = 97.35 GiB here -- so
    "60 GB free" is not the precondition, and a rung that waited on it raced a
    foreign container that appeared in the gap and died with
    ``Free memory on device cuda:0 (82.37/121.69 GiB) on startup is less than
    desired GPU memory utilization``.

    This box is shared and holders come and go every few minutes, so the wait
    is re-checked immediately before *each* server start rather than once for
    the whole ladder. ``timeout_s = 0`` waits indefinitely.
    """
    start = time.monotonic()
    announced = False
    while True:
        holders = foreign_gpu_holders(own_container)
        used_gib = sum(h["used_mib"] for h in holders.get("processes", [])) / 1024.0
        free_gib = max(0.0, total_gib - used_gib)
        if free_gib >= need_gib:
            return {
                "waited_s": time.monotonic() - start,
                "free_gib": free_gib,
                "need_gib": need_gib,
                "holders": holders,
                "ok": True,
            }
        if not announced:
            announced = True
            names = ", ".join(
                f"{h['process_name']}({h['used_mib']} MiB)"
                for h in holders.get("processes", [])
                if h["used_mib"] >= 1024
            )
            print(
                f"[gpu] waiting for {need_gib:.0f} GiB free (have {free_gib:.1f}); "
                f"holders: {names or 'none'}",
                flush=True,
            )
        if timeout_s and (time.monotonic() - start) >= timeout_s:
            return {
                "waited_s": time.monotonic() - start,
                "free_gib": free_gib,
                "need_gib": need_gib,
                "holders": holders,
                "ok": False,
            }
        time.sleep(interval_s)


def run_overhead(
    *,
    root: Path,
    repeats: int = 30,
    prompts: int = 30,
    model: str = "diffusiongemma-26b-a4b-it",
    level: str = "off",
    serve_script: str = "scripts/serve_diffusiongemma.sh",
    manage_server: bool = True,
    server_url: str | None = None,
    sink_endpoint: str | None = None,
    max_tokens: int = 64,
) -> dict[str, Any]:
    """Measure one rung of the instrumentation overhead ladder.

    The four rungs isolate each cost:

    ``off``        no side channel consumer and no server-side emission;
    ``nopayload``  sampler trace generation without per-token payload transfer;
    ``full``       compact transfer enabled and received, nothing parsed;
    ``parse``      decode + AST parse + canonicalise + gate on every revision,
                   with no tool ever launched.

    This function owns the whole rung: it binds the revision sink *before*
    launching the server (so no revision can be lost to a missing listener),
    runs matched requests, and reports both request latency and what actually
    arrived on the side channel.  A rung that claims to transfer revisions but
    received none is reported as a failure rather than as a cheap measurement.
    """
    if level not in OVERHEAD_LEVELS:
        raise ValueError(f"unknown overhead level {level!r}; have {OVERHEAD_LEVELS}")
    # Hoisted: an early server failure must not turn into a NameError here.
    sink_revisions = 0
    root = Path(root)
    url = server_url or os.environ.get("CRYSTALCALL_SERVER_URL", "http://127.0.0.1:8000/v1")
    host, port = parse_sink_endpoint(sink_endpoint)
    sink_str = f"{host}:{port}"

    sink: RevisionSink | None = None
    runtime: DenoisePtcRuntime | None = None
    registry = None
    impls: dict[str, Any] = {}
    if level != "off":
        sink = RevisionSink(host=host, port=port).start()
        if sink.port != port:
            sink.close()
            raise RuntimeError(f"revision sink bound {sink.port}, expected {port}")
    if level == "parse":
        from .manifest import load_manifest, registry_for

        # A real per-prompt registry, so canonicalisation cost is realistic.
        dev = load_manifest(root / "experiments/manifests/dev.jsonl")
        registry = registry_for(dev[0])
        model_dir = resolve_model_dir("google/diffusiongemma-26B-A4B-it")
        runtime = DenoisePtcRuntime(
            sink=sink,
            decoder=CanvasDecoder(tokenizer=load_tokenizer(model_dir)),
            policy=policy_by_name("full_gate"),
            costs=CostModel(),
            remaining_model=_load_remaining_model(root),
            instrumentation=Instrumentation.PARSE,
        )

    started = False
    try:
        if manage_server:
            _stop_server(root, serve_script)
            # Re-check headroom immediately before starting: the box is
            # shared, and a holder that appears in the gap between a one-shot
            # gate and the launch kills the rung.
            attempts = 0
            while True:
                attempts += 1
                gpu_wait = wait_for_gpu_headroom(
                    need_gib=float(os.environ.get("CRYSTALCALL_GPU_NEED_GIB", "98")),
                    interval_s=float(os.environ.get("CRYSTALCALL_GPU_POLL_S", "30")),
                )
                try:
                    _start_server(root, serve_script, level=level, sink=sink_str)
                    break
                except RuntimeError:
                    if attempts >= 5:
                        raise
                    print(
                        f"[gpu] rung {level!r} lost the race to a foreign holder "
                        f"(attempt {attempts}); waiting again",
                        flush=True,
                    )
                    time.sleep(15.0)
            started = True
        if not _server_ready(url, timeout=30.0):
            raise RuntimeError(f"model server not ready at {url}")

        gpu_before = foreign_gpu_holders()
        latencies: list[float] = []
        tokens: list[int] = []
        prompt_indices: list[int] = []
        request_occurrence_ids: list[int] = []
        rt_reports: list[dict[str, Any]] = []
        streams_closed = 0
        trace_drain_timeout = float(
            os.environ.get("CRYSTALCALL_TRACE_DRAIN_TIMEOUT_S", "30")
        )
        client = httpx.Client(timeout=1800.0)
        try:
            # Warm-ups open trace streams too. Drain each one by its own
            # response id: dropping "whatever is present" leaves a late warm-up
            # stream in the sink, where the next occurrence would count it as
            # its own and the measured request's tail would be missed.
            warmup_posts = 0
            warmup_empty = 0
            warmup_ids: list[str] = []
            for _ in range(3):  # warm before timing
                # `/v1/models` answers before the engine core is live on this
                # build, and such a premature request comes
                # back 200 with an empty body. Retry until the server returns a
                # real completion id: without one the warm-up's trace stream
                # cannot be attributed, let alone drained.
                #
                # Every POST here consumes sampler draws, so the count is part
                # of the rung's workload and is persisted: rungs whose warm-up
                # sequences differ are not comparable.
                warm_id = ""
                for _attempt in range(6):
                    warm = client.post(
                        url.rstrip("/") + "/chat/completions",
                        json={
                            "model": model,
                            "messages": [{"role": "user", "content": OVERHEAD_PROMPTS[0]}],
                            "max_tokens": max_tokens,
                            "ignore_eos": True,
                            "temperature": 0.0,
                        },
                    )
                    warmup_posts += 1
                    warm.raise_for_status()
                    try:
                        warm_id = str((warm.json() or {}).get("id") or "")
                    except ValueError:
                        warm_id = ""
                    if warm_id:
                        warmup_ids.append(warm_id)
                        break
                    warmup_empty += 1
                    time.sleep(10.0)
                if not warm_id:
                    raise RuntimeError(
                        "warm-up never returned a completion id: the server "
                        "accepted requests before the engine core was live"
                    )
                if sink is not None:
                    warm_stream = sink.wait_for_stream(warm_id, trace_drain_timeout)
                    if warm_stream is None or not warm_stream.wait_closed(
                        trace_drain_timeout
                    ):
                        raise RuntimeError(
                            f"warm-up stream {warm_id} did not close within "
                            f"{trace_drain_timeout:.1f}s"
                        )
                    sink.drop(warm_stream.request_id)
            if sink is not None and sink.request_ids():
                raise RuntimeError(
                    f"sink still holds {list(sink.request_ids())} after warm-up drain"
                )

            for i in range(prompts):
                prompt = OVERHEAD_PROMPTS[i % len(OVERHEAD_PROMPTS)]
                rt = None
                holder: dict[str, Any] = {}
                if runtime is not None and registry is not None:
                    def _bind(stream, _h=holder):
                        if "rt" not in _h:
                            _h["rt"] = runtime.begin(
                                f"overhead-{i}", registry, impls, stream, keep_parse_results=False
                            )
                            _h["stream"] = stream

                    sink.set_new_stream_hook(_bind)  # type: ignore[union-attr]
                t0 = time.monotonic_ns()
                r = client.post(
                    url.rstrip("/") + "/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": max_tokens,
                        "ignore_eos": True,
                        "temperature": 0.0,
                    },
                )
                dt = (time.monotonic_ns() - t0) / 1e6
                r.raise_for_status()
                body = r.json()
                latencies.append(dt)
                tokens.append(int((body.get("usage") or {}).get("completion_tokens", 0)))
                prompt_indices.append(i % len(OVERHEAD_PROMPTS))
                request_occurrence_ids.append(i)

                # The HTTP response and the private sender are asynchronous, and
                # only this response's own id identifies its stream. Wait for
                # that exact stream to open and to receive `request_end` before
                # accounting or dropping it; otherwise the parse rung scores a
                # prefix of the revision stream (or another request's tail).
                measured: RequestStream | None = None
                if sink is not None:
                    server_id = str(body.get("id") or "")
                    if not server_id:
                        raise RuntimeError(
                            f"overhead occurrence {i} response carried no request id"
                        )
                    measured = sink.wait_for_stream(server_id, trace_drain_timeout)
                    if measured is None:
                        raise RuntimeError(
                            f"trace stream {server_id} did not open for overhead "
                            f"occurrence {i} within {trace_drain_timeout:.1f}s"
                        )
                    if not measured.wait_closed(trace_drain_timeout):
                        raise RuntimeError(
                            f"trace stream {server_id} did not receive request_end "
                            f"for overhead occurrence {i} within "
                            f"{trace_drain_timeout:.1f}s"
                        )
                    streams_closed += 1

                if runtime is not None:
                    sink.set_new_stream_hook(None)  # type: ignore[union-attr]
                    # `_ingest` swallows hook exceptions, so a failed bind is
                    # invisible: the stream still opens, closes and is counted
                    # while nothing parsed it. Demand the binding per
                    # occurrence; the rung-level `parses == 0` check only
                    # catches total failure, not partial under-measurement.
                    bound = holder.get("stream")
                    rt = holder.get("rt")
                    if bound is None or rt is None:
                        raise RuntimeError(
                            f"parse runtime did not bind occurrence {i} "
                            f"({'no stream' if bound is None else 'no runtime'}): "
                            "the rung would score an unparsed request"
                        )
                    if measured is not None and bound.request_id != measured.request_id:
                        raise RuntimeError(
                            f"parse runtime bound {bound.request_id} but occurrence "
                            f"{i} measured {measured.request_id}"
                        )
                    rt_reports.append(runtime.end(rt, bound))
                if sink is not None and measured is not None:
                    # Count BEFORE dropping: the streams are released each
                    # iteration to bound memory, so counting after the loop would
                    # see an empty sink and report `rev/req 0.0` for every
                    # non-parse rung.
                    sink_revisions += len(measured.snapshot())
                    sink.drop(measured.request_id)
                    if sink.request_ids():
                        raise RuntimeError(
                            f"occurrence {i} left {list(sink.request_ids())} in the sink"
                        )
        finally:
            client.close()
    finally:
        if started:
            _stop_server(root, serve_script)
        if runtime is not None:
            runtime.close()
        if sink is not None:
            sink.close()

    ordered = sorted(latencies)

    def pct(p: float) -> float:
        if not ordered:
            return 0.0
        idx = min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))
        return ordered[idx]

    sink_stats = sink.stats() if sink is not None else {"requests": 0, "events": 0}
    revisions = sum(r["stats"]["revisions"] for r in rt_reports)
    parses = sum(r["stats"]["parses"] for r in rt_reports)
    # `rt_reports` only exists for the `parse` level, where a runtime is
    # attached. The `nopayload` and `full` rungs transfer revisions with no
    # runtime, so counting only through the runtime would report `revisions: 0`
    # for them and make the "claims to transfer but received none" guard
    # unfireable. Count what the sink actually received.
    if level != "off":
        revisions = max(revisions, sink_revisions)
    per_rev_decode = statistics.fmean(
        [r["stats"]["decode_ms_per_revision"] for r in rt_reports if r["stats"]["revisions"]]
    ) if rt_reports else None
    per_rev_parse = statistics.fmean(
        [r["stats"]["parse_ms_per_revision"] for r in rt_reports if r["stats"]["revisions"]]
    ) if rt_reports else None

    problems: list[str] = []
    if level != "off" and sink_stats.get("events", 0) == 0:
        problems.append("no side-channel events received: the rung measured a dead sink")
    if level == "parse" and parses == 0:
        problems.append("no revision was parsed: the parse rung measured nothing")
    # Exactly the measured responses must have closed: a smaller count means
    # some rung row was scored on an unfinished stream.
    if level != "off" and streams_closed != len(latencies):
        problems.append(
            f"only {streams_closed} of {len(latencies)} measured requests had a "
            "closed trace stream: the accounting is not attributable"
        )
    # `full` and `parse` claim to move per-token payload; if none arrived, the
    # rung measured a cheaper path than the one it names.
    if level in ("full", "parse") and revisions == 0:
        problems.append(
            f"the {level!r} rung transferred no revision: it did not measure the "
            "path it claims to"
        )
    if sink_stats.get("malformed", 0):
        problems.append(f"{sink_stats['malformed']} malformed side-channel frames")

    gpu_after = foreign_gpu_holders()
    contended = bool(gpu_before.get("foreign") or gpu_after.get("foreign"))
    if contended:
        problems.append(
            "a foreign GPU process held the accelerator during this rung "
            f"(before {gpu_before.get('foreign_mib', 0)} MiB, "
            f"after {gpu_after.get('foreign_mib', 0)} MiB): the latency is not "
            "attributable to instrumentation"
        )

    record = {
        "schema_version": "crystalcall-overhead/4",
        "gpu_exclusive": not contended,
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "level": level,
        "tracing_enabled": level != "off",
        "parsing_enabled": level == "parse",
        "url": url,
        "model": model,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "ladder_seed": os.environ.get("CRYSTALCALL_LADDER_SEED"),
        "ladder_order": os.environ.get("CRYSTALCALL_LADDER_ORDER"),
        "n": len(latencies),
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "mean_ms": statistics.fmean(latencies) if latencies else 0.0,
        "stdev_ms": statistics.pstdev(latencies) if len(latencies) > 1 else 0.0,
        "completion_tokens": tokens,
        "prompt_indices": prompt_indices,
        "request_occurrence_ids": request_occurrence_ids,
        "n_clusters": len(set(prompt_indices)),
        "completion_tokens_mean": statistics.fmean(tokens) if tokens else 0.0,
        "streams_closed": streams_closed,
        "stream_accounting_provenance": (
            "request_end observed for every counted stream before drop"
        ),
        "warmup": {
            "requests": 3,
            "posts": warmup_posts,
            "empty_body_retries": warmup_empty,
            "ids": warmup_ids,
        },
        "trace_drain_timeout_s": trace_drain_timeout,
        "sink_stats": sink_stats,
        "revisions": revisions,
        "parses": parses,
        "revisions_per_request": (revisions / len(latencies)) if latencies else 0.0,
        "decode_ms_per_revision": per_rev_decode,
        "parse_ms_per_revision": per_rev_parse,
        "latencies_ms": latencies,
        "problems": problems,
        "ok": not problems,
        "unix_ns": time.time_ns(),
    }
    out = root / "experiments/derived/overhead"
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / f"overhead-{level}.json", record)
    return record


def consolidate_overhead(root: Path) -> int:
    """Flatten the overhead-ladder rung files into one JSONL for aggregation.

    Each rung writes `experiments/derived/overhead/overhead-<level>.json`;
    `aggregate()` consumes `experiments/derived/overhead_microbench.jsonl`.
    The relative cost of each rung against the uninstrumented rung is computed
    here so no report has to recompute it.
    """
    d = root / "experiments/derived/overhead"
    if not d.exists():
        return 0
    rungs: dict[str, dict[str, Any]] = {}
    for level in OVERHEAD_LEVELS:
        f = d / f"overhead-{level}.json"
        if f.exists():
            try:
                rungs[level] = json.loads(f.read_text())
            except json.JSONDecodeError:
                continue
    if not rungs:
        return 0
    base = (rungs.get("off") or {}).get("p50_ms")
    rows: list[dict[str, Any]] = []
    for level in OVERHEAD_LEVELS:
        rec = rungs.get(level)
        if rec is None:
            continue
        row = {k: v for k, v in rec.items() if k != "latencies_ms"}
        row["schema_version"] = "crystalcall-overhead-microbench/1"
        row["relative_p50_vs_off"] = (
            (rec["p50_ms"] / base - 1.0) if base else None
        )
        row["absolute_p50_delta_ms"] = (rec["p50_ms"] - base) if base else None
        rows.append(row)
    return write_jsonl(root / "experiments/derived/overhead_microbench.jsonl", rows)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str | None:
    import hashlib

    if not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _cached_doctor_digest(root: Path) -> str | None:
    p = root / "reports/doctor.json"
    if p.exists():
        try:
            return json.loads(p.read_text()).get("digest")
        except json.JSONDecodeError:
            return None
    return None


def _load_cost_model(root: Path) -> CostModel:
    p = root / "configs/frozen_policy.json"
    if p.exists():
        try:
            d = json.loads(p.read_text()).get("costs") or {}
            return CostModel(
                trace_ms_per_revision=float(d.get("trace_ms_per_revision", 0.0)),
                parse_ms_per_revision=float(d.get("parse_ms_per_revision", 0.0)),
                schedule_ms_per_launch=float(d.get("schedule_ms_per_launch", 0.0)),
                contention_ms_per_launch=d.get("contention_ms_per_launch", {}) or {},
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return CostModel()


def _load_remaining_model(root: Path) -> RemainingDecodeModel:
    p = root / "configs/frozen_policy.json"
    if p.exists():
        try:
            d = json.loads(p.read_text()).get("remaining_decode") or {}
            return RemainingDecodeModel(
                expected_steps_per_canvas=float(d.get("expected_steps_per_canvas", 24.0)),
                expected_canvases=float(d.get("expected_canvases", 1.0)),
                default_step_ms=float(d.get("default_step_ms", 40.0)),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return RemainingDecodeModel()
