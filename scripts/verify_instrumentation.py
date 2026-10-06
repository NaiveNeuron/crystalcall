#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27"]
# ///
"""Prove the DiffusionGemma sampler instrumentation is observationally pure.

Two arms, same container image, same patched sampler file, switched only by
``CRYSTALCALL_TRACE``:

  arm A (baseline)  CRYSTALCALL_TRACE=0  -> completions + latency
  arm B (traced)    CRYSTALCALL_TRACE=1  -> completions + latency + revisions

The script fails (non-zero exit) if

  * any completion differs byte-for-byte between the arms,
  * a revision has the wrong array lengths / implausible entropy or confidence,
  * ``step`` is not 0,1,2,... within a canvas or does not reset per canvas,
  * ``phase``/``converged`` do not agree with ``max_denoising_steps``,
  * ``prefix_len`` does not track the committed token count,
  * the emitter reports any drop / error, or the sink reports malformed lines.

Usage:
  uv run scripts/verify_instrumentation.py                 # both arms
  uv run scripts/verify_instrumentation.py --arm baseline  # only arm A
  uv run scripts/verify_instrumentation.py --arm traced --baseline-json out.json
  uv run scripts/verify_instrumentation.py --keep-server
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from crystalcall.sidechannel import RevisionSink  # noqa: E402

TRACE_DIR = Path(os.environ.get("CRYSTALCALL_TRACE_DIR", "/tmp/cc-trace"))
SERVE = REPO / "scripts" / "serve_diffusiongemma.sh"
EVIDENCE = REPO / "docs" / "evidence"
CONTAINER = "cc-diffusiongemma"

# Three short deterministic prompts (the assignment's evidence set) plus one
# long prompt so the multi-canvas path (canvas_id > 0, prefix_len > 0) is
# actually exercised: max_tokens > canvas_length forces a second canvas.
PROMPTS = [
    "Name the capital of France in one word.",
    "What is 17 times 3? Answer with the number only.",
    "List three primary colours, comma separated.",
    "Explain in detail, in at least 400 words, how a bicycle derailleur "
    "shifts gears, covering the cable, the parallelogram, the jockey wheels "
    "and the chain.",
]
# Indices whose completions MUST be byte-identical across arms: the three short
# single-canvas prompts from the assignment. Index 3 exists to exercise the
# multi-canvas path (canvas_id > 0, prefix_len > 0); this engine build is not
# bit-reproducible over ~640 tokens even with tracing off (MoE reductions +
# async scheduling), so it is scored but not asserted -- the `control` arm
# measures that directly.
REQUIRED_IDENTICAL = (0, 1, 2)
MAX_TOKENS = 768
VOCAB = 262144
MAX_ENTROPY = math.log(VOCAB)


def source_provenance() -> dict[str, str]:
    """Record the exact checked-out source used by a traced run."""
    out: dict[str, str] = {}
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    out["git_commit"] = commit
    for name, path in (
        ("sampler", REPO / "patches/build/diffusion_gemma.py"),
        ("trace_patch", REPO / "patches/vllm-trace.patch"),
        ("emitter", REPO / "patches/crystalcall_trace.py"),
    ):
        digest = hashlib.sha256()
        try:
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    digest.update(chunk)
            out[f"{name}_sha256"] = digest.hexdigest()
        except OSError:
            out[f"{name}_sha256"] = "missing"
    return out


def provenance_errors(
    label: str, blob: dict[str, Any], current: dict[str, str]
) -> list[str]:
    """Refuse to compare an arm against evidence from a different source tree.

    The three fidelity arms are separate server starts, so identity claims are
    only meaningful if every arm ran the same patched sampler. Evidence written
    before provenance stamping (or against different patched bytes) must not
    silently validate a new run.

    Only the ``*_sha256`` digests gate: they are the bytes the server executes.
    ``git_commit`` is recorded for traceability but deliberately does not gate,
    because editing docs or host-side tests cannot change engine behaviour and
    a 24-minute three-arm GPU re-run per commit would be pure ceremony.
    """
    recorded = blob.get("source_provenance")
    if not isinstance(recorded, dict):
        return [f"{label} evidence has no source_provenance; re-run that arm"]
    keys = {k for k in set(recorded) | set(current) if k.endswith("_sha256")}
    differing = sorted(k for k in keys if recorded.get(k) != current.get(k))
    if differing:
        return [
            f"{label} evidence came from a different source tree "
            f"({', '.join(differing)} differ); re-run that arm"
        ]
    return []


# --------------------------------------------------------------------------- #
# server control
# --------------------------------------------------------------------------- #


def start_server(trace: bool, sink: str | None) -> None:
    env = dict(os.environ)
    env["CRYSTALCALL_TRACE"] = "1" if trace else "0"
    env["CRYSTALCALL_TRACE_SINK"] = sink or ""
    env["CRYSTALCALL_TRACE_LEVEL"] = os.environ.get("CRYSTALCALL_TRACE_LEVEL", "full")
    env["CRYSTALCALL_TRACE_DIR"] = str(TRACE_DIR)
    print(
        f"[server] starting (trace={env['CRYSTALCALL_TRACE']} "
        f"sink={env['CRYSTALCALL_TRACE_SINK']!r})",
        flush=True,
    )
    subprocess.run([str(SERVE)], env=env, check=True)


def stop_server() -> None:
    subprocess.run(
        [str(SERVE), "--stop"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # give the emitter's atexit hook a moment to flush
    time.sleep(1.0)


def server_up(base: str) -> bool:
    try:
        return httpx.get(f"{base}/v1/models", timeout=3.0).status_code == 200
    except httpx.HTTPError:
        return False


# --------------------------------------------------------------------------- #
# requests
# --------------------------------------------------------------------------- #


class ServerReplyError(RuntimeError):
    """The server answered, but not with the JSON the OpenAI surface promises."""


def dump_container_log(tag: str) -> Path | None:
    """Persist the server's container log before it is torn down.

    A 20-stage pipeline that destroys the only copy of the server log on
    failure is not debuggable, and the container is removed on stop.
    """
    out = EVIDENCE / f"server-{tag}.log"
    try:
        proc = subprocess.run(
            [str(SERVE), "--logs", "--tail=400"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:  # noqa: BLE001 - diagnostics must never mask the real error
        print(f"[server] could not capture container log: {exc}", file=sys.stderr)
        return None
    blob = (proc.stdout or "") + (proc.stderr or "")
    if not blob.strip():
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(blob)
    print(f"[server] container log -> {out}", file=sys.stderr)
    return out


def run_prompts(base: str, model: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with httpx.Client(timeout=1800.0) as client:
        for prompt in PROMPTS:
            body = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "top_p": 1,
                "seed": 12345,
                "max_tokens": MAX_TOKENS,
                "stream": False,
            }
            t0 = time.monotonic()
            r = client.post(f"{base}/v1/chat/completions", json=body)
            dt = time.monotonic() - t0
            r.raise_for_status()
            try:
                data = r.json()
            except json.JSONDecodeError as exc:
                # A bare "Expecting value: line 1 column 1" tells the operator
                # nothing and the container is usually gone by the time they
                # look.  Report what the server actually sent.
                body_preview = r.content[:400].decode("utf-8", "replace")
                raise ServerReplyError(
                    f"non-JSON reply from {base}/v1/chat/completions: "
                    f"HTTP {r.status_code} {r.reason_phrase!r}, "
                    f"content-type={r.headers.get('content-type')!r}, "
                    f"{len(r.content)} bytes, preview={body_preview!r} ({exc})"
                ) from exc
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            out.append(
                {
                    "prompt": prompt,
                    "server_request_id": data.get("id") or "",
                    "content": choice["message"].get("content") or "",
                    "reasoning": choice["message"].get("reasoning_content") or "",
                    "tool_calls": choice["message"].get("tool_calls") or [],
                    "finish_reason": choice.get("finish_reason"),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "latency_s": dt,
                }
            )
            print(
                f"[req] {dt:7.3f}s  {out[-1]['completion_tokens']:4d} tok  "
                f"{prompt[:38]!r} -> {out[-1]['content'][:60]!r}",
                flush=True,
            )
    return out


def resolve_model(base: str) -> str:
    r = httpx.get(f"{base}/v1/models", timeout=10.0)
    r.raise_for_status()
    return r.json()["data"][0]["id"]


# --------------------------------------------------------------------------- #
# revision validation
# --------------------------------------------------------------------------- #


def check_streams(
    sink: RevisionSink,
    canvas_length: int,
    max_steps: int,
    *,
    expect_ids: Sequence[str] = (),
) -> list[str]:
    """Validate the side-channel streams belonging to the measured prompts.

    ``expect_ids`` are the server request ids returned for the prompts.  The
    server also answers the launch-time decode readiness probe, which is a
    real traced request, so the check is "every prompt I sent produced a valid
    stream", never "the sink saw exactly N streams".
    """
    errors: list[str] = []
    all_rids = sink.request_ids()
    wanted = [r for r in expect_ids if r]
    if wanted:
        # The OpenAI surface returns `chatcmpl-<id>` while the sampler tags its
        # side-channel events `chatcmpl-<id>-<engine suffix>`, so the join is by
        # prefix.  An ambiguous prefix is an error, never a silent pick.
        rids: list[str] = []
        for want in wanted:
            if want in all_rids:
                rids.append(want)
                continue
            hits = [r for r in all_rids if r.startswith(f"{want}-")]
            if len(hits) == 1:
                rids.append(hits[0])
            elif not hits:
                errors.append(
                    f"no side-channel stream for request id {want!r}; sink saw {all_rids}"
                )
            else:
                errors.append(f"ambiguous side-channel streams for {want!r}: {hits}")
    else:
        rids = all_rids
        if len(rids) != len(PROMPTS):
            errors.append(f"expected {len(PROMPTS)} traced requests, got {len(rids)}: {rids}")
    for rid in rids:
        stream = sink.stream(rid)
        assert stream is not None
        revs = stream.snapshot()
        if not revs:
            errors.append(f"{rid}: no revisions")
            continue
        if stream.begin_ns == 0:
            errors.append(f"{rid}: no request_begin")
        if stream.end_ns == 0:
            errors.append(f"{rid}: no request_end")
        if stream.prompt_tokens <= 0:
            errors.append(f"{rid}: prompt_tokens={stream.prompt_tokens}")

        by_canvas: dict[int, list] = {}
        for rev in revs:
            by_canvas.setdefault(rev.canvas_id, []).append(rev)

        if sorted(by_canvas) != list(range(len(by_canvas))):
            errors.append(f"{rid}: canvas ids not 0..n: {sorted(by_canvas)}")

        expected_prefix = 0
        for cid in sorted(by_canvas):
            group = by_canvas[cid]
            steps = [r.step for r in group]
            phases = [r.phase for r in group]

            # step resets to 0 per canvas and increments by one per denoise
            denoise = [r for r in group if r.phase == "denoise"]
            if [r.step for r in denoise] != list(range(len(denoise))):
                errors.append(f"{rid}/c{cid}: denoise steps not 0..n: {steps}")
            if len(denoise) > max_steps:
                errors.append(
                    f"{rid}/c{cid}: {len(denoise)} denoise steps > max_denoising_steps={max_steps}"
                )
            commits = [r for r in group if r.phase == "commit"]
            if len(commits) > 1:
                errors.append(f"{rid}/c{cid}: {len(commits)} commit revisions")
            if commits and phases[-1] != "commit":
                errors.append(f"{rid}/c{cid}: commit is not the last revision: {phases}")

            # the last denoise revision before a commit must be the converged one
            if commits:
                if not denoise:
                    errors.append(f"{rid}/c{cid}: commit without any denoise step")
                elif not denoise[-1].converged:
                    errors.append(
                        f"{rid}/c{cid}: last denoise step {denoise[-1].step} not converged"
                    )
                if len(denoise) == max_steps and not denoise[-1].converged:
                    errors.append(f"{rid}/c{cid}: step cap reached without converged flag")
                if cid in stream.canvas_done_ns:
                    pass
                else:
                    errors.append(f"{rid}/c{cid}: missing canvas_done")
            # converged must only be set on the final denoise step
            early = [r.step for r in denoise[:-1] if r.converged]
            if early:
                errors.append(f"{rid}/c{cid}: converged set early at steps {early}")

            for rev in group:
                if rev.prefix_len != expected_prefix:
                    errors.append(
                        f"{rid}/c{cid}/s{rev.step}: prefix_len={rev.prefix_len} "
                        f"expected {expected_prefix}"
                    )
                    break
                if rev.n != canvas_length:
                    errors.append(
                        f"{rid}/c{cid}/s{rev.step}: {rev.n} token ids, expected {canvas_length}"
                    )
                    break
                if len(rev.accepted_mask) != canvas_length:
                    errors.append(f"{rid}/c{cid}/s{rev.step}: bad accepted_mask length")
                    break
                if rev.token_entropy is None or len(rev.token_entropy) != canvas_length:
                    errors.append(f"{rid}/c{cid}/s{rev.step}: bad token_entropy length")
                    break
                if rev.token_confidence is None or len(rev.token_confidence) != canvas_length:
                    errors.append(f"{rid}/c{cid}/s{rev.step}: bad token_confidence length")
                    break
                if any(not (0 <= t < VOCAB) for t in rev.predicted_token_ids):
                    errors.append(f"{rid}/c{cid}/s{rev.step}: token id out of range")
                    break
                if any(not (-1e-4 <= e <= MAX_ENTROPY + 1e-3) for e in rev.token_entropy):
                    errors.append(f"{rid}/c{cid}/s{rev.step}: entropy out of [0, ln V]")
                    break
                if any(not (0.0 <= c <= 1.0 + 1e-6) for c in rev.token_confidence):
                    errors.append(f"{rid}/c{cid}/s{rev.step}: confidence out of [0,1]")
                    break
                if not (-1e-4 <= rev.mean_entropy <= MAX_ENTROPY + 1e-3):
                    errors.append(f"{rid}/c{cid}/s{rev.step}: mean_entropy out of range")
                    break
            if commits:
                expected_prefix += canvas_length
    return errors


def revision_report(sink: RevisionSink, canvas_length: int) -> dict[str, Any]:
    rows = []
    for rid in sink.request_ids():
        stream = sink.stream(rid)
        assert stream is not None
        revs = stream.snapshot()
        canvases = sorted({r.canvas_id for r in revs})
        rows.append(
            {
                "request_id": rid,
                "prompt_tokens": stream.prompt_tokens,
                "output_tokens": stream.output_tokens,
                "revisions": len(revs),
                "canvases": len(canvases),
                "denoise_steps_per_canvas": [
                    sum(1 for r in revs if r.canvas_id == c and r.phase == "denoise")
                    for c in canvases
                ],
                "first_mean_entropy": round(revs[0].mean_entropy, 5) if revs else None,
                "last_mean_entropy": round(revs[-1].mean_entropy, 5) if revs else None,
                "accepted_first": revs[0].accepted_count if revs else None,
                "accepted_last": revs[-1].accepted_count if revs else None,
                "wall_ms": round((stream.end_ns - stream.begin_ns) / 1e6, 2),
            }
        )
    return {"streams": rows, "sink_stats": sink.stats()}


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def diff_indices(ref: list[dict], got: list[dict]) -> list[int]:
    """Indices whose completion is not byte-identical between two arms."""
    out: list[int] = []
    for i, (a, b) in enumerate(zip(ref, got)):
        same = all(
            a[k] == b[k]
            for k in ("content", "reasoning", "finish_reason", "completion_tokens")
        ) and json.dumps(a["tool_calls"], sort_keys=True) == json.dumps(
            b["tool_calls"], sort_keys=True
        )
        if not same:
            out.append(i)
    return out


def compare_arms(
    ref: list[dict],
    got: list[dict],
    label: str,
    required: set[int] | None = None,
) -> tuple[list[str], list[str]]:
    """Byte-for-byte comparison of two arms' completions.

    ``required`` restricts the hard assertion to the request indices the engine
    itself was shown to reproduce (see the ``control`` arm). Mismatches outside
    that set are returned as warnings: this engine build is not bit-reproducible
    across restarts for long generations even with tracing off, so those
    requests carry no information about the instrumentation.
    """
    errors: list[str] = []
    warnings: list[str] = []
    if len(got) != len(ref):
        errors.append(f"arm size mismatch: {len(got)} vs {len(ref)}")
    for i in diff_indices(ref, got):
        a, b = ref[i], got[i]
        detail = "; ".join(
            f"{k}: baseline={a[k]!r} {label}={b[k]!r}"
            if k != "content"
            else f"content differs ({len(a[k])} vs {len(b[k])} chars)"
            for k in ("content", "reasoning", "finish_reason", "completion_tokens")
            if a[k] != b[k]
        )
        msg = f"request {i} ({PROMPTS[i][:40]!r}): {detail}"
        if required is None or i in required:
            errors.append(msg)
        else:
            warnings.append(msg + "  [engine not reproducible here; not a purity signal]")
    return errors, warnings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", choices=("both", "baseline", "control", "traced"), default="both"
    )
    ap.add_argument("--base-url", default=os.environ.get("CC_BASE_URL", "http://127.0.0.1:8000"))
    ap.add_argument("--canvas-length", type=int, default=256)
    ap.add_argument("--max-denoising-steps", type=int, default=48)
    ap.add_argument("--baseline-json", default=str(EVIDENCE / "baseline.json"))
    ap.add_argument("--traced-json", default=str(EVIDENCE / "traced.json"))
    ap.add_argument("--reuse-server", action="store_true",
                    help="do not start/stop containers; the caller manages them")
    ap.add_argument("--keep-server", action="store_true")
    args = ap.parse_args()

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    base = args.base_url
    baseline_path = Path(args.baseline_json)
    traced_path = Path(args.traced_json)
    stats_host = TRACE_DIR / "stats.json"
    provenance = source_provenance()

    # ---- arm A: no tracing ------------------------------------------------ #
    if args.arm in ("both", "baseline"):
        # Same ownership rule as the control arm: whatever this arm starts, it
        # releases, including on exceptions during startup or the run.
        released = False
        try:
            if not args.reuse_server:
                stop_server()
                start_server(trace=False, sink=None)
            if not server_up(base):
                print("FAIL: baseline server not reachable", file=sys.stderr)
                return 1
            model = resolve_model(base)
            t0 = time.monotonic()
            baseline = run_prompts(base, model)
            base_wall = time.monotonic() - t0
            baseline_path.write_text(
                json.dumps(
                    {
                        "model": model,
                        "wall_s": base_wall,
                        "results": baseline,
                        "source_provenance": provenance,
                    },
                    indent=2,
                )
            )
            print(f"[arm A] {base_wall:.3f}s total, wrote {baseline_path}")
            if args.arm == "baseline":
                return 0
            # arm A of a `both` run: hand the GPU to the traced arm.
            if not args.reuse_server:
                stop_server()
                released = True
        finally:
            if not released and not args.reuse_server:
                if args.arm != "baseline" or not args.keep_server:
                    stop_server()
    # ---- control arm: tracing OFF twice ----------------------------------- #
    # The entropy-bound sampler renoises the canvas with Gumbel noise and
    # ignores the request temperature, so decoding is stochastic; the engine
    # seed pins the RNG stream. This arm measures how far that gets us: it
    # records which request indices a freshly restarted, UNINSTRUMENTED server
    # reproduces byte-for-byte. Only those indices can carry information about
    # whether the instrumentation perturbs decoding.
    if args.arm == "control":
        # Prerequisites first: discovering a stale baseline after an 8-minute
        # run wastes the run and the GPU.
        if not baseline_path.exists():
            print(f"FAIL: no baseline evidence at {baseline_path}", file=sys.stderr)
            return 1
        baseline_blob = json.loads(baseline_path.read_text())
        stale = provenance_errors("baseline", baseline_blob, provenance)
        if stale:
            for err in stale:
                print(f"FAIL: {err}", file=sys.stderr)
            return 1
        ref = baseline_blob["results"]
        # The startup itself is inside the try: a launcher that creates the
        # container and then raises must not leave it holding the GPU.
        try:
            if not args.reuse_server:
                stop_server()
                start_server(trace=False, sink=None)
            if not server_up(base):
                print("FAIL: control server not reachable", file=sys.stderr)
                return 1
            model = resolve_model(base)
            control = run_prompts(base, model)
            diverged = diff_indices(ref, control)
            reproducible = [i for i in range(len(ref)) if i not in diverged]
            required = sorted(set(reproducible) & set(REQUIRED_IDENTICAL))
            (EVIDENCE / "control.json").write_text(
                json.dumps(
                    {
                        "model": model,
                        "results": control,
                        "reproducible_indices": reproducible,
                        "diverged_indices": diverged,
                        "source_provenance": provenance,
                    },
                    indent=2,
                )
            )
        finally:
            # Every path out of the measured section, including exceptions,
            # must release the ~99 GB server this arm started.
            if not args.keep_server and not args.reuse_server:
                stop_server()
        print(
            f"\n[control] engine-reproducible request indices: {reproducible}; "
            f"diverged with tracing OFF: {diverged}"
        )
        if required != sorted(REQUIRED_IDENTICAL):
            print(
                "FAIL: the short deterministic prompts "
                f"{sorted(REQUIRED_IDENTICAL)} are not reproducible even with "
                f"tracing off (reproducible: {reproducible})",
                file=sys.stderr,
            )
            return 1
        print(
            "PASS: with tracing off, a restarted server reproduces the required "
            f"prompts {required} byte-for-byte."
        )
        return 0



    # ---- arm B: tracing on ------------------------------------------------ #
    if not baseline_path.exists():
        print(f"FAIL: no baseline evidence at {baseline_path}", file=sys.stderr)
        return 1
    baseline_blob = json.loads(baseline_path.read_text())
    stale = provenance_errors("baseline", baseline_blob, provenance)
    if stale:
        for err in stale:
            print(f"FAIL: {err}", file=sys.stderr)
        return 1
    baseline = baseline_blob["results"]

    # Only indices the uninstrumented engine reproduces across restarts can test
    # purity; the control arm records which those are. Gate it before the GPU
    # run, not after: stale control evidence invalidates the comparison, and
    # discovering that afterwards throws away the whole traced run.
    control_path = EVIDENCE / "control.json"
    required: set[int] = set(REQUIRED_IDENTICAL)
    if control_path.exists():
        ctl = json.loads(control_path.read_text())
        stale = provenance_errors("control", ctl, provenance)
        if stale:
            for err in stale:
                print(f"FAIL: {err}", file=sys.stderr)
            return 1
        if "reproducible_indices" in ctl:
            required = set(ctl["reproducible_indices"])
            print(f"[arm B] identity asserted on control-reproducible {sorted(required)}")
        else:
            print(f"[arm B] control evidence has no indices; asserting {sorted(required)}")
    else:
        print(f"[arm B] no control evidence; identity asserted on {sorted(required)}")

    sink = RevisionSink(host="127.0.0.1", port=0).start()
    print(f"[sink] listening on {sink.endpoint}")
    try:
        if not args.reuse_server:
            if stats_host.exists():
                stats_host.unlink()
            start_server(trace=True, sink=sink.endpoint)
        if not server_up(base):
            print("FAIL: traced server not reachable", file=sys.stderr)
            return 1
        model = resolve_model(base)
        t0 = time.monotonic()
        try:
            traced = run_prompts(base, model)
        except (ServerReplyError, httpx.HTTPError) as exc:
            print(f"FAIL: {exc}", file=sys.stderr)
            dump_container_log(f"{args.arm}-failure")
            return 1
        traced_wall = time.monotonic() - t0
        # let the last request_end / revisions land
        for rid in sink.request_ids():
            stream = sink.stream(rid)
            if stream is not None:
                stream.wait_closed(10.0)
        time.sleep(1.0)

        errors = check_streams(
            sink,
            args.canvas_length,
            args.max_denoising_steps,
            expect_ids=[str(r.get("server_request_id") or "") for r in traced],
        )

        if not set(REQUIRED_IDENTICAL) <= required:
            errors.append(
                f"control shows the required prompts {sorted(REQUIRED_IDENTICAL)} are "
                f"not engine-reproducible (reproducible: {sorted(required)})"
            )
        cmp_errors, cmp_warnings = compare_arms(baseline, traced, "traced", required)
        errors += cmp_errors

        report = revision_report(sink, args.canvas_length)
        emitter_stats: dict[str, Any] = {}
        if stats_host.exists():
            emitter_stats = json.loads(stats_host.read_text())
            if emitter_stats.get("dropped", 0) != 0:
                errors.append(f"emitter dropped {emitter_stats['dropped']} events")
            if emitter_stats.get("errors", 0) != 0:
                errors.append(f"emitter reported {emitter_stats['errors']} errors")
        else:
            errors.append(f"no emitter stats at {stats_host}")
        if report["sink_stats"]["malformed"]:
            errors.append(f"sink saw {report['sink_stats']['malformed']} malformed lines")

        n_rev = sum(r["revisions"] for r in report["streams"])
        per_rev = {
            "bytes_per_revision": (
                round(emitter_stats["bytes"] / n_rev, 1) if n_rev and emitter_stats else None
            ),
            "encode_ns_per_event": (
                round(emitter_stats["encode_ns"] / emitter_stats["sent"], 1)
                if emitter_stats.get("sent") else None
            ),
            "queue_max": emitter_stats.get("queue_max"),
        }

        traced_path.write_text(
            json.dumps(
                {
                    "model": model,
                    "wall_s": traced_wall,
                    "baseline_wall_s": baseline_blob["wall_s"],
                    "results": traced,
                    "emitter_stats": emitter_stats,
                    "per_revision": per_rev,
                    "revisions": report,
                    "clock_skew": sink.clock_skew_ns(),
                    "identity_required_indices": sorted(required),
                    "identity_warnings": cmp_warnings,
                    "errors": errors,
                    "source_provenance": provenance,
                },
                indent=2,
            )
        )
        print(f"[arm B] {traced_wall:.3f}s total, wrote {traced_path}")
        print(json.dumps({"per_revision": per_rev, "emitter": emitter_stats}, indent=2))
        for row in report["streams"]:
            print(json.dumps(row))
        for w in cmp_warnings:
            print(f"[warn] {w}")

        if errors:
            print(f"\nFAIL ({len(errors)} problems):", file=sys.stderr)
            for e in errors:
                print(f"  - {e}", file=sys.stderr)
            return 1
        print(
            "\nPASS: completions byte-identical with tracing off vs on for "
            f"requests {sorted(required)}; {n_rev} revisions validated; 0 drops."
        )
        return 0
    finally:
        sink.close()
        if not args.keep_server and not args.reuse_server:
            stop_server()


if __name__ == "__main__":
    raise SystemExit(main())
