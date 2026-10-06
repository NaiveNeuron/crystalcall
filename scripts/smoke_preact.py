# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "tokenizers>=0.20", "pyyaml>=6", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""GPU acceptance test for the P-ReAct-style arm: SEMANTIC health, not timing.

This is the single command to run once the GPU is free. It brings up the pinned
DiffusionGemma server twice from the SAME image and the SAME patched sampler
file -- once with ``CRYSTALCALL_PREACT=0`` (the standard / CrystalCall arm) and
once with ``CRYSTALCALL_PREACT=1`` -- and drives both through the **production
path**: `crystalcall.harness.run_trace_arm`, i.e. real OMP-Lite, the single
`eval` tool, the real `HostToolRouter`, the real canonicaliser and the real
frozen local tools. Nothing here re-implements execution; a bespoke runner
could diverge from the arms it is supposed to validate, and constraint 3 fixes
the executor across arms.

Asserted, all hard failures:

1. **Same action.** For every prompt, the canonical authoritative calls the two
   arms actually executed are identical. Priority decoding may change
   *scheduling*; if it changes or drops a call, the baseline is crippled and
   every latency advantage measured against it is an artefact.
2. **Reasoning not truncated.** The final assistant text of each arm's turn is
   non-empty and structurally comparable. The harness runs the whole OMP turn
   loop, so this is the real closing turn after the tool results, which is
   exactly the "reasoning continues" property P-ReAct claims.
3. **Benchmark correctness preserved.** Neither arm regresses to an incorrect
   call set on a prompt the other got right.
4. **The mechanism fired.** The preact arm produced at least one
   `tool_call_ready`, at most one per (request_id, canvas_id), and every event's
   span re-derives exactly from the revision recorded at that step.
5. **The mechanism is inert when off.** The standard arm produced *zero*
   readiness events: the runtime proof that the added code does not touch the
   CrystalCall arm.

Run:  uv run scripts/smoke_preact.py [--prompts 6] [--json path]
Exit: 0 on success.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crystalcall.harness import ARMS, TraceRunConfig, parse_sink_endpoint, run_trace_arm  # noqa: E402
from crystalcall.replay import PREACT_ARM, load_trajectories  # noqa: E402

SERVE = REPO / "scripts" / "serve_diffusiongemma.sh"
PREACT_CFG = REPO / "configs" / "preact.yml"
MANIFEST = REPO / "experiments" / "manifests" / "dev.jsonl"
BASE = os.environ.get("CRYSTALCALL_SERVER_URL", "http://127.0.0.1:8000/v1")
SINK = os.environ.get("CRYSTALCALL_TRACE_SINK", "127.0.0.1:9802")
SEED = 0


# --------------------------------------------------------------------------- #
# server
# --------------------------------------------------------------------------- #


def start_server(*, preact: bool, mode: str, bound: float | None) -> None:
    env = dict(os.environ)
    env["CRYSTALCALL_TRACE"] = "1"
    env["CRYSTALCALL_TRACE_SINK"] = SINK
    env["CRYSTALCALL_TRACE_LEVEL"] = "full"
    env["CRYSTALCALL_PREACT"] = "1" if preact else "0"
    env["CRYSTALCALL_PREACT_MODE"] = mode
    if bound is not None:
        env["CRYSTALCALL_PREACT_BOUND"] = str(bound)
    print(f"[server] starting (preact={env['CRYSTALCALL_PREACT']} mode={mode})", flush=True)
    subprocess.run([str(SERVE)], env=env, check=True)


def stop_server() -> None:
    subprocess.run(
        [str(SERVE), "--stop"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1.0)


# --------------------------------------------------------------------------- #
# one arm, through the production path
# --------------------------------------------------------------------------- #


def run_arm(arm: str, out_dir: Path, limit: int) -> dict[str, Any]:
    cfg = TraceRunConfig(
        root=REPO,
        manifest=MANIFEST,
        arm=arm,
        seeds=(SEED,),
        out_dir=out_dir,
        limit=limit,
        server_url=BASE,
        trace_sink_endpoint=SINK,
    )
    summary = run_trace_arm(cfg)
    shard = Path(summary["shards"][0]["path"])
    trajectories = load_trajectories([shard], verify=True)
    return {"summary": summary, "shard": shard, "trajectories": trajectories}


def canonical_calls(traj: Any) -> list[str]:
    """The canonical identities OMP actually executed, in a stable order."""
    out: list[str] = []
    for call in traj.authoritative_calls or ():
        tool = call.get("tool") or call.get("name") or ""
        args = call.get("args") or {}
        body = ",".join(
            f"{k}={json.dumps(args[k], sort_keys=True, separators=(',', ':'))}"
            for k in sorted(args)
        )
        out.append(f"{tool}({body})")
    return sorted(out)


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #


def check_arm_health(res: dict[str, Any], arm: str) -> list[str]:
    errs: list[str] = []
    summary = res["summary"]
    if summary.get("failures"):
        errs.append(f"{arm}: {summary['failures']} turns failed")
    if not res["trajectories"]:
        errs.append(f"{arm}: no trajectory recovered from {res['shard']}")
    for traj in res["trajectories"]:
        where = f"{arm}[{traj.prompt_id}]"
        if not traj.revisions:
            errs.append(f"{where}: no revisions recorded; the side channel did not join")
        if not (traj.final_text or "").strip():
            errs.append(
                f"{where}: the final assistant text is empty -- the turn was truncated "
                "(this is the closing turn after the tool results, so it must not be)"
            )
        if not traj.authoritative_calls:
            errs.append(f"{where}: OMP executed no authoritative call")
        if not traj.valid_final_parse:
            errs.append(f"{where}: the final program did not parse")
    return errs


def check_same_action(std: dict[str, Any], pre: dict[str, Any]) -> tuple[list[str], list[int]]:
    errs: list[str] = []
    lost: list[str] = []
    a = {t.prompt_id: t for t in std["trajectories"]}
    b = {t.prompt_id: t for t in pre["trajectories"]}
    if set(a) != set(b):
        errs.append(f"prompt sets differ: standard={sorted(a)} preact={sorted(b)}")
    for pid in sorted(set(a) & set(b)):
        ca, cb = canonical_calls(a[pid]), canonical_calls(b[pid])
        if ca != cb:
            errs.append(
                f"[{pid}]: canonical calls differ: standard={ca} preact={cb} "
                "-- priority decoding may change scheduling, never the action"
            )
            if ca and not cb:
                lost.append(pid)
        if a[pid].benchmark_correct and not b[pid].benchmark_correct:
            errs.append(f"[{pid}]: preact is incorrect where standard is correct")
        if b[pid].benchmark_correct and not a[pid].benchmark_correct:
            errs.append(f"[{pid}]: standard is incorrect where preact is correct")
    return errs, lost


def check_ready(res: dict[str, Any], arm: str, *, expect: bool) -> list[str]:
    errs: list[str] = []
    total = 0
    for traj in res["trajectories"]:
        where = f"{arm}[{traj.prompt_id}]"
        events = traj.ready or []
        total += len(events)
        if not expect:
            if events:
                errs.append(
                    f"{where}: {len(events)} tool_call_ready events with "
                    "CRYSTALCALL_PREACT=0: the added code is NOT inert"
                )
            continue
        seen: set[tuple[int, int]] = set()
        by_step = {(int(r.canvas_id), int(r.step)): r for r in traj.revisions}
        for ev in events:
            key = (int(ev["canvas_id"]), 0)
            if key in seen:
                errs.append(f"{where}: more than one readiness event for canvas {key[0]}")
            seen.add(key)
            rev = by_step.get((int(ev["canvas_id"]), int(ev["step"])))
            if rev is None:
                errs.append(
                    f"{where}: readiness at canvas {ev['canvas_id']} step {ev['step']} "
                    "has no matching revision; the join is broken"
                )
                continue
            toks = list(rev.predicted_token_ids or ())
            if not toks:
                errs.append(
                    f"{where}: the revision at step {ev['step']} carries no token payload; "
                    "the span cannot be re-derived (is CRYSTALCALL_TRACE_LEVEL 'full'?)"
                )
                continue
            open_id, close_id = _delimiters()
            want_lo = toks.index(open_id) if open_id in toks else -1
            want_hi = (
                toks.index(close_id, want_lo + 1) + 1
                if want_lo >= 0 and close_id in toks[want_lo + 1 :]
                else -1
            )
            if (want_lo, want_hi) != (int(ev["span_lo"]), int(ev["span_hi"])):
                errs.append(
                    f"{where}: span {ev['span_lo']}:{ev['span_hi']} does not re-derive from "
                    f"the recorded revision (expected {want_lo}:{want_hi})"
                )
    if expect and total == 0:
        errs.append(f"{arm}: no tool_call_ready event at all; the mechanism never fired")
    return errs


def _delimiters() -> tuple[int, int]:
    doc = yaml.safe_load(PREACT_CFG.read_text())
    toks = doc["tokens"]
    return int(toks["open_token_id"]), int(toks["close_token_id"])


# --------------------------------------------------------------------------- #


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", type=int, default=6)
    ap.add_argument("--mode", default=os.environ.get("CRYSTALCALL_PREACT_MODE", "both"))
    ap.add_argument(
        "--bound",
        type=float,
        default=(float(os.environ["CRYSTALCALL_PREACT_BOUND"])
                 if os.environ.get("CRYSTALCALL_PREACT_BOUND") else None),
        help="override the in-span entropy bound (default: configs/preact.yml)",
    )
    ap.add_argument("--json", help="write the full report here")
    ap.add_argument("--keep", action="store_true", help="retain the temporary trace workspace")
    args = ap.parse_args(argv)

    if not (REPO / "patches/build/diffusion_gemma.py").is_file():
        print("smoke_preact: run patches/apply_patch.sh first", file=sys.stderr)
        return 2
    for arm in ("diffusion_standard", PREACT_ARM):
        if arm not in ARMS:
            print(f"smoke_preact: unknown arm {arm!r}", file=sys.stderr)
            return 2
    parse_sink_endpoint(SINK)  # fail early on a malformed endpoint

    doc = yaml.safe_load(PREACT_CFG.read_text())
    bound = args.bound
    if bound is None:
        bound = float(doc["defaults"]["span_entropy_bound"])
        # The bound is a development-selected parameter. Running the arm for
        # results on an unvalidated value is how a crippled baseline reaches a
        # paper, so refuse rather than warn. An explicit
        # --bound is allowed: that is how the sweep itself probes rungs.
        sel = doc.get("selection") or {}
        if str(sel.get("status")) != "selected":
            print(
                f"smoke_preact: configs/preact.yml selection.status is "
                f"{sel.get('status')!r}, not 'selected'. Run "
                f"`uv run scripts/preact_bound_sweep.py` (pipeline stage "
                f"`preact-bound`) and record the result first, or pass an "
                f"explicit --bound to probe a value deliberately.",
                file=sys.stderr,
            )
            return 2
        report_path = REPO / str(sel.get("sweep_report") or "")
        if not report_path.is_file():
            print(f"smoke_preact: sweep report {report_path} is missing", file=sys.stderr)
            return 2
        selected = json.loads(report_path.read_text()).get("selected")
        if selected is None or float(selected) != bound:
            print(
                f"smoke_preact: configs/preact.yml has span_entropy_bound={bound} "
                f"but {report_path.name} selected {selected}; the arm would not be "
                "the tuned baseline.",
                file=sys.stderr,
            )
            return 2

    work = Path(tempfile.mkdtemp(prefix="crystalcall-smoke-preact-"))
    report: dict[str, Any] = {
        "schema_version": "crystalcall-smoke-preact/2",
        "seed": SEED,
        "prompts": args.prompts,
        "mode": args.mode,
        "span_entropy_bound": bound,
        "arms": {},
        "errors": [],
    }
    results: dict[str, dict[str, Any]] = {}
    try:
        for arm, preact in (("diffusion_standard", False), (PREACT_ARM, True)):
            stop_server()
            start_server(preact=preact, mode=args.mode, bound=bound if preact else None)
            try:
                res = run_arm(arm, work / arm, args.prompts)
            finally:
                stop_server()
            results[arm] = res
            report["arms"][arm] = {
                "summary": {k: v for k, v in res["summary"].items() if k != "shards"},
                "shard": str(res["shard"]),
                "prompts": [
                    {
                        "prompt_id": t.prompt_id,
                        "calls": canonical_calls(t),
                        "final_text_chars": len((t.final_text or "").strip()),
                        "correct": t.benchmark_correct,
                        "revisions": len(t.revisions),
                        "ready": t.ready,
                    }
                    for t in res["trajectories"]
                ],
            }
            report["errors"] += check_arm_health(res, arm)
            report["errors"] += check_ready(res, arm, expect=preact)

        errs, lost = check_same_action(results["diffusion_standard"], results[PREACT_ARM])
        report["errors"] += errs
        report["lost_calls"] = {"prompt_ids": lost, "count": len(lost)}
    finally:
        if args.keep:
            print(f"workspace kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    report["ok"] = not report["errors"]
    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
        print(f"report      {out}")
    for arm, blob in report["arms"].items():
        s = blob["summary"]
        print(
            f"{arm:20s} prompts={s.get('prompts')} turns={s.get('turns')} "
            f"failures={s.get('failures')} "
            f"ready={sum(len(p['ready']) for p in blob['prompts'])}"
        )
    if report.get("lost_calls", {}).get("count"):
        print(f"LOST CALLS  {report['lost_calls']['prompt_ids']} at bound={bound}")
    for err in report["errors"]:
        print(f"FAIL {err}", file=sys.stderr)
    print("SMOKE PASS" if report["ok"] else "SMOKE FAIL")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
