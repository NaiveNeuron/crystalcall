# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "tokenizers>=0.20", "pyyaml>=6", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Select the P-ReAct-style arm's in-span entropy bound on DEVELOPMENT data.

Why this exists: a bound of 0.6 nats looks "clearly looser" than the
sampler's own 0.1-nat cumulative rule, but measured on six development prompts
it **destroys the baseline** -- 4 of 6 prompts lost every tool call and 2 final
programs stopped parsing, because freezing in-span positions at 0.6 nats locks
in tokens the canvas would have revised.

A crippled baseline is worse than no baseline: every latency advantage
CrystalCall shows against it would be an artefact of having broken it. So the
bound is a *development-tuned* parameter, tuned **in the baseline's own
favour**: pick the largest bound (most priority, earliest readiness) that
preserves the arm's action set, its parse rate and its benchmark correctness
against the standard arm on the same prompts and seed.

The standard arm is run once as the reference; each bound then needs one server
restart, because the setting is baked into the container environment.

**Scored against ground truth, not against another stochastic run.** This
engine is not bit-reproducible across restarts for multi-canvas generations
even with a pinned server seed: the entropy-bound sampler renoises unaccepted
positions with Gumbel noise from the engine's own generator.
Comparing one arm's per-prompt output to one restart of another arm therefore
cannot attribute a difference to the mechanism, and turning that comparison
into an acceptance rule would invite a post-hoc tolerance. Each rung is
instead scored on **absolute quality against the manifest's ground truth** --
BFCL correctness (computed by the harness from the manifest's expected calls),
valid final-parse rate, and whether the turn executed any authoritative call at
all -- and required not to degrade against the standard arm's rates on the same
prompts.

Selection rule, predeclared:
  1. BFCL correctness rate >= the standard arm's;
  2. valid final-parse rate >= the standard arm's;
  3. rate of turns that executed at least one authoritative call >= the
     standard arm's -- this is the crippling mode, where the arm simply stops
     calling tools;
  4. at least one readiness event, or the mechanism is inert and the arm is not
     a tool-priority baseline at all;
  5. tie-break among the survivors: the LARGEST bound -- the most priority, and
     so the earliest readiness, that costs nothing measurable.

A second standard run (`control`) is included to **report** the engine's
run-to-run spread on these prompts. It is descriptive only: it never widens an
acceptance threshold and never excuses an incorrect call.

If nothing survives, that is reported as a negative finding about P-ReAct-style
priority on this model, not papered over.

Run:  uv run scripts/preact_bound_sweep.py [--bounds 0.05,0.1,0.2,0.3,0.45,0.6]
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
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crystalcall.harness import TraceRunConfig, run_trace_arm  # noqa: E402
from crystalcall.manifest import load_manifest, sha256_file  # noqa: E402
from crystalcall.replay import PREACT_ARM, load_trajectories  # noqa: E402

SERVE = REPO / "scripts" / "serve_diffusiongemma.sh"
PREACT_CFG = REPO / "configs" / "preact.yml"
DEV = REPO / "experiments" / "manifests" / "dev.jsonl"
OUT = REPO / "experiments" / "derived" / "preact_bound_sweep.json"
BASE = os.environ.get("CRYSTALCALL_SERVER_URL", "http://127.0.0.1:8000/v1")
SINK = os.environ.get("CRYSTALCALL_TRACE_SINK", "127.0.0.1:9802")
SEED = 0


def stratified_subset(per_category: int) -> list[Any]:
    """A deterministic spread across the four BFCL categories.

    `dev.jsonl` is grouped by category, so a bare `--limit` would tune on one
    category only.
    """
    by_cat: dict[str, list[Any]] = defaultdict(list)
    for rec in load_manifest(DEV):
        by_cat[rec.category].append(rec)
    out: list[Any] = []
    for cat in sorted(by_cat):
        out.extend(by_cat[cat][:per_category])
    return out


def write_subset(records: list[Any], path: Path) -> Path:
    """A development-only manifest slice; the shard header records its sha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_json(), sort_keys=True) + "\n")
    return path


def start_server(*, preact: bool, bound: float | None, mode: str) -> None:
    env = dict(os.environ)
    env["CRYSTALCALL_TRACE"] = "1"
    env["CRYSTALCALL_TRACE_SINK"] = SINK
    env["CRYSTALCALL_TRACE_LEVEL"] = "full"
    env["CRYSTALCALL_PREACT"] = "1" if preact else "0"
    env["CRYSTALCALL_PREACT_MODE"] = mode
    if bound is not None:
        env["CRYSTALCALL_PREACT_BOUND"] = str(bound)
    subprocess.run([str(SERVE)], env=env, check=True)


def stop_server() -> None:
    subprocess.run(
        [str(SERVE), "--stop"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    time.sleep(1.0)


def canonical_calls(traj: Any) -> list[str]:
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


def load_arm_from_shard(shard: Path, records: list[Any]) -> dict[str, Any]:
    """Reuse an already-recorded run as the reference.

    The shard is hash-chain verified on load, so a reused reference is exactly
    as trustworthy as a freshly generated one.
    """
    trajs = load_trajectories([shard], verify=True)
    if not trajs:
        raise SystemExit(f"reference shard {shard} contains no trajectory")
    arms = {t.arm for t in trajs}
    if arms != {"diffusion_standard"}:
        raise SystemExit(
            f"reference shard {shard} is arm(s) {sorted(arms)}, expected "
            "diffusion_standard: the reference must be the unmodified sampler"
        )
    ready = sum(len(t.ready or ()) for t in trajs)
    if ready:
        raise SystemExit(
            f"reference shard {shard} carries {ready} tool_call_ready events, so it "
            "was produced with CRYSTALCALL_PREACT=1; it cannot be the reference"
        )
    # `rates()` scores each arm over its own `by_prompt`, so a reference on a
    # different prompt set or seed would silently produce non-comparable rates.
    want = {r.prompt_id for r in records}
    got = {t.prompt_id for t in trajs}
    if got != want:
        raise SystemExit(
            f"reference shard {shard} covers {len(got)} prompts, the manifest {len(want)}; "
            f"missing {sorted(want - got)[:5]}, extra {sorted(got - want)[:5]}"
        )
    seeds = {int(t.seed) for t in trajs}
    if seeds != {SEED}:
        raise SystemExit(f"reference shard {shard} has seeds {sorted(seeds)}, expected [{SEED}]")
    return {
        "summary": {"prompts": len(trajs), "turns": None, "reused_shard": str(shard)},
        "by_prompt": {
            t.prompt_id: {
                "calls": canonical_calls(t),
                "parse_ok": bool(t.valid_final_parse),
                "correct": t.benchmark_correct,
                "category": t.category,
                "ready": list(t.ready or ()),
                "revisions": len(t.revisions),
                "decode_ms": t.decode_ms,
                "final_text_chars": len((t.final_text or "").strip()),
            }
            for t in trajs
        },
    }


def run_arm(arm: str, manifest: Path, out_dir: Path) -> dict[str, Any]:
    cfg = TraceRunConfig(
        root=REPO,
        manifest=manifest,
        arm=arm,
        seeds=(SEED,),
        out_dir=out_dir,
        server_url=BASE,
        trace_sink_endpoint=SINK,
    )
    summary = run_trace_arm(cfg)
    shard = Path(summary["shards"][0]["path"])
    trajs = load_trajectories([shard], verify=True)
    return {
        "summary": {k: v for k, v in summary.items() if k != "shards"},
        "by_prompt": {
            t.prompt_id: {
                "calls": canonical_calls(t),
                "parse_ok": bool(t.valid_final_parse),
                "correct": t.benchmark_correct,
                "category": t.category,
                "ready": list(t.ready or ()),
                "revisions": len(t.revisions),
                "decode_ms": t.decode_ms,
                "final_text_chars": len((t.final_text or "").strip()),
            }
            for t in trajs
        },
    }


def rates(arm: dict[str, Any]) -> dict[str, Any]:
    """Absolute quality of one run, scored against the manifest's ground truth.

    ``benchmark_correct`` is BFCL AST correctness computed by the harness from
    the manifest's expected calls, so these are ground-truth rates and not
    agreement with another stochastic run.
    """
    by = arm["by_prompt"]
    rows = list(by.values())
    n = len(rows) or 1
    correct = sum(1 for r in rows if r["correct"] is True)
    parsed = sum(1 for r in rows if r["parse_ok"])
    called = sum(1 for r in rows if r["calls"])
    return {
        "n_prompts": len(rows),
        "correct": correct,
        "correct_rate": correct / n,
        "parse_ok": parsed,
        "parse_rate": parsed / n,
        "called_any": called,
        "called_any_rate": called / n,
        "ready_events": sum(len(r["ready"]) for r in rows),
        "prompts_with_ready": sum(1 for r in rows if r["ready"]),
        "no_call_prompt_ids": sorted(pid for pid, r in by.items() if not r["calls"]),
        "incorrect_prompt_ids": sorted(pid for pid, r in by.items() if r["correct"] is not True),
        "unparsed_prompt_ids": sorted(pid for pid, r in by.items() if not r["parse_ok"]),
    }


def divergence(reference: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Per-prompt output differences. Descriptive only, never an acceptance
    criterion: a plain restart of the same arm also differs."""
    ref, got = reference["by_prompt"], arm["by_prompt"]
    shared = sorted(set(ref) & set(got))
    differing = [pid for pid in shared if ref[pid]["calls"] != got[pid]["calls"]]
    return {
        "n_shared": len(shared),
        "differing_calls": differing,
        "differ_rate": len(differing) / (len(shared) or 1),
    }


def score(reference: dict[str, Any], arm: dict[str, Any]) -> dict[str, Any]:
    """Judge one rung: no degradation in any ground-truth rate, and it fired."""
    ref_r, got_r = rates(reference), rates(arm)
    reasons: list[str] = []
    for key, label in (
        ("correct_rate", "BFCL correctness"),
        ("parse_rate", "valid final parse"),
        ("called_any_rate", "turns that executed a call"),
    ):
        if got_r[key] < ref_r[key]:
            reasons.append(f"{label} degraded: {got_r[key]:.0%} vs standard {ref_r[key]:.0%}")
    if got_r["ready_events"] == 0:
        reasons.append("no readiness event: the mechanism never fired")
    return {
        "rates": got_r,
        "reference_rates": ref_r,
        "divergence": divergence(reference, arm),
        "reject_reasons": reasons,
        "acceptable": not reasons,
    }


def rescore(path: Path) -> int:
    """Re-apply the selection rule offline to a stored report.

    The per-prompt records already hold everything the rule needs, so revising
    the rule never costs another hour of GPU time, and the recomputation stays
    auditable against the same raw rows.
    """
    report = json.loads(path.read_text(encoding="utf-8"))
    reference = report["reference"]
    report["reference_rates"] = rates(reference)
    for rung in report["rungs"]:
        rung["score"] = score(reference, rung["arm"])
        r = rung["score"]["rates"]
        verdict = (
            "OK"
            if rung["score"]["acceptable"]
            else "REJECT: " + "; ".join(rung["score"]["reject_reasons"])
        )
        print(
            f"bound {float(rung['bound']):<5} correct={r['correct_rate']:.0%} "
            f"parse={r['parse_rate']:.0%} called={r['called_any_rate']:.0%} "
            f"ready={r['ready_events']} -> {verdict}"
        )
    if isinstance(report.get("control"), dict) and "arm" in report["control"]:
        report["control"]["rates"] = rates(report["control"]["arm"])
        report["control"]["divergence_vs_reference"] = divergence(
            reference, report["control"]["arm"]
        )
    ok = [r for r in report["rungs"] if r["score"]["acceptable"]]
    report["selected"] = max((float(r["bound"]) for r in ok), default=None)
    report["selection_rule"] = SELECTION_RULE
    report["rescored"] = True
    path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    ref = report["reference_rates"]
    print(
        f"\nstandard    correct={ref['correct_rate']:.0%} parse={ref['parse_rate']:.0%} "
        f"called={ref['called_any_rate']:.0%} (n={ref['n_prompts']})"
    )
    if report["selected"] is None:
        print("\nNO ACCEPTABLE BOUND at any swept value.", file=sys.stderr)
        return 1
    print(f"\nSELECTED span_entropy_bound = {report['selected']}")
    return 0


SELECTION_RULE = (
    "largest bound whose BFCL correctness, valid-parse and executed-a-call "
    "rates are all >= the standard arm's on the same prompts, and which "
    "produced at least one readiness event; per-prompt agreement is reported "
    "but is never an acceptance criterion, because the engine is not "
    "restart-reproducible"
)


def write_selection(report: dict[str, Any]) -> None:
    """Record the outcome in `configs/preact.yml` so the arm's guard is
    satisfied mechanically, never by hand.

    On success the bound and its provenance are written and `status` becomes
    `selected`. On failure the mechanism is switched to `ready_signal`, which
    emits readiness without touching the acceptance mask and therefore cannot
    change the trajectory -- the arm is degraded honestly rather than shipped
    with a knowingly destructive setting.
    """
    import datetime

    selected = report.get("selected")
    text = PREACT_CFG.read_text(encoding="utf-8")
    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    digest = sha256_file(OUT) if OUT.is_file() else None
    if selected is None:
        block = [
            "selection:",
            "  status: rejected",
            f"  selected_on: {stamp}",
            f"  sweep_report: {OUT.relative_to(REPO).as_posix()}",
            f"  sweep_sha256: {digest}",
            "  note: >",
            "    No swept bound preserved the arm's ground-truth rates, so priority",
            "    acceptance is disabled and the arm runs readiness-only.",
        ]
        text = _replace_scalar(text, "mode", "ready_signal")
    else:
        block = [
            "selection:",
            "  status: selected",
            f"  selected_on: {stamp}",
            f"  sweep_report: {OUT.relative_to(REPO).as_posix()}",
            f"  sweep_sha256: {digest}",
            f"  prompts: {len(report.get('prompt_ids') or ())}",
            f"  rule: {json.dumps(report.get('selection_rule', ''))}",
        ]
        text = _replace_scalar(text, "span_entropy_bound", repr(float(selected)))
    text = _replace_block(text, "selection:", "\n".join(block))
    PREACT_CFG.write_text(text, encoding="utf-8")
    print(f"recorded selection in {PREACT_CFG.relative_to(REPO)}")


def _replace_scalar(text: str, key: str, value: str) -> str:
    out = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{key}:") and line.startswith("  "):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{key}: {value}")
        else:
            out.append(line)
    return "\n".join(out) + "\n"


def _replace_block(text: str, header: str, block: str) -> str:
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.startswith(header))
    except StopIteration:
        return text.rstrip("\n") + "\n\n" + block + "\n"
    end = start + 1
    while end < len(lines) and (not lines[end].strip() or lines[end].startswith((" ", "\t"))):
        end += 1
    return "\n".join(lines[:start] + block.splitlines() + lines[end:]) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bounds", default="0.05,0.1,0.2,0.3,0.45,0.6")
    ap.add_argument("--per-category", type=int, default=3)
    ap.add_argument("--mode", default="both")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--no-control", action="store_true", help="skip the descriptive second standard run")
    ap.add_argument(
        "--write-selection",
        action="store_true",
        help="record the outcome into configs/preact.yml's `selection:` block, so "
        "the arm's guard is satisfied mechanically instead of by hand",
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        help="score on this manifest instead of a stratified dev subset "
        "(use experiments/manifests/dev.jsonl for the full development set)",
    )
    ap.add_argument(
        "--reference-shard",
        type=Path,
        help="reuse an existing standard-arm shard as the reference instead of "
        "re-running it; must have been produced with CRYSTALCALL_PREACT=0 on the "
        "same manifest and seed",
    )
    ap.add_argument(
        "--rescore",
        metavar="REPORT",
        help="re-apply the selection rule to a previous report's stored per-prompt "
        "data and exit, without touching the GPU",
    )
    args = ap.parse_args(argv)

    if args.rescore:
        return rescore(Path(args.rescore))

    bounds = sorted({float(x) for x in args.bounds.split(",") if x.strip()})
    work = Path(tempfile.mkdtemp(prefix="crystalcall-bound-sweep-"))
    if args.manifest:
        manifest = args.manifest
        records = load_manifest(manifest)
        label = f"manifest {manifest.name}"
    else:
        records = stratified_subset(args.per_category)
        if not records:
            print("preact_bound_sweep: empty development subset", file=sys.stderr)
            return 2
        manifest = write_subset(records, work / "dev_subset.jsonl")
        label = f"stratified subset -> {manifest.name}"
    print(
        f"prompts     {len(records)}, "
        f"{sorted({r.category for r in records})}, {label}",
        flush=True,
    )

    report: dict[str, Any] = {
        "schema_version": "crystalcall-preact-sweep/1",
        "seed": SEED,
        "mode": args.mode,
        "bounds": bounds,
        "per_category": args.per_category,
        "prompt_ids": [r.prompt_id for r in records],
        "rungs": [],
    }
    try:
        if args.reference_shard:
            reference = load_arm_from_shard(args.reference_shard, records)
            print(f"reference   reused {args.reference_shard}", flush=True)
        else:
            stop_server()
            start_server(preact=False, bound=None, mode=args.mode)
            try:
                reference = run_arm("diffusion_standard", manifest, work / "standard")
            finally:
                stop_server()
        report["reference"] = reference
        report["reference_shard"] = str(args.reference_shard) if args.reference_shard else None
        report["reference_rates"] = rates(reference)
        rr = report["reference_rates"]
        print(
            f"reference   turns={reference['summary'].get('turns')} "
            f"prompts={rr['n_prompts']} correct={rr['correct_rate']:.0%} "
            f"parse={rr['parse_rate']:.0%} called={rr['called_any_rate']:.0%}",
            flush=True,
        )

        # Descriptive only: the same arm and settings, second server restart.
        # It quantifies the engine's run-to-run spread on these prompts and
        # never widens an acceptance threshold.
        if not args.no_control:
            stop_server()
            start_server(preact=False, bound=None, mode=args.mode)
            try:
                control = run_arm("diffusion_standard", manifest, work / "control")
            finally:
                stop_server()
            report["control"] = {
                "arm": control,
                "rates": rates(control),
                "divergence_vs_reference": divergence(reference, control),
                "note": "nondeterminism measurement; not an acceptance criterion",
            }
            cr = report["control"]["rates"]
            print(
                f"control     correct={cr['correct_rate']:.0%} parse={cr['parse_rate']:.0%} "
                f"called={cr['called_any_rate']:.0%} differ_vs_ref="
                f"{report['control']['divergence_vs_reference']['differ_rate']:.0%}"
                "  <- engine nondeterminism, descriptive",
                flush=True,
            )

        for bound in bounds:
            stop_server()
            start_server(preact=True, bound=bound, mode=args.mode)
            try:
                arm = run_arm(PREACT_ARM, manifest, work / f"b{bound}")
            finally:
                stop_server()
            sc = score(reference, arm)
            report["rungs"].append({"bound": bound, "score": sc, "arm": arm})
            r = sc["rates"]
            verdict = "OK" if sc["acceptable"] else "REJECT: " + "; ".join(sc["reject_reasons"])
            print(
                f"bound {bound:<5} correct={r['correct_rate']:.0%} "
                f"parse={r['parse_rate']:.0%} called={r['called_any_rate']:.0%} "
                f"ready={r['ready_events']} -> {verdict}",
                flush=True,
            )

        ok = [r for r in report["rungs"] if r["score"]["acceptable"]]
        report["selected"] = max((float(r["bound"]) for r in ok), default=None)
        report["selection_rule"] = SELECTION_RULE
    finally:
        if args.keep:
            print(f"workspace kept at {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report      {OUT}")

    if args.write_selection:
        write_selection(report)

    if report.get("selected") is None:
        print(
            "\nNO ACCEPTABLE BOUND. Priority acceptance cannot be made "
            "non-destructive on this model at any swept value: every bound "
            "either loses a tool call the standard arm makes, breaks the final "
            "parse, regresses correctness, or never fires. This is a finding "
            "about P-ReAct-style priority on DiffusionGemma, and the arm must "
            "be reported as ready-signal-only (mode=ready_signal, which cannot "
            "change the trajectory) rather than silently shipped broken.",
            file=sys.stderr,
        )
        return 1
    print(f"\nSELECTED span_entropy_bound = {report['selected']}")
    print(f"Write it into {PREACT_CFG.relative_to(REPO)} and re-freeze.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
