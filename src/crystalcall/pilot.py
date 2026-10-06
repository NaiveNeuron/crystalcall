"""Phase 1: the 100-prompt kill-or-continue pilot, and development-set
threshold selection.

The pilot is computed entirely from stored standard-DiffusionGemma traces, so
it is reproducible and costs no model time to re-evaluate.  Its decision is
mechanical: every gate must pass.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .replay import (
    LATENCY_CONDITIONS_MS,
    Trajectory,
    analyze_trajectory,
    build_views,
    load_trajectories,
    simulate_policy,
    write_jsonl,
)
from .scheduler import ConfidenceMode, CostModel, GatePolicy, RemainingDecodeModel

__all__ = ["PILOT_GATES", "run_pilot", "select_thresholds", "PilotGate"]


@dataclass(frozen=True, slots=True)
class PilotGate:
    key: str
    description: str
    threshold: float
    comparison: str = ">="
    unit: str = ""

    def passes(self, value: float | None) -> bool:
        if value is None:
            return False
        return value >= self.threshold if self.comparison == ">=" else value <= self.threshold


PILOT_GATES: tuple[PilotGate, ...] = (
    PilotGate("valid_final_ptc_rate", "Valid final PTC programs", 0.70, unit="fraction"),
    PilotGate(
        "oracle_precommit_lead_ok",
        "Median oracle pre-commit lead >= 200 ms OR >= 15% of remaining decode",
        1.0,
        unit="boolean",
    ),
    PilotGate("gate_precision", "Conservative launch precision", 0.98, unit="fraction"),
    PilotGate("eligible_coverage", "Coverage of eligible final calls", 0.35, unit="fraction"),
    PilotGate("median_speedup_500ms", "Projected median speedup at 500 ms tools", 0.15),
    PilotGate("median_speedup_2s", "Projected median speedup at 2 s tools", 0.25),
)

#: The conservative gate used for the pilot decision, before any tuning.
CONSERVATIVE_GATE = GatePolicy(
    name="pilot_conservative",
    k=2,
    tau=0.9,
    confidence_mode=ConfidenceMode.MIN,
    min_overlap_ms=50.0,
    max_inflight=4,
)

#: Development-set search space.  These are the ONLY tuned quantities.
K_GRID: tuple[int, ...] = (1, 2, 3, 4)
TAU_GRID: tuple[float, ...] = (0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)
OVERLAP_GRID_MS: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0)


# --------------------------------------------------------------------------- #
# metric computation
# --------------------------------------------------------------------------- #


def _median(xs: Sequence[float]) -> float | None:
    vals = [x for x in xs if x is not None]
    return statistics.median(vals) if vals else None


def _percentile(xs: Sequence[float], p: float) -> float | None:
    vals = sorted(x for x in xs if x is not None)
    if not vals:
        return None
    idx = min(len(vals) - 1, int(round(p / 100.0 * (len(vals) - 1))))
    return vals[idx]


def crystallization_summary(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Milestone distributions over every authoritative final call."""
    calls = [c for a in analyses for c in a.get("final_calls", [])]
    eligible = [c for c in calls if c.get("eligible")]

    def rel(field: str, base: str = "response_done_ns") -> list[float]:
        out: list[float] = []
        for c in calls:
            v, b = c.get(field), c.get(base)
            if v and b:
                out.append((b - v) / 1e6)
        return out

    milestones = {
        "first_final_name": rel("first_name_ns"),
        "first_final_arguments": rel("first_args_ns"),
        "first_executable_final_match": rel("first_exec_ns"),
        "first_invariant_through_final": rel("invariant_ns"),
        "first_fully_committed": rel("committed_ns"),
        "canvas_completion": rel("canvas_done_ns"),
    }
    leads = [c["lead_ms"] for c in calls if c.get("lead_ms") is not None]
    eligible_leads = [c["lead_ms"] for c in eligible if c.get("lead_ms") is not None]
    decode_ms = [a["decode_ms"] for a in analyses if a.get("decode_ms")]

    lead_fractions: list[float] = []
    for c in calls:
        lead, resp, inv = c.get("lead_ms"), c.get("response_done_ns"), c.get("invariant_ns")
        if lead is None or not resp or not inv:
            continue
        remaining_ms = (resp - inv) / 1e6
        if remaining_ms > 0:
            lead_fractions.append(min(1.0, lead / remaining_ms))

    return {
        "final_calls": len(calls),
        "eligible_final_calls": len(eligible),
        "trajectories": len(analyses),
        "milestones_ms_before_response": {
            k: {
                "n": len(v),
                "p50": _median(v),
                "p95": _percentile(v, 95),
                "mean": statistics.fmean(v) if v else None,
            }
            for k, v in milestones.items()
        },
        "precommit_lead_ms": {
            "n": len(leads),
            "p50": _median(leads),
            "p95": _percentile(leads, 95),
            "eligible_p50": _median(eligible_leads),
        },
        "precommit_lead_fraction_of_remaining": {
            "n": len(lead_fractions),
            "p50": _median(lead_fractions),
        },
        "decode_ms": {"p50": _median(decode_ms), "p95": _percentile(decode_ms, 95)},
        "mutations": {
            "p50": _median([a.get("mutations", 0) for a in analyses]),
            "p95": _percentile([a.get("mutations", 0) for a in analyses], 95),
        },
        "first_canvas_only_fraction": (
            sum(1 for c in calls if c.get("first_canvas_only")) / len(calls) if calls else None
        ),
        "by_category": _by_category(analyses),
    }


def _by_category(analyses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for a in analyses:
        cat = a.get("category", "unknown")
        bucket = out.setdefault(
            cat, {"trajectories": 0, "valid_parse": 0, "final_calls": 0, "leads": []}
        )
        bucket["trajectories"] += 1
        bucket["valid_parse"] += 1 if a.get("valid_final_parse") else 0
        bucket["final_calls"] += len(a.get("final_calls", []))
        for c in a.get("final_calls", []):
            if c.get("lead_ms") is not None:
                bucket["leads"].append(c["lead_ms"])
    for cat, b in out.items():
        leads = b.pop("leads")
        b["lead_ms_p50"] = _median(leads)
        b["valid_parse_rate"] = b["valid_parse"] / b["trajectories"] if b["trajectories"] else None
    return out


def gate_metrics(
    trajectories: Sequence[Trajectory],
    views_by_id: Mapping[str, Sequence[Any]],
    analyses_by_id: Mapping[str, Any],
    registries: Mapping[str, Any],
    policy: GatePolicy,
    *,
    remaining_model: RemainingDecodeModel,
    costs: CostModel,
    latencies: Sequence[float] = LATENCY_CONDITIONS_MS,
) -> dict[str, Any]:
    """Precision, coverage and projected speedups for one gate policy."""
    launches = 0
    true_launches = 0
    eligible = 0
    covered = 0
    per_latency: dict[float, list[float]] = {lat: [] for lat in latencies}
    baseline_per_latency: dict[float, list[float]] = {lat: [] for lat in latencies}

    for traj in trajectories:
        reg = registries.get(traj.prompt_id)
        analysis = analyses_by_id.get(traj.request_id)
        if reg is None or analysis is None:
            continue
        views = views_by_id[traj.request_id]
        for lat in latencies:
            out = simulate_policy(
                traj,
                views,
                analysis,
                reg,
                policy,
                lat,
                remaining_model=remaining_model,
                costs=costs,
            )
            per_latency[lat].append(out.completion_ms)
            baseline_per_latency[lat].append(out.baseline_completion_ms)
            if lat == latencies[0]:
                launches += out.launches
                true_launches += out.true_launches
                eligible += out.eligible_final_calls
                covered += out.covered_final_calls

    speedups: dict[str, Any] = {}
    for lat in latencies:
        pairs = [
            (b - c) / b
            for c, b in zip(per_latency[lat], baseline_per_latency[lat], strict=True)
            if b > 0
        ]
        speedups[f"{int(lat)}ms"] = {
            "n": len(pairs),
            "median_gain": _median(pairs),
            "p95_gain": _percentile(pairs, 95),
            "completion_p50_ms": _median(per_latency[lat]),
            "baseline_p50_ms": _median(baseline_per_latency[lat]),
        }

    denom = true_launches + (launches - true_launches)
    return {
        "policy": policy.to_json(),
        "policy_fingerprint": policy.fingerprint(),
        "launches": launches,
        "true_launches": true_launches,
        "false_launches": launches - true_launches,
        "precision": (true_launches / denom) if denom else None,
        "eligible_final_calls": eligible,
        "covered_final_calls": covered,
        "coverage": (covered / eligible) if eligible else None,
        "speedups": speedups,
    }


# --------------------------------------------------------------------------- #
# threshold selection (development data only)
# --------------------------------------------------------------------------- #


def select_thresholds(
    trajectories: Sequence[Trajectory],
    views_by_id: Mapping[str, Sequence[Any]],
    analyses_by_id: Mapping[str, Any],
    registries: Mapping[str, Any],
    *,
    remaining_model: RemainingDecodeModel,
    costs: CostModel,
    min_precision: float = 0.99,
) -> dict[str, Any]:
    """Grid-search k, tau and minimum useful overlap on development data.

    Objective: maximise eligible-call coverage subject to launch precision at
    or above ``min_precision``; ties are broken by the larger median gain at
    500 ms and then by the smaller k.
    """
    rows: list[dict[str, Any]] = []
    for k in K_GRID:
        for tau in TAU_GRID:
            for overlap in OVERLAP_GRID_MS:
                pol = replace(
                    GatePolicy(name=f"dev_k{k}_tau{tau:g}_ov{int(overlap)}"),
                    k=k,
                    tau=tau,
                    min_overlap_ms=overlap,
                )
                m = gate_metrics(
                    trajectories,
                    views_by_id,
                    analyses_by_id,
                    registries,
                    pol,
                    remaining_model=remaining_model,
                    costs=costs,
                    latencies=(500.0, 2000.0),
                )
                rows.append(
                    {
                        "k": k,
                        "tau": tau,
                        "min_overlap_ms": overlap,
                        "precision": m["precision"],
                        "coverage": m["coverage"],
                        "launches": m["launches"],
                        "median_gain_500ms": m["speedups"]["500ms"]["median_gain"],
                        "median_gain_2000ms": m["speedups"]["2000ms"]["median_gain"],
                    }
                )

    feasible = [
        r
        for r in rows
        if r["precision"] is not None
        and r["precision"] >= min_precision
        and r["coverage"] is not None
        and r["launches"] > 0
    ]
    pool = feasible or [r for r in rows if r["launches"] > 0]
    if not pool:
        return {"status": "no_launches", "grid": rows}
    best = max(
        pool,
        key=lambda r: (
            r["coverage"] or 0.0,
            r["median_gain_500ms"] or 0.0,
            -r["k"],
            -r["min_overlap_ms"],
        ),
    )
    return {
        "status": "ok" if feasible else "precision_target_unreachable",
        "min_precision": min_precision,
        "selected": best,
        "grid": rows,
    }


# --------------------------------------------------------------------------- #
# the pilot
# --------------------------------------------------------------------------- #


def run_pilot(
    *, root: Path, attempt: int = 1, shards: Sequence[Path] | None = None
) -> dict[str, Any]:
    from .harness import load_tokenizer, resolve_model_dir
    from .manifest import load_manifest, registry_for
    from .parse_ptc import PtcExtractor
    from .tokens import CanvasDecoder

    root = Path(root)
    shards = list(shards or sorted((root / "experiments/raw/diffusion_standard").glob("*.jsonl")))
    if not shards:
        raise FileNotFoundError(
            "no standard DiffusionGemma trace shards under experiments/raw/diffusion_standard"
        )

    trajectories = load_trajectories(shards, verify=True)
    dev = load_manifest(root / "experiments/manifests/dev.jsonl")
    registries = {rec.prompt_id: registry_for(rec) for rec in dev}
    dev_ids = set(registries)
    trajectories = [t for t in trajectories if t.prompt_id in dev_ids]
    if not trajectories:
        raise RuntimeError("no development trajectories matched the frozen dev manifest")

    decoder = CanvasDecoder(tokenizer=load_tokenizer(resolve_model_dir("google/diffusiongemma-26B-A4B-it")), cache_size=64)
    costs = _pilot_costs(root)
    remaining = _pilot_remaining(root, trajectories)

    views_by_id: dict[str, Any] = {}
    analyses_by_id: dict[str, Any] = {}
    analyses: list[dict[str, Any]] = []
    for traj in trajectories:
        reg = registries[traj.prompt_id]
        views = build_views(traj, PtcExtractor(registry=reg), decoder)
        analysis = analyze_trajectory(traj, views, reg)
        views_by_id[traj.request_id] = views
        analyses_by_id[traj.request_id] = analysis
        analyses.append(analysis.to_json())

    crystal = crystallization_summary(analyses)
    conservative = gate_metrics(
        trajectories,
        views_by_id,
        analyses_by_id,
        registries,
        CONSERVATIVE_GATE,
        remaining_model=remaining,
        costs=costs,
    )
    selection = select_thresholds(
        trajectories,
        views_by_id,
        analyses_by_id,
        registries,
        remaining_model=remaining,
        costs=costs,
    )

    valid_rate = (
        sum(1 for a in analyses if a.get("valid_final_parse")) / len(analyses) if analyses else 0.0
    )
    lead_p50 = crystal["precommit_lead_ms"]["p50"]
    lead_frac_p50 = crystal["precommit_lead_fraction_of_remaining"]["p50"]
    lead_ok = 1.0 if ((lead_p50 or 0) >= 200.0 or (lead_frac_p50 or 0) >= 0.15) else 0.0

    values = {
        "valid_final_ptc_rate": valid_rate,
        "oracle_precommit_lead_ok": lead_ok,
        "gate_precision": conservative["precision"],
        "eligible_coverage": conservative["coverage"],
        "median_speedup_500ms": conservative["speedups"]["500ms"]["median_gain"],
        "median_speedup_2s": conservative["speedups"]["2000ms"]["median_gain"],
    }
    results = [
        {
            "gate": g.key,
            "description": g.description,
            "threshold": g.threshold,
            "value": values.get(g.key),
            "passed": g.passes(values.get(g.key)),
        }
        for g in PILOT_GATES
    ]
    decision = "continue" if all(r["passed"] for r in results) else "stop"

    report = {
        "schema_version": "crystalcall-pilot/1",
        "attempt": attempt,
        "unix_ns": time.time_ns(),
        # The identity of the *configuration* this attempt measured. The
        # pilot allows one formatting-only prompt repair, so "which attempt
        # is this" is a question about the prompt, not about which files happen
        # to exist: a fresh clone reproducing attempt 2 has attempt 2's prompt
        # and must be allowed to run.
        "system_prompt_sha256": _sha256_or_none(root / "configs/system_prompt.md"),
        "shards": [str(s) for s in shards],
        "trajectories": len(trajectories),
        "excluded": [a["trajectory_id"] for a in analyses if a.get("exclusion")],
        "decision": decision,
        "gates": results,
        "crystallization": crystal,
        "conservative_gate": conservative,
        "threshold_selection": selection,
        "costs": costs.to_json(),
        "remaining_decode_model": remaining.to_json(),
        "decoder_fallbacks": decoder.fallbacks,
    }

    derived = root / "experiments/derived"
    write_jsonl(derived / f"pilot_analyses_attempt{attempt}.jsonl", analyses)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / f"reports/pilot_attempt{attempt}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n"
    )
    md = render_pilot_report(report)
    (root / "reports/pilot_report.md").write_text(md)
    if decision == "continue" and selection.get("status") in ("ok",):
        _freeze_policy(root, selection, costs, remaining)
    report["summary"] = md
    return report


def _freeze_policy(
    root: Path, selection: Mapping[str, Any], costs: CostModel, remaining: RemainingDecodeModel
) -> Path:
    sel = selection["selected"]
    frozen = {
        "schema_version": "crystalcall-policy/1",
        "selected_on": "experiments/manifests/dev.jsonl",
        "min_precision": selection.get("min_precision"),
        "gate": {
            "k": sel["k"],
            "tau": sel["tau"],
            "min_overlap_ms": sel["min_overlap_ms"],
            "confidence_mode": "min",
        },
        "dev_metrics": {
            "precision": sel["precision"],
            "coverage": sel["coverage"],
            "median_gain_500ms": sel["median_gain_500ms"],
            "median_gain_2000ms": sel["median_gain_2000ms"],
        },
        "costs": costs.to_json(),
        "remaining_decode": remaining.to_json(),
        "frozen_unix_ns": time.time_ns(),
    }
    p = root / "configs/frozen_policy.json"
    p.write_text(json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    return p


def _pilot_costs(root: Path) -> CostModel:
    """Cost model from the measured Phase 0 overhead ladder, if present."""
    d = root / "experiments/derived/overhead"
    trace_ms = parse_ms = 0.0
    if d.exists():
        levels: dict[str, float] = {}
        revisions: dict[str, float] = {}
        for p in sorted(d.glob("overhead-*.json")):
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            levels[rec.get("level", "off")] = float(rec.get("p50_ms", 0.0))
            revisions[rec.get("level", "off")] = float(rec.get("revisions_per_request", 0.0) or 0.0)
        n = max(revisions.values() or [0.0]) or 1.0
        if "full" in levels and "off" in levels:
            trace_ms = max(0.0, (levels["full"] - levels["off"]) / n)
        if "parse" in levels and "full" in levels:
            parse_ms = max(0.0, (levels["parse"] - levels["full"]) / n)
    return CostModel(trace_ms_per_revision=trace_ms, parse_ms_per_revision=parse_ms)


def _sha256_or_none(path: Path) -> str | None:
    from .manifest import sha256_file

    return sha256_file(path) if path.is_file() else None


def _pilot_remaining(root: Path, trajectories: Sequence[Trajectory]) -> RemainingDecodeModel:
    """Calibrate the remaining-decode estimator on development trajectories."""
    steps: list[int] = []
    canvases: list[int] = []
    step_ms: list[float] = []
    for t in trajectories:
        by_canvas: dict[int, list[Any]] = {}
        for r in t.revisions:
            by_canvas.setdefault(r.canvas_id, []).append(r)
        if not by_canvas:
            continue
        canvases.append(len(by_canvas))
        for revs in by_canvas.values():
            steps.append(len(revs))
        ordered = sorted(t.revisions, key=lambda r: r.monotonic_ns)
        for a, b in zip(ordered, ordered[1:], strict=False):
            dt = (b.monotonic_ns - a.monotonic_ns) / 1e6
            if 0 < dt < 5000:
                step_ms.append(dt)
    return RemainingDecodeModel(
        expected_steps_per_canvas=statistics.fmean(steps) if steps else 24.0,
        expected_canvases=statistics.fmean(canvases) if canvases else 1.0,
        default_step_ms=statistics.median(step_ms) if step_ms else 40.0,
    )


# --------------------------------------------------------------------------- #
# report rendering
# --------------------------------------------------------------------------- #


def _fmt(v: Any, pct: bool = False, digits: int = 3) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if pct:
        return f"{100 * float(v):.1f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def render_pilot_report(report: Mapping[str, Any]) -> str:
    c = report["crystallization"]
    lines: list[str] = []
    lines.append("# CrystalCall pilot report")
    lines.append("")
    lines.append(f"- attempt: **{report['attempt']}**")
    lines.append(f"- decision: **{report['decision'].upper()}**")
    lines.append(f"- development trajectories: {report['trajectories']}")
    lines.append(f"- authoritative final calls: {c['final_calls']} "
                 f"(speculation-eligible: {c['eligible_final_calls']})")
    lines.append(f"- excluded trajectories: {len(report['excluded'])}")
    lines.append(f"- shards: {', '.join(Path(s).name for s in report['shards'])}")
    lines.append("")
    lines.append("## Go/no-go gates")
    lines.append("")
    lines.append("| Gate | Required | Observed | Result |")
    lines.append("| --- | ---: | ---: | :--- |")
    for g in report["gates"]:
        pct = g["gate"] not in ("oracle_precommit_lead_ok",)
        lines.append(
            f"| {g['description']} | {_fmt(g['threshold'], pct=pct)} | "
            f"{_fmt(g['value'], pct=pct)} | {'PASS' if g['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append("## Crystallization milestones")
    lines.append("")
    lines.append("Milliseconds *before* response completion (larger = earlier).")
    lines.append("")
    lines.append("| Milestone | n | p50 ms | p95 ms |")
    lines.append("| --- | ---: | ---: | ---: |")
    for k, v in c["milestones_ms_before_response"].items():
        lines.append(f"| {k} | {v['n']} | {_fmt(v['p50'], digits=1)} | {_fmt(v['p95'], digits=1)} |")
    lines.append("")
    lines.append(
        f"Pre-commit lead: p50 {_fmt(c['precommit_lead_ms']['p50'], digits=1)} ms, "
        f"p95 {_fmt(c['precommit_lead_ms']['p95'], digits=1)} ms, "
        f"median fraction of remaining decode "
        f"{_fmt(c['precommit_lead_fraction_of_remaining']['p50'], pct=True)}."
    )
    lines.append(
        f"Decode time: p50 {_fmt(c['decode_ms']['p50'], digits=1)} ms; "
        f"provisional mutations per trajectory: p50 {_fmt(c['mutations']['p50'], digits=1)}; "
        f"single-canvas calls: {_fmt(c['first_canvas_only_fraction'], pct=True)}."
    )
    lines.append("")
    lines.append("## Conservative gate")
    lines.append("")
    cg = report["conservative_gate"]
    lines.append(
        f"Policy `{cg['policy']['name']}` (k={cg['policy']['k']}, tau={cg['policy']['tau']}, "
        f"min overlap {cg['policy']['min_overlap_ms']} ms): "
        f"{cg['launches']} launches, {cg['false_launches']} false, "
        f"precision {_fmt(cg['precision'], pct=True)}, coverage {_fmt(cg['coverage'], pct=True)}."
    )
    lines.append("")
    lines.append("| Tool duration | median gain | p95 gain | completion p50 ms | baseline p50 ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for lat, s in cg["speedups"].items():
        lines.append(
            f"| {lat} | {_fmt(s['median_gain'], pct=True)} | {_fmt(s['p95_gain'], pct=True)} | "
            f"{_fmt(s['completion_p50_ms'], digits=1)} | {_fmt(s['baseline_p50_ms'], digits=1)} |"
        )
    lines.append("")
    lines.append("## Category breakdown")
    lines.append("")
    lines.append("| Category | trajectories | valid final PTC | final calls | lead p50 ms |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for cat, b in sorted(c["by_category"].items()):
        lines.append(
            f"| {cat} | {b['trajectories']} | {_fmt(b['valid_parse_rate'], pct=True)} | "
            f"{b['final_calls']} | {_fmt(b['lead_ms_p50'], digits=1)} |"
        )
    lines.append("")
    sel = report["threshold_selection"]
    lines.append("## Development threshold selection")
    lines.append("")
    lines.append(f"status: `{sel.get('status')}`")
    if sel.get("selected"):
        s = sel["selected"]
        lines.append("")
        lines.append(
            f"selected k={s['k']}, tau={s['tau']}, min overlap {s['min_overlap_ms']} ms "
            f"-> precision {_fmt(s['precision'], pct=True)}, coverage {_fmt(s['coverage'], pct=True)}, "
            f"median gain at 500 ms {_fmt(s['median_gain_500ms'], pct=True)}, "
            f"at 2 s {_fmt(s['median_gain_2000ms'], pct=True)}"
        )
    lines.append("")
    lines.append("## Instrumentation cost used in the projection")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['costs'], indent=2, sort_keys=True)}\n```")
    lines.append("")
    lines.append(f"```json\n{json.dumps(report['remaining_decode_model'], indent=2, sort_keys=True)}\n```")
    lines.append("")
    return "\n".join(lines) + "\n"
