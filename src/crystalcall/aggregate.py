"""Aggregate derived JSONL into ``reports/results.json`` and ``reports/results.md``.

Everything in the report is computed here from the files in
``experiments/derived/``; no number is ever typed by hand, and the markdown
tables are generated (never edited).  Missing inputs are *reported* as
``{"status": "missing"}`` sections rather than crashing the run, so a partial
experiment still produces a readable report that shows exactly what is absent.

Recognised inputs (all optional):

======================================  ==================================
``trajectory_analyses.jsonl``           ``TrajectoryAnalysis.to_json()``
``policy_outcomes.jsonl``               ``PolicyOutcome.to_json()``
``gate_ablations.jsonl``                ``PolicyOutcome.to_json()`` rows whose
                                        ``policy`` is a gate ablation
``overhead_microbench.jsonl``           ``{"component": str, <numeric>: float}``
``contention.jsonl``                    ``{"scenario": str, <numeric>: float}``
======================================  ==================================
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from . import metrics
from .metrics import (
    ArmPolicy,
    binom_ci_exact,
    bootstrap_ci_clustered,
    deadline_success,
    mcnemar_exact,
    noninferiority_success,
    paired_gain,
    paired_randomization,
    percentiles,
    summarize_outcomes,
    wilcoxon_paired,
)

__all__ = [
    "SCHEMA_VERSION",
    "MAIN_TABLE_SCHEMA",
    "MAIN_TABLE_PATH",
    "DERIVED_SOURCES",
    "MainTableSpec",
    "aggregate",
    "load_jsonl",
    "load_main_table",
    "sha256_file",
    "read_source",
]

SCHEMA_VERSION = "crystalcall-results/1"

#: Schema of the declarative main-comparison-table spec.
MAIN_TABLE_SCHEMA = "crystalcall-main-table/1"

#: Default location of that spec.
MAIN_TABLE_PATH = Path("configs/experiments/main_table.yaml")

#: logical section name -> file name inside the derived directory
DERIVED_SOURCES: dict[str, str] = {
    "trajectory_analyses": "trajectory_analyses.jsonl",
    "policy_outcomes": "policy_outcomes.jsonl",
    "gate_ablations": "gate_ablations.jsonl",
    "overhead": "overhead_microbench.jsonl",
    "contention": "contention.jsonl",
    "toolsandbox_episodes": "toolsandbox_episodes.jsonl",
}

#: keys used to group the free-form microbenchmark / contention rows
_GROUP_KEY_CANDIDATES: tuple[str, ...] = (
    "component",
    "scenario",
    "case",
    "condition",
    "name",
    "label",
)

_BOOT_N = 10_000
_BOOT_SEED = 0
_ALPHA = 0.05


# --------------------------------------------------------------------------- #
# io helpers
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> tuple[list[dict[str, Any]], int]:
    """Read a JSONL file; returns ``(rows, malformed_line_count)``."""
    rows: list[dict[str, Any]] = []
    malformed = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                malformed += 1
    return rows, malformed


def read_source(derived_dir: str | Path, name: str) -> dict[str, Any]:
    """Describe one logical input: presence, hash, size and parsed rows."""
    fname = DERIVED_SOURCES[name]
    path = Path(derived_dir) / fname
    if not path.is_file():
        return {
            "name": name,
            "file": fname,
            "path": str(path),
            "status": "missing",
            "rows": 0,
            "malformed_lines": 0,
            "sha256": None,
            "bytes": 0,
            "_rows": [],
        }
    rows, malformed = load_jsonl(path)
    return {
        "name": name,
        "file": fname,
        "path": str(path),
        "status": "present" if rows else "empty",
        "rows": len(rows),
        "malformed_lines": malformed,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "_rows": rows,
    }


# --------------------------------------------------------------------------- #
# the declarative main comparison table
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class MainTableSpec:
    """``configs/experiments/main_table.yaml`` as data.

    Nothing about the comparison table is selected in code: the row order, the
    ``(arm, policy)`` pair behind every row, the reference denominator, the two
    decisive comparisons, the ablation set and the latency / deadline grids all
    come from the file.
    """

    path: Path
    sha256: str
    schema_version: str
    rows: tuple[dict[str, Any], ...]
    reference: ArmPolicy
    decisive: tuple[ArmPolicy, ...]
    candidate: ArmPolicy
    validation: tuple[ArmPolicy, ...]
    ablation_arm: str
    ablation_policies: tuple[str, ...]
    latency_conditions_ms: tuple[float, ...]
    deadline_budgets_s: tuple[float, ...]

    def arm_policy(self, row: Mapping[str, Any]) -> ArmPolicy:
        return ArmPolicy(
            arm=str(row["arm"]),
            policy=str(row["policy"]),
            id=str(row.get("id", "")),
            label=str(row.get("label", "")),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "status": "present",
            "path": str(self.path),
            "sha256": self.sha256,
            "schema_version": self.schema_version,
            "reference_row": self.reference.to_json(),
            "decisive_rows": [d.to_json() for d in self.decisive],
            "candidate_row": self.candidate.to_json(),
            "validation_rows": [v.to_json() for v in self.validation],
            "ablation_arm": self.ablation_arm,
            "ablation_policies": list(self.ablation_policies),
            "latency_conditions_ms": list(self.latency_conditions_ms),
            "deadline_budgets_s": list(self.deadline_budgets_s),
            "declared_rows": [dict(r) for r in self.rows],
        }


def _pick_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """The row the submission gates judge: ``crystalcall`` if declared, else the
    first non-oracle speculative row (declaration order, deterministic)."""
    for row in rows:
        if str(row.get("id", "")) == "crystalcall":
            return row
    for row in rows:
        if row.get("speculative") and not row.get("oracle"):
            return row
    return None


def _resolve_spec_path(path: str | Path) -> Path:
    """Locate the spec relative to the CWD, else relative to the repo root.

    ``crystalcall analyze`` may run from anywhere in the tree, so a relative
    default falls back to the checkout that contains this module.
    """
    p = Path(path)
    if p.is_file() or p.is_absolute():
        return p
    root_relative = Path(__file__).resolve().parents[2] / p
    if root_relative.is_file():
        return root_relative
    raise FileNotFoundError(
        f"main comparison table spec not found: {p} (also tried {root_relative}); "
        "the table is data, it is not reconstructed in code"
    )


def load_main_table(path: str | Path = MAIN_TABLE_PATH) -> MainTableSpec:
    """Parse the main-table spec; raises on a missing/invalid declaration."""
    p = _resolve_spec_path(path)
    payload = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    schema = str(payload.get("schema_version", ""))
    if schema != MAIN_TABLE_SCHEMA:
        raise ValueError(
            f"{p}: expected schema_version {MAIN_TABLE_SCHEMA!r}, got {schema!r}"
        )
    rows = [dict(r) for r in (payload.get("rows") or [])]
    if not rows:
        raise ValueError(f"{p}: declares no rows")
    by_id = {str(r.get("id", "")): r for r in rows}

    def resolve(row_id: str, what: str) -> ArmPolicy:
        row = by_id.get(str(row_id))
        if row is None:
            raise ValueError(
                f"{p}: {what} {row_id!r} is not one of the declared row ids "
                f"{sorted(by_id)}"
            )
        return ArmPolicy(
            arm=str(row["arm"]),
            policy=str(row["policy"]),
            id=str(row.get("id", "")),
            label=str(row.get("label", "")),
        )

    reference = resolve(payload["reference_row"], "reference_row")
    decisive = tuple(
        resolve(rid, "decisive_rows entry") for rid in (payload.get("decisive_rows") or [])
    )
    validation = tuple(
        resolve(rid, "validation_rows entry") for rid in (payload.get("validation_rows") or [])
    )
    primary_id = payload.get("primary_row")
    cand_row = by_id.get(str(primary_id)) if primary_id else None
    if primary_id and cand_row is None:
        raise ValueError(
            f"{p}: primary_row {primary_id!r} is not one of the declared row ids "
            f"{sorted(by_id)}"
        )
    if cand_row is None:
        cand_row = _pick_candidate(rows)
    if cand_row is None:
        raise ValueError(f"{p}: no speculative row to use as the gate candidate")
    candidate = ArmPolicy(
        arm=str(cand_row["arm"]),
        policy=str(cand_row["policy"]),
        id=str(cand_row.get("id", "")),
        label=str(cand_row.get("label", "")),
    )
    return MainTableSpec(
        path=p,
        sha256=sha256_file(p),
        schema_version=schema,
        rows=tuple(rows),
        reference=reference,
        decisive=decisive,
        candidate=candidate,
        validation=validation,
        ablation_arm=str(payload.get("ablation_arm", metrics.ABLATION_ARM)),
        ablation_policies=tuple(str(x) for x in (payload.get("ablation_policies") or [])),
        latency_conditions_ms=tuple(
            float(x) for x in (payload.get("latency_conditions_ms") or [])
        ),
        deadline_budgets_s=tuple(float(x) for x in (payload.get("deadline_budgets_s") or [])),
    )

# --------------------------------------------------------------------------- #
# section builders
# --------------------------------------------------------------------------- #


def _missing(reason: str, file: str) -> dict[str, Any]:
    return {"status": "missing", "reason": reason, "file": file}


def _json_safe(obj: Any) -> Any:
    """Replace non-finite floats with ``null`` so results.json is strict JSON.

    ``nan`` means "not estimable" (empty sample, ``n == 0``); emitting it as
    ``NaN`` would produce a file that only Python's json module can read.
    """
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def _num(row: Mapping[str, Any], key: str) -> float | None:
    v = row.get(key)
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def crystallization_section(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Crystallization milestones and pre-commit lead, from trajectory analyses."""
    exclusions: Counter = Counter()
    kept: list[Mapping[str, Any]] = []
    for r in rows:
        reason = r.get("exclusion")
        if reason:
            exclusions[str(reason)] += 1
            continue
        kept.append(r)

    n = len(kept)
    valid = sum(1 for r in kept if r.get("valid_final_parse"))
    v_point, v_lo, v_hi = binom_ci_exact(valid, n, _ALPHA)

    lead_by_prompt: dict[str, list[float]] = defaultdict(list)
    leads: list[float] = []
    lead_vs_response: list[float] = []
    milestone_hits: Counter = Counter()
    milestone_total = 0
    eligible_calls = 0
    total_calls = 0
    mutations: list[float] = []
    canvas_only = 0
    rejections: Counter = Counter()
    by_category: dict[str, list[float]] = defaultdict(list)

    for r in kept:
        prompt = str(r.get("prompt_id", ""))
        cat = str(r.get("category", "unknown"))
        for k, v in (r.get("rejection_counts") or {}).items():
            rejections[str(k)] += int(v)
        for call in r.get("final_calls") or []:
            total_calls += 1
            milestone_total += 1
            if call.get("eligible"):
                eligible_calls += 1
            for key in (
                "first_name_ns",
                "first_args_ns",
                "first_exec_ns",
                "invariant_ns",
                "committed_ns",
            ):
                if int(call.get(key, 0) or 0) > 0:
                    milestone_hits[key] += 1
            if call.get("first_canvas_only"):
                canvas_only += 1
            mut = _num(call, "mutations")
            if mut is not None:
                mutations.append(mut)
            lead = _num(call, "lead_ms")
            if lead is not None:
                leads.append(lead)
                lead_by_prompt[prompt].append(lead)
                by_category[cat].append(lead)
            lvr = _num(call, "lead_vs_response_ms")
            if lvr is not None:
                lead_vs_response.append(lvr)

    lead_point, lead_lo, lead_hi = bootstrap_ci_clustered(
        lead_by_prompt, None, _BOOT_N, _BOOT_SEED, _ALPHA
    )
    cov_point, cov_lo, cov_hi = binom_ci_exact(eligible_calls, total_calls, _ALPHA)

    return {
        "status": "present" if n else "empty",
        "n_trajectories": n,
        "n_excluded": int(sum(exclusions.values())),
        "exclusions": {k: int(v) for k, v in sorted(exclusions.items())},
        "valid_final_parse": {
            "k": valid,
            "n": n,
            "point": v_point,
            "ci_lo": v_lo,
            "ci_hi": v_hi,
        },
        "final_calls": {
            "total": total_calls,
            "eligible": eligible_calls,
            "eligible_fraction": cov_point,
            "eligible_ci_lo": cov_lo,
            "eligible_ci_hi": cov_hi,
            "first_canvas_only": canvas_only,
        },
        "milestone_reached": {
            k: int(milestone_hits[k])
            for k in (
                "first_name_ns",
                "first_args_ns",
                "first_exec_ns",
                "invariant_ns",
                "committed_ns",
            )
        },
        "milestone_denominator": milestone_total,
        "lead_ms": percentiles(leads),
        "lead_ms_median_ci": {
            "median": lead_point,
            "ci_lo": lead_lo,
            "ci_hi": lead_hi,
            "clusters": len(lead_by_prompt),
        },
        "lead_vs_response_ms": percentiles(lead_vs_response),
        "mutations_per_call": percentiles(mutations),
        "decode_ms": percentiles([_num(r, "decode_ms") or float("nan") for r in kept]),
        "revision_count": percentiles(
            [_num(r, "revision_count") or float("nan") for r in kept]
        ),
        "parse_failures_total": int(sum(int(r.get("parse_failures", 0) or 0) for r in kept)),
        "rejection_counts": {k: int(v) for k, v in sorted(rejections.items())},
        "lead_ms_by_category": {
            cat: percentiles(vals) for cat, vals in sorted(by_category.items())
        },
    }


def _episode(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row.get("prompt_id", "")), int(row.get("seed", 0)))


def paired_tests_section(
    rows: Sequence[Mapping[str, Any]], *, reference: ArmPolicy
) -> dict[str, Any]:
    """Paired latency and correctness tests of every ``(arm, policy)`` row.

    Rows are bucketed by ``(arm, policy, tool_duration_ms)``; pairs are formed on
    ``(prompt_id, seed)`` against the declared reference row at the same tool
    duration.
    """
    by_key: dict[
        tuple[str, str, float], dict[tuple[str, int], Mapping[str, Any]]
    ] = defaultdict(dict)
    excluded: Counter = Counter()
    dupes = 0
    for r in rows:
        reason = r.get("exclusion")
        if reason:
            excluded[str(reason)] += 1
            continue
        key = (
            str(r.get("arm", "")),
            str(r.get("policy", "")),
            float(r.get("tool_duration_ms", 0.0) or 0.0),
        )
        ep = _episode(r)
        if ep in by_key[key]:
            dupes += 1
            continue
        by_key[key][ep] = r

    out: list[dict[str, Any]] = []
    durations = sorted({d for _, _, d in by_key})
    for arm, policy, dur in sorted(by_key, key=lambda kv: (kv[0], kv[1], kv[2])):
        if (arm, policy) == reference.key:
            continue
        base_map = by_key.get((reference.arm, reference.policy, dur), {})
        row_id = {"arm": arm, "policy": policy, "row_key": f"{arm}/{policy}"}
        if not base_map:
            out.append(
                {
                    **row_id,
                    "tool_duration_ms": dur,
                    "status": "missing",
                    "reason": (
                        f"no {reference.arm}/{reference.policy} rows at duration {dur}"
                    ),
                }
            )
            continue
        cand_map = by_key[(arm, policy, dur)]
        shared = sorted(set(cand_map) & set(base_map))
        a_ms = [float(cand_map[e].get("completion_ms", float("nan"))) for e in shared]
        b_ms = [float(base_map[e].get("completion_ms", float("nan"))) for e in shared]

        corr_pairs = [
            (bool(cand_map[e].get("correct")), bool(base_map[e].get("correct")))
            for e in shared
            if cand_map[e].get("correct") is not None and base_map[e].get("correct") is not None
        ]
        a_ok = [p[0] for p in corr_pairs]
        b_ok = [p[1] for p in corr_pairs]
        n_disc_b = sum(1 for x, y in corr_pairs if x and not y)
        n_disc_c = sum(1 for x, y in corr_pairs if y and not x)

        out.append(
            {
                **row_id,
                "tool_duration_ms": dur,
                "status": "present",
                "n_pairs": len(shared),
                "unpaired_candidate": len(set(cand_map) - set(base_map)),
                "unpaired_baseline": len(set(base_map) - set(cand_map)),
                "correctness_pairs": len(corr_pairs),
                "correctness_unknown": len(shared) - len(corr_pairs),
                "wilcoxon_completion_ms": wilcoxon_paired(a_ms, b_ms),
                "randomization_completion_ms": paired_randomization(
                    a_ms, b_ms, _BOOT_N, _BOOT_SEED
                ),
                "mcnemar_correct": mcnemar_exact(n_disc_b, n_disc_c),
                "noninferiority_correct": noninferiority_success(
                    a_ok, b_ok, metrics.NONINFERIORITY_MARGIN, _ALPHA
                ),
            }
        )

    return {
        "status": "present" if out else "empty",
        "reference": reference.to_json(),
        "tool_durations_ms": durations,
        "n_excluded": int(sum(excluded.values())),
        "exclusions": {k: int(v) for k, v in sorted(excluded.items())},
        "duplicate_episode_rows": dupes,
        "comparisons": out,
    }


def _row_stats(
    rows: Sequence[Mapping[str, Any]],
    target: ArmPolicy,
    reference: ArmPolicy,
    *,
    duration: float | None,
) -> dict[str, Any]:
    """Every main-table column for one ``(arm, policy)`` row at one duration."""
    sel = [
        r
        for r in rows
        if not r.get("exclusion")
        and (str(r.get("arm", "")), str(r.get("policy", ""))) == target.key
        and (duration is None or float(r.get("tool_duration_ms", 0.0) or 0.0) == duration)
    ]
    if not sel:
        return {
            "status": "missing",
            "reason": (
                f"no outcomes for {target.arm}/{target.policy}"
                + (f" at tool_duration_ms={duration:g}" if duration is not None else "")
            ),
            "n": 0,
        }

    correct_k = sum(1 for r in sel if r.get("correct") is True)
    correct_n = sum(1 for r in sel if r.get("correct") is not None)
    s_point, s_lo, s_hi = binom_ci_exact(correct_k, correct_n, _ALPHA)

    true_l = sum(int(r.get("true_launches", 0) or 0) for r in sel)
    false_l = sum(int(r.get("false_launches", 0) or 0) for r in sel)
    p_point, p_lo, p_hi = binom_ci_exact(true_l, true_l + false_l, _ALPHA)

    covered = sum(int(r.get("covered_final_calls", 0) or 0) for r in sel)
    eligible = sum(int(r.get("eligible_final_calls", 0) or 0) for r in sel)
    c_point, c_lo, c_hi = binom_ci_exact(min(covered, eligible), eligible, _ALPHA)

    wasted = sum(float(r.get("wasted_tool_ms", 0.0) or 0.0) for r in sel)
    gain = (
        paired_gain(
            rows,
            target,
            reference,
            duration=duration,
            n_boot=_BOOT_N,
            seed=_BOOT_SEED,
            alpha=_ALPHA,
        )
        if target.key != reference.key
        else None
    )
    # One implementation of the fold, shared with `metrics` so the main table and
    # the per-policy grid cannot disagree about which rows are retrospective.
    retrospective = metrics.all_retrospective(sel)
    return {
        "status": "present",
        "retrospective": retrospective,
        "n": len(sel),
        "n_prompts": len({str(r.get("prompt_id", "")) for r in sel}),
        "completion_ms": percentiles([r.get("completion_ms") for r in sel]),
        "success": {
            "k": correct_k,
            "n": correct_n,
            "point": s_point,
            "ci_lo": s_lo,
            "ci_hi": s_hi,
            "unknown": len(sel) - correct_n,
        },
        "semantic_divergence": int(sum(1 for r in sel if bool(r.get("semantic_divergence")))),
        "verification": metrics.verification_totals(sel),
        "precision": {
            "k": true_l,
            "n": true_l + false_l,
            "point": p_point,
            "ci_lo": p_lo,
            "ci_hi": p_hi,
        },
        "coverage": {
            "k": covered,
            "n": eligible,
            "point": c_point,
            "ci_lo": c_lo,
            "ci_hi": c_hi,
        },
        "wasted_tool_ms_total": wasted,
        "wasted_tool_ms_per_episode": wasted / len(sel),
        "cancelled_launches": false_l,
        "duplicate_launches": int(
            sum(
                max(
                    0,
                    int(r.get("launches", 0) or 0)
                    - int(r.get("claimed", 0) or 0)
                    - int(r.get("false_launches", 0) or 0),
                )
                for r in sel
            )
        ),
        "hidden_overlap_ms": percentiles([r.get("hidden_overlap_ms") for r in sel]),
        "gain": None
        if gain is None
        else {
            "median": gain["median"],
            "ci_lo": gain["ci_lo"],
            "ci_hi": gain["ci_hi"],
            "n_paired": gain["n_paired"],
            "unpaired_candidate": gain["unpaired_candidate"],
            "unpaired_reference": gain["unpaired_reference"],
            "wilcoxon_p": gain["wilcoxon"]["p_value"],
            "randomization_p": gain["randomization"]["p_value"],
        },
    }


def main_table_section(
    spec: MainTableSpec, rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Build the declared main comparison table from the outcome rows.

    One entry per declared row, in declared order.  A declared ``(arm, policy)``
    pair with no outcomes is emitted with ``status="missing"`` -- it is never
    dropped, because a silently vanishing arm reads as a passing table.
    """
    durations = list(spec.latency_conditions_ms) or sorted(
        {float(r.get("tool_duration_ms", 0.0) or 0.0) for r in rows if not r.get("exclusion")}
    )
    out_rows: list[dict[str, Any]] = []
    missing = 0
    for decl in spec.rows:
        target = spec.arm_policy(decl)
        pooled = _row_stats(rows, target, spec.reference, duration=None)
        if pooled.get("status") == "missing":
            missing += 1
        out_rows.append(
            {
                "id": str(decl.get("id", "")),
                "label": str(decl.get("label", "")) or target.display,
                "arm": target.arm,
                "policy": target.policy,
                "row_key": f"{target.arm}/{target.policy}",
                "source": str(decl.get("source", "")),
                "speculative": bool(decl.get("speculative", False)),
                "oracle": bool(decl.get("oracle", False)),
                "note": str(decl.get("note", "") or "").strip(),
                "is_reference": target.key == spec.reference.key,
                "is_decisive": target.key in {d.key for d in spec.decisive},
                "is_candidate": target.key == spec.candidate.key,
                "status": pooled["status"],
                "pooled": pooled,
                "by_duration": [
                    {
                        "tool_duration_ms": d,
                        **_row_stats(rows, target, spec.reference, duration=d),
                    }
                    for d in durations
                ],
            }
        )
    return {
        "status": "present" if out_rows else "empty",
        "spec": spec.to_json(),
        "latency_conditions_ms": [float(d) for d in durations],
        "n_declared": len(out_rows),
        "n_missing": missing,
        "rows": out_rows,
    }


def toolsandbox_section(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """ToolSandbox replay check: one generation per episode, replayed against a
    clone of the initial world.  Divergence here feeds gate S6 exactly like the
    per-claim verifier does; excluded episodes are reported, never counted clean.
    """
    totals = metrics.external_divergence_totals(rows)
    exclusions: Counter = Counter()
    by_arm: dict[str, dict[str, int]] = defaultdict(
        lambda: {"episodes": 0, "divergent": 0, "success": 0, "excluded": 0}
    )
    for r in rows:
        arm = str(r.get("arm", ""))
        if r.get("exclusion"):
            exclusions[str(r["exclusion"]).split(":", 1)[0]] += 1
            by_arm[arm]["excluded"] += 1
            continue
        by_arm[arm]["episodes"] += 1
        by_arm[arm]["divergent"] += int(r.get("semantic_divergence", 0) or 0)
        by_arm[arm]["success"] += 1 if r.get("success") else 0
    return {
        "status": "present" if rows else "empty",
        "totals": totals,
        "n_input_rows": len(rows),
        "exclusions": {k: int(v) for k, v in sorted(exclusions.items())},
        "by_arm": {arm: dict(sorted(v.items())) for arm, v in sorted(by_arm.items())},
    }


#: Phase 0 overhead-ladder columns, in report order.
_LADDER_FIELDS: tuple[str, ...] = (
    "level",
    "p50_ms",
    "p95_ms",
    "revisions_per_request",
    "relative_p50_vs_off",
    "absolute_p50_delta_ms",
    "parse_ms_per_revision",
    "decode_ms_per_revision",
    "problems",
    "ok",
)

_LADDER_ORDER = ("off", "trace", "parse", "full", "gate")


def instrumentation_section(
    outcome_rows: Sequence[Mapping[str, Any]],
    ladder_rows: Sequence[Mapping[str, Any]],
    *,
    ladder_status: str = "present",
    ladder_file: str = "overhead_microbench.jsonl",
) -> dict[str, Any]:
    """In-timeline instrumentation cost next to the Phase 0 overhead ladder.

    ``instrumentation_in_timeline_ms`` is the trace+parse cost the replayed
    decode timeline already contains: every arm replays the *same* instrumented
    timeline, so it is never charged to completion a second time and never
    credited back to the reference arm.  The measured
    per-rung ladder is reported beside it, unmodified.
    """
    in_timeline = metrics.instrumentation_totals(outcome_rows)

    def rung_order(row: Mapping[str, Any]) -> tuple[int, str]:
        level = str(row.get("level", ""))
        return (
            _LADDER_ORDER.index(level) if level in _LADDER_ORDER else len(_LADDER_ORDER),
            level,
        )

    rungs = [
        {field: row.get(field) for field in _LADDER_FIELDS} | {"n": row.get("n")}
        for row in sorted(ladder_rows, key=rung_order)
    ]
    failing = [str(r.get("level")) for r in rungs if r.get("ok") is False]
    return {
        "status": "present" if (in_timeline["requests"] or rungs) else "empty",
        "in_timeline": in_timeline,
        "ladder": {
            "status": ladder_status if ladder_rows else ("empty" if ladder_status == "present" else ladder_status),
            "file": ladder_file,
            "fields": list(_LADDER_FIELDS),
            "rungs": rungs,
            "failing_rungs": failing,
        },
        "note": (
            "all arms replay the same instrumented timeline; the in-timeline cost is "
            "reported, not charged again to completion and not credited back to the "
            "reference arm"
        ),
    }


def numeric_group_section(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Percentile summary of every numeric field, grouped by the first key field."""
    if not rows:
        return {"status": "empty", "groups": []}
    key_field = None
    for cand in _GROUP_KEY_CANDIDATES:
        if any(cand in r for r in rows):
            key_field = cand
            break
    buckets: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        label = str(r.get(key_field, "all")) if key_field else "all"
        buckets[label].append(r)

    numeric_fields: set[str] = set()
    for r in rows:
        for k in r:
            if _num(r, k) is not None:
                numeric_fields.add(k)

    groups: list[dict[str, Any]] = []
    for label in sorted(buckets):
        grp = buckets[label]
        fields: dict[str, Any] = {}
        for field in sorted(numeric_fields):
            vals = [v for v in (_num(r, field) for r in grp) if v is not None]
            if vals:
                fields[field] = percentiles(vals)
        groups.append({"key": label, "n": len(grp), "fields": fields})
    return {
        "status": "present",
        "key_field": key_field,
        "numeric_fields": sorted(numeric_fields),
        "groups": groups,
    }


# --------------------------------------------------------------------------- #
# markdown rendering
# --------------------------------------------------------------------------- #


def _f(x: Any, nd: int = 3) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _pct(x: Any, nd: int = 1) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(v):
        return "n/a"
    return f"{100.0 * v:.{nd}f}%"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        # A dropped or duplicated cell shifts every later column silently, and
        # a rendered table gives no hint. The header is the contract, checked
        # here so every table in this module is covered, not just one renderer.
        if len(row) != len(header):
            raise AssertionError(
                f"table row has {len(row)} cells, header declares {len(header)}: {list(row)!r}"
            )
        out.append("| " + " | ".join(row) + " |")
    return out


def _method_rows(summary: Mapping[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for g in summary.get("groups", []):
        comp = g["completion_ms"]
        gain = g["gain"]
        prec = g["precision"]
        cov = g["coverage"]
        rows.append(
            [
                g.get("row_key", g.get("policy", "")),
                _f(g["tool_duration_ms"], 0),
                str(g["n"]),
                _f(comp.get("p50"), 1),
                _f(comp.get("p95"), 1),
                _f(comp.get("p99"), 1),
                f"{_pct(gain.get('median'))} [{_pct(gain.get('ci_lo'))}, {_pct(gain.get('ci_hi'))}]",
                str(gain.get("n_paired", 0)),
                str(gain.get("unpaired", 0)),
                f"{_pct(prec.get('point'))} [{_pct(prec.get('ci_lo'))}, {_pct(prec.get('ci_hi'))}]"
                + f" ({prec.get('k')}/{prec.get('n')})"
                # Same rule as the main table: with no launches there is no
                # precision to qualify, so the marker would label an empty cell.
                + (
                    " _retrospective_"
                    if g.get("retrospective") and (prec.get("n") or 0) > 0
                    else ""
                ),
                f"{_pct(cov.get('point'))} ({cov.get('k')}/{cov.get('n')})",
                _f(g["waste"].get("wasted_tool_ms_total"), 1),
                str(g["waste"].get("cancelled_launches")),
                str(g["waste"].get("duplicate_launches")),
                _f(g["hidden_overlap_ms"].get("p50"), 1),
                str(g.get("semantic_divergence")),
            ]
        )
    return rows


_METHOD_HEADER = (
    "arm/policy",
    "tool ms",
    "n",
    "p50 ms",
    "p95 ms",
    "p99 ms",
    "median gain [95% CI]",
    "paired",
    "unpaired",
    "spec. precision [95% CI]",
    "coverage",
    "wasted ms",
    "cancelled",
    "duplicate",
    "hidden p50 ms",
    "divergences",
)


#: Columns of the declared main comparison table.
_MAIN_TABLE_HEADER = (
    "Method",
    "Success",
    "Semantic divergence",
    "p50",
    "p95",
    "p99",
    "Precision",
    "Coverage",
    "Wasted work",
)

#: The retrospective marker is data-driven: `replay` records
#: `retrospective: true` on outcomes whose launches were resolved from
#: `analysis.final_calls`, and `_row_stats` folds that into the row. It is
#: annotated on the Precision cell only -- coverage and latency in those rows are
#: real measurements.


def _main_row_cells(row: Mapping[str, Any], stats: Mapping[str, Any]) -> list[str]:
    label = row["label"]
    marks = []
    if row.get("is_reference"):
        marks.append("reference")
    if row.get("is_candidate"):
        marks.append("candidate")
    if row.get("is_decisive"):
        marks.append("decisive")
    if row.get("oracle"):
        marks.append("oracle")
    method = label + (f" _({', '.join(marks)})_" if marks else "")
    if stats.get("status") != "present":
        return [method] + ["missing"] + ["-"] * (len(_MAIN_TABLE_HEADER) - 2)
    succ = stats["success"]
    comp = stats["completion_ms"]
    prec = stats["precision"]
    cov = stats["coverage"]
    retrospective = bool(stats.get("retrospective"))
    if prec["n"] == 0:
        precision_cell = "n/a (non-speculative)"
    else:
        precision_cell = (
            f"{_pct(prec['point'])} [{_pct(prec['ci_lo'])}, {_pct(prec['ci_hi'])}]"
            f" ({prec['k']}/{prec['n']})"
        )
        if retrospective:
            precision_cell += " _retrospective: matched_final by construction_"
    return [
        method,
        f"{_pct(succ['point'])} [{_pct(succ['ci_lo'])}, {_pct(succ['ci_hi'])}]"
        f" ({succ['k']}/{succ['n']})",
        str(stats["semantic_divergence"]),
        _f(comp.get("p50"), 1),
        _f(comp.get("p95"), 1),
        _f(comp.get("p99"), 1),
        precision_cell,
        "n/a" if cov["n"] == 0 else f"{_pct(cov['point'])} ({cov['k']}/{cov['n']})",
        f"{_f(stats['wasted_tool_ms_per_episode'], 1)} ms/episode"
        f" ({stats['cancelled_launches']} cancelled, {stats['duplicate_launches']} duplicate)",
    ]


def _main_table_lines(block: Mapping[str, Any]) -> list[str]:
    """Render the declared main comparison table, one sub-table per latency condition."""
    if not block or block.get("status") == "missing":
        return [
            f"_missing: {block.get('reason', 'main table not built')} "
            f"(`{block.get('file', 'policy_outcomes.jsonl')}`)_"
        ]
    spec = block.get("spec", {})
    out: list[str] = [
        f"Declared by `{spec.get('path')}` (schema `{spec.get('schema_version')}`, sha256 "
        f"`{(spec.get('sha256') or '-')[:16]}`); {block['n_declared']} rows, "
        f"{block['n_missing']} without outcomes.",
        "",
        "Rows are ``(arm, policy)`` pairs in declared order; a declared pair with no "
        "outcomes is kept with `missing` rather than dropped.",
    ]
    for dur in block.get("latency_conditions_ms", []):
        out += ["", f"### Tool latency D = {_f(dur, 0)} ms", ""]
        rows: list[list[str]] = []
        for row in block["rows"]:
            stats = next(
                (
                    d
                    for d in row.get("by_duration", [])
                    if float(d.get("tool_duration_ms", -1.0)) == float(dur)
                ),
                {"status": "missing"},
            )
            rows.append(_main_row_cells(row, stats))
        out += _table(_MAIN_TABLE_HEADER, rows)

    out += ["", "### Paired latency gain vs the reference row", ""]
    grows: list[list[str]] = []
    for row in block["rows"]:
        for d in row.get("by_duration", []):
            gain = d.get("gain") if d.get("status") == "present" else None
            if not gain:
                continue
            grows.append(
                [
                    row["label"],
                    _f(d["tool_duration_ms"], 0),
                    f"{_pct(gain['median'])} [{_pct(gain['ci_lo'])}, {_pct(gain['ci_hi'])}]",
                    str(gain["n_paired"]),
                    str(gain["unpaired_candidate"]),
                    f"{gain['wilcoxon_p']:.3g}",
                    f"{gain['randomization_p']:.3g}",
                ]
            )
    if grows:
        out += _table(
            (
                "Method",
                "tool ms",
                "median gain [95% CI]",
                "paired",
                "unpaired",
                "wilcoxon p",
                "randomization p",
            ),
            grows,
        )
    else:
        out += ["_no row could be paired against the reference row_"]
    return out


_GATE_HEADER = (
    "id",
    "gate",
    "required",
    "threshold",
    "point",
    "95% CI",
    "n",
    "result",
    "notes",
)


def _gate_notes(gate: Mapping[str, Any]) -> str:
    d = gate.get("detail") or {}
    bits: list[str] = []
    if gate.get("status") != "present":
        bits.append(f"status={gate.get('status')}")
    if "reason" in d:
        bits.append(str(d["reason"]))
    for key, label in (
        ("true_launches", "true launches"),
        ("false_launches", "false launches"),
        ("covered_final_calls", "covered"),
        ("eligible_final_calls", "eligible"),
    ):
        if key in d:
            bits.append(f"{label}={d[key]}")
    if "margin" in d:
        bits.append(f"margin={_f(d['margin'], 3)}")
        bits.append(f"n10={d.get('n10')}, n01={d.get('n01')}")
        if d.get("continuity_applied"):
            bits.append("continuity correction applied (no discordant pair)")
    for key, label, fmt in (
        ("wilcoxon_p", "wilcoxon p", "g"),
        ("randomization_p", "randomization p", "g"),
        ("mcnemar_p", "mcnemar p", "g"),
        ("rule_of_three_upper", "rule-of-three upper", "f"),
    ):
        if key in d and d[key] is not None:
            v = float(d[key])
            bits.append(f"{label}={v:.3g}" if fmt == "g" else f"{label}={v:.4f}")
    if "in_target_band" in d:
        band = gate.get("target_band")
        bits.append(
            ("in" if d["in_target_band"] else "outside")
            + f" target band [{_f(band[0], 2)}, {_f(band[1], 2)}]"
        )
    if "divergences" in d:
        bits.append(
            f"{d['divergences']} divergence(s) over {d.get('claims_verified', 0)} verified "
            f"claim(s) of {d.get('claims_claimed', 0)} claimed"
        )
        bits.append(
            f"match={d.get('claims_verified_match')}, "
            f"divergent={d.get('claims_verified_divergent')}, "
            f"verifier_error={d.get('claims_verification_error')}, "
            f"unverified={d.get('claims_unverified')}, "
            f"unaccounted={d.get('claims_unaccounted')}"
        )
        ext = d.get("external_replay") or {}
        if ext:
            bits.append(
                f"replay check: {ext.get('divergent')} divergent over "
                f"{ext.get('episodes')} episode(s), {ext.get('excluded_episodes')} excluded"
            )
        if d.get("fail_reasons"):
            bits.append("FAIL CLOSED: " + "; ".join(d["fail_reasons"]))
    if "ci_excludes_zero" in d:
        bits.append("CI excludes 0" if d["ci_excludes_zero"] else "CI includes 0")
    if "decisive_companion" in d:
        bits.append(f"read with {d['decisive_companion']}")
    if d.get("pareto"):
        cand = d["pareto"]["candidate"]
        ref = d["pareto"]["reference"]
        bits.append(
            "Pareto: precision "
            f"{_pct(cand['precision']['point'])} vs {_pct(ref['precision']['point'])}, "
            f"wasted/episode {_f(cand['wasted_tool_ms_per_episode'], 1)} vs "
            f"{_f(ref['wasted_tool_ms_per_episode'], 1)} ms, gain vs baseline "
            f"{_pct(cand['median_gain_vs_baseline'])} vs "
            f"{_pct(ref['median_gain_vs_baseline'])}"
        )
    return "; ".join(bits) if bits else "-"


def _instrumentation_lines(sec: Mapping[str, Any]) -> list[str]:
    """In-timeline instrumentation cost plus the Phase 0 overhead ladder."""
    if not sec or sec.get("status") == "missing":
        return [f"_missing: {sec.get('reason', 'not measured')} (`{sec.get('file', '-')}`)_"]
    it = sec["in_timeline"]
    out = [
        "**All arms replay the same instrumented timeline.** The trace+parse cost below is "
        "already contained in that shared decode timeline: it is neither charged to "
        "completion a second time nor credited back to the reference arm, because doing "
        "either would build an unfair \"instrumented candidate vs stripped baseline\" "
        "comparison. It therefore cancels in every paired "
        "comparison and is reported on its own here.",
        "",
    ]
    if it["requests"]:
        out += _table(
            ("in-timeline instrumentation", "value"),
            [
                ("requests measured", str(it["requests"])),
                ("per-episode p50 ms", _f(it["in_timeline_ms"].get("p50"), 3)),
                ("per-episode p95 ms", _f(it["in_timeline_ms"].get("p95"), 3)),
                ("per-episode p99 ms", _f(it["in_timeline_ms"].get("p99"), 3)),
                ("per-episode mean ms", _f(it["in_timeline_ms"].get("mean"), 3)),
                ("total ms over all requests", _f(it["in_timeline_ms_total"], 1)),
                ("charged to completion", str(it["charged_to_completion"])),
            ],
        )
        if it["by_arm"]:
            out += ["", "Per-episode in-timeline cost by arm (ms):", ""]
            out += _table(
                ("arm", "n", "p50", "p95"),
                [
                    (arm, _f(v.get("n"), 0), _f(v.get("p50"), 3), _f(v.get("p95"), 3))
                    for arm, v in it["by_arm"].items()
                ],
            )
    else:
        out += ["_no replayed episode carried an in-timeline instrumentation cost_"]

    ladder = sec["ladder"]
    out += ["", "### Phase 0 overhead ladder", ""]
    if ladder["status"] == "missing":
        out += [f"_missing: input file not found (`{ladder['file']}`)_"]
        return out
    if not ladder["rungs"]:
        out += [f"_input present but contained no rungs (`{ladder['file']}`)_"]
        return out
    out += _table(
        (
            "level",
            "n",
            "p50 ms",
            "p95 ms",
            "revisions/request",
            "relative p50 vs off",
            "absolute p50 delta ms",
            "parse ms/revision",
            "decode ms/revision",
            "ok",
            "problems",
        ),
        [
            [
                str(r.get("level")),
                _f(r.get("n"), 0),
                _f(r.get("p50_ms"), 2),
                _f(r.get("p95_ms"), 2),
                _f(r.get("revisions_per_request"), 2),
                _pct(r.get("relative_p50_vs_off"), 2),
                _f(r.get("absolute_p50_delta_ms"), 2),
                _f(r.get("parse_ms_per_revision"), 4),
                _f(r.get("decode_ms_per_revision"), 3),
                "yes" if r.get("ok") else "NO",
                "; ".join(str(p) for p in (r.get("problems") or [])) or "-",
            ]
            for r in ladder["rungs"]
        ],
    )
    if ladder["failing_rungs"]:
        out += [
            "",
            "**Failing rungs (`ok: false`): "
            + ", ".join(f"`{lvl}`" for lvl in ladder["failing_rungs"])
            + "** - the measurement itself is suspect, not just the number.",
        ]
    return out


def _fidelity_lines(mc: Mapping[str, Any]) -> list[str]:
    """Online future table vs offline replay of the same frozen gate."""
    if not isinstance(mc, dict) or "summary" not in mc:
        return ["_missing: no policy outcomes, so no online run could be compared_"]
    fid = mc["summary"]["accounting"]["replay_fidelity"]
    obs = mc["summary"]["accounting"]["observed"]
    out = _table(
        ("quantity", "value"),
        [
            ("requests compared (arm, trajectory, own frozen gate)", str(fid["compared"])),
            ("matching", str(fid["matching"])),
            ("mismatching", str(fid["mismatching"])),
            ("rows not comparable (different policy: expected to differ)", str(fid["not_comparable"])),
            ("requests with an online observation", str(obs["requests_with_observation"])),
            ("online launched / claimed / wasted", 
             f"{obs['observed_launched']} / {obs['observed_claimed']} / {obs['observed_wasted']}"),
        ],
    )
    if "candidate_precision_online" in fid:
        on = fid["candidate_precision_online"]
        re_ = fid["candidate_precision_replayed"]
        out += [
            "",
            f"Candidate row `{fid['candidate']['arm']}/{fid['candidate']['policy']}` "
            "launch precision, measured online vs re-derived offline:",
            "",
        ]
        out += _table(
            ("source", "k / n", "precision [95% CI]", "definition"),
            [
                (
                    "online future table",
                    f"{on['k']} / {on['n']}",
                    f"{_pct(on['point'])} [{_pct(on['ci_lo'])}, {_pct(on['ci_hi'])}]",
                    on["definition"],
                ),
                (
                    "offline replay",
                    f"{re_['k']} / {re_['n']}",
                    f"{_pct(re_['point'])} [{_pct(re_['ci_lo'])}, {_pct(re_['ci_hi'])}]",
                    re_["definition"],
                ),
            ],
        )
        out += ["", f"Difference (online - replayed): {_pct(fid['candidate_precision_delta'], 2)}."]
    if not fid["mismatching"]:
        out += [
            "",
            "Every comparable request reproduced its online launch count under replay.",
        ]
        return out
    out += [
        "",
        f"**REPLAY FIDELITY FAILURE: {fid['mismatching']} request(s) did not reproduce.** "
        "This is a bug signal, not a result: the offline replay of a frozen gate must "
        "reproduce what the online future table did on the same trace. Every mismatching "
        "request is listed below and none of them is averaged into any other table.",
        "",
    ]
    out += _table(
        (
            "arm",
            "trajectory_id",
            "policy",
            "observed launched",
            "replayed launches",
            "observed claimed",
            "replayed claimed",
            "tool ms",
        ),
        [
            [
                r["arm"],
                r["trajectory_id"],
                r["policy"],
                str(r["observed_launched"]),
                str(r["replayed_launches"]),
                str(r["observed_claimed"]),
                str(r["replayed_claimed"]),
                _f(r["tool_duration_ms"], 0),
            ]
            for r in fid["mismatching_rows"]
        ],
    )
    return out


def _verification_lines(results: Mapping[str, Any]) -> list[str]:
    """Per-claim verification coverage plus the ToolSandbox replay check."""
    mc = results["sections"].get("method_comparison", {})
    if not isinstance(mc, dict) or "summary" not in mc:
        return [
            "_missing: no policy outcomes, so no claim was verified - S6 cannot pass_"
        ]
    acc = mc["summary"]["accounting"]
    cand = acc["candidate_verification"]
    allrows = acc["verification"]
    out = _table(
        ("counter", "candidate row", "all rows"),
        [
            (label, str(cand[key]), str(allrows[key]))
            for label, key in (
                ("episodes", "episodes"),
                ("claims claimed", "claims_claimed"),
                ("claims verified (match)", "claims_verified_match"),
                ("claims verified (divergent)", "claims_verified_divergent"),
                ("claims verification error", "claims_verification_error"),
                ("claims unverified", "claims_unverified"),
                ("claims unaccounted", "claims_unaccounted"),
                ("claims verified (total)", "claims_verified"),
                ("episodes with incomplete verification", "episodes_verification_incomplete"),
                ("episodes flagged semantic_divergence", "episodes_flagged_divergent"),
            )
        ],
    )
    out += [
        "",
        "Every claimed speculative result is re-executed normally by "
        "`HostToolRouter(verify_claims=True)` and the two result digests are compared, so the "
        "model is held fixed and a mismatch can only come from scheduling. A verifier error "
        "is **not** evidence of losslessness: S6 fails closed unless every claim is verified, "
        "and the rule-of-three bound is taken over verified claims only.",
    ]

    ts = results["sections"].get("toolsandbox", {})
    out += ["", "### ToolSandbox replay check", ""]
    if not isinstance(ts, dict) or ts.get("status") == "missing":
        out += [f"_missing: {ts.get('reason', 'not run')} (`{ts.get('file', '-')}`)_"]
        return out
    if ts.get("status") == "empty":
        out += ["_input present but contained no episodes_"]
        return out
    out += _table(
        ("arm", "episodes", "divergent", "success", "excluded"),
        [
            (
                arm,
                str(v["episodes"]),
                str(v["divergent"]),
                str(v["success"]),
                str(v["excluded"]),
            )
            for arm, v in ts["by_arm"].items()
        ],
    )
    tot = ts["totals"]
    out += [
        "",
        f"Totals: {tot['episodes']} episode(s) ({tot['speculative_episodes']} speculative), "
        f"{tot['divergent']} divergent, {tot['excluded_episodes']} excluded. "
        "One generation per episode is replayed under ordinary scheduling against a clone of "
        "the initial world; any divergence fails S6.",
    ]
    return out


def _gate_lines(block: Mapping[str, Any]) -> list[str]:
    if not block or block.get("status") == "missing":
        return [
            f"_missing: {block.get('reason', 'no submission gate block')} "
            f"(`{block.get('file', 'policy_outcomes.jsonl')}`) - gates are NOT met_"
        ]
    rows: list[list[str]] = []
    for g in block.get("gates", []):
        rows.append(
            [
                g["id"],
                g["key"] + (" (decisive)" if g.get("decisive") else ""),
                "required" if g["required"] else "preferred",
                g["comparator"]
                if any(ch.isdigit() for ch in str(g["comparator"]))
                else f"{g['comparator']} {_f(g['required_threshold'], 3)}",
                _f(g["point"], 4),
                f"[{_f(g['ci_lo'], 4)}, {_f(g['ci_hi'], 4)}]",
                str(g["n"]),
                "PASS" if g["pass"] else "FAIL",
                _gate_notes(g),
            ]
        )
    out = _table(_GATE_HEADER, rows)
    out += [
        "",
        f"Candidate arm: `{block.get('candidate_policy')}`; baseline `{block.get('baseline_policy')}`; "
        f"decisive references `{block.get('earliest_committed_policy')}` (S7) and "
        f"`{block.get('preact_policy') or 'P-ReAct-style arm absent'}` (S8).",
        f"Required gates passed: {block.get('required_passed')} / {block.get('required_total')}; "
        f"all required pass: {bool(block.get('all_required_pass'))}. "
        "S3/S4 must never be read without S7 (and S8) beside them: a gain over the "
        "post-response baseline alone is not sufficient evidence.",
    ]
    return out


def render_markdown(results: Mapping[str, Any]) -> str:
    lines: list[str] = [
        "# CrystalCall / DenoisePTC results",
        "",
        f"Generated by `crystalcall.aggregate` (schema `{results['schema']}`).",
        "**Do not edit by hand** - regenerate with "
        "`crystalcall.aggregate.aggregate(derived_dir, out_json, out_md)`.",
        "",
        "## Inputs",
        "",
    ]
    lines += _table(
        ("source", "file", "status", "rows", "malformed", "bytes", "sha256"),
        [
            (
                s["name"],
                f"`{s['file']}`",
                s["status"],
                str(s["rows"]),
                str(s["malformed_lines"]),
                str(s["bytes"]),
                (s["sha256"] or "-")[:16],
            )
            for s in results["sources"]
        ],
    )

    lines += ["", "## Main comparison table", ""]
    lines += _main_table_lines(results.get("main_table", {}))

    lines += ["", "## Submission gates (S1-S8)", ""]
    lines += _gate_lines(results.get("submission_gates", {}))

    lines += ["", "## Verification coverage (gate S6)", ""]
    lines += _verification_lines(results)

    lines += ["", "## Frozen policy", ""]
    fp = results.get("frozen_policy", {})
    if fp.get("status") != "present":
        lines += [
            f"_missing: {fp.get('reason', 'not recorded')} "
            f"(`{fp.get('path', 'configs/frozen_policy.json')}`; written by `crystalcall pilot`)_"
        ]
    else:
        pol = fp.get("policy") or {}
        flat = {k: v for k, v in sorted(pol.items()) if not isinstance(v, (dict, list))}
        lines += _table(
            ("threshold", "frozen value"),
            [(k, str(v)) for k, v in flat.items()],
        )
        lines += ["", f"Source: `{fp['path']}` (sha256 `{fp['sha256'][:16]}`)."]

    lines += ["", "## Exclusions", ""]
    exc = results.get("exclusions", {})
    if exc.get("rows"):
        lines += _table(
            ("scope", "reason", "count"),
            [(scope, reason, str(count)) for scope, reason, count in exc.get("rows", [])],
        )
        lines += ["", f"Total excluded rows: {exc.get('total', 0)}."]
    else:
        lines += ["_no exclusions recorded_"]

    sections = results["sections"]

    lines += ["", "## Method comparison", ""]
    mc = sections["method_comparison"]
    if mc.get("status") == "missing":
        lines += [f"_missing: {mc.get('reason')} (`{mc.get('file')}`)_"]
    else:
        lines += _table(_METHOD_HEADER, _method_rows(mc["summary"]))
        lines += [
            "",
            f"Reference row: `{mc['summary']['reference']['arm']}/"
            f"{mc['summary']['reference']['policy']}`; "
            f"gain = 1 - completion / reference completion on the same (prompt_id, seed); "
            f"rows are keyed on (arm, policy, tool duration) and never pooled across arms; "
            f"CI = prompt-clustered percentile bootstrap "
            f"(n={mc['summary']['bootstrap']['n']}, seed={mc['summary']['bootstrap']['seed']}).",
            "",
            "### Paired tests vs baseline",
            "",
        ]
        pt = sections["paired_tests"]
        prows: list[list[str]] = []
        for c in pt.get("comparisons", []):
            if c.get("status") == "missing":
                prows.append(
                    [
                        c.get("row_key", c.get("policy", "")),
                        _f(c["tool_duration_ms"], 0),
                        "missing",
                        c.get("reason", ""),
                    ]
                    + [""] * 4
                )
                continue
            ni = c["noninferiority_correct"]
            prows.append(
                [
                    c.get("row_key", c.get("policy", "")),
                    _f(c["tool_duration_ms"], 0),
                    str(c["n_pairs"]),
                    _f(c["randomization_completion_ms"]["observed_diff"], 2),
                    f"{c['wilcoxon_completion_ms']['p_value']:.3g}",
                    f"{c['randomization_completion_ms']['p_value']:.3g}",
                    f"{c['mcnemar_correct']['p_value']:.3g}"
                    f" (b={c['mcnemar_correct']['b']}, c={c['mcnemar_correct']['c']})",
                    f"{_f(ni['diff'], 4)} [{_f(ni['ci_lo'], 4)}, {_f(ni['ci_hi'], 4)}] -> "
                    + ("non-inferior" if ni["non_inferior"] else "NOT non-inferior"),
                ]
            )
        lines += _table(
            (
                "arm/policy",
                "tool ms",
                "pairs",
                "mean dt ms",
                "wilcoxon p",
                "randomization p",
                "mcnemar p",
                f"success diff [95% CI] (margin {metrics.NONINFERIORITY_MARGIN})",
            ),
            prows,
        )

        lines += ["", "### Deadline success Q(B)", ""]
        ds = sections["deadline_success"]
        drows: list[list[str]] = []
        budgets = ds.get("budgets_s", [])
        for g in ds.get("groups", []):
            cells = [g["policy"], _f(g["tool_duration_ms"], 0), str(g["n"])]
            for b in g["budgets"]:
                cells.append(f"{_pct(b['q'])} [{_pct(b['ci_lo'])}, {_pct(b['ci_hi'])}]")
            cells.append(str(g["unknown_correct"]))
            drows.append(cells)
        lines += _table(
            ["policy", "tool ms", "n"] + [f"Q({_f(b, 0)}s)" for b in budgets] + ["unknown"],
            drows,
        )

        lines += ["", "### Paired deadline success (McNemar exact)", ""]
        dp = sections["deadline_paired"]
        if dp.get("status") != "present":
            lines += [f"_{dp.get('status', 'empty')}_"]
        else:
            lines += [
                "Paired by `(prompt_id, seed)` on the indicator "
                "`correct AND completion_ms <= B`. Only `live` cells have "
                "discordant pairs; `saturated` / `floored` / `tied` cells are "
                "listed so that no live cell can be quoted as if the grid were "
                f"uniformly informative. Live cells: **{dp['n_live_cells']}** of "
                f"{dp['n_cells']}; no multiplicity correction is applied.",
                "",
            ]
            prows = [
                [
                    c["policy"],
                    c["comparator"],
                    _f(c["tool_duration_ms"], 0),
                    _f(c["budget_s"], 0),
                    f"{c['k_candidate']}/{c['n_pairs']}",
                    f"{c['k_comparator']}/{c['n_pairs']}",
                    f"+{c['mcnemar']['b']} / -{c['mcnemar']['c']}",
                    f"{c['mcnemar']['p_value']:.4g}",
                    c["regime"],
                ]
                for c in dp["cells"]
                if c["regime"] == "live"
            ]
            lines += _table(
                (
                    "policy",
                    "comparator",
                    "tool ms",
                    "budget s",
                    "hits",
                    "comparator hits",
                    "discordant",
                    "exact p",
                    "regime",
                ),
                prows,
            )
            regimes = Counter(c["regime"] for c in dp["cells"])
            lines += [
                "",
                "Cell regimes: "
                + ", ".join(f"`{k}` {v}" for k, v in sorted(regimes.items()))
                + ".",
            ]
            gaps = [
                c for c in dp.get("comparator_coverage", []) if not c["n_cells"]
            ]
            if gaps:
                lines += [
                    "",
                    "Comparators declared but not testable here: "
                    + ", ".join(f"`{c['comparator']}` ({c['reason']})" for c in gaps)
                    + ".",
                ]

    lines += ["", "## Crystallization", ""]
    cs = sections["crystallization"]
    if cs.get("status") == "missing":
        lines += [f"_missing: {cs.get('reason')} (`{cs.get('file')}`)_"]
    else:
        vf = cs["valid_final_parse"]
        fc = cs["final_calls"]
        lead = cs["lead_ms"]
        lci = cs["lead_ms_median_ci"]
        lines += _table(
            ("quantity", "value"),
            [
                ("trajectories", str(cs["n_trajectories"])),
                ("excluded", str(cs["n_excluded"])),
                (
                    "valid final parse",
                    f"{_pct(vf['point'])} [{_pct(vf['ci_lo'])}, {_pct(vf['ci_hi'])}] "
                    f"({vf['k']}/{vf['n']})",
                ),
                (
                    "speculation-eligible final calls",
                    f"{_pct(fc['eligible_fraction'])} ({fc['eligible']}/{fc['total']})",
                ),
                ("pre-commit lead p50 ms", _f(lead.get("p50"), 1)),
                ("pre-commit lead p95 ms", _f(lead.get("p95"), 1)),
                (
                    "pre-commit lead median [95% CI] ms",
                    f"{_f(lci['median'], 1)} [{_f(lci['ci_lo'], 1)}, {_f(lci['ci_hi'], 1)}]",
                ),
                ("lead vs response p50 ms", _f(cs["lead_vs_response_ms"].get("p50"), 1)),
                ("mutations per call p50", _f(cs["mutations_per_call"].get("p50"), 2)),
                ("decode ms p50", _f(cs["decode_ms"].get("p50"), 1)),
                ("parse failures", str(cs["parse_failures_total"])),
            ],
        )
        lines += ["", "Milestone reachability (calls with a non-zero timestamp):", ""]
        lines += _table(
            ("milestone", "count", "of"),
            [
                (k, str(v), str(cs["milestone_denominator"]))
                for k, v in cs["milestone_reached"].items()
            ],
        )
        if cs["lead_ms_by_category"]:
            lines += ["", "Pre-commit lead by category (ms):", ""]
            lines += _table(
                ("category", "n", "p50", "p95"),
                [
                    (cat, _f(v.get("n"), 0), _f(v.get("p50"), 1), _f(v.get("p95"), 1))
                    for cat, v in cs["lead_ms_by_category"].items()
                ],
            )
        if cs["rejection_counts"]:
            lines += ["", "Parse rejections:", ""]
            lines += _table(
                ("reason", "count"), [(k, str(v)) for k, v in cs["rejection_counts"].items()]
            )

    lines += ["", "## Gate ablation", ""]
    ga = sections["gate_ablation"]
    if ga.get("status") == "missing":
        lines += [f"_missing: {ga.get('reason')} (`{ga.get('file')}`)_"]
    else:
        lines += [
            f"Declared ablation arm: `{ga.get('arm')}`; declared policies: "
            + ", ".join(f"`{p}`" for p in ga.get("declared_policies", []))
            + f" ({ga.get('n_selected_rows')} of {ga.get('n_input_rows')} rows selected).",
            "",
        ]
        if ga.get("missing_policies"):
            lines += [
                "Declared but absent from the outcomes: "
                + ", ".join(f"`{p}`" for p in ga["missing_policies"])
                + ".",
                "",
            ]
        lines += _table(_METHOD_HEADER, _method_rows(ga["summary"]))

    lines += ["", "## Instrumentation cost", ""]
    lines += _instrumentation_lines(sections.get("instrumentation", {}))

    lines += ["", "## Replay fidelity (online future table vs offline replay)", ""]
    lines += _fidelity_lines(sections.get("method_comparison", {}))

    lines += ["", "## Contention", ""]
    sec = sections["contention"]
    if sec.get("status") == "missing":
        lines += [f"_missing: {sec.get('reason')} (`{sec.get('file')}`)_"]
    elif sec.get("status") == "empty":
        lines += ["_input present but contained no rows_"]
    else:
        fields = sec["numeric_fields"]
        lines += _table(
            [sec.get("key_field") or "group", "n"] + [f"{f} p50" for f in fields],
            [
                [g["key"], str(g["n"])]
                + [_f((g["fields"].get(f) or {}).get("p50"), 3) for f in fields]
                for g in sec["groups"]
            ],
        )

    lines += [
        "",
        "## Provenance",
        "",
        f"- generated_at (UTC, informational only): {results['generated_at']}",
        f"- bootstrap: n={_BOOT_N}, seed={_BOOT_SEED}, cluster=prompt_id",
        f"- alpha: {_ALPHA}",
        "",
    ]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def read_frozen_policy(path: str | Path | None) -> dict[str, Any]:
    """Load the frozen gate thresholds written by ``crystalcall pilot``.

    Thresholds (``k``, ``tau``, ``min_overlap_ms``, ...) are never hardcoded
    here: they are read from ``configs/frozen_policy.json`` and echoed into the
    report together with the file's sha256.  A missing file is reported, not
    invented.
    """
    if path is None:
        return {"status": "missing", "reason": "no frozen policy path supplied"}
    p = Path(path)
    if not p.is_file():
        return {"status": "missing", "reason": "file not found", "path": str(p)}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return {"status": "invalid", "reason": str(exc), "path": str(p)}
    return {
        "status": "present",
        "path": str(p),
        "sha256": sha256_file(p),
        "policy": payload,
    }


def aggregate(
    derived_dir: Path,
    out_json: Path,
    out_md: Path,
    *,
    frozen_policy_path: str | Path | None = Path("configs/frozen_policy.json"),
    main_table_path: str | Path = MAIN_TABLE_PATH,
) -> dict:
    """Build ``results.json`` + ``results.md`` from a derived directory.

    Returns the same dictionary that is written to ``out_json`` (the JSON file is
    a verbatim dump).  Missing inputs become sections with
    ``"status": "missing"``; every dropped row is counted in ``exclusions``; the
    S1-S8 submission gate block is promoted to the top level.
    """
    derived = Path(derived_dir)
    sources = {name: read_source(derived, name) for name in sorted(DERIVED_SOURCES)}
    spec = load_main_table(main_table_path)

    sections: dict[str, Any] = {}
    exclusion_rows: list[tuple[str, str, int]] = []

    traj = sources["trajectory_analyses"]
    if traj["status"] == "missing":
        sections["crystallization"] = _missing("input file not found", traj["file"])
    else:
        # Crystallization milestones are a property of a *diffusion* draft
        # stream.  The autoregressive control has no drafts (its revisions are
        # a monotone projection of the token stream), so pooling its
        # trajectories would silently move the parse rate, the decode median
        # and every milestone denominator.
        diffusion_rows = [
            r
            for r in traj["_rows"]
            if str(r.get("trajectory_id", "")).split("/")[0] == "diffusion_standard"
        ]
        sections["crystallization"] = crystallization_section(diffusion_rows)
        sections["crystallization"]["scope"] = "diffusion_standard arm only (AR control and online validation arm excluded)"
        for reason, count in sections["crystallization"]["exclusions"].items():
            exclusion_rows.append(("trajectory_analyses", reason, count))
        cs = sections["crystallization"]
        if cs.get("parse_failures_total"):
            exclusion_rows.append(
                ("trajectory_analyses", "parse_failure", int(cs["parse_failures_total"]))
            )
        for reason, count in (cs.get("rejection_counts") or {}).items():
            exclusion_rows.append(("canonicalization", reason, int(count)))
        invalid_final = int(cs["valid_final_parse"]["n"]) - int(cs["valid_final_parse"]["k"])
        if invalid_final:
            exclusion_rows.append(
                ("trajectory_analyses", "invalid_final_parse", invalid_final)
            )

    sandbox = sources["toolsandbox_episodes"]
    sandbox_rows = sandbox["_rows"]
    if sandbox["status"] == "missing":
        sections["toolsandbox"] = _missing("input file not found", sandbox["file"])
    else:
        sections["toolsandbox"] = toolsandbox_section(sandbox_rows)
        for reason, count in sections["toolsandbox"]["exclusions"].items():
            exclusion_rows.append(("toolsandbox_episodes", reason, int(count)))

    outcomes = sources["policy_outcomes"]
    if outcomes["status"] == "missing":
        sections["method_comparison"] = _missing("input file not found", outcomes["file"])
        sections["paired_tests"] = _missing("input file not found", outcomes["file"])
        sections["deadline_success"] = _missing("input file not found", outcomes["file"])
        sections["deadline_paired"] = _missing("input file not found", outcomes["file"])
        sections["main_table"] = _missing("input file not found", outcomes["file"])
    else:
        summary = summarize_outcomes(
            outcomes["_rows"],
            reference=spec.reference,
            candidate=spec.candidate,
            decisive=spec.decisive,
            verification=spec.validation or (metrics.VERIFICATION_ROW,),
            external_divergence_rows=sandbox_rows,
            n_boot=_BOOT_N,
            seed=_BOOT_SEED,
            alpha=_ALPHA,
        )
        sections["method_comparison"] = {
            "status": outcomes["status"],
            "summary": summary,
        }
        sections["main_table"] = main_table_section(spec, outcomes["_rows"])
        sections["paired_tests"] = paired_tests_section(
            outcomes["_rows"], reference=spec.reference
        )
        sections["deadline_success"] = deadline_success(
            outcomes["_rows"], spec.deadline_budgets_s or metrics.DEFAULT_BUDGETS_S, alpha=_ALPHA
        )
        # Marginal Q(B) cannot carry a claim; the paired grid can. Comparators
        # come from the manifest (reference + decisive rows), never hardcoded.
        sections["deadline_paired"] = metrics.deadline_paired(
            outcomes["_rows"],
            spec.deadline_budgets_s or metrics.DEFAULT_BUDGETS_S,
            comparators=[
                ap.key
                for ap in [ArmPolicy.coerce(spec.reference)]
                + [ArmPolicy.coerce(d) for d in spec.decisive]
            ],
            alpha=_ALPHA,
        )
        for reason, count in summary["exclusions"].items():
            exclusion_rows.append(("policy_outcomes", reason, count))
        acc = summary["accounting"]
        for label in ("unpaired_episodes", "cancelled_launches", "duplicate_launches"):
            if acc[label]:
                exclusion_rows.append(("policy_outcomes", label, int(acc[label])))
        if acc["correct_unknown"]:
            exclusion_rows.append(
                ("policy_outcomes", "correctness_unknown", int(acc["correct_unknown"]))
            )
        # Verification coverage holes are exclusions from the losslessness claim.
        ver = acc["verification"]
        for label in (
            "claims_verification_error",
            "claims_unverified",
            "claims_unaccounted",
            "episodes_verification_incomplete",
        ):
            if ver[label]:
                exclusion_rows.append(("claim_verification", label, int(ver[label])))

    ablation = sources["gate_ablations"]
    if ablation["status"] == "missing":
        sections["gate_ablation"] = _missing("input file not found", ablation["file"])
    else:
        declared = set(spec.ablation_policies)
        ab_rows = [
            r
            for r in ablation["_rows"]
            if str(r.get("arm", "")) in {spec.ablation_arm, spec.reference.arm}
            and (not declared or str(r.get("policy", "")) in declared | {spec.reference.policy})
        ]
        ab_summary = summarize_outcomes(
            ab_rows,
            reference=spec.reference,
            candidate=spec.candidate,
            decisive=spec.decisive,
            n_boot=_BOOT_N,
            seed=_BOOT_SEED,
            alpha=_ALPHA,
            gates=False,
        )
        sections["gate_ablation"] = {
            "status": ablation["status"],
            "arm": spec.ablation_arm,
            "declared_policies": list(spec.ablation_policies),
            "n_input_rows": len(ablation["_rows"]),
            "n_selected_rows": len(ab_rows),
            "summary": ab_summary,
            "missing_policies": [
                p
                for p in spec.ablation_policies
                if not any(
                    str(r.get("policy", "")) == p and str(r.get("arm", "")) == spec.ablation_arm
                    for r in ab_rows
                )
            ],
        }
        for reason, count in ab_summary["exclusions"].items():
            exclusion_rows.append(("gate_ablations", reason, count))

    overhead = sources["overhead"]
    sections["instrumentation"] = instrumentation_section(
        outcomes["_rows"] if outcomes["status"] != "missing" else [],
        overhead["_rows"],
        ladder_status="missing" if overhead["status"] == "missing" else "present",
        ladder_file=overhead["file"],
    )
    for rung in sections["instrumentation"]["ladder"]["failing_rungs"]:
        exclusion_rows.append(("overhead_ladder", f"rung_not_ok:{rung}", 1))

    contention = sources["contention"]
    if contention["status"] == "missing":
        sections["contention"] = _missing("input file not found", contention["file"])
    else:
        sections["contention"] = numeric_group_section(contention["_rows"])

    if outcomes["status"] != "missing":
        fidelity = summary["accounting"]["replay_fidelity"]
        if fidelity["mismatching"]:
            exclusion_rows.append(
                ("replay_fidelity", "observed_vs_replay_mismatch", int(fidelity["mismatching"]))
            )

    for name in sorted(sources):
        bad = sources[name]["malformed_lines"]
        if bad:
            exclusion_rows.append((name, "malformed_json_line", int(bad)))

    mc = sections.get("method_comparison", {})
    gate_block = (
        mc["summary"]["submission_gates"]
        if isinstance(mc, dict) and "summary" in mc and "submission_gates" in mc["summary"]
        else _missing("policy outcomes absent, gates cannot be evaluated", "policy_outcomes.jsonl")
    )

    exclusion_rows.sort()
    spec_source = {
        "name": "main_table",
        "file": spec.path.name,
        "path": str(spec.path),
        "status": "present",
        "rows": len(spec.rows),
        "malformed_lines": 0,
        "sha256": spec.sha256,
        "bytes": spec.path.stat().st_size,
    }

    results: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "generated_by": "crystalcall.aggregate",
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "derived_dir": str(derived),
        "alpha": _ALPHA,
        "bootstrap": {"n": _BOOT_N, "seed": _BOOT_SEED, "cluster": "prompt_id"},
        "noninferiority_margin": metrics.NONINFERIORITY_MARGIN,
        "reference_row": spec.reference.to_json(),
        "candidate_row": spec.candidate.to_json(),
        "decisive_rows": [d.to_json() for d in spec.decisive],
        "main_table_spec": spec.to_json(),
        "main_table": sections.get("main_table", _missing("not built", "policy_outcomes.jsonl")),
        "frozen_policy": read_frozen_policy(frozen_policy_path),
        "submission_gates": gate_block,
        "sources": [
            {k: v for k, v in sorted(sources[name].items()) if k != "_rows"}
            for name in sorted(sources)
        ]
        + [dict(sorted(spec_source.items()))],
        "exclusions": {
            "total": int(sum(c for _, _, c in exclusion_rows)),
            "rows": [[scope, reason, count] for scope, reason, count in exclusion_rows],
        },
        "sections": sections,
    }

    out_json = Path(out_json)
    out_md = Path(out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(_json_safe(results), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    out_md.write_text(render_markdown(results), encoding="utf-8")
    return results
