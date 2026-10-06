# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20", "numpy>=2.1", "scipy>=1.14", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""How much of the oracle gap can a *learned* launch rule recover?

Post-hoc, exploratory analysis; the launch rules were fixed before it was
run.  It answers a question the oracle ceiling leaves open: the final-match
oracle bounds every policy, but does not say how much of the gap between
CrystalCall and that bound an online rule could attain.

Design
------
* Every (candidate, revision) pair of the recorded development traces becomes
  one row.  Features are exactly what an online rule can see at that revision
  (``CandidateState`` + ``CanvasRevision`` + registry + remaining-decode
  estimate); the label is whether the candidate's identity appears in the
  final program.
* A logistic-regression scorer is fitted with leave-one-prompt-out
  cross-validation: the score for every row of prompt *p* comes from a model
  that never saw prompt *p*.
* The learned rule launches an eligible candidate at the first revision whose
  held-out score reaches a threshold.  A grid of thresholds is replayed through
  the unchanged latency simulation (``replay.simulate_policy``) next to the
  library policies, and a *selected* threshold is chosen per fold on the
  training prompts only (max median all-prompt gain at 500 ms), so no
  held-out prompt influences its own threshold.

Writes ``experiments/derived/learned_gate.json``.  CPU only.
"""

from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crystalcall.harness import _load_cost_model, load_tokenizer, resolve_model_dir  # noqa: E402
from crystalcall.manifest import load_manifest, registry_for, sha256_file  # noqa: E402
from crystalcall.parse_ptc import PtcExtractor  # noqa: E402
from crystalcall.pilot import CONSERVATIVE_GATE, _pilot_remaining  # noqa: E402
from crystalcall.replay import (  # noqa: E402
    EXTERNAL_LAUNCHERS,
    analyze_trajectory,
    build_views,
    load_trajectories,
    simulate_policy,
)
from crystalcall.scheduler import GatePolicy, RevisionTracker  # noqa: E402
from crystalcall.tokens import CanvasDecoder  # noqa: E402

SHARD_DIR = REPO / "experiments/raw/diffusion_standard"


def shards() -> list[Path]:
    """Every recorded diffusion seed on the development prompts."""
    return sorted(SHARD_DIR.glob("diffusion_standard-seed*.jsonl"))
DURATIONS = (100.0, 500.0, 2000.0, 5000.0)
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
SELECT_DURATION = 500.0
CATEGORIES = ("multiple", "parallel", "parallel_multiple", "simple")
FEATURES = (
    "step",
    "canvas_id",
    "prefix_len",
    "consecutive",
    "total_seen",
    "disappearances",
    "max_consecutive",
    "steps_since_first_seen",
    "span_accepted",
    "accepted_fraction",
    "min_conf",
    "mean_conf",
    "span_tokens",
    "mean_entropy",
    "converged",
    "phase_commit",
    "remaining_ms",
    "expected_overlap_ms",
    "candidates_in_revision",
    *(f"cat_{c}" for c in CATEGORIES),
)

#: A policy that observes every candidate at every revision and never fires,
#: so the tracker's online state can be read out as features.
NEVER = GatePolicy(
    name="never",
    use_persistence=False,
    use_confidence=False,
    use_overlap=False,
    use_resource=False,
    fixed_step=-1,
)


def _rows_for(traj, views, analysis, reg, remaining, costs) -> list[dict[str, Any]]:
    final_ids = {c.identity for c in analysis.final_calls}
    tracker = RevisionTracker(
        registry=reg,
        policy=NEVER,
        remaining_model=remaining,
        costs=costs,
        request_id=traj.request_id,
        duration_override_ms=SELECT_DURATION,
    )
    rows: list[dict[str, Any]] = []
    for vi, v in enumerate(views):
        n_cands = len(v.parse.candidates)
        for state, decision in tracker.observe(v.revision, v.parse, v.decoded):
            if not decision.checks.get("effect_eligible", False):
                continue
            lo, hi = state.last_token_span
            mask = v.revision.accepted_mask
            span = max(0, hi - lo)
            acc = float(sum(mask[lo:hi]) / span) if span and mask else 0.0
            rows.append(
                {
                    "prompt_id": traj.prompt_id,
                    "trajectory_id": traj.request_id,
                    "seed": traj.seed,
                    "identity": state.identity,
                    "view_index": vi,
                    "label": 1.0 if state.identity in final_ids else 0.0,
                    "step": float(v.revision.step),
                    "canvas_id": float(v.revision.canvas_id),
                    "prefix_len": float(v.revision.prefix_len or 0),
                    "consecutive": float(state.consecutive),
                    "total_seen": float(state.total_seen),
                    "disappearances": float(state.disappearances),
                    "max_consecutive": float(state.max_consecutive),
                    "steps_since_first_seen": float(v.revision.step - state.first_seen_step),
                    "span_accepted": 1.0 if state.first_committed_step >= 0 else 0.0,
                    "accepted_fraction": acc,
                    "min_conf": float(state.last_min_conf if state.last_min_conf is not None else 0.0),
                    "mean_conf": float(state.last_mean_conf if state.last_mean_conf is not None else 0.0),
                    "span_tokens": float(span),
                    "mean_entropy": float(v.revision.mean_entropy or 0.0),
                    "converged": 1.0 if v.revision.converged else 0.0,
                    "phase_commit": 1.0 if v.revision.phase == "commit" else 0.0,
                    "remaining_ms": float(decision.remaining_ms or 0.0),
                    "expected_overlap_ms": float(decision.expected_overlap_ms or 0.0),
                    "candidates_in_revision": float(n_cands),
                    **{f"cat_{c}": 1.0 if traj.category == c else 0.0 for c in CATEGORIES},
                }
            )
    return rows


def _standardize(train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scaler fitted on training rows only; the held-out prompt must not
    contribute its mean or variance to its own features."""
    return train.mean(0), train.std(0) + 1e-9


def _fit(X: np.ndarray, y: np.ndarray, l2: float = 1.0) -> np.ndarray:
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])

    def loss(w: np.ndarray) -> tuple[float, np.ndarray]:
        z = Xb @ w
        p = 1.0 / (1.0 + np.exp(-z))
        eps = 1e-12
        nll = -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
        reg = 0.5 * l2 * np.sum(w[:-1] ** 2) / n
        g = Xb.T @ (p - y) / n
        g[:-1] += l2 * w[:-1] / n
        return float(nll + reg), g

    res = minimize(loss, np.zeros(d + 1), jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"logistic fit did not converge: {res.message}")
    return res.x


def _predict(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(np.hstack([X, np.ones((X.shape[0], 1))]) @ w)))


def _auc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "median": None, "mean": None, "positive": 0}
    return {"n": len(xs), "median": st.median(xs), "mean": st.fmean(xs), "positive": sum(1 for x in xs if x > 0)}


def _launches(rows: list[dict[str, Any]], scores: np.ndarray, theta: float) -> list[tuple[str, int, str]]:
    first: dict[str, tuple[int, float]] = {}
    for r, s in zip(rows, scores, strict=True):
        if s < theta or r["identity"] in first:
            continue
        first[r["identity"]] = (int(r["view_index"]), float(s))
    return [(ident, vi, f"learned_p={s:.3f}") for ident, (vi, s) in first.items()]


def _replay(prepared, launch_table, name, remaining, costs) -> dict[str, Any]:
    """Replay one learned rule (launch_table: trajectory_id -> launches) at every duration."""
    EXTERNAL_LAUNCHERS[name] = lambda traj, views, analysis, reg, dur: launch_table.get(traj.request_id, [])
    per_duration: dict[str, Any] = {}
    for dur in DURATIONS:
        gains = {"all": [], "executed": []}
        launches = false = 0
        per_prompt: dict[str, float] = {}
        for pid, (t, views, analysis, reg) in prepared.items():
            base = simulate_policy(t, views, analysis, reg, "diffusion_final_response", dur, remaining_model=remaining, costs=costs)
            got = simulate_policy(t, views, analysis, reg, name, dur, remaining_model=remaining, costs=costs)
            launches += got.launches
            false += got.false_launches
            if base.completion_ms > 0 and got.completion_ms > 0:
                g = 1.0 - got.completion_ms / base.completion_ms
                gains["all"].append(g)
                per_prompt[pid] = g
                if t.authoritative_calls:
                    gains["executed"].append(g)
        per_duration[str(int(dur))] = {
            "gain": {k: _stats(v) for k, v in gains.items()},
            "gain_per_prompt": per_prompt,
            "launches": launches,
            "false_launches": false,
            "precision": (1.0 - false / launches) if launches else None,
        }
    del EXTERNAL_LAUNCHERS[name]
    return per_duration


def main() -> int:
    recs = {r.prompt_id: r for r in load_manifest(REPO / "experiments/manifests/dev.jsonl")}
    decoder = CanvasDecoder(tokenizer=load_tokenizer(resolve_model_dir("google/diffusiongemma-26B-A4B-it")), cache_size=64)
    costs = _load_cost_model(REPO)
    shard_paths = shards()
    trajs = load_trajectories(shard_paths, verify=True)
    remaining = _pilot_remaining(REPO, trajs)

    # Keyed by trajectory (prompt x seed); folds are grouped by prompt so every
    # seed of a held-out prompt leaves the training set together.
    prepared: dict[str, Any] = {}
    rows_by_traj: dict[str, list[dict[str, Any]]] = {}
    for t in trajs:
        reg = registry_for(recs[t.prompt_id]) if t.prompt_id in recs else None
        if reg is None:
            continue
        views = build_views(t, PtcExtractor(registry=reg), decoder)
        analysis = analyze_trajectory(t, views, reg)
        prepared[t.request_id] = (t, views, analysis, reg)
        rows_by_traj[t.request_id] = _rows_for(t, views, analysis, reg, remaining, costs)

    trajectories = sorted(prepared)
    prompts = sorted({prepared[k][0].prompt_id for k in trajectories})
    prompt_of = {k: prepared[k][0].prompt_id for k in trajectories}
    all_rows = [r for k in trajectories for r in rows_by_traj[k]]
    X_all = np.array([[r[f] for f in FEATURES] for r in all_rows])
    y_all = np.array([r["label"] for r in all_rows])
    owner = np.array([r["prompt_id"] for r in all_rows])
    # ---- leave-one-prompt-out scores + per-fold threshold selection ------ #
    heldout = np.zeros(len(all_rows))
    selected_theta: dict[str, float] = {}
    coef_sum = np.zeros(len(FEATURES) + 1)
    for p in prompts:
        tr = owner != p
        mu, sd = _standardize(X_all[tr])
        Xtr = (X_all[tr] - mu) / sd
        Xte = (X_all[~tr] - mu) / sd
        w = _fit(Xtr, y_all[tr])
        coef_sum += w
        heldout[~tr] = _predict(w, Xte)
        # Threshold selection on TRAINING prompts only: replay each grid value
        # over the training prompts at the selection duration and keep the
        # best median all-prompt gain.  In-sample scores are used here, which
        # only makes the selection optimistic about the *training* prompts;
        # the held-out prompt never sees its own scores.
        tr_scores = _predict(w, Xtr)
        best: tuple[float, float] = (-1.0, THRESHOLDS[0])
        for theta in THRESHOLDS:
            table: dict[str, list[tuple[str, int, str]]] = {}
            idx = np.where(tr)[0]
            per: dict[str, list[tuple[dict[str, Any], float]]] = {}
            for i, s in zip(idx, tr_scores, strict=True):
                per.setdefault(all_rows[i]["trajectory_id"], []).append((all_rows[i], float(s)))
            for q, items in per.items():
                rs = [it[0] for it in items]
                ss = np.array([it[1] for it in items])
                table[q] = _launches(rs, ss, theta)
            name = f"_sel_{p}_{theta}"
            EXTERNAL_LAUNCHERS[name] = lambda traj, views, analysis, reg, dur, _t=table: _t.get(traj.request_id, [])
            gs = []
            for q in trajectories:
                if prompt_of[q] == p:
                    continue
                t, views, analysis, reg = prepared[q]
                base = simulate_policy(t, views, analysis, reg, "diffusion_final_response", SELECT_DURATION, remaining_model=remaining, costs=costs)
                got = simulate_policy(t, views, analysis, reg, name, SELECT_DURATION, remaining_model=remaining, costs=costs)
                if base.completion_ms > 0 and got.completion_ms > 0:
                    gs.append(1.0 - got.completion_ms / base.completion_ms)
            del EXTERNAL_LAUNCHERS[name]
            m = st.median(gs) if gs else -1.0
            if m > best[0]:
                best = (m, theta)
        selected_theta[p] = best[1]

    out: dict[str, Any] = {
        "schema_version": "crystalcall-learned-gate/1",
        "shards": {str(sp.relative_to(REPO)): sha256_file(sp) for sp in shard_paths},
        "trajectories": len(trajectories),
        "seeds": sorted({prepared[k][0].seed for k in trajectories}),
        "protocol": "post-hoc, exploratory; the launch rules were fixed before this analysis",
        "model": "L2 logistic regression on online features; leave-one-prompt-out over all seeds of a prompt, scaler and threshold fitted per fold on training prompts only",
        "features": list(FEATURES),
        "rows": len(all_rows),
        "positive_rows": int(y_all.sum()),
        "prompts": len(prompts),
        "heldout_auc": _auc(heldout, y_all),
        "heldout_logloss": float(-np.mean(y_all * np.log(heldout + 1e-12) + (1 - y_all) * np.log(1 - heldout + 1e-12))),
        "mean_coefficients": dict(zip([*FEATURES, "bias"], (coef_sum / len(prompts)).round(4).tolist(), strict=True)),
        "selected_threshold_histogram": {str(t): sum(1 for v in selected_theta.values() if v == t) for t in THRESHOLDS},
        "durations_ms": list(DURATIONS),
        "heldout_rows": [
            {"prompt_id": r["prompt_id"], "trajectory_id": r["trajectory_id"], "seed": r["seed"], "identity": r["identity"], "view_index": r["view_index"], "label": r["label"], "score": round(float(s), 4)}
            for r, s in zip(all_rows, heldout, strict=True)
        ],
        "policies": {},
    }

    traj_of_row = np.array([r["trajectory_id"] for r in all_rows])

    def table_for(theta_of: dict[str, float]) -> dict[str, list[tuple[str, int, str]]]:
        table: dict[str, list[tuple[str, int, str]]] = {}
        for k in trajectories:
            idx = np.where(traj_of_row == k)[0]
            rs = [all_rows[i] for i in idx]
            table[k] = _launches(rs, heldout[idx], theta_of[prompt_of[k]])
        return table

    for theta in THRESHOLDS:
        out["policies"][f"learned_t{theta}"] = _replay(prepared, table_for({p: theta for p in prompts}), f"learned_t{theta}", remaining, costs)
    out["policies"]["learned_selected"] = _replay(prepared, table_for(selected_theta), "learned_selected", remaining, costs)

    # Reference points from the library, same simulation, same prompts.
    for label, pol in (
        ("crystalcall_conservative", CONSERVATIVE_GATE),
        ("diffusion_earliest_committed", "diffusion_earliest_committed"),
        ("oracle_final_match", "oracle_final_match"),
    ):
        per_duration: dict[str, Any] = {}
        for dur in DURATIONS:
            gains = {"all": [], "executed": []}
            launches = false = 0
            for pid, (t, views, analysis, reg) in prepared.items():
                base = simulate_policy(t, views, analysis, reg, "diffusion_final_response", dur, remaining_model=remaining, costs=costs)
                got = simulate_policy(t, views, analysis, reg, pol, dur, remaining_model=remaining, costs=costs)
                launches += got.launches
                false += got.false_launches
                if base.completion_ms > 0 and got.completion_ms > 0:
                    g = 1.0 - got.completion_ms / base.completion_ms
                    gains["all"].append(g)
                    if t.authoritative_calls:
                        gains["executed"].append(g)
            per_duration[str(int(dur))] = {
                "gain": {k: _stats(v) for k, v in gains.items()},
                "launches": launches,
                "false_launches": false,
                "precision": (1.0 - false / launches) if launches else None,
            }
        out["policies"][label] = per_duration

    path = REPO / "experiments/derived/learned_gate.json"
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO)}  rows={len(all_rows)} pos={int(y_all.sum())} auc={out['heldout_auc']:.3f}")
    print(f"selected thresholds: {out['selected_threshold_histogram']}")
    print(f"  {'policy':30s}{'500ms all/exec/prec':>26s}{'2s all/exec/prec':>26s}")
    for label, per in out["policies"].items():
        cells = []
        for dur in ("500", "2000"):
            g = per[dur]["gain"]
            pr = per[dur]["precision"]
            cells.append(f"{g['all']['median']:.1%}/{g['executed']['median']:.1%}/{(pr if pr is not None else float('nan')):.1%}".rjust(26))
        print(f"  {label:30s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
