"""Paper figures: matplotlib only, Agg backend, deterministic, PDF + PNG.

Every function takes *data* plus an output directory and returns the list of
written paths.  Nothing here invents data: a figure whose input is absent is
skipped by :func:`render_all` (and reported on stdout), and
:func:`fig_architecture_timeline` raises rather than drawing a placeholder
trajectory.

Output goes to ``experiments/figures/<stem>.pdf`` and ``<stem>.png`` at 200 dpi.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow use("Agg"))
import numpy as np  # noqa: E402

from .aggregate import DERIVED_SOURCES, load_jsonl  # noqa: E402
from .metrics import (  # noqa: E402
    ArmPolicy,
    DECISIVE_ROWS,
    REFERENCE_ROW,
    binom_ci_exact,
    summarize_outcomes,
)

__all__ = [
    "DPI",
    "FIGURE_STEMS",
    "fig_architecture_timeline",
    "fig_crystallization_cdfs",
    "fig_latency_pareto",
    "fig_category_breakdown",
    "fig_gate_ablation",
    "fig_contention",
    "render_all",
]

DPI = 200

FIGURE_STEMS: tuple[str, ...] = (
    "fig_architecture_timeline",
    "fig_crystallization_cdfs",
    "fig_latency_pareto",
    "fig_latency_pareto_decisive",
    "fig_category_breakdown",
    "fig_gate_ablation",
    "fig_contention",
)

#: Fixed, colour-blind-safe palette (Okabe-Ito); indexed deterministically.
_PALETTE: tuple[str, ...] = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#000000",
    "#F0E442",
)

_MILESTONES: tuple[tuple[str, str, str], ...] = (
    ("first_name_ns", "tool name readable", "o"),
    ("first_args_ns", "arguments readable", "s"),
    ("first_exec_ns", "full call readable (executable)", "^"),
    ("invariant_ns", "call stops changing", "D"),
    ("committed_ns", "call locked in (committed)", "*"),
)

_CDF_SERIES: tuple[tuple[str, str], ...] = tuple(
    (key, label) for key, label, _ in _MILESTONES
) + (
    ("canvas_done_ns", "draft finished"),
    ("response_done_ns", "answer finished"),
)

_CONTENTION_X = ("workers", "concurrency", "parallelism", "level", "n_tools", "clients", "load")
_CONTENTION_Y = (
    "decode_ms",
    "decode_ms_per_revision",
    "tokens_per_s",
    "completion_ms",
    "latency_ms",
    "p50_ms",
    "throughput",
)

_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.hashsalt": "crystalcall",
    "figure.constrained_layout.use": True,
}

_PDF_METADATA = {
    "Creator": "crystalcall.figures",
    "Producer": "crystalcall.figures",
    "CreationDate": None,
}
_PNG_METADATA = {"Software": "crystalcall.figures"}

# This module exists solely to render paper figures, so the deterministic style
# is applied globally at import: rcParams are consulted at draw time, and a
# context manager around figure *creation* alone would not cover savefig.
plt.rcParams.update(_RC)


# --------------------------------------------------------------------------- #
# plumbing
# --------------------------------------------------------------------------- #


class MissingFigureData(ValueError):
    """Raised when a figure is asked to draw without the data it depends on."""


def _save(fig: "plt.Figure", out_dir: str | Path, stem: str) -> list[Path]:
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    pdf = d / f"{stem}.pdf"
    png = d / f"{stem}.png"
    fig.savefig(pdf, dpi=DPI, metadata=_PDF_METADATA)
    fig.savefig(png, dpi=DPI, metadata=_PNG_METADATA)
    plt.close(fig)
    return [pdf, png]


def _num(row: Mapping[str, Any], key: str) -> float | None:
    v = row.get(key)
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)) and math.isfinite(float(v)):
        return float(v)
    return None


def _finite_or(value: Any, fallback: float) -> float:
    """``value`` as a float when it is finite, otherwise ``fallback`` (sort-safe)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return fallback
    return v if math.isfinite(v) else fallback


def _t0_ns(analysis: Mapping[str, Any]) -> float | None:
    """Decode start of a trajectory, in ns, derived from the analysis row."""
    calls = list(analysis.get("final_calls") or [])
    ends = [float(c.get("response_done_ns", 0) or 0) for c in calls]
    ends = [e for e in ends if e > 0]
    decode_ms = _num(analysis, "decode_ms")
    if ends and decode_ms is not None and decode_ms > 0:
        return max(ends) - decode_ms * 1e6
    stamps = [
        float(c.get(k, 0) or 0)
        for c in calls
        for k, _, _ in _MILESTONES
        if float(c.get(k, 0) or 0) > 0
    ]
    return min(stamps) if stamps else None


def _ecdf(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray([v for v in values if math.isfinite(v)], dtype=float))
    if arr.size == 0:
        return np.empty(0), np.empty(0)
    y = np.arange(1, arr.size + 1, dtype=float) / arr.size
    return arr, y


def _paired_gains(
    rows: Sequence[Mapping[str, Any]],
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any] = REFERENCE_ROW,
) -> list[dict[str, Any]]:
    """Per-episode relative gain against the reference ``(arm, policy)`` row."""
    ref = ArmPolicy.coerce(reference)
    baseline: dict[tuple[float, str, int], float] = {}
    for r in rows:
        if r.get("exclusion") or (str(r.get("arm", "")), str(r.get("policy", ""))) != ref.key:
            continue
        comp = _num(r, "completion_ms")
        if comp and comp > 0:
            baseline[
                (
                    float(r.get("tool_duration_ms", 0.0) or 0.0),
                    str(r.get("prompt_id", "")),
                    int(r.get("seed", 0)),
                )
            ] = comp
    out: list[dict[str, Any]] = []
    for r in rows:
        if r.get("exclusion"):
            continue
        dur = float(r.get("tool_duration_ms", 0.0) or 0.0)
        key = (dur, str(r.get("prompt_id", "")), int(r.get("seed", 0)))
        base = baseline.get(key)
        comp = _num(r, "completion_ms")
        if base is None or comp is None:
            continue
        arm = str(r.get("arm", ""))
        policy = str(r.get("policy", ""))
        out.append(
            {
                "arm": arm,
                "policy": policy,
                "row_key": f"{arm}/{policy}",
                "tool_duration_ms": dur,
                "category": str(r.get("category", "unknown")),
                "prompt_id": key[1],
                "seed": key[2],
                "gain": 1.0 - comp / base,
            }
        )
    return out


# --------------------------------------------------------------------------- #
# 1. architecture + early-issue / late-commit timeline
# --------------------------------------------------------------------------- #


def _architecture_panel(ax: "plt.Axes") -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    boxes = (
        "DiffusionGemma\ndenoiser",
        "canvas revision\n(uncommitted)",
        "PTC parse +\ncanonicalize",
        "revision gate\n(k, tau, overlap)",
        "future table\n(quarantined)",
        "OMP host tools\n(authoritative)",
    )
    width = 10.0 / len(boxes)
    for i, label in enumerate(boxes):
        x = i * width
        ax.add_patch(
            plt.Rectangle(
                (x + 0.12, 0.5),
                width - 0.24,
                1.1,
                facecolor="white",
                edgecolor=_PALETTE[i % len(_PALETTE)],
                linewidth=1.1,
            )
        )
        ax.text(x + width / 2.0, 1.05, label, ha="center", va="center", fontsize=7.0)
        if i + 1 < len(boxes):
            ax.annotate(
                "",
                xy=(x + width + 0.10, 1.05),
                xytext=(x + width - 0.10, 1.05),
                arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#444444"},
            )
    ax.text(
        0.0,
        0.05,
        "early issue: speculative launch on an uncommitted revision   |   "
        "late commit: result claimed only if the final canonical call matches",
        fontsize=7.0,
        va="bottom",
    )


def fig_architecture_timeline(
    analysis: Mapping[str, Any] | None,
    out_dir: str | Path,
    *,
    stem: str = "fig_architecture_timeline",
    max_calls: int = 6,
) -> list[Path]:
    """Architecture schematic plus the measured early-issue/late-commit timeline.

    ``analysis`` must be a real ``TrajectoryAnalysis.to_json()`` dict; the
    pre-commit interval (invariant -> committed) is drawn and annotated **in
    milliseconds** from that measurement.  Passing ``None`` (or an analysis with
    no usable timestamps) raises :class:`MissingFigureData` -- this figure is
    never drawn from synthetic data.
    """
    if not analysis:
        raise MissingFigureData(
            "fig_architecture_timeline requires a measured trajectory analysis; "
            "refusing to draw synthetic data"
        )
    calls = [c for c in (analysis.get("final_calls") or []) if int(c.get("invariant_ns", 0) or 0)]
    if not calls:
        raise MissingFigureData(
            f"trajectory {analysis.get('trajectory_id')!r} has no final call with an "
            "invariant timestamp; nothing measured to draw"
        )
    t0 = _t0_ns(analysis)
    if t0 is None:
        raise MissingFigureData("cannot locate decode start (no decode_ms, no timestamps)")

    calls = sorted(calls, key=lambda c: (int(c.get("invariant_ns", 0)), str(c.get("identity"))))
    calls = calls[:max_calls]

    def ms(ns: Any) -> float | None:
        v = float(ns or 0)
        return (v - t0) / 1e6 if v > 0 else None

    fig = plt.figure(figsize=(7.2, 2.0 + 0.55 * len(calls) + 1.1))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.15 + 0.32 * len(calls)])
    _architecture_panel(fig.add_subplot(gs[0]))
    ax = fig.add_subplot(gs[1])

    canvas_done = ms(max((c.get("canvas_done_ns", 0) or 0) for c in calls))
    response_done = ms(max((c.get("response_done_ns", 0) or 0) for c in calls))

    for lane, call in enumerate(calls):
        y = len(calls) - 1 - lane
        inv = ms(call.get("invariant_ns"))
        com = ms(call.get("committed_ns"))
        if inv is not None and com is not None and com >= inv:
            ax.barh(
                y,
                com - inv,
                left=inv,
                height=0.34,
                color="#0072B2",
                alpha=0.18,
                edgecolor="#0072B2",
                linewidth=0.6,
                zorder=1,
            )
            ax.annotate(
                f"pre-commit lead {com - inv:.1f} ms",
                xy=((inv + com) / 2.0, y + 0.26),
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#0072B2",
            )
        for i, (key, label, marker) in enumerate(_MILESTONES):
            t = ms(call.get(key))
            if t is None:
                continue
            ax.plot(
                [t],
                [y],
                marker=marker,
                markersize=5.5,
                color=_PALETTE[i % len(_PALETTE)],
                linestyle="none",
                label=label if lane == 0 else None,
                zorder=3,
            )
        ax.text(
            -0.01,
            y,
            f"{call.get('tool', '?')}",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=7.0,
        )

    for t, label, color in (
        (canvas_done, "canvas complete", "#009E73"),
        (response_done, "response complete", "#D55E00"),
    ):
        if t is None:
            continue
        ax.axvline(t, color=color, linestyle="--", linewidth=0.9, zorder=2)
        ax.annotate(
            f"{label} @ {t:.1f} ms",
            xy=(t, 0.98),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-3, 0),
            textcoords="offset points",
            rotation=90,
            fontsize=6.8,
            color=color,
            ha="right",
            va="top",
        )

    ax.set_yticks(range(len(calls)))
    ax.set_yticklabels([])
    ax.set_ylim(-0.6, len(calls) - 0.15)
    ax.margins(x=0.06)
    ax.set_xlabel("time since decode start (ms)")
    ax.set_title(
        f"early issue / late commit on trajectory {analysis.get('trajectory_id', '?')} "
        f"(prompt {analysis.get('prompt_id', '?')}, seed {analysis.get('seed', '?')})"
    )
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, framealpha=0.9)
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# 2. crystallization CDFs
# --------------------------------------------------------------------------- #


def fig_crystallization_cdfs(
    analyses: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "fig_crystallization_cdfs",
) -> list[Path]:
    """Empirical CDFs of every crystallization milestone (ms since decode start)."""
    series: dict[str, list[float]] = {label: [] for _, label in _CDF_SERIES}
    for analysis in analyses or ():
        if analysis.get("exclusion"):
            continue
        t0 = _t0_ns(analysis)
        if t0 is None:
            continue
        for call in analysis.get("final_calls") or []:
            for key, label in _CDF_SERIES:
                raw = float(call.get(key, 0) or 0)
                if raw > 0:
                    series[label].append((raw - t0) / 1e6)
    if not any(series.values()):
        raise MissingFigureData("no crystallization timestamps found in the supplied analyses")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for i, (_, label) in enumerate(_CDF_SERIES):
        xs, ys = _ecdf(series[label])
        if xs.size == 0:
            continue
        color = _PALETTE[i % len(_PALETTE)]
        for ax in axes:
            ax.step(
                np.concatenate([[0.0], xs]),
                np.concatenate([[0.0], ys]),
                where="post",
                label=f"{label} (n={xs.size})",
                color=color,
                linewidth=1.2,
            )
    axes[0].set_xlabel("time since generation start (ms)")
    axes[0].set_ylabel("fraction of final calls")
    axes[0].set_title("when a tool call reaches each milestone")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("time since generation start (ms, log)")
    axes[1].set_title("same, log scale")
    for ax in axes:
        ax.set_ylim(0.0, 1.02)
    axes[1].legend(loc="lower right", fontsize=6.5)
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# 3. latency-risk Pareto
# --------------------------------------------------------------------------- #


def fig_latency_pareto(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "fig_latency_pareto",
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any] = REFERENCE_ROW,
    n_boot: int = 2000,
    seed: int = 0,
) -> list[Path]:
    """Median latency gain against speculative precision and wasted tool work.

    ``reference`` is the ``(arm, policy)`` row every gain is paired against; pass
    ``DECISIVE_ROWS[0]`` to draw the decisive (S7) comparison.
    """
    if not rows:
        raise MissingFigureData("no policy outcome rows supplied")
    ref = ArmPolicy.coerce(reference)
    summary = summarize_outcomes(
        rows, reference=ref, n_boot=n_boot, seed=seed, gates=False
    )
    groups = [g for g in summary["groups"] if g["gain"]["n_paired"] > 0]
    if not groups:
        raise MissingFigureData(
            f"no episode could be paired against {ref.arm}/{ref.policy}"
        )

    by_duration: dict[float, list[Mapping[str, Any]]] = defaultdict(list)
    for g in groups:
        by_duration[float(g["tool_duration_ms"])].append(g)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for i, dur in enumerate(sorted(by_duration)):
        color = _PALETTE[i % len(_PALETTE)]
        grp = sorted(
            by_duration[dur],
            key=lambda g: (_finite_or(g["precision"]["point"], -1.0), g["row_key"]),
        )
        prec = [g["precision"]["point"] for g in grp]
        gain = [g["gain"]["median"] for g in grp]
        lo = [max(0.0, g["gain"]["median"] - g["gain"]["ci_lo"]) for g in grp]
        hi = [max(0.0, g["gain"]["ci_hi"] - g["gain"]["median"]) for g in grp]
        axes[0].errorbar(
            prec,
            gain,
            yerr=[lo, hi],
            marker="o",
            markersize=4,
            linewidth=1.0,
            capsize=2,
            color=color,
            label=f"D = {dur:g} ms",
        )
        for j, (g, x, y) in enumerate(zip(grp, prec, gain, strict=True)):
            if x is None or not math.isfinite(x):
                continue
            axes[0].annotate(
                g["row_key"],
                xy=(x, y),
                xytext=(4, 4 if j % 2 == 0 else -9),
                textcoords="offset points",
                fontsize=6.0,
                color=color,
                clip_on=False,
            )
        waste = [
            g["waste"]["wasted_tool_ms_total"] / g["n"] if g["n"] else float("nan") for g in grp
        ]
        order = np.argsort(np.asarray(waste, dtype=float), kind="stable")
        axes[1].plot(
            [waste[j] for j in order],
            [gain[j] for j in order],
            marker="s",
            markersize=4,
            linewidth=1.0,
            color=color,
            label=f"D = {dur:g} ms",
        )
        for rank, j in enumerate(order):
            axes[1].annotate(
                grp[j]["row_key"],
                xy=(waste[j], gain[j]),
                xytext=(4, 4 if rank % 2 == 0 else -9),
                textcoords="offset points",
                fontsize=6.0,
                color=color,
                clip_on=False,
            )
        axes[0].margins(x=0.18, y=0.12)
        axes[1].margins(x=0.20, y=0.12)

    ref_label = f"{ref.arm}/{ref.policy}"
    axes[0].set_xlabel("speculative precision (true launches / launches)")
    axes[0].set_ylabel(f"median paired gain vs {ref_label}")
    axes[0].set_title(f"latency-risk Pareto: gain vs precision\n(paired vs {ref_label})")
    axes[1].set_xlabel("wasted tool time per episode (ms)")
    axes[1].set_ylabel(f"median paired gain vs {ref_label}")
    axes[1].set_title(f"gain vs wasted work\n(paired vs {ref_label})")
    axes[0].axhline(0.0, color="#666666", linewidth=0.7, linestyle=":")
    axes[1].axhline(0.0, color="#666666", linewidth=0.7, linestyle=":")
    axes[0].legend(loc="best")
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# 4. category breakdown
# --------------------------------------------------------------------------- #


def fig_category_breakdown(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "fig_category_breakdown",
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any] = REFERENCE_ROW,
) -> list[Path]:
    """Median paired latency gain per prompt category, one panel per tool duration."""
    ref = ArmPolicy.coerce(reference)
    gains = _paired_gains(rows or (), ref)
    if not gains:
        raise MissingFigureData(
            "no paired episodes available for a category breakdown vs "
            f"{ref.arm}/{ref.policy}"
        )

    buckets: dict[tuple[float, str, str], list[float]] = defaultdict(list)
    for g in gains:
        buckets[(g["tool_duration_ms"], g["category"], g["row_key"])].append(g["gain"])
    durations = sorted({k[0] for k in buckets})
    categories = sorted({k[1] for k in buckets})
    policies = sorted({k[2] for k in buckets})

    fig, axes = plt.subplots(
        1, len(durations), figsize=(max(3.4, 2.6 * len(durations)), 3.0), squeeze=False
    )
    width = 0.8 / max(1, len(policies))
    idx = np.arange(len(categories), dtype=float)
    for a, dur in enumerate(durations):
        ax = axes[0][a]
        for p, policy in enumerate(policies):
            vals = [
                float(np.median(buckets[(dur, cat, policy)]))
                if buckets.get((dur, cat, policy))
                else float("nan")
                for cat in categories
            ]
            ax.bar(
                idx + p * width - 0.4 + width / 2.0,
                vals,
                width=width,
                color=_PALETTE[p % len(_PALETTE)],
                label=policy if a == 0 else None,
                edgecolor="none",
            )
        ax.set_xticks(idx)
        ax.set_xticklabels(categories, rotation=30, ha="right")
        ax.axhline(0.0, color="#666666", linewidth=0.7)
        ax.set_title(f"D = {dur:g} ms")
        if a == 0:
            ax.set_ylabel(f"median paired gain vs {ref.arm}/{ref.policy}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        axes[0][0].legend(loc="best", fontsize=6.5, ncols=1)
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# 5. gate ablation
# --------------------------------------------------------------------------- #


def fig_gate_ablation(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "fig_gate_ablation",
    n_boot: int = 2000,
    seed: int = 0,
) -> list[Path]:
    """Gate ablation: speculative precision (exact CI) and gain per tool duration."""
    if not rows:
        raise MissingFigureData("no gate ablation rows supplied")
    summary = summarize_outcomes(rows, n_boot=n_boot, seed=seed, gates=False)
    if not summary["groups"]:
        raise MissingFigureData("gate ablation rows contained no usable group")

    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    gain_by_policy: dict[str, dict[float, float]] = defaultdict(dict)
    for g in summary["groups"]:
        agg[g["row_key"]][0] += int(g["precision"]["k"])
        agg[g["row_key"]][1] += int(g["precision"]["n"])
        gain_by_policy[g["row_key"]][float(g["tool_duration_ms"])] = g["gain"]["median"]
    policies = sorted(agg)
    durations = sorted({d for m in gain_by_policy.values() for d in m})
    arms = sorted({g["arm"] for g in summary["groups"]})
    # One ablation arm is the normal case: label ticks with the policy alone and
    # name the arm in the title instead of repeating it on every tick.
    tick_labels = (
        [p.split("/", 1)[1] for p in policies] if len(arms) == 1 else list(policies)
    )
    arm_suffix = f" ({arms[0]})" if len(arms) == 1 else ""

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    xs = np.arange(len(policies), dtype=float)
    points, los, his = [], [], []
    for policy in policies:
        k, n = agg[policy]
        point, lo, hi = binom_ci_exact(k, n)
        points.append(point)
        los.append(0.0 if not math.isfinite(point) else max(0.0, point - lo))
        his.append(0.0 if not math.isfinite(point) else max(0.0, hi - point))
    axes[0].bar(xs, points, width=0.62, color=_PALETTE[0], edgecolor="none")
    axes[0].errorbar(
        xs, points, yerr=[los, his], fmt="none", ecolor="#222222", capsize=2, linewidth=0.9
    )
    for x, policy in zip(xs, policies, strict=True):
        k, n = agg[policy]
        axes[0].annotate(
            f"{k}/{n}", xy=(x, 0.02), ha="center", va="bottom", fontsize=6.0, rotation=90
        )
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(tick_labels, rotation=35, ha="right")
    axes[0].set_ylabel("speculative precision (exact 95% CI)")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title(f"gate ablation: launch precision{arm_suffix}")

    for i, dur in enumerate(durations):
        ys = [gain_by_policy[p].get(dur, float("nan")) for p in policies]
        axes[1].plot(
            xs,
            ys,
            marker="o",
            markersize=4,
            linewidth=1.0,
            color=_PALETTE[i % len(_PALETTE)],
            label=f"D = {dur:g} ms",
        )
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(tick_labels, rotation=35, ha="right")
    axes[1].axhline(0.0, color="#666666", linewidth=0.7, linestyle=":")
    axes[1].set_ylabel("median paired latency gain")
    axes[1].set_title("gate ablation: latency gain")
    if durations:
        axes[1].legend(loc="best")
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# 6. contention
# --------------------------------------------------------------------------- #


def fig_contention(
    rows: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    *,
    stem: str = "fig_contention",
) -> list[Path]:
    """Measured GPU/host contention: metric vs load level, one line per scenario."""
    if not rows:
        raise MissingFigureData("no contention measurements supplied")
    x_field = next((f for f in _CONTENTION_X if any(f in r for r in rows)), None)
    y_field = next((f for f in _CONTENTION_Y if any(_num(r, f) is not None for r in rows)), None)
    if y_field is None:
        numeric = sorted(
            {k for r in rows for k in r if _num(r, k) is not None and k != x_field}
        )
        y_field = numeric[0] if numeric else None
    if y_field is None:
        raise MissingFigureData("contention rows contain no numeric metric to plot")

    scen_field = next(
        (f for f in ("scenario", "case", "condition", "name", "label") if any(f in r for r in rows)),
        None,
    )
    buckets: dict[str, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        scen = str(r.get(scen_field, "all")) if scen_field else "all"
        y = _num(r, y_field)
        if y is None:
            continue
        x = _num(r, x_field) if x_field else None
        buckets[scen][0.0 if x is None else x].append(y)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    if x_field is None:
        labels = sorted(buckets)
        vals = [float(np.median(buckets[s][0.0])) for s in labels]
        ax.bar(np.arange(len(labels), dtype=float), vals, width=0.6, color=_PALETTE[0])
        ax.set_xticks(np.arange(len(labels), dtype=float))
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_xlabel("scenario")
    else:
        for i, scen in enumerate(sorted(buckets)):
            xs = sorted(buckets[scen])
            ys = [float(np.median(buckets[scen][x])) for x in xs]
            ax.plot(
                xs,
                ys,
                marker="o",
                markersize=4,
                linewidth=1.1,
                color=_PALETTE[i % len(_PALETTE)],
                label=scen,
            )
        ax.set_xlabel(x_field)
        if len(buckets) > 1:
            ax.legend(loc="best")
    ax.set_ylabel(f"median {y_field}")
    ax.set_title("speculation contention cost")
    return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# CLI-callable driver
# --------------------------------------------------------------------------- #


def _pick_example(analyses: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Deterministically choose the median-lead trajectory as the timeline example."""
    scored: list[tuple[float, str, Mapping[str, Any]]] = []
    for a in analyses:
        if a.get("exclusion"):
            continue
        leads = [
            float(c["lead_ms"])
            for c in (a.get("final_calls") or [])
            if isinstance(c.get("lead_ms"), (int, float)) and math.isfinite(float(c["lead_ms"]))
        ]
        if not leads:
            continue
        scored.append((float(np.median(leads)), str(a.get("trajectory_id", "")), a))
    if not scored:
        return None
    scored.sort(key=lambda t: (t[0], t[1]))
    return scored[len(scored) // 2][2]


def render_all(derived_dir: str | Path, out_dir: str | Path) -> list[Path]:
    """Render every figure whose input data exists; report the rest as skipped.

    Skips are printed as ``skip <stem>: <reason>`` and never fabricate data.
    Returns the sorted list of written file paths.
    """
    derived = Path(derived_dir)
    data: dict[str, list[dict[str, Any]]] = {}
    for name, fname in sorted(DERIVED_SOURCES.items()):
        path = derived / fname
        if path.is_file():
            rows, malformed = load_jsonl(path)
            if malformed:
                print(f"warn {fname}: {malformed} malformed line(s) ignored")
            data[name] = rows
        else:
            data[name] = []

    analyses = data["trajectory_analyses"]
    outcomes = data["policy_outcomes"]
    ablations = data["gate_ablations"] or outcomes

    jobs: list[tuple[str, Any]] = [
        (
            "fig_architecture_timeline",
            lambda: fig_architecture_timeline(_pick_example(analyses), out_dir),
        ),
        ("fig_crystallization_cdfs", lambda: fig_crystallization_cdfs(analyses, out_dir)),
        ("fig_latency_pareto", lambda: fig_latency_pareto(outcomes, out_dir)),
        (
            "fig_latency_pareto_decisive",
            lambda: fig_latency_pareto(
                outcomes,
                out_dir,
                stem="fig_latency_pareto_decisive",
                reference=DECISIVE_ROWS[0],
            ),
        ),
        ("fig_category_breakdown", lambda: fig_category_breakdown(outcomes, out_dir)),
        ("fig_gate_ablation", lambda: fig_gate_ablation(ablations, out_dir)),
        ("fig_contention", lambda: fig_contention(data["contention"], out_dir)),
    ]

    written: list[Path] = []
    skipped: list[str] = []
    with matplotlib.rc_context(_RC):
        for stem, fn in jobs:
            try:
                paths = fn()
            except MissingFigureData as exc:
                skipped.append(stem)
                print(f"skip {stem}: {exc}")
                continue
            written.extend(paths)
            print(f"wrote {stem}: " + ", ".join(str(p) for p in paths))
    print(
        f"figures: {len(jobs) - len(skipped)} rendered, {len(skipped)} skipped"
        + (f" ({', '.join(skipped)})" if skipped else "")
    )
    return sorted(written)
