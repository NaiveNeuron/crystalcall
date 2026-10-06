# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.8", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Render the milestone CDF figure and the launch-rule ceiling figure.

`fig_crystallization_cdfs` -- complete final-matching tool calls appear well
before they commit. Empirical CDFs of each crystallization milestone, in
milliseconds before response completion.

`fig_oracle_ceiling` -- median latency gain against final-response execution
as a function of tool duration, for the realised gate, the mandatory
earliest-committed baseline, and two retrospective oracles that upper-bound
every online policy including a trained predictor. The latency targets are
drawn as horizontal lines, so the reader can see that the 2 s target sits
above the oracle.

Both read only checksummed artifacts: `experiments/derived/pilot_analyses_attempt2.jsonl`
and `experiments/derived/oracle_ceiling.json`.

Run:  uv run scripts/negative_figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments/figures"
DPI = 200

MILESTONES = [
    ("first_name_ms", "first final tool name"),
    ("first_args_ms", "first final required argument"),
    ("first_exec_ms", "first executable final-matching call"),
    ("invariant_ms", "first invariant-through-canvas call"),
    ("committed_ms", "first fully committed valid call"),
    ("canvas_done_ms", "canvas completion"),
]


def _load_analyses(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _cdf(ax: Any, values: list[float], label: str) -> None:
    if not values:
        return
    xs = sorted(values)
    ys = [(i + 1) / len(xs) for i in range(len(xs))]
    ax.step(xs, ys, where="post", label=f"{label} (n={len(xs)})", linewidth=1.6)


def crystallization(rows: list[dict[str, Any]]) -> Path | None:
    """Milliseconds before response completion, per milestone."""
    series: dict[str, list[float]] = {k: [] for k, _ in MILESTONES}
    for row in rows:
        for call in row.get("final_calls") or ():
            resp = call.get("response_done_ns") or 0
            if not resp:
                continue
            for key, _label in MILESTONES:
                raw = call.get(key)
                if raw is None:
                    ns = call.get(key.replace("_ms", "_ns"))
                    if not ns:
                        continue
                    raw = (resp - int(ns)) / 1e6
                if raw is None or raw <= 0:
                    continue
                series[key].append(float(raw))
    if not any(series.values()):
        print("crystallization: no milestone data in the analyses", file=sys.stderr)
        return None
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key, label in MILESTONES:
        _cdf(ax, series[key], label)
    ax.axvline(200.0, color="crimson", linewidth=1.0, linestyle=":")
    ax.annotate("pilot gate: 200 ms", (215, 0.05), fontsize=7, color="crimson")
    ax.set_xlabel("milliseconds before response completion (larger = earlier)")
    ax.set_ylabel("fraction of authoritative final calls")
    ax.set_title(
        "Crystallization of final tool calls in uncommitted canvases\n"
        "attempt 2, 100 development prompts"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_crystallization_cdfs.{ext}", dpi=DPI)
    plt.close(fig)
    return OUT / "fig_crystallization_cdfs.pdf"


LINES = [
    ("diffusion_earliest_committed", "launch when locked in", "o", "-"),
    ("crystalcall_conservative", "CrystalCall (k=2, tau=0.9)", "s", "-"),
    ("oracle_earliest_invariant", "oracle: launch once the call stops changing", "^", "--"),
    ("oracle_final_match", "oracle: knows the final answer", "v", "--"),
]


def oracle_ceiling(
    ceiling: dict[str, Any], attempt: str = "attempt2", view: str = "executed"
) -> Path | None:
    att = ceiling["attempts"].get(attempt)
    if not att or att.get("status") == "missing":
        print(f"oracle_ceiling: {attempt} missing", file=sys.stderr)
        return None
    durations = [int(d) for d in ceiling["durations_ms"]]
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    for key, label, marker, style in LINES:
        per = att["policies"].get(key)
        if not per:
            continue
        ys = []
        for d in durations:
            g = per[str(d)]["gain"][view]["median"]
            ys.append(None if g is None else g * 100.0)
        ax.plot(durations, ys, marker=marker, linestyle=style, label=label, linewidth=1.6)
    ax.axhline(15.0, color="crimson", linewidth=1.0, linestyle=":")
    ax.annotate("target at 500 ms: 15%", (durations[0], 15.4), fontsize=7, color="crimson")
    ax.axhline(25.0, color="darkred", linewidth=1.0, linestyle=":")
    ax.annotate("target at 2 s: 25%", (durations[0], 25.4), fontsize=7, color="darkred")
    ax.set_xscale("log")
    ax.set_xticks(durations)
    ax.set_xticklabels([f"{d} ms" if d < 1000 else f"{d // 1000} s" for d in durations])
    ax.set_xlabel("tool delay")
    ax.set_ylabel("median task time saved (%)")
    ax.set_title(
        f"Ceiling for any launch rule\n"
        f"tasks whose calls ran (n={att['trajectories'] - att['executed_nothing']})"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_oracle_ceiling.{ext}", dpi=DPI)
    plt.close(fig)
    return OUT / "fig_oracle_ceiling.pdf"


def main() -> int:
    made: list[Path] = []
    analyses = REPO / "experiments/derived/pilot_analyses_attempt2.jsonl"
    if analyses.is_file():
        p = crystallization(_load_analyses(analyses))
        if p:
            made.append(p)
    else:
        print(f"missing {analyses}", file=sys.stderr)
    ceiling = REPO / "experiments/derived/oracle_ceiling.json"
    if ceiling.is_file():
        p = oracle_ceiling(json.loads(ceiling.read_text(encoding="utf-8")))
        if p:
            made.append(p)
    else:
        print(f"missing {ceiling}", file=sys.stderr)
    for p in made:
        print(f"wrote {p.relative_to(REPO)}")
    return 0 if len(made) == 2 else 1


if __name__ == "__main__":
    raise SystemExit(main())
