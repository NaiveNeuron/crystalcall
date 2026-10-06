# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Freeze the latency ceiling of *any* launch policy on the recorded traces.

The pilot's projected-speedup gates are the ones that decide the submission, so
the decisive question is not "how good is our gate" but "how good could any
gate be". Two retrospective oracles answer it:

* ``oracle_earliest_invariant`` launches at the earliest revision from which
  the canonical call never changes again -- the best any *online* policy could
  do, because nothing observable distinguishes that revision from its
  predecessors;
* ``oracle_final_match`` launches at the first revision where the exact final
  call is executable -- not implementable at all, since it needs the answer.

Both are upper bounds, including on a trained predictor, which is why a failure
here is a statement about the model and not about our gate.

**Three views, all reported.** A median over all 100 development prompts is
contaminated: a trajectory whose program executed no tool has a baseline with
no tool duration in it, so its gain is exactly 0 by construction, and
``completion = max(per_call_done)`` makes a trajectory with one unlaunched
parallel sibling look like no gain either. Conditioning is therefore reported
explicitly rather than chosen:

* ``all``           -- every trajectory, forced zeros included;
* ``executed``      -- trajectories where OMP executed at least one
                       authoritative call, so a speculative result could have
                       been claimed at all;
* ``fully_covered`` -- of those, the ones where the policy launched every
                       eligible final call, which is where the mechanism is
                       actually being measured rather than diluted.

Run:  uv run scripts/oracle_ceiling.py
"""

from __future__ import annotations

import json
import statistics as st
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO / "src"))

from crystalcall.harness import (  # noqa: E402
    _load_cost_model,
    load_tokenizer,
    resolve_model_dir,
)
from crystalcall.manifest import load_manifest, registry_for, sha256_file  # noqa: E402
from crystalcall.parse_ptc import PtcExtractor  # noqa: E402
from crystalcall.pilot import CONSERVATIVE_GATE, _pilot_remaining  # noqa: E402
from crystalcall.replay import (  # noqa: E402
    analyze_trajectory,
    build_views,
    load_trajectories,
    simulate_policy,
)
from crystalcall.scheduler import POLICY_LIBRARY  # noqa: E402
from crystalcall.tokens import CanvasDecoder  # noqa: E402

DURATIONS = (100.0, 500.0, 2000.0, 5000.0)
SHARDS = {
    "attempt1": ["experiments/raw/_attempt1/diffusion_standard/diffusion_standard-seed0.jsonl"],
    # Every recorded seed of the current prompt; pooled, paired per trajectory.
    "attempt2": sorted(
        str(p.relative_to(REPO)) for p in (REPO / "experiments/raw/diffusion_standard").glob("diffusion_standard-seed*.jsonl")
    ),
}
POLICIES: dict[str, Any] = {
    "diffusion_final_canvas": "diffusion_final_canvas",
    "diffusion_earliest_committed": "diffusion_earliest_committed",
    "crystalcall_full_gate": POLICY_LIBRARY["full_gate"],
    "crystalcall_conservative": CONSERVATIVE_GATE,
    "oracle_earliest_invariant": "oracle_earliest_invariant",
    "oracle_final_match": "oracle_final_match",
}


def _stats(xs: list[float]) -> dict[str, Any]:
    if not xs:
        return {"n": 0, "median": None, "p25": None, "p75": None}
    xs = sorted(xs)
    return {
        "n": len(xs),
        "median": st.median(xs),
        "p25": xs[len(xs) // 4],
        "p75": xs[(3 * len(xs)) // 4],
    }


def main() -> int:
    out: dict[str, Any] = {
        "schema_version": "crystalcall-oracle-ceiling/1",
        "durations_ms": list(DURATIONS),
        "views": {
            "all": "every trajectory; gain forced to 0 where nothing executed",
            "executed": "trajectories with >=1 authoritative call",
            "fully_covered": "executed AND the policy launched every eligible final call",
        },
        "attempts": {},
    }
    recs = {r.prompt_id: r for r in load_manifest(REPO / "experiments/manifests/dev.jsonl")}
    decoder = CanvasDecoder(
        tokenizer=load_tokenizer(resolve_model_dir("google/diffusiongemma-26B-A4B-it")),
        cache_size=64,
    )
    costs = _load_cost_model(REPO)

    for tag, rels in SHARDS.items():
        shard_paths = [REPO / rel for rel in rels]
        if not shard_paths or not all(sp.is_file() for sp in shard_paths):
            out["attempts"][tag] = {"status": "missing", "shard": rels}
            continue
        trajs = load_trajectories(shard_paths, verify=True)
        remaining = _pilot_remaining(REPO, trajs)
        prepared = {}
        for t in trajs:
            reg = registry_for(recs[t.prompt_id]) if t.prompt_id in recs else None
            if reg is None:
                continue
            views = build_views(t, PtcExtractor(registry=reg), decoder)
            prepared[t.request_id] = (t, views, analyze_trajectory(t, views, reg), reg)

        blob: dict[str, Any] = {
            "shards": {str(rel): sha256_file(REPO / rel) for rel in rels},
            "seeds": sorted({t.seed for t in trajs}),
            "trajectories": len(trajs),
            "tool_proxy_programs": sum(1 for t in trajs if "tool." in (t.final_program or "")),
            "executed_nothing": sum(1 for t in trajs if not t.authoritative_calls),
            "benchmark_correct": sum(1 for t in trajs if t.benchmark_correct is True),
            "valid_final_parse": sum(1 for t in trajs if t.valid_final_parse),
            "remaining_decode_model": remaining.to_json(),
            "policies": {},
        }
        for label, policy in POLICIES.items():
            per_duration: dict[str, Any] = {}
            for dur in DURATIONS:
                views_gain: dict[str, list[float]] = {"all": [], "executed": [], "fully_covered": []}
                launches = false_launches = 0
                launches_x = false_x = 0
                for _pid, (t, views, analysis, reg) in prepared.items():
                    base = simulate_policy(
                        t, views, analysis, reg, "diffusion_final_response", dur,
                        remaining_model=remaining, costs=costs,
                    )
                    got = simulate_policy(
                        t, views, analysis, reg, policy, dur,
                        remaining_model=remaining, costs=costs,
                    )
                    launches += got.launches
                    false_launches += got.false_launches
                    if t.authoritative_calls:
                        launches_x += got.launches
                        false_x += got.false_launches
                    if not (base.completion_ms > 0 and got.completion_ms > 0):
                        continue
                    gain = 1.0 - got.completion_ms / base.completion_ms
                    views_gain["all"].append(gain)
                    if not t.authoritative_calls:
                        continue
                    views_gain["executed"].append(gain)
                    eligible = got.eligible_final_calls or 0
                    if eligible and (got.covered_final_calls or 0) >= eligible:
                        views_gain["fully_covered"].append(gain)
                per_duration[str(int(dur))] = {
                    "gain": {k: _stats(v) for k, v in views_gain.items()},
                    "launches": launches,
                    "false_launches": false_launches,
                    "precision": (1.0 - false_launches / launches) if launches else None,
                    # Conditioned on an authoritative call existing at all: a
                    # launch on a trajectory that executed nothing can never be
                    # claimed, so it measures the model's instruction
                    # following, not the gate's precision.
                    "launches_executed": launches_x,
                    "false_launches_executed": false_x,
                    "precision_executed": (1.0 - false_x / launches_x) if launches_x else None,
                }
            blob["policies"][label] = per_duration
        out["attempts"][tag] = blob

    path = REPO / "experiments/derived/oracle_ceiling.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {path.relative_to(REPO)}")

    for tag, blob in out["attempts"].items():
        if blob.get("status") == "missing":
            print(f"\n{tag}: MISSING {blob['shard']}")
            continue
        print(
            f"\n{tag}: n={blob['trajectories']} parse_ok={blob['valid_final_parse']} "
            f"tool.-proxy={blob['tool_proxy_programs']} executed_nothing={blob['executed_nothing']} "
            f"correct={blob['benchmark_correct']}"
        )
        print(f"  {'policy':30s}" + "".join(f"{int(d)}ms".rjust(24) for d in (500.0, 2000.0)))
        print(f"  {'':30s}" + "".join("   all / exec / covered".rjust(24) for _ in range(2)))
        for label, per in blob["policies"].items():
            cells = []
            for dur in (500.0, 2000.0):
                g = per[str(int(dur))]["gain"]
                cells.append(
                    "".join(
                        f"{g[v]['median']:.1%}".rjust(8) if g[v]["median"] is not None else "n/a".rjust(8)
                        for v in ("all", "executed", "fully_covered")
                    )
                )
            print(f"  {label:30s}" + "".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
