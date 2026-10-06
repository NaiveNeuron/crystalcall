#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["crystalcall"]
#
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Emit `paper/numbers.tex`: every headline number as a LaTeX macro.

No number in the paper is typed by hand. Each macro here is read out of a
committed artifact, so recompiling after `crystalcall analyze` propagates the
change instead of leaving stale prose.

Each macro records its source in a comment so a reader of the `.tex` can find
the artifact field behind any figure in the text.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def load(rel: str) -> Any:
    return json.loads((ROOT / rel).read_text())


def jsonl(rel: str) -> list[dict]:
    path = ROOT / rel
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def pct(x: float, nd: int = 1) -> str:
    return f"{100.0 * float(x):.{nd}f}"


def num(x: float, nd: int = 1) -> str:
    return f"{float(x):.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/numbers.tex")
    args = ap.parse_args()

    res = load("reports/results.json")
    sec = res["sections"]
    cs = sec["crystallization"]
    oracle = load("experiments/derived/oracle_ceiling.json")
    a2 = oracle["attempts"]["attempt2"]["policies"]
    # Rows are keyed `arm/policy`: `crystalcall/full_gate` is the *online* arm
    # (never run, deliberately) while `diffusion_standard/full_gate` is the
    # offline replay the paper reports. Keying by policy alone silently picks
    # the empty online row.
    rows = {f"{r['arm']}/{r['policy']}": r for r in sec["main_table"]["rows"]}
    doctor = load("reports/doctor.json")

    def cell(row_key: str, dur: float) -> dict:
        for d in rows[row_key]["by_duration"]:
            if float(d["tool_duration_ms"]) == dur:
                if d.get("status") != "present":
                    raise KeyError(f"{row_key}@{dur}: {d.get('reason')}")
                return d
        raise KeyError((row_key, dur))

    def dl_cell(policy: str, comparator: str, dur: float, budget: float, arm: str = "diffusion_standard") -> dict:
        # Keyed on the arm too: the live `crystalcall` arm replays the same
        # `full_gate` policy, and its cell against the diffusion reference is
        # empty (different prompts), so a policy-only match would pick it.
        for c in sec["deadline_paired"]["cells"]:
            if (
                c.get("arm", "diffusion_standard") == arm
                and c["policy"] == policy
                and c["comparator"] == comparator
                and c["tool_duration_ms"] == dur
                and c["budget_s"] == budget
            ):
                return c
        raise KeyError((policy, comparator, dur, budget))

    out: list[tuple[str, str, str]] = []  # macro, value, source

    def put(macro: str, value: str, source: str) -> None:
        out.append((macro, value, source))

    # ---- crystallization -------------------------------------------------- #
    vf = cs["valid_final_parse"]
    put("ParseRate", pct(vf["point"], 0), "crystallization.valid_final_parse.point")
    put("ParseK", str(vf["k"]), "crystallization.valid_final_parse.k")
    put("ParseN", str(vf["n"]), "crystallization.valid_final_parse.n")
    fc = cs["final_calls"]
    put("CallsTotal", str(fc["total"]), "crystallization.final_calls.total")
    put("CallsEligible", str(fc["eligible"]), "crystallization.final_calls.eligible")
    put("EligibleFrac", pct(fc["eligible_fraction"]), "crystallization.final_calls.eligible_fraction")
    lead = cs["lead_ms"]
    put("LeadMedian", num(lead["p50"]), "crystallization.lead_ms.p50")
    put("LeadPninetyfive", num(lead["p95"]), "crystallization.lead_ms.p95")
    put("LeadN", str(int(lead["n"])), "crystallization.lead_ms.n")
    lci = cs["lead_ms_median_ci"]
    put("LeadCiLo", num(lci["ci_lo"]), "crystallization.lead_ms_median_ci.ci_lo")

    # The pooled lead statistic mixes three populations; the paper quotes the
    # real-commit cohort only, so the split is recomputed here from the
    # trajectory rows rather than transcribed.
    traj_all = jsonl("experiments/derived/trajectory_analyses.jsonl")
    # Cohort statistics below are diffusion-only; the AR control has no drafts.
    traj = [t for t in traj_all if str(t["trajectory_id"]).split("/")[0] == "diffusion_standard"]
    put("Prompts", str(len({t["prompt_id"] for t in traj})), "trajectory_analyses distinct diffusion prompt_ids")
    put("Seeds", str(len({int(t["seed"]) for t in traj})), "trajectory_analyses distinct diffusion seeds")
    put("Trajectories", str(len(traj)), "trajectory_analyses diffusion trajectories")
    calls = [c for t in traj for c in t["final_calls"] if c.get("eligible")]

    def committed(c: dict) -> bool:
        # `committed_step == 0` is a real commit at the first revision, and `0`
        # is falsy: `c.get("committed_step") or -1` would bucket it as imputed.
        step = c.get("committed_step")
        return step is not None and step >= 0

    real = [c for c in calls if committed(c) and c.get("lead_ms") is not None]
    imputed = [
        c
        for c in calls
        if not committed(c) and c.get("appearances") and c.get("lead_ms") is not None
    ]
    unseen = [c for c in calls if not c.get("appearances")]

    def med(xs: list[float]) -> float:
        """Interpolated median (mean of the two central values for even n).

        The project's `percentiles()` returns an order statistic; for the
        real-commit cohort both central values coincide so the two agree, but
        for the drafter tails they do not, and the paper interpolates.
        """
        s = sorted(xs)
        n = len(s)
        return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])

    # Anchor named explicitly: the stored `lead_ms` field, which is what the
    # artifact commits to (exec-ready -> commit). Recomputing it from
    # `committed_ns - first_exec_ns` gives a slightly different value, so the
    # paper quotes this one throughout.
    put("RealLead", num(med([c["lead_ms"] for c in real])), "trajectory_analyses stored lead_ms, real-commit cohort, interpolated median")
    put("RealLeadN", str(len(real)), "trajectory_analyses real-commit cohort size")
    put("ImputedN", str(len(imputed)), "trajectory_analyses imputed-commit cohort size")
    put("UnseenN", str(len(unseen)), "trajectory_analyses never-observed calls")
    put("CoverageCeiling", pct((len(calls) - len(unseen)) / len(calls)), "trace coverage ceiling")
    put("LeadCiHi", num(lci["ci_hi"]), "crystallization.lead_ms_median_ci.ci_hi")
    dec = cs["decode_ms"]
    put("DecodeMedian", num(dec["p50"]), "crystallization.decode_ms.p50")

    # ---- mechanism: lead quantized in revisions; extractor vs acceptance --- #
    steps = [int(c["committed_step"]) - int(c["first_exec_step"]) for c in real]
    put("LeadStepsZero", str(sum(1 for s in steps if s == 0)), "trajectory_analyses real-commit lead == 0 revisions")
    put("LeadStepsOne", str(sum(1 for s in steps if s == 1)), "trajectory_analyses real-commit lead == 1 revision")
    put("LeadStepsTwo", str(sum(1 for s in steps if s == 2)), "trajectory_analyses real-commit lead == 2 revisions")
    put("LeadStepsThreePlus", str(sum(1 for s in steps if s >= 3)), "trajectory_analyses real-commit lead >= 3 revisions")
    put("LeadStepsMedian", str(int(med([float(s) for s in steps]))), "trajectory_analyses real-commit lead in revisions, median")
    tve = load("experiments/derived/text_vs_exec_lead.json")["summary"]
    put("TextBeforeExecShare", num(tve["text_before_exec_steps"]["share_gt0"]), "text_vs_exec_lead text_before_exec_steps.share_gt0")
    put("AcceptLagN", str(int(tve["accepted_text_before_exec_steps"]["n"])), "text_vs_exec_lead accepted_text_before_exec_steps.n")
    put("AcceptLagMedian", str(int(-tve["accepted_text_before_exec_steps"]["p50"])), "text_vs_exec_lead -accepted_text_before_exec_steps.p50 (revisions from readable to accepted)")
    put("AcceptLagMin", str(int(-tve["accepted_text_before_exec_steps"]["max"])), "text_vs_exec_lead -accepted_text_before_exec_steps.max")
    put("LeadStepsTwoPlusPct", pct(sum(1 for s in steps if s >= 2) / len(steps), 0), "trajectory_analyses real-commit share with lead >= 2 revisions")

    # ---- lead uncertainty on the *same* 162-call cohort ------------------- #
    # `crystallization.lead_ms_median_ci` is computed over the pooled 196-call
    # sample (real + imputed commits); quoting it next to the real-cohort
    # median mixes cohorts. Bootstrap here, clustered by trajectory like the
    # rest of the paper, so point estimate and interval share a population.
    import random  # noqa: PLC0415 - local, deterministic seed below

    # Clustered by *prompt*, so all seeds of one prompt resample together.
    real_by_prompt: dict[str, list[float]] = {}
    for t in traj:
        vals = [
            c["lead_ms"]
            for c in t["final_calls"]
            if c.get("eligible") and committed(c) and c.get("lead_ms") is not None
        ]
        if vals:
            real_by_prompt.setdefault(str(t["prompt_id"]), []).extend(vals)
    clusters = list(real_by_prompt.values())
    rng = random.Random(20260906)
    boots: list[float] = []
    for _ in range(int(res["bootstrap"]["n"])):
        sample = [v for k in rng.choices(clusters, k=len(clusters)) for v in k]
        boots.append(med(sample))
    boots.sort()
    alpha = float(res["alpha"])
    put("RealLeadCiLo", num(boots[int(alpha / 2 * len(boots))]), "trajectory_analyses real-commit lead_ms median, cluster bootstrap ci_lo")
    put("RealLeadCiHi", num(boots[int((1 - alpha / 2) * len(boots)) - 1]), "trajectory_analyses real-commit lead_ms median, cluster bootstrap ci_hi")
    real_sorted = sorted(c["lead_ms"] for c in real)
    put("RealLeadPninetyfive", num(real_sorted[min(len(real_sorted) - 1, int(round(0.95 * (len(real_sorted) - 1))))]), "trajectory_analyses real-commit lead_ms p95 (nearest-rank)")

    # ---- pilot: thresholds and both attempts ------------------------------ #
    for attempt in (1, 2):
        pil = load(f"reports/pilot_attempt{attempt}.json")
        gates = {g["gate"]: g for g in pil["gates"]}
        tag = "One" if attempt == 1 else "Two"
        put(f"PilotPrec{tag}", pct(gates["gate_precision"]["value"]), f"pilot_attempt{attempt} gate_precision.value")
        put(f"PilotSpeedFivehundred{tag}", pct(gates["median_speedup_500ms"]["value"]), f"pilot_attempt{attempt} median_speedup_500ms.value")
        put(f"PilotSpeedTwosec{tag}", pct(gates["median_speedup_2s"]["value"]), f"pilot_attempt{attempt} median_speedup_2s.value")
        if attempt == 2:
            put("PilotPrecTarget", pct(gates["gate_precision"]["threshold"], 0), "pilot gate_precision.threshold")
            put("PilotSpeedFivehundredTarget", pct(gates["median_speedup_500ms"]["threshold"], 0), "pilot median_speedup_500ms.threshold")
            put("PilotSpeedTwosecTarget", pct(gates["median_speedup_2s"]["threshold"], 0), "pilot median_speedup_2s.threshold")
            put("PilotN", str(gates["median_speedup_500ms"].get("n") or pil["conservative_gate"]["speedups"]["500ms"]["n"]), "pilot speedup denominator (prompts)")

    # ---- the acceptance signal run *forward* as a gate -------------------- #
    # `diffusion_earliest_committed` is resolved retrospectively and cannot
    # mis-launch by construction; `acceptance_only` fires on the same signal as
    # a real forward policy and shows what acceptance costs on a renoising sampler.
    fwd = [
        r
        for r in jsonl("experiments/derived/acceptance/policy_outcomes_attempt2.jsonl")
        if r["policy"] == "acceptance_only" and r["tool_duration_ms"] == 2000.0
    ]
    fwd_true = sum(r.get("true_launches", 0) or 0 for r in fwd)
    fwd_false = sum(r.get("false_launches", 0) or 0 for r in fwd)
    put("FwdTrue", str(int(fwd_true)), "acceptance_only@2000 true_launches")
    put("FwdFalse", str(int(fwd_false)), "acceptance_only@2000 false_launches")
    put("FwdPrec", pct(fwd_true / (fwd_true + fwd_false)), "acceptance_only@2000 precision")
    put("FwdWasteS", num(sum(r.get("wasted_tool_ms", 0) or 0 for r in fwd) / 1000.0, 0), "acceptance_only@2000 wasted tool seconds")

    # ---- realised gate at 500 ms and 2 s ---------------------------------- #
    for dur, tag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
        c = cell("diffusion_standard/full_gate", dur)
        put(f"Prec{tag}", pct(c["precision"]["point"]), f"main_table full_gate@{dur:.0f} precision")
        put(f"PrecLo{tag}", pct(c["precision"]["ci_lo"]), f"main_table full_gate@{dur:.0f} precision.ci_lo")
        put(f"PrecHi{tag}", pct(c["precision"]["ci_hi"]), f"main_table full_gate@{dur:.0f} precision.ci_hi")
        put(f"PrecK{tag}", str(c["precision"]["k"]), f"main_table full_gate@{dur:.0f} precision.k")
        put(f"PrecN{tag}", str(c["precision"]["n"]), f"main_table full_gate@{dur:.0f} precision.n")
        put(f"Cov{tag}", pct(c["coverage"]["point"]), f"main_table full_gate@{dur:.0f} coverage")
        put(f"Gain{tag}", pct(c["gain"]["median"]), f"main_table full_gate@{dur:.0f} gain.median")
        put(f"GainLo{tag}", pct(c["gain"]["ci_lo"]), f"main_table full_gate@{dur:.0f} gain.ci_lo")
        put(f"GainHi{tag}", pct(c["gain"]["ci_hi"]), f"main_table full_gate@{dur:.0f} gain.ci_hi")
        put(f"Div{tag}", str(c["semantic_divergence"]), f"main_table full_gate@{dur:.0f} semantic_divergence")
        put(f"Waste{tag}", num(c["wasted_tool_ms_per_episode"], 0), f"main_table full_gate@{dur:.0f} wasted_tool_ms_per_episode")
        e = cell("diffusion_standard/diffusion_earliest_committed", dur)
        put(f"EarliestGain{tag}", pct(e["gain"]["median"]), f"main_table earliest@{dur:.0f} gain.median")
        put(f"EarliestPrec{tag}", pct(e["precision"]["point"]), f"main_table earliest@{dur:.0f} precision")

    # ---- oracle ceiling (attempt 2, executed cohort) ---------------------- #
    for dur, tag in (("500", "Fivehundred"), ("2000", "Twosec")):
        g = a2["oracle_final_match"][dur]["gain"]
        put(f"OracleExec{tag}", pct(g["executed"]["median"]), f"oracle_ceiling attempt2 oracle_final_match {dur} executed")
        put(f"OracleAll{tag}", pct(g["all"]["median"]), f"oracle_ceiling attempt2 oracle_final_match {dur} all")

    # ---- paired deadline grid --------------------------------------------- #
    outcomes = jsonl("experiments/derived/policy_outcomes.jsonl")
    dp = sec["deadline_paired"]
    ref = "diffusion_standard/diffusion_final_response"
    dcell = dl_cell("full_gate", ref, 500.0, 2.0)
    put("DlHits", str(dcell["k_candidate"]), "deadline_paired full_gate hits")
    put("DlRefHits", str(dcell["k_comparator"]), "deadline_paired comparator hits")
    put("DlN", str(dcell["n_pairs"]), "deadline_paired n_pairs")
    put("DlB", str(dcell["mcnemar"]["b"]), "deadline_paired mcnemar.b")
    put("DlC", str(dcell["mcnemar"]["c"]), "deadline_paired mcnemar.c")
    pv = dcell["mcnemar"]["p_value"]
    put("DlP", (f"{pv:.4f}" if pv >= 1e-4 else "<10^{-4}"), "deadline_paired mcnemar.p_value")
    # The pooled cell above counts 300 (prompt, seed) pairs, which are 100
    # dependent prompt clusters, so its exact test is not valid as a single
    # test.  Per seed the 100 prompts are independent, so each seed gets its
    # own exact McNemar and the paper quotes the per-seed tests plus the pooled
    # counts descriptively.
    from crystalcall.metrics import mcnemar_exact  # noqa: PLC0415

    def hit(r: dict) -> bool | None:
        if r.get("correct") is None or r.get("completion_ms") is None:
            return None
        return bool(r["correct"]) and float(r["completion_ms"]) <= 2000.0

    by_seed: dict[int, dict[str, dict[str, bool | None]]] = {}
    for r in outcomes:
        if r["arm"] != "diffusion_standard" or r["tool_duration_ms"] != 500.0:
            continue
        if r["policy"] not in ("full_gate", "diffusion_final_response"):
            continue
        by_seed.setdefault(int(r["seed"]), {}).setdefault(r["policy"], {})[r["prompt_id"]] = hit(r)
    seed_tags = {0: "Zero", 1: "One", 2: "Two"}
    worst_p = 0.0
    for seed in sorted(by_seed):
        cand, base = by_seed[seed]["full_gate"], by_seed[seed]["diffusion_final_response"]
        b = c = k1 = k2 = 0
        for pid in cand:
            h1, h2 = cand[pid], base.get(pid)
            if h1 is None or h2 is None:
                continue
            k1 += h1
            k2 += h2
            b += h1 and not h2
            c += h2 and not h1
        m = mcnemar_exact(b, c)
        tag = seed_tags[seed]
        put(f"DlSeed{tag}Hits", str(k1), f"policy_outcomes seed {seed} full_gate deadline hits @500/2s")
        put(f"DlSeed{tag}Ref", str(k2), f"policy_outcomes seed {seed} final_response deadline hits @500/2s")
        put(f"DlSeed{tag}B", str(b), f"policy_outcomes seed {seed} discordant gained")
        put(f"DlSeed{tag}C", str(c), f"policy_outcomes seed {seed} discordant lost")
        put(f"DlSeed{tag}P", f"{m['p_value']:.4f}", f"exact McNemar seed {seed}")
        worst_p = max(worst_p, float(m["p_value"]))
    put("DlSeedMaxP", f"{worst_p:.4f}", "largest per-seed exact McNemar p @500/2s")
    put("DlLive", str(dp["n_live_cells"]), "deadline_paired.n_live_cells")
    put("DlCells", str(dp["n_cells"]), "deadline_paired.n_cells")
    shared = [
        c
        for c in dp["cells"]
        if c.get("arm", "diffusion_standard") == "diffusion_standard"
        and c["comparator"] == ref
        and c["tool_duration_ms"] == 500.0
        and c["budget_s"] == 2.0
        and c["mcnemar"]["b"] == dcell["mcnemar"]["b"]
        and c["mcnemar"]["c"] == dcell["mcnemar"]["c"]
    ]
    put("DlShared", str(len(shared)), "deadline_paired policies sharing the cell")
    put("DlSharedOthers", str(len(shared) - 1), "deadline_paired sharing policies minus full_gate")
    sig = [c for c in dp["cells"] if c["regime"] == "live" and c["mcnemar"]["p_value"] < 0.05]
    put("DlSig", str(len(sig)), "deadline_paired live cells with p<0.05")
    vs_e = [
        c
        for c in dp["cells"]
        if c.get("arm", "diffusion_standard") == "diffusion_standard"
        and c["policy"] == "full_gate"
        and c["comparator"] == "diffusion_standard/diffusion_earliest_committed"
    ]
    put("DlVsEarliestCells", str(len(vs_e)), "deadline_paired full_gate vs earliest cells")
    put("DlVsEarliestDisc", str(sum(c["mcnemar"]["n_discordant"] for c in vs_e)), "deadline_paired discordant total")
    put("DlVsEarliestLive", str(sum(1 for c in vs_e if c["mcnemar"]["n_discordant"])), "deadline_paired cells vs earliest with any discordant pair")

    # ---- safety core ------------------------------------------------------ #
    # `accounting.verification` counts *online* claim re-execution, and the
    # online arm was deliberately never run, so those fields are 0 and must not
    # be quoted as "0 divergences over 0 claims" (that sentence rendered in a
    # draft PDF). The offline replay grid and the stateful stub arm are the two
    # populations that actually carry released claims.
    put("GridRows", f"{len(outcomes):,}".replace(",", "{,}"), "policy_outcomes rows")
    put("GridPolicies", str(len({r["policy"] for r in outcomes})), "policy_outcomes distinct policies")
    put("GridDiv", str(sum(r.get("semantic_divergence", 0) or 0 for r in outcomes)), "policy_outcomes semantic_divergence total")
    fg500 = [
        r
        for r in outcomes
        if r["policy"] == "full_gate" and r["tool_duration_ms"] == 500.0
    ]
    put("GateClaims", str(sum(r.get("claimed", 0) or 0 for r in fg500)), "policy_outcomes full_gate@500 claimed")
    put("GateClaimDiv", str(sum(r.get("claims_verified_divergent", 0) or 0 for r in fg500)), "policy_outcomes full_gate@500 claims_verified_divergent")
    ts = sec["toolsandbox"]
    put("TsEpisodesTotal", str(ts["n_input_rows"]), "toolsandbox.n_input_rows")
    put("TsEpisodes", str(ts["totals"]["episodes"]), "toolsandbox.totals.episodes")
    put("TsSpeculative", str(ts["totals"]["speculative_episodes"]), "toolsandbox.totals.speculative_episodes")
    put("TsDivergent", str(ts["totals"]["divergent"]), "toolsandbox.totals.divergent")
    put("TsExcluded", str(ts["totals"]["excluded_episodes"]), "toolsandbox.totals.excluded_episodes")
    eps = jsonl("experiments/derived/toolsandbox_episodes.jsonl")
    spec = [e for e in eps if e.get("arm") == "toolsandbox_crystalcall"]
    put("TsLaunches", str(sum(e.get("speculative_launches", 0) for e in spec)), "toolsandbox_episodes speculative_launches")
    put("TsClaimed", str(sum(e.get("speculative_claimed", 0) for e in spec)), "toolsandbox_episodes speculative_claimed")
    put("TsAuthCalls", str(sum(e.get("authoritative_calls", 0) for e in spec)), "toolsandbox_episodes authoritative_calls")
    arms = ts["by_arm"]
    succ = {a: v["success"] for a, v in arms.items()}
    put("TsSuccess", str(sorted(set(succ.values()))[0]), "toolsandbox.by_arm success (identical across arms)")
    put("TsArmEpisodes", str(sorted({v["episodes"] for v in arms.values()})[0]), "toolsandbox.by_arm episodes")

    # ---- cross-family drafter rungs --------------------------------------- #
    for name, tag in (("lead_block4", "Bfour"), ("lead_block8", "Beight")):
        blob = load(f"experiments/sweeps/dflash2/{name}.json")
        m = blob["summary"]["measured"]
        rws = blob["rows"]
        put(f"Dflash{tag}Tokens", num(blob["summary"]["mean_tokens_per_event"], 2), f"{name} mean_tokens_per_event")
        put(f"Dflash{tag}Ms", num(m["ms_per_acceptance_event_mean_over_rounds"]["p50"]), f"{name} ms_per_event p50")
        put(f"Dflash{tag}MsLo", num(min(r["ms_per_event"] for r in rws)), f"{name} ms_per_event min")
        put(f"Dflash{tag}MsHi", num(max(r["ms_per_event"] for r in rws)), f"{name} ms_per_event max")
        tails = sorted(r["tail_tokens"] for r in rws)
        # Interpolated, like every other median in the paper: the block-4
        # tails are 7,8,8,9,9,9,10,11,11,12,12,15, so an order statistic would
        # print 10 instead of the interpolated 9.5.
        put(f"Dflash{tag}TailTok", num(med([float(x) for x in tails])), f"{name} tail_tokens interpolated median")
        ev = [round(r["call_to_done_ms"] / r["ms_per_event"]) for r in rws]
        put(f"Dflash{tag}TailEvLo", str(min(ev)), f"{name} tail in events min")
        put(f"Dflash{tag}TailEvHi", str(max(ev)), f"{name} tail in events max")
        put(f"Dflash{tag}Hist", str(len(rws[0]["spec_accept_histogram"])), f"{name} spec_accept_histogram width")
        put(f"Dflash{tag}N", str(len(rws)), f"{name} rows")
        ms_med = m["ms_per_acceptance_event_mean_over_rounds"]["p50"]
        put(f"Dflash{tag}TokPerSec", num(1000.0 * blob["summary"]["mean_tokens_per_event"] / ms_med), f"{name} tokens per second (mean tokens/event over p50 ms/event)")
        put(f"Dflash{tag}TailMs", num(med([float(r["call_to_done_ms"]) for r in rws]), 0), f"{name} call_to_done_ms interpolated median")

    # ---- post-hoc learned launch rule (scripts/learned_gate.py) ----------- #
    lg = load("experiments/derived/learned_gate.json")
    put("LearnedRows", str(lg["rows"]), "learned_gate rows")
    put("LearnedPos", str(lg["positive_rows"]), "learned_gate positive_rows")
    put("LearnedAuc", num(lg["heldout_auc"], 2), "learned_gate heldout_auc")
    put("LearnedFeatures", str(len(lg["features"])), "learned_gate feature count")
    for key, tag in (("learned_selected", "Learned"), ("crystalcall_conservative", "LgCc"), ("diffusion_earliest_committed", "LgEc"), ("oracle_final_match", "LgOracle")):
        for dur, dtag in (("500", "Fivehundred"), ("2000", "Twosec")):
            g = lg["policies"][key][dur]["gain"]
            put(f"{tag}MeanAll{dtag}", pct(g["all"]["mean"]), f"learned_gate {key}@{dur} gain.all.mean")
            put(f"{tag}MedAll{dtag}", pct(g["all"]["median"]), f"learned_gate {key}@{dur} gain.all.median")
            put(f"{tag}MedExec{dtag}", pct(g["executed"]["median"]), f"learned_gate {key}@{dur} gain.executed.median")
            put(f"{tag}Pos{dtag}", str(g["all"]["positive"]), f"learned_gate {key}@{dur} trajectories with gain > 0")
            put(f"{tag}ExecN{dtag}", str(g["executed"]["n"]), f"learned_gate {key}@{dur} executed trajectories")
            pr = lg["policies"][key][dur]["precision"]
            put(f"{tag}Prec{dtag}", pct(pr) if pr is not None else "---", f"learned_gate {key}@{dur} precision")
    sel = lg["selected_threshold_histogram"]
    put("LearnedThetaMode", max(sel, key=lambda k: sel[k]), "learned_gate most-selected threshold")

    # ---- matched autoregressive control (post-hoc; dev split) ------------- #
    for pol, tag in (("ar_streaming", "ArStream"), ("ar_sptc", "ArSptc")):
        for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
            c = cell(f"gemma4_ar/{pol}", dur)
            put(f"{tag}Ms{dtag}", num(c["completion_ms"]["p50"], 0), f"main_table {pol}@{dur:.0f} completion p50")
    for pol, tag in (("full_gate", "Cc"), ("diffusion_final_response", "Ref")):
        for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
            c = cell(f"diffusion_standard/{pol}", dur)
            put(f"{tag}Ms{dtag}", num(c["completion_ms"]["p50"], 0), f"main_table {pol}@{dur:.0f} completion p50")
    # `ar_sptc` overlaps independent calls, the same execution discipline every
    # diffusion policy uses, so it carries the matched headline ratio;
    # `ar_streaming` is serial by construction (replay.py "Execution
    # discipline") and its ratio is reported separately as a serial baseline.
    for pol, tag in (("ar_sptc", "ArSlowdown"), ("ar_streaming", "ArSerialSlowdown")):
        for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
            put(
                f"{tag}{dtag}",
                num(cell(f"gemma4_ar/{pol}", dur)["completion_ms"]["p50"] / cell("diffusion_standard/full_gate", dur)["completion_ms"]["p50"], 1),
                f"{pol}/full_gate completion p50 ratio @{dur:.0f}",
            )

    # ---- output-token throughput (retokenised final text; scripts/throughput.py)
    # `tokens` here is `retokenized_final_output_tokens`: `final_text` encoded
    # under the arm's own tokenizer without special tokens. It is NOT the
    # engine's emitted-token counter (the side channel records none), so the
    # paper must call it a re-tokenised output rate.
    tp = load("experiments/derived/throughput.json")["arms"]
    for arm, tag in (("diffusion_standard", "Diff"), ("gemma4_ar", "Ar")):
        s0 = tp[arm]["per_seed"]["0"]
        put(f"{tag}TokPerSec", num(s0["aggregate_tokens_per_s"], 1), f"throughput {arm} seed 0 aggregate_tokens_per_s")
        put(f"{tag}TokPerSecMedian", num(s0["tokens_per_s"]["p50"], 1), f"throughput {arm} seed 0 tokens_per_s.p50")
        put(f"{tag}OutTokMedian", num(s0["tokens"]["p50"], 0), f"throughput {arm} seed 0 tokens.p50 (retokenised final_text)")
    put(
        "DiffTokPerSecPooled",
        num(tp["diffusion_standard"]["pooled"]["aggregate_tokens_per_s"], 1),
        "throughput diffusion_standard pooled over seeds aggregate_tokens_per_s",
    )

    # Seed-0-only completion medians, so the AR comparison (AR has one seed) is
    # like-for-like: same prompts, same seed index, paired episodes.
    s0_rows = [r for r in outcomes if r.get("seed") == 0 and r["arm"] == "diffusion_standard"]
    for pol, tag in (("full_gate", "CcSeedZero"), ("diffusion_final_response", "RefSeedZero")):
        for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
            xs = sorted(r["completion_ms"] for r in s0_rows if r["policy"] == pol and r["tool_duration_ms"] == dur and r.get("completion_ms"))
            put(f"{tag}Ms{dtag}", num(med(xs), 0), f"policy_outcomes {pol}@{dur:.0f} seed-0 completion median")
    for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
        xs = sorted(r["completion_ms"] for r in s0_rows if r["policy"] == "full_gate" and r["tool_duration_ms"] == dur and r.get("completion_ms"))
        put(f"ArSlowdownSeedZero{dtag}", num(cell("gemma4_ar/ar_sptc", dur)["completion_ms"]["p50"] / med(xs), 1), f"ar_sptc / seed-0 full_gate completion p50 ratio @{dur:.0f}")

    # Per-arm generation quality, never pooled (docs/baselines.md).
    arm_traj: dict[str, list[dict]] = {}
    for t in traj_all:
        arm_traj.setdefault(str(t["trajectory_id"]).split("/")[0], []).append(t)
    for arm, tag in (("diffusion_standard", "Diff"), ("gemma4_ar", "Ar")):
        rs = arm_traj[arm]
        put(f"{tag}DecodeMs", num(med([float(r["decode_ms"]) for r in rs]), 0), f"trajectory_analyses {arm} decode_ms median")
        put(f"{tag}Parse", str(sum(1 for r in rs if r.get("valid_final_parse"))), f"trajectory_analyses {arm} valid_final_parse count")
    for arm, tag in (("diffusion_standard", "Diff"), ("gemma4_ar", "Ar")):
        finals = [
            json.loads(line)
            for shard in sorted((ROOT / f"experiments/raw/{arm}").glob(f"{arm}-seed*.jsonl"))
            for line in shard.read_text().splitlines()
            if line.strip() and json.loads(line).get("kind") == "final"
        ]
        if not finals:
            print(f"paper_numbers: no raw shards under experiments/raw/{arm}; skipping its raw-shard macros", file=sys.stderr)
            continue
        put(f"{tag}Finals", str(len(finals)), f"raw {arm} final records over all seeds")
        put(f"{tag}Correct", str(sum(1 for f in finals if f.get("benchmark_correct") is True)), f"raw {arm} benchmark_correct count over all seeds")
        put(f"{tag}CorrectPct", pct(sum(1 for f in finals if f.get("benchmark_correct") is True) / len(finals), 0), f"raw {arm} benchmark_correct share")
        put(f"{tag}Executed", str(sum(1 for f in finals if f.get("authoritative_calls"))), f"raw {arm} trajectories executing >=1 call over all seeds")
        put(f"{tag}ExecutedPct", pct(sum(1 for f in finals if f.get("authoritative_calls")) / len(finals), 0), f"raw {arm} executing share")
        put(f"{tag}ParsePct", pct(sum(1 for f in finals if f.get("valid_final_parse")) / len(finals), 0), f"raw {arm} valid_final_parse share")
        # Seed 0 alone, for the paired comparison against the single-seed AR control.
        s0 = [f for f in finals if str(f.get("request_id", "")).endswith("/s0")]
        put(f"{tag}CorrectSeedZero", str(sum(1 for f in s0 if f.get("benchmark_correct") is True)), f"raw {arm} seed-0 benchmark_correct count")

    # Paired deadline success against the AR control.
    for dur, budget, tag in ((500.0, 2.0, "ArDlTight"), (500.0, 5.0, "ArDlLoose")):
        c = dl_cell("full_gate", "gemma4_ar/ar_sptc", dur, budget)
        # The serial AR row yields the identical cell at both budgets (its
        # completion exceeds every budget that binds here either way); the
        # assert keeps the paper from attributing a concurrency artifact to
        # the decoder.
        ser = dl_cell("full_gate", "gemma4_ar/ar_streaming", dur, budget)
        assert (c["k_candidate"], c["k_comparator"], c["mcnemar"]["b"], c["mcnemar"]["c"]) == (
            ser["k_candidate"], ser["k_comparator"], ser["mcnemar"]["b"], ser["mcnemar"]["c"]
        ), "AR deadline cell differs between serial and concurrent execution"
        put(f"{tag}Hits", str(c["k_candidate"]), f"deadline_paired full_gate vs ar_sptc (identical to ar_streaming) @{dur:.0f}/{budget:.0f}s hits")
        put(f"{tag}Ref", str(c["k_comparator"]), f"deadline_paired ar_sptc @{dur:.0f}/{budget:.0f}s hits")
        put(f"{tag}B", str(c["mcnemar"]["b"]), f"deadline_paired vs ar_sptc @{dur:.0f}/{budget:.0f}s mcnemar.b")
        put(f"{tag}C", str(c["mcnemar"]["c"]), f"deadline_paired vs ar_sptc @{dur:.0f}/{budget:.0f}s mcnemar.c")
        put(f"{tag}P", f"{c['mcnemar']['p_value']:.4f}", f"deadline_paired vs ar_sptc @{dur:.0f}/{budget:.0f}s exact p")

    # ---- live online validation of the speculative runtime ---------------- #
    # experiments/raw/crystalcall_tool{500,2000}: the frozen rule run live on
    # 60 held-out prompts with the given injected tool delay.  Every result the
    # runtime handed to the agent was re-executed after the turn and compared.
    live_total = {"claimed": 0, "verified_match": 0, "verified_divergent": 0, "verification_error": 0, "launches": 0, "finals": 0}
    for d in ("crystalcall_tool500", "crystalcall_tool2000"):
        for shard in sorted((ROOT / "experiments/raw" / d).glob("crystalcall-seed*.jsonl")):
            for line in shard.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("kind") != "final":
                    continue
                live_total["finals"] += 1
                c = rec.get("claims") or {}
                for k in ("claimed", "verified_match", "verified_divergent", "verification_error"):
                    live_total[k] += int(c.get(k, 0) or 0)
                live_total["launches"] += int(((rec.get("runtime") or {}).get("stats") or {}).get("launches", 0) or 0)
    if not live_total["finals"]:
        print("paper_numbers: no live-run shards under experiments/raw/crystalcall_tool*; skipping the live-run macros", file=sys.stderr)
    else:
        put("LivePrompts", str(live_total["finals"] // 2 if live_total["finals"] else 0), "live shards: prompts per pass")
        put("LivePasses", "2", "live shards: injected tool delays 500 ms and 2 s")
        put("LiveLaunches", str(live_total["launches"]), "live shards: runtime.stats.launches summed")
        put("LiveClaimed", str(live_total["claimed"]), "live shards: claims.claimed summed")
        put("LiveVerifiedMatch", str(live_total["verified_match"]), "live shards: claims.verified_match summed")
        put("LiveVerifiedDivergent", str(live_total["verified_divergent"]), "live shards: claims.verified_divergent summed")
        # Per-launch fidelity: every launch the live runtime made (identity and
        # launch revision, from the recorded future table) against the launch the
        # offline replay of the same drafts makes under the same rule and tool
        # delay.  Equal *counts* would not establish this; identities and steps do.
        from crystalcall.harness import _load_cost_model, load_tokenizer, resolve_model_dir  # noqa: PLC0415
        from crystalcall.manifest import load_manifest, registry_for  # noqa: PLC0415
        from crystalcall.parse_ptc import PtcExtractor  # noqa: PLC0415
        from crystalcall.pilot import _pilot_remaining  # noqa: PLC0415
        from crystalcall.replay import _policy_launches, analyze_trajectory, build_views, load_trajectories  # noqa: PLC0415
        from crystalcall.scheduler import POLICY_LIBRARY  # noqa: PLC0415
        from crystalcall.tokens import CanvasDecoder  # noqa: PLC0415

        test_recs = {r.prompt_id: r for r in load_manifest(ROOT / "experiments/manifests/test.jsonl")}
        fid_decoder = CanvasDecoder(tokenizer=load_tokenizer(resolve_model_dir("google/diffusiongemma-26B-A4B-it")), cache_size=64)
        fid_costs = _load_cost_model(ROOT)
        fid_online = fid_same_identity = fid_same_step = fid_replay_only = 0
        fid_traj = 0
        for d, dur in (("crystalcall_tool500", 500.0), ("crystalcall_tool2000", 2000.0)):
            shard_paths = sorted((ROOT / "experiments/raw" / d).glob("crystalcall-seed*.jsonl"))
            if not shard_paths:
                continue
            online: dict[str, dict[str, int]] = {}
            for shard in shard_paths:
                for line in shard.read_text().splitlines():
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if rec.get("kind") == "final":
                        online[rec["request_id"]] = {
                            f["call"]["identity"]: int(f["launch_revision"])
                            for f in ((rec.get("runtime") or {}).get("futures") or [])
                        }
            fid_trajs = load_trajectories(shard_paths, verify=True)
            fid_remaining = _pilot_remaining(ROOT, fid_trajs)
            for t in fid_trajs:
                reg = registry_for(test_recs[t.prompt_id])
                views = build_views(t, PtcExtractor(registry=reg), fid_decoder)
                analysis = analyze_trajectory(t, views, reg)
                replayed = {
                    l.identity: l.launch_step
                    for l in _policy_launches(t, views, analysis, reg, POLICY_LIBRARY["full_gate"], remaining_model=fid_remaining, costs=fid_costs, tool_duration_ms=dur)
                    if l.reason != "not_speculated"
                }
                fid_traj += 1
                on = online.get(t.request_id, {})
                for ident, step in on.items():
                    fid_online += 1
                    if ident in replayed:
                        fid_same_identity += 1
                        fid_same_step += replayed[ident] == step
                fid_replay_only += sum(1 for ident in replayed if ident not in on)
        put("LiveReplayTraj", str(fid_traj), "live trajectories compared against replay")
        put("LiveFidelityOnline", str(fid_online), "live launches (future table)")
        put("LiveFidelityIdentity", str(fid_same_identity), "live launches whose call identity the replay also launches")
        put("LiveFidelityStep", str(fid_same_step), "live launches replayed at the same launch revision")
        put("LiveFidelityReplayOnly", str(fid_replay_only), "replayed launches with no live counterpart")

    # ---- held-out test split (300 prompts, seed 0, separate derived tree) --- #
    # Replayed from experiments/raw/heldout/** into experiments/derived_heldout
    # and aggregated into reports/results_heldout.json; never pooled with the
    # development rows (cli.REPLAY_EXCLUDED_SUBTREES).  The AR rows appear once
    # experiments/raw/heldout/gemma4_ar exists; macros for them are emitted
    # only when their cells are present so the paper build never fails on a
    # missing run.
    ho_path = ROOT / "reports/results_heldout.json"
    if ho_path.is_file():
        ho = load("reports/results_heldout.json")
        hsec = ho["sections"]
        hrows = {f"{r['arm']}/{r['policy']}": r for r in hsec["main_table"]["rows"]}

        def hcell(row_key: str, dur: float) -> dict | None:
            for d in hrows.get(row_key, {}).get("by_duration", []):
                if float(d["tool_duration_ms"]) == dur and d.get("status") == "present":
                    return d
            return None

        hcs = hsec["crystallization"]
        put("HeldoutN", str(hcs["valid_final_parse"]["n"]), "results_heldout crystallization.valid_final_parse.n")
        put("HeldoutParseK", str(hcs["valid_final_parse"]["k"]), "results_heldout crystallization.valid_final_parse.k")
        put("HeldoutCallsEligible", str(hcs["final_calls"]["eligible"]), "results_heldout crystallization.final_calls.eligible")
        put("HeldoutLead", num(hcs["lead_ms"]["p50"]), "results_heldout crystallization.lead_ms.p50")
        put("HeldoutLeadN", str(int(hcs["lead_ms"]["n"])), "results_heldout crystallization.lead_ms.n")
        for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
            c = hcell("diffusion_standard/full_gate", dur)
            if c:
                put(f"HeldoutPrec{dtag}", pct(c["precision"]["point"]), f"results_heldout full_gate@{dur:.0f} precision")
                put(f"HeldoutCov{dtag}", pct(c["coverage"]["point"]), f"results_heldout full_gate@{dur:.0f} coverage")
                put(f"HeldoutCcMs{dtag}", num(c["completion_ms"]["p50"], 0), f"results_heldout full_gate@{dur:.0f} completion p50")
            rc = hcell("diffusion_standard/diffusion_final_response", dur)
            if rc:
                put(f"HeldoutRefMs{dtag}", num(rc["completion_ms"]["p50"], 0), f"results_heldout final_response@{dur:.0f} completion p50")
            for pol, tag in (("ar_sptc", "HeldoutArSptcMs"), ("ar_streaming", "HeldoutArStreamMs")):
                ac = hcell(f"gemma4_ar/{pol}", dur)
                if ac:
                    put(f"{tag}{dtag}", num(ac["completion_ms"]["p50"], 0), f"results_heldout {pol}@{dur:.0f} completion p50")
                    if c:
                        put(f"{tag.replace('Ms', 'Ratio')}{dtag}", num(ac["completion_ms"]["p50"] / c["completion_ms"]["p50"], 1), f"results_heldout {pol}/full_gate p50 ratio @{dur:.0f}")
        # Executed-task and all-prompt reductions, same definitions as the body.
        hout = jsonl("experiments/derived_heldout/policy_outcomes.jsonl")
        href = {(r["prompt_id"], r["tool_duration_ms"]): r for r in hout if r["policy"] == "diffusion_final_response" and r["arm"] == "diffusion_standard"}
        for pol, tag in (("full_gate", "HeldoutCc"), ("diffusion_earliest_committed", "HeldoutEc"), ("oracle_final_match", "HeldoutOracle")):
            for dur, dtag in ((500.0, "Fivehundred"), (2000.0, "Twosec")):
                g_all: list[float] = []
                g_exec: list[float] = []
                for r in hout:
                    if r["policy"] != pol or r["arm"] != "diffusion_standard" or r["tool_duration_ms"] != dur:
                        continue
                    b = href[(r["prompt_id"], dur)]
                    if b.get("completion_ms") and r.get("completion_ms"):
                        x = 1.0 - r["completion_ms"] / b["completion_ms"]
                        g_all.append(x)
                        if (b.get("eligible_final_calls") or 0) > 0:
                            g_exec.append(x)
                put(f"{tag}MeanAll{dtag}", pct(sum(g_all) / len(g_all)), f"heldout {pol}@{dur:.0f} mean reduction, all prompts")
                put(f"{tag}MedExec{dtag}", pct(med(g_exec)), f"heldout {pol}@{dur:.0f} median reduction, prompts that call a tool")
                put(f"{tag}ExecN{dtag}", str(len(g_exec)), f"heldout {pol}@{dur:.0f} prompts that call a tool")
        # Deadline cell on 300 independent prompts (one seed): the exact test is valid here.
        for c in hsec["deadline_paired"]["cells"]:
            if c["policy"] == "full_gate" and c["comparator"].endswith("diffusion_final_response") and c["tool_duration_ms"] == 500.0 and c["budget_s"] == 2.0:
                put("HeldoutDlHits", str(c["k_candidate"]), "results_heldout deadline full_gate hits @500/2s")
                put("HeldoutDlRef", str(c["k_comparator"]), "results_heldout deadline final_response hits @500/2s")
                put("HeldoutDlB", str(c["mcnemar"]["b"]), "results_heldout deadline gained")
                put("HeldoutDlC", str(c["mcnemar"]["c"]), "results_heldout deadline lost")
                pv = c["mcnemar"]["p_value"]
                put("HeldoutDlP", (f"={pv:.4f}" if pv >= 1e-4 else "<10^{-4}"), "results_heldout deadline exact p")
            if c["policy"] == "full_gate" and c["comparator"].endswith("ar_sptc") and c["tool_duration_ms"] == 500.0 and c["budget_s"] == 2.0:
                put("HeldoutArDlHits", str(c["k_candidate"]), "results_heldout deadline full_gate vs ar_sptc hits")
                put("HeldoutArDlRef", str(c["k_comparator"]), "results_heldout deadline ar_sptc hits")
                put("HeldoutArDlB", str(c["mcnemar"]["b"]), "results_heldout vs ar_sptc gained")
                put("HeldoutArDlC", str(c["mcnemar"]["c"]), "results_heldout vs ar_sptc lost")
        # Per-arm quality on the held-out split, from the raw shards.
        for arm, tag in (("diffusion_standard", "HeldoutDiff"), ("gemma4_ar", "HeldoutAr")):
            shard_dir = ROOT / f"experiments/raw/heldout/{arm}"
            finals = [
                json.loads(line)
                for shard in sorted(shard_dir.glob(f"{arm}-seed*.jsonl"))
                for line in shard.read_text().splitlines()
                if line.strip() and json.loads(line).get("kind") == "final"
            ]
            if finals:
                put(f"{tag}CorrectPct", pct(sum(1 for f in finals if f.get("benchmark_correct") is True) / len(finals), 0), f"raw heldout {arm} benchmark_correct share")
                put(f"{tag}ParsePct", pct(sum(1 for f in finals if f.get("valid_final_parse")) / len(finals), 0), f"raw heldout {arm} valid_final_parse share")
                put(f"{tag}ExecutedPct", pct(sum(1 for f in finals if f.get("authoritative_calls")) / len(finals), 0), f"raw heldout {arm} executing share")

    # ---- statistical method and frozen gate ------------------------------- #
    put("Alpha", f"{res['alpha']:g}", "results.json alpha")
    put("Margin", pct(res["noninferiority_margin"], 0), "results.json noninferiority_margin")
    put("BootN", f"{res['bootstrap']['n']:,}".replace(",", "{,}"), "results.json bootstrap.n")
    put("BootCluster", str(res["bootstrap"]["cluster"]).replace("_", r"\_"), "results.json bootstrap.cluster")
    put("LadderN", str(sec["instrumentation"]["ladder"]["rungs"][0]["n"]), "instrumentation.ladder rung n")
    put("LadderRungs", str(len(sec["instrumentation"]["ladder"]["rungs"])), "instrumentation.ladder rung count")
    # The frozen gate lives in code, not in an artifact (configs/frozen_policy.json
    # is deliberately absent because no held-out run was ever protected), so it
    # is read out of the dataclass defaults rather than typed into the paper.
    from crystalcall.scheduler import GatePolicy  # noqa: PLC0415 - late, optional

    gate = GatePolicy(name="full_gate")
    put("GateK", str(gate.k), "scheduler.GatePolicy.k")
    put("GateTau", f"{gate.tau:g}", "scheduler.GatePolicy.tau")
    put("GateOverlap", num(gate.min_overlap_ms, 0), "scheduler.GatePolicy.min_overlap_ms")

    # ---- environment ------------------------------------------------------ #
    put("OmpPin", str(doctor["pins"]["omp"]["version"]).replace("_", r"\_"), "doctor.pins.omp.version")
    # Hand-carried: model turns spent through the pilot phase span the two
    # pilot attempts plus the smoke and sweep runs, which no single artifact
    # totals. Flagged so the "generated" guarantee stays honest.
    put("Turns", "644", "HAND-CARRIED: model turns spent through the pilot phase")

    dest = ROOT / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Generated by scripts/paper_numbers.py -- do not edit by hand.",
        "% Every macro is read from a committed artifact; the comment names the field.",
        f"% source: reports/results.json generated_at {res['generated_at']}",
        "",
    ]
    width = max(len(m) for m, _, _ in out)
    for macro, value, source in out:
        lines.append(f"\\newcommand{{\\{macro}}}{{{value}}}{' ' * (width - len(macro))} % {source}")
    dest.write_text("\n".join(lines) + "\n")
    print(f"wrote {dest} ({len(out)} macros)")

    # ---- appendix tables, generated the same way as the macros ------------ #
    def tex(s: str) -> str:
        return s.replace("_", r"\_")

    # Artifact policy identifiers, rendered with the names the body uses so a
    # reader never has to map `full_gate` onto "CrystalCall" by themselves.
    POLICY_LABELS = {
        "full_gate": "CrystalCall",
        "diffusion_final_response": "after the answer (reference)",
        "diffusion_final_canvas": "at final draft",
        "diffusion_earliest_committed": "at acceptance (hindsight)",
        "oracle_earliest_invariant": "oracle: call stops changing",
        "oracle_final_match": "oracle: knows the answer",
        "acceptance_only": "at acceptance (online)",
        "acceptance_confidence": "at acceptance + confidence (online)",
        "syntax_effect_only": "ablation: syntax + effect only",
        "persistence_only": "ablation: persistence only",
        "confidence_only": "ablation: confidence only",
        "persistence_confidence": "ablation: persistence + confidence",
        "fixed_step_2": "ablation: fixed step 2",
        "random_matched": "ablation: random 50\\%",
        "ar_streaming": "AR, serial",
        "ar_sptc": "AR, matched concurrency",
        "preact_ready": "P-ReAct-style",
    }

    def label(policy: str) -> str:
        return POLICY_LABELS.get(policy, tex(policy))

    def ci(d: dict, scale_pct: bool = True) -> str:
        if not d or d.get("n") in (0, None) or d.get("point") is None:
            return "---"
        f = pct if scale_pct else num
        return f"{f(d['point'])} [{f(d['ci_lo'])}, {f(d['ci_hi'])}]"

    parts: dict[str, list[str]] = {}

    def emit(name: str, body: list[str]) -> None:
        header = ["% Generated by scripts/paper_numbers.py -- do not edit by hand.", ""]
        (dest.parent / f"{name}.tex").write_text("\n".join(header + body) + "\n")
        parts[name] = body

    def longtable(cols: str, header: str, caption: str) -> list[str]:
        """A table that may run past one page.

        `tabular` cannot break, so a grid with hundreds of rows silently
        overflows the page instead of continuing (an overfull \\vbox, with the
        remaining rows simply not rendered). Every generated appendix grid is
        therefore a `longtable` with a repeated header.
        """
        return [
            rf"\begin{{longtable}}{{{cols}}}",
            rf"\caption{{{caption}}}\\",
            r"\toprule",
            rf"{header} \\",
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{" + str(len(cols.replace("|", ""))) + r"}{l}{\emph{\dots\ continued}}\\",
            r"\toprule",
            rf"{header} \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{" + str(len(cols.replace("|", ""))) + r"}{r}{\emph{continued on the next page}}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]

    tbl: list[str] = []

    # A: every policy x duration of the replay grid.
    tbl += longtable(
        "llrrrrr",
        r"Policy & Arm & Tool ms & p50 ms & Precision [95\% CI] & Coverage [95\% CI] & Gain",
        "Every present (policy, arm, tool duration) cell of the replay grid.",
    )
    for r in sec["main_table"]["rows"]:
        for d in r["by_duration"]:
            if d.get("status") != "present":
                continue
            gain = d.get("gain") or {}
            gtxt = "---" if gain.get("median") is None else f"{pct(gain['median'])}\\%"
            tbl.append(
                f"{label(r['policy'])} & {tex(r['arm'])} & {d['tool_duration_ms']:.0f} & "
                f"{num(d['completion_ms']['p50'])} & {ci(d.get('precision'))} & "
                f"{ci(d.get('coverage'))} & {gtxt} \\\\"
            )
    tbl += [r"\end{longtable}"]
    emit("tab_grid", tbl)

    # B: instrumentation overhead ladder.
    tbl = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Rung & $n$ & p50 ms & p95 ms & vs.\ off & Revisions/req \\",
        r"\midrule",
    ]
    for rung in sec["instrumentation"]["ladder"]["rungs"]:
        rel = rung.get("relative_p50_vs_off")
        tbl.append(
            f"{tex(str(rung['level']))} & {rung['n']} & {num(rung['p50_ms'])} & "
            f"{num(rung['p95_ms'])} & "
            + ("---" if rel is None else f"{pct(rel)}\\%")
            + f" & {num(rung.get('revisions_per_request') or 0.0)} \\\\"
        )
    tbl += [r"\bottomrule", r"\end{tabular}"]
    emit("tab_ladder", tbl)

    # C: every live cell of the paired deadline grid.
    tbl = longtable(
        "llrrrr",
        r"Policy & Comparator & Tool ms & $B$ s & Discordant & Exact $p$",
        "Every live cell of the paired deadline grid (at least one discordant pair).",
    )
    for c in sec["deadline_paired"]["cells"]:
        if c["regime"] != "live":
            continue
        comp = c["comparator"].split("/")[-1]
        tbl.append(
            f"{label(c['policy'])} & {label(comp)} & {c['tool_duration_ms']:.0f} & "
            f"{c['budget_s']:.0f} & $+{c['mcnemar']['b']}/-{c['mcnemar']['c']}$ & "
            f"{c['mcnemar']['p_value']:.4g} \\\\"
        )
    tbl += [r"\end{longtable}"]
    emit("tab_deadline", tbl)

    # D: per-prompt drafter rungs.
    tbl = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Rung & Prompt & Tokens/event & Rounds & ms/event & Tail ms & Tail tokens \\",
        r"\midrule",
    ]
    for name, label in (("lead_block8", "block 8"), ("lead_block4", "block 4")):  # noqa: B007
        for r in load(f"experiments/sweeps/dflash2/{name}.json")["rows"]:
            tbl.append(
                f"{label} & {tex(r['prompt_id'])} & {num(r['tokens_per_event'], 2)} & "
                f"{r['verify_rounds']} & {num(r['ms_per_event'])} & "
                f"{num(r['call_to_done_ms'])} & {r['tail_tokens']} \\\\"
            )
    tbl += [r"\bottomrule", r"\end{tabular}"]
    emit("tab_rungs", tbl)

    rows_written = sum(
        1 for body in parts.values() for l in body if l.endswith(chr(92) * 2)
    )
    print(f"wrote {len(parts)} appendix tables ({rows_written} data rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
