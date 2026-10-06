"""Unit and property tests for :mod:`crystalcall.metrics`.

Reference values are re-derived inside the test from an *independent* closed
form (F-distribution quantiles / exact binomial tail identities) rather than
pasted in, so the tests check the estimator rather than a copy of its output.
The two published Clopper-Pearson numbers for (k=3, n=10) are additionally
asserted at 4 dp against the textbook interval (0.0667, 0.6525).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pytest
from scipy import stats

from crystalcall.aggregate import aggregate, load_main_table
from crystalcall.metrics import (
    binom_ci_exact,
    bootstrap_ci_clustered,
    deadline_paired,
    deadline_success,
    mcnemar_exact,
    noninferiority_success,
    paired_gain,
    paired_randomization,
    percentiles,
    rule_of_three_upper,
    submission_gates,
    summarize_outcomes,
    wilcoxon_paired,
)

ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #


def _cp_reference(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Clopper-Pearson bounds via the F-distribution identity (independent form)."""
    if k == 0:
        lo = 0.0
    else:
        f = stats.f.ppf(1.0 - alpha / 2.0, 2 * (n - k + 1), 2 * k)
        lo = k / (k + (n - k + 1) * f)
    if k == n:
        hi = 1.0
    else:
        f = stats.f.ppf(1.0 - alpha / 2.0, 2 * (k + 1), 2 * (n - k))
        hi = (k + 1) * f / ((n - k) + (k + 1) * f)
    return float(lo), float(hi)


def test_binom_ci_exact_zero_successes():
    point, lo, hi = binom_ci_exact(0, 10)
    assert point == 0.0
    assert lo == 0.0
    # closed form at k = 0: hi = 1 - (alpha/2)**(1/n)
    assert hi == pytest.approx(1.0 - (ALPHA / 2.0) ** (1.0 / 10.0), abs=1e-6)
    assert (lo, hi) == pytest.approx(_cp_reference(0, 10), abs=1e-6)


def test_binom_ci_exact_all_successes():
    point, lo, hi = binom_ci_exact(10, 10)
    assert point == 1.0
    assert hi == 1.0
    # closed form at k = n: lo = (alpha/2)**(1/n)
    assert lo == pytest.approx((ALPHA / 2.0) ** (1.0 / 10.0), abs=1e-6)
    assert (lo, hi) == pytest.approx(_cp_reference(10, 10), abs=1e-6)


def test_binom_ci_exact_three_of_ten_matches_reference_to_six_dp():
    point, lo, hi = binom_ci_exact(3, 10)
    assert point == pytest.approx(0.3, abs=1e-12)
    ref_lo, ref_hi = _cp_reference(3, 10)
    assert lo == pytest.approx(ref_lo, abs=1e-6)
    assert hi == pytest.approx(ref_hi, abs=1e-6)
    # textbook interval for 3/10 at 95%
    assert round(lo, 4) == 0.0667
    assert round(hi, 4) == 0.6525


def test_binom_ci_exact_coverage_identity():
    """The bounds must solve the defining exact tail equations."""
    k, n = 3, 10
    _, lo, hi = binom_ci_exact(k, n)
    assert stats.binom.sf(k - 1, n, lo) == pytest.approx(ALPHA / 2.0, abs=1e-9)
    assert stats.binom.cdf(k, n, hi) == pytest.approx(ALPHA / 2.0, abs=1e-9)


def test_binom_ci_exact_degenerate_and_validation():
    assert binom_ci_exact(0, 0)[1:] == (0.0, 1.0)
    assert math.isnan(binom_ci_exact(0, 0)[0])
    with pytest.raises(ValueError):
        binom_ci_exact(5, 3)


def test_rule_of_three_upper():
    assert rule_of_three_upper(300) == pytest.approx(0.01)
    assert rule_of_three_upper(0) == 1.0
    assert rule_of_three_upper(1) == 1.0
    # rule of three tracks the exact upper bound at k = 0 within 20% for n >= 30
    for n in (30, 100, 1000):
        exact = binom_ci_exact(0, n)[2]
        assert abs(rule_of_three_upper(n) - exact) / exact < 0.2


# --------------------------------------------------------------------------- #
# percentiles
# --------------------------------------------------------------------------- #


def test_percentiles_basic_and_empty():
    got = percentiles(list(range(1, 101)))
    assert got["p50"] == pytest.approx(50.5)
    assert got["n"] == 100.0
    empty = percentiles([])
    assert empty["n"] == 0.0
    assert all(math.isnan(empty[k]) for k in ("p50", "p95", "p99", "mean"))


# --------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------- #


def test_bootstrap_ci_clustered_point_is_pooled_statistic():
    data = {"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]}
    point, lo, hi = bootstrap_ci_clustered(data, None, 500, 0)
    assert point == pytest.approx(3.5)
    assert lo <= point <= hi


def test_bootstrap_ci_clustered_is_deterministic():
    data = {f"p{i}": [float(i) + j for j in range(4)] for i in range(12)}
    first = bootstrap_ci_clustered(data, None, 800, 7)
    second = bootstrap_ci_clustered(data, None, 800, 7)
    assert first == second


def test_bootstrap_ci_clustered_empty():
    assert all(math.isnan(v) for v in bootstrap_ci_clustered({}, None, 100, 0))
    assert all(math.isnan(v) for v in bootstrap_ci_clustered({"a": []}, None, 100, 0))


def test_bootstrap_ci_clustered_covers_true_median():
    """Coverage on a synthetic clustered generator (fixed seed, loose bound).

    Generator: 40 clusters, 5 observations each; cluster effect and within
    -cluster noise are both symmetric around 0, so the population median is
    exactly 0.  The nominal level is 95%; we require >= 0.85 empirical coverage
    over 200 replicates, which tolerates the known finite-sample
    under-coverage of the percentile bootstrap for a median.
    """
    rng = np.random.default_rng(20260903)
    n_reps = 200
    hits = 0
    for rep in range(n_reps):
        clusters: dict[str, list[float]] = {}
        for c in range(40):
            effect = float(rng.normal(0.0, 1.0))
            noise = rng.normal(0.0, 0.5, size=5)
            clusters[f"c{c:03d}"] = [effect + float(x) for x in noise]
        _, lo, hi = bootstrap_ci_clustered(clusters, None, 400, rep)
        if lo <= 0.0 <= hi:
            hits += 1
    coverage = hits / n_reps
    assert coverage >= 0.85, f"cluster-bootstrap coverage too low: {coverage}"


def test_bootstrap_ci_clustered_custom_stat():
    data = {"a": [0.0, 0.0], "b": [10.0, 10.0]}
    point, lo, hi = bootstrap_ci_clustered(data, lambda x: float(np.mean(x)), 500, 1)
    assert point == pytest.approx(5.0)
    assert lo >= 0.0 and hi <= 10.0


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #


def test_wilcoxon_identical_inputs_gives_p_one():
    xs = [1.0, 2.0, 3.0, 4.0, 5.0]
    res = wilcoxon_paired(xs, xs)
    assert res["p_value"] == 1.0
    assert res["statistic"] == 0.0
    assert res["n"] == 0
    assert res["n_pairs"] == 5
    assert res["degenerate"] is True


def test_wilcoxon_detects_consistent_shift():
    a = [float(i) for i in range(1, 21)]
    b = [x + 5.0 for x in a]
    res = wilcoxon_paired(a, b)
    assert res["n"] == 20
    assert res["p_value"] < 0.01
    assert res["degenerate"] is False


def test_wilcoxon_length_mismatch():
    with pytest.raises(ValueError):
        wilcoxon_paired([1.0, 2.0], [1.0])


def test_paired_randomization_exact_symmetric_case():
    a = [1.0, 2.0, 3.0]
    b = [2.0, 1.0, 3.0]  # differences -1, +1, 0
    res = paired_randomization(a, b)
    assert res["exact"] is True
    assert res["n_nonzero"] == 2
    assert res["n_resamples"] == 4
    assert res["observed_diff"] == pytest.approx(0.0)
    assert res["p_value"] == pytest.approx(1.0)


def test_paired_randomization_all_zero_differences():
    res = paired_randomization([1.0, 2.0], [1.0, 2.0])
    assert res["p_value"] == 1.0
    assert res["observed_diff"] == pytest.approx(0.0)
    assert res["n_nonzero"] == 0


def test_paired_randomization_exact_matches_enumeration():
    a = [3.0, 4.0, 5.0, 6.0]
    b = [1.0, 1.0, 1.0, 1.0]
    res = paired_randomization(a, b)
    # all four differences positive => only the identity flip attains the max
    assert res["exact"] is True
    assert res["p_value"] == pytest.approx(2.0 / 16.0)


def test_paired_randomization_monte_carlo_is_seeded_and_bounded():
    rng = np.random.default_rng(3)
    a = rng.normal(1.0, 1.0, size=40)
    b = rng.normal(0.0, 1.0, size=40)
    r1 = paired_randomization(a, b, 2000, 5)
    r2 = paired_randomization(a, b, 2000, 5)
    assert r1 == r2
    assert r1["exact"] is False
    assert 0.0 < r1["p_value"] <= 1.0
    assert r1["p_value"] < 0.05


def test_mcnemar_equal_discordants_gives_p_one():
    for n in (1, 5, 17):
        res = mcnemar_exact(n, n)
        assert res["p_value"] == pytest.approx(1.0)
        assert res["n_discordant"] == 2 * n


def test_mcnemar_no_discordants():
    res = mcnemar_exact(0, 0)
    assert res["p_value"] == 1.0
    assert res["degenerate"] is True


def test_mcnemar_matches_binomtest_and_rejects_asymmetry():
    res = mcnemar_exact(20, 2)
    assert res["p_value"] == pytest.approx(stats.binomtest(2, 22, 0.5).pvalue)
    assert res["p_value"] < 0.001
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


# --------------------------------------------------------------------------- #
# non-inferiority
# --------------------------------------------------------------------------- #


def test_noninferiority_flags_five_point_drop():
    n = 1000
    b = [True] * n
    a = [False] * 50 + [True] * (n - 50)
    res = noninferiority_success(a, b)
    assert res["diff"] == pytest.approx(-0.05)
    assert res["non_inferior"] is False
    assert res["ci_hi"] < -res["margin"]
    assert res["n01"] == 50 and res["n10"] == 0


def test_noninferiority_accepts_zero_difference():
    n = 1000
    rng = np.random.default_rng(11)
    outcome = list(rng.random(n) < 0.9)
    res = noninferiority_success(outcome, outcome)
    assert res["diff"] == pytest.approx(0.0)
    assert res["continuity_applied"] is True
    assert res["non_inferior"] is True
    # continuity keeps the interval non-degenerate
    assert res["ci_lo"] < 0.0 < res["ci_hi"]
    assert res["ci_lo"] > -res["margin"]


def test_noninferiority_small_sample_is_not_declared_from_no_evidence():
    outcome = [True] * 10
    res = noninferiority_success(outcome, outcome)
    assert res["diff"] == pytest.approx(0.0)
    assert res["non_inferior"] is False  # 1.96/10 = 0.196 > margin


def test_noninferiority_validation():
    with pytest.raises(ValueError):
        noninferiority_success([True], [True, False])
    empty = noninferiority_success([], [])
    assert empty["non_inferior"] is False
    assert empty["n"] == 0


# --------------------------------------------------------------------------- #
# outcome aggregation
# --------------------------------------------------------------------------- #


#: The declared rows the fixtures use, mirroring configs/experiments/main_table.yaml.
REFERENCE = ("diffusion_standard", "diffusion_final_response")
EARLIEST = ("diffusion_standard", "diffusion_earliest_committed")
PREACT = ("preact_style", "preact_ready")
CANDIDATE = ("crystalcall", "full_gate")


def _row(
    policy: str, prompt: str, seed: int, completion: float, arm: str = "diffusion_standard", **kw
) -> dict:
    """One ``PolicyOutcome.to_json()``-shaped row for an ``(arm, policy)`` pair.

    Claim verification defaults to "every claim re-executed and matched", so a
    test only states the coverage hole it wants to exercise.
    """
    claimed = int(kw.pop("claimed", 0))
    row = {
        "trajectory_id": f"{arm}-{prompt}-{seed}-{policy}",
        "arm": arm,
        "prompt_id": prompt,
        "category": kw.pop("category", "simple"),
        "seed": seed,
        "policy": policy,
        "tool_duration_ms": kw.pop("tool_duration_ms", 500.0),
        "completion_ms": completion,
        "baseline_completion_ms": kw.pop("baseline_completion_ms", completion),
        "launches": kw.pop("launches", 0),
        "true_launches": kw.pop("true_launches", 0),
        "false_launches": kw.pop("false_launches", 0),
        "claimed": claimed,
        "wasted_tool_ms": kw.pop("wasted_tool_ms", 0.0),
        "hidden_overlap_ms": kw.pop("hidden_overlap_ms", 0.0),
        "eligible_final_calls": kw.pop("eligible_final_calls", 1),
        "covered_final_calls": kw.pop("covered_final_calls", 0),
        "correct": kw.pop("correct", True),
        "semantic_divergence": kw.pop("semantic_divergence", False),
        "claims_claimed": int(kw.pop("claims_claimed", claimed)),
        "claims_verified_match": int(kw.pop("claims_verified_match", claimed)),
        "claims_verified_divergent": int(kw.pop("claims_verified_divergent", 0)),
        "claims_verification_error": int(kw.pop("claims_verification_error", 0)),
        "claims_unverified": int(kw.pop("claims_unverified", 0)),
        "verification_complete": bool(kw.pop("verification_complete", True)),
        "exclusion": kw.pop("exclusion", None),
    }
    row.update(kw)
    return row


def test_summarize_outcomes_paired_gain_on_four_row_fixture():
    """Hand-built fixture: gains are 1 - 80/100 = 0.20 and 1 - 50/200 = 0.75.

    The median paired relative gain is therefore (0.20 + 0.75) / 2 = 0.475.
    """
    rows = [
        _row("diffusion_final_response", "p1", 0, 100.0),
        _row("diffusion_final_response", "p2", 0, 200.0),
        _row("full_gate", "p1", 0, 80.0, arm="crystalcall", launches=1, true_launches=1,
             claimed=1, covered_final_calls=1, hidden_overlap_ms=20.0),
        _row("full_gate", "p2", 0, 50.0, arm="crystalcall", launches=2, true_launches=1,
             false_launches=1, claimed=1, covered_final_calls=1, wasted_tool_ms=500.0,
             hidden_overlap_ms=150.0),
    ]
    summary = summarize_outcomes(rows, n_boot=500, seed=0)
    assert summary["n_input_rows"] == 4
    assert summary["n_excluded"] == 0
    assert [r["row_key"] for r in summary["rows"]] == [
        "crystalcall/full_gate",
        "diffusion_standard/diffusion_final_response",
    ]
    assert summary["arms"] == ["crystalcall", "diffusion_standard"]
    assert summary["tool_durations_ms"] == [500.0]

    by_row = {g["row_key"]: g for g in summary["groups"]}
    spec = by_row["crystalcall/full_gate"]
    assert spec["gain"]["n_paired"] == 2
    assert spec["gain"]["unpaired"] == 0
    assert spec["gain"]["median"] == pytest.approx(0.475)
    assert spec["gain"]["mean"] == pytest.approx(0.475)
    assert spec["gain"]["ci_lo"] <= 0.475 <= spec["gain"]["ci_hi"]
    assert spec["gain"]["clusters"] == 2

    assert spec["completion_ms"]["p50"] == pytest.approx(65.0)
    assert spec["launches"] == {
        "total": 3,
        "true": 2,
        "false": 1,
        "claimed": 2,
        "duplicate": 0,
        "per_episode": 1.5,
    }
    assert spec["precision"]["k"] == 2 and spec["precision"]["n"] == 3
    assert spec["precision"]["point"] == pytest.approx(2.0 / 3.0)
    assert spec["coverage"]["point"] == pytest.approx(1.0)
    assert spec["waste"]["wasted_tool_ms_total"] == pytest.approx(500.0)
    assert spec["waste"]["cancelled_launches"] == 1
    assert spec["hidden_overlap_ms"]["p50"] == pytest.approx(85.0)
    assert spec["semantic_divergence"] == 0
    assert spec["correct"] == {"true": 2, "false": 0, "unknown": 0}

    base = by_row["diffusion_standard/diffusion_final_response"]
    assert base["gain"]["median"] == pytest.approx(0.0)


def test_summarize_outcomes_reports_unpaired_and_exclusions():
    rows = [
        _row("diffusion_final_response", "p1", 0, 100.0),
        _row("full_gate", "p1", 0, 50.0, arm="crystalcall"),
        _row("full_gate", "p9", 0, 10.0, arm="crystalcall"),  # no reference partner
        _row("full_gate", "p2", 0, 10.0, arm="crystalcall", exclusion="tokenizer_mismatch"),
    ]
    summary = summarize_outcomes(rows, n_boot=200, seed=0)
    assert summary["n_excluded"] == 1
    assert summary["exclusions"] == {"tokenizer_mismatch": 1}
    spec = next(g for g in summary["groups"] if g["row_key"] == "crystalcall/full_gate")
    assert spec["n"] == 2
    assert spec["gain"]["n_paired"] == 1
    assert spec["gain"]["unpaired"] == 1
    assert spec["gain"]["unpaired_examples"] == ["p9#0"]


def test_summarize_outcomes_duplicate_launch_accounting():
    rows = [
        _row("diffusion_final_response", "p1", 0, 100.0),
        _row("full_gate", "p1", 0, 90.0, arm="crystalcall", launches=5, true_launches=1,
             false_launches=1, claimed=1),
    ]
    spec = next(
        g
        for g in summarize_outcomes(rows, n_boot=100, seed=0)["groups"]
        if g["row_key"] == "crystalcall/full_gate"
    )
    assert spec["launches"]["duplicate"] == 3
    assert spec["waste"]["duplicate_launches"] == 3


def test_summarize_outcomes_empty():
    summary = summarize_outcomes([])
    assert summary["groups"] == []
    assert summary["n_input_rows"] == 0


def test_deadline_success_counts_correct_and_on_time():
    rows = [
        _row("full_gate", "p1", 0, 900.0, correct=True),
        _row("full_gate", "p2", 0, 1500.0, correct=True),
        _row("full_gate", "p3", 0, 900.0, correct=False),
        _row("full_gate", "p4", 0, 900.0, correct=None),
    ]
    res = deadline_success(rows, (1.0, 2.0))
    grp = res["groups"][0]
    assert grp["arm"] == "diffusion_standard"
    assert grp["row_key"] == "diffusion_standard/full_gate"
    assert grp["n"] == 4
    assert grp["unknown_correct"] == 1
    b1, b2 = grp["budgets"]
    assert b1["budget_s"] == 1.0
    assert b1["k"] == 1 and b1["q"] == pytest.approx(0.25)
    assert b1["complete_rate"] == pytest.approx(0.75)
    assert b2["k"] == 2 and b2["q"] == pytest.approx(0.5)
    assert b2["complete_rate"] == pytest.approx(1.0)
    assert b1["ci_lo"] <= b1["q"] <= b1["ci_hi"]


def test_deadline_success_default_budgets_and_exclusions():
    rows = [_row("full_gate", "p1", 0, 100.0, exclusion="bad_trace")]
    res = deadline_success(rows)
    assert res["budgets_s"] == [1.0, 2.0, 5.0, 10.0]
    assert res["n_excluded"] == 1
    assert res["groups"] == []


def test_deadline_paired_pairs_by_prompt_and_seed():
    """The whole point of the paired test: two policies with the *same* marginal
    Q(B) on disjoint episodes must show discordance, not a tie."""
    rows = [
        _row("full_gate", "p1", 0, 900.0, correct=True),
        _row("full_gate", "p2", 0, 3000.0, correct=True),
        _row("diffusion_final_response", "p1", 0, 3000.0, correct=True),
        _row("diffusion_final_response", "p2", 0, 900.0, correct=True),
    ]
    res = deadline_paired(
        rows, (1.0,), comparators=[("diffusion_standard", "diffusion_final_response")]
    )
    cell = next(c for c in res["cells"] if c["policy"] == "full_gate")
    assert cell["n_pairs"] == 2
    assert cell["n_prompts"] == 2 and cell["n_seeds"] == 1
    # Same marginal rate (1/2 each) but opposite episodes -> both discordant.
    assert cell["k_candidate"] == 1 and cell["k_comparator"] == 1
    assert cell["mcnemar"]["b"] == 1 and cell["mcnemar"]["c"] == 1
    assert cell["regime"] == "live"


def test_deadline_paired_pairing_is_seed_aware():
    """One prompt, two seeds, outcomes swapped across seeds. Correct pairing
    sees two discordant pairs; pairing that ignored `seed` would collapse this
    to a tie, so this is the discriminator the prompt-level test cannot be."""
    rows = [
        _row("full_gate", "p1", 0, 900.0, correct=True),
        _row("diffusion_final_response", "p1", 0, 3000.0, correct=True),
        _row("full_gate", "p1", 1, 3000.0, correct=True),
        _row("diffusion_final_response", "p1", 1, 900.0, correct=True),
    ]
    res = deadline_paired(
        rows, (1.0,), comparators=[("diffusion_standard", "diffusion_final_response")]
    )
    cell = next(c for c in res["cells"] if c["policy"] == "full_gate")
    assert cell["n_prompts"] == 1 and cell["n_seeds"] == 2
    assert cell["n_pairs"] == 2
    assert cell["mcnemar"]["b"] == 1 and cell["mcnemar"]["c"] == 1
    assert cell["regime"] == "live"


def test_deadline_paired_counts_unpaired_and_labels_empty():
    """Episodes present on one side only are counted, never quietly dropped, and
    a cell with no shared pairs is `empty` rather than `floored`."""
    rows = [
        _row("full_gate", "p1", 0, 900.0, correct=True),
        _row("full_gate", "p2", 0, 900.0, correct=True),
        _row("diffusion_final_response", "p3", 0, 900.0, correct=True),
    ]
    res = deadline_paired(
        rows, (1.0,), comparators=[("diffusion_standard", "diffusion_final_response")]
    )
    cell = next(c for c in res["cells"] if c["policy"] == "full_gate")
    assert cell["n_pairs"] == 0
    assert cell["unpaired_candidate"] == 2
    assert cell["unpaired_baseline"] == 1
    assert cell["regime"] == "empty"


def test_deadline_paired_labels_regimes_and_drops_unknowns():
    rows = [
        # p1: both on time and correct at every budget -> saturated
        _row("full_gate", "p1", 0, 500.0, correct=True),
        _row("diffusion_final_response", "p1", 0, 600.0, correct=True),
        # p2: no ground truth on one side -> pair dropped, counted
        _row("full_gate", "p2", 0, 500.0, correct=True),
        _row("diffusion_final_response", "p2", 0, 600.0, correct=None),
    ]
    res = deadline_paired(
        rows, (1.0, 10.0), comparators=[("diffusion_standard", "diffusion_final_response")]
    )
    cells = [c for c in res["cells"] if c["policy"] == "full_gate"]
    assert {c["unknown_pairs"] for c in cells} == {1}
    assert {c["n_pairs"] for c in cells} == {1}
    assert {c["regime"] for c in cells} == {"saturated"}
    assert res["n_live_cells"] == 0


def test_deadline_paired_floors_and_reports_absent_comparators():
    rows = [
        _row("full_gate", "p1", 0, 9000.0, correct=True),
        _row("diffusion_final_response", "p1", 0, 9000.0, correct=True),
    ]
    res = deadline_paired(
        rows,
        (1.0,),
        comparators=[
            ("diffusion_standard", "diffusion_final_response"),
            ("preact_style", "preact_ready"),
        ],
    )
    assert [c["regime"] for c in res["cells"]] == ["floored"]
    gap = next(c for c in res["comparator_coverage"] if c["comparator"] == "preact_style/preact_ready")
    assert gap["n_cells"] == 0 and gap["reason"]


# --------------------------------------------------------------------------- #
# submission gates S1-S8
# --------------------------------------------------------------------------- #


def _gate_fixture(
    n_prompts: int = 12, durations: tuple[float, ...] = (500.0, 2000.0)
) -> list[dict]:
    """The four declared rows with known completion ratios at D = 500 and 2000 ms."""
    rows: list[dict] = []
    # candidate is faster than every reference; the P-ReAct-style arm sits between
    frac = {
        REFERENCE: 0.0,
        EARLIEST: 0.40,
        PREACT: 0.50,
        CANDIDATE: 0.80,
    }
    for i in range(n_prompts):
        prompt = f"p{i:02d}"
        decode = 1000.0
        for dur in durations:
            for (arm, policy), f in frac.items():
                launches = 0 if (arm, policy) == REFERENCE else 1
                rows.append(
                    _row(
                        policy,
                        prompt,
                        0,
                        decode + dur * (1.0 - f),
                        arm=arm,
                        tool_duration_ms=dur,
                        launches=launches,
                        true_launches=launches,
                        claimed=launches,
                        covered_final_calls=launches,
                        eligible_final_calls=1,
                        correct=True,
                    )
                )
    return rows


def test_submission_gates_ids_order_and_required_flags():
    block = submission_gates(_gate_fixture(), n_boot=400, seed=0)
    ids = [g["id"] for g in block["gates"]]
    assert ids == ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
    keys = [g["key"] for g in block["gates"]]
    assert keys == [
        "speculative_precision",
        "eligible_call_coverage",
        "median_gain_500ms",
        "median_gain_2000ms",
        "eventual_success_noninferior",
        "semantic_divergence_zero",
        "advantage_over_earliest_committed",
        "advantage_over_preact_style",
    ]
    required = {g["id"]: g["required"] for g in block["gates"]}
    assert required == {
        "S1": True,
        "S2": True,
        "S3": True,
        "S4": True,
        "S5": True,
        "S6": True,
        "S7": True,
        "S8": False,
    }
    for g in block["gates"]:
        assert set(g) >= {
            "id",
            "key",
            "description",
            "required_threshold",
            "point",
            "ci_lo",
            "ci_hi",
            "n",
            "pass",
            "required",
        }
    assert {g["id"] for g in block["gates"] if g["decisive"]} == {"S7", "S8"}


def test_submission_gates_thresholds_are_the_contract():
    block = submission_gates(_gate_fixture(), n_boot=200, seed=0)
    thresholds = {g["id"]: g["required_threshold"] for g in block["gates"]}
    assert thresholds["S1"] == 0.99
    assert thresholds["S2"] == 0.40
    assert thresholds["S3"] == 0.15
    assert thresholds["S4"] == 0.25
    assert thresholds["S5"] == pytest.approx(-0.01)
    assert thresholds["S6"] == 0.0
    assert thresholds["S7"] == 0.0
    assert thresholds["S8"] == 0.10
    s3 = next(g for g in block["gates"] if g["id"] == "S3")
    assert s3["target_band"] == [0.15, 0.20]
    assert next(g for g in block["gates"] if g["id"] == "S4")["target_band"] == [0.25, 0.30]


def test_submission_gates_pass_on_a_clean_fixture():
    rows = _gate_fixture(n_prompts=220)  # >= 196 pairs so S5 clears the 1pp margin
    block = submission_gates(rows, n_boot=400, seed=0)
    by_id = {g["id"]: g for g in block["gates"]}
    # precision: every launch matched a final call
    assert by_id["S1"]["point"] == pytest.approx(1.0)
    assert by_id["S1"]["pass"] is True
    # coverage: every eligible final call covered
    assert by_id["S2"]["point"] == pytest.approx(1.0)
    assert by_id["S2"]["pass"] is True
    # gain at D=500: 1 - (1000 + 100) / 1500
    assert by_id["S3"]["point"] == pytest.approx(1.0 - 1100.0 / 1500.0)
    assert by_id["S3"]["pass"] is True
    # gain at D=2000: 1 - (1000 + 400) / 3000
    assert by_id["S4"]["point"] == pytest.approx(1.0 - 1400.0 / 3000.0)
    assert by_id["S4"]["pass"] is True
    assert by_id["S5"]["pass"] is True
    assert by_id["S5"]["detail"]["mcnemar_p"] == 1.0
    assert by_id["S6"]["pass"] is True
    assert by_id["S6"]["point"] == 0.0
    assert by_id["S6"]["detail"]["rule_of_three_upper"] == pytest.approx(3.0 / 220.0)
    assert by_id["S7"]["pass"] is True
    assert by_id["S7"]["detail"]["ci_excludes_zero"] is True
    assert len(by_id["S7"]["detail"]["per_duration"]) == 2
    assert block["all_required_pass"] is True
    assert block["required_passed"] == block["required_total"] == 7
    assert block["decisive"]["S7"] is True


def _is(row: Mapping[str, Any], target: tuple[str, str]) -> bool:
    return (row["arm"], row["policy"]) == target


def test_submission_gates_s1_fails_on_bad_precision_and_reports_ci():
    rows = _gate_fixture(n_prompts=20)
    for r in rows:
        if _is(r, CANDIDATE):
            r["false_launches"] = 1
            r["launches"] = 2
    by_id = {g["id"]: g for g in submission_gates(rows, n_boot=200, seed=0)["gates"]}
    assert by_id["S1"]["point"] == pytest.approx(0.5)
    assert by_id["S1"]["pass"] is False
    assert by_id["S1"]["ci_lo"] < 0.5 < by_id["S1"]["ci_hi"]
    # 20 executions x (1 true + 1 false) launch; the two latency rows per
    # execution are the SAME execution and must not double the denominator.
    assert by_id["S1"]["n"] == 40
    assert by_id["S1"]["detail"]["executions"] == 20


def _s6(rows, **kw) -> dict:
    block = submission_gates(rows, n_boot=100, seed=0, **kw)
    return next(g for g in block["gates"] if g["id"] == "S6")


def test_submission_gates_s6_fails_on_any_divergence():
    rows = _gate_fixture(n_prompts=6)
    for r in rows:
        if _is(r, CANDIDATE) and r["prompt_id"] == "p00":
            r["semantic_divergence"] = True
            r["claims_verified_match"] = 0
            r["claims_verified_divergent"] = 1
    s6 = _s6(rows)
    assert s6["pass"] is False
    assert s6["point"] > 0.0
    # Counters are per source execution, not per replayed latency row.
    assert s6["detail"]["claims_verified_divergent"] == 1
    assert any("diverged" in r for r in s6["detail"]["fail_reasons"])


def test_submission_gates_s6_passes_only_when_fully_verified():
    """12 executions x 1 claim = 12 verified claims, zero divergence.

    The fixture replays each execution at two latencies; the evidence is the
    12 executions, not the 24 rows.
    """
    s6 = _s6(_gate_fixture(n_prompts=12))
    assert s6["pass"] is True
    assert s6["detail"]["fail_reasons"] == []
    assert s6["detail"]["claims_claimed"] == 12
    assert s6["detail"]["claims_verified_match"] == 12
    assert s6["detail"]["claims_verified"] == 12
    assert s6["detail"]["claims_unverified"] == 0
    assert s6["detail"]["claims_unaccounted"] == 0
    assert s6["n"] == 12
    assert s6["point"] == 0.0
    # rule of three is taken over VERIFIED claims, never over launches
    assert s6["detail"]["rule_of_three_upper"] == pytest.approx(3.0 / 12.0)
    assert s6["detail"]["rule_of_three_denominator"] == (
        "claims_verified_match + claims_verified_divergent"
    )
    assert s6["ci_hi"] == pytest.approx(binom_ci_exact(0, 12)[2])


def test_submission_gates_s6_fails_on_a_verification_error():
    """Zero divergence but one claim the verifier could not check must not pass."""
    rows = _gate_fixture(n_prompts=6)
    hit = next(r for r in rows if _is(r, CANDIDATE))
    hit["claims_verified_match"] = 0
    hit["claims_verification_error"] = 1
    hit["verification_complete"] = False
    s6 = _s6(rows)
    assert s6["pass"] is False
    assert s6["detail"]["claims_verified_divergent"] == 0
    assert s6["detail"]["claims_verification_error"] == 1
    assert s6["detail"]["episodes_verification_incomplete"] == 1
    assert any("verifier could not check" in r for r in s6["detail"]["fail_reasons"])
    # the bound is over the 5 executions whose claims WERE verified
    assert s6["n"] == 5


def test_submission_gates_s6_fails_when_claims_are_unverified():
    """claims_claimed=5, verified_match=3, divergent=0 -> two unverified -> FAIL."""
    rows = [
        _row("diffusion_final_response", "p0", 0, 1500.0),
        _row(
            "full_gate",
            "p0",
            0,
            1100.0,
            arm="crystalcall",
            launches=5,
            true_launches=5,
            claimed=5,
            covered_final_calls=1,
            claims_claimed=5,
            claims_verified_match=3,
            claims_verified_divergent=0,
            claims_unverified=2,
            verification_complete=False,
        ),
    ]
    s6 = _s6(rows)
    assert s6["pass"] is False
    assert s6["detail"]["claims_claimed"] == 5
    assert s6["detail"]["claims_verified"] == 3
    assert s6["detail"]["claims_unverified"] == 2
    assert s6["n"] == 3
    assert any("never verified" in r for r in s6["detail"]["fail_reasons"])


def test_submission_gates_s6_fails_when_counters_do_not_add_up():
    """A row whose counters leave claims unaccounted for is a coverage hole."""
    rows = [
        _row("diffusion_final_response", "p0", 0, 1500.0),
        _row(
            "full_gate",
            "p0",
            0,
            1100.0,
            arm="crystalcall",
            launches=4,
            true_launches=4,
            claimed=4,
            claims_claimed=4,
            claims_verified_match=2,
            claims_unverified=0,
        ),
    ]
    s6 = _s6(rows)
    assert s6["pass"] is False
    assert s6["detail"]["claims_unaccounted"] == 2


def test_submission_gates_s6_fails_vacuously_when_nothing_was_claimed():
    rows = [
        _row("diffusion_final_response", "p0", 0, 1500.0),
        _row("full_gate", "p0", 0, 1100.0, arm="crystalcall", launches=0, claimed=0),
    ]
    s6 = _s6(rows)
    assert s6["pass"] is False
    assert s6["detail"]["claims_claimed"] == 0
    assert s6["n"] == 0
    assert any("no speculative claim" in r for r in s6["detail"]["fail_reasons"])
    assert s6["detail"]["rule_of_three_upper"] == 1.0


def test_submission_gates_s6_fails_on_toolsandbox_replay_divergence():
    rows = _gate_fixture(n_prompts=6)
    clean = [
        {"arm": "crystalcall", "seed": 0, "speculate": True, "semantic_divergence": 0,
         "success": True, "exclusion": ""},
        {"arm": "crystalcall", "seed": 1, "speculate": True, "semantic_divergence": 0,
         "success": True, "exclusion": ""},
    ]
    assert _s6(rows, external_divergence_rows=clean)["pass"] is True

    dirty = clean + [
        {"arm": "crystalcall", "seed": 2, "speculate": True, "semantic_divergence": 1,
         "success": False, "exclusion": ""},
        {"arm": "crystalcall", "seed": 3, "speculate": True, "semantic_divergence": 0,
         "success": False, "exclusion": "episode_error:boom"},
    ]
    s6 = _s6(rows, external_divergence_rows=dirty)
    assert s6["pass"] is False
    ext = s6["detail"]["external_replay"]
    assert ext == {
        "episodes": 3,
        "speculative_episodes": 3,
        "divergent": 1,
        "excluded_episodes": 1,
    }
    assert s6["point"] == 1.0
    assert any("replay episode(s) diverged" in r for r in s6["detail"]["fail_reasons"])


def test_summarize_outcomes_reports_verification_accounting():
    rows = _gate_fixture(n_prompts=4)
    rows[-1]["claims_verified_match"] = 0
    rows[-1]["claims_verification_error"] = 1
    summary = summarize_outcomes(
        rows,
        n_boot=100,
        seed=0,
        external_divergence_rows=[
            {"arm": "crystalcall", "speculate": True, "semantic_divergence": 0, "exclusion": ""}
        ],
    )
    ver = summary["accounting"]["verification"]
    # 4 executions x 3 speculating arms.  The two latency rows per execution
    # are the SAME execution and must not double-count the evidence.
    assert ver["claims_claimed"] == 12
    assert ver["claims_verification_error"] == 1
    assert summary["accounting"]["candidate_verification"]["claims_claimed"] == 4
    assert summary["accounting"]["external_replay"]["episodes"] == 1
    grp = next(g for g in summary["groups"] if g["row_key"] == "crystalcall/full_gate")
    assert grp["verification"]["claims_claimed"] == 4


def test_submission_gates_s7_requires_ci_excluding_zero():
    """A candidate that ties the first decisive row must fail the decisive gate."""
    rows = _gate_fixture(n_prompts=15)
    for r in rows:
        if _is(r, CANDIDATE):
            r["completion_ms"] = next(
                x["completion_ms"]
                for x in rows
                if _is(x, EARLIEST)
                and x["prompt_id"] == r["prompt_id"]
                and x["tool_duration_ms"] == r["tool_duration_ms"]
            )
    by_id = {g["id"]: g for g in submission_gates(rows, n_boot=300, seed=0)["gates"]}
    assert by_id["S7"]["point"] == pytest.approx(0.0)
    assert by_id["S7"]["pass"] is False
    assert by_id["S7"]["detail"]["ci_excludes_zero"] is False


def test_submission_gates_s8_missing_arm_is_reported_not_invented():
    rows = [r for r in _gate_fixture(n_prompts=8) if not _is(r, PREACT)]
    block = submission_gates(rows, n_boot=200, seed=0)
    s8 = next(g for g in block["gates"] if g["id"] == "S8")
    assert s8["status"] == "missing"
    assert s8["pass"] is False
    assert s8["required"] is False
    assert "no outcomes for the declared P-ReAct-style row" in s8["detail"]["reason"]
    assert s8["detail"]["reference"]["arm"] == PREACT[0]
    # a missing preferred gate must not sink the required set
    assert block["required_total"] == 7


def test_submission_gates_s8_failure_reports_pareto_comparison():
    rows = _gate_fixture(n_prompts=10)
    for r in rows:
        if _is(r, PREACT):
            r["completion_ms"] = (
                next(
                    x["completion_ms"]
                    for x in rows
                    if _is(x, CANDIDATE)
                    and x["prompt_id"] == r["prompt_id"]
                    and x["tool_duration_ms"] == r["tool_duration_ms"]
                )
                * 1.01  # only 1% slower -> below the 10% preference
            )
            r["false_launches"] = 3
            r["launches"] = 4
            r["wasted_tool_ms"] = 3 * r["tool_duration_ms"]
    block = submission_gates(rows, n_boot=200, seed=0)
    s8 = next(g for g in block["gates"] if g["id"] == "S8")
    assert s8["pass"] is False
    pareto = s8["detail"]["pareto"]
    assert (pareto["candidate"]["arm"], pareto["candidate"]["policy"]) == CANDIDATE
    assert (pareto["reference"]["arm"], pareto["reference"]["policy"]) == PREACT
    assert pareto["candidate"]["precision"]["point"] > pareto["reference"]["precision"]["point"]
    assert (
        pareto["candidate"]["wasted_tool_ms_per_episode"]
        < pareto["reference"]["wasted_tool_ms_per_episode"]
    )


def test_submission_gates_missing_candidate_arm_never_passes():
    rows = [r for r in _gate_fixture(n_prompts=5) if not _is(r, CANDIDATE)]
    block = submission_gates(rows, n_boot=100, seed=0)
    assert block["all_required_pass"] is False
    assert all(g["pass"] is False for g in block["gates"])
    assert {g["status"] for g in block["gates"]} == {"missing"}


def test_summarize_outcomes_embeds_gate_block_and_accounting():
    rows = _gate_fixture(n_prompts=10)
    summary = summarize_outcomes(rows, n_boot=200, seed=0)
    assert (summary["candidate"]["arm"], summary["candidate"]["policy"]) == CANDIDATE
    assert (summary["reference"]["arm"], summary["reference"]["policy"]) == REFERENCE
    assert [g["id"] for g in summary["submission_gates"]["gates"]] == [
        "S1",
        "S2",
        "S3",
        "S4",
        "S5",
        "S6",
        "S7",
        "S8",
    ]
    acc = summary["accounting"]
    assert acc["unpaired_episodes"] == 0
    assert acc["cancelled_launches"] == 0
    assert acc["duplicate_launches"] == 0
    assert acc["semantic_divergence_total"] == 0
    without = summarize_outcomes(rows, n_boot=100, seed=0, gates=False)
    assert "submission_gates" not in without


def test_paired_gain_pools_durations_and_counts_unpaired():
    rows = _gate_fixture(n_prompts=4)
    rows.append(
        _row("full_gate", "orphan", 0, 100.0, arm="crystalcall", tool_duration_ms=500.0)
    )
    pooled = paired_gain(rows, CANDIDATE, REFERENCE, n_boot=200, seed=0)
    assert pooled["n_paired"] == 8  # 4 prompts x 2 durations
    assert pooled["unpaired_candidate"] == 1
    assert pooled["clusters"] == 4
    at_500 = paired_gain(rows, CANDIDATE, REFERENCE, duration=500.0, n_boot=200, seed=0)
    assert at_500["n_paired"] == 4
    assert at_500["median"] == pytest.approx(1.0 - 1100.0 / 1500.0)
    assert pooled["candidate"]["arm"] == CANDIDATE[0]
    assert pooled["reference"]["policy"] == REFERENCE[1]


# --------------------------------------------------------------------------- #
# arm keying: the same policy on two arms is two different results
# --------------------------------------------------------------------------- #


def test_same_policy_on_two_arms_is_never_pooled():
    """`diffusion_final_canvas` exists on two arms; the rows must stay separate."""
    rows: list[dict] = []
    for i in range(8):
        prompt = f"p{i:02d}"
        rows.append(_row("diffusion_final_response", prompt, 0, 1500.0))
        # same policy name, two arms, deliberately different completion times
        rows.append(
            _row(
                "diffusion_final_canvas",
                prompt,
                0,
                1400.0,
                arm="diffusion_standard",
                launches=1,
                true_launches=1,
                claimed=1,
                covered_final_calls=1,
            )
        )
        rows.append(
            _row(
                "diffusion_final_canvas",
                prompt,
                0,
                1200.0,
                arm="preact_style",
                launches=2,
                true_launches=1,
                false_launches=1,
                claimed=1,
                covered_final_calls=1,
                wasted_tool_ms=500.0,
            )
        )
    summary = summarize_outcomes(rows, n_boot=200, seed=0, gates=False)

    keys = [g["row_key"] for g in summary["groups"]]
    assert keys == [
        "diffusion_standard/diffusion_final_canvas",
        "diffusion_standard/diffusion_final_response",
        "preact_style/diffusion_final_canvas",
    ]
    by_row = {g["row_key"]: g for g in summary["groups"]}
    std = by_row["diffusion_standard/diffusion_final_canvas"]
    pre = by_row["preact_style/diffusion_final_canvas"]
    # 8 episodes each, not 16 pooled into one group
    assert std["n"] == pre["n"] == 8
    assert std["completion_ms"]["p50"] == pytest.approx(1400.0)
    assert pre["completion_ms"]["p50"] == pytest.approx(1200.0)
    assert std["gain"]["median"] == pytest.approx(1.0 - 1400.0 / 1500.0)
    assert pre["gain"]["median"] == pytest.approx(1.0 - 1200.0 / 1500.0)
    # risk columns stay separate too
    assert std["precision"]["point"] == pytest.approx(1.0)
    assert pre["precision"]["point"] == pytest.approx(0.5)
    assert std["waste"]["wasted_tool_ms_total"] == pytest.approx(0.0)
    assert pre["waste"]["wasted_tool_ms_total"] == pytest.approx(4000.0)

    # pairing also respects the arm
    std_gain = paired_gain(
        rows, ("diffusion_standard", "diffusion_final_canvas"), REFERENCE, n_boot=200, seed=0
    )
    pre_gain = paired_gain(
        rows, ("preact_style", "diffusion_final_canvas"), REFERENCE, n_boot=200, seed=0
    )
    assert std_gain["n_paired"] == pre_gain["n_paired"] == 8
    assert std_gain["median"] != pre_gain["median"]

    # and so does the deadline curve
    dl = deadline_success(rows, (2.0,))
    assert [g["row_key"] for g in dl["groups"]] == keys


# --------------------------------------------------------------------------- #
# declared main table: a missing row is reported, never dropped
# --------------------------------------------------------------------------- #


_MAIN_TABLE_FIXTURE = """
schema_version: crystalcall-main-table/1
reference_row: diffusion_final_response
decisive_rows:
  - diffusion_earliest_committed
  - preact_style
rows:
  - id: diffusion_final_response
    label: Diffusion final-response
    arm: diffusion_standard
    policy: diffusion_final_response
    source: derived
    speculative: false
  - id: diffusion_earliest_committed
    label: Diffusion earliest committed call
    arm: diffusion_standard
    policy: diffusion_earliest_committed
    source: derived
    speculative: false
  - id: preact_style
    label: P-ReAct-style tool-priority diffusion
    arm: preact_style
    policy: preact_ready
    source: generated
    speculative: true
  - id: gemma4_ar_sptc
    label: Gemma 4 AR sPTC
    arm: gemma4_ar
    policy: ar_sptc
    source: derived
    speculative: true
  - id: crystalcall
    label: CrystalCall (DenoisePTC)
    arm: crystalcall
    policy: full_gate
    source: online
    speculative: true
ablation_arm: diffusion_standard
ablation_policies: [full_gate, random_matched]
latency_conditions_ms: [500, 2000]
deadline_budgets_s: [1, 2]
"""


def test_main_table_keeps_declared_row_without_outcomes(tmp_path):
    """`gemma4_ar/ar_sptc` is declared but has no outcomes: it must be reported."""
    spec_path = tmp_path / "main_table.yaml"
    spec_path.write_text(_MAIN_TABLE_FIXTURE, encoding="utf-8")
    spec = load_main_table(spec_path)
    assert (spec.reference.arm, spec.reference.policy) == REFERENCE
    assert [(d.arm, d.policy) for d in spec.decisive] == [EARLIEST, PREACT]
    assert (spec.candidate.arm, spec.candidate.policy) == CANDIDATE
    assert spec.latency_conditions_ms == (500.0, 2000.0)

    derived = tmp_path / "derived"
    derived.mkdir()
    rows = _gate_fixture(n_prompts=6)  # every declared row EXCEPT gemma4_ar/ar_sptc
    with open(derived / "policy_outcomes.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")

    results = aggregate(
        derived,
        tmp_path / "results.json",
        tmp_path / "results.md",
        frozen_policy_path=None,
        main_table_path=spec_path,
    )
    table = results["main_table"]
    assert [r["id"] for r in table["rows"]] == [
        "diffusion_final_response",
        "diffusion_earliest_committed",
        "preact_style",
        "gemma4_ar_sptc",
        "crystalcall",
    ]
    statuses = {r["id"]: r["status"] for r in table["rows"]}
    assert statuses["gemma4_ar_sptc"] == "missing"
    assert statuses["crystalcall"] == "present"
    assert table["n_declared"] == 5
    assert table["n_missing"] == 1
    absent = next(r for r in table["rows"] if r["id"] == "gemma4_ar_sptc")
    assert absent["pooled"]["n"] == 0
    assert "no outcomes for gemma4_ar/ar_sptc" in absent["pooled"]["reason"]
    assert all(d["status"] == "missing" for d in absent["by_duration"])
    assert [d["tool_duration_ms"] for d in absent["by_duration"]] == [500.0, 2000.0]

    # the spec is hashed into the source list, and the markdown keeps the row
    names = {s["name"]: s for s in results["sources"]}
    assert names["main_table"]["sha256"] == spec.sha256
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "Gemma 4 AR sPTC" in md
    assert "| Method | Success | Semantic divergence | p50 | p95 | p99 |" in md
    # declared deadline budgets drive the Q(B) table
    assert results["sections"]["deadline_success"]["budgets_s"] == [1.0, 2.0]
    # gates read the declared rows, not hardcoded names
    gates = results["submission_gates"]
    assert (gates["candidate"]["arm"], gates["candidate"]["policy"]) == CANDIDATE
    assert [(d["arm"], d["policy"]) for d in gates["decisive_rows"]] == [EARLIEST, PREACT]


# --------------------------------------------------------------------------- #
# verification coverage is per source execution, not per replayed row
# --------------------------------------------------------------------------- #


def _claim_row(arm, traj, duration, policy="full_gate", **over):
    row = {
        "arm": arm,
        "trajectory_id": traj,
        "prompt_id": traj.split("/")[1] if "/" in traj else traj,
        "seed": 0,
        "policy": policy,
        "tool_duration_ms": float(duration),
        "completion_ms": 100.0,
        "baseline_completion_ms": 200.0,
        "correct": True,
        "launches": 1,
        "true_launches": 1,
        "false_launches": 0,
        "claimed": 1,
        "eligible_final_calls": 1,
        "covered_final_calls": 1,
        "wasted_tool_ms": 0.0,
        "hidden_overlap_ms": 10.0,
        "semantic_divergence": False,
        "claims_claimed": 2,
        "claims_verified_match": 2,
        "claims_verified_divergent": 0,
        "claims_verification_error": 0,
        "claims_unverified": 0,
        "verification_complete": True,
    }
    row.update(over)
    return row


def test_verification_totals_are_per_execution_not_per_row():
    """Replay emits one row per (trajectory, policy, latency) and copies the
    trajectory's claim counters into each.  Summing rows would multiply the
    evidence by the number of policies and latency conditions."""
    from crystalcall.metrics import verification_totals

    one_latency = [_claim_row("crystalcall", f"crystalcall/p{i}/s0", 500) for i in range(12)]
    totals_one = verification_totals(one_latency)
    assert totals_one["episodes"] == 12
    assert totals_one["claims_claimed"] == 24
    assert totals_one["claims_verified"] == 24

    # Same 12 executions, now replayed at four latencies and three policies.
    many = [
        _claim_row("crystalcall", f"crystalcall/p{i}/s0", d, policy=pol)
        for i in range(12)
        for d in (100, 500, 2000, 5000)
        for pol in ("full_gate", "persistence_only", "confidence_only")
    ]
    totals_many = verification_totals(many)
    assert totals_many["rows"] == 12 * 4 * 3
    # Adding latency conditions and policies must not change the evidence.
    assert totals_many["episodes"] == totals_one["episodes"]
    assert totals_many["claims_claimed"] == totals_one["claims_claimed"]
    assert totals_many["claims_verified"] == totals_one["claims_verified"]


def test_verification_totals_separate_arms_with_the_same_trajectory_index():
    from crystalcall.metrics import verification_totals

    rows = [
        _claim_row("crystalcall", "crystalcall/p1/s0", 500),
        _claim_row("preact_style", "preact_style/p1/s0", 500),
    ]
    assert verification_totals(rows)["episodes"] == 2


def test_verification_totals_flag_inconsistent_duplicates_and_fail_closed():
    from crystalcall.metrics import verification_totals

    rows = [
        _claim_row("crystalcall", "crystalcall/p1/s0", 500),
        _claim_row(
            "crystalcall",
            "crystalcall/p1/s0",
            2000,
            claims_claimed=3,
            claims_verified_match=2,
            claims_unverified=1,
            verification_complete=False,
        ),
    ]
    totals = verification_totals(rows)
    assert totals["episodes"] == 1
    assert totals["inconsistent_executions"] == 1
    assert totals["claims_claimed"] == 3
    assert totals["episodes_verification_incomplete"] == 1


def test_rule_of_three_bound_does_not_tighten_with_latency_conditions():
    from crystalcall.metrics import rule_of_three_upper, verification_totals

    one = verification_totals(
        [_claim_row("crystalcall", f"crystalcall/p{i}/s0", 500) for i in range(10)]
    )
    many = verification_totals(
        [
            _claim_row("crystalcall", f"crystalcall/p{i}/s0", d)
            for i in range(10)
            for d in (100, 500, 2000, 5000)
        ]
    )
    assert one["claims_verified"] == many["claims_verified"] == 20
    assert rule_of_three_upper(one["claims_verified"]) == rule_of_three_upper(
        many["claims_verified"]
    )


def test_s1_and_s2_are_invariant_to_the_number_of_latency_conditions():
    """Launch and coverage counts describe one model execution.  Replaying the
    same executions at more latencies must not narrow the exact intervals."""
    two = _gate_fixture(n_prompts=15)
    four = _gate_fixture(n_prompts=15, durations=(100.0, 500.0, 2000.0, 5000.0))

    g2 = {g["id"]: g for g in submission_gates(two, n_boot=200, seed=0)["gates"]}
    g4 = {g["id"]: g for g in submission_gates(four, n_boot=200, seed=0)["gates"]}

    for gid in ("S1", "S2"):
        assert g2[gid]["n"] == g4[gid]["n"], gid
        assert g2[gid]["point"] == pytest.approx(g4[gid]["point"]), gid
        assert g2[gid]["ci_lo"] == pytest.approx(g4[gid]["ci_lo"]), gid
        assert g2[gid]["ci_hi"] == pytest.approx(g4[gid]["ci_hi"]), gid
    assert g2["S1"]["detail"]["executions"] == g4["S1"]["detail"]["executions"] == 15
    assert g2["S2"]["detail"]["executions"] == g4["S2"]["detail"]["executions"] == 15


def _fidelity_row(arm, traj, duration, *, policy="full_gate", **over):
    row = {
        "arm": arm,
        "trajectory_id": traj,
        "prompt_id": traj.split("/")[1] if "/" in traj else traj,
        "seed": 0,
        "policy": policy,
        "tool_duration_ms": float(duration),
        "completion_ms": 100.0,
        "baseline_completion_ms": 200.0,
        "correct": True,
        "launches": 2,
        "true_launches": 2,
        "false_launches": 0,
        "claimed": 2,
        "eligible_final_calls": 2,
        "covered_final_calls": 2,
        "wasted_tool_ms": 0.0,
        "hidden_overlap_ms": 10.0,
        "semantic_divergence": False,
        "claims_claimed": 2,
        "claims_verified_match": 2,
        "claims_verified_divergent": 0,
        "claims_verification_error": 0,
        "claims_unverified": 0,
        "verification_complete": True,
        "observed_launched": 2,
        "observed_claimed": 2,
        "observed_wasted": 0,
        "observed_matches_replay": True,
    }
    row.update(over)
    return row


def test_replay_fidelity_lists_mismatching_executions():
    from crystalcall.metrics import replay_fidelity

    rows = [_fidelity_row("crystalcall", f"crystalcall/p{i}/s0", 500) for i in range(3)]
    rows.append(
        _fidelity_row(
            "crystalcall",
            "crystalcall/p9/s0",
            500,
            observed_launched=3,
            observed_matches_replay=False,
        )
    )
    fid = replay_fidelity(rows)
    assert fid["compared"] == 4
    assert fid["matching"] == 3
    assert fid["mismatching"] == 1
    assert fid["clean"] is False
    assert ["crystalcall", "crystalcall/p9/s0"] in fid["mismatching_keys"]


def test_replay_fidelity_excludes_non_comparable_rows_from_compared():
    from crystalcall.metrics import replay_fidelity

    rows = [
        _fidelity_row("crystalcall", "crystalcall/p0/s0", 500),
        # A policy that is not the arm's own frozen gate: expected to differ.
        _fidelity_row(
            "crystalcall",
            "crystalcall/p0/s0",
            500,
            policy="persistence_only",
            observed_matches_replay=None,
        ),
    ]
    fid = replay_fidelity(rows)
    assert fid["compared"] == 1
    assert fid["matching"] == 1
    assert fid["not_comparable"] == 1
    assert fid["clean"] is True


def test_replay_fidelity_precision_is_invariant_to_latency_conditions():
    """The candidate's replayed precision interval must not tighten when the
    same executions are replayed at more tool durations."""
    from crystalcall.metrics import ArmPolicy, replay_fidelity

    cand = ArmPolicy("crystalcall", "full_gate")
    two = [
        _fidelity_row("crystalcall", f"crystalcall/p{i}/s0", d)
        for i in range(8)
        for d in (500, 2000)
    ]
    four = [
        _fidelity_row("crystalcall", f"crystalcall/p{i}/s0", d)
        for i in range(8)
        for d in (100, 500, 2000, 5000)
    ]
    a = replay_fidelity(two, candidate=cand)["candidate_precision_replayed"]
    b = replay_fidelity(four, candidate=cand)["candidate_precision_replayed"]
    assert a["n"] == b["n"] == 16
    assert a["executions"] == b["executions"] == 8
    assert a["ci_lo"] == pytest.approx(b["ci_lo"])
    assert a["ci_hi"] == pytest.approx(b["ci_hi"])

    on_a = replay_fidelity(two, candidate=cand)["candidate_precision_online"]
    on_b = replay_fidelity(four, candidate=cand)["candidate_precision_online"]
    assert on_a["n"] == on_b["n"]
    assert on_a["ci_hi"] == pytest.approx(on_b["ci_hi"])


# --------------------------------------------------------------------------- #
# report surfacing: instrumentation cost and replay fidelity
# --------------------------------------------------------------------------- #


def _write_derived(tmp_path, name: str, rows) -> None:
    derived = tmp_path / "derived"
    derived.mkdir(exist_ok=True)
    with open(derived / name, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, sort_keys=True) + "\n")


_LADDER_ROWS = [
    {
        "schema_version": "crystalcall-overhead-microbench/1",
        "level": "off",
        "n": 30,
        "p50_ms": 800.0,
        "p95_ms": 900.0,
        "revisions_per_request": 0.0,
        "parse_ms_per_revision": 0.0,
        "decode_ms_per_revision": 0.0,
        "relative_p50_vs_off": 0.0,
        "absolute_p50_delta_ms": 0.0,
        "problems": [],
        "ok": True,
    },
    {
        "schema_version": "crystalcall-overhead-microbench/1",
        "level": "parse",
        "n": 30,
        "p50_ms": 826.0,
        "p95_ms": 932.0,
        "revisions_per_request": 24.0,
        "parse_ms_per_revision": 0.42,
        "decode_ms_per_revision": 0.02,
        "relative_p50_vs_off": 826.0 / 800.0 - 1.0,
        "absolute_p50_delta_ms": 26.0,
        "problems": ["1 malformed side-channel frames"],
        "ok": False,
    },
]


def test_instrumentation_section_reports_in_timeline_cost_and_ladder(tmp_path):
    """The in-timeline cost is reported, never charged, next to the ladder rungs."""
    spec_path = tmp_path / "main_table.yaml"
    spec_path.write_text(_MAIN_TABLE_FIXTURE, encoding="utf-8")
    rows = _gate_fixture(n_prompts=4)
    for r in rows:
        r["instrumentation_in_timeline_ms"] = 10.08
    _write_derived(tmp_path, "policy_outcomes.jsonl", rows)
    _write_derived(tmp_path, "overhead_microbench.jsonl", _LADDER_ROWS)

    results = aggregate(
        tmp_path / "derived",
        tmp_path / "results.json",
        tmp_path / "results.md",
        frozen_policy_path=None,
        main_table_path=spec_path,
    )
    sec = results["sections"]["instrumentation"]
    it = sec["in_timeline"]
    # one value per execution, not per (execution x tool duration) row
    assert it["requests"] == 4 * 4  # 4 prompts x 4 declared rows present in the fixture
    assert it["in_timeline_ms"]["p50"] == pytest.approx(10.08)
    assert it["charged_to_completion"] is False
    assert [r["level"] for r in sec["ladder"]["rungs"]] == ["off", "parse"]
    assert sec["ladder"]["fields"] == [
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
    ]
    parse_rung = next(r for r in sec["ladder"]["rungs"] if r["level"] == "parse")
    assert parse_rung["relative_p50_vs_off"] == pytest.approx(826.0 / 800.0 - 1.0)
    assert parse_rung["absolute_p50_delta_ms"] == pytest.approx(26.0)
    assert parse_rung["ok"] is False
    assert sec["ladder"]["failing_rungs"] == ["parse"]
    # a rung that failed its own sanity checks is counted, not quietly averaged
    assert ["overhead_ladder", "rung_not_ok:parse", 1] in results["exclusions"]["rows"]

    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "All arms replay the same instrumented timeline" in md
    assert "nor credited back to the reference arm" in md
    assert "### Phase 0 overhead ladder" in md
    assert "| revisions/request |" in md
    assert "Failing rungs" in md
    # the completion times must not include the instrumentation charge twice:
    # the candidate's p50 at D=500 is decode + D*(1 - hidden fraction)
    crystal = next(r for r in results["main_table"]["rows"] if r["id"] == "crystalcall")
    at_500 = next(d for d in crystal["by_duration"] if d["tool_duration_ms"] == 500.0)
    assert at_500["completion_ms"]["p50"] == pytest.approx(1000.0 + 500.0 * 0.2)


def test_replay_fidelity_mismatch_is_surfaced_loudly(tmp_path):
    spec_path = tmp_path / "main_table.yaml"
    spec_path.write_text(_MAIN_TABLE_FIXTURE, encoding="utf-8")
    rows = _gate_fixture(n_prompts=5)
    for r in rows:
        own_gate = (r["arm"], r["policy"]) == CANDIDATE
        r["observed_launched"] = r["launches"]
        r["observed_claimed"] = r["claimed"]
        r["observed_wasted"] = 0
        r["observed_matches_replay"] = True if own_gate else None
    broken = [r for r in rows if _is(r, CANDIDATE) and r["prompt_id"] == "p03"]
    for r in broken:
        r["observed_launched"] = r["launches"] + 1
        r["observed_matches_replay"] = False
    _write_derived(tmp_path, "policy_outcomes.jsonl", rows)

    results = aggregate(
        tmp_path / "derived",
        tmp_path / "results.json",
        tmp_path / "results.md",
        frozen_policy_path=None,
        main_table_path=spec_path,
    )
    fid = results["sections"]["method_comparison"]["summary"]["accounting"]["replay_fidelity"]
    assert fid["compared"] == 5
    assert fid["matching"] == 4
    assert fid["mismatching"] == 1
    assert fid["clean"] is False
    assert [k[0] for k in fid["mismatching_keys"]] == ["crystalcall"]
    assert fid["mismatching_rows"][0]["observed_launched"] == 2
    assert fid["mismatching_rows"][0]["replayed_launches"] == 1
    # rows of other policies are not comparable and must not count as matches
    assert fid["not_comparable"] > 0
    assert ["replay_fidelity", "observed_vs_replay_mismatch", 1] in results["exclusions"]["rows"]

    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "REPLAY FIDELITY FAILURE" in md
    assert "bug signal, not a result" in md
    assert "p03" in md  # the offending trajectory is named, not averaged away
    assert "online future table" in md and "offline replay" in md


def test_replay_fidelity_clean_run_says_so(tmp_path):
    spec_path = tmp_path / "main_table.yaml"
    spec_path.write_text(_MAIN_TABLE_FIXTURE, encoding="utf-8")
    rows = _gate_fixture(n_prompts=3)
    for r in rows:
        own_gate = (r["arm"], r["policy"]) == CANDIDATE
        r["observed_launched"] = r["launches"]
        r["observed_claimed"] = r["claimed"]
        r["observed_wasted"] = 0
        r["observed_matches_replay"] = True if own_gate else None
    _write_derived(tmp_path, "policy_outcomes.jsonl", rows)
    results = aggregate(
        tmp_path / "derived",
        tmp_path / "results.json",
        tmp_path / "results.md",
        frozen_policy_path=None,
        main_table_path=spec_path,
    )
    fid = results["sections"]["method_comparison"]["summary"]["accounting"]["replay_fidelity"]
    assert fid["mismatching"] == 0
    assert fid["clean"] is True
    assert fid["mismatching_rows"] == []
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "REPLAY FIDELITY FAILURE" not in md
    assert "Every comparable request reproduced its online launch count" in md
    assert not any(row[0] == "replay_fidelity" for row in results["exclusions"]["rows"])


# --------------------------------------------------------------------------- #
# S6 reads the ONLINE validation rows; S3/S4 read the derived candidate
# --------------------------------------------------------------------------- #

DERIVED_CANDIDATE = ("diffusion_standard", "full_gate")
ONLINE_VALIDATION = ("crystalcall", "full_gate")


def _split_fixture(n_prompts=12, *, online=True, online_kw=None):
    """The realistic shape: the primary CrystalCall row
    is DERIVED from the standard trajectories and carries no claim counters,
    while a bounded online arm carries the verification evidence."""
    rows = []
    frac = {REFERENCE: 0.0, EARLIEST: 0.40, PREACT: 0.50, DERIVED_CANDIDATE: 0.80}
    for i in range(n_prompts):
        prompt = f"p{i:02d}"
        for dur in (500.0, 2000.0):
            for (arm, policy), f in frac.items():
                launches = 0 if (arm, policy) == REFERENCE else 1
                rows.append(
                    _row(
                        policy,
                        prompt,
                        0,
                        1000.0 + dur * (1.0 - f),
                        arm=arm,
                        tool_duration_ms=dur,
                        launches=launches,
                        true_launches=launches,
                        claimed=launches,
                        covered_final_calls=launches,
                        eligible_final_calls=1,
                        correct=True,
                        # A derived row has no live claims to verify.
                        claims_claimed=0,
                        claims_verified_match=0,
                        claims_verified_divergent=0,
                        claims_verification_error=0,
                        claims_unverified=0,
                        verification_complete=True,
                    )
                )
    if online:
        kw = {
            "claims_claimed": 1,
            "claims_verified_match": 1,
            "claims_verified_divergent": 0,
            "claims_verification_error": 0,
            "claims_unverified": 0,
            "verification_complete": True,
        }
        kw.update(online_kw or {})
        for i in range(6):  # bounded subset
            rows.append(
                _row(
                    ONLINE_VALIDATION[1],
                    f"p{i:02d}",
                    0,
                    1400.0,
                    arm=ONLINE_VALIDATION[0],
                    tool_duration_ms=500.0,
                    launches=1,
                    true_launches=1,
                    claimed=1,
                    covered_final_calls=1,
                    eligible_final_calls=1,
                    correct=True,
                    **kw,
                )
            )
    return rows


def _gates_split(rows, **kw):
    block = submission_gates(
        rows,
        candidate=DERIVED_CANDIDATE,
        reference=REFERENCE,
        decisive=(EARLIEST, PREACT),
        verification=(ONLINE_VALIDATION,),
        n_boot=100,
        seed=0,
        **kw,
    )
    return {g["id"]: g for g in block["gates"]}


def test_s6_reads_the_online_validation_rows_not_the_derived_candidate():
    by_id = _gates_split(_split_fixture())
    s6 = by_id["S6"]
    assert s6["pass"] is True
    assert s6["detail"]["verification_rows"] == ["crystalcall/full_gate"]
    assert s6["detail"]["claims_claimed"] == 6
    assert s6["detail"]["claims_verified"] == 6
    assert s6["n"] == 6
    assert s6["detail"]["rule_of_three_upper"] == pytest.approx(3.0 / 6.0)


def test_s6_fails_closed_when_the_validation_arm_is_absent():
    """A derived-only run has no claim counters at all: S6 must not pass
    vacuously just because the derived rows report zero divergence."""
    by_id = _gates_split(_split_fixture(online=False))
    s6 = by_id["S6"]
    assert s6["pass"] is False
    assert s6["status"] == "missing"
    assert any("no online validation episodes" in r for r in s6["detail"]["fail_reasons"])


def test_s6_fails_on_validation_divergence_without_touching_the_latency_gates():
    clean = _gates_split(_split_fixture())
    dirty = _gates_split(
        _split_fixture(
            online_kw={
                "claims_verified_match": 0,
                "claims_verified_divergent": 1,
                "semantic_divergence": True,
            }
        )
    )
    assert clean["S6"]["pass"] is True
    assert dirty["S6"]["pass"] is False
    assert dirty["S6"]["detail"]["claims_verified_divergent"] == 6
    # The latency estimates come from the DERIVED candidate and must be identical.
    for gid in ("S3", "S4", "S7", "S8"):
        assert clean[gid]["point"] == pytest.approx(dirty[gid]["point"]), gid
        assert clean[gid]["n"] == dirty[gid]["n"], gid
    # ...as must precision and coverage.
    for gid in ("S1", "S2"):
        assert clean[gid]["point"] == pytest.approx(dirty[gid]["point"]), gid
        assert clean[gid]["n"] == dirty[gid]["n"], gid


def test_latency_gates_come_from_the_derived_candidate():
    by_id = _gates_split(_split_fixture())
    # gain at D=500 for the derived candidate: 1 - (1000 + 100) / 1500
    assert by_id["S3"]["point"] == pytest.approx(1.0 - 1100.0 / 1500.0)
    assert by_id["S4"]["point"] == pytest.approx(1.0 - 1400.0 / 3000.0)
    assert by_id["S1"]["detail"]["executions"] == 12
