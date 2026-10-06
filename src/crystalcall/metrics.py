"""Statistical estimators for the CrystalCall / DenoisePTC evaluation.

Every number that appears in the paper is produced by one of the estimators in
this module, computed from a data file.  Nothing here reads a hard-coded
measurement, and every randomised procedure takes an explicit seed so that two
runs on the same input produce byte-identical output.

Design rules obeyed by every function below:

* ``numpy`` + ``scipy`` only (no statsmodels, no seaborn, no plotly).
* Deterministic iteration: dictionaries are traversed in ``sorted`` key order
  and every resampling procedure is driven by ``numpy.random.default_rng`` with
  an explicit ``seed``.
* Degenerate inputs (empty samples, zero discordant pairs, all-zero paired
  differences, ``k == 0`` or ``k == n`` binomial counts) return a well-defined
  value instead of raising.
* Nothing is silently dropped: rows that cannot be paired or that carry an
  ``exclusion`` reason are *counted and reported*.

See ``docs/metrics.md`` for the estimator-by-estimator description, the
assumptions each one makes and the submission gate it feeds.
"""

from __future__ import annotations

import itertools
import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

__all__ = [
    "ArmPolicy",
    "REFERENCE_ROW",
    "CANDIDATE_ROW",
    "VERIFICATION_ROW",
    "DECISIVE_ROWS",
    "ABLATION_ARM",
    "DEFAULT_ALPHA",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_PERCENTILES",
    "NONINFERIORITY_MARGIN",
    "percentiles",
    "bootstrap_ci_clustered",
    "wilcoxon_paired",
    "paired_randomization",
    "mcnemar_exact",
    "noninferiority_success",
    "binom_ci_exact",
    "rule_of_three_upper",
    "verification_totals",
    "external_divergence_totals",
    "observed_totals",
    "replay_fidelity",
    "instrumentation_totals",
    "summarize_outcomes",
    "deadline_success",
    "deadline_paired",
    "paired_gain",
    "submission_gates",
    "SUBMISSION_GATE_SPECS",
]

DEFAULT_PERCENTILES: tuple[int, ...] = (50, 95, 99)
DEFAULT_BOOTSTRAP = 10_000
DEFAULT_ALPHA = 0.05

#: One percentage point of task success, in absolute terms.
NONINFERIORITY_MARGIN = 0.01

#: Budgets (seconds) for the deadline-success curve Q(B).
DEFAULT_BUDGETS_S: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0)


@dataclass(frozen=True, slots=True)
class ArmPolicy:
    """One row of the comparison table: an ``(arm, policy)`` pair.

    The same offline policy replayed against two arms' traces is two *different*
    results, so every grouping, pairing and table row is keyed on the pair, never
    on the bare policy name.  ``id`` / ``label`` carry the declaration from
    ``configs/experiments/main_table.yaml`` when the pair came from there.
    """

    arm: str
    policy: str
    id: str = ""
    label: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.arm, self.policy)

    @property
    def display(self) -> str:
        return self.label or self.id or f"{self.arm}/{self.policy}"

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id or f"{self.arm}/{self.policy}",
            "label": self.display,
            "arm": self.arm,
            "policy": self.policy,
        }

    @staticmethod
    def coerce(value: Any) -> ArmPolicy:
        """Accept an :class:`ArmPolicy`, an ``(arm, policy)`` pair or a mapping."""
        if isinstance(value, ArmPolicy):
            return value
        if isinstance(value, Mapping):
            return ArmPolicy(
                arm=str(value["arm"]),
                policy=str(value["policy"]),
                id=str(value.get("id", "")),
                label=str(value.get("label", "")),
            )
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return ArmPolicy(arm=str(value[0]), policy=str(value[1]))
        raise TypeError(
            f"cannot interpret {value!r} as an (arm, policy) row; pass ArmPolicy, "
            "a 2-tuple or a mapping with 'arm' and 'policy'"
        )


#: Reference denominator of every paired latency claim: the standard diffusion
#: arm issuing its tool calls only after the response is complete.  A gain over
#: this row alone is explicitly not sufficient evidence (see S7/S8).
REFERENCE_ROW = ArmPolicy(
    arm="diffusion_standard",
    policy="diffusion_final_response",
    id="diffusion_final_response",
    label="Diffusion final-response",
)

#: The arm/policy pair the submission gates judge.
CANDIDATE_ROW = ArmPolicy(
    arm="crystalcall", policy="full_gate", id="crystalcall", label="CrystalCall (DenoisePTC)"
)

#: Where the CLAIM-VERIFICATION evidence for S6 comes from.  The primary
#: candidate row is derived by replaying the standard trajectories, so it carries
#: no claim counters at all: only a live speculative run does.  S6 therefore
#: reads the bounded online validation arm (plus the ToolSandbox replay check),
#: while S1-S5/S7/S8 read the primary candidate.
VERIFICATION_ROW = ArmPolicy(
    arm="crystalcall",
    policy="full_gate",
    id="crystalcall_online",
    label="CrystalCall online validation (bounded subset)",
)

#: The two decisive comparisons: S7 (index 0) and S8 (index 1).
DECISIVE_ROWS: tuple[ArmPolicy, ...] = (
    ArmPolicy(
        arm="diffusion_standard",
        policy="diffusion_earliest_committed",
        id="diffusion_earliest_committed",
        label="Diffusion earliest committed call",
    ),
    ArmPolicy(
        arm="preact_style",
        policy="preact_ready",
        id="preact_style",
        label="P-ReAct-style tool-priority diffusion",
    ),
)

#: Arm the gate ablations are replayed from.
ABLATION_ARM = "diffusion_standard"

#: The submission gates: ordered, id-keyed, with required thresholds.
SUBMISSION_GATE_SPECS: tuple[dict[str, Any], ...] = (
    {
        "id": "S1",
        "key": "speculative_precision",
        "description": "speculative precision of the candidate arm",
        "statistic": "exact Clopper-Pearson on launches",
        "comparator": ">=",
        "threshold": 0.99,
        "required": True,
    },
    {
        "id": "S2",
        "key": "eligible_call_coverage",
        "description": "fraction of speculation-eligible final calls covered in flight",
        "statistic": "exact Clopper-Pearson on eligible final calls",
        "comparator": ">=",
        "threshold": 0.40,
        "required": True,
    },
    {
        "id": "S3",
        "key": "median_gain_500ms",
        "description": "paired median relative gain vs diffusion_final_response at D=500 ms",
        "statistic": "cluster-bootstrap median + Wilcoxon / randomization p",
        "comparator": ">=",
        "threshold": 0.15,
        "target_band": (0.15, 0.20),
        "required": True,
    },
    {
        "id": "S4",
        "key": "median_gain_2000ms",
        "description": "paired median relative gain vs diffusion_final_response at D=2000 ms",
        "statistic": "cluster-bootstrap median + Wilcoxon / randomization p",
        "comparator": ">=",
        "threshold": 0.25,
        "target_band": (0.25, 0.30),
        "required": True,
    },
    {
        "id": "S5",
        "key": "eventual_success_noninferior",
        "description": "paired task-success difference within 1 percentage point",
        "statistic": "paired-difference Wald CI (McNemar exact p reported)",
        "comparator": "ci_lo >",
        "threshold": -NONINFERIORITY_MARGIN,
        "required": True,
    },
    {
        "id": "S6",
        "key": "semantic_divergence_zero",
        "description": "no claimed speculative result diverged from authoritative execution",
        "statistic": "observed count + rule-of-three 95% upper bound on the rate",
        "comparator": "==",
        "threshold": 0.0,
        "required": True,
    },
    {
        "id": "S7",
        "key": "advantage_over_earliest_committed",
        "description": (
            "paired median gain vs the first decisive row (strongest non-speculative"
            " diffusion baseline) > 0 with a CI excluding 0 (DECISIVE)"
        ),
        "statistic": "cluster-bootstrap median + Wilcoxon / randomization p",
        "comparator": "> 0 and ci_lo > 0",
        "threshold": 0.0,
        "required": True,
        "decisive": True,
    },
    {
        "id": "S8",
        "key": "advantage_over_preact_style",
        "description": (
            "relative gain vs the second decisive row (P-ReAct-style arm), or a"
            " materially better latency/risk trade-off (preferred, not required)"
        ),
        "statistic": "cluster-bootstrap median + Pareto comparison on failure",
        "comparator": ">=",
        "threshold": 0.10,
        "required": False,
        "decisive": True,
    },
)

_EXACT_RANDOMIZATION_LIMIT = 20  # 2**20 == 1_048_576 sign vectors


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #


def _as_float_array(xs: Any) -> np.ndarray:
    arr = np.asarray(list(xs) if not isinstance(xs, np.ndarray) else xs, dtype=float)
    return arr.reshape(-1)


def _finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def _paired(a: Any, b: Any) -> tuple[np.ndarray, np.ndarray]:
    x = _as_float_array(a)
    y = _as_float_array(b)
    if x.size != y.size:
        raise ValueError(f"paired inputs must have equal length, got {x.size} and {y.size}")
    return x, y


def _median(arr: np.ndarray) -> float:
    return float(np.median(arr)) if arr.size else float("nan")


# --------------------------------------------------------------------------- #
# descriptive statistics
# --------------------------------------------------------------------------- #


def percentiles(
    xs: Sequence[float] | np.ndarray,
    ps: Sequence[int | float] = DEFAULT_PERCENTILES,
) -> dict[str, float]:
    """Linear-interpolated percentiles plus the sample size.

    Keys are ``"p50"``, ``"p95"``, ... (``:g`` formatted) and ``"n"``.  Empty or
    all-non-finite samples give ``nan`` percentiles and ``n == 0`` rather than
    raising, so that a missing arm shows up as ``nan`` in a table instead of
    aborting the whole aggregation.
    """
    arr = _finite(_as_float_array(xs))
    out: dict[str, float] = {}
    if arr.size == 0:
        for p in ps:
            out[f"p{p:g}"] = float("nan")
        out["n"] = 0.0
        out["mean"] = float("nan")
        return out
    qs = np.percentile(arr, [float(p) for p in ps], method="linear")
    for p, q in zip(ps, np.atleast_1d(qs), strict=True):
        out[f"p{p:g}"] = float(q)
    out["n"] = float(arr.size)
    out["mean"] = float(arr.mean())
    return out


# --------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------- #


def bootstrap_ci_clustered(
    values_by_cluster: Mapping[str, Sequence[float]],
    stat: Callable[[np.ndarray], float] | None = None,
    n: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> tuple[float, float, float]:
    """Cluster (block) bootstrap percentile CI.

    Observations are grouped into clusters -- in this project one cluster is one
    ``prompt_id``, because repeated seeds of the same prompt are strongly
    dependent.  Each replicate resamples ``m`` *clusters* with replacement
    (``m`` = number of observed clusters) and pools **all** observations of every
    drawn cluster, which preserves the within-cluster correlation instead of
    assuming i.i.d. rows.

    ``stat`` defaults to the median of the pooled values.  Returns
    ``(point, lo, hi)`` where ``point`` is the statistic on the observed data and
    ``lo``/``hi`` are the ``alpha/2`` and ``1 - alpha/2`` percentiles of the
    bootstrap distribution.  Fully empty input yields ``(nan, nan, nan)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")

    arrays: list[np.ndarray] = []
    for key in sorted(values_by_cluster):
        arr = _finite(_as_float_array(values_by_cluster[key]))
        if arr.size:
            arrays.append(arr)
    if not arrays:
        return (float("nan"), float("nan"), float("nan"))

    pooled = np.concatenate(arrays)
    fn = stat if stat is not None else _median
    point = float(fn(pooled))

    m = len(arrays)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, m, size=(int(n), m))

    sizes = {a.size for a in arrays}
    if stat is None and len(sizes) == 1:
        # Fast, fully vectorised path: equal-size clusters form a matrix.
        mat = np.stack(arrays)  # (m, s)
        reps = np.median(mat[draws].reshape(int(n), -1), axis=1)
    else:
        reps = np.empty(int(n), dtype=float)
        for i in range(int(n)):
            sample = np.concatenate([arrays[j] for j in draws[i]])
            reps[i] = float(fn(sample))

    reps = _finite(reps)
    if reps.size == 0:
        return (point, float("nan"), float("nan"))
    lo, hi = np.percentile(reps, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    return (point, float(lo), float(hi))


# --------------------------------------------------------------------------- #
# paired tests
# --------------------------------------------------------------------------- #


def wilcoxon_paired(a: Sequence[float], b: Sequence[float]) -> dict[str, Any]:
    """Two-sided Wilcoxon signed-rank test on paired observations.

    ``zero_method="wilcox"`` discards zero differences (the classic Wilcoxon
    convention).  When *every* difference is zero -- or when no non-zero pair
    survives -- ``scipy`` refuses to run; that case is exactly "the two arms are
    indistinguishable", so it is reported as ``p_value = 1.0`` with a zero
    statistic instead of an exception.

    ``n`` is the number of non-zero pairs actually used by the test;
    ``n_pairs`` is the input length.
    """
    x, y = _paired(a, b)
    diff = x - y
    nz = diff[np.isfinite(diff) & (diff != 0.0)]
    if nz.size == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n": 0,
            "n_pairs": int(diff.size),
            "degenerate": True,
        }
    res = stats.wilcoxon(nz, zero_method="wilcox", alternative="two-sided")
    return {
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
        "n": int(nz.size),
        "n_pairs": int(diff.size),
        "degenerate": False,
    }


def paired_randomization(
    a: Sequence[float],
    b: Sequence[float],
    n: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
) -> dict[str, Any]:
    """Sign-flip (randomisation) test on paired differences.

    Under the sharp null of exchangeability within a pair, the sign of every
    non-zero difference is equally likely to flip.  With ``m`` non-zero pairs the
    test is computed **exactly** by enumerating all ``2**m`` sign vectors when
    ``m <= 20``; otherwise ``n`` Monte-Carlo sign vectors are drawn and the
    add-one estimator ``(1 + #{|d*| >= |d|}) / (n + 1)`` is used, which is valid
    (never anti-conservative) for a finite number of draws.

    Zero differences are invariant under sign flips and are kept in the
    denominator of the mean, so ``observed_diff`` is the mean over *all* pairs.
    """
    x, y = _paired(a, b)
    diff = x - y
    diff = diff[np.isfinite(diff)]
    total = diff.size
    if total == 0:
        return {
            "observed_diff": float("nan"),
            "p_value": 1.0,
            "n": 0,
            "n_nonzero": 0,
            "exact": True,
            "n_resamples": 0,
        }
    observed = float(diff.mean())
    nz = diff[diff != 0.0]
    m = int(nz.size)
    if m == 0:
        return {
            "observed_diff": observed,
            "p_value": 1.0,
            "n": total,
            "n_nonzero": 0,
            "exact": True,
            "n_resamples": 1,
        }

    target = abs(float(nz.sum()))
    tol = 1e-12 * max(1.0, target)

    if m <= _EXACT_RANDOMIZATION_LIMIT:
        signs = np.array(list(itertools.product((1.0, -1.0), repeat=m)), dtype=float)
        sums = np.abs(signs @ nz)
        p = float(np.count_nonzero(sums >= target - tol)) / float(sums.size)
        return {
            "observed_diff": observed,
            "p_value": min(1.0, p),
            "n": total,
            "n_nonzero": m,
            "exact": True,
            "n_resamples": int(sums.size),
        }

    rng = np.random.default_rng(seed)
    hits = 0
    chunk = max(1, min(int(n), max(1, 4_000_000 // m)))
    drawn = 0
    while drawn < int(n):
        k = min(chunk, int(n) - drawn)
        signs = rng.integers(0, 2, size=(k, m)).astype(float) * 2.0 - 1.0
        sums = np.abs(signs @ nz)
        hits += int(np.count_nonzero(sums >= target - tol))
        drawn += k
    p = (1.0 + hits) / (1.0 + float(drawn))
    return {
        "observed_diff": observed,
        "p_value": min(1.0, p),
        "n": total,
        "n_nonzero": m,
        "exact": False,
        "n_resamples": drawn,
    }


def mcnemar_exact(b: int, c: int) -> dict[str, Any]:
    """Exact (binomial) McNemar test on discordant pair counts.

    ``b`` and ``c`` are the two discordant cells of the paired 2x2 table
    (``b`` = arm A correct & arm B wrong, ``c`` = A wrong & B correct).  The
    exact test is a two-sided binomial test of ``min(b, c)`` successes out of
    ``b + c`` trials at ``p = 0.5``.  With no discordant pairs the arms are
    identical on every episode and ``p_value = 1.0``.
    """
    b = int(b)
    c = int(c)
    if b < 0 or c < 0:
        raise ValueError(f"discordant counts must be non-negative, got b={b}, c={c}")
    nd = b + c
    if nd == 0:
        return {
            "b": b,
            "c": c,
            "n_discordant": 0,
            "statistic": 0,
            "p_value": 1.0,
            "degenerate": True,
        }
    res = stats.binomtest(min(b, c), nd, 0.5)
    return {
        "b": b,
        "c": c,
        "n_discordant": nd,
        "statistic": int(min(b, c)),
        "p_value": float(res.pvalue),
        "degenerate": False,
    }


def noninferiority_success(
    a_success: Sequence[bool],
    b_success: Sequence[bool],
    margin: float = NONINFERIORITY_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired non-inferiority test for a binary outcome.

    ``a_success`` is the candidate arm (speculative execution), ``b_success`` the
    reference arm; both are equal-length boolean sequences indexed by the *same*
    episodes.  The estimand is the paired difference in success rates
    ``diff = mean(a) - mean(b) = (n10 - n01) / n`` where ``n10`` counts pairs
    with ``a`` correct and ``b`` wrong and ``n01`` the reverse.

    The CI is the Wald interval for a paired difference of proportions,

        se = sqrt(n10 + n01 - (n10 - n01)^2 / n) / n

    **Continuity handling.** With no discordant pairs (``n10 + n01 == 0``) the
    Wald ``se`` collapses to exactly 0 and the CI would be the degenerate point
    ``[0, 0]``, which would declare non-inferiority from zero evidence.  We
    therefore add 0.5 to *each* discordant cell whenever ``n10 + n01 == 0``
    (Haldane-Anscombe style correction), giving ``se = sqrt(1)/n = 1/n`` and a
    CI that shrinks with sample size instead of pretending to be exact.  The flag
    ``continuity_applied`` records when this happened.  The same correction is
    applied when the radicand is non-positive (possible when every pair is
    discordant in one direction).

    Non-inferiority is declared when the lower ``1 - alpha`` two-sided bound is
    strictly greater than ``-margin``.
    """
    if margin < 0.0:
        raise ValueError(f"margin must be non-negative, got {margin}")
    a = np.asarray(list(a_success), dtype=bool)
    bb = np.asarray(list(b_success), dtype=bool)
    if a.size != bb.size:
        raise ValueError(f"paired inputs must have equal length, got {a.size} and {bb.size}")
    n = int(a.size)
    if n == 0:
        return {
            "diff": float("nan"),
            "ci_lo": float("nan"),
            "ci_hi": float("nan"),
            "margin": float(margin),
            "non_inferior": False,
            "n": 0,
            "n10": 0,
            "n01": 0,
            "rate_a": float("nan"),
            "rate_b": float("nan"),
            "continuity_applied": False,
        }

    n10 = int(np.count_nonzero(a & ~bb))
    n01 = int(np.count_nonzero(~a & bb))
    diff = (n10 - n01) / n

    f10, f01 = float(n10), float(n01)
    continuity = False
    radicand = f10 + f01 - (f10 - f01) ** 2 / n
    if (f10 + f01) == 0.0 or radicand <= 0.0:
        continuity = True
        f10 += 0.5
        f01 += 0.5
        radicand = f10 + f01 - (f10 - f01) ** 2 / n
        radicand = max(radicand, 0.0)
    se = math.sqrt(radicand) / n

    z = float(stats.norm.ppf(1.0 - alpha / 2.0))
    ci_lo = diff - z * se
    ci_hi = diff + z * se
    return {
        "diff": float(diff),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "margin": float(margin),
        "non_inferior": bool(ci_lo > -float(margin)),
        "n": n,
        "n10": n10,
        "n01": n01,
        "rate_a": float(np.count_nonzero(a) / n),
        "rate_b": float(np.count_nonzero(bb) / n),
        "se": float(se),
        "alpha": float(alpha),
        "continuity_applied": continuity,
    }


# --------------------------------------------------------------------------- #
# binomial intervals
# --------------------------------------------------------------------------- #


def binom_ci_exact(k: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float, float]:
    """Clopper-Pearson exact interval, returned as ``(point, lo, hi)``.

    Implemented with the Beta quantile identity
    ``lo = Beta(alpha/2; k, n-k+1)`` and ``hi = Beta(1-alpha/2; k+1, n-k)``, with
    the boundary conventions ``k == 0 -> lo = 0`` and ``k == n -> hi = 1``.
    ``n == 0`` gives ``(nan, 0.0, 1.0)``: no information, full interval.
    """
    k = int(k)
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"k must satisfy 0 <= k <= n, got k={k}, n={n}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if n == 0:
        return (float("nan"), 0.0, 1.0)
    point = k / n
    lo = 0.0 if k == 0 else float(stats.beta.ppf(alpha / 2.0, k, n - k + 1))
    hi = 1.0 if k == n else float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))
    return (float(point), lo, hi)


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on an unobserved event rate after ``n`` clean trials.

    ``3 / n`` is the standard rule-of-three approximation to the Clopper-Pearson
    upper limit at ``k = 0``; ``n == 0`` returns 1.0 (no information).
    """
    n = int(n)
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return 1.0
    return min(1.0, 3.0 / n)


# --------------------------------------------------------------------------- #
# outcome aggregation
# --------------------------------------------------------------------------- #


def _episode_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return (str(row.get("prompt_id", "")), int(row.get("seed", 0)))


def _duration(row: Mapping[str, Any]) -> float:
    return float(row.get("tool_duration_ms", 0.0) or 0.0)


def _arm(row: Mapping[str, Any]) -> str:
    return str(row.get("arm", ""))


def _row_key(row: Mapping[str, Any]) -> tuple[str, str]:
    """The ``(arm, policy)`` identity of an outcome row."""
    return (_arm(row), str(row.get("policy", "")))


def _group_gain_rows(
    rows: Sequence[Mapping[str, Any]],
    reference: ArmPolicy,
) -> tuple[
    dict[tuple[str, str, float], list[Mapping[str, Any]]],
    dict[tuple[float, tuple[str, int]], float],
    Counter,
]:
    """Bucket usable rows by ``(arm, policy, duration)`` and index the reference row.

    The same policy replayed against two arms' traces is two different results,
    so the bucket key carries the arm; the reference index is the completion time
    of the declared ``(arm, policy)`` reference row per ``(duration, episode)``.
    """
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    baseline: dict[tuple[float, tuple[str, int]], float] = {}
    excluded: Counter = Counter()
    for row in rows:
        reason = row.get("exclusion")
        if reason:
            excluded[str(reason)] += 1
            continue
        arm, policy = _row_key(row)
        dur = _duration(row)
        groups[(arm, policy, dur)].append(row)
        if (arm, policy) == reference.key:
            comp = float(row.get("completion_ms", float("nan")))
            key = (dur, _episode_key(row))
            if math.isfinite(comp) and comp > 0.0:
                baseline[key] = comp
    return groups, baseline, excluded


def _row_index(
    rows: Sequence[Mapping[str, Any]], target: ArmPolicy
) -> dict[tuple[float, str, int], Mapping[str, Any]]:
    """Index the non-excluded rows of one ``(arm, policy)`` by ``(duration, prompt, seed)``."""
    out: dict[tuple[float, str, int], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("exclusion") or _row_key(row) != target.key:
            continue
        prompt, seed = _episode_key(row)
        out.setdefault((_duration(row), prompt, seed), row)
    return out


def all_retrospective(rows: Sequence[Mapping[str, Any]]) -> bool:
    """True when every launch-bearing outcome in ``rows`` was resolved
    retrospectively from ``analysis.final_calls``.

    `replay` sets `retrospective` per outcome and leaves it ``None`` where the
    policy launched nothing, which carries no information either way. Precision
    computed over retrospective launches cannot fall below 100%, so every
    renderer that prints it must label it.
    """
    flags = [bool(r["retrospective"]) for r in rows if r.get("retrospective") is not None]
    return bool(flags) and all(flags)


def verification_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Per-claim verification coverage over a set of outcome rows.

    ``HostToolRouter(verify_claims=True)`` re-executes every claimed speculative
    result normally and compares result digests, so each claim ends up in exactly
    one bucket: ``verified_match``, ``verified_divergent``, ``verification_error``
    (the verifier itself raised) or ``unverified``.  ``claims_unaccounted`` is the
    residual ``claimed - (match + divergent + error + unverified)``: a set whose
    counters do not add up is a coverage hole, not a pass.

    **Counters are per source execution, not per row.**  Replay emits one row per
    (trajectory, policy, tool duration) and copies the trajectory's claim
    counters into each, so summing rows would multiply the evidence by the number
    of policies and latency conditions and spuriously tighten the rule-of-three
    bound.  Rows are therefore deduplicated on ``(arm, trajectory_id)``; when two
    rows for the same execution disagree, the disagreement is reported in
    ``inconsistent_executions`` and the larger claim count is kept so the
    coverage check still fails closed.
    """
    out = {
        "rows": 0,
        "episodes": 0,
        "claims_claimed": 0,
        "claims_verified_match": 0,
        "claims_verified_divergent": 0,
        "claims_verification_error": 0,
        "claims_unverified": 0,
        "episodes_verification_incomplete": 0,
        "episodes_flagged_divergent": 0,
        "inconsistent_executions": 0,
    }
    keys = (
        "claims_claimed",
        "claims_verified_match",
        "claims_verified_divergent",
        "claims_verification_error",
        "claims_unverified",
    )
    by_execution: dict[tuple[str, str], dict[str, Any]] = {}
    inconsistent: set[tuple[str, str]] = set()
    # Worst-case reconciliation, so the result is independent of row order and
    # always fails closed: the most claims, the fewest verified matches, and the
    # most divergences, errors and unverified claims that any row reported.
    worst = {
        "claims_claimed": max,
        "claims_verified_match": min,
        "claims_verified_divergent": max,
        "claims_verification_error": max,
        "claims_unverified": max,
    }
    for row in rows:
        out["rows"] += 1
        ident = (
            str(row.get("arm", "")),
            str(row.get("trajectory_id") or row.get("request_id") or row.get("prompt_id", "")),
        )
        current = {k: int(row.get(k, 0) or 0) for k in keys}
        current["verification_complete"] = bool(row.get("verification_complete", False))
        current["semantic_divergence"] = bool(row.get("semantic_divergence"))
        previous = by_execution.get(ident)
        if previous is None:
            by_execution[ident] = current
            continue
        if any(previous[k] != current[k] for k in keys):
            inconsistent.add(ident)
        for k, pick in worst.items():
            previous[k] = pick(previous[k], current[k])
        previous["verification_complete"] = (
            previous["verification_complete"] and current["verification_complete"]
        )
        previous["semantic_divergence"] = (
            previous["semantic_divergence"] or current["semantic_divergence"]
        )
    for rec in by_execution.values():
        out["episodes"] += 1
        for key in keys:
            out[key] += rec[key]
        if not rec["verification_complete"]:
            out["episodes_verification_incomplete"] += 1
        if rec["semantic_divergence"]:
            out["episodes_flagged_divergent"] += 1
    out["inconsistent_executions"] = len(inconsistent)
    out["claims_verified"] = out["claims_verified_match"] + out["claims_verified_divergent"]
    out["claims_unaccounted"] = max(
        0,
        out["claims_claimed"]
        - out["claims_verified"]
        - out["claims_verification_error"]
        - out["claims_unverified"],
    )
    return out


def external_divergence_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Divergence tally of an external replay check (e.g. ToolSandbox episodes).

    ``semantic_divergence`` is a count (or bool) per episode; excluded episodes
    (``exclusion`` set) never ran and are reported, not counted as clean.
    """
    episodes = 0
    divergent = 0
    excluded = 0
    speculative = 0
    for row in rows or ():
        if row.get("exclusion"):
            excluded += 1
            continue
        episodes += 1
        if row.get("speculate"):
            speculative += 1
        divergent += int(row.get("semantic_divergence", 0) or 0)
    return {
        "episodes": episodes,
        "speculative_episodes": speculative,
        "divergent": divergent,
        "excluded_episodes": excluded,
    }


def observed_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """What the ONLINE future table actually did, de-duplicated per request.

    ``observed_*`` is a property of the recorded request, so the same trajectory
    replayed at four tool durations must be counted once, not four times.
    """
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        if row.get("exclusion"):
            continue
        seen.setdefault(execution_id(row), row)
    launched = claimed = wasted = with_obs = 0
    for row in seen.values():
        if row.get("observed_launched") is None:
            continue
        with_obs += 1
        launched += int(row.get("observed_launched") or 0)
        claimed += int(row.get("observed_claimed") or 0)
        wasted += int(row.get("observed_wasted") or 0)
    return {
        "requests": len(seen),
        "requests_with_observation": with_obs,
        "observed_launched": launched,
        "observed_claimed": claimed,
        "observed_wasted": wasted,
    }


def replay_fidelity(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: ArmPolicy | tuple[str, str] | Mapping[str, Any] | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Did replaying the frozen gate reproduce what the online runtime did?

    ``observed_matches_replay`` is ``None`` for any policy that is not that arm's
    own frozen gate -- a different policy is *expected* to differ -- so those
    rows are counted in ``not_comparable`` and never as a match.  A mismatch is a
    bug signal, not a result: every mismatching ``(arm, trajectory_id)`` is listed
    verbatim instead of being averaged away.

    When ``candidate`` is given, the online and replayed launch precision of that
    row are reported side by side so a reader sees the measured number next to
    the re-derived one.
    """
    compared: dict[tuple[str, str, str], bool] = {}
    mismatch_detail: dict[tuple[str, str, str], dict[str, Any]] = {}
    not_comparable_rows = 0
    compared_rows = 0
    for row in rows:
        if row.get("exclusion"):
            continue
        flag = row.get("observed_matches_replay")
        if flag is None:
            not_comparable_rows += 1
            continue
        compared_rows += 1
        arm, traj = execution_id(row)
        key = (arm, traj, str(row.get("policy", "")))
        compared[key] = bool(compared.get(key, True)) and bool(flag)
        if not flag:
            mismatch_detail.setdefault(
                key,
                {
                    "arm": arm,
                    "trajectory_id": traj,
                    "policy": str(row.get("policy", "")),
                    "observed_launched": row.get("observed_launched"),
                    "replayed_launches": int(row.get("launches", 0) or 0),
                    "observed_claimed": row.get("observed_claimed"),
                    "replayed_claimed": int(row.get("claimed", 0) or 0),
                    "observed_wasted": row.get("observed_wasted"),
                    "replayed_wasted": int(row.get("false_launches", 0) or 0),
                    "tool_duration_ms": _duration(row),
                },
            )
    matching = sum(1 for ok in compared.values() if ok)
    mismatching = len(compared) - matching

    out: dict[str, Any] = {
        "compared": len(compared),
        "matching": matching,
        "mismatching": mismatching,
        "not_comparable": not_comparable_rows,
        "compared_rows": compared_rows,
        "mismatching_keys": [[k[0], k[1]] for k in sorted(mismatch_detail)],
        "mismatching_rows": [mismatch_detail[k] for k in sorted(mismatch_detail)],
        "clean": mismatching == 0,
    }
    if candidate is not None:
        cand = ArmPolicy.coerce(candidate)
        cand_rows = [r for r in rows if not r.get("exclusion") and _row_key(r) == cand.key]
        obs = observed_totals(cand_rows)
        on_point, on_lo, on_hi = binom_ci_exact(
            min(obs["observed_claimed"], obs["observed_launched"]),
            obs["observed_launched"],
            alpha,
        )
        # Launch counts belong to the source execution and are copied into
        # every simulated latency row: deduplicate before the exact interval,
        # exactly as `observed_totals` already does.
        cand_exec_rows = dedupe_executions(cand_rows)
        true_l = sum(int(r.get("true_launches", 0) or 0) for r in cand_exec_rows)
        false_l = sum(int(r.get("false_launches", 0) or 0) for r in cand_exec_rows)
        re_point, re_lo, re_hi = binom_ci_exact(true_l, true_l + false_l, alpha)
        delta = (
            on_point - re_point
            if math.isfinite(on_point) and math.isfinite(re_point)
            else float("nan")
        )
        out["candidate"] = cand.to_json()
        out["candidate_observed"] = obs
        out["candidate_precision_online"] = {
            "k": obs["observed_claimed"],
            "n": obs["observed_launched"],
            "point": on_point,
            "ci_lo": on_lo,
            "ci_hi": on_hi,
            "definition": "observed_claimed / observed_launched, from the online future table",
        }
        out["candidate_precision_replayed"] = {
            "k": true_l,
            "n": true_l + false_l,
            "point": re_point,
            "ci_lo": re_lo,
            "ci_hi": re_hi,
            "executions": len(cand_exec_rows),
            "definition": "true_launches / (true_launches + false_launches), re-derived offline",
        }
        out["candidate_precision_delta"] = delta
    return out


def instrumentation_totals(
    rows: Sequence[Mapping[str, Any]],
    ps: Sequence[int | float] = DEFAULT_PERCENTILES,
) -> dict[str, Any]:
    """In-timeline instrumentation cost, de-duplicated per request.

    ``instrumentation_in_timeline_ms`` is the trace+parse cost the replayed
    decode timeline already contains.  It is NOT charged again to completion and
    is NOT credited back to the reference arm: every arm replays the same
    instrumented timeline, so the cost cancels in every paired comparison and is
    reported on its own here.
    """
    seen: dict[tuple[str, str], float] = {}
    per_arm: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("exclusion"):
            continue
        val = row.get("instrumentation_in_timeline_ms")
        if val is None:
            continue
        key = execution_id(row)
        if key in seen:
            continue
        fval = float(val)
        if not math.isfinite(fval):
            continue
        seen[key] = fval
        per_arm[key[0]].append(fval)
    values = [seen[k] for k in sorted(seen)]
    return {
        "requests": len(values),
        "in_timeline_ms": percentiles(values, ps),
        "in_timeline_ms_total": float(sum(values)),
        "by_arm": {arm: percentiles(sorted(v), ps) for arm, v in sorted(per_arm.items())},
        "charged_to_completion": False,
        "note": (
            "already inside the replayed decode timeline shared by every arm; never "
            "charged twice and never credited back to the reference arm"
        ),
    }


def paired_gain(
    rows: Sequence[Mapping[str, Any]],
    candidate: ArmPolicy | tuple[str, str] | Mapping[str, Any],
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any],
    *,
    duration: float | None = None,
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired relative latency gain of ``candidate`` over ``reference``.

    A pair is one ``(tool_duration_ms, prompt_id, seed)`` episode present under
    both ``(arm, policy)`` rows; the gain is
    ``1 - completion_candidate / completion_reference``.  With
    ``duration=None`` every duration is pooled (each duration contributes its own
    pairs, clustered by ``prompt_id``).  Reported: the median with a
    prompt-clustered bootstrap CI, the Wilcoxon signed-rank p-value and the
    sign-flip randomisation p-value on the raw completion times, plus the number
    of episodes that could not be paired.
    """
    cand_row = ArmPolicy.coerce(candidate)
    ref_row = ArmPolicy.coerce(reference)
    cand = _row_index(rows, cand_row)
    ref = _row_index(rows, ref_row)
    if duration is not None:
        cand = {k: v for k, v in cand.items() if k[0] == duration}
        ref = {k: v for k, v in ref.items() if k[0] == duration}

    shared = sorted(set(cand) & set(ref))
    gains_by_prompt: dict[str, list[float]] = defaultdict(list)
    gains: list[float] = []
    a_ms: list[float] = []
    b_ms: list[float] = []
    unusable = 0
    for key in shared:
        c = float(cand[key].get("completion_ms", float("nan")))
        r = float(ref[key].get("completion_ms", float("nan")))
        if not (math.isfinite(c) and math.isfinite(r) and r > 0.0):
            unusable += 1
            continue
        gains.append(1.0 - c / r)
        gains_by_prompt[key[1]].append(1.0 - c / r)
        a_ms.append(c)
        b_ms.append(r)

    point, lo, hi = bootstrap_ci_clustered(gains_by_prompt, None, n_boot, seed, alpha)
    return {
        "candidate": cand_row.to_json(),
        "reference": ref_row.to_json(),
        "tool_duration_ms": duration,
        "n_paired": len(gains),
        "clusters": len(gains_by_prompt),
        "unpaired_candidate": len(set(cand) - set(ref)),
        "unpaired_reference": len(set(ref) - set(cand)),
        "unusable_pairs": unusable,
        "median": point,
        "ci_lo": lo,
        "ci_hi": hi,
        "mean": float(np.mean(gains)) if gains else float("nan"),
        "wilcoxon": wilcoxon_paired(a_ms, b_ms),
        "randomization": paired_randomization(a_ms, b_ms, n_boot, seed),
    }


def _paired_success(
    rows: Sequence[Mapping[str, Any]],
    candidate: ArmPolicy,
    reference: ArmPolicy,
) -> tuple[list[bool], list[bool], int]:
    """De-duplicated per-episode success pairs (correctness is duration-invariant)."""
    cand = _row_index(rows, candidate)
    ref = _row_index(rows, reference)
    a: dict[tuple[str, int], bool] = {}
    b: dict[tuple[str, int], bool] = {}
    unknown = 0
    for key in sorted(set(cand) & set(ref)):
        ep = (key[1], key[2])
        ca = cand[key].get("correct")
        cb = ref[key].get("correct")
        if ca is None or cb is None:
            unknown += 1
            continue
        a.setdefault(ep, bool(ca))
        b.setdefault(ep, bool(cb))
    eps = sorted(a)
    return [a[e] for e in eps], [b[e] for e in eps], unknown


def _gate_row(spec: Mapping[str, Any], **fields: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": spec["id"],
        "key": spec["key"],
        "description": spec["description"],
        "statistic": spec["statistic"],
        "comparator": spec["comparator"],
        "required_threshold": spec["threshold"],
        "required": bool(spec["required"]),
        "decisive": bool(spec.get("decisive", False)),
        "point": float("nan"),
        "ci_lo": float("nan"),
        "ci_hi": float("nan"),
        "n": 0,
        "pass": False,
        "status": "present",
        "detail": {},
    }
    if "target_band" in spec:
        row["target_band"] = list(spec["target_band"])
    row.update(fields)
    return row


def execution_id(row: Mapping[str, Any]) -> tuple[str, str]:
    """Identity of the source model execution a replayed row came from."""
    return (
        str(row.get("arm", "")),
        str(row.get("trajectory_id") or row.get("request_id") or row.get("prompt_id", "")),
    )


def dedupe_executions(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """One row per source execution, keeping the first occurrence.

    Replay emits one row per (trajectory, policy, tool duration) and copies the
    execution's *count* fields -- ``true_launches``, ``false_launches``,
    ``claimed``, ``eligible_final_calls``, ``covered_final_calls`` and the claim
    counters -- unchanged into every row, because they describe what the model
    and the gate did once, not what the latency simulation did.  Summing rows
    would therefore multiply every Clopper-Pearson denominator by the number of
    latency conditions and report spuriously narrow intervals.

    Rate-and-time fields (``completion_ms``, gains) legitimately vary per
    duration and must NOT be deduplicated; use the full row set for those.
    """
    seen: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        seen.setdefault(execution_id(row), row)
    return list(seen.values())


def _row_profile(
    rows: Sequence[Mapping[str, Any]],
    target: ArmPolicy,
    reference: ArmPolicy,
    *,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Latency / risk profile of one ``(arm, policy)`` row, for the S8 Pareto view."""
    arm_rows = [r for r in rows if not r.get("exclusion") and _row_key(r) == target.key]
    # Count fields describe one model execution and are repeated across latency
    # rows; deduplicate before any exact-binomial interval.
    exec_rows = dedupe_executions(arm_rows)
    true_l = sum(int(r.get("true_launches", 0) or 0) for r in exec_rows)
    false_l = sum(int(r.get("false_launches", 0) or 0) for r in exec_rows)
    prec, prec_lo, prec_hi = binom_ci_exact(true_l, true_l + false_l)
    wasted = sum(float(r.get("wasted_tool_ms", 0.0) or 0.0) for r in arm_rows)
    gain = paired_gain(rows, target, reference, n_boot=n_boot, seed=seed)
    return {
        "row": target.to_json(),
        "arm": target.arm,
        "policy": target.policy,
        "n": len(arm_rows),
        "executions": len(exec_rows),
        "completion_ms": percentiles([r.get("completion_ms") for r in arm_rows]),
        "precision": {
            "k": true_l,
            "n": true_l + false_l,
            "point": prec,
            "ci_lo": prec_lo,
            "ci_hi": prec_hi,
        },
        "wasted_tool_ms_per_episode": wasted / len(arm_rows) if arm_rows else float("nan"),
        "median_gain_vs_reference": gain["median"],
        "median_gain_ci": [gain["ci_lo"], gain["ci_hi"]],
    }


def submission_gates(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: ArmPolicy | tuple[str, str] | Mapping[str, Any] = CANDIDATE_ROW,
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any] = REFERENCE_ROW,
    decisive: Sequence[ArmPolicy | tuple[str, str] | Mapping[str, Any]] = DECISIVE_ROWS,
    verification: Sequence[ArmPolicy | tuple[str, str] | Mapping[str, Any]] = (
        VERIFICATION_ROW,
    ),
    external_divergence_rows: Sequence[Mapping[str, Any]] = (),
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    margin: float = NONINFERIORITY_MARGIN,
) -> dict[str, Any]:
    """Evaluate the S1-S8 submission gates on policy-outcome rows.

    ``candidate``, ``reference`` and ``decisive`` are ``(arm, policy)`` rows —
    normally taken straight from ``configs/experiments/main_table.yaml``
    (``reference_row`` and ``decisive_rows``), never hardcoded policy names.
    ``decisive[0]`` drives S7 (strongest non-speculative diffusion baseline) and
    ``decisive[1]`` drives S8 (the P-ReAct-style arm).

    Each returned gate row carries ``id``, ``key``, ``description``,
    ``required_threshold``, ``point``, ``ci_lo``/``ci_hi``, ``n``, ``pass`` and
    ``required``.  A gate passes only when its **point estimate** meets the
    threshold; the CI is always reported alongside (S7 additionally requires the
    CI to exclude 0).  Missing inputs give ``status="missing"`` and never pass.

    S7 and S8 are the decisive pair: a gain over the reference row alone
    (S3/S4) is scientifically insufficient, so S3/S4 are never reported without
    S7 beside them.
    """
    specs = {s["id"]: s for s in SUBMISSION_GATE_SPECS}
    cand_row = ArmPolicy.coerce(candidate)
    ref_row = ArmPolicy.coerce(reference)
    decisive_rows = [ArmPolicy.coerce(d) for d in decisive]
    s7_row = decisive_rows[0] if decisive_rows else None
    s8_row = decisive_rows[1] if len(decisive_rows) > 1 else None
    cand_rows = [r for r in rows if not r.get("exclusion") and _row_key(r) == cand_row.key]
    gates: list[dict[str, Any]] = []

    # ---- S1 speculative precision ------------------------------------------
    # Launch counts belong to the source execution, not to a latency condition.
    cand_exec_rows = dedupe_executions(cand_rows)
    true_l = sum(int(r.get("true_launches", 0) or 0) for r in cand_exec_rows)
    false_l = sum(int(r.get("false_launches", 0) or 0) for r in cand_exec_rows)
    n_launch = true_l + false_l
    p1, lo1, hi1 = binom_ci_exact(true_l, n_launch, alpha)
    gates.append(
        _gate_row(
            specs["S1"],
            point=p1,
            ci_lo=lo1,
            ci_hi=hi1,
            n=n_launch,
            status="present" if n_launch else "missing",
            **{"pass": bool(n_launch and math.isfinite(p1) and p1 >= specs["S1"]["threshold"])},
            detail={
                "true_launches": true_l,
                "false_launches": false_l,
                "claimed": sum(int(r.get("claimed", 0) or 0) for r in cand_exec_rows),
                "executions": len(cand_exec_rows),
                "row": cand_row.to_json(),
            },
        )
    )

    # ---- S2 eligible-call coverage -----------------------------------------
    covered = sum(int(r.get("covered_final_calls", 0) or 0) for r in cand_exec_rows)
    eligible = sum(int(r.get("eligible_final_calls", 0) or 0) for r in cand_exec_rows)
    p2, lo2, hi2 = binom_ci_exact(min(covered, eligible), eligible, alpha)
    gates.append(
        _gate_row(
            specs["S2"],
            point=p2,
            ci_lo=lo2,
            ci_hi=hi2,
            n=eligible,
            status="present" if eligible else "missing",
            **{"pass": bool(eligible and math.isfinite(p2) and p2 >= specs["S2"]["threshold"])},
            detail={
                "covered_final_calls": covered,
                "eligible_final_calls": eligible,
                "executions": len(cand_exec_rows),
            },
        )
    )

    # ---- S3 / S4 gain vs the post-response baseline -------------------------
    for gid, dur in (("S3", 500.0), ("S4", 2000.0)):
        spec = specs[gid]
        g = paired_gain(
            rows,
            cand_row,
            ref_row,
            duration=dur,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
        ok = bool(g["n_paired"] and math.isfinite(g["median"]) and g["median"] >= spec["threshold"])
        band = spec["target_band"]
        gates.append(
            _gate_row(
                spec,
                point=g["median"],
                ci_lo=g["ci_lo"],
                ci_hi=g["ci_hi"],
                n=g["n_paired"],
                status="present" if g["n_paired"] else "missing",
                **{"pass": ok},
                detail={
                    "tool_duration_ms": dur,
                    "reference": ref_row.to_json(),
                    "wilcoxon_p": g["wilcoxon"]["p_value"],
                    "randomization_p": g["randomization"]["p_value"],
                    "in_target_band": bool(
                        math.isfinite(g["median"]) and band[0] <= g["median"] <= band[1]
                    ),
                    "clusters": g["clusters"],
                    "unpaired_candidate": g["unpaired_candidate"],
                    "unpaired_reference": g["unpaired_reference"],
                    "decisive_companion": "S7",
                },
            )
        )

    # ---- S5 eventual-success non-inferiority --------------------------------
    a_ok, b_ok, unknown = _paired_success(rows, cand_row, ref_row)
    ni = noninferiority_success(a_ok, b_ok, margin, alpha)
    mc = mcnemar_exact(ni["n10"], ni["n01"])
    gates.append(
        _gate_row(
            specs["S5"],
            point=ni["diff"],
            ci_lo=ni["ci_lo"],
            ci_hi=ni["ci_hi"],
            n=ni["n"],
            status="present" if ni["n"] else "missing",
            **{"pass": bool(ni["non_inferior"])},
            detail={
                "margin": ni["margin"],
                "rate_candidate": ni["rate_a"],
                "rate_reference": ni["rate_b"],
                "n10": ni["n10"],
                "n01": ni["n01"],
                "continuity_applied": ni["continuity_applied"],
                "mcnemar_p": mc["p_value"],
                "mcnemar_discordant": mc["n_discordant"],
                "correctness_unknown_pairs": unknown,
            },
        )
    )

    # ---- S6 zero semantic divergence, fail-closed on verification coverage ---
    # Every claimed speculative result is re-executed normally by
    # HostToolRouter(verify_claims=True) and the two result digests are compared.
    # A verifier error is NOT evidence of losslessness, and neither is a claim
    # nobody verified, so S6 passes only when the verified set covers every claim.
    # S6's evidence is the LIVE arm, not the derived candidate: only a real
    # speculative run produces claim counters.  Missing validation rows are a
    # fail-closed condition, never a silent pass.
    ver_keys = {ArmPolicy.coerce(v).key for v in (verification or ())}
    ver_rows = [r for r in rows if not r.get("exclusion") and _row_key(r) in ver_keys]
    v = verification_totals(ver_rows)
    ext = external_divergence_totals(external_divergence_rows)
    verified = v["claims_verified"]
    divergences = v["claims_verified_divergent"] + ext["divergent"]
    p6, lo6, hi6 = binom_ci_exact(
        min(v["claims_verified_divergent"], verified), verified, alpha
    )
    fail_reasons: list[str] = []
    if not ver_rows:
        fail_reasons.append(
            "no online validation episodes for "
            + ", ".join(sorted(f"{a}/{p}" for a, p in ver_keys))
            + ": claim verification evidence is absent, so losslessness is not demonstrated"
        )
    if v["claims_claimed"] == 0:
        fail_reasons.append("no speculative claim was made: losslessness is not demonstrated")
    if v["claims_verified_divergent"]:
        fail_reasons.append(f"{v['claims_verified_divergent']} verified claim(s) diverged")
    if v["claims_verification_error"]:
        fail_reasons.append(
            f"{v['claims_verification_error']} claim(s) the verifier could not check"
        )
    if v["claims_unverified"] or v["claims_unaccounted"]:
        fail_reasons.append(
            f"{v['claims_unverified'] + v['claims_unaccounted']} claim(s) never verified"
        )
    if v["episodes_verification_incomplete"]:
        fail_reasons.append(
            f"{v['episodes_verification_incomplete']} episode(s) with incomplete verification"
        )
    if v["episodes_flagged_divergent"] and not v["claims_verified_divergent"]:
        fail_reasons.append(
            f"{v['episodes_flagged_divergent']} episode(s) flagged semantic_divergence"
            " without a per-claim count"
        )
    if ext["divergent"]:
        fail_reasons.append(f"{ext['divergent']} replay episode(s) diverged")
    gates.append(
        _gate_row(
            specs["S6"],
            point=float(divergences),
            ci_lo=lo6,
            ci_hi=hi6,
            n=verified,
            status="present" if ver_rows else "missing",
            **{"pass": not fail_reasons},
            detail={
                "divergences": divergences,
                "verification_rows": sorted(f"{a}/{p}" for a, p in ver_keys),
                "episodes": len(ver_rows),
                "claims_claimed": v["claims_claimed"],
                "claims_verified_match": v["claims_verified_match"],
                "claims_verified_divergent": v["claims_verified_divergent"],
                "claims_verification_error": v["claims_verification_error"],
                "claims_unverified": v["claims_unverified"],
                "claims_unaccounted": v["claims_unaccounted"],
                "claims_verified": verified,
                "episodes_verification_incomplete": v["episodes_verification_incomplete"],
                "episodes_flagged_divergent": v["episodes_flagged_divergent"],
                "rate_point": p6,
                # The bound is over claims actually VERIFIED, never over claims launched.
                "rule_of_three_upper": rule_of_three_upper(verified),
                "rule_of_three_denominator": "claims_verified_match + claims_verified_divergent",
                "external_replay": ext,
                "fail_reasons": fail_reasons,
            },
        )
    )

    # ---- S7 decisive: advantage over the first decisive row -----------------
    spec7 = specs["S7"]
    if s7_row is None:
        gates.append(
            _gate_row(
                spec7,
                status="missing",
                detail={"reason": "no decisive row declared for S7", "pareto": None},
            )
        )
    else:
        g7 = paired_gain(
            rows,
            cand_row,
            s7_row,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
        per_duration = [
            paired_gain(
                rows,
                cand_row,
                s7_row,
                duration=d,
                n_boot=n_boot,
                seed=seed,
                alpha=alpha,
            )
            for d in sorted({_duration(r) for r in rows if _row_key(r) == cand_row.key})
        ]
        ok7 = bool(
            g7["n_paired"]
            and math.isfinite(g7["median"])
            and g7["median"] > spec7["threshold"]
            and math.isfinite(g7["ci_lo"])
            and g7["ci_lo"] > spec7["threshold"]
        )
        gates.append(
            _gate_row(
                spec7,
                point=g7["median"],
                ci_lo=g7["ci_lo"],
                ci_hi=g7["ci_hi"],
                n=g7["n_paired"],
                status="present" if g7["n_paired"] else "missing",
                **{"pass": ok7},
                detail={
                    "reference": s7_row.to_json(),
                    "wilcoxon_p": g7["wilcoxon"]["p_value"],
                    "randomization_p": g7["randomization"]["p_value"],
                    "ci_excludes_zero": bool(math.isfinite(g7["ci_lo"]) and g7["ci_lo"] > 0.0),
                    "clusters": g7["clusters"],
                    "unpaired_candidate": g7["unpaired_candidate"],
                    "unpaired_reference": g7["unpaired_reference"],
                    "per_duration": [
                        {
                            "tool_duration_ms": d["tool_duration_ms"],
                            "median": d["median"],
                            "ci_lo": d["ci_lo"],
                            "ci_hi": d["ci_hi"],
                            "n_paired": d["n_paired"],
                        }
                        for d in per_duration
                    ],
                },
            )
        )

    # ---- S8 preferred: advantage over the second decisive row ---------------
    spec8 = specs["S8"]
    if s8_row is None:
        gates.append(
            _gate_row(
                spec8,
                status="missing",
                detail={
                    "reason": "no second decisive row (P-ReAct-style arm) declared",
                    "pareto": None,
                },
            )
        )
    elif not any(not r.get("exclusion") and _row_key(r) == s8_row.key for r in rows):
        gates.append(
            _gate_row(
                spec8,
                status="missing",
                detail={
                    "reason": (
                        "no outcomes for the declared P-ReAct-style row "
                        f"{s8_row.arm}/{s8_row.policy}"
                    ),
                    "reference": s8_row.to_json(),
                    "pareto": None,
                },
            )
        )
    else:
        g8 = paired_gain(rows, cand_row, s8_row, n_boot=n_boot, seed=seed, alpha=alpha)
        ok8 = bool(
            g8["n_paired"] and math.isfinite(g8["median"]) and g8["median"] >= spec8["threshold"]
        )
        detail: dict[str, Any] = {
            "reference": s8_row.to_json(),
            "wilcoxon_p": g8["wilcoxon"]["p_value"],
            "randomization_p": g8["randomization"]["p_value"],
            "ci_excludes_zero": bool(math.isfinite(g8["ci_lo"]) and g8["ci_lo"] > 0.0),
            "clusters": g8["clusters"],
            "unpaired_candidate": g8["unpaired_candidate"],
            "unpaired_reference": g8["unpaired_reference"],
        }
        if not ok8:
            # Preferred gate: when the 10% gain is not reached, the claim becomes a
            # latency/risk trade-off, so report the Pareto comparison instead of a
            # bare boolean.
            detail["pareto"] = {
                "candidate": _row_profile(
                    rows, cand_row, ref_row, n_boot=n_boot, seed=seed
                ),
                "reference": _row_profile(rows, s8_row, ref_row, n_boot=n_boot, seed=seed),
            }
        gates.append(
            _gate_row(
                spec8,
                point=g8["median"],
                ci_lo=g8["ci_lo"],
                ci_hi=g8["ci_hi"],
                n=g8["n_paired"],
                status="present" if g8["n_paired"] else "missing",
                **{"pass": ok8},
                detail=detail,
            )
        )

    required = [g for g in gates if g["required"]]
    return {
        "candidate": cand_row.to_json(),
        "reference": ref_row.to_json(),
        "decisive_rows": [d.to_json() for d in decisive_rows],
        "margin": float(margin),
        "alpha": float(alpha),
        "bootstrap": {"n": int(n_boot), "seed": int(seed), "cluster": "prompt_id"},
        "gates": gates,
        "by_id": {g["id"]: g["pass"] for g in gates},
        "required_passed": sum(1 for g in required if g["pass"]),
        "required_total": len(required),
        "all_required_pass": all(g["pass"] for g in required) if required else False,
        "preferred_passed": sum(1 for g in gates if not g["required"] and g["pass"]),
        "decisive": {g["id"]: g["pass"] for g in gates if g["decisive"]},
    }


def summarize_outcomes(
    rows: Sequence[Mapping[str, Any]],
    *,
    reference: ArmPolicy | tuple[str, str] | Mapping[str, Any] = REFERENCE_ROW,
    candidate: ArmPolicy | tuple[str, str] | Mapping[str, Any] = CANDIDATE_ROW,
    decisive: Sequence[ArmPolicy | tuple[str, str] | Mapping[str, Any]] = DECISIVE_ROWS,
    verification: Sequence[ArmPolicy | tuple[str, str] | Mapping[str, Any]] = (
        VERIFICATION_ROW,
    ),
    external_divergence_rows: Sequence[Mapping[str, Any]] = (),
    n_boot: int = DEFAULT_BOOTSTRAP,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    percentile_points: Sequence[int | float] = DEFAULT_PERCENTILES,
    gates: bool = True,
) -> dict[str, Any]:
    """Aggregate ``PolicyOutcome.to_json()`` rows into the main results table.

    Rows are grouped by ``(arm, policy, tool_duration_ms)`` -- the same offline
    policy replayed against two arms' traces is two different results and is
    never pooled.  Every group reports:

    * ``completion_ms`` p50/p95/p99,
    * the **median paired relative gain** ``1 - completion / reference`` against
      the declared ``reference`` ``(arm, policy)`` row for the *same*
      ``(prompt_id, seed)`` and the same tool duration, with a prompt-clustered
      bootstrap CI,
    * launch counts, speculative precision (with an exact binomial CI),
      eligible-call coverage, wasted / cancelled / duplicate work, hidden
      overlap and the semantic-divergence count.

    Episodes with an ``exclusion`` reason are removed and counted in
    ``exclusions``; rows without a usable baseline partner are counted in
    ``gain.unpaired`` and totalled in ``accounting.unpaired_episodes`` (never
    silently dropped).  Unless ``gates=False``, the S1-S8 submission gate block
    (see :func:`submission_gates`) is attached under ``submission_gates``.
    """
    rows = list(rows)
    ref_row = ArmPolicy.coerce(reference)
    cand_row = ArmPolicy.coerce(candidate)
    groups, baseline, excluded = _group_gain_rows(rows, ref_row)

    out_groups: list[dict[str, Any]] = []
    for arm, policy, dur in sorted(groups, key=lambda kv: (kv[0], kv[1], kv[2])):
        grp = groups[(arm, policy, dur)]
        comps = [float(r.get("completion_ms", float("nan"))) for r in grp]

        gains_by_prompt: dict[str, list[float]] = defaultdict(list)
        gains: list[float] = []
        unpaired = 0
        unpaired_examples: list[str] = []
        for r in grp:
            ekey = _episode_key(r)
            base = baseline.get((dur, ekey))
            comp = float(r.get("completion_ms", float("nan")))
            if base is None or not math.isfinite(comp):
                unpaired += 1
                if len(unpaired_examples) < 8:
                    unpaired_examples.append(f"{ekey[0]}#{ekey[1]}")
                continue
            gain = 1.0 - comp / base
            gains.append(gain)
            gains_by_prompt[ekey[0]].append(gain)
        point, lo, hi = bootstrap_ci_clustered(
            gains_by_prompt, None, n_boot, seed, alpha
        )

        true_l = int(sum(int(r.get("true_launches", 0) or 0) for r in grp))
        false_l = int(sum(int(r.get("false_launches", 0) or 0) for r in grp))
        launches = int(sum(int(r.get("launches", 0) or 0) for r in grp))
        claimed = int(sum(int(r.get("claimed", 0) or 0) for r in grp))
        duplicate = int(
            sum(
                max(
                    0,
                    int(r.get("launches", 0) or 0)
                    - int(r.get("claimed", 0) or 0)
                    - int(r.get("false_launches", 0) or 0),
                )
                for r in grp
            )
        )
        prec_point, prec_lo, prec_hi = binom_ci_exact(true_l, true_l + false_l, alpha)

        eligible = int(sum(int(r.get("eligible_final_calls", 0) or 0) for r in grp))
        covered = int(sum(int(r.get("covered_final_calls", 0) or 0) for r in grp))
        cov_point, cov_lo, cov_hi = binom_ci_exact(min(covered, eligible), eligible, alpha)

        correct_true = sum(1 for r in grp if r.get("correct") is True)
        correct_false = sum(1 for r in grp if r.get("correct") is False)
        correct_unknown = sum(1 for r in grp if r.get("correct") is None)
        divergences = sum(1 for r in grp if bool(r.get("semantic_divergence")))

        out_groups.append(
            {
                "arm": arm,
                "policy": policy,
                "row_key": f"{arm}/{policy}",
                "tool_duration_ms": dur,
                "n": len(grp),
                "n_prompts": len({_episode_key(r)[0] for r in grp}),
                "n_seeds": len({_episode_key(r)[1] for r in grp}),
                "completion_ms": percentiles(comps, percentile_points),
                "gain": {
                    "n_paired": len(gains),
                    "unpaired": unpaired,
                    "unpaired_examples": unpaired_examples,
                    "median": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "mean": float(np.mean(gains)) if gains else float("nan"),
                    "clusters": len(gains_by_prompt),
                },
                "launches": {
                    "total": launches,
                    "true": true_l,
                    "false": false_l,
                    "claimed": claimed,
                    "duplicate": duplicate,
                    "per_episode": launches / len(grp) if grp else float("nan"),
                },
                # Data-driven from the outcome records (see `replay.PolicyOutcome`):
                # a retrospective policy resolves launches from
                # `analysis.final_calls`, so this precision is 100% by
                # construction and every renderer must label it.
                "retrospective": all_retrospective(grp),
                "precision": {
                    "k": true_l,
                    "n": true_l + false_l,
                    "point": prec_point,
                    "ci_lo": prec_lo,
                    "ci_hi": prec_hi,
                },
                "coverage": {
                    "k": covered,
                    "n": eligible,
                    "point": cov_point,
                    "ci_lo": cov_lo,
                    "ci_hi": cov_hi,
                },
                "waste": {
                    "wasted_tool_ms_total": float(
                        sum(float(r.get("wasted_tool_ms", 0.0) or 0.0) for r in grp)
                    ),
                    "wasted_tool_ms_median": _median(
                        _as_float_array([r.get("wasted_tool_ms", 0.0) or 0.0 for r in grp])
                    ),
                    "cancelled_launches": false_l,
                    "duplicate_launches": duplicate,
                },
                "hidden_overlap_ms": percentiles(
                    [r.get("hidden_overlap_ms", 0.0) or 0.0 for r in grp], percentile_points
                ),
                "semantic_divergence": int(divergences),
                "verification": verification_totals(grp),
                "instrumentation_in_timeline_ms": percentiles(
                    [
                        r.get("instrumentation_in_timeline_ms")
                        for r in grp
                        if r.get("instrumentation_in_timeline_ms") is not None
                    ],
                    percentile_points,
                ),
                "observed": observed_totals(grp),
                "correct": {
                    "true": int(correct_true),
                    "false": int(correct_false),
                    "unknown": int(correct_unknown),
                },
            }
        )

    kept_rows = [r for r in rows if not r.get("exclusion")]
    overall_verification = verification_totals(kept_rows)
    ver_keys_acc = {ArmPolicy.coerce(v).key for v in (verification or ())}
    ver_rows_acc = [r for r in rows if not r.get("exclusion") and _row_key(r) in ver_keys_acc]
    validation_verification = verification_totals(ver_rows_acc)
    candidate_verification = verification_totals(
        [r for r in kept_rows if _row_key(r) == cand_row.key]
    )
    accounting = {
        "unpaired_episodes": int(sum(g["gain"]["unpaired"] for g in out_groups)),
        "cancelled_launches": int(sum(g["waste"]["cancelled_launches"] for g in out_groups)),
        "duplicate_launches": int(sum(g["waste"]["duplicate_launches"] for g in out_groups)),
        "wasted_tool_ms_total": float(
            sum(g["waste"]["wasted_tool_ms_total"] for g in out_groups)
        ),
        "semantic_divergence_total": int(sum(g["semantic_divergence"] for g in out_groups)),
        "correct_unknown": int(sum(g["correct"]["unknown"] for g in out_groups)),
        # Claim-verification coverage: an unverified or unchecked claim is a hole
        # in the losslessness argument, so it is counted here and fails S6.
        "verification": overall_verification,
        "candidate_verification": candidate_verification,
        # S6's actual evidence: only a live speculative run produces claim
        # counters, so the derived primary candidate carries none.
        "validation_verification": validation_verification,
        "validation_rows": sorted(f"{a}/{p}" for a, p in ver_keys_acc),
        "external_replay": external_divergence_totals(external_divergence_rows),
        # Instrumentation already inside the shared replayed timeline.
        "instrumentation": instrumentation_totals(kept_rows, percentile_points),
        # Online future table vs offline replay of the same frozen gate.
        "observed": observed_totals(kept_rows),
        "replay_fidelity": replay_fidelity(rows, candidate=cand_row, alpha=alpha),
    }
    out: dict[str, Any] = {
        "reference": ref_row.to_json(),
        "candidate": cand_row.to_json(),
        "alpha": float(alpha),
        "bootstrap": {"n": int(n_boot), "seed": int(seed), "cluster": "prompt_id"},
        "n_input_rows": len(rows),
        "n_excluded": int(sum(excluded.values())),
        "exclusions": {k: int(v) for k, v in sorted(excluded.items())},
        "accounting": accounting,
        "arms": sorted({a for a, _, _ in groups}),
        "rows": [
            {"arm": a, "policy": p, "row_key": f"{a}/{p}"}
            for a, p in sorted({(a, p) for a, p, _ in groups})
        ],
        "tool_durations_ms": sorted({d for _, _, d in groups}),
        "groups": out_groups,
    }
    if gates:
        out["submission_gates"] = submission_gates(
            rows,
            candidate=cand_row,
            reference=ref_row,
            decisive=decisive,
            verification=verification,
            external_divergence_rows=external_divergence_rows,
            n_boot=n_boot,
            seed=seed,
            alpha=alpha,
        )
    return out


def deadline_success(
    rows: Sequence[Mapping[str, Any]],
    budgets_s: Sequence[float] = DEFAULT_BUDGETS_S,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Deadline-success curve ``Q(B)``.

    ``Q(B)`` is the fraction of episodes that are **both** benchmark-correct and
    complete within ``B`` seconds, per ``(arm, policy, tool_duration_ms)``, with exact
    Clopper-Pearson intervals.  Episodes whose ``correct`` field is ``None``
    (no ground truth available) can never count as a success; they stay in the
    denominator and are reported as ``unknown_correct`` so the deficit is
    visible.  ``complete_rate`` gives the timing-only fraction for reference.
    """
    budgets = [float(b) for b in budgets_s]
    groups: dict[tuple[str, str, float], list[Mapping[str, Any]]] = defaultdict(list)
    excluded: Counter = Counter()
    for row in rows:
        reason = row.get("exclusion")
        if reason:
            excluded[str(reason)] += 1
            continue
        arm, policy = _row_key(row)
        groups[(arm, policy, _duration(row))].append(row)

    out: list[dict[str, Any]] = []
    for arm, policy, dur in sorted(groups, key=lambda kv: (kv[0], kv[1], kv[2])):
        grp = groups[(arm, policy, dur)]
        n = len(grp)
        unknown = sum(1 for r in grp if r.get("correct") is None)
        per_budget: list[dict[str, Any]] = []
        for b in budgets:
            limit_ms = b * 1000.0
            k = 0
            k_complete = 0
            for r in grp:
                comp = float(r.get("completion_ms", float("inf")))
                on_time = math.isfinite(comp) and comp <= limit_ms
                if on_time:
                    k_complete += 1
                    if r.get("correct") is True:
                        k += 1
            point, lo, hi = binom_ci_exact(k, n, alpha)
            c_point, c_lo, c_hi = binom_ci_exact(k_complete, n, alpha)
            per_budget.append(
                {
                    "budget_s": b,
                    "k": k,
                    "n": n,
                    "q": point,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "complete_rate": c_point,
                    "complete_ci_lo": c_lo,
                    "complete_ci_hi": c_hi,
                }
            )
        out.append(
            {
                "arm": arm,
                "policy": policy,
                "row_key": f"{arm}/{policy}",
                "tool_duration_ms": dur,
                "n": n,
                "unknown_correct": int(unknown),
                "budgets": per_budget,
            }
        )

    return {
        "budgets_s": budgets,
        "alpha": float(alpha),
        "n_input_rows": len(list(rows)),
        "n_excluded": int(sum(excluded.values())),
        "exclusions": {k: int(v) for k, v in sorted(excluded.items())},
        "groups": out,
    }


def deadline_paired(
    rows: Sequence[Mapping[str, Any]],
    budgets_s: Sequence[float] = DEFAULT_BUDGETS_S,
    *,
    comparators: Sequence[tuple[str, str]],
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Paired McNemar-exact tests on the deadline-success indicator.

    ``deadline_success`` reports ``Q(B)`` marginally, per ``(arm, policy,
    duration)``.  Marginal rates cannot support a claim: two policies can have
    the same ``Q(B)`` while succeeding on disjoint episodes.  This function
    pairs the *same* ``(prompt_id, seed)`` episode across a candidate row and a
    comparator row at one ``tool_duration_ms`` and one budget, and tests the
    discordant pairs exactly.

    The indicator is ``correct is True and completion_ms <= B * 1000``.  Pairs
    where either side has no ground truth (``correct is None``) are dropped from
    the test and counted in ``unknown_pairs``: charging an unknown as a failure
    would manufacture discordance wherever coverage differs between arms.

    Every cell carries a ``regime`` so no cell can be quoted out of context:

    ``saturated``
        both sides succeed on every pair -- the budget is not binding.
    ``floored``
        neither side ever succeeds -- the budget is below both.
    ``tied``
        no discordant pairs, but not at a ceiling or floor.
    ``live``
        there is something to test.

    ``n_live_cells`` is the multiplicity denominator: the grid is large and
    only a few cells are informative, so any single cell must be read against
    it.  No correction is applied here; the count is reported instead.
    """
    budgets = [float(b) for b in budgets_s]
    comps = [(str(a), str(p)) for a, p in comparators]
    by_key: dict[tuple[str, str, float], dict[tuple[str, int], Mapping[str, Any]]] = (
        defaultdict(dict)
    )
    excluded: Counter = Counter()
    dupes = 0
    for row in rows:
        reason = row.get("exclusion")
        if reason:
            excluded[str(reason)] += 1
            continue
        arm, policy = _row_key(row)
        ep = _episode_key(row)
        slot = by_key[(arm, policy, _duration(row))]
        if ep in slot:
            dupes += 1
            continue
        slot[ep] = row

    def hit(row: Mapping[str, Any], limit_ms: float) -> bool:
        comp = float(row.get("completion_ms", float("nan")))
        return row.get("correct") is True and math.isfinite(comp) and comp <= limit_ms

    cells: list[dict[str, Any]] = []
    for arm, policy, dur in sorted(by_key, key=lambda kv: (kv[0], kv[1], kv[2])):
        cand_map = by_key[(arm, policy, dur)]
        for c_arm, c_policy in comps:
            if (arm, policy) == (c_arm, c_policy):
                continue
            base_map = by_key.get((c_arm, c_policy, dur), {})
            if not base_map:
                continue
            shared = sorted(set(cand_map) & set(base_map))
            known = [
                e
                for e in shared
                if cand_map[e].get("correct") is not None
                and base_map[e].get("correct") is not None
            ]
            for b_s in budgets:
                limit = b_s * 1000.0
                a_hits = [hit(cand_map[e], limit) for e in known]
                b_hits = [hit(base_map[e], limit) for e in known]
                n_b = sum(1 for x, y in zip(a_hits, b_hits) if x and not y)
                n_c = sum(1 for x, y in zip(a_hits, b_hits) if y and not x)
                k_a = sum(a_hits)
                k_b = sum(b_hits)
                n = len(known)
                if not n:
                    regime = "empty"
                elif n_b or n_c:
                    regime = "live"
                elif k_a == n and k_b == n:
                    regime = "saturated"
                elif k_a == 0 and k_b == 0:
                    regime = "floored"
                else:
                    regime = "tied"
                cells.append(
                    {
                        "arm": arm,
                        "policy": policy,
                        "row_key": f"{arm}/{policy}",
                        "comparator": f"{c_arm}/{c_policy}",
                        "tool_duration_ms": dur,
                        "budget_s": b_s,
                        "n_pairs": n,
                        "n_prompts": len({e[0] for e in known}),
                        "n_seeds": len({e[1] for e in known}),
                        "unknown_pairs": len(shared) - n,
                        # Nothing is silently dropped: episodes present on only
                        # one side never enter the test and are counted here.
                        "unpaired_candidate": len(set(cand_map) - set(base_map)),
                        "unpaired_baseline": len(set(base_map) - set(cand_map)),
                        "k_candidate": int(k_a),
                        "k_comparator": int(k_b),
                        "regime": regime,
                        "mcnemar": mcnemar_exact(n_b, n_c),
                    }
                )

    # A comparator named in the manifest but absent from the outcome rows would
    # otherwise vanish silently, making the grid look more complete than it is.
    coverage = []
    for c_arm, c_policy in comps:
        label = f"{c_arm}/{c_policy}"
        n_c_cells = sum(1 for c in cells if c["comparator"] == label)
        coverage.append(
            {
                "comparator": label,
                "n_cells": n_c_cells,
                "reason": None if n_c_cells else "no outcome rows for this row key",
            }
        )

    return {
        "status": "present" if cells else "empty",
        "budgets_s": budgets,
        "comparators": [f"{a}/{p}" for a, p in comps],
        "comparator_coverage": coverage,
        "alpha": float(alpha),
        "n_cells": len(cells),
        "n_live_cells": sum(1 for c in cells if c["regime"] == "live"),
        "duplicate_episode_rows": dupes,
        "n_excluded": int(sum(excluded.values())),
        "exclusions": {k: int(v) for k, v in sorted(excluded.items())},
        "cells": cells,
    }
