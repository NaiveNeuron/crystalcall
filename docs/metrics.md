# CrystalCall / DenoisePTC: statistics reference

Every number in the paper is produced by `crystalcall.metrics` (estimators),
`crystalcall.aggregate` (tables — `reports/results.json` + `reports/results.md`)
and `crystalcall.figures` (plots), computed from files under
`experiments/derived/`. Nothing is hand-entered; `reports/results.md` is
generated and must never be edited.

The **submission gates are S1-S8** (§4). Everything in §2-§3 is the
*estimator machinery* that feeds those gates — it is not a competing gate list.

Global conventions:

| convention | value |
|---|---|
| nominal level | two-sided 95% (`alpha = 0.05`) |
| resampling | `numpy.random.default_rng(seed)`, `seed = 0`, `n = 10000` replicates |
| bootstrap cluster | `prompt_id` (seeds of one prompt are dependent) |
| non-inferiority margin | 1 percentage point of task success (`margin = 0.01`) |
| table rows | `(arm, policy)` pairs declared by `configs/experiments/main_table.yaml` (schema `crystalcall-main-table/1`) |
| grouping key | `(arm, policy, tool_duration_ms)` — the same offline policy replayed against two arms' traces is two different results and is never pooled |
| pairing key | `(prompt_id, seed)` within one `tool_duration_ms` |
| candidate row | the declared `crystalcall` row (`crystalcall/full_gate`); falls back to the first declared non-oracle speculative row |
| reference row | `reference_row` from the spec (`diffusion_standard/diffusion_final_response`) — tools issued only after the response completes |
| decisive references | `decisive_rows` from the spec: `diffusion_standard/diffusion_earliest_committed` (S7) and `preact_style/preact_ready` (S8) |
| ablation set | `ablation_arm` + `ablation_policies` from the spec |
| latency / deadline grids | `latency_conditions_ms` and `deadline_budgets_s` from the spec |
| frozen thresholds | `k`, `tau`, `min_overlap_ms`, ... read from `configs/frozen_policy.json` (written by `crystalcall pilot`), never hardcoded |

Degenerate inputs never raise in an aggregation path: they return `nan`
percentiles, `p_value = 1.0`, or the full `[0, 1]` interval, and the reason is
recorded (`degenerate`, `continuity_applied`, `unpaired`, `status = "missing"`).
A missing measurement is never treated as a pass.

---

## 1. Derived quantities

Definitions, not estimators; computed per episode from
`PolicyOutcome.to_json()` / `TrajectoryAnalysis.to_json()` rows.

| quantity | definition | source fields |
|---|---|---|
| completion time | wall-clock ms from decode start until the last tool result needed by the authoritative program is available (instrumentation cost charged to every arm) | `completion_ms` |
| relative latency gain | `1 - completion_candidate / completion_reference` on the same `(tool_duration_ms, prompt_id, seed)` | `completion_ms` |
| speculative precision | `true_launches / (true_launches + false_launches)`; a launch is *true* iff its canonical identity equals a final authoritative call | `true_launches`, `false_launches` |
| eligible-call coverage | `covered_final_calls / eligible_final_calls`: speculation-eligible authoritative calls whose result was already in flight at commit time | `covered_final_calls`, `eligible_final_calls` |
| wasted work | tool time spent on launches that never matched a final call | `wasted_tool_ms` |
| cancelled launches | pre-commit eligible launches that did not match the final call (discarded at commit) — identical to `false_launches` | `false_launches` |
| duplicate launches | `max(0, launches - claimed - false_launches)`: launched work neither claimed nor charged as a false launch (ineligible-at-commit or post-commit duplicates), reported separately so waste is never double counted | `launches`, `claimed`, `false_launches` |
| hidden overlap | tool time overlapped with remaining decode — the part of tool latency the user never observes | `hidden_overlap_ms` |
| semantic divergence | a claimed speculative result whose digest differs from a normal re-execution of the same call. Must be **zero**: this is the losslessness counter | `semantic_divergence`, `claims_verified_divergent` |
| claim verification | per episode, every claimed speculative result is re-executed by `HostToolRouter(verify_claims=True)` and the two result digests are compared, so the model is held fixed and a mismatch can only come from scheduling. Each claim lands in exactly one bucket | `claims_claimed`, `claims_verified_match`, `claims_verified_divergent`, `claims_verification_error`, `claims_unverified`, `verification_complete` |
| pre-commit lead | `committed_ns - invariant_ns` in ms: how long a call was stable *and* executable before the model committed it | `FinalCallAnalysis.lead_ms()` |
| in-timeline instrumentation | trace+parse cost already contained in the replayed decode timeline that every arm shares; reported, never charged to completion again and never credited back to the reference arm (§5.1) | `instrumentation_in_timeline_ms` |
| online observation | what the live future table did on that request: launches, claims and discarded work, plus whether the offline replay of the arm's own frozen gate reproduced the launch count (§5.2) | `observed_launched`, `observed_claimed`, `observed_wasted`, `observed_matches_replay` |

### Exclusion accounting (nothing is silently dropped)

`summarize_outcomes` returns `exclusions` (by reason) and an `accounting` block
(`unpaired_episodes`, `cancelled_launches`, `duplicate_launches`,
`wasted_tool_ms_total`, `semantic_divergence_total`, `correct_unknown`, plus the
`verification` / `candidate_verification` / `external_replay` sub-blocks that
carry `claims_claimed`, `claims_verified_match`, `claims_verified_divergent`,
`claims_verification_error`, `claims_unverified`, `claims_unaccounted`,
`episodes_verification_incomplete` and `episodes_flagged_divergent`).
`aggregate` promotes the verification holes — `claims_verification_error`,
`claims_unverified`, `claims_unaccounted`, `episodes_verification_incomplete` —
into the `claim_verification` scope of the top-level `exclusions` table, and the
ToolSandbox episode errors into the `toolsandbox_episodes` scope.
`aggregate` merges these with the trajectory-side counts — `parse_failure`,
`invalid_final_parse`, per-reason canonicalisation rejections
(`rejection_counts`) and `malformed_json_line` — into the single top-level
`exclusions` table of `reports/results.json` / `results.md`. Episodes with no
partner in the reference arm are counted (`gain.unpaired`, with up to eight
example keys, and `paired_gain(...)["unpaired_candidate" | "unpaired_reference"]`),
never discarded quietly.

`aggregate` also promotes a failing overhead-ladder rung (`ok: false`) under the
`overhead_ladder` scope and any online-vs-replay launch-count mismatch under the
`replay_fidelity` scope; the accounting block additionally carries
`instrumentation` (§5.1), `observed` and `replay_fidelity` (§5.2).

---

## 2. Estimators

### 2.1 `percentiles(xs, ps=(50, 95, 99))`

* **Estimand:** distribution of a latency (completion time, hidden overlap,
  pre-commit lead, microbenchmark cost).
* **Estimator:** linearly interpolated sample quantiles
  (`numpy.percentile(..., method="linear")`), plus `n` and `mean`.
* **Assumptions:** none beyond i.i.d.-ness *within* the reported sample; no
  distributional shape assumed. Descriptive only — inferential claims use
  §2.2-§2.7.
* **Degenerate:** empty / all-non-finite sample gives `nan` percentiles, `n = 0`.
* **Feeds:** the p50/p95/p99 columns of the method-comparison table, the
  overhead and contention tables, and the S8 Pareto profile.

### 2.2 `bootstrap_ci_clustered(values_by_cluster, stat, n, seed, alpha)`

* **Estimand:** median of the paired relative gain (default `stat`), or of the
  pre-commit lead.
* **Estimator:** cluster (block) percentile bootstrap. Each of `n` replicates
  resamples `m` *clusters* with replacement (`m` = number of observed clusters)
  and pools **all** observations of every drawn cluster; the CI is the
  `alpha/2` / `1 - alpha/2` percentile pair of the replicate statistics. The
  point estimate is the statistic on the observed pooled sample.
* **Assumptions:** clusters (prompts) exchangeable and independent; observations
  *within* a cluster may be arbitrarily dependent — which is the point, since
  repeated seeds of one prompt share the trajectory shape. The percentile
  bootstrap is first-order accurate and slightly under-covers a median in small
  samples; the regression test therefore asserts a loose `>= 0.85` empirical
  coverage against a nominal 95%.
* **Degenerate:** no finite observation gives `(nan, nan, nan)`; `m = 1` still
  returns an interval (no within-cluster bootstrap is substituted, so the
  interval stays honest about having one cluster).
* **Determinism:** sorted cluster order, one seeded generator, so the CI is
  reproducible bit for bit. With the default median and equal-size clusters a
  vectorised path computes the identical result.
* **Feeds:** S3, S4, S7, S8 (gain CIs) and the pre-commit lead CI.

### 2.3 `wilcoxon_paired(a, b)`

* **Estimand:** whether the paired completion-time difference is centred at zero.
* **Estimator:** `scipy.stats.wilcoxon(diff, zero_method="wilcox",
  alternative="two-sided")` on the non-zero differences.
* **Assumptions:** paired observations, differences symmetric about the median
  under the null, pairs independent. Since seeds within a prompt are *not*
  independent, this p-value is a secondary (anti-conservative) check; primary
  evidence is the clustered bootstrap CI (§2.2) plus the randomisation test
  (§2.4).
* **Degenerate:** all-zero differences returns `statistic = 0.0`,
  `p_value = 1.0`, `degenerate = True` instead of the `scipy` exception.
* **Feeds:** reported alongside S3, S4, S7, S8.

### 2.4 `paired_randomization(a, b, n, seed)`

* **Estimand:** mean paired difference under the sharp null of within-pair
  exchangeability.
* **Estimator:** sign-flip randomisation on the non-zero differences. With
  `m <= 20` non-zero pairs all `2**m` sign vectors are enumerated (**exact**);
  otherwise `n` Monte-Carlo sign vectors are drawn and
  `p = (1 + #{|d*| >= |d|}) / (1 + n)` is used, which is valid for finite `n`.
  `exact` / `n_resamples` record which branch ran.
* **Assumptions:** only within-pair sign exchangeability — no normality, no
  symmetry of the difference distribution.
* **Degenerate:** no pairs or all-zero differences returns `p_value = 1.0`.
* **Feeds:** the primary p-value reported with S3, S4, S7, S8.

### 2.5 `mcnemar_exact(b, c)`

* **Estimand:** whether speculation changes task correctness.
* **Estimator:** exact binomial McNemar,
  `scipy.stats.binomtest(min(b, c), b + c, 0.5)` (two-sided), on the discordant
  cells `b` (candidate right / reference wrong) and `c` (the reverse).
* **Assumptions:** paired binary outcomes, independent pairs; only discordant
  pairs are informative. Exact — no chi-square or continuity approximation.
* **Degenerate:** `b + c == 0` returns `p_value = 1.0`, `degenerate = True`.
* **Feeds:** reported inside S5 (`detail.mcnemar_p`); S5 itself is *decided* by
  the paired-difference CI, not by this p-value.

### 2.6 `noninferiority_success(a_success, b_success, margin=0.01, alpha=0.05)`

* **Estimand:** paired difference in task-success rate,
  `diff = (n10 - n01) / n`, with `n10` = candidate right & reference wrong and
  `n01` the reverse.
* **Estimator:** Wald CI for a paired difference of proportions,
  `se = sqrt(n10 + n01 - (n10 - n01)^2 / n) / n`,
  `ci = diff +- z_{1-alpha/2} * se`. Non-inferiority is declared iff
  `ci_lo > -margin`.
* **Continuity handling (documented and flagged):** when `n10 + n01 == 0` (or
  the radicand is non-positive, possible when all discordant pairs point one
  way) the Wald `se` collapses to exactly 0 and the interval would degenerate to
  `[0, 0]`, declaring non-inferiority from zero evidence. In that case 0.5 is
  added to *each* discordant cell (Haldane-Anscombe style), giving `se = 1 / n`;
  `continuity_applied` records it. Consequence, deliberately: perfect agreement
  clears a 1 pp margin only once `1.96 / n < 0.01`, i.e. `n > 196` episodes.
* **Assumptions:** paired binary outcomes; normal approximation to the paired
  difference (adequate at the intended `n > 196`, conservative under the
  correction). Pairs whose ground truth is unknown are excluded from the test
  and counted (`correctness_unknown_pairs`).
* **Degenerate:** `n == 0` returns `nan` bounds and `non_inferior = False`.
* **Feeds:** S5.

### 2.7 `binom_ci_exact(k, n, alpha)` and `rule_of_three_upper(n)`

* **Estimand:** a single binomial rate — speculative precision, eligible-call
  coverage, valid-final-parse rate, deadline success `Q(B)`, semantic-divergence
  rate.
* **Estimator:** Clopper-Pearson exact interval via the Beta quantile identity
  `lo = Beta(alpha/2; k, n-k+1)`, `hi = Beta(1-alpha/2; k+1, n-k)`, with the
  boundary conventions `k = 0 -> lo = 0`, `k = n -> hi = 1`. The bounds solve the
  defining tail equations `P(X >= k | lo) = alpha/2` and
  `P(X <= k | hi) = alpha/2` (asserted in `tests/test_metrics.py`).
* **Assumptions:** independent Bernoulli trials at the *trial* level (one
  launch, one authoritative call, one episode). Cross-prompt dependence is
  handled by the clustered bootstrap wherever a *median* is reported instead.
* **Degenerate:** `n == 0` returns `(nan, 0.0, 1.0)` — no information.
* **`rule_of_three_upper(n) = min(1, 3/n)`:** 95% upper bound on an unobserved
  event rate after `n` clean trials (standard approximation to the
  Clopper-Pearson bound at `k = 0`); used to quantify the losslessness claim when
  zero divergences are observed. For S6 the denominator `n` is the number of
  claims **actually verified** (`claims_verified_match + claims_verified_divergent`),
  never the number of claims launched or claimed.
* **Feeds:** S1, S2, S6, and the `Q(B)` intervals.

### 2.8 `paired_gain(rows, candidate, reference, duration=None, ...)`

``candidate`` and ``reference`` are `(arm, policy)` rows (an `ArmPolicy`, a
2-tuple or a mapping with `arm`/`policy`). A pair is one
`(tool_duration_ms, prompt_id, seed)` episode present under **both** rows; the
function returns the median relative gain with its prompt-clustered bootstrap CI
(§2.2), the Wilcoxon (§2.3) and randomisation (§2.4) p-values on the raw
completion times, the cluster count, and the unpaired / unusable counts on both
sides. With `duration=None` all durations are pooled. This is the single
estimator behind S3, S4, S7 and S8.

### 2.9 `summarize_outcomes(rows)`

Groups `PolicyOutcome.to_json()` rows by `(arm, policy, tool_duration_ms)` and
emits, per group (`arm`, `policy`, `row_key = "<arm>/<policy>"`):
`completion_ms` p50/p95/p99 (§2.1); the median paired relative gain
against the declared reference row with its clustered bootstrap CI (§2.2);
launch counts (`total`/`true`/`false`/`claimed`/`duplicate`, per-episode rate);
speculative precision and eligible-call coverage with exact CIs (§2.7); wasted,
cancelled and duplicate work; hidden-overlap percentiles; the
semantic-divergence count; and the correctness tally. It also returns the
`exclusions` + `accounting` blocks and, unless `gates=False`, the
`submission_gates` block (§4). This function *is* the main method-comparison
table.

### 2.10 `deadline_success(rows, budgets_s=(1, 2, 5, 10))`

`Q(B)` = fraction of episodes both benchmark-correct and complete within `B`
seconds, per `(policy, tool_duration_ms)`, with exact binomial CIs (§2.7).
Episodes with unavailable ground truth (`correct is None`) can never count as a
success but stay in the denominator and are reported as `unknown_correct`, so
the deficit is visible rather than flattering; `complete_rate` gives the
timing-only fraction. Supporting evidence for the deadline-utility discussion;
it does not decide a gate.

---

## 3. Estimator-to-gate map

| estimator | used by |
|---|---|
| `binom_ci_exact` (§2.7) | S1, S2, S6, `Q(B)`, valid-final-parse rate |
| `rule_of_three_upper` (§2.7) | S6 (upper bound on the divergence rate) |
| `paired_gain` = `bootstrap_ci_clustered` + `wilcoxon_paired` + `paired_randomization` (§2.2-§2.4, §2.8) | S3, S4, S7, S8 |
| `noninferiority_success` (§2.6) | S5 (decision) |
| `mcnemar_exact` (§2.5) | S5 (reported alongside) |
| `percentiles` (§2.1) | S8 Pareto profile, method / overhead / contention tables |
| `deadline_success` (§2.10) | deadline-utility discussion |

---

## 4. Submission gates S1-S8

Produced by `metrics.submission_gates(rows, candidate=…, reference=…, decisive=…)`,
embedded in `summarize_outcomes(...)["submission_gates"]` and promoted to the top
level of `reports/results.json` (also rendered as a table in
`reports/results.md`). The three row arguments come from
`configs/experiments/main_table.yaml` — `reference_row` and `decisive_rows` —
so no policy name is hardcoded in the gate logic: `decisive_rows[0]` drives S7
and `decisive_rows[1]` drives S8. Every gate row carries `id`, `key`,
`description`, `statistic`, `comparator`, `required_threshold`, `point`,
`ci_lo`, `ci_hi`, `n`, `pass`, `required`, `status` and `detail`.

**Decision rule.** A gate passes only when its **point estimate** meets the
threshold; the CI is always reported next to it. S7 additionally requires the CI
to exclude 0. `status = "missing"` never passes.

| id | key | required | threshold | estimator | notes |
|---|---|---|---|---|---|
| **S1** | `speculative_precision` | required | `>= 0.99` | exact Clopper-Pearson over all candidate launches (§2.7) | point must clear 0.99; `ci_lo` reported alongside; detail carries true/false launch counts |
| **S2** | `eligible_call_coverage` | required | `>= 0.40` | exact Clopper-Pearson over eligible final calls (§2.7) | detail carries covered / eligible counts |
| **S3** | `median_gain_500ms` | required | `>= 0.15` (target band 0.15-0.20) | `paired_gain` vs the declared reference row at `D = 500 ms` (§2.8) | reports cluster-bootstrap CI plus Wilcoxon and randomisation p; `in_target_band` flag; **never present without S7** |
| **S4** | `median_gain_2000ms` | required | `>= 0.25` (target band 0.25-0.30) | `paired_gain` vs the declared reference row at `D = 2000 ms` (§2.8) | same estimators and companion rule as S3 |
| **S5** | `eventual_success_noninferior` | required | paired success difference `ci_lo > -0.01` | `noninferiority_success` (§2.6); `mcnemar_exact` (§2.5) reported | decision by the paired-difference CI lower bound; detail carries `n10`, `n01`, `mcnemar_p`, `continuity_applied` |
| **S6** | `semantic_divergence_zero` | required | observed divergences `== 0` **and** full verification coverage (see §4.2) | `verification_totals` over candidate episodes + `external_divergence_totals` over the ToolSandbox replay + `rule_of_three_upper(claims_verified)` (§2.7) | fails closed: detail carries `claims_claimed`, `claims_verified_match`, `claims_verified_divergent`, `claims_verification_error`, `claims_unverified`, `claims_unaccounted`, `claims_verified`, `episodes_verification_incomplete`, `episodes_flagged_divergent`, `external_replay`, `rule_of_three_upper` and an explicit `fail_reasons` list |
| **S7** | `advantage_over_earliest_committed` | required, **DECISIVE** | median gain `> 0` **and** `ci_lo > 0` | `paired_gain` vs `decisive_rows[0]` (`diffusion_standard/diffusion_earliest_committed`), pooled over durations plus per-duration detail (§2.8) | this is the gate that shows the *uncommitted* denoising state carries the win, not merely early issue at commit time |
| **S8** | `advantage_over_preact_style` | **preferred** (not required), decisive | `>= 0.10` relative gain vs `decisive_rows[1]`, **or** a materially better latency/risk trade-off | `paired_gain` vs `decisive_rows[1]` (`preact_style/preact_ready`) (§2.8); on failure a Pareto comparison | when the 10% gain is not reached the detail carries `pareto` = completion percentiles, precision (with CI), wasted ms per episode and gain vs the reference row for both rows, so the claim can be argued as a trade-off instead of a bare boolean; an undeclared or outcome-less arm is reported (`status = "missing"`), never invented |

**S7 and S8 are the decisive pair.** A gain over the reference row alone
(S3/S4) is explicitly scientifically insufficient — a post-response baseline is
trivially beatable — so S3/S4 are always rendered with S7 (and S8) beside them,
and each S3/S4 row carries `detail.decisive_companion = "S7"`.

Roll-up fields: `required_passed` / `required_total`, `all_required_pass`,
`preferred_passed`, `by_id`, and `decisive` (the S7/S8 pass pair).

Frozen policy thresholds (`k`, `tau`, `min_overlap_ms`, ...) are **read** from
`configs/frozen_policy.json` (written by `crystalcall pilot`) by
`aggregate.read_frozen_policy` and echoed into the report with the file's
sha256; a missing file is reported, never substituted with defaults.

### 4.1 The declared comparison table

`configs/experiments/main_table.yaml` (schema `crystalcall-main-table/1`) *is*
the main table: `rows[]` (`id`, `label`, `arm`, `policy`, `source`,
`speculative`, optional `oracle` / `note`), `reference_row`, `decisive_rows`,
`ablation_arm`, `ablation_policies`, `latency_conditions_ms` and
`deadline_budgets_s`. `aggregate.load_main_table` parses it,
`aggregate.main_table_section` evaluates it, and the file's sha256 is recorded
in `results["sources"]` next to the derived inputs.

`results["main_table"]` holds one entry per declared row **in declared order**,
each with `status`, a `pooled` block and a `by_duration` block over
`latency_conditions_ms`. A declared `(arm, policy)` pair with no outcomes is
emitted with `status: "missing"` (and a reason) rather than dropped, because a
silently vanishing arm reads as a passing table. The rendered
`reports/results.md` main table carries exactly these columns —
Method | Success | Semantic divergence | p50 | p95 | p99 | Precision |
Coverage | Wasted work — labelled with `row.label`, one sub-table per latency
condition, followed by the paired-gain table, the gate ablation over
`ablation_policies` and the S1-S8 gate table.

### 4.2 S6 fails closed on verification coverage

S6 is the losslessness gate, so absence of evidence is never evidence of
absence. It passes **only** when every one of the following holds over the
candidate row's non-excluded episodes:

1. `sum(claims_verified_divergent) == 0` — no verified claim diverged;
2. `sum(claims_verification_error) == 0` — a verifier that raised has checked
   nothing, so it cannot support the claim;
3. `sum(claims_unverified) == 0` **and** `claims_unaccounted == 0`, i.e.
   `sum(claims_claimed) == sum(claims_verified_match) + sum(claims_verified_divergent)`;
4. every episode reports `verification_complete == true`;
5. `sum(claims_claimed) > 0` — a run that claimed nothing has demonstrated
   nothing and must not pass vacuously;
6. no episode is flagged `semantic_divergence` without a per-claim count; and
7. the ToolSandbox replay check (`toolsandbox_episodes.jsonl`, one generation per
   episode replayed against a `WorldState.clone()` of the initial world) reports
   zero divergent episodes when that file is present.

Every one of those numbers is reported in the gate row's `detail`, together with
a `fail_reasons` list naming exactly which condition failed, and in the
"Verification coverage (gate S6)" table of `reports/results.md`. The
rule-of-three 95% upper bound on the divergence rate uses
`claims_verified_match + claims_verified_divergent` as its denominator
(`rule_of_three_denominator` records this), never the number of claims launched.

---

## 5. Instrumentation cost and replay fidelity

### 5.1 In-timeline instrumentation (`instrumentation_in_timeline_ms`)

The trace+parse cost of an instrumented run is **already inside** the replayed
decode timeline that every arm shares. It is therefore:

* **not** charged to `completion_ms` a second time — that would double-count it; and
* **not** credited back to the reference arm — that would build exactly the
  "fully instrumented candidate versus unrealistically stripped baseline"
  comparison.

Because every arm replays the same instrumented timeline, the cost cancels in
every paired comparison and is reported on its own:
`metrics.instrumentation_totals` de-duplicates per source execution (the same
trajectory appears once per tool duration) and returns per-episode p50/p95/p99,
the total, a per-arm breakdown and `charged_to_completion: false`.
`aggregate.instrumentation_section` renders it in the "Instrumentation cost"
table of `reports/results.md` **next to the Phase 0 overhead ladder** read from
`overhead_microbench.jsonl` (`level`, `n`, `p50_ms`, `p95_ms`,
`revisions_per_request`, `relative_p50_vs_off`, `absolute_p50_delta_ms`,
`parse_ms_per_revision`, `decode_ms_per_revision`, `ok`, `problems`), rungs in
ladder order. A rung with `ok: false` is named as a failing rung and counted in
the exclusion table under `overhead_ladder` — the measurement itself is suspect,
not merely the number.

### 5.2 Replay fidelity (`observed_*`)

Each outcome row also carries what the **online** future table actually did on
that request — `observed_launched`, `observed_claimed`, `observed_wasted` — and
`observed_matches_replay`: whether replaying the same frozen gate over the same
trace reproduced the launch count. `observed_matches_replay` is `None` for any
policy that is not that arm's own frozen gate, because a different policy is
*expected* to differ; those rows land in `not_comparable` and are never counted
as a match.

`metrics.replay_fidelity` (surfaced as `accounting.replay_fidelity`) reports
`compared`, `matching`, `mismatching`, `not_comparable`, `compared_rows`, the
mismatching `(arm, trajectory_id)` list (`mismatching_keys`) and a
`mismatching_rows` detail list carrying observed vs replayed launch and claim
counts. **A mismatch is a bug signal, not a result:** `reports/results.md`
prints a `REPLAY FIDELITY FAILURE` block that names every offending request
instead of averaging it into any table, and `aggregate` records the count under
the `replay_fidelity` scope of the exclusion table. A clean run states that
every comparable request reproduced its online launch count.

The same block carries the candidate row's launch precision **measured online**
(`observed_claimed / observed_launched`) beside the **offline re-derivation**
(`true_launches / (true_launches + false_launches)`), each with an exact
binomial CI, the number of executions behind it and the signed difference, so a
reader can see the measured number next to the re-derived one. Both sides are
de-duplicated per execution (`metrics.dedupe_executions`), so neither interval
tightens when the same executions are replayed at more tool durations.

---

## 6. File layout and reproduction

```
experiments/raw/<arm>/<shard>.jsonl(+.meta.json)   append-only hash-chained traces
experiments/derived/trajectory_analyses.jsonl      TrajectoryAnalysis.to_json()
experiments/derived/policy_outcomes.jsonl          PolicyOutcome.to_json()
experiments/derived/gate_ablations.jsonl           PolicyOutcome.to_json() (declared ablation arm)
experiments/derived/overhead_microbench.jsonl      {"component": str, <numeric>: float}
experiments/derived/contention.jsonl               optional {"scenario": str, "workers": int, <numeric>: float}; contention was not measured, so the table reports it missing
experiments/derived/toolsandbox_episodes.jsonl     ToolSandbox replay check (feeds S6)
configs/experiments/main_table.yaml                the declared comparison table (schema crystalcall-main-table/1)
experiments/figures/*.pdf, *.png                   200 dpi, deterministic
configs/frozen_policy.json                         frozen k / tau / min_overlap_ms (from `crystalcall pilot`)
reports/results.json                               schema "crystalcall-results/1"
reports/results.md                                 generated tables (gates first)
```

Reproduce the report and the figures:

```python
from crystalcall.aggregate import aggregate
from crystalcall.figures import render_all

aggregate("experiments/derived", "reports/results.json", "reports/results.md")
render_all("experiments/derived", "experiments/figures")
```

`reports/results.json` records, for every input file, its name, path, byte size,
row count, malformed-line count and sha256, so any table can be traced back to
the exact bytes it was computed from. Figures are byte-identical across runs on
identical input (fixed style, fixed seeds, no timestamps in PDF/PNG metadata);
`render_all` skips and reports any figure whose input is absent, and
`fig_architecture_timeline` raises rather than drawing an invented trajectory.
