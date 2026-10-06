# Baselines

Nine arms. Every arm shares the same OMP-Lite configuration, the same fixed system prompt
(`configs/system_prompt.md`, digest asserted by `crystalcall doctor`), the same per-prompt
tool schemas, the same canonicaliser, the same tool implementations and the same retry
policy. Scheduling is the only treatment.

Every arm's reported end-to-end latency includes decode, instrumentation, parsing,
scheduling, tool execution, claim and cancellation. A stripped baseline is never compared
against a fully instrumented CrystalCall: arms that consume the revision stream are
charged the measured per-revision trace and parse cost from `configs/frozen_policy.json`
(`CostModel`), and `simulate_policy` applies that charge in `replay.py`.

| # | Arm | Kind | Model generations | Launch point |
| --- | --- | --- | --- | --- |
| A | Diffusion final-response | derived | reuses standard traces | response completion |
| B | Diffusion final-canvas | derived | reuses standard traces | containing canvas finalises |
| C | Diffusion earliest committed valid call | derived | reuses standard traces | first revision where every token of a valid canonical call is inside the entropy-bound accepted set, else canvas commit |
| D | P-ReAct-style tool-priority diffusion | **online** | 300 test x 3 seeds | sampler-side span priority; launch at `tool_call_ready` |
| E | Gemma 4 AR streaming | **online** | 300 test x 3 seeds | first complete valid call in the irreversible token stream |
| F | Gemma 4 AR sPTC | derived from E when replay-valid | shares E's generations | shadow-runtime future on the AR prefix |
| G | **CrystalCall (DenoisePTC)** | **derived** | reuses standard traces | frozen pre-commit gate on the predicted-clean canvas |
| G' | CrystalCall online validation | **online** | 60 test prompts x 1 seed (+ same subset at 500 ms tool duration) | same frozen gate, live; validates the invariant, not the latency estimate |
| H | Oracle earliest invariant | derived | reuses standard traces | first revision after which the canonical call never differs from the final call |
| I | Oracle perfect final-match | derived | reuses standard traces | earliest executable final-matching candidate, even if it later disappears |

`BASELINE_POLICIES` in `src/crystalcall/replay.py` names the derived policies; the gate
ablations live in `POLICY_LIBRARY` in `src/crystalcall/scheduler.py`.

**Why G is derived, not generated.** DiffusionGemma's sampler is stochastic even at a
pinned server seed and each arm advances the same generator, so a
separately generated CrystalCall arm would produce *different* trajectories for the same
`(prompt, seed)` — confounding sampling variance with scheduling in exactly the paired
comparison that carries the claim — and would add ~900 model turns. The frozen gate is therefore replayed over the same standard trajectories every
diffusion baseline row uses, and the live behaviour is validated separately on a bounded
subset. Gate S6 reads that online subset, because only a live run
produces claim-verification counters.

## A. Diffusion final-response

Wait for the whole assistant response, then run the authoritative program normally. This
is the reference denominator for every paired gain. A win over A alone is explicitly
**not** sufficient evidence for the paper's claim.

## B. Diffusion final-canvas

Execute a call as soon as the 256-token canvas containing it is finalised, while later
canvases keep generating. Launch time is `canvas_done_ns` for the call's canvas, recorded
by the patched sampler at each commit step.

## C. Diffusion earliest committed valid call

The strongest non-speculative diffusion baseline and one of the two decisive comparisons.

DiffusionGemma's commitment granularity needs care. The vLLM
sampler renoises unaccepted positions in place and only emits a canvas when the request
converges, so an individual token is not irrevocably committed mid-canvas. The
entropy-bound acceptance mask is nevertheless a real per-step signal. C therefore launches
at the earlier of:

1. the first revision in which **every** token of the call's span lies inside the
   entropy-bound accepted set (`FinalCallAnalysis.committed_step` / `committed_ns`); and
2. the containing canvas's commit timestamp (`canvas_done_ns`).

Both quantities are stored separately for every final call, so C can be reported under
either reading and the paper must state which it uses. Reading 1 makes C *earlier* and
therefore harder to beat; it is the primary reading precisely because the weaker reading
would flatter CrystalCall.

## D. P-ReAct-style tool-priority diffusion

**This arm is an approximation and is labelled `P-ReAct-style` everywhere.** DLLM-Searcher's
P-ReAct guides diffusion decoding so that the tool-call instruction is prioritised and
becomes available while reasoning continues. We do not have their guided-decoding
implementation, and we do not claim to reproduce its internals. What we do reproduce is
the *mechanism class*: altered token-selection priority inside the canvas.

The prompt, the model, the tool schema, the canonicaliser, the executor and the retry
policy are **identical** to every other arm. The only thing
that differs is server-side decoding, enabled by `CRYSTALCALL_PREACT=1`:

1. **Span detection.** The tool-call span is located in the predicted-clean canvas as the
   first Gemma `<|tool_call>` delimiter and the first `<tool_call|>` after it. The ids are
   resolved from the checkpoint tokenizer and pinned in `configs/preact.yml`; the
   in-container module refuses to run without them rather than falling back to a default,
   because a wrong id would make the whole baseline vacuously fast.
2. **Priority acceptance.** Inside the span, a position is additionally accepted when its
   per-token entropy is below a looser bound (`CRYSTALCALL_PREACT_BOUND`) than the global
   cumulative entropy bound. Union only: a position the ordinary rule accepted is never
   un-accepted, and positions outside the span follow the ordinary rule exactly.
3. **Readiness signalling.** When the span has been byte-identical for
   `CRYSTALCALL_PREACT_STABLE` consecutive steps and every in-span position is accepted,
   the sampler emits `tool_call_ready` once and the executor launches the parsed call
   there, while generation continues. **Convergence is never touched.** Ending denoising
   early would commit the whole argmax canvas including unstable out-of-span positions,
   corrupting the authoritative turn and truncating the very reasoning P-ReAct is meant to
   overlap with.

`CRYSTALCALL_PREACT_MODE` decomposes the mechanism into `span_accept`, `ready_signal` or
`both` (the arm default), so the two halves are separable in the ablation table.

**The in-span bound is development-tuned, in the baseline's favour.** The first guessed
value (0.6 nats) destroyed the arm: it lost every tool call on 4 of 6 development prompts
and two final programs stopped parsing, because freezing in-span positions that loosely
locks in tokens the canvas would have revised. A crippled baseline
would hand CrystalCall a purely artefactual advantage on the S8 comparison, so
`scripts/preact_bound_sweep.py` selects the **largest** bound over a stratified 12-prompt
development subset that shows zero lost calls, zero parse regressions, zero correctness
regressions against the standard arm on the same prompts and seed, and at least one
readiness event. Every rejected rung is preserved in
`experiments/derived/preact_bound_sweep.json` with its failing criterion, and
`tests/test_preact_selection.py` fails if the configured bound is not the selected one.
If no value survives, the arm runs `mode=ready_signal` -- readiness without touching the
acceptance mask, so the trajectory is provably unchanged -- and that is reported as a
finding about P-ReAct-style priority acceptance on this model.

Documented deviations from the paper:

- P-ReAct's priority is learned/guided; ours is a fixed entropy-threshold relaxation over
  a syntactically located span. No reward or training signal is used (the workshop version
  trains nothing).
- Priority is applied to the whole tool-call span, not to individual instruction tokens in
  the order the paper's guidance would prefer.
- The reported acceleration is therefore not expected to match the ~15% quoted by
  DLLM-Searcher on their setup.

Because the mechanism changes decoding, this arm is **generated**, not derived: its
trajectories differ from the standard arm's and cannot be replayed from them. CrystalCall's
decoding is untouched, and with `CRYSTALCALL_PREACT` unset the patched sampler is
behaviourally identical to the trace-only build -- that is the invariant `tests/test_preact.py`
pins.

Launch accounting: the offline policy is `preact_ready` (`crystalcall.replay`), and it is
offered *only* to this arm -- a standard trace carries no readiness events, so the policy
raises rather than silently degrading into a different baseline. A readiness launch is
**speculative**: the span can settle to a call the final program does not contain, so this
row is scored with the same precision, wasted-work and false-launch accounting as
CrystalCall. Global safety still applies: no baseline launches a `Write`
or a `LiveRead` early either. A final call that never received a readiness signal falls
back to response completion -- deliberately the unflattering fallback, since this arm does
not watch canvas commits.

Arm C (earliest committed) remains **mandatory** and is the primary committed-call
comparison; D is the closest available prior-art-shaped baseline.

## E. Gemma 4 AR streaming

The matched autoregressive control, `google/gemma-4-26B-A4B-it`, same prompt and tool
surface, `tools.format: native`. The runtime watches the irreversible token stream and
launches a call as soon as a complete valid canonical call has been emitted. Because an AR
prefix is monotonic, there is nothing to revise: precision is 100% by construction and
the interesting quantity is *how late* the launch is relative to CrystalCall.

This is a **scheduling** comparison. It is not a claim that DiffusionGemma and Gemma 4
have equal base task quality; eventual task success is reported per arm and never pooled.

## F. Gemma 4 AR sPTC

Alex Zhang's speculative programmatic tool calling, pinned at
`alexzhang13/spec-ptc@9b78b7d6ceeaf8afd1557c4e3a999ce653fc0e17`
(`experiments/manifests/vendor_sources.json`).

What is reused: the shadow-runtime/future **semantics** — start external work from an
incomplete stream, represent it as a future, reconcile against the authoritative final
program.

What is deliberately **not** reused: the canonicaliser, the effect registry, the future
table and the tool implementations. Those are ours, because the comparison is only
meaningful if the two arms differ solely in where the speculation signal comes from (an
irreversible AR prefix versus a revisable diffusion canvas). Reusing two different
canonicalisers would confound identity matching with scheduling.

F may replay E's generations rather than spending new model turns, but only when replay
validity is demonstrable: the AR token stream is monotonic and fully recorded, so a
prefix-time policy is a pure function of that recording. The harness asserts this by
re-deriving E's own launch decisions from the recording and checking they match the online
decisions exactly; if the assertion fails, F is run online and the failure is reported.

## G. CrystalCall / DenoisePTC

Unmodified DiffusionGemma decoding plus the frozen pre-commit gate
(`configs/frozen_policy.json`): valid AST, statically closed arguments, `Pure` or
`SnapshotRead` effect, identical canonical call for `k` consecutive revisions, span
confidence above `tau`, sufficient expected remaining decode, compatible resource class,
no duplicate future. Results are quarantined and released only on an exact
`(request_id, canonical_tool, canonical_arguments, immutable_snapshot_id)` match.

## H. Oracle earliest invariant

Retrospective upper bound: launch at the first revision after which the canonical call
never differs from the final call (`invariant_ns`). Not implementable online; it bounds
how much headroom a better gate could recover.

## I. Oracle perfect final-match classifier

Looser upper bound: launch at the earliest executable final-matching candidate
(`first_exec_ns`), even if the candidate later temporarily disappears. Reported to show
the cost of the persistence requirement.

## Ablations replayed from the same standard traces

From `POLICY_LIBRARY`: `syntax_effect_only`, `persistence_only`, `confidence_only`,
`persistence_confidence`, `full_gate`, `fixed_step_2`, `random_matched` (launch-rate
matched, deterministic per-identity hash so the ablation is reproducible), plus the two
oracles. Swept: `k in {1,2,3,4}`, development-selected `tau` grid, tool duration in
{100, 500, 2000, 5000} ms, task category, first-canvas versus multi-canvas calls, name
versus argument versus full-call stability, CPU/storage/GPU resource class, and adaptive
stopping on/off on a small development subset.

No policy ablation re-runs the model.
