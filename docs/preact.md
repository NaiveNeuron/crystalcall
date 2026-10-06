# P-ReAct-style tool-priority diffusion decoding (mandatory baseline arm D)

This is the reproduction of DLLM-Searcher's **P-ReAct** tool-priority decoding
as faithfully as its public description permits, labelled **P-ReAct-style**
everywhere because we do not have their implementation. It is a
within-canvas decoding-priority mechanism, not a prompt change: OMP, prompts,
tool schemas, canonicalisation, tool implementations and retry policy are
**fixed** across arms. The arm runs `configs/system_prompt.md`, byte-identical
to every other arm; the only difference is one environment variable on the
server.

---

## 1. What the mechanism does

The intervention lives inside the denoising loop and has exactly two effects.
Both are **union-only**, and neither touches convergence.

### Span detection

At every denoising step, over the **predicted-clean canvas** (`argmax_canvas`,
i.e. `argmax` of the current step's processed logits — the thing the sampler
would commit):

* `lo` = index of the **first** `<|tool_call>` token;
* `hi` = index of the **first** `<tool_call|>` token strictly after `lo`, plus
  one.

The span is the half-open range `[lo, hi)` and it **covers both delimiters**. A
closer that appears before any opener is ignored; a second opener is ignored. If
there is no opener, or no closer after the opener, the mechanism is inert for
that step.

### Effect 1 — priority acceptance (`span_accept`)

The entropy-bound sampler accepts a position when the **cumulative** entropy
rule holds (sort by per-token entropy, accept the prefix whose
`cumsum - cummax <= entropy_bound`, with `entropy_bound = 0.1` nats over the
whole 256-token canvas). Inside the span, a position is **additionally**
accepted when its own entropy is `<= span_entropy_bound` (default **0.6 nats**,
a deliberately looser *per-token* bound — that looseness is what "priority"
means here). Outside the span, acceptance is exactly the ordinary rule.

The result is a union: a position the ordinary rule accepted is **never**
un-accepted. Accepted positions keep their sampled token instead of being
renoised with Gumbel noise, so the tool-call span settles in fewer steps while
the rest of the canvas keeps denoising normally.

### Effect 2 — readiness signalling (`ready_signal`)

When the span has been **byte-identical for `require_stable_steps` consecutive
denoising steps** (default 1, matching the sampler's own
`stability_threshold = 1`) **and** every in-span position clears
`span_entropy_bound`, the sampler emits exactly one side-channel event for that
`(request_id, canvas_id)`:

```json
{"kind":"tool_call_ready","request_id":"...","canvas_id":0,"step":11,
 "monotonic_ns":123456789,"span":[41,73],"span_stable":true}
```

`span` is the same half-open `[lo, hi)`, so the offline replay can reparse the
revision recorded at that `step` and reconstruct exactly the launched call. The
executor may launch the parsed call on this event while generation continues.

### What the mechanism deliberately does NOT do

It **never overrides convergence**. `converged` ends denoising, and
DiffusionGemma then commits the *whole* argmax canvas — including out-of-span
positions that are still unaccepted and unstable. Forcing convergence when the
span settles would (a) write garbage into the authoritative assistant turn,
(b) truncate exactly the reasoning P-ReAct exists to let continue, and (c) make
the baseline look fast for an illegitimate reason. Scheduling may change; final
action semantics may not, and the baseline is held to the same standard as
CrystalCall itself.

This is mechanised, not merely documented — see §6.

### Modes

`CRYSTALCALL_PREACT_MODE` selects `span_accept`, `ready_signal`, or `both`
(default, and what the arm runs). The decomposition exists so the ablation table
can attribute the effect to acceptance priority versus early launch.

---

## 2. Deviations from the paper

Every one of these is a deviation forced by the public description being a
description, not an implementation. They are listed so a reader can weigh
them, and they are why the arm is labelled `P-ReAct-style`.

| # | Paper (as described) | Here | Why |
|---|---|---|---|
| 1 | "Guided decoding prioritises the tool-call instruction" — the guidance signal is not specified | A **looser per-token entropy acceptance bound inside the detected tool-call span** | The pinned sampler's only token-selection knob is the entropy-bound acceptance mask. Loosening it in-span is the smallest faithful realisation of "this span gets priority" that reuses the model's own uncertainty signal. |
| 2 | Tool-call region presumably identified by the guidance model / decoding constraint | Identified by the **checkpoint's own in-band delimiter token ids** in the predicted-clean canvas | No constrained-decoding grammar is available in the pinned build, and `--tool-call-parser gemma4` already makes those delimiters the wire format. |
| 3 | Priority may reorder generation so the call is produced first | **No reordering at all**; the span is prioritised wherever the model puts it | A diffusion canvas has no generation order to reorder. Forcing position would be a prompt/format intervention, which would break the prompt shared by every arm. |
| 4 | The call becomes "available while reasoning continues" | Readiness is a **side-channel event**; the canvas is not committed early and denoising is not shortened | See §1: committing early corrupts the turn. Availability is delivered by the event, which is exactly what "available while reasoning continues" requires. |
| 5 | No stability criterion described | The span must be byte-identical for `require_stable_steps` steps | A diffusion canvas revises; without a stability test the "ready" signal would fire on a span that is still moving. Default 1 matches the sampler's own `stability_threshold`. |
| 6 | Bound value not given | `span_entropy_bound = 0.6` nats, in `configs/preact.yml` | Chosen as a clearly-looser-than-0.1 per-token bound (≈ 0.75 top-1 probability on a two-way ambiguity), overridable per run and swept in the ablation. |
| 7 | Their model, prompt and tools | **Our** model, prompt, tool schema and executor, unchanged from every other arm | Every arm uses the same model, prompt, tool schema and executor; that is the point of the comparison. |

---

## 3. The delimiter token ids

Resolved from the pinned checkpoint's own tokenizer, never guessed and never
defaulted:

| token | id |
|---|---|
| `<\|tool_call>` (opener) | **48** |
| `<tool_call\|>` (closer) | **49** |

Note the asymmetric pipe placement. Recorded for provenance in
`configs/preact.yml` together with `<\|tool>` 46, `<tool\|>` 47,
`<\|tool_response>` 50, `<tool_response\|>` 51, `<\|turn>` 105, `<turn\|>` 106.

* source: `~/.cache/huggingface/hub/models--google--diffusiongemma-26B-A4B-it/snapshots/f7f5b7f5fa82ffc52addd066915886d497f5517b/tokenizer.json`
* tokenizer sha256: `cc8d3a0ce36466ccc1278bf987df5f71db1719b9ca6b4118264f45cb627bfe0f`
* regenerate / verify:

```
uv run scripts/resolve_gemma_tokens.py            # print JSON
uv run scripts/resolve_gemma_tokens.py --write    # update configs/preact.yml
uv run scripts/resolve_gemma_tokens.py --check    # fail if the yml is stale
```

`tests/test_preact.py` re-resolves the ids from the live checkpoint and compares
them (and the tokenizer sha256) against `configs/preact.yml`, so a checkpoint
swap fails the suite instead of silently producing a span that never matches —
a vacuous arm that would still look like a real measurement.

---

## 4. Where the code lives

| file | role |
|---|---|
| `src/crystalcall/preact.py` | Reference implementation: pure Python, no torch, unit-tested. `PreactConfig`, `find_tool_call_span`, `priority_accept_mask`, `span_is_accepted`, `span_is_stable`, `should_signal_ready`, `ReadyLatch`. |
| `patches/preact_priority.py` | The same rules as vectorised torch, mounted into the container as `crystalcall_preact`. Stdlib + torch only. Carries a differential `--self-test` against the reference. |
| `patches/vllm-preact.patch` | 9 hunks against the **trace-patched** sampler. |
| `configs/preact.yml` | The token ids, the defaults, the environment variables and the exact server commands. |
| `scripts/resolve_gemma_tokens.py` | Resolves the ids from the checkpoint. |
| `scripts/smoke_preact.py` | The GPU validation: semantic health, not timing. |
| `tests/test_preact.py` | GPU-free tests, including the two guarantees in §6. |

### Patch layout and hashes

Two patches applied in order by `patches/apply_patch.sh`, both landing in one
file, because the same image and the same file must serve every arm:

| artifact | lines | sha256 |
|---|---|---|
| `patches/build/diffusion_gemma.py.orig` (pristine, from the image) | 1369 | `5974f9c1e1090ff528e2ebe76a949c48feb696ed49884ad8bcf880ccbccb0486` |
| `patches/build/diffusion_gemma.py` (trace + preact) | 1780 | `7f39041b0fc887723a64d2084d8a5ba2b6299b55bc94e0f2e10306a89aa9d464` |
| `patches/vllm-trace.patch` | 8 hunks | `905943c16434821bbcc80a3987dc1b9dcb7cae5b40ef1379f2f137d6362bcedf` |
| `patches/vllm-preact.patch` | 9 hunks | `4c20f483850bd90797f7bf744b5cf410e0d5a48226730f88b69add1321f9be18` |

Two patch files rather than one, on purpose: the trace patch is load-bearing for
*every* arm, while the preact patch is arm-specific and can be inspected and
reverted on its own.

The nine preact hunks: the `crystalcall_preact` import + no-op stub; the eleven
extra `_compiled_sample_step` parameters; the priority call inside the compiled
step; the four persistent buffers in `DiffusionGemmaRequestStates.__init__`; the
per-slot reset in its `add_request`; the settings resolution in
`DiffusionSampler.__init__`; the `_emit_preact` method; the call-site kwargs;
and the emit call.

### Cost when the mechanism is off

`crystalcall_preact.enabled()` is False, so no buffer is allocated, every added
argument is `None`/`False`, and the compiled step's `if preact_span_hist is not
None:` guard keeps every added op out of the traced graph — the same technique
the trace hunks already use. The CrystalCall / standard arm therefore runs the
identical file with the mechanism compiled out.

### Cost when it is on

Span detection is two `==` comparisons plus a masked `amin` over the position
index; stability is an unrolled `S`-step ring compare (the same pattern upstream
uses for `accepted_canvas_history`); acceptance is one `|`. Every input is a
tensor the compiled step already holds (`argmax_tokens`, `token_entropy`,
`eb_mask`) — there is **no logits copy, no probability transfer and no host
synchronisation** inside the compiled region. Emission costs one
`[num_decode, 3]` int copy per step: three ints per request, never a
vocabulary-sized transfer, and never a full-vocabulary transfer off the GPU.

---

## 5. Running the arm

```
# build the patched sampler (both patches) once
bash patches/apply_patch.sh
bash patches/apply_patch.sh --check     # re-verify the dry runs only

# CrystalCall / standard arm -- mechanism OFF (the default)
SEED=12345 CRYSTALCALL_TRACE=1 CRYSTALCALL_TRACE_SINK=127.0.0.1:8765 \
  bash scripts/serve_diffusiongemma.sh

# P-ReAct-style arm -- mechanism ON
SEED=12345 CRYSTALCALL_TRACE=1 CRYSTALCALL_TRACE_SINK=127.0.0.1:8765 \
  CRYSTALCALL_PREACT=1 CRYSTALCALL_PREACT_MODE=both \
  bash scripts/serve_diffusiongemma.sh

bash scripts/serve_diffusiongemma.sh --stop
```

Environment variables (the container does **not** inherit host env; the serve
script forwards each of these explicitly, and
`scripts/serve_diffusiongemma.sh --print-docker-args` renders the argv so the
test can prove it):

| variable | default | meaning |
|---|---|---|
| `CRYSTALCALL_PREACT` | `0` | master switch |
| `CRYSTALCALL_PREACT_MODE` | `both` | `span_accept` \| `ready_signal` \| `both` |
| `CRYSTALCALL_PREACT_BOUND` | `0.6` (from `configs/preact.yml`) | in-span per-token entropy bound |
| `CRYSTALCALL_PREACT_STABLE` | `1` (from `configs/preact.yml`) | `require_stable_steps` |
| `CRYSTALCALL_PREACT_OPEN_ID` | `48` (from `configs/preact.yml`) | opener token id |
| `CRYSTALCALL_PREACT_CLOSE_ID` | `49` (from `configs/preact.yml`) | closer token id |

`harness.ARMS["preact_style"].sampler_env` sets the first two; the serve script
fills the rest from `configs/preact.yml` and **fails loudly** if an id is
missing or not an integer. The in-container module has no id fallback and raises
at configuration time rather than running with a plausible-looking wrong id.

---

## 6. The two guarantees, mechanised

`tests/test_preact.py` asserts both, so they cannot rot into documentation:

1. **The CrystalCall arm's decoding is untouched.** With `enabled=False` every
   reference function is the identity on its inputs: the mask comes back equal
   to the base mask, no span is found, readiness is never signalled, and the
   latch never fires.
2. **Nothing can override convergence.** No public function takes or returns a
   convergence/commit flag; no added line in `patches/vllm-preact.patch`
   mentions `is_encoder_phase` or assigns/ORs `converged`; the generated
   `patches/build/diffusion_gemma.py` still contains the upstream convergence
   expression byte-for-byte, has exactly one assignment to `converged`, and no
   `preact*` line touches either symbol; and `priority_step` returns only an
   acceptance mask.

Plus: the rendered container argv carries all six `CRYSTALCALL_PREACT*`
variables, and the config ids still match the live checkpoint tokenizer.

---

## 7. Verification performed

| check | command | result |
|---|---|---|
| Reference semantics, config, inertness, no-convergence, argv rendering | `uv run --project . python -m pytest tests/test_preact.py -q` | 35 passed |
| Both patches apply to the image's pristine file, host and in-image | `bash patches/apply_patch.sh --check` | exit 0 |
| Tensor form == reference, 600 randomised cases over all 3 modes, 2 stability depths, with canvas boundaries | `docker run --rm -v "$PWD:/repo:ro" -e PYTHONPATH=/repo/src --entrypoint python3 vllm/vllm-openai:gemma /repo/patches/preact_priority.py --self-test` | `PREACT SELF-TEST PASS` (mask 600, span 600, ready 600, fires 10) |
| Emitter self-test under the `/2` schema | `python3 patches/crystalcall_trace.py --self-test` | `SELF-TEST PASS`, exactly the four `/1` kinds on a non-preact run |
| Token ids match the checkpoint | `uv run scripts/resolve_gemma_tokens.py --check` | exit 0 |

### GPU smoke test

```
uv run scripts/smoke_preact.py --json experiments/evidence/preact_smoke.json
```

It brings the server up twice from the same image and patched file (preact off,
then on), with the same prompt, tool surface and seed, and **asserts semantics,
never timing**:

1. the preact arm's final assistant turn contains a parseable, complete tool
   call whose name is in the declared surface and whose arguments are a JSON
   object;
2. the closing prose turn is non-empty — the turn was not truncated;
3. the turn is structurally comparable to the standard arm's on the same
   prompt: same set of called tool names, same `finish_reason`, and prose that
   has not collapsed;
4. at least one `tool_call_ready` arrived, at most one per
   `(request_id, canvas_id)`, and its `span` re-derives exactly from the
   revision recorded at that `step`;
5. the standard arm emitted **zero** `tool_call_ready` events, which is the
   runtime proof that the added code is inert for the CrystalCall arm.

---

## 8. Side-channel schema

`patches/crystalcall_trace.py` reports `crystalcall-revision/2`. `/2` is
`/1` plus the optional, purely additive `tool_call_ready` kind; a producer that
never emits it (every non-preact run) is still a valid `/2` producer, because
the version describes the producer's capability, not the run. The consumer half
lives in `src/crystalcall/sidechannel.py` (`RequestStream.ready`).

### Guarantees the offline `preact_ready` policy joins on

* **`step` always names a revision emitted in the same step.** `_emit_preact`
  is called immediately after `_emit_trace`, with the *same* pre-step index
  snapshot (`trace_step_pre`), under the same `self._trace_on and _TRACE_SLOTS`
  guard, so the `revision` for that step is enqueued **before** the
  `tool_call_ready` for that step. The emitter queue is FIFO with one consumer,
  so a consumer that has the readiness event has already seen the revision
  unless the revision itself was dropped under back-pressure —
  `scripts/smoke_preact.py` fails loudly on exactly that case ("no revision at
  step N"), so it can never degrade into a silently slow arm.
* **The latch is per `(request_id, canvas_id)`, not per request.** `fired` is
  cleared on every commit step (the compiled step writes
  `torch.where(is_denoise, prev_fired | now, False)`) and again in
  `DiffusionGemmaRequestStates.add_request`, and `canvas_id` advances only on
  commit — so a multi-canvas response signals once per canvas, and readiness
  can never fire on a commit step (where `is_denoise` is False).
* **Run the arm at `CRYSTALCALL_TRACE_LEVEL=full`.** At `nopayload` the
  revisions carry empty token arrays, so the offline reparse at `step` has
  nothing to read. `full` is the serve-script default and what
  `scripts/smoke_preact.py` sets.
* Semantics the offline side applies, recorded here so the two descriptions
  match: a readiness launch whose parsed call is absent from the final program
  counts as a **false launch** (precision < 1 for row D), and candidates are
  filtered by `speculation_eligible`, so no `Write` / `LiveRead` tool is ever
  launched from a readiness signal.