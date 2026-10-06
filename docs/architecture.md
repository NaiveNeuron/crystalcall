# Architecture

```text
                                 benchmark prompt (frozen manifest)
                                              |
                                              v
   +----------------------------------  OMP-Lite  ----------------------------------+
   |  omp 18.1.4, --mode rpc, --profile crystalcall, --config configs/omp-lite.yml  |
   |  --tools eval, one fresh session per prompt, one fixed system prompt           |
   +---+---------------------------------------------------------------+------------+
       |                                                              |
       | OpenAI-compatible stream (authoritative)                      | host_tool_call
       v                                                              |   (authoritative
   instrumented vLLM DiffusionGemma                                   |    tool dispatch)
   +-------------------------------------------+                      |
   |  entropy-bound sampler, canvas 256        |                      |
   |  patches/vllm-trace.patch (every arm)     |                      |
   |  patches/vllm-preact.patch (P-ReAct arm,  |                      |
   |    CRYSTALCALL_PREACT=1, off by default)  |                      |
   +---+---------------------------------------+                      |
       |                                                              |
       | private CanvasRevision side channel (NDJSON over loopback,   |
       |   schema crystalcall-revision/2; `tool_call_ready` only when |
       |   priority decoding is on)                                   |
       v                                                              v
   RevisionSink  ->  CanvasDecoder  ->  PtcExtractor  ->  canonicalize  ->  HostToolRouter
                                              |                              |
                                              v                              | exact-key
                                        RevisionTracker (launch gate)         | claim
                                              |                              |
                                              v                              v
                                    FutureTable (quarantined)  <-------------+
                                              |
                                              +--> Pure / SnapshotRead tool workers
```

Two logically separate streams leave the server. The **authoritative** stream is the ordinary
OpenAI-compatible completion that OMP consumes and acts on. The **revision** stream is private
telemetry consumed only by CrystalCall; nothing on it ever enters the model context, the OMP
transcript, the user-visible output, a downstream tool, or any persisted state a model later
reads.

## The authoritative path

The model's action surface is Python. It emits one native tool call to OMP's `eval` tool
carrying a program; OMP runs that program in its retained Python kernel; the program calls
`tool.<name>(...)`, which is OMP's own loopback tool bridge
(`POST $PI_TOOL_BRIDGE_URL/v1/tool`); because our benchmark tools were registered over RPC with
`set_host_tools`, the bridge dispatch round-trips back to this process as a `host_tool_call`
frame. `HostToolRouter` is therefore the single choke point through which every authoritative
tool invocation passes, which is exactly where canonicalisation, the exact-key claim and the
per-call verification live.

Verified empirically end to end without a GPU by `scripts/smoke_omp_ptc.py`: OMP-Lite ->
native tool call -> Python kernel -> `tool.math_gcd(num1=12, num2=18)` -> `host_tool_call` ->
canonicalise -> claim of a pre-launched future -> `host_tool_result` -> closing turn.

## The speculative path

| Stage | Module | Guarantee |
| --- | --- | --- |
| receive | `sidechannel.py` | one `CanvasRevision` per denoising step; bounded queue, drop-on-full accounting, never blocks the sampler |
| decode | `tokens.py` | canvas token ids -> text with exact per-token character offsets (piecewise, validated against `decode`, exact quadratic fallback) |
| parse | `parse_ptc.py` | real `ast` parse; only top-level, unconditional, statically-closed, allow-listed calls survive; every rejection carries a machine-readable reason |
| identify | `canonicalize.py` | one canonicaliser for speculation, authoritative execution, replay, baselines and tests; total-or-reject, never silent repair |
| gate | `scheduler.py` | AST validity, static closure, eligible effect, `k` consecutive revisions, span confidence above `tau`, sufficient expected remaining decode, compatible resource class, no duplicate future |
| launch | `futures.py` | quarantined future keyed by `(request_id, canonical_tool, canonical_arguments, immutable_snapshot_id)` |
| claim | `omp_bridge.py` | exact-key claim only; otherwise execute normally; every unmatched future discarded |

## Why the result is lossless

Let `P` be the final generated program, `R` the ordinary OMP runtime and `R_s` the same runtime
with speculation. The invariant `Obs(R(P)) = Obs(R_s(P))` rests on six properties, each of which
is enforced mechanically rather than asserted:

1. **Only eligible effects are launched.** `FutureTable.launch` raises on anything that is not
   `Pure` or `SnapshotRead`, and `assert_no_write_speculation` guards every launch site.
   Tested: `test_effect_safety.py`.
2. **Speculative results are unreachable without an exact key.** The payload lives inside
   `_Quarantined`, whose `release` requires the full four-part key and whose `__str__` and
   `__iter__` raise. The trace stores digests and timings, never values.
   Tested: `test_claim_discard.py`.
3. **Identity is canonical, not textual.** Equivalent spellings collide; inequivalent calls
   never do; a different snapshot is a different key. Tested: `test_canonicalize.py`.
4. **Cancellation is an optimisation, not a safety property.** A speculation that ignores
   cancellation and completes after invalidation still leaves nothing observable.
   Tested: `test_effect_safety.py::test_cancellation_failure_creates_no_semantic_effect`.
5. **Decoding is untouched.** The patch only *writes* already-computed tensors into trace
   buffers and copies them out; `scripts/verify_instrumentation.py` proves byte-identical
   completions with tracing on versus off, against a baseline-versus-baseline control that
   accounts for the sampler's own stochasticity.
6. **Every claim is checked.** With `verify_claims`, each claimed result is re-executed
   normally and the digests compared. This holds the model fixed by construction, so a mismatch
   can only come from scheduling. An unverified or erroring claim is *not* evidence of
   losslessness and makes gate S6 fail closed.

## Trajectory scoping

An OMP turn issues several model requests: one that emits the program, then one that writes the
closing prose after the tool results return. Only the program-producing request is the analysed
trajectory, and its end is the authoritative execution point. Later requests restart
`canvas_id` and `step` at 0, so merging them would collide canvases and let prose revisions
perturb the program trajectory; their identity and timing are still recorded in
`meta.model_requests`. Tested: `test_trace_join.py`.

## Commitment granularity

DiffusionGemma renoises unaccepted canvas positions in place and only emits a canvas when the
request converges, so a token is not irrevocably committed mid-canvas. Both readings are
recorded separately for every final call — the first revision in which every token of the call's
span lies inside the entropy-bound accepted set (`committed_step`), and the containing canvas's
commit (`canvas_done_ns`) — so the earliest-committed baseline can be reported under either, and
the primary reading is the *earlier* one, which is the harder baseline to beat.

## Offline replay

Every policy comparison and latency regime is a pure function of the stored traces, so nothing
except the four online arms costs model time. `replay.py` recomputes the crystallization
milestones, resolves each policy's launch time, and simulates end-to-end completion with the
measured instrumentation cost charged to every arm that observes revisions.
