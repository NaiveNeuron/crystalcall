# DiffusionGemma sampler instrumentation

How CrystalCall observes every predicted-clean canvas revision inside the
native vLLM DiffusionGemma sampler, and the evidence that observing it does not
change what the model decodes.

## 1. Pinned environment

| item | value |
| --- | --- |
| image | `vllm/vllm-openai:gemma` |
| image digest | `sha256:9c719fc0c869092c7d0533f8357d6985a38d5ff03b20ffb6a4620c2b4806dd4b` |
| image id | `sha256:de71fdd23e93ac46abf251e56992fd91547746010e0256c32551b6660a879589` |
| vLLM version | `0.22.1rc1.dev357+g74b5964f0` |
| site-packages | `/usr/local/lib/python3.12/dist-packages` |
| patched file | `vllm/model_executor/models/diffusion_gemma.py` (1369 lines pristine, 1843 patched) |
| model | `google/diffusiongemma-26B-A4B-it` rev `f7f5b7f5fa82ffc52addd066915886d497f5517b` |
| architecture | `DiffusionGemmaForBlockDiffusion`, resolved by the engine |
| attention backend | `TRITON_ATTN`, forced by the engine (heterogeneous head dims: `head_dim=256`, `global_head_dim=512`; FA4 unavailable) |
| host | NVIDIA GB10 (sm_121), driver 580.173.02, CUDA 13.0, aarch64, 20 cores, 121 GB unified |
| cold load | ~470 s to `GET /v1/models` → 200 |

File digests (sha256, first 24 hex):

```
0567c04dc13fcd11f18a63b1  patches/build/diffusion_gemma.py        (patched, mounted at run time)
ffebec8f079124d6c39eccfd3  patches/vllm-trace.patch
438a8f1ea0c3a6b93ba60da78b3bc4e6  patches/crystalcall_trace.py
```

**The image's sampler is an older revision than upstream `main`.** The
reference copy fetched from `main` (1425 lines) differs from the image's copy
(1369 lines) in 33 hunks: the image build has no tile loop in
`DiffusionSampler.__call__` (one `_compiled_sample_step` call per step, not one
per tile), no tensor-parallel vocab sharding in the self-conditioning matmul, no
`top_k`/`top_p` application, `sampled.zero_()`/`num_sampled.zero_()` moved
*inside* the compiled step, `init_canvas` taking numpy indices, and
`compute_topk_logprobs` instead of `compute_topk_scores`. The patch in this repo
is derived against **the image's** file, which is authoritative.

## 2. Delivery mechanism: run-time bind mounts

`patches/apply_patch.sh` extracts the pristine sampler from the image, applies
`patches/vllm-trace.patch` into `patches/build/diffusion_gemma.py`, and records
the image metadata in `patches/build/image-env.sh`. The serve script then
bind-mounts, at container start:

```
-v patches/build/diffusion_gemma.py:/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/diffusion_gemma.py:ro
-v patches/crystalcall_trace.py:/opt/crystalcall/crystalcall_trace.py:ro
-e PYTHONPATH=/opt/crystalcall
```

No derived image is built. Reasons: the pinned image is ~21 GB, a derived layer
would copy the site-packages layer on every patch edit, and — the decisive
one — the baseline and traced arms must be the *same* binary path. With bind
mounts, both arms run the identical image and the identical patched file, and
differ only in `CRYSTALCALL_TRACE`. With tracing off the emitter allocates no
thread, no queue and no socket; the sampler allocates no trace buffers (the
attributes are `None`), `torch.compile` traces the upstream graph because the
extra tensor arguments are `None`, and no device→host copy is issued.

## 3. Patched symbols

All eight hunks, in file order:

| # | symbol | change |
| --- | --- | --- |
| 1 | module scope, after `logger = init_logger(__name__)` | `import crystalcall_trace` with an inert stub fallback on `ImportError`; `_TraceRequest` (per-slot `req_id`/`canvas_id`/`prefix_len`); `_TRACE_SLOTS: dict[int, _TraceRequest]` |
| 2 | `_compiled_sample_step` signature | four optional trace-output tensors appended: `trace_accept_mask`, `trace_token_conf`, `trace_token_entropy`, `trace_mean_entropy`, all defaulting to `None` |
| 3 | `_compiled_sample_step` body, immediately before `return scaled` | guarded by `if trace_accept_mask is not None:` — writes `eb_mask`, `probs.max(dim=-1).values`, `token_entropy`, `mean_entropy` into those buffers at `decode_slots`. No existing statement is moved, no RNG is consumed, `return scaled` is untouched |
| 4 | `DiffusionGemmaRequestStates.__init__` | allocates `trace_accept_mask` `[max_num_reqs, CL]` bool, `trace_token_conf` / `trace_token_entropy` `[max_num_reqs, CL]` fp32, `trace_mean_entropy` `[max_num_reqs]` fp32 — **only** when `crystalcall_trace.enabled()`; otherwise the four attributes are `None` |
| 5 | `DiffusionGemmaModelState._trace_sampler_info` (new), `.add_request`, `.remove_request` | registers slot → `_TraceRequest`, emits `request_begin` with the prompt length from `new_req_data` and the pinned sampler settings, emits `request_end` on removal. `_warmup_*` request ids are skipped, so warmup traffic never enters the trace |
| 6 | `DiffusionSampler.__init__` | `self._trace_on = crystalcall_trace.enabled()`, evaluated once |
| 7 | `DiffusionSampler._emit_trace` (new) | one fused device→host transfer per step; see below |
| 8 | `DiffusionSampler.__call__` | snapshots `trace_step_pre = states.step[decode_slots].clone()` next to the existing `is_committing` snapshot (the compiled step mutates `states.step`); passes the four trace buffers to `_compiled_sample_step`; calls `_emit_trace` after it returns, guarded by `if self._trace_on and _TRACE_SLOTS:` |

`_emit_trace` uses a persistent ring of pinned host buffers. It issues
non-blocking copies (`non_blocking=True`) for the compact top-1 token IDs,
accepted mask, confidence, entropy and scalar summaries, records a
`torch.cuda.Event`, and hands the buffer to `crystalcall_trace.defer_staged`
with `put_nowait`. No `.tolist()` or event wait runs on the sampler thread.
The stdlib staging worker waits on that event, materialises the pinned arrays,
and queues plain Python events to the separate JSON/socket worker. The ring is
bounded; if it fills, the affected request is marked desynchronised and later
revisions are suppressed rather than emitting a corrupt canvas sequence.

```
tokens  int32 [n, CL]       mask uint8 [n, CL]
conf    float32 [n, CL]     entropy float32 [n, CL]
scalars int32 [n, 3]        mean_entropy float32 [n]
```

No full-vocabulary tensor ever leaves the GPU. At
`CRYSTALCALL_TRACE_LEVEL=nopayload`, the four per-token arrays are not copied;
only the scalar tensors travel. `sample_monotonic_ns` records sampler-side
enqueue time for diagnostics. The producer's event-complete time is the wire
`monotonic_ns`; **live** sink ingestion replaces that with receive time before
live replay or online launch accounting. Raw emitter captures without a
persisted receive timestamp are structural-only for timing purposes.

Field semantics:

* `step` — the **pre-step** denoising index within the current canvas (the
  compiled step increments it on denoise and resets it to 0 on commit, so the
  post-step value would be off by one).
* `converged` — post-step `is_encoder_phase`, i.e. "this request commits next".
* `phase` — `"commit"` when `is_committing` was true on entry, else `"denoise"`.
* `canvas_id` — per-request counter, incremented after each observed commit.
* `prefix_len` — running committed-token count from earlier canvases of the same
  request (`+= valid_canvas_len` at each commit).

## 4. Emitter (`patches/crystalcall_trace.py`)

Stdlib only (`socket`, `json`, `threading`, `queue`, `os`, `time`, `atexit`).
Surface: `ENABLED`, `enabled()`, `want_payload()`, `begin_request`,
`defer_staged`, `emit_revision`, `canvas_done`, `end_request`, `flush`,
`flush_staged`, `stats`, `self_test`.

* The sampler uses bounded `put_nowait` for staging and never waits on the
  event or calls `.tolist()`. A bounded staging worker waits for event
  completion and a separate sender worker handles JSON and socket/file I/O.
* A staging queue or pinned-ring overflow increments `dropped`; the affected
  request is marked desynchronised and subsequent revisions are suppressed.
  Request teardown waits for staged work so a valid request never ends before
  its queued revisions.
* JSON encoding and float rounding (5 decimals) happen on the sender worker;
  tracing off allocates no queue or worker.
* `socket.IPPROTO_TCP, TCP_NODELAY`; reconnect with exponential backoff capped
  at 2 s; while the socket is down events are drained and dropped.
* `CRYSTALCALL_TRACE_FILE` appends the same NDJSON to a path (used by the
  overhead ladder to measure encode cost with no receiver).
* `CRYSTALCALL_TRACE_STATS` snapshots `stats()` as JSON whenever the queue
  drains — diagnostics only, not part of the wire schema.

Environment variables: `CRYSTALCALL_TRACE`, `CRYSTALCALL_TRACE_SINK`,
`CRYSTALCALL_TRACE_LEVEL` (`full` | `nopayload`), `CRYSTALCALL_TRACE_QUEUE`,
`CRYSTALCALL_TRACE_STAGE_QUEUE`, `CRYSTALCALL_TRACE_RING`,
`CRYSTALCALL_TRACE_FILE`, `CRYSTALCALL_TRACE_STATS`.

## 5. Wire format

NDJSON, one object per line, plain TCP to `$CRYSTALCALL_TRACE_SINK`, consumed by
`src/crystalcall/sidechannel.py` (schema `crystalcall-revision/2`). The producer
sets `monotonic_ns` to event-complete host time and includes
`sample_monotonic_ns` for the sampler enqueue time. On live ingestion the sink
preserves the producer value as `producer_monotonic_ns` and replaces
`monotonic_ns` with the host receive timestamp; live replay and launch
accounting consume this conservative receive time. A raw emitter capture has no
receive timestamp unless the capture layer persists `receive_monotonic_ns`, so
`replay_events()` treats such files as structural-replay inputs only.

```json
{"kind":"request_begin","request_id":"...","monotonic_ns":0,"prompt_tokens":18,"sampler":{...}}
{"kind":"revision","request_id":"...","canvas_id":0,"step":0,"monotonic_ns":0,
 "sample_monotonic_ns":0,"predicted_token_ids":[...256...],
 "accepted_mask":[...256...],"token_confidence":[...256...],
 "token_entropy":[...256...],"mean_entropy":0.00185,
 "converged":true,"phase":"denoise","prefix_len":0}
{"kind":"canvas_done","request_id":"...","canvas_id":0,"monotonic_ns":0}
{"kind":"request_end","request_id":"...","monotonic_ns":0,"output_tokens":256,"finish_reason":"removed"}
```

`request_begin.sampler` carries the pins the sampler actually used:
`{schema, sampler:"entropy_bound", entropy_bound, canvas_length,
max_denoising_steps, stability_threshold, confidence_threshold, t_min, t_max,
vocab_size}`.

`finish_reason` is `"removed"`: the sampler observes the request leaving the
batch, not the HTTP-level stop reason. The OpenAI `finish_reason` is available
to the harness from the HTTP response.

## 6. Exact commands

```bash
# 0. emitter self-test (no GPU, no container)
PYTHONPATH=patches python3 -m crystalcall_trace --self-test

# 1. extract the image's pristine sampler, apply the patch, verify it applies
patches/apply_patch.sh            # also runs `patch -p1 --dry-run` inside the image
patches/apply_patch.sh --check    # dry-run only

# 2. serve (env selects the arm; same image, same patched file both ways)
scripts/serve_diffusiongemma.sh                       # baseline, tracing off
CRYSTALCALL_TRACE=1 CRYSTALCALL_TRACE_SINK=127.0.0.1:9099 \
  scripts/serve_diffusiongemma.sh                     # traced
scripts/serve_diffusiongemma.sh --stop
scripts/serve_diffusiongemma.sh --logs --tail=100
scripts/serve_gemma4.sh                               # autoregressive control arm

# 3. proof, in this order
uv run scripts/verify_instrumentation.py --arm baseline   # tracing off  -> docs/evidence/baseline.json
uv run scripts/verify_instrumentation.py --arm control    # tracing off  -> docs/evidence/control.json
uv run scripts/verify_instrumentation.py --arm traced     # tracing on   -> docs/evidence/traced.json
```

Each arm restarts the container itself, so each arm starts from a fresh RNG
stream. `--reuse-server` reuses an already-running server, `--keep-server`
leaves it up.

## 7. Emitter self-test

```
$ PYTHONPATH=patches python3 -m crystalcall_trace --self-test
[full] {"arrays_ok": true, "bytes": 275038, "bytes_per_revision": 5729, "dropped": 0,
        "encode_ns_per_event": 262970, "errors": 0, "expected_events": 51, "file_lines": 51,
        "malformed": 0, "ok": true, "queue_max": 50, "received_events": 51,
        "received_kinds": {"canvas_done": 1, "request_begin": 1, "request_end": 1, "revision": 48},
        "sent": 51, "steps_ok": true}
[nopayload] {"bytes": 2391, "bytes_per_revision": 298, "dropped": 0, "empty_arrays": true,
             "events": 10, "ok": true, "revisions": 8}
[drop] {"dropped": 200, "emit_ns_per_call": 3767, "errors": 0, "ok": true, "sent": 0}
[off] {"calls": 20000, "ns_per_call": 74.478, "ok": true, "worker_started": false}
SELF-TEST PASS
```

Four cases: a full round trip through a throwaway TCP listener *and* a file
sink; `nopayload` (empty per-token arrays); a dead sink (200 events dropped, no
block, no raise); and tracing off (no worker thread ever started). The staging
worker is additionally exercised by `tests/test_trace_transfer.py` with a
fake CUDA event and production-shaped pinned-buffer dictionary.

### In-container transport check (no GPU required)

The emitter also passes its self-test *inside* the pinned image, and the
container→host path was verified against the frozen consumer:

```bash
docker run --rm --network host --entrypoint python3 \
  -v "$PWD/patches/crystalcall_trace.py:/opt/crystalcall/crystalcall_trace.py:ro" \
  -e PYTHONPATH=/opt/crystalcall vllm/vllm-openai:gemma -m crystalcall_trace --self-test
# SELF-TEST PASS  (in-image python 3.12; encode 263.8 us/event, 0 drops)
```

and, emitting from inside the container into a host-side
`crystalcall.sidechannel.RevisionSink`:

```
received revisions: 4  prompt_tokens: 21  output_tokens: 256  closed: True
phases: ['denoise','denoise','denoise','commit']  steps: [0,1,2,3]  n: {256}
canvas_done: dict_keys([0])  sink stats: {"events": 7, "malformed": 0}
CONTAINER->HOST TRANSPORT OK  skew_ns: median 1.71 ms
```

This confirms the mount layout, the in-image interpreter, `--network host`
loopback reachability, that `RevisionSink` parses the emitter's bytes into
`CanvasRevision` objects unmodified, and that `CLOCK_MONOTONIC` is shared with
the host (1.7 ms apparent skew = transport latency, not a clock offset).

## 8. Observational purity

**The sampler is not deterministic.** `_compiled_sample_step` renoises the
canvas with Gumbel noise (`torch.rand_like` → `-log(-log u)`) and reinitialises
it with `torch.randint`; the temperature comes from the diffusion schedule
(`t_min`/`t_max`), **not** from the request's `temperature`, which the diffusion
path ignores. So `temperature=0` alone does not make two runs comparable. The
serve script therefore pins `--seed 12345`, and the proof uses three arms:

| arm | tracing | purpose |
| --- | --- | --- |
| baseline | off | reference completions |
| control | off | how much of the reference a *restarted uninstrumented* server reproduces |
| traced | on | must reproduce the reference on everything the control arm showed is reproducible |

Measured (4 prompts; indices 0–2 are short single-canvas prompts, index 3 is a
~640-token prompt included to exercise `canvas_id > 0` / `prefix_len > 0`):

* **control arm**: indices `[0, 1, 2]` reproduced byte-for-byte; index `3`
  diverged **with tracing off**. This build is not bit-reproducible over ~640
  tokens across restarts even uninstrumented (MoE reductions plus async
  scheduling); such requests carry no information about the instrumentation, so
  the verifier asserts identity only on the control-reproducible set and reports
  the rest as warnings.
* **traced arm**: indices `[0, 1, 2]` were byte-identical to the baseline —
  `"Paris"`, `"51"` and `"Red, blue, blue"` — and the long request was the
  expected uninstrumented restart divergence. The traced arm emitted 64
  validated revisions with zero drops.

Evidence: `docs/evidence/baseline.json`, `control.json`, `traced.json`.

Exact PASS line emitted by the traced arm:

```
PASS: completions byte-identical with tracing off vs on for requests [0, 1, 2]; 64 revisions validated; 0 drops.
```

**Provenance of these three files, stated exactly.** `baseline.json`,
`control.json` and `traced.json` contain the three separate container starts
of the same pinned image and patched file, with `--seed 12345`, tracing
off / off / on. Each file carries a `source_provenance` block — the git commit
plus SHA-256 of the patched sampler, the trace patch and the emitter — and all
three carry the same `git_commit` and `sampler_sha256: 0567c04d…`, so the arms are
pinned to one source state rather than asserted to be. The gate is enforced,
not decorative: the control and traced arms exit 1 if the evidence they consume
has no provenance block or different digests. A differing `git_commit` alone
does not gate — host-side docs and tests cannot change engine behaviour. The
traced run exited 0 with zero errors and zero warnings on the
control-reproducible set; its long-request warning is the same restart
nondeterminism seen by the control arm.

The sink timestamps every received event at `_ingest` and preserves the
producer timestamp as a diagnostic. The verifier's stream checks remain:
256-length `predicted_token_ids`, `accepted_mask`, `token_confidence`,
`token_entropy`; token ids in `[0, 262144)`; entropies in `[0, ln 262144]`;
confidences in `[0, 1]`; `step` exactly `0,1,2,…` within a canvas and reset at
each `canvas_id`; at most one `commit` revision per canvas and it is last;
`converged` set on the final denoise step and on no earlier one; denoise steps
per canvas ≤ `max_denoising_steps=48`; `canvas_done` present for every
committed canvas; `prefix_len` equal to `256 × (number of earlier commits)`;
`request_begin` / `request_end` present with `prompt_tokens > 0`. These checks
were exercised against deliberately broken streams (dropped step, short array,
wrong `prefix_len`, out-of-range entropy) and each was caught.


## 9. Measured cost

The native fidelity run emitted 64 revisions over four
requests with zero drops and zero malformed frames:

| metric | value |
| --- | --- |
| bytes per revision (on the wire, NDJSON) | **5160.5 B** |
| encode time per event (JSON + rounding, sender worker) | **340.4 µs** |
| total encoded | 330 271 B in 26.55 ms of sender CPU |
| dropped | **0** |
| emitter errors | 0 |
| max sender queue depth observed | **1** (of 4096) |
| socket reconnects | 1 (the initial connect) |
| sink malformed lines | 0 |
| producer→host receive skew | median 0.68 ms, max 1.50 ms |
| `nopayload` bytes per revision | 298 B (self-test) |
| emit call cost with tracing off | 74 ns (self-test run) |

The sampler-side transfer is persistent pinned staging plus
`non_blocking=True`; CUDA-event waiting and host `.tolist()` run on the staging
worker, and JSON/socket work runs on the sender worker. During live ingestion
the host sink replaces the producer event timestamp with receive time before
constructing `CanvasRevision`; raw emitter captures are not backlog-safe unless
they persist a receive timestamp.

### 9.1 Exclusive matched overhead ladder

The original blocking `.cpu()` path was measured at **+16.5%** end to end.
The ladder below measures the optimized path. It waits for `request_end` on
every trace stream, so it never scores a prefix of the revision stream, and it
records warm-up accounting. Deterministic randomised order (`full`, `nopayload`,
`parse`, `off`, seed `20260904`), five repeated prompt clusters, `max_tokens=64`,
`ignore_eos=true`, n=120:

| rung | p50 ms | p95 ms | revisions/request | streams closed | warm-up POSTs | vs off |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 1559.0 | 3123.4 | 0.0 | 0 / 0 | 3 | — |
| nopayload | 1617.5 | 3348.9 | 8.6 | 120 / 120 | 3 | +3.8% |
| full | 1414.4 | 2638.6 | 8.3 | 120 / 120 | 3 | −9.3% |
| parse | 1619.2 | 3120.9 | 8.4 | 120 / 120 | 3 | +3.9% |

All 480 requests were accelerator-exclusive, every rung returned 120/120
successful records with no problems, the `parse` rung parsed **1005 of 1005**
revisions, and all four rungs recorded exactly **3 warm-up POSTs with 0
empty-body retries** — so no rung consumed extra sampler draws before timing
started. The parse rung's direct accounting is **0.298 ms decode + 0.184 ms
parse/gate per revision**.

The p50 column still carries restart/order drift and no latency claim is drawn
from it: `full` came out 9.3% *faster* than `off` while `nopayload` and `parse`
came out ~3.8% slower, which is not a coherent cost ordering (`nopayload`
transfers strictly less than `full`). The clustered paired bootstrap over the
five prompt blocks gives `[-1.6%, +4.5%]`, straddling zero. The optimized
trace-transfer result is therefore **not resolvable by end-to-end wall time,
with no evidence of the former >10% blocking cost**; the direct per-revision
accounting is the useful overhead estimate. The complete ladder and raw
per-request arrays are in
`experiments/derived/overhead/overhead-{off,nopayload,full,parse}.json`, each
carrying `streams_closed`, `stream_accounting_provenance`, `warmup.{posts,
empty_body_retries,ids}`, `n_clusters: 5`, `prompt_indices`,
`request_occurrence_ids`, exact token counts and ladder order.

A rung is comparable only when every traced stream closed (`streams_closed ==
n`) and all rungs recorded the same warm-up count. Every warm-up POST consumes
sampler draws, so an unrecorded count is not evidence of equal workload; the `off` rung emits nothing and can only ever be witnessed by this
host-side counter, never by a sink.

The native fidelity check passes:
`scripts/verify_instrumentation.py --arm traced` → zero drops and identity on
the control-reproducible requests `[0, 1, 2]`; the long request remained the
known uninstrumented restart divergence.

## 10. Observations about the pinned build

Recorded because they affect how the revision stream should be read, not because
the instrumentation caused them:

* **`stability_threshold` is passed through, not incremented.** This build calls
  `DiffusionGemmaRequestStates(stability_threshold=self.gen_config["stability_threshold"])`
  = 1 (upstream `main` passes `+1`). With `ST = 1` the stability loop
  (`for h in range(1, ST)`) is empty, so `stable` is trivially true and
  `new_hist_len >= ST` holds after the very first step. Consequently a canvas
  whose `mean_entropy` is already below `confidence_threshold = 0.005` converges
  in a **single** denoise step. Observed: `denoise_steps_per_canvas` of
  `[1, 1]`, `[1, 1]`, `[3, 1]` for the short prompts and `[12, 27, 10, 1]` for
  the long one. The 48-step cap was never reached.
* Requests always end mid-canvas: a commit writes all 256 canvas tokens, the
  detokenizer stops at EOS, and the next canvas has already begun denoising, so
  the last `canvas_id` of a request typically has no `commit` revision. The
  verifier expects this.
* `request_end.output_tokens` is the number of *committed* tokens (a multiple of
  256), not the number of tokens the client received.

## 11. Energy telemetry scope

`nvidia-smi --query-gpu=power.draw,power.draw.instant,temperature.gpu,utilization.gpu,clocks.sm`
returns real values on this box (e.g. `11.83 W, 11.83 W, 51, 2 %, 2405 MHz`
sampled during a traced run; `5.08 W ... 208 MHz` at idle). However
`power.max_limit` and `memory.total` report `[N/A]` on GB10, and `dcgmi` is not
installed (no sudo on this box). Any energy claim must therefore be scoped as
**GPU-module power integrated from `power.draw` samples**, with no board or
system-level power, no memory-capacity normalisation, and no DCGM cross-check.

## 12. Status of each artifact

| artifact | state |
| --- | --- |
| `patches/crystalcall_trace.py` | done; `--self-test` passes (4 cases); staging and sender workers are bounded and non-blocking |
| `patches/vllm-trace.patch` | done; `patch -p1 --dry-run` clean on the host extract **and** inside the image; event completion is drained off the sampler thread |
| `patches/vllm-preact.patch` | done; 9 hunks, applied on top of the trace patch into the same file; dry-run clean on host and in-image. Enables the P-ReAct-style arm's token-selection priority, off unless `CRYSTALCALL_PREACT=1` |
| `patches/preact_priority.py` | done; mounted at `/opt/crystalcall/crystalcall_preact.py`; in-image CPU differential self-test passes 600/600 against the pure reference |
| `patches/apply_patch.sh` | done; run end-to-end, produced `patches/build/{diffusion_gemma.py,diffusion_gemma.py.orig,image-env.sh}` |
| `scripts/serve_diffusiongemma.sh` | done; used for all three arms; every recipe flag accepted by the pinned build, none rejected |
| `scripts/serve_gemma4.sh` | written and syntax-checked; **not launched** — it needs the GPU and the diffusion arm had priority. Its flag set is a strict subset of the verified diffusion invocation (same image, same `--tool-call-parser gemma4` / `--reasoning-parser gemma4` / `--generation-config vllm`, minus `--hf-overrides` and `--diffusion-config`), so the only untested part is that `google/gemma-4-26B-A4B-it` loads |
| `configs/diffusiongemma.yaml` | done; flat keys, parses, includes the pinned `seed: 12345` and `attention_backend: TRITON_ATTN` |
| `scripts/verify_instrumentation.py` | done; native traced arm exits 0: `PASS: completions byte-identical with tracing off vs on for requests [0, 1, 2]; 64 revisions validated; 0 drops.` Readiness requires `/health`; sink receive timestamps are retained separately from producer diagnostics; every arm stamps `source_provenance` (git commit + sampler/patch/emitter SHA-256) into its evidence file and rejects consumed evidence whose digests differ; control/baseline arms tear down the container they started (pinned by `tests/test_verify_instrumentation.py`) |
| `docs/evidence/*.json` | captured from the three-arm native run; traced evidence records 64 revisions, zero drops, sink skew, and matching `source_provenance` across all three arms |

### 12.1 Readiness must be observationally pure

Nothing may touch the model between server start and the measured requests. A
`max_tokens=1` warm-up completion was tried as a readiness probe and rejected:
the very next run failed byte-identity on request 2 (15 vs 17 characters). The
entropy-bound sampler renoises unaccepted positions with Gumbel noise drawn
from the engine's own generator and ignores the request temperature, so
any request issued before the measured ones shifts every subsequent
completion. `/health` answers only once the engine core is live and consumes no
draws, so that is the gate.

Flags rejected by the pinned build: **none**. All of `--diffusion-config`,
`--hf-overrides`, `--generation-config vllm`, `--enable-auto-tool-choice`,
`--tool-call-parser gemma4`, `--reasoning-parser gemma4`, `--max-num-seqs 1`,
`--gpu-memory-utilization`, `--max-model-len`, `--served-model-name` and
`--seed` were accepted and are visible in the engine's `non-default args` line.
Note that `--hf-overrides {"diffusion_sampler":...,"diffusion_entropy_bound":...}`
is *inert* on this build: the sampler reads `sampler_config.entropy_bound` from
`generation_config.json`, not from the HF config. It is kept because the recipe
pins it and it does no harm; the value that actually took effect is echoed in
every `request_begin.sampler`.
