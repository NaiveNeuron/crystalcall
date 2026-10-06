# CrystalCall

Code and data for the paper **Where Should a Diffusion-LM Agent Start Its Tools?
The Acceptance Lead in Speculative Tool Calling**, by Marek Šuppa and Jaroslav
Kopcan, NeurIPS 2026 Workshop on SLMs for Agentic Systems (SLM-Agents), Paris.

Paper: https://openreview.net/forum?id=nVaHns2ZDR

A diffusion language model holds a draft of its whole answer at every decoding
step, so a tool call can be readable before generation ends. CrystalCall reads
these drafts, launches side-effect-free tool calls early, and hands a result to
the agent only if the final program makes exactly that call. The final program
is always authoritative: CrystalCall changes when a call runs, not which calls
run.

## Repository layout

```
src/crystalcall/   runtime, trace recording, offline replay, metrics, CLI
patches/           vLLM sampler instrumentation (draft side channel)
configs/           model, tool, harness profile, policy and experiment configs
scripts/           servers, manifest builders, analyses, paper numbers
experiments/
  manifests/       BFCL development (100) and test (300) prompts, ToolSandbox (64)
  raw/             per-run metadata and checksums of the recorded traces
  derived/         per-trajectory analyses and replayed policy outcomes
  derived_heldout/ the same for the 300 test prompts
  sweeps/          block-diffusion drafter runs, ToolSandbox runs
  figures/         figures rendered from the derived data
reports/           aggregated results, pilot results, environment record
tests/             unit and regression tests
docs/              design notes: architecture, instrumentation, metrics, tools
```

## What is included

- The full code used for the paper, with the configuration it ran with.
- The prompt manifests and the pinned sources they were built from
  (`experiments/manifests/vendor_sources.json`).
- For every recorded run: its metadata, the checksum of each trace shard and the
  final hash of its hash chain (`experiments/raw/**`).
- Everything computed from the traces: per-trajectory analyses, replayed policy
  outcomes for every launch rule and tool delay, the learned launch rule, the
  oracle ceiling, throughput, and the aggregated results in
  `reports/results.json` and `reports/results_heldout.json`.
- The environment record of the measurement machine (`reports/doctor.json`).

Not included: model weights, the third-party sources fetched by
`scripts/build_manifests.py --fetch`, the harness binary fetched by
`scripts/fetch_omp.sh`, and the raw trace shards (`experiments/raw/**/*.jsonl`).

## Quick start (CPU only)

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run scripts/build_manifests.py --fetch   # BFCL, ToolSandbox and sPTC at pinned commits
uv run --with pytest pytest                 # tests
uv run scripts/paper_numbers.py             # writes paper/numbers.tex and the appendix tables
```

`paper_numbers.py` reads only committed artifacts. The appendix tables and most
numbers are recomputed from the derived data; the few numbers that are counted
directly from raw trace shards (task success, parse rates and the live run) need
those shards in `experiments/raw/`.

## Full pipeline (GPU)

The measurements ran on one NVIDIA DGX Spark (GB10, arm64, 128 GB unified
memory) with:

| Component | Version |
| --- | --- |
| Diffusion model | `google/diffusiongemma-26B-A4B-it` @ `f7f5b7f5` |
| AR control | `google/gemma-4-26B-A4B-it` |
| Serving | `vllm/vllm-openai@sha256:9c719fc0…` (vLLM `0.22.1rc1.dev357+g74b5964f0`) |
| Harness | Oh My Pi `omp/18.1.4`, restricted to the profile in `configs/omp-lite.yml` |

Exact pins and file hashes are in `reports/doctor.json`;
`patches/build/image-env.sh` holds the image digest.

`scripts/run_all.sh` runs the measurement stages in order (`--list` shows them,
`--dry-run` prints every command). The paper's results come from these steps:

```bash
# Setup
uv run scripts/build_manifests.py --fetch
bash scripts/fetch_omp.sh                       # pinned harness binary
uv run scripts/build_snapshots.py               # tool corpora
uv run crystalcall doctor --out reports/doctor.json

# The sampler seed is a server flag, so each seed needs its own server.
# serve_diffusiongemma.sh applies the instrumentation patch inside the image.

# Development traces: diffusion model, seeds 0-2
for seed in 0 1 2; do
  SEED=$seed bash scripts/serve_diffusiongemma.sh
  uv run crystalcall trace --manifest experiments/manifests/dev.jsonl \
      --arm diffusion_standard --seed $seed
  bash scripts/serve_diffusiongemma.sh --stop
done
uv run crystalcall pilot --attempt 2

# Development traces: AR control
SEED=0 bash scripts/serve_gemma4.sh
uv run crystalcall trace --manifest experiments/manifests/dev.jsonl --arm gemma4_ar --seed 0

# Test prompts, seed 0
uv run crystalcall run --manifest experiments/manifests/test.jsonl --arm gemma4_ar \
    --seed 0 --out experiments/raw/heldout/gemma4_ar
bash scripts/serve_gemma4.sh --stop
SEED=0 bash scripts/serve_diffusiongemma.sh
uv run crystalcall run --manifest experiments/manifests/test.jsonl --arm diffusion_standard \
    --seed 0 --out experiments/raw/heldout/diffusion_standard

# Live run of the frozen rule on 60 test prompts, at two tool delays
for ms in 500 2000; do
  uv run crystalcall run --manifest experiments/manifests/test.jsonl --arm crystalcall \
      --seed 0 --limit 60 --tool-sleep-ms $ms --out experiments/raw/crystalcall_tool$ms
done

# Replay every launch rule at tool delays of 100, 500, 2000 and 5000 ms
uv run crystalcall replay --policy all --latency-ms 100 500 2000 5000
uv run crystalcall replay --shard experiments/raw/heldout/*/*.jsonl \
    --out experiments/derived_heldout
uv run crystalcall verify-traces
uv run crystalcall analyze                      # reports/results.{json,md}
uv run python -c "from crystalcall.aggregate import aggregate; \
aggregate('experiments/derived_heldout', 'reports/results_heldout.json', 'reports/results_heldout.md')"

# Analyses
uv run scripts/learned_gate.py                  # learned launch rule (CPU)
uv run scripts/oracle_ceiling.py                # latency ceiling of any launch rule
uv run scripts/throughput.py
uv run scripts/text_vs_exec_lead.py
uv run scripts/negative_figures.py              # milestone CDFs and ceiling figure
bash scripts/overhead_ladder.sh 30              # recording overhead, 4 levels

# Stateful condition: ToolSandbox with a scripted model, against a cloned world
for seed in 0 1 2; do for arm in toolsandbox_baseline toolsandbox_crystalcall; do
  uv run scripts/run_toolsandbox.py --stub --arm $arm --seed $seed \
      --out experiments/sweeps/toolsandbox_stub
done; done

# Paper numbers and appendix tables
uv run scripts/paper_numbers.py
```

Traces are append-only and hash-chained; every derived output records the hash
of the trace it was computed from. The harness is a third-party program and is
used unmodified (`patches/omp-provider.md`); any runtime with the same
properties can replace it.

### Block-diffusion drafter

The cross-family check in the paper's appendix used `Qwen/Qwen3.8-27B` as target
and the `z-lab/Qwen3.8-27B-DFlash2` drafter, served by SGLang with
`--speculative-num-draft-tokens` set to 8 or 4, on the first 12 development
prompts:

```bash
uv run scripts/dflash_lead.py --base-url http://127.0.0.1:8000 --label dflash2_block8 \
    --out experiments/sweeps/dflash2/lead_block8.json
```

SGLang reports the per-request acceptance telemetry this script reads, so no
patch is needed. Only post-verification tokens are visible, so the lead on this
decoder is an upper bound, not a measurement.

## Citation

```bibtex
@inproceedings{suppa2026crystalcall,
  author    = {Marek {\v{S}}uppa and Jaroslav Kopcan},
  title     = {Where Should a Diffusion-{LM} Agent Start Its Tools?
               The Acceptance Lead in Speculative Tool Calling},
  booktitle = {NeurIPS 2026 Workshop on SLMs for Agentic Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=nVaHns2ZDR}
}
```
