#!/usr/bin/env bash
# CrystalCall / DenoisePTC -- the one command that reproduces every experiment.
#
#   scripts/run_all.sh                 # every stage, in order, from a clean checkout
#   scripts/run_all.sh --list          # numbered stage list, nothing else
#   scripts/run_all.sh --dry-run       # print every command that would run
#   scripts/run_all.sh --stage 15      # run exactly one stage (number or name)
#   scripts/run_all.sh --from pilot    # resume the pipeline at a stage
#   scripts/run_all.sh --resume        # skip stages already stamped as done
#
# Design rules:
#   * fail fast -- `set -Eeuo pipefail`, no stage swallows a non-zero exit;
#   * every command is echoed with a leading `+ ` before it runs;
#   * every stage is independently runnable and stamps
#     experiments/derived/.run_all/<stage>.done on success, so `--resume`
#     continues an interrupted run without redoing model time;
#   * this script owns the model servers: it starts them, waits for readiness
#     and stops them again, one instance per (arm, seed) for the held-out matrix
#     because the sampler seed is a *server* flag.
#
# Environment knobs: SEEDS (default "0 1 2"), TS_SEEDS, OVERHEAD_N,
# SINK_PORT, DIFFUSION_MODEL, GEMMA4_MODEL, PORT.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-0 1 2}"
TS_SEEDS="${TS_SEEDS:-0 1 2}"
# 64 scenarios x 3 seeds = 192 episodes, generated ONCE under speculative
# scheduling.  The ordinary-scheduling counterfactual is not a second
# generation: `run_toolsandbox.py` replays each episode's recorded
# authoritative call sequence against a clone of its INITIAL world with
# futures disabled, which holds the model fixed and makes the comparison a
# real test of scheduling.
TS_ARMS="${TS_ARMS:-toolsandbox_crystalcall}"
# CrystalCall is DERIVED by replaying the standard trajectories with the frozen
# gate, so it is deliberately NOT in the generation loop: a second, separately
# sampled 900-turn arm would unpair the comparison and overshoot the
# ~3,300-3,600 turn budget.  Its online behaviour is validated
# on a bounded subset by the `online-validation` stage.
MATRIX_ARMS="${MATRIX_ARMS:-diffusion_standard preact_style}"
# Bounded online validation: prompts x 1 seed, semantic invariant + real tools.
ONLINE_VALIDATION_N="${ONLINE_VALIDATION_N:-60}"
ONLINE_VALIDATION_SEED="${ONLINE_VALIDATION_SEED:-0}"
OVERHEAD_N="${OVERHEAD_N:-30}"
SINK_PORT="${SINK_PORT:-9801}"
SINK="127.0.0.1:${SINK_PORT}"
PORT="${PORT:-8000}"
DIFFUSION_MODEL="${DIFFUSION_MODEL:-diffusiongemma-26b-a4b-it}"
GEMMA4_MODEL="${GEMMA4_MODEL:-gemma-4-26b-a4b-it}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT}/v1}"
TEST_MANIFEST="experiments/manifests/test.jsonl"
DEV_MANIFEST="experiments/manifests/dev.jsonl"
STAMP_DIR="$ROOT/experiments/derived/.run_all"

DRY_RUN=0
RESUME=0
ONLY_STAGES=()
FROM_STAGE=""

# --------------------------------------------------------------------------- #
# stage table (order is the pipeline order)
# --------------------------------------------------------------------------- #

STAGE_NAMES=(
  doctor
  selftest
  manifests
  snapshots
  tools
  patch
  smoke
  instrumentation
  overhead
  dev-trace
  pilot
  preact-bound
  preact
  freeze
  matrix
  online-validation
  toolsandbox
  dev-ablations
  replay
  verify-traces
  analyze
  figures
  ceiling
)

STAGE_DESCS=(
  "environment, pins, hashes, telemetry"
  "safety core test suite"
  "rebuild + verify the frozen prompt manifests"
  "rebuild + verify the frozen corpus snapshots"
  "validate the real tool registry against the snapshots"
  "check the vLLM instrumentation patch still applies (dry run)"
  "GPU-free end-to-end OMP-Lite PTC smoke test"
  "instrumentation fidelity: traced vs baseline decoding"
  "Phase 0 instrumentation overhead ladder (4 rungs)"
  "dev-manifest trace, diffusion_standard, seed 0"
  "Phase 1 crystallization pilot and go/no-go gate"
  "P-ReAct-style priority bound: development selection on the dev manifest"
  "P-ReAct-style priority decoding: semantic health, on vs off"
  "freeze manifests/configs/thresholds and verify no drift"
  "held-out matrix: 3 generated arms x seeds ${SEEDS}, one server per (arm, seed)"
  "bounded online CrystalCall validation: semantic invariant + real tools"
  "ToolSandbox: 2 arms x seeds ${TS_SEEDS}"
  "sampler-level development ablations: adaptive stopping on/off"
  "offline policy and latency replay"
  "verify append-only trace hash chains"
  "aggregate derived data into reports/results.{json,md}"
  "render every figure"
  "oracle latency ceiling and the milestone/ceiling figures"
)

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

log()  { printf '%s\n' "$*"; }
warn() { printf '%s\n' "$*" >&2; }
die()  { printf 'run_all: %s\n' "$*" >&2; exit 2; }

# Echo a command, then run it (unless --dry-run).
run() {
  printf '+ %s\n' "$*"
  (( DRY_RUN )) && return 0
  "$@"
}

usage() {
  sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
}

list_stages() {
  local i
  for i in "${!STAGE_NAMES[@]}"; do
    printf '%2d  %-16s %s\n' "$((i + 1))" "${STAGE_NAMES[$i]}" "${STAGE_DESCS[$i]}"
  done
}

# stage number (1-based) for a number or a name; 0 when unknown.
stage_index() {
  local want="$1" i
  if [[ "$want" =~ ^[0-9]+$ ]]; then
    (( want >= 1 && want <= ${#STAGE_NAMES[@]} )) && { printf '%s' "$want"; return 0; }
    printf '0'; return 0
  fi
  for i in "${!STAGE_NAMES[@]}"; do
    if [[ "${STAGE_NAMES[$i]}" == "$want" ]]; then printf '%s' "$((i + 1))"; return 0; fi
  done
  printf '0'
}

selected() {
  local idx="$1" want
  if (( ${#ONLY_STAGES[@]} )); then
    for want in "${ONLY_STAGES[@]}"; do [[ "$want" == "$idx" ]] && return 0; done
    return 1
  fi
  if [[ -n "$FROM_STAGE" ]]; then (( idx >= FROM_STAGE )) || return 1; fi
  return 0
}

# --------------------------------------------------------------------------- #
# model servers (this script owns their lifecycle)
# --------------------------------------------------------------------------- #

DIFFUSION_UP=0
GEMMA4_UP=0

start_diffusion() {  # start_diffusion <seed> <trace 0|1> [extra server args]
  local seed="$1" trace="${2:-1}" extra="${3:-}"
  stop_servers
  # CRYSTALCALL_PREACT is pinned OFF, never inherited: an ambient 1 in the
  # caller's environment would silently give the standard and CrystalCall arms
  # priority decoding and destroy the fixed-arm comparison.
  run env SEED="$seed" PORT="$PORT" \
    CRYSTALCALL_TRACE="$trace" \
    CRYSTALCALL_TRACE_SINK="127.0.0.1:${SINK_PORT}" \
    CRYSTALCALL_TRACE_LEVEL=full \
    CRYSTALCALL_PREACT=0 \
    EXTRA_ARGS="$extra" \
    bash scripts/serve_diffusiongemma.sh
  DIFFUSION_UP=1
}

start_preact() {  # start_preact <seed>
  # The P-ReAct-style baseline is a *sampler* change, so it
  # needs its own server instance: same image, same patched file, same prompt,
  # priority decoding switched on by environment.
  local seed="$1"
  stop_servers
  run env SEED="$seed" PORT="$PORT" \
    CRYSTALCALL_TRACE=1 \
    CRYSTALCALL_TRACE_SINK="127.0.0.1:${SINK_PORT}" \
    CRYSTALCALL_TRACE_LEVEL=full \
    CRYSTALCALL_PREACT=1 \
    CRYSTALCALL_PREACT_MODE="${PREACT_MODE:-both}" \
    bash scripts/serve_diffusiongemma.sh
  DIFFUSION_UP=1
}

start_arm_server() {  # start_arm_server <arm> <seed>
  case "$1" in
    preact_style) start_preact "$2" ;;
    *) start_diffusion "$2" 1 ;;
  esac
}

start_gemma4() {  # start_gemma4 <seed>
  local seed="$1"
  stop_servers
  run env SEED="$seed" PORT="$PORT" bash scripts/serve_gemma4.sh
  GEMMA4_UP=1
}

stop_servers() {
  if (( DIFFUSION_UP )); then
    run bash scripts/serve_diffusiongemma.sh --stop
    DIFFUSION_UP=0
  fi
  if (( GEMMA4_UP )); then
    run bash scripts/serve_gemma4.sh --stop
    GEMMA4_UP=0
  fi
}

cleanup() {
  local rc=$?
  stop_servers || true
  exit "$rc"
}

# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

stage_doctor() {
  run uv run --project . crystalcall doctor --out reports/doctor.json
}

stage_selftest() {
  run uv run --project . crystalcall selftest
}

stage_manifests() {
  run uv run scripts/build_manifests.py --verify
}

stage_snapshots() {
  run uv run scripts/build_snapshots.py --verify
}

stage_tools() {
  run uv run --project . crystalcall tools validate
}

stage_patch() {
  run bash patches/apply_patch.sh --check
}

stage_smoke() {
  run uv run scripts/smoke_omp_ptc.py
  # The production arm path (`run_trace_arm`) against a stub that impersonates
  # the instrumented server on BOTH channels: authoritative SSE + revisions.
  run uv run scripts/smoke_trace_arm.py --prompts 4 --port "$PORT" --sink "$SINK"
}

stage_instrumentation() {
  # verify_instrumentation.py starts and stops its own containers.
  stop_servers
  run uv run scripts/verify_instrumentation.py --arm traced
}

stage_preact_bound() {
  # The baseline's own priority setting is tuned on DEVELOPMENT data, in the
  # baseline's favour, on absolute ground-truth rates rather than per-prompt
  # agreement, because this engine is not restart-reproducible. A crippled
  # baseline would hand CrystalCall an artefactual S8 advantage.
  stop_servers
  run uv run scripts/preact_bound_sweep.py \
    --manifest experiments/manifests/dev.jsonl \
    --reference-shard experiments/raw/diffusion_standard/diffusion_standard-seed0.jsonl \
    --bounds 0.2,0.6 \
    --write-selection
}

stage_preact() {
  # The P-ReAct-style baseline is a sampler change, so its correctness is a
  # SEMANTIC question, not a timing one: the arm must still emit a complete,
  # parseable tool call with the SAME canonical arguments as the standard arm
  # and a non-empty closing turn, and the standard arm must emit no readiness
  # events at all.  smoke_preact.py brings the server up twice itself.
  stop_servers
  run uv run scripts/smoke_preact.py --json experiments/evidence/preact_smoke.json
}

stage_overhead() {
  # overhead_ladder.sh restarts the server once per rung itself.
  #
  # The ladder is a latency measurement, so it needs the accelerator to itself:
  # a foreign process does not add noise, it invents instrumentation cost.
  # Wait for exclusivity rather than recording a number that would
  # misattribute someone else's workload.
  stop_servers
  # 98 GB, not 60: vLLM refuses to start unless free memory covers
  # gpu_memory_utilization x total (0.80 x 121.69 = 97.35 GiB). `run_overhead`
  # re-checks this before every rung, because holders come and go on this box.
  run bash scripts/wait_for_gpu.sh --need-gb 98 --exclusive --interval 60
  run env CRYSTALCALL_TRACE_SINK="$SINK" bash scripts/overhead_ladder.sh "$OVERHEAD_N"
}

stage_dev_trace() {
  start_diffusion 0 1
  run uv run --project . crystalcall trace \
    --manifest "$DEV_MANIFEST" --arm diffusion_standard --seed 0
  stop_servers
}

stage_pilot() {
  # Which attempt this is follows from the *prompt*, not from which report
  # files happen to exist: the reports are committed evidence, so deriving it
  # from their presence refuses a fresh clone that is merely reproducing
  # attempt 2. The resolver exits 3 on a genuine third
  # distinct configuration.
  local attempt
  if ! attempt="$(uv run scripts/pilot_attempt.py)"; then
    (( DRY_RUN )) && { printf '+ %s\n' "uv run --project . crystalcall pilot --attempt <refused>"; return 0; }
    return 1
  fi
  printf '+ %s\n' "uv run --project . crystalcall pilot --attempt $attempt"
  (( DRY_RUN )) && return 0
  if uv run --project . crystalcall pilot --attempt "$attempt"; then
    return 0
  fi
  # The pilot records each go/no-go gate in reports/pilot_attempt<N>.json.
  # In the paper both attempts missed the precision and speed-up targets; the
  # remaining stages were still run, and the paper reports both attempts.
  warn "run_all: pilot attempt $attempt did not pass every gate (see reports/pilot_attempt${attempt}.json); continuing"
  return 0
}

stage_freeze() {
  run uv run --project . crystalcall freeze-manifest
  run uv run --project . crystalcall verify-freeze
}

stage_matrix() {
  # One server instance per seed: the sampler seed is a server flag, not a
  # request parameter, so a seed change means a restart.
  local seed arm
  for seed in $SEEDS; do
    # One server per (arm, seed): the seed is a server flag and the P-ReAct-style
    # arm additionally needs priority decoding enabled server-side.
    for arm in $MATRIX_ARMS; do
      start_arm_server "$arm" "$seed"
      run uv run --project . crystalcall run \
        --manifest "$TEST_MANIFEST" --arm "$arm" --seed "$seed" --resume
    done
    stop_servers
    start_gemma4 "$seed"
    run uv run --project . crystalcall run \
      --manifest "$TEST_MANIFEST" --arm gemma4_ar --seed "$seed" --resume
    stop_servers
  done
}

stage_online_validation() {
  # The frozen gate, live through OMP-Lite, on a bounded held-out subset.  This
  # is where the semantic invariant is measured against real work: every claimed
  # result is re-executed normally after the turn and the digests compared.  It
  # is NOT the source of the headline latency estimate -- that comes from
  # replaying the full standard trajectories with the same gate.
  start_diffusion "$ONLINE_VALIDATION_SEED" 1
  run uv run --project . crystalcall run \
    --manifest "$TEST_MANIFEST" --arm crystalcall \
    --seed "$ONLINE_VALIDATION_SEED" --limit "$ONLINE_VALIDATION_N" --resume
  # Same subset again with real immutable local tool durations injected, so the
  # deterministic-duration replay is validated against real tool behaviour.
  run env CRYSTALCALL_TOOL_SLEEP_MS=500 uv run --project . crystalcall run \
    --manifest "$TEST_MANIFEST" --arm crystalcall \
    --seed "$ONLINE_VALIDATION_SEED" --limit "$ONLINE_VALIDATION_N" \
    --tool-sleep-ms 500 --out experiments/raw/crystalcall_tool500 --resume
  stop_servers
}

stage_toolsandbox() {
  # Same rule as the held-out matrix: the replicate seed is a server flag,
  # so each seed gets its own server instance and both arms share it.
  local arm seed
  for seed in $TS_SEEDS; do
    start_diffusion "$seed" 1
    for arm in $TS_ARMS; do
      run uv run scripts/run_toolsandbox.py \
        --arm "$arm" --seed "$seed" --server-url "$SERVER_URL" --resume
    done
    stop_servers
  done
}

stage_dev_ablations() {
  # The only ablations that cannot be replayed from a stored trace, because they
  # change the sampler and therefore the trajectory. Development subset only:
  # they never touch the held-out manifest and never move the frozen thresholds.
  local spec="configs/experiments/dev_ablations.yaml"
  local n=25
  local off_args
  off_args='--hf-overrides {"diffusion_sampler":"entropy_bound","diffusion_entropy_bound":0.1,"confidence_threshold":-1.0,"stability_threshold":9999}'

  start_diffusion 0 1
  run uv run --project . crystalcall trace \
    --manifest experiments/manifests/dev.jsonl --arm diffusion_standard --seed 0 \
    --limit "$n" --out experiments/raw/dev_ablation_adaptive_on --resume
  stop_servers

  start_diffusion 0 1 "$off_args"
  run uv run --project . crystalcall trace \
    --manifest experiments/manifests/dev.jsonl --arm diffusion_standard --seed 0 \
    --limit "$n" --out experiments/raw/dev_ablation_adaptive_off --resume
  stop_servers

  echo "+ dev ablation spec: $spec"
}

stage_replay() {
  run uv run --project . crystalcall replay --policy all --latency-ms 100 500 2000 5000
}

stage_verify_traces() {
  run uv run --project . crystalcall verify-traces
}

stage_analyze() {
  run uv run --project . crystalcall analyze
}

stage_figures() {
  run uv run --project . crystalcall figures
}

stage_ceiling() {
  # The latency ceiling of any launch rule, and the two figures behind the
  # paper's Figure 1 (milestone CDFs and the launch-rule ceiling).
  run uv run scripts/oracle_ceiling.py
  run uv run scripts/negative_figures.py
}

# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #

while (( $# )); do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --resume)  RESUME=1 ;;
    --list)    list_stages; exit 0 ;;
    --stage)
      [[ $# -ge 2 ]] || die "--stage needs a stage number or name"
      idx="$(stage_index "$2")"
      [[ "$idx" != "0" ]] || die "unknown stage '$2' (see --list)"
      ONLY_STAGES+=("$idx"); shift ;;
    --stage=*)
      idx="$(stage_index "${1#*=}")"
      [[ "$idx" != "0" ]] || die "unknown stage '${1#*=}' (see --list)"
      ONLY_STAGES+=("$idx") ;;
    --from)
      [[ $# -ge 2 ]] || die "--from needs a stage number or name"
      FROM_STAGE="$(stage_index "$2")"
      [[ "$FROM_STAGE" != "0" ]] || die "unknown stage '$2' (see --list)"
      shift ;;
    --from=*)
      FROM_STAGE="$(stage_index "${1#*=}")"
      [[ "$FROM_STAGE" != "0" ]] || die "unknown stage '${1#*=}' (see --list)" ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument '$1' (see --help)" ;;
  esac
  shift
done

# Stage stamps are run bookkeeping, not derived data: the directory ignores
# itself so `--resume` state never lands in a commit.
if (( ! DRY_RUN )); then
  mkdir -p "$STAMP_DIR"
  [[ -f "$STAMP_DIR/.gitignore" ]] || printf '*\n' >"$STAMP_DIR/.gitignore"
fi
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

log "CrystalCall measurement pipeline"
log "root        : $ROOT"
log "seeds       : $SEEDS"
log "dry-run     : $DRY_RUN   resume: $RESUME"
log ""

STARTED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
for i in "${!STAGE_NAMES[@]}"; do
  idx=$((i + 1))
  name="${STAGE_NAMES[$i]}"
  desc="${STAGE_DESCS[$i]}"
  selected "$idx" || continue
  stamp="$STAMP_DIR/${name}.done"
  if (( RESUME )) && [[ -f "$stamp" ]]; then
    log "== [$idx/${#STAGE_NAMES[@]}] $name -- SKIPPED (already done: $stamp)"
    continue
  fi
  log "== [$idx/${#STAGE_NAMES[@]}] $name -- $desc"
  "stage_${name//-/_}"
  if (( ! DRY_RUN )); then
    printf '%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >"$stamp"
  fi
  log ""
done

log "run_all: finished (started $STARTED_AT, ended $(date -u +%Y-%m-%dT%H:%M:%SZ))"
