#!/usr/bin/env bash
# Launch the instrumented DiffusionGemma server for CrystalCall / DenoisePTC.
#
#   scripts/serve_diffusiongemma.sh [--foreground] [--no-wait] [--stop] [--logs]
#                                   [--print-docker-args]
#
# The patched sampler and the stdlib emitter are bind-mounted over the pinned
# image at run time (see patches/apply_patch.sh); the SAME container image and
# the SAME patched file serve every experiment arm, switched only by
# CRYSTALCALL_TRACE and CRYSTALCALL_PREACT, so the uninstrumented baseline is
# the identical code path with the taps and the priority mechanism compiled out
# (None buffers, no allocations, no copies).
#
# --print-docker-args renders the exact container argv and exits without
# touching docker: it is how the tests prove the container actually receives
# every CRYSTALCALL_* variable (the container does NOT inherit host env).
#
# Env overrides (all optional):
#   CRYSTALCALL_TRACE=0|1            enable the side channel
#   CRYSTALCALL_TRACE_SINK=host:port NDJSON TCP sink (host loopback; --network host)
#   CRYSTALCALL_TRACE_LEVEL=full|nopayload
#   CRYSTALCALL_TRACE_QUEUE=4096
#   CRYSTALCALL_TRACE_FILE=/path     append NDJSON inside the container
#   CRYSTALCALL_PREACT=0|1           P-ReAct-style tool priority (default 0)
#   CRYSTALCALL_PREACT_MODE          span_accept|ready_signal|both (default both)
#   CRYSTALCALL_PREACT_BOUND         in-span per-token entropy bound
#   CRYSTALCALL_PREACT_STABLE        consecutive identical span steps
#   CRYSTALCALL_PREACT_OPEN_ID       tool-call opener id  (default: configs/preact.yml)
#   CRYSTALCALL_PREACT_CLOSE_ID      tool-call closer id  (default: configs/preact.yml)
#   PORT, HOST, MODEL, SERVED_NAME, MAX_MODEL_LEN, MAX_NUM_SEQS,
#   GPU_MEMORY_UTILIZATION, CANVAS_LENGTH, GENERATION_CONFIG, ENTROPY_BOUND,
#   CPUSET,
#   CRYSTALCALL_VLLM_IMAGE, NAME, HF_CACHE, EXTRA_ARGS
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATCHES="$REPO/patches"
ENVFILE="$PATCHES/build/image-env.sh"

NAME="${NAME:-cc-diffusiongemma}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MODEL="${MODEL:-google/diffusiongemma-26B-A4B-it}"
SERVED_NAME="${SERVED_NAME:-diffusiongemma-26b-a4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# Exactly one in-flight request: CrystalCall speculates on a single canvas
# stream, and the diffusion state buffers scale as
# max_num_seqs x canvas_length x vocab_size, so 1 also minimises memory.
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
CANVAS_LENGTH="${CANVAS_LENGTH:-256}"
ENTROPY_BOUND="${ENTROPY_BOUND:-0.1}"
# The sampler reads `entropy_bound` and `stability_threshold` from the
# *generation* config (`patches/build/diffusion_gemma.py:1067,1094-1101`), NOT
# from the HF model config, so `--hf-overrides diffusion_entropy_bound` is
# silently inert: a rung passed that way reported `entropy_bound: 0.1` in the
# sampler's own emitted meta. Sweeping a sampler knob means
# substituting a generation config directory here.
GENERATION_CONFIG="${GENERATION_CONFIG:-vllm}"
# The entropy-bound sampler renoises the canvas with Gumbel noise, so decoding
# is stochastic regardless of the request's `temperature`. A pinned engine seed
# makes a fresh server replay the same RNG stream, which is what makes the
# tracing-off vs tracing-on comparison meaningful. Set SEED= to disable.
SEED="${SEED-12345}"
# Pin to the ten Cortex-X925 performance cores (the A725 cores are 0-4,10-14).
CPUSET="${CPUSET:-5-9,15-19}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

# ---- P-ReAct-style tool-priority decoding (baseline arm D) ---------------- #
# Off by default: with CRYSTALCALL_PREACT=0 the container runs the identical
# patched sampler with the mechanism compiled out, which is the CrystalCall /
# standard arm. The container does NOT inherit host env, so every variable
# below needs its own explicit -e line further down.
PREACT_CFG="$REPO/configs/preact.yml"
PREACT="${CRYSTALCALL_PREACT:-0}"
PREACT_MODE="${CRYSTALCALL_PREACT_MODE:-both}"
PREACT_BOUND="${CRYSTALCALL_PREACT_BOUND:-}"
PREACT_STABLE="${CRYSTALCALL_PREACT_STABLE:-}"
PREACT_OPEN_ID="${CRYSTALCALL_PREACT_OPEN_ID:-}"
PREACT_CLOSE_ID="${CRYSTALCALL_PREACT_CLOSE_ID:-}"

die() { echo "serve: $*" >&2; exit 1; }

case "${1:-}" in
  --stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "stopped $NAME" || echo "$NAME not running"
    exit 0
    ;;
  --logs)
    shift
    exec docker logs "${1:---tail=200}" "$NAME"
    ;;
esac

# Narrow scalar reader for configs/preact.yml: `  key: value` with an optional
# trailing comment. Deliberately not a YAML parser -- it reads exactly the four
# keys below and every one of them is validated after reading.
yml_scalar() {
  [[ -f "$PREACT_CFG" ]] || return 1
  sed -n -E "s/^[[:space:]]*$1:[[:space:]]*\"?([^\"#[:space:]]+)\"?.*/\1/p" "$PREACT_CFG" | sed -n 1p
}

FOREGROUND=0
WAIT=1
PRINT_ARGS=0
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    --no-wait) WAIT=0 ;;
    --print-docker-args) PRINT_ARGS=1 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

# Resolve the tool-call delimiter ids from configs/preact.yml. They are data,
# never a default: a wrong or empty id would make the span never match and the
# P-ReAct-style arm silently identical to the standard arm, i.e. a fake null.
[[ -n "$PREACT_OPEN_ID" ]] || PREACT_OPEN_ID="$(yml_scalar open_token_id || true)"
[[ -n "$PREACT_CLOSE_ID" ]] || PREACT_CLOSE_ID="$(yml_scalar close_token_id || true)"
[[ -n "$PREACT_BOUND" ]] || PREACT_BOUND="$(yml_scalar span_entropy_bound || true)"
[[ -n "$PREACT_STABLE" ]] || PREACT_STABLE="$(yml_scalar require_stable_steps || true)"

case "$PREACT" in
  0|false|no|off|"") PREACT=0 ;;
  *)
    [[ -f "$PREACT_CFG" ]] || die "CRYSTALCALL_PREACT=$PREACT but $PREACT_CFG is missing"
    case "$PREACT_MODE" in
      span_accept|ready_signal|both) ;;
      *) die "CRYSTALCALL_PREACT_MODE=$PREACT_MODE not in span_accept|ready_signal|both" ;;
    esac
    [[ "$PREACT_OPEN_ID" =~ ^[0-9]+$ ]] \
      || die "could not read tokens.open_token_id from $PREACT_CFG (got '$PREACT_OPEN_ID'); run: uv run scripts/resolve_gemma_tokens.py --write"
    [[ "$PREACT_CLOSE_ID" =~ ^[0-9]+$ ]] \
      || die "could not read tokens.close_token_id from $PREACT_CFG (got '$PREACT_CLOSE_ID'); run: uv run scripts/resolve_gemma_tokens.py --write"
    [[ "$PREACT_OPEN_ID" != "$PREACT_CLOSE_ID" ]] \
      || die "open_token_id == close_token_id == $PREACT_OPEN_ID in $PREACT_CFG"
    [[ "$PREACT_BOUND" =~ ^[0-9]+([.][0-9]+)?$ ]] \
      || die "bad span_entropy_bound '$PREACT_BOUND' in $PREACT_CFG"
    [[ "$PREACT_STABLE" =~ ^[1-9][0-9]*$ ]] \
      || die "bad require_stable_steps '$PREACT_STABLE' in $PREACT_CFG"
    ;;
esac

[[ -f "$ENVFILE" ]] || die "run patches/apply_patch.sh first (missing $ENVFILE)"
# shellcheck source=/dev/null
source "$ENVFILE"
IMAGE="${CRYSTALCALL_VLLM_IMAGE:?}"
SITE_PACKAGES="${CRYSTALCALL_SITE_PACKAGES:?}"
SAMPLER_REL="${CRYSTALCALL_SAMPLER_REL:?}"
PATCHED="$PATCHES/build/diffusion_gemma.py"
[[ -f "$PATCHED" ]] || die "missing $PATCHED (run patches/apply_patch.sh)"

(( PRINT_ARGS )) || docker rm -f "$NAME" >/dev/null 2>&1 || true

# Host-visible directory for emitter diagnostics (counters, optional NDJSON
# dump). Mounted at the same path inside the container so paths are identical.
TRACE_DIR="${CRYSTALCALL_TRACE_DIR:-/tmp/cc-trace}"
(( PRINT_ARGS )) || mkdir -p "$TRACE_DIR"

DOCKER_ARGS=(
  run --name "$NAME"
  --gpus all --ipc=host --network host
  --cpuset-cpus "$CPUSET"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -v "$PATCHED:$SITE_PACKAGES/$SAMPLER_REL:ro"
  -v "$PATCHES/crystalcall_trace.py:/opt/crystalcall/crystalcall_trace.py:ro"
  -v "$PATCHES/preact_priority.py:/opt/crystalcall/crystalcall_preact.py:ro"
  -e PYTHONPATH=/opt/crystalcall
  -e HF_HOME=/root/.cache/huggingface
  -e "CRYSTALCALL_TRACE=${CRYSTALCALL_TRACE:-0}"
  -e "CRYSTALCALL_TRACE_SINK=${CRYSTALCALL_TRACE_SINK:-}"
  -e "CRYSTALCALL_TRACE_LEVEL=${CRYSTALCALL_TRACE_LEVEL:-full}"
  -e "CRYSTALCALL_TRACE_QUEUE=${CRYSTALCALL_TRACE_QUEUE:-4096}"
  -e "CRYSTALCALL_TRACE_FILE=${CRYSTALCALL_TRACE_FILE:-}"
  -e "CRYSTALCALL_TRACE_STATS=${CRYSTALCALL_TRACE_STATS:-$TRACE_DIR/stats.json}"
  -v "$TRACE_DIR:$TRACE_DIR"
  -e "CRYSTALCALL_PREACT=$PREACT"
  -e "CRYSTALCALL_PREACT_MODE=$PREACT_MODE"
  -e "CRYSTALCALL_PREACT_BOUND=$PREACT_BOUND"
  -e "CRYSTALCALL_PREACT_STABLE=$PREACT_STABLE"
  -e "CRYSTALCALL_PREACT_OPEN_ID=$PREACT_OPEN_ID"
  -e "CRYSTALCALL_PREACT_CLOSE_ID=$PREACT_CLOSE_ID"
)
[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
[[ -n "${VLLM_EXTRA_ENV:-}" ]] && DOCKER_ARGS+=(-e "$VLLM_EXTRA_ENV")

# The pinned invocation carries exactly two override keys, because
# `reports/doctor.json` records this JSON verbatim and its digest is stamped
# into every run.json: adding a key unconditionally would break the digest the
# 1111-record corpus was traced under. `STABILITY_THRESHOLD` therefore only
# appears when the caller sets it -- and note that the sampler reads that value
# from the generation config, so the key is documentation of intent, not a
# working sweep route (use `GENERATION_CONFIG`).
HF_OVERRIDES="{\"diffusion_sampler\":\"entropy_bound\",\"diffusion_entropy_bound\":$ENTROPY_BOUND"
if [[ -n "${STABILITY_THRESHOLD:-}" ]]; then
  HF_OVERRIDES+=",\"diffusion_stability_threshold\":$STABILITY_THRESHOLD"
fi
HF_OVERRIDES+="}"

SERVER_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --generation-config "$GENERATION_CONFIG"
  --hf-overrides "$HF_OVERRIDES"
  --diffusion-config "{\"canvas_length\": $CANVAS_LENGTH}"
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
  --host "$HOST"
  --port "$PORT"
)
[[ -n "$SEED" ]] && SERVER_ARGS+=(--seed "$SEED")
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && SERVER_ARGS+=(${EXTRA_ARGS})

if (( PRINT_ARGS )); then
  # Render the exact container argv and exit. Host-shell env is NOT evidence
  # that the container got a variable: only the -e lines below are.
  printf '%s\n' docker "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVER_ARGS[@]}"
  exit 0
fi

if (( FOREGROUND )); then
  exec docker "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVER_ARGS[@]}"
fi

docker "${DOCKER_ARGS[@]}" -d "$IMAGE" "${SERVER_ARGS[@]}" >/dev/null
echo "started $NAME (trace=${CRYSTALCALL_TRACE:-0} sink=${CRYSTALCALL_TRACE_SINK:-none} level=${CRYSTALCALL_TRACE_LEVEL:-full} preact=$PREACT mode=$PREACT_MODE)"

(( WAIT )) || exit 0

deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/v1/models" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    # `/v1/models` answers as soon as the HTTP layer is up, while the engine
    # core may still be warming; a completion sent in that window comes back as
    # an empty HTTP 200. `/health` is 200 only once the
    # engine core is live, and unlike a warm-up completion it consumes no draws
    # from the sampler's generator -- the entropy-bound sampler renoises with
    # Gumbel noise from that generator, so any request issued before the
    # measured ones shifts every subsequent completion and destroys the
    # byte-identity the instrumentation check exists to prove.
    hcode="$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/health" 2>/dev/null || true)"
    if [[ "$hcode" == "200" ]]; then
      echo "ready: http://$HOST:$PORT/v1/models (engine core healthy)"
      exit 0
    fi
    # A 404 here is the early window, not a missing endpoint: this build serves
    # `/health` once the engine core is live (that is how every successful run
    # reports readiness). Treating 404 as "no /health on this build" let a
    # server that had only the HTTP layer up pass the gate, after which the
    # first completion either returned an empty 200 or hit a dead port. Wait
    # for the gate instead; if a build genuinely lacks /health, the timeout
    # below says so with the container log.
    echo "serve: models endpoint up, engine core not healthy yet (/health=$hcode)" >&2
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "serve: container exited before becoming ready. Full log: $TRACE_DIR/$NAME.log" >&2
    docker logs "$NAME" >"$TRACE_DIR/$NAME.log" 2>&1 || true
    echo "serve: first error in the log:" >&2
    grep -m1 -A20 -E 'Error|ERROR|Exception|error:' "$TRACE_DIR/$NAME.log" 2>/dev/null \
      | sed 's/^/  /' >&2 || docker logs --tail=40 "$NAME" 2>&1 | sed 's/^/  /' >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
      | sed 's/^/  gpu: /' >&2 || true
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    echo "serve: timed out after ${READY_TIMEOUT}s. Full log: $TRACE_DIR/$NAME.log" >&2
    docker logs "$NAME" >"$TRACE_DIR/$NAME.log" 2>&1 || true
    docker logs --tail=60 "$NAME" 2>&1 | sed 's/^/  /' >&2 || true
    exit 1
  fi
  sleep 3
done
