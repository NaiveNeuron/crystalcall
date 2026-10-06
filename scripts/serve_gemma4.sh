#!/usr/bin/env bash
# Launch the autoregressive control server (Gemma 4 26B-A4B-it).
#
#   scripts/serve_gemma4.sh [--foreground] [--no-wait] [--stop] [--logs]
#
# Matched to scripts/serve_diffusiongemma.sh on everything that is not the
# decoding algorithm: same image, same tool-call/reasoning parsers, same
# max_model_len / max_num_seqs / gpu_memory_utilization, same CPU affinity,
# same HF cache mount. No diffusion flags, no sampler patch, no side channel:
# this arm is the AR baseline for the CrystalCall latency comparison.
#
# Env overrides: PORT, HOST, MODEL, SERVED_NAME, MAX_MODEL_LEN, MAX_NUM_SEQS,
# GPU_MEMORY_UTILIZATION, CPUSET, NAME, HF_CACHE, EXTRA_ARGS,
# CRYSTALCALL_VLLM_IMAGE.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVFILE="$REPO/patches/build/image-env.sh"

NAME="${NAME:-cc-gemma4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8001}"
MODEL="${MODEL:-google/gemma-4-26B-A4B-it}"
SERVED_NAME="${SERVED_NAME:-gemma-4-26b-a4b-it}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
CPUSET="${CPUSET:-5-9,15-19}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
READY_TIMEOUT="${READY_TIMEOUT:-1800}"

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

FOREGROUND=0
WAIT=1
for arg in "$@"; do
  case "$arg" in
    --foreground) FOREGROUND=1 ;;
    --no-wait) WAIT=0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

IMAGE="${CRYSTALCALL_VLLM_IMAGE:-vllm/vllm-openai:gemma}"
if [[ -f "$ENVFILE" ]]; then
  # shellcheck source=/dev/null
  source "$ENVFILE"
  IMAGE="${CRYSTALCALL_VLLM_IMAGE:-$IMAGE}"
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true

DOCKER_ARGS=(
  run --name "$NAME"
  --gpus all --ipc=host --network host
  --cpuset-cpus "$CPUSET"
  -v "$HF_CACHE:/root/.cache/huggingface"
  -e HF_HOME=/root/.cache/huggingface
)
[[ -n "${HF_TOKEN:-}" ]] && DOCKER_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")

SERVER_ARGS=(
  --model "$MODEL"
  --served-model-name "$SERVED_NAME"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --generation-config vllm
  --enable-auto-tool-choice
  --tool-call-parser gemma4
  --reasoning-parser gemma4
  --host "$HOST"
  --port "$PORT"
)
# shellcheck disable=SC2206
[[ -n "${EXTRA_ARGS:-}" ]] && SERVER_ARGS+=(${EXTRA_ARGS})

if (( FOREGROUND )); then
  exec docker "${DOCKER_ARGS[@]}" "$IMAGE" "${SERVER_ARGS[@]}"
fi

docker "${DOCKER_ARGS[@]}" -d "$IMAGE" "${SERVER_ARGS[@]}" >/dev/null
echo "started $NAME (autoregressive control)"

(( WAIT )) || exit 0

deadline=$(( $(date +%s) + READY_TIMEOUT ))
while :; do
  code="$(curl -s -o /dev/null -w '%{http_code}' "http://$HOST:$PORT/v1/models" 2>/dev/null || true)"
  if [[ "$code" == "200" ]]; then
    echo "ready: http://$HOST:$PORT/v1/models"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NAME"; then
    echo "serve: container exited before becoming ready; first error and GPU state:" >&2
    docker logs "$NAME" 2>&1 | grep -m1 -A20 -E 'Error|ERROR|Exception|error:' \
      | sed 's/^/  /' >&2 || docker logs --tail=40 "$NAME" 2>&1 | sed 's/^/  /' >&2
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv 2>/dev/null \
      | sed 's/^/  gpu: /' >&2 || true
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    echo "serve: timed out after ${READY_TIMEOUT}s; last 60 log lines:" >&2
    docker logs --tail=60 "$NAME" 2>&1 | sed 's/^/  /' >&2 || true
    exit 1
  fi
  sleep 3
done
