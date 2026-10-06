#!/usr/bin/env bash
# Block until the GPU has enough free unified memory for DiffusionGemma, then exit 0.
#
# DiffusionGemma 26B-A4B is bf16: ~52 GB of weights plus activation and diffusion
# state, so the server needs roughly 60 GB free of the box's 121 GB. On GB10
# `nvidia-smi --query-gpu=memory.used/memory.total` reports [N/A], so free memory
# is derived from the per-process compute-app accounting instead.
#
# This exists because the experiment is single-GPU and other work on the box can
# hold the accelerator. It NEVER stops anybody else's process; it only waits.
#
# --exclusive additionally requires that NOBODY else holds the accelerator.
# Free memory is not a sufficient condition: vLLM sizes its KV budget as a
# fraction of *total* device memory, so with 46 GiB held by a stranger it still
# fails engine-core init even though 75 GB of 121 GB is free. A latency
# measurement needs exclusivity anyway -- a foreign process does not add noise
# to an overhead rung, it invents instrumentation cost.
#
# Usage:
#   scripts/wait_for_gpu.sh [--need-gb 60] [--exclusive] [--timeout 0]
#                           [--interval 60] [--quiet]
#   scripts/wait_for_gpu.sh --exclusive && bash scripts/run_all.sh --stage overhead
set -Eeuo pipefail

NEED_GB=60
EXCLUSIVE=0
TIMEOUT=0          # 0 = wait forever
INTERVAL=60
QUIET=0
TOTAL_GB="${CRYSTALCALL_GPU_TOTAL_GB:-121}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --need-gb) NEED_GB="$2"; shift 2 ;;
    --exclusive) EXCLUSIVE=1; shift ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    --interval) INTERVAL="$2"; shift 2 ;;
    --quiet) QUIET=1; shift ;;
    -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 2; }

used_gb() {
  # Sum of every compute app's GPU memory, in GB. Empty output = nothing resident.
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{s=0} /^[0-9]+$/{s+=$1} END{printf "%.1f", s/1024}'
}

holder_gb() {
  # Largest single compute app, in GB. Sub-GiB processes are the display stack.
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{m=0} /^[0-9]+$/{ if ($1 > m) m=$1 } END{printf "%.1f", m/1024}'
}

describe_holders() {
  nvidia-smi --query-compute-apps=pid,used_memory,process_name \
    --format=csv,noheader 2>/dev/null | sed 's/^/    /'
  if command -v docker >/dev/null; then
    docker ps --format '    container {{.Names}} ({{.Image}}) {{.Status}}' 2>/dev/null || true
  fi
}

start=$(date +%s)
first=1
while true; do
  used=$(used_gb)
  free=$(awk -v t="$TOTAL_GB" -v u="$used" 'BEGIN{printf "%.1f", t-u}')
  biggest=$(holder_gb)
  enough=0
  awk -v f="$free" -v n="$NEED_GB" 'BEGIN{exit !(f >= n)}' && enough=1
  exclusive_ok=1
  if [[ "$EXCLUSIVE" == 1 ]]; then
    awk -v b="$biggest" 'BEGIN{exit !(b >= 1.0)}' && exclusive_ok=0
  fi
  if [[ "$enough" == 1 && "$exclusive_ok" == 1 ]]; then
    if [[ "$QUIET" != 1 ]]; then
      echo "GPU available: ${free} GB free of ${TOTAL_GB} GB (need ${NEED_GB} GB)$(
        [[ "$EXCLUSIVE" == 1 ]] && echo ', exclusive')"
    fi
    exit 0
  fi
  if [[ "$first" == 1 ]]; then
    first=0
    if [[ "$EXCLUSIVE" == 1 && "$exclusive_ok" == 0 ]]; then
      echo "waiting for EXCLUSIVE use of the GPU: another process holds ${biggest} GB"
      echo "  (free memory is not sufficient: vLLM sizes its budget from total"
      echo "   device memory, and an overhead rung must not measure someone else)"
    else
      echo "waiting for the GPU: ${free} GB free of ${TOTAL_GB} GB, need ${NEED_GB} GB"
    fi
    echo "  current holders (this script never stops them):"
    describe_holders
  fi
  now=$(date +%s)
  elapsed=$((now - start))
  if [[ "$TIMEOUT" != 0 && "$elapsed" -ge "$TIMEOUT" ]]; then
    echo "timed out after ${elapsed}s with ${free} GB free" >&2
    describe_holders >&2
    exit 1
  fi
  [[ "$QUIET" == 1 ]] || printf '\r  %6ds elapsed, %s GB free, util %s%%   ' \
    "$elapsed" "$free" "$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)"
  sleep "$INTERVAL"
done
