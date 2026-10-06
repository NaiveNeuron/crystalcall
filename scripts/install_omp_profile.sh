#!/usr/bin/env bash
# Install the pinned OMP-Lite profile.
#
# A named omp profile relocates ALL omp user state (settings, models, sessions,
# blobs, agent.db, skills, commands) under ~/.omp/profiles/<name>/, so this
# cannot disturb the default profile or the local-qwen profile on this box.
#
# Usage: scripts/install_omp_profile.sh [profile-name]
set -euo pipefail

PROFILE="${1:-crystalcall}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="${HOME}/.omp/profiles/${PROFILE}/agent"

mkdir -p "${AGENT_DIR}"

for f in models.yml; do
  src="${ROOT}/configs/${f}"
  dst="${AGENT_DIR}/${f}"
  if [[ -f "${dst}" ]] && ! cmp -s "${src}" "${dst}"; then
    cp -a "${dst}" "${dst}.bak.$(date +%s)"
  fi
  cp "${src}" "${dst}"
  echo "installed ${dst}"
done

# The settings overlay is passed per run with --config so it stays visible in
# the command line and in every run record; it is deliberately NOT copied into
# the profile as config.yml.
echo "overlay: ${ROOT}/configs/omp-lite.yml (pass with --config)"

# A malformed models.yml is skipped SILENTLY by omp — the registry keeps working
# with built-in models — so listing the model is the only real parse test.
export OMP_PROFILE="${PROFILE}"
echo "--- verifying model registry"
if ! omp --profile "${PROFILE}" models find diffusiongemma --json 2>/dev/null | head -c 400; then
  echo "FAILED: omp did not list diffusiongemma; models.yml probably did not parse" >&2
  exit 1
fi
echo
omp --profile "${PROFILE}" models find gemma-4 --json 2>/dev/null | head -c 400 || true
echo
echo "OK: profile ${PROFILE} installed"
