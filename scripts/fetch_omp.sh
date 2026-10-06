#!/usr/bin/env bash
# Vendor the *pinned* omp build next to the corpus it produced.
#
# The host binary can be upgraded out from under a frozen corpus (omp/18.1.4 ->
# omp/18.1.11 silently broke the stateful arm's native tool-call path). This fetches the exact build named by
# `reports/doctor.json` -> `pins.omp.version` from the upstream release, checks
# it against the release SHA256SUMS, and installs it under
# `experiments/vendor/omp/<version>/`, which `resolve_omp_binary` prefers over
# `PATH`.
#
# Idempotent: re-running verifies an existing copy instead of re-downloading.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="${OMP_RELEASE_REPO:-can1357/oh-my-pi}"

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  VERSION="$(python3 - "$ROOT" <<'PY'
import json, sys, pathlib
doctor = pathlib.Path(sys.argv[1]) / "reports/doctor.json"
pins = json.loads(doctor.read_text()).get("pins") or {}
print(str((pins.get("omp") or {}).get("version") or "").split("/")[-1])
PY
)"
fi
[[ -n "$VERSION" ]] || { echo "no omp pin in reports/doctor.json and no version argument" >&2; exit 2; }

case "$(uname -m)" in
  aarch64|arm64) ARCH=arm64 ;;
  x86_64)        ARCH=x64 ;;
  *) echo "unsupported machine $(uname -m)" >&2; exit 2 ;;
esac
ASSET="omp-linux-${ARCH}"
DEST="$ROOT/experiments/vendor/omp/$VERSION"
BASE="https://github.com/$REPO/releases/download/v$VERSION"

mkdir -p "$DEST"
cd "$DEST"

# The checksum file is the authority; fetch it every time so a silently
# re-cut release cannot pass on a stale digest.
curl -fsSL -o SHA256SUMS.txt "$BASE/SHA256SUMS.txt"
grep -q "$ASSET" SHA256SUMS.txt || { echo "release v$VERSION has no $ASSET" >&2; exit 3; }

if [[ -f "$ASSET" ]] && grep "$ASSET" SHA256SUMS.txt | sha256sum -c - >/dev/null 2>&1; then
  echo "already vendored and verified: $DEST/$ASSET"
else
  curl -fsSL -o "$ASSET" "$BASE/$ASSET"
  grep "$ASSET" SHA256SUMS.txt | sha256sum -c -
fi

chmod +x "$ASSET"
printf 'pinned  %s\n' "$("./$ASSET" --version)"
printf 'path    %s\n' "$DEST/$ASSET"
printf 'host    %s\n' "$(omp --version 2>/dev/null || echo 'not on PATH')"
