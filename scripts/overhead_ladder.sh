#!/usr/bin/env bash
# Phase 0 instrumentation overhead ladder.
#
# Four rungs, each a single `crystalcall overhead` invocation that owns its own
# rung end to end: it binds the revision sink FIRST, then starts the server with
# the matching CRYSTALCALL_TRACE* environment, runs matched requests at
# temperature 0, stops the server, and reports both request latency and what
# actually arrived on the side channel.
#
#   off        no side channel consumer, no server-side emission
#   nopayload  sampler trace generation without per-token payload transfer
#   full       compact transfer enabled and received, nothing parsed
#   parse      decode + AST parse + canonicalise + gate per revision, no launch
#
# Binding the sink before the server matters: the emitter reconnects with
# backoff, so a sink started afterwards would silently lose the first
# revisions and the rung would measure a cheaper path than it claims.
#
# A rung that received no side-channel events, or a `parse` rung that parsed
# nothing, is reported as `ok: false` and fails this script.
#
# Usage: scripts/overhead_ladder.sh [n_requests]
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
N="${1:-30}"
SINK="${CRYSTALCALL_TRACE_SINK:-127.0.0.1:9801}"
OUT="${ROOT}/experiments/derived/overhead"
SERVE="${ROOT}/scripts/serve_diffusiongemma.sh"

if [[ ! -f "${SERVE}" ]]; then
  echo "missing ${SERVE} — install the instrumentation patch first" >&2
  exit 2
fi
mkdir -p "${OUT}"

SEED="${CRYSTALCALL_LADDER_SEED:-20260903}"
read -r -a LEVELS <<< "$(
  python3 -c '
import random, sys
levels = ["off", "nopayload", "full", "parse"]
random.Random(int(sys.argv[1])).shuffle(levels)
print(" ".join(levels))
' "$SEED"
)"
ORDER="${LEVELS[*]}"
echo "=== randomized rung order (seed ${SEED}): ${ORDER}"

cleanup() { bash "${SERVE}" --stop >/dev/null 2>&1 || true; }
trap cleanup EXIT INT TERM

for level in "${LEVELS[@]}"; do
  echo "=== rung ${level}"
  echo "+ uv run --project ${ROOT} crystalcall overhead --level ${level} --prompts ${N} --sink ${SINK}"
  CRYSTALCALL_LADDER_SEED="$SEED" CRYSTALCALL_LADDER_ORDER="$ORDER" \
    uv run --project "${ROOT}" crystalcall overhead \
      --level "${level}" --prompts "${N}" --sink "${SINK}"
done

echo
echo "=== ladder summary (measured order: ${ORDER})"
uv run --project "${ROOT}" python - <<'PY'
import json, pathlib, sys

out = pathlib.Path("experiments/derived/overhead")
order = ["off", "nopayload", "full", "parse"]
rows = {}
for level in order:
    p = out / f"overhead-{level}.json"
    if p.exists():
        rows[level] = json.loads(p.read_text())

base = (rows.get("off") or {}).get("p50_ms")
hdr = f"{'rung':<11}{'n':>4}{'p50 ms':>11}{'p95 ms':>11}{'rev/req':>9}{'vs off':>9}{'gpu':>6}  status"
print(hdr)
print("-" * len(hdr))
bad = []
for level in order:
    r = rows.get(level)
    if r is None:
        print(f"{level:<11}{'—':>4}{'—':>11}{'—':>11}{'—':>9}{'—':>9}{'—':>6}  MISSING")
        bad.append(f"{level}: missing")
        continue
    rel = "" if not base else f"{100 * (r['p50_ms'] / base - 1):+.1f}%"
    status = "ok" if r.get("ok") else "; ".join(r.get("problems") or ["not ok"])
    # `excl` means the accelerator was ours alone; a shared rung measures
    # someone else's workload, not instrumentation.
    # A rung with no exclusivity evidence is refused, not trusted: a rung
    # written by a pre-detection build (schema /2) could be contaminated and
    # would otherwise be read as valid.
    if "gpu_exclusive" not in r:
        excl = "UNKNOWN"
        bad.append(
            f"{level}: no gpu_exclusive field (schema {r.get('schema_version')}); "
            "re-measure -- a rung without exclusivity evidence is not usable"
        )
    else:
        excl = "excl" if r.get("gpu_exclusive") else "SHARED"
    print(
        f"{level:<11}{r['n']:>4}{r['p50_ms']:>11.1f}{r['p95_ms']:>11.1f}"
        f"{r['revisions_per_request']:>9.1f}{rel:>9}{excl:>6}  {status}"
    )
    if not r.get("ok"):
        bad.append(f"{level}: {status}")

parse = (rows.get("parse") or {}).get("p50_ms")
if base and parse:
    # Completion length, prompt identity, and request occurrence are part of
    # measured work. Fixed `ignore_eos` requests should match exactly across
    # every rung; refuse a latency verdict otherwise.
    off_r = rows.get("off") or {}
    parse_r = rows.get("parse") or {}
    off_l = list(off_r.get("latencies_ms") or [])
    par_l = list(parse_r.get("latencies_ms") or [])
    off_tokens = list(off_r.get("completion_tokens") or [])
    off_ids = list(off_r.get("prompt_indices") or [
        i % 5 for i in range(len(off_l))
    ])
    off_occ = list(off_r.get("request_occurrence_ids") or range(len(off_l)))
    workload_match = True
    for other in order:
        r = rows.get(other) or {}
        lat = list(r.get("latencies_ms") or [])
        tokens = list(r.get("completion_tokens") or [])
        ids = list(r.get("prompt_indices") or [i % 5 for i in range(len(lat))])
        occ = list(r.get("request_occurrence_ids") or range(len(lat)))
        if tokens != off_tokens or ids != off_ids or occ != off_occ:
            workload_match = False
            break
    if not workload_match:
        print(
            "\nworkload mismatch: rung token, prompt-index, or occurrence-id "
            "sequences differ; raw latency is not a matched comparison"
        )

    # Pair off and parse by request occurrence, then resample prompt clusters.
    # The bootstrap statistic is the median of per-request paired relative
    # deltas, not a ratio of two independently resampled medians.
    parse_occ = list(parse_r.get("request_occurrence_ids") or range(len(par_l)))
    off_pos = {int(occ): i for i, occ in enumerate(off_occ)}
    parse_pos = {int(occ): i for i, occ in enumerate(parse_occ)}
    paired = [
        (off_pos[int(occ)], parse_pos[int(occ)])
        for occ in off_occ
        if int(occ) in parse_pos
    ]
    deltas_by_prompt = {}
    deltas = []
    for oi, pi in paired:
        if off_l[oi] <= 0:
            continue
        pid = int(off_ids[oi])
        delta_i = 100 * (par_l[pi] / off_l[oi] - 1)
        deltas_by_prompt.setdefault(pid, []).append(delta_i)
        deltas.append(delta_i)
    cluster_ids = sorted(deltas_by_prompt)
    lo = hi = None
    if workload_match and len(deltas) >= 5 and len(cluster_ids) >= 2:
        import random
        rng = random.Random(0)
        boot = []
        for _ in range(5000):
            chosen = [rng.choice(cluster_ids) for _ in cluster_ids]
            vals = sorted(
                value
                for cid in chosen
                for value in deltas_by_prompt[cid]
            )
            boot.append(vals[len(vals) // 2])
        boot.sort()
        lo = boot[int(0.025 * len(boot))]
        hi = boot[int(0.975 * len(boot))]
    if not workload_match:
        verdict = "NOT RESOLVABLE: request workloads differ"
    elif lo is None or hi is None:
        verdict = (
            "NOT RESOLVABLE: too few matched prompt clusters to bound the delta"
        )
    elif hi < 0:
        verdict = (
            f"NOT RESOLVABLE: CI is wholly negative [{lo:+.1f}%, {hi:+.1f}%]; "
            "rung order or workload drift is confounded"
        )
    elif lo <= 0 <= hi:
        verdict = (
            f"NOT RESOLVABLE at this n: 95% CI [{lo:+.1f}%, {hi:+.1f}%] spans zero"
        )
    elif hi <= 5:
        verdict = f"OK (<=5%): 95% CI [{lo:+.1f}%, {hi:+.1f}%]"
    elif hi <= 10:
        verdict = f"INVESTIGATE (5-10%): 95% CI [{lo:+.1f}%, {hi:+.1f}%]"
    else:
        verdict = f"BLOCKING (>10%): 95% CI [{lo:+.1f}%, {hi:+.1f}%]"
    point = sorted(deltas)[len(deltas) // 2] if deltas else float("nan")
    print(
        f"\npaired trace-and-parser path: {point:+.1f}% median per-request delta "
        f"(prompt clusters={len(cluster_ids)}) -> {verdict}"
    )
    per_rev = rows["parse"].get("parse_ms_per_revision")
    dec_rev = rows["parse"].get("decode_ms_per_revision")
    print(f"per revision: decode {dec_rev} ms, parse+gate {per_rev} ms")
else:
    bad.append("cannot compute the trace-and-parser delta: a rung is missing")

if bad:
    print("\nFAILED RUNGS:", file=sys.stderr)
    for b in bad:
        print(f"  - {b}", file=sys.stderr)
    raise SystemExit(1)
print("\nladder complete")
PY
