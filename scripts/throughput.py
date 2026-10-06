#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20", "crystalcall"]
#
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Output-token throughput per arm, straight off the raw trace shards.

The engine never recorded an emitted-token count in the side channel, so the
only honest token quantity available post-hoc is the *re-tokenisation* of
`final_text` under the arm's own tokenizer: `retokenized_final_output_tokens`.
That is what this script reports, and the JSON says so in
`token_quantity`/`token_quantity_note` so a reader cannot mistake it for the
sampler's step count (a diffusion step can accept several tokens at once, and
rejected/renoised positions never appear in `final_text` at all).

Seconds use exactly the boundary the paper's `decode_ms` already uses --
`(response_done_ns - request_start_ns)` of the program-producing request
(`src/crystalcall/replay.py:136-137`) -- so tokens/s here is commensurable with
every latency number in the paper rather than a second, subtly different clock.

Shards are globbed at run time: a diffusion seed traced later is picked up
without editing this file. Shards are read-only here; nothing is rewritten.

    uv run scripts/throughput.py [--out experiments/derived/throughput.json]
"""

from __future__ import annotations

import argparse
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tokenizers import Tokenizer  # noqa: E402

from crystalcall.harness import resolve_model_dir  # noqa: E402
from crystalcall.manifest import sha256_file  # noqa: E402

#: Arm -> (shard glob, tokenizer/model id, serving script).
ARMS: dict[str, tuple[str, str, str]] = {
    "diffusion_standard": (
        "experiments/raw/diffusion_standard/*.jsonl",
        "google/diffusiongemma-26B-A4B-it",
        "scripts/serve_diffusiongemma.sh",
    ),
    "gemma4_ar": (
        "experiments/raw/gemma4_ar/*.jsonl",
        "google/gemma-4-26B-A4B-it",
        "scripts/serve_gemma4.sh",
    ),
}

#: Serving knobs read out of the launcher scripts. Each entry is the shell
#: variable's default assignment (`VAR="${VAR:-value}"`), which is the value
#: every run in `experiments/raw` was traced under -- no env override is
#: recorded in the shards, so the script default *is* the configuration.
_DEFAULT_RE = r'^{var}="\$\{{{var}:-(?P<value>[^}}]*)\}}"'
_SCALARS = (
    ("max_model_len", "MAX_MODEL_LEN"),
    ("max_num_seqs", "MAX_NUM_SEQS"),
    ("gpu_memory_utilization", "GPU_MEMORY_UTILIZATION"),
    ("canvas_length", "CANVAS_LENGTH"),
    ("entropy_bound", "ENTROPY_BOUND"),
    ("generation_config", "GENERATION_CONFIG"),
)


def _shell_default(text: str, var: str) -> str | None:
    m = re.search(_DEFAULT_RE.format(var=var), text, re.MULTILINE)
    return m.group("value") if m else None


def _serving(script_rel: str) -> dict[str, Any]:
    """Record exactly what the launcher script pins, nothing inferred."""
    text = (REPO / script_rel).read_text()
    out: dict[str, Any] = {"script": script_rel}
    for key, var in _SCALARS:
        val = _shell_default(text, var)
        if val is not None:
            out[key] = val
    # `--seed` is only appended when SEED is non-empty; `SEED="${SEED-12345}"`
    # uses `-` (unset), not `:-`, so it needs its own pattern.
    seed = re.search(r'^SEED="\$\{SEED-(?P<value>[^}]*)\}"', text, re.MULTILINE)
    if seed:
        out["engine_seed"] = seed.group("value")
    # Image: the diffusion launcher takes it from the generated
    # patches/build/image-env.sh (required); the AR launcher inlines a default.
    img = re.search(r'^IMAGE="\$\{CRYSTALCALL_VLLM_IMAGE:-(?P<value>[^}]*)\}"', text, re.MULTILINE)
    if img:
        out["image"] = img.group("value")
        out["image_source"] = "script default (CRYSTALCALL_VLLM_IMAGE override)"
    elif 'IMAGE="${CRYSTALCALL_VLLM_IMAGE:?}"' in text:
        env = (REPO / "patches/build/image-env.sh").read_text()
        pin = re.search(r"^CRYSTALCALL_VLLM_IMAGE='(?P<value>[^']*)'", env, re.MULTILINE)
        digest = re.search(r"^CRYSTALCALL_VLLM_DIGEST='(?P<value>[^']*)'", env, re.MULTILINE)
        out["image"] = pin.group("value") if pin else None
        out["image_digest"] = digest.group("value") if digest else None
        out["image_source"] = "patches/build/image-env.sh"
    # Diffusion-specific server flags, verbatim from the SERVER_ARGS array.
    diff_flags = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("--diffusion-config") or line.strip().startswith("--hf-overrides")
    ]
    if diff_flags:
        out["diffusion_flags"] = diff_flags
        hf = re.search(r'^HF_OVERRIDES="(?P<value>.*)"$', text, re.MULTILINE)
        if hf:
            tmpl = hf.group("value").replace('\\"', '"')
            out["hf_overrides_template"] = tmpl
            # The closing brace is appended by a later line; STABILITY_THRESHOLD
            # is only spliced in when the caller exports it (unset by default,
            # and inert anyway), so the value the traced
            # server actually received is the template plus "}".
            out["hf_overrides_effective"] = (
                tmpl.replace("$ENTROPY_BOUND", out.get("entropy_bound", "")) + "}"
            )
        # No denoising-step count is pinned anywhere in the launcher: the
        # entropy-bound sampler stops on its own criterion, so recording a
        # `denoising_steps` value would be an invention.
        out["denoising_steps"] = None
        out["denoising_steps_note"] = (
            "not a server flag: the entropy_bound sampler decides step count at run time"
        )
    return out


def _stats(xs: list[float]) -> dict[str, float | None]:
    if not xs:
        return {"p50": None, "mean": None}
    return {"p50": float(st.median(xs)), "mean": float(st.fmean(xs))}


def _summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    toks = [float(r["tokens"]) for r in rows]
    secs = [float(r["seconds"]) for r in rows]
    tps = [float(r["tokens_per_s"]) for r in rows if r["tokens_per_s"] is not None]
    tot_t, tot_s = sum(toks), sum(secs)
    return {
        "n_trajectories": len(rows),
        "n_empty_final_text": sum(1 for r in rows if r["tokens"] == 0),
        "tokens": _stats(toks),
        "seconds": _stats(secs),
        "tokens_per_s": _stats(tps),
        "aggregate_tokens_per_s": (tot_t / tot_s) if tot_s > 0 else None,
        "total_tokens": int(tot_t),
        "total_seconds": tot_s,
    }


def _trajectories(shard: Path) -> list[dict[str, Any]]:
    """Pair each `final` with the request carrying the same request_id.

    The shards hold exactly one `request` per `request_id` (verified: 100
    requests / 100 finals / 100 distinct ids per shard), so the
    program-producing request is that request; a `final` without its request,
    or a request without a `final` (a shard still being appended to), is
    dropped and counted.
    """
    requests: dict[str, dict[str, Any]] = {}
    finals: dict[str, dict[str, Any]] = {}
    for line in shard.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "request":
            requests[rec["request_id"]] = rec
        elif kind == "final":
            finals[rec["request_id"]] = rec
    rows = []
    for rid, fin in finals.items():
        req = requests.get(rid)
        if req is None:
            continue
        rows.append(
            {
                "request_id": rid,
                "prompt_id": req.get("prompt_id"),
                "category": req.get("category"),
                "seed": req.get("seed"),
                "final_text": fin.get("final_text") or "",
                "request_start_ns": int(req["request_start_ns"]),
                "response_done_ns": int(fin["response_done_ns"]),
            }
        )
    rows.sort(key=lambda r: str(r["request_id"]))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="experiments/derived/throughput.json")
    args = ap.parse_args()

    out: dict[str, Any] = {
        "token_quantity": "retokenized_final_output_tokens",
        "token_quantity_note": (
            "tokens = len(tokenizer.encode(final_text, add_special_tokens=False)) under the "
            "arm's own tokenizer; this is a post-hoc re-tokenisation of the final text, NOT the "
            "engine's emitted-token count (the side channel records no emitted-token counter)."
        ),
        "seconds_definition": (
            "(response_done_ns - request_start_ns)/1e9 of the program-producing request, the "
            "same boundary as Trajectory.decode_ms (src/crystalcall/replay.py:136-137)"
        ),
        "arms": {},
    }
    table: list[tuple[str, str, int, float, float, float, float]] = []

    for arm, (pattern, model_id, serve_script) in ARMS.items():
        shards = sorted(REPO.glob(pattern))
        model_dir = resolve_model_dir(model_id)
        # `crystalcall.harness.load_tokenizer` returns a decode-only adapter;
        # re-tokenising text needs the encoder, so the same tokenizer.json is
        # loaded directly here (identical vocabulary and merges).
        tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        arm_rows: list[dict[str, Any]] = []
        per_seed: dict[str, Any] = {}
        shard_meta: list[dict[str, Any]] = []
        for shard in shards:
            rows = _trajectories(shard)
            for r in rows:
                text = r["final_text"]
                n = len(tok.encode(text, add_special_tokens=False).ids) if text else 0
                secs = (r["response_done_ns"] - r["request_start_ns"]) / 1e9
                r["tokens"] = n
                r["seconds"] = secs
                r["tokens_per_s"] = (n / secs) if secs > 0 else None
            seeds = sorted({r["seed"] for r in rows})
            key = ",".join(str(s) for s in seeds) if seeds else shard.stem
            summary = _summarise(rows)
            summary["shard"] = str(shard.relative_to(REPO))
            summary["shard_sha256"] = sha256_file(shard)
            summary["seeds"] = seeds
            per_seed[key] = summary
            shard_meta.append(
                {
                    "shard": summary["shard"],
                    "sha256": summary["shard_sha256"],
                    "seeds": seeds,
                    "n_trajectories": summary["n_trajectories"],
                }
            )
            arm_rows.extend(rows)
            table.append(
                (
                    arm,
                    f"seed {key}",
                    summary["n_trajectories"],
                    summary["tokens"]["p50"] or 0.0,
                    summary["seconds"]["p50"] or 0.0,
                    summary["tokens_per_s"]["p50"] or 0.0,
                    summary["aggregate_tokens_per_s"] or 0.0,
                )
            )
        pooled = _summarise(arm_rows)
        out["arms"][arm] = {
            "tokenizer_model_id": model_id,
            "tokenizer_dir": str(model_dir),
            "shards": shard_meta,
            "serving": _serving(serve_script),
            "per_seed": per_seed,
            "pooled": pooled,
        }
        table.append(
            (
                arm,
                "pooled",
                pooled["n_trajectories"],
                pooled["tokens"]["p50"] or 0.0,
                pooled["seconds"]["p50"] or 0.0,
                pooled["tokens_per_s"]["p50"] or 0.0,
                pooled["aggregate_tokens_per_s"] or 0.0,
            )
        )

    dest = REPO / args.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")

    hdr = f"{'arm':<20} {'slice':<12} {'n':>4} {'tok p50':>8} {'s p50':>8} {'tok/s p50':>10} {'tok/s agg':>10}"
    print(hdr)
    print("-" * len(hdr))
    for arm, slice_, n, tp, sp, tps, agg in table:
        print(f"{arm:<20} {slice_:<12} {n:>4} {tp:>8.0f} {sp:>8.2f} {tps:>10.2f} {agg:>10.2f}")
    print(f"\nwrote {dest.relative_to(REPO)} (sha256 {sha256_file(dest)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
