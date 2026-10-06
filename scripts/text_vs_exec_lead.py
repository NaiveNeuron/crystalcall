#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["tokenizers>=0.20", "crystalcall"]
#
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Upper-bound what any extractor could recover from a canvas.

The measured pre-commit lead is flat at ~200 ms regardless of program length
(`corr(program_len, lead_ms) = 0.019` over 61 trajectories), which is consistent
with two different binding constraints:

  (a) the sampler's adaptive stopper halts one step after the argmax canvas
      stabilises, so any retrospectively measured lead is ~1 step by
      construction;
  (b) the extractor requires a whole region to `ast.parse`, and since
      non-accepted positions are renoised to uniformly random tokens, the parse
      only succeeds once nearly everything is accepted.

This script separates them without a GPU and without writing a new extractor.
For every final call it finds `first_text_step`: the first revision whose
decoded canvas contains the call's final `source_text` byte-for-byte. That is an
upper bound on what ANY parser could extract, because no extractor can recover a
call whose text is not yet on the canvas.

  first_text_step << first_exec_step  -> the extractor is binding (b)
  first_text_step == first_exec_step  -> the text itself settles one step before
                                         acceptance and no parser can help (a)

Both `accepted` and `text-only` variants are reported: a call's text can be
present while its tokens are still unaccepted, and only the accepted variant is
launchable under the effect/quarantine rules.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from crystalcall.harness import load_tokenizer, resolve_model_dir
from crystalcall.manifest import load_manifest, registry_for
from crystalcall.parse_ptc import PtcExtractor
from crystalcall.tokens import CanvasDecoder

MODEL = "google/diffusiongemma-26B-A4B-it"


def pct(values: list[float], p: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    idx = min(len(xs) - 1, max(0, int(round(p * (len(xs) - 1)))))
    return round(xs[idx], 1)


def load_revisions(shard: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    finals: dict[str, dict[str, Any]] = {}
    for line in shard.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        kind = rec.get("kind")
        if kind == "revision":
            out[rec["request_id"]].append(rec)
        elif kind == "final":
            finals[rec["request_id"]] = rec
    for rid in out:
        out[rid].sort(key=lambda r: (r["canvas_id"], r["step"]))
    return out, finals


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--shard",
        default="experiments/raw/diffusion_standard/diffusion_standard-seed0.jsonl",
    )
    ap.add_argument("--manifest", default="experiments/manifests/dev.jsonl")
    ap.add_argument("--out", default="experiments/derived/text_vs_exec_lead.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    shard = (root / args.shard) if not Path(args.shard).is_absolute() else Path(args.shard)
    revisions, _finals = load_revisions(shard)

    tok = load_tokenizer(resolve_model_dir(MODEL))
    decoder = CanvasDecoder(tok)
    manifest = root / args.manifest
    extractors = {
        rec.prompt_id: PtcExtractor(registry=registry_for(rec))
        for rec in load_manifest(manifest)
    }

    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    for rid, revs in revisions.items():
        prompt_id = rid.split("/")[1] if "/" in rid else rid
        extractor = extractors.get(prompt_id)
        if extractor is None:
            skipped.append(rid)
            continue
        by_canvas: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rev in revs:
            by_canvas[rev["canvas_id"]].append(rev)
        for canvas_id, crevs in sorted(by_canvas.items()):
            decoded = [decoder.decode(r["predicted_token_ids"]) for r in crevs]
            final_result = extractor.extract(decoded[-1].text)
            for cand in final_result.candidates:
                text = cand.source_text
                first_text = first_text_accepted = first_exec = -1
                for i, (rev, dc) in enumerate(zip(crevs, decoded)):
                    pos = dc.text.find(text)
                    if pos >= 0:
                        if first_text < 0:
                            first_text = rev["step"]
                        if first_text_accepted < 0:
                            lo, hi = dc.token_range(pos, pos + len(text))
                            span = rev["accepted_mask"][lo:hi]
                            if span and all(span):
                                first_text_accepted = rev["step"]
                    if first_exec < 0:
                        res = extractor.extract(dc.text)
                        if any(c.identity == cand.identity for c in res.candidates):
                            first_exec = rev["step"]
                    if first_text >= 0 and first_text_accepted >= 0 and first_exec >= 0:
                        break
                rows.append(
                    {
                        "request_id": rid,
                        "canvas_id": canvas_id,
                        "identity": cand.identity,
                        "tool": cand.call.tool,
                        "steps_in_canvas": crevs[-1]["step"] + 1,
                        "first_text_step": first_text,
                        "first_text_accepted_step": first_text_accepted,
                        "first_exec_step": first_exec,
                        "text_len": len(text),
                    }
                )

    def deltas(key: str) -> list[float]:
        return [
            float(r["first_exec_step"] - r[key])
            for r in rows
            if r["first_exec_step"] >= 0 and r[key] >= 0
        ]

    text_delta = deltas("first_text_step")
    acc_delta = deltas("first_text_accepted_step")
    summary = {
        "shard": str(shard.relative_to(root)),
        "calls": len(rows),
        "text_before_exec_steps": {
            "n": len(text_delta),
            "p50": pct(text_delta, 0.5),
            "p90": pct(text_delta, 0.9),
            "max": max(text_delta) if text_delta else None,
            "share_gt0": (
                round(100.0 * sum(1 for d in text_delta if d > 0) / len(text_delta), 1)
                if text_delta
                else None
            ),
            "mean": round(statistics.fmean(text_delta), 2) if text_delta else None,
        },
        "accepted_text_before_exec_steps": {
            "n": len(acc_delta),
            "p50": pct(acc_delta, 0.5),
            "p90": pct(acc_delta, 0.9),
            "max": max(acc_delta) if acc_delta else None,
            "share_gt0": (
                round(100.0 * sum(1 for d in acc_delta if d > 0) / len(acc_delta), 1)
                if acc_delta
                else None
            ),
        },
        "verdict_rule": (
            "text_before_exec p50 > 0 => the whole-region parse is binding and a "
            "finer extractor can recover lead; == 0 => the canvas text itself only "
            "settles when acceptance completes and no extractor can help"
        ),
    }
    out = (root / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
