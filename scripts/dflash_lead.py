#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "jinja2>=3.1", "crystalcall"]
#
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Acceptance granularity on a block-diffusion speculative decoder, and the
resulting UPPER BOUND on any pre-commit lead.

On DiffusionGemma, the lead available to a tool-executing agent runtime
equals ONE acceptance event (~200 ms), invariant to the acceptance threshold.
The law that predicts: the lead is one acceptance
event, whose wall-clock size is set by tokens-per-acceptance-event.

WHAT THIS SCRIPT CAN AND CANNOT DO — read before quoting any number from it.

Measured directly, from the server's own per-request telemetry:
  * `tokens_per_event`  = `spec_accept_length`, tokens the target accepted per
                          verify round.
  * `verify_rounds`     = `spec_verify_ct`.
  * `ms_per_event`      = decode wall time / verify rounds. A mean over rounds,
                          not a per-round measurement.
  * `call_to_done_ms`   = streaming time from the first *committed* text that
                          yields an extractable canonical call, to end of
                          generation. This is the post-commit tail, the analogue
                          of DiffusionGemma's `commit -> canvas emit`.

NOT measured, and deliberately not claimed:
  * The pre-commit lead itself. SGLang streams only **post-verification**
    tokens; rejected draft content is never exposed by the API, and even
    `SGLANG_DSPARK_DEBUG_DUMP` yields draft *lengths* (`correct_len`,
    `commit_lens`), not draft text. So the first stream position at which a call
    is extractable is its COMMIT time, not the time it first became readable in
    a draft.

What follows is therefore a bound, and is labelled as one in the output:

    pre_commit_lead <= ms_per_event

i.e. a call cannot become exploitable more than one verify round before it is
committed, because a drafted block is either accepted whole-prefix or truncated
at the first rejection. Reporting this as a measured lead would present a
derived quantity as measured.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import httpx
from jinja2 import Environment, StrictUndefined

from crystalcall.manifest import load_manifest, registry_for
from crystalcall.parse_ptc import PtcExtractor

SYSTEM_PROMPT = Path("configs/system_prompt.md")


def pct(xs: list[float], p: float) -> float | None:
    if not xs:
        return None
    ys = sorted(xs)
    return round(ys[min(len(ys) - 1, max(0, int(round(p * (len(ys) - 1)))))], 1)


CHAT_TEMPLATE = (
    Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    / "hub/models--Qwen--Qwen3.8-27B/snapshots/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
    / "chat_template.jinja"
)


def build_prompt(system: str, rec: Any, template: Any) -> str:
    """Render the target's own chat template.

    `/generate` takes raw text, and raw text makes an instruct model emit EOS
    immediately (measured: `completion_tokens: 1`, empty output). The OpenAI
    chat route applies the template but does not surface `spec_*` telemetry,
    so the template is applied here and `/generate` keeps the telemetry.
    """
    decls = [
        t.to_json() if hasattr(t, "to_json") else dict(getattr(t, "__dict__", {}) or {})
        for t in (rec.tools or ())
    ]
    tools = json.dumps(decls, indent=2, default=str) if decls else "[]"
    user = (
        f"# Available tools (call them as tool.<name>(...))\n{tools}\n\n"
        f"# Task\n{rec.user_message}\n\n"
        "Reply with a single ```python block that calls the required tools."
    )
    return template.render(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        # The Qwen3.8 template references `tools` and the thinking switches even
        # when no native tool schema is passed; the PTC surface is Python inside
        # a code block, not native tool calls, so these stay empty/off.
        tools=None,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--manifest", default="experiments/manifests/dev.jsonl")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--label", default="dflash2_block8")
    ap.add_argument("--out", default="experiments/sweeps/dflash2/lead.json")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    system = (root / SYSTEM_PROMPT).read_text(encoding="utf-8")
    template = Environment(undefined=StrictUndefined, keep_trailing_newline=True).from_string(
        CHAT_TEMPLATE.read_text(encoding="utf-8")
    )
    records = list(load_manifest(root / args.manifest))[: args.limit]
    rows: list[dict[str, Any]] = []
    with httpx.Client(timeout=600.0) as client:
        for rec in records:
            extractor = PtcExtractor(registry=registry_for(rec))
            payload = {
                "text": build_prompt(system, rec, template),
                "sampling_params": {
                    "max_new_tokens": args.max_tokens,
                    "temperature": 0.0,
                },
                "stream": True,
            }
            text = ""
            t0 = time.monotonic()
            first_token_ms: float | None = None
            commit_ms: float | None = None
            identity: str | None = None
            meta: dict[str, Any] = {}
            tokens_emitted = 0
            tokens_at_commit: int | None = None
            lines_seen = 0
            with client.stream("POST", f"{args.base_url}/generate", json=payload) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    lines_seen += 1
                    if not line or not line.startswith("data:"):
                        continue
                    body = line[len("data:") :].strip()
                    if body == "[DONE]":
                        break
                    chunk = json.loads(body)
                    if first_token_ms is None and chunk.get("text"):
                        first_token_ms = (time.monotonic() - t0) * 1000.0
                    text = chunk.get("text", text)
                    meta = chunk.get("meta_info") or meta
                    ids = chunk.get("output_ids") or []
                    if ids:
                        tokens_emitted = len(ids)
                    if commit_ms is None and text:
                        res = extractor.extract(text)
                        if res.candidates:
                            commit_ms = (time.monotonic() - t0) * 1000.0
                            identity = res.candidates[0].identity
                            # Exactly counted but chunk-granular: extraction is
                            # tried once per SSE chunk, so the call became
                            # parseable somewhere inside the tokens this chunk
                            # delivered. `tail_tokens` is a lower bound, short
                            # by up to one stream chunk (server-side
                            # `stream_interval`, not recorded here).
                            tokens_at_commit = tokens_emitted
            done_ms = (time.monotonic() - t0) * 1000.0
            if not lines_seen:
                raise RuntimeError(f"no stream lines for {rec.prompt_id}")
            rounds = int(meta.get("spec_verify_ct") or 0)
            rows.append(
                {
                    "prompt_id": rec.prompt_id,
                    "category": rec.category,
                    "completion_tokens": meta.get("completion_tokens"),
                    "e2e_ms": round(done_ms, 1),
                    # measured
                    "tokens_per_event": meta.get("spec_accept_length"),
                    "verify_rounds": rounds,
                    "spec_accept_rate": meta.get("spec_accept_rate"),
                    "spec_accept_histogram": meta.get("spec_accept_histogram"),
                    # Decode-only mean over verify rounds: prefill is excluded by
                    # starting the clock at the first streamed token, otherwise
                    # time-to-first-token is charged to the acceptance event.
                    "ttft_ms": round(first_token_ms, 1) if first_token_ms else None,
                    "ms_per_event": (
                        round((done_ms - first_token_ms) / rounds, 1)
                        if rounds and first_token_ms
                        else None
                    ),
                    # commit time of the first extractable call, NOT its draft time
                    "call_commit_ms": round(commit_ms, 1) if commit_ms else None,
                    "call_to_done_ms": (
                        round(done_ms - commit_ms, 1) if commit_ms else None
                    ),
                    "tokens_at_commit": tokens_at_commit,
                    "tail_tokens": (
                        tokens_emitted - tokens_at_commit
                        if tokens_at_commit is not None
                        else None
                    ),
                    "tokens_emitted": tokens_emitted,
                    "first_call_identity": identity,
                    "final_text_head": text[:200],
                }
            )
            print(json.dumps({k: rows[-1][k] for k in
                              ("prompt_id", "tokens_per_event", "verify_rounds",
                               "ms_per_event", "call_commit_ms", "call_to_done_ms")}))

    def col(key: str) -> list[float]:
        return [float(r[key]) for r in rows if r.get(key) is not None]

    def band(key: str) -> dict[str, float | None]:
        xs = col(key)
        return {"p25": pct(xs, 0.25), "p50": pct(xs, 0.5), "p75": pct(xs, 0.75), "n": len(xs)}

    summary = {
        "label": args.label,
        "n_prompts": len(rows),
        "n_with_extractable_call": sum(1 for r in rows if r["call_commit_ms"]),
        "measured": {
            "tokens_per_acceptance_event": band("tokens_per_event"),
            "ms_per_acceptance_event_mean_over_rounds": band("ms_per_event"),
            "post_commit_tail_ms": band("call_to_done_ms"),
            "post_commit_tail_tokens": band("tail_tokens"),
            "e2e_ms": band("e2e_ms"),
        },
        "derived_bound": {
            "pre_commit_lead_upper_bound_ms": pct(col("ms_per_event"), 0.5),
            "why": (
                "SGLang streams only post-verification tokens and never exposes "
                "rejected draft content, so the draft time of a call is not "
                "observable. A drafted block is accepted as a prefix or truncated "
                "at the first rejection, so no call can become exploitable more "
                "than one verify round before it commits. This is a bound, not a "
                "measurement."
            ),
        },
        "mean_tokens_per_event": (
            round(statistics.fmean(col("tokens_per_event")), 2)
            if col("tokens_per_event")
            else None
        ),
    }
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
