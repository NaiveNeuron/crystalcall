#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Resolve the Gemma tool-call delimiter token ids from the pinned checkpoint.

The P-ReAct-style arm's tool-priority decoding needs the *exact* vocabulary ids
of the in-band tool-call delimiters, because the priority mechanism detects the
call span on device with a token-id comparison.  Guessing them is not an
option, so they are read out of the checkpoint's own ``tokenizer.json`` and
written into ``configs/preact.yml``: the ids are data, never a magic number
baked into the vLLM patch.

Usage:
  uv run scripts/resolve_gemma_tokens.py                 # print JSON
  uv run scripts/resolve_gemma_tokens.py --write         # + update configs/preact.yml
  uv run scripts/resolve_gemma_tokens.py --check         # fail if the yml disagrees

Exit codes: 0 ok, 1 a required token is absent from the vocabulary, 2 the
checkpoint could not be located, 3 ``--check`` mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CONFIG = REPO / "configs/preact.yml"

# Gemma's in-band tool-call delimiters.  Note the asymmetric pipe placement:
# the opener is ``<|tool_call>`` and the closer is ``<tool_call|>``.
REQUIRED = {
    "open_token_id": "<|tool_call>",
    "close_token_id": "<tool_call|>",
}
# Recorded for provenance / debugging; not required by the mechanism.
EXTRA = {
    "tool_open_token_id": "<|tool>",
    "tool_close_token_id": "<tool|>",
    "tool_response_open_token_id": "<|tool_response>",
    "tool_response_close_token_id": "<tool_response|>",
    "turn_open_token_id": "<|turn>",
    "turn_close_token_id": "<turn|>",
}


def find_tokenizer(explicit: str | None = None) -> Path:
    """Locate the pinned checkpoint's tokenizer.json, or exit 2."""
    if explicit:
        p = Path(explicit).expanduser()
        if p.is_dir():
            p = p / "tokenizer.json"
        if not p.is_file():
            print(f"resolve_gemma_tokens: no tokenizer.json at {p}", file=sys.stderr)
            raise SystemExit(2)
        return p
    hf = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface"))
    repo = os.environ.get("CRYSTALCALL_MODEL_REPO", "google/diffusiongemma-26B-A4B-it")
    root = hf / "hub" / ("models--" + repo.replace("/", "--")) / "snapshots"
    cands = sorted(root.glob("*/tokenizer.json")) if root.is_dir() else []
    if not cands:
        print(
            f"resolve_gemma_tokens: no tokenizer.json under {root}; pass --tokenizer PATH",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return cands[-1]


def resolve(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    tk = json.loads(raw)
    table: dict[str, int] = {}
    for entry in tk.get("added_tokens") or ():
        table[entry["content"]] = int(entry["id"])
    vocab = (tk.get("model") or {}).get("vocab") or {}
    for content, tid in vocab.items():
        table.setdefault(content, int(tid))

    out: dict[str, object] = {}
    missing: list[str] = []
    for key, content in {**REQUIRED, **EXTRA}.items():
        tid = table.get(content)
        if tid is None:
            if key in REQUIRED:
                missing.append(content)
            continue
        out[key] = tid
    if missing:
        print(
            "resolve_gemma_tokens: the checkpoint vocabulary does not contain "
            + ", ".join(missing)
            + " -- the P-ReAct-style priority mechanism cannot be built without it",
            file=sys.stderr,
        )
        raise SystemExit(1)
    out["open_token"] = REQUIRED["open_token_id"]
    out["close_token"] = REQUIRED["close_token_id"]
    out["tokenizer_sha256"] = hashlib.sha256(raw).hexdigest()
    out["tokenizer_snapshot"] = path.parent.name
    out["tokenizer_path"] = str(path)
    return out


_SCALAR = re.compile(r"^(\s*)([A-Za-z0-9_]+):\s*(?:#.*)?(.*)$")


def rewrite_yaml(text: str, values: dict[str, object]) -> str:
    """Replace ``key: value`` scalars in place, preserving comments and order."""
    remaining = dict(values)
    lines = text.splitlines()
    for i, line in enumerate(lines):
        m = _SCALAR.match(line)
        if not m:
            continue
        indent, key, _rest = m.groups()
        if key not in remaining:
            continue
        val = remaining.pop(key)
        rendered = json.dumps(val) if isinstance(val, str) else str(val)
        comment = ""
        if "#" in line:
            comment = "  " + line[line.index("#") :].strip()
        lines[i] = f"{indent}{key}: {rendered}{comment}"
    if remaining:
        print(
            "resolve_gemma_tokens: configs/preact.yml has no slot for "
            + ", ".join(sorted(remaining)),
            file=sys.stderr,
        )
        raise SystemExit(3)
    return "\n".join(lines) + "\n"


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", help="path to tokenizer.json or its directory")
    ap.add_argument("--write", action="store_true", help="update configs/preact.yml")
    ap.add_argument("--check", action="store_true", help="fail if configs/preact.yml disagrees")
    args = ap.parse_args(argv)

    info = resolve(find_tokenizer(args.tokenizer))
    print(json.dumps(info, indent=2, sort_keys=True))

    if not (args.write or args.check):
        return 0

    ids: dict[str, object] = {k: v for k, v in info.items() if k.endswith("_token_id")}
    ids["tokenizer_sha256"] = info["tokenizer_sha256"]
    ids["tokenizer_snapshot"] = info["tokenizer_snapshot"]
    if not CONFIG.is_file():
        print(f"resolve_gemma_tokens: missing {CONFIG}", file=sys.stderr)
        raise SystemExit(2)
    text = CONFIG.read_text()
    updated = rewrite_yaml(text, ids)
    if updated == text:
        return 0
    if args.check:
        print(
            "resolve_gemma_tokens: configs/preact.yml is out of date (run with --write)",
            file=sys.stderr,
        )
        return 3
    CONFIG.write_text(updated)
    print(f"updated {CONFIG.relative_to(REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
