# /// script
# requires-python = ">=3.12"
# dependencies = ["crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Decide which pilot attempt the next run is, from the *configuration*.

The pilot allows the original attempt plus exactly one formatting-only
prompt repair. Deriving that from which report files exist is wrong on a fresh
clone: `reports/pilot_attempt{1,2}.json` are committed evidence, so a clone
reproducing attempt 2 would be told it is attempting a third and refused, while
`dev-trace` cheerfully regenerates the shard.

The question is really about the prompt. Each report records the
`system_prompt_sha256` it measured, so:

* the current prompt matches a recorded attempt -> that attempt number, and
  re-running it is a faithful reproduction that may overwrite its own record;
* no attempt recorded yet -> 1;
* exactly one recorded and the prompt differs -> 2, the permitted repair;
* two recorded and the prompt matches neither -> refuse: that is a third
  distinct configuration, i.e. retuning against the outcome.

Prints the attempt number on stdout, or a reason on stderr with exit 3.

Run:  uv run scripts/pilot_attempt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from crystalcall.manifest import sha256_file  # noqa: E402


def main() -> int:
    prompt = REPO / "configs/system_prompt.md"
    if not prompt.is_file():
        print(f"pilot_attempt: {prompt} missing", file=sys.stderr)
        return 3
    current = sha256_file(prompt)

    recorded: dict[int, str | None] = {}
    for n in (1, 2):
        path = REPO / f"reports/pilot_attempt{n}.json"
        if not path.is_file():
            continue
        try:
            recorded[n] = json.loads(path.read_text(encoding="utf-8")).get(
                "system_prompt_sha256"
            )
        except (json.JSONDecodeError, OSError):
            recorded[n] = None

    for n, sha in sorted(recorded.items()):
        if sha == current:
            print(n)
            return 0

    if not recorded:
        print(1)
        return 0
    if len(recorded) == 1:
        print(2)
        return 0

    known = ", ".join(f"attempt{n}={str(s)[:12]}" for n, s in sorted(recorded.items()))
    print(
        "pilot_attempt: refusing a third attempt.\n"
        f"  configs/system_prompt.md is {current[:12]}, which matches neither "
        f"recorded attempt ({known}).\n"
        "  The pilot allows one formatting-only prompt repair and two attempts; "
        "a third distinct prompt would be retuning against the pilot outcome.\n"
        "  To start over deliberately: archive reports/pilot_attempt*.json, "
        "reports/attempt*/ and experiments/raw/ , then re-run.",
        file=sys.stderr,
    )
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
