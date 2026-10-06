"""Shared fixtures for the CrystalCall safety core tests.

The registry here is deliberately hand-built from :mod:`crystalcall.effects`
rather than imported from the benchmark tool layer, so the safety tests pin the
*effect rules* and not one particular tool implementation.
"""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Mapping
from typing import Any

import pytest

from crystalcall.effects import (
    EffectKind,
    EffectRegistry,
    ParamKind,
    ParamSpec,
    ResourceClass,
    ToolSpec,
)

SNAPSHOT_A = "repo@aaaaaaaaaaaaaaaa"
SNAPSHOT_B = "repo@bbbbbbbbbbbbbbbb"


def _snapshot_resolver(snapshot_id: str):
    def resolve(_args: Mapping[str, Any]) -> str:
        return snapshot_id

    return resolve


def _arg_snapshot_resolver(args: Mapping[str, Any]) -> str:
    """Snapshot bound to the ``database`` argument, as a real frozen DB would be."""
    db = str(args.get("database", ""))
    if not db:
        raise ValueError("no database argument")
    return f"sqlite:{db}@{hashlib.sha256(db.encode()).hexdigest()[:16]}"


@pytest.fixture
def registry() -> EffectRegistry:
    reg = EffectRegistry(path_root="/srv/frozen")
    reg.register(
        ToolSpec(
            name="calc_sum",
            effect=EffectKind.PURE,
            resource=ResourceClass.CPU,
            params=(
                ParamSpec("a", ParamKind.FLOAT),
                ParamSpec("b", ParamKind.FLOAT),
                ParamSpec("precision", ParamKind.INT, required=False, default=2),
            ),
            expected_duration_ms=500.0,
            aliases=frozenset({"sum_numbers"}),
            description="Deterministic addition.",
        )
    )
    reg.register(
        ToolSpec(
            name="search_repo",
            effect=EffectKind.SNAPSHOT_READ,
            resource=ResourceClass.STORAGE,
            params=(
                ParamSpec("query", ParamKind.STR),
                ParamSpec("path", ParamKind.PATH, required=False, default="."),
                ParamSpec("max_results", ParamKind.INT, required=False, default=50),
            ),
            expected_duration_ms=500.0,
            snapshot_resolver=_snapshot_resolver(SNAPSHOT_A),
            description="Search a frozen repository snapshot.",
        )
    )
    reg.register(
        ToolSpec(
            name="sqlite_query",
            effect=EffectKind.SNAPSHOT_READ,
            resource=ResourceClass.STORAGE,
            params=(
                ParamSpec("database", ParamKind.STR),
                ParamSpec("sql", ParamKind.STR),
            ),
            expected_duration_ms=500.0,
            snapshot_resolver=_arg_snapshot_resolver,
            description="Read a frozen SQLite database.",
        )
    )
    reg.register(
        ToolSpec(
            name="lookup_symbol",
            effect=EffectKind.PURE,
            resource=ResourceClass.CPU,
            params=(ParamSpec("name", ParamKind.STR),),
            expected_duration_ms=500.0,
            description="Deterministic symbol lookup.",
        )
    )
    reg.register(
        ToolSpec(
            name="tag_set",
            effect=EffectKind.PURE,
            resource=ResourceClass.CPU,
            params=(
                ParamSpec(
                    "tags",
                    ParamKind.LIST,
                    order_sensitive=False,
                    element_kind=ParamKind.STR,
                ),
            ),
            expected_duration_ms=100.0,
            description="Order-insensitive tag set.",
        )
    )
    reg.register(
        ToolSpec(
            name="read_live_clock",
            effect=EffectKind.LIVE_READ,
            resource=ResourceClass.CPU,
            params=(),
            expected_duration_ms=10.0,
            description="Live mutable read; never speculated.",
        )
    )
    reg.register(
        ToolSpec(
            name="write_note",
            effect=EffectKind.WRITE,
            resource=ResourceClass.STORAGE,
            params=(
                ParamSpec("path", ParamKind.PATH),
                ParamSpec("text", ParamKind.STR),
            ),
            expected_duration_ms=20.0,
            description="Write; never speculated.",
        )
    )
    return reg


class RecordingTools:
    """Deterministic tool implementations that count their own invocations."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.side_effects: list[str] = []
        self.lock = threading.Lock()
        self.delay_ms = 0.0

    def _record(self, name: str, args: Mapping[str, Any]) -> None:
        with self.lock:
            self.calls.append((name, dict(args)))

    def _maybe_delay(self, cancel: threading.Event) -> None:
        if self.delay_ms <= 0:
            return
        cancel.wait(self.delay_ms / 1000.0)

    def calc_sum(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("calc_sum", args)
        self._maybe_delay(cancel)
        return {"result": round(args["a"] + args["b"], int(args.get("precision", 2)))}

    def search_repo(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("search_repo", args)
        self._maybe_delay(cancel)
        return {"matches": [f"{args['path']}:{args['query']}"], "n": args["max_results"]}

    def sqlite_query(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("sqlite_query", args)
        self._maybe_delay(cancel)
        return {"rows": [args["database"], args["sql"]]}

    def lookup_symbol(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("lookup_symbol", args)
        self._maybe_delay(cancel)
        return {"symbol": args["name"], "kind": "function"}

    def tag_set(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("tag_set", args)
        return {"tags": list(args["tags"])}

    def read_live_clock(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("read_live_clock", args)
        return {"now": len(self.calls)}

    def write_note(self, args: Mapping[str, Any], cancel: threading.Event) -> Any:
        self._record("write_note", args)
        with self.lock:
            self.side_effects.append(f"{args['path']}<-{args['text']}")
        return {"written": args["path"]}

    def impls(self) -> dict[str, Any]:
        return {
            "calc_sum": self.calc_sum,
            "search_repo": self.search_repo,
            "sqlite_query": self.sqlite_query,
            "lookup_symbol": self.lookup_symbol,
            "tag_set": self.tag_set,
            "read_live_clock": self.read_live_clock,
            "write_note": self.write_note,
        }


@pytest.fixture
def tools() -> RecordingTools:
    return RecordingTools()


# --------------------------------------------------------------------------- #
# synthetic diffusion trajectories
# --------------------------------------------------------------------------- #


class CharTokenizer:
    """Character-level tokenizer.

    Exact by construction for :class:`crystalcall.tokens.CanvasDecoder`, which
    lets the tests reason about token spans and per-token confidence without
    pulling in a real model tokenizer.
    """

    PAD = 0x20

    def encode(self, text: str, width: int | None = None) -> list[int]:
        ids = [ord(c) for c in text]
        if width is not None:
            ids = ids[:width] + [self.PAD] * max(0, width - len(ids))
        return ids

    def decode(self, ids, **_kwargs) -> str:
        return "".join(chr(int(i)) for i in ids)

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [chr(int(i)) for i in ids]


@pytest.fixture
def tokenizer() -> CharTokenizer:
    return CharTokenizer()


def make_trajectory(
    canvas_texts,
    *,
    request_id: str = "req-1",
    width: int = 256,
    step_ns: int = 40_000_000,
    start_ns: int = 1_000_000_000,
    confidence: float = 0.99,
    accepted_from_step: int | None = None,
    tokenizer: "CharTokenizer | None" = None,
):
    """Build a synthetic revision list from a sequence of canvas texts.

    ``accepted_from_step`` marks every token as accepted from that step
    onwards, modelling entropy-bound acceptance; before it, nothing is
    accepted.  The final revision is always ``phase="commit"``.
    """
    from crystalcall.trace import CanvasRevision

    tok = tokenizer or CharTokenizer()
    n = len(canvas_texts)
    revs = []
    for i, text in enumerate(canvas_texts):
        ids = tok.encode(text, width=width)
        accepted = accepted_from_step is not None and i >= accepted_from_step
        revs.append(
            CanvasRevision(
                request_id=request_id,
                canvas_id=0,
                step=i,
                monotonic_ns=start_ns + i * step_ns,
                predicted_token_ids=tuple(ids),
                accepted_mask=tuple([accepted] * width),
                token_confidence=tuple([confidence] * width),
                mean_entropy=0.05 if accepted else 0.4,
                converged=(i == n - 1),
                phase="commit" if i == n - 1 else "denoise",
            )
        )
    return revs
