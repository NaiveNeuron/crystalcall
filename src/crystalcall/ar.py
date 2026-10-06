"""Autoregressive control arms (Gemma 4 AR streaming and AR sPTC).

An autoregressive prefix is *monotonic*: once text is emitted, earlier bytes do
not change. That is exactly the property CrystalCall does not have, and it is
what makes the comparison informative.

To analyse both model families with one code path, an AR stream is projected
into the same :class:`~crystalcall.trace.CanvasRevision` shape: one revision per
observed text delta, whose ``predicted_token_ids`` are the tokens of the prefix
so far and whose ``accepted_mask`` is all-``True`` (a monotonic prefix is
committed by construction). ``analyze_trajectory`` then yields the same
milestones, and for an AR trace the first executable final-matching candidate,
the first invariant candidate and the first fully committed candidate coincide
by construction -- a fact the analysis asserts rather than assumes.

The two AR policies differ in *execution discipline*, which is the actual sPTC
contribution:

``ar_streaming``
    Calls are executed in stream order, serially: the runtime blocks on each
    call before starting the next, as a straightforward streaming executor
    would.

``ar_sptc``
    Each call becomes a future the moment it is textually complete, so
    independent calls overlap, and results are reconciled against the
    authoritative final program.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .trace import CanvasRevision

__all__ = ["TextDelta", "build_ar_revisions", "AR_POLICIES", "assert_monotonic"]

#: Policy names handled by :func:`crystalcall.replay.simulate_policy`.
AR_POLICIES: tuple[str, ...] = ("ar_streaming", "ar_sptc")


@dataclass(frozen=True, slots=True)
class TextDelta:
    monotonic_ns: int
    text: str


def _tokenize(tokenizer: Any, text: str) -> list[int]:
    ids = getattr(tokenizer, "encode_ids", None)
    if callable(ids):
        return list(ids(text))
    enc = getattr(tokenizer, "encode", None)
    if callable(enc):
        out = enc(text)
        if hasattr(out, "ids"):
            return list(out.ids)
        return list(out)
    # Last resort: a stable byte-level projection. Offsets stay exact because
    # CanvasDecoder validates piece concatenation against `decode`.
    return list(text.encode("utf-8"))


def build_ar_revisions(
    deltas: Sequence[TextDelta],
    *,
    request_id: str,
    tokenizer: Any,
    logprob_confidence: Sequence[float] | None = None,
) -> list[CanvasRevision]:
    """Project a monotonic AR stream into revision form.

    ``canvas_id`` is always 0 and ``phase`` is always ``"commit"``: every token
    an AR model has emitted is already committed. The last revision is marked
    ``converged``.
    """
    revs: list[CanvasRevision] = []
    text = ""
    for i, d in enumerate(deltas):
        text += d.text
        ids = _tokenize(tokenizer, text)
        n = len(ids)
        if logprob_confidence is not None:
            conf = tuple(float(c) for c in logprob_confidence[:n])
            if len(conf) < n:
                conf = conf + (1.0,) * (n - len(conf))
        else:
            # No per-token logprobs requested from the server: a committed AR
            # token is certain for gating purposes, and the AR arms are not
            # gated on confidence anyway.
            conf = (1.0,) * n
        revs.append(
            CanvasRevision(
                request_id=request_id,
                canvas_id=0,
                step=i,
                monotonic_ns=d.monotonic_ns,
                predicted_token_ids=tuple(ids),
                accepted_mask=(True,) * n,
                token_confidence=conf,
                mean_entropy=0.0,
                converged=(i == len(deltas) - 1),
                phase="commit",
                prefix_len=0,
            )
        )
    return revs


def assert_monotonic(revs: Sequence[CanvasRevision]) -> dict[str, Any]:
    """Verify the AR prefix property, so replay validity is demonstrated and
    not assumed. Returns a machine-readable check record."""
    violations: list[dict[str, Any]] = []
    for a, b in zip(revs, revs[1:], strict=False):
        pa, pb = a.predicted_token_ids, b.predicted_token_ids
        if len(pb) < len(pa) or pb[: len(pa)] != pa:
            violations.append({"from_step": a.step, "to_step": b.step})
    return {
        "revisions": len(revs),
        "monotonic": not violations,
        "violations": violations,
    }


def deltas_from_events(events: Sequence[dict[str, Any]]) -> list[TextDelta]:
    """Extract timestamped assistant text deltas from an OMP RPC event list."""
    out: list[TextDelta] = []
    for ev in events:
        if ev.get("type") != "message_update":
            continue
        ame = ev.get("assistantMessageEvent") or {}
        if ame.get("type") not in ("text_delta", "toolcall_delta"):
            continue
        piece = ame.get("delta")
        if not isinstance(piece, str) or not piece:
            piece = ame.get("partial") or ame.get("text") or ""
        if not isinstance(piece, str) or not piece:
            continue
        ns = int(ev.get("_recv_ns") or 0)
        out.append(TextDelta(monotonic_ns=ns, text=piece))
    return out
