"""Canvas decoding with exact per-token character offsets.

The launch gate needs to align a parsed candidate's *character* span in the
decoded canvas with the sampler's per-token confidence and accepted mask.  That
requires a decode that reports, for every token position, the half-open
character range it contributed.

Strategy: decode piecewise from token pieces (fast, O(n)) and validate the
concatenation against the tokenizer's own ``decode`` output.  On any mismatch
fall back to exact prefix decoding for that canvas, which is O(n^2) but
correct.  The fallback rate is recorded so it can be reported.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["CanvasDecoder", "DecodedCanvas", "TokenizerLike"]


class TokenizerLike(Protocol):
    def decode(self, ids: Sequence[int], **kwargs: Any) -> str: ...
    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]: ...


@dataclass(frozen=True, slots=True)
class DecodedCanvas:
    text: str
    #: ``offsets[i] == (start, end)`` character range contributed by token ``i``.
    offsets: tuple[tuple[int, int], ...]
    exact: bool

    def token_range(self, char_lo: int, char_hi: int) -> tuple[int, int]:
        """Smallest token index range covering ``[char_lo, char_hi)``."""
        lo = None
        hi = None
        for i, (a, b) in enumerate(self.offsets):
            if b <= char_lo:
                continue
            if a >= char_hi:
                break
            if lo is None:
                lo = i
            hi = i + 1
        if lo is None or hi is None:
            return (0, 0)
        return (lo, hi)


@dataclass
class CanvasDecoder:
    """Decodes canvases and caches offsets per token-id tuple."""

    tokenizer: TokenizerLike
    #: SentencePiece word-boundary marker used by the Gemma tokenizer family.
    space_marker: str = "\u2581"
    cache_size: int = 512
    fallbacks: int = 0
    decodes: int = 0
    _cache: dict[tuple[int, ...], DecodedCanvas] = field(default_factory=dict, repr=False)

    def decode(self, ids: Sequence[int]) -> DecodedCanvas:
        key = tuple(int(i) for i in ids)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        out = self._decode_uncached(key)
        if len(self._cache) >= self.cache_size:
            self._cache.clear()
        self._cache[key] = out
        return out

    def _decode_uncached(self, ids: tuple[int, ...]) -> DecodedCanvas:
        self.decodes += 1
        reference = self.tokenizer.decode(ids, skip_special_tokens=False)
        pieces = self.tokenizer.convert_ids_to_tokens(ids)
        rendered: list[str] = []
        for p in pieces:
            if p is None:
                rendered.append("")
            elif p.startswith(self.space_marker):
                rendered.append(" " + p[len(self.space_marker) :])
            else:
                rendered.append(p.replace(self.space_marker, " "))
        joined = "".join(rendered)
        if joined == reference:
            offsets: list[tuple[int, int]] = []
            pos = 0
            for r in rendered:
                offsets.append((pos, pos + len(r)))
                pos += len(r)
            return DecodedCanvas(text=joined, offsets=tuple(offsets), exact=True)

        # Exact but quadratic fallback.
        self.fallbacks += 1
        offsets = []
        prev = 0
        for i in range(len(ids)):
            cur = len(self.tokenizer.decode(ids[: i + 1], skip_special_tokens=False))
            offsets.append((prev, max(prev, cur)))
            prev = max(prev, cur)
        return DecodedCanvas(text=reference, offsets=tuple(offsets), exact=False)


class OffsetTokenizerAdapter:
    """Wraps a HuggingFace tokenizer for :class:`CanvasDecoder`."""

    def __init__(self, hf_tokenizer: Any) -> None:
        self._t = hf_tokenizer

    def decode(self, ids: Sequence[int], **kwargs: Any) -> str:
        return self._t.decode(list(ids), **kwargs)

    def convert_ids_to_tokens(self, ids: Sequence[int]) -> list[str]:
        return list(self._t.convert_ids_to_tokens(list(ids)))
