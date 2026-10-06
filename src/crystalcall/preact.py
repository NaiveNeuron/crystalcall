"""P-ReAct-style tool-priority diffusion decoding (mandatory baseline arm D).

This module is the *reference* implementation of the mechanism: plain Python
over sequences, no torch, no GPU, fully unit-testable.  The in-container tensor
form lives in ``patches/preact_priority.py`` and is called from the patched
DiffusionGemma sampler; the two implement the same rules and
``tests/test_preact.py`` pins the semantics here.

The mechanism has exactly two effects, and **neither touches convergence**:

* **span detection** -- the first ``open_token_id`` in the predicted-clean
  canvas (the argmax canvas), and the first ``close_token_id`` strictly after
  it.  The span is the half-open range ``[lo, hi)`` covering *both*
  delimiters.  No opener, or no closer after the opener, means the mechanism
  is inert for that step.
* **priority acceptance** (``span_accept``) -- inside the span, a position is
  additionally accepted when its per-token entropy is
  ``<= span_entropy_bound``, a deliberately looser bound than the sampler's
  ordinary *cumulative* entropy bound.  Outside the span acceptance is exactly
  the ordinary rule.  Union only: a position the ordinary rule accepted is
  never un-accepted.  This is the token-selection-priority intervention: the
  tool-call span settles early while the rest of the canvas keeps denoising
  normally.
* **readiness signalling** (``ready_signal``) -- when the span has been
  byte-identical for ``require_stable_steps`` consecutive denoising steps AND
  every in-span position clears the in-span bound, the sampler emits ONE
  ``tool_call_ready`` side-channel event for that request.  Nothing else
  changes: no flag is flipped, no step is skipped, no canvas is committed
  early.  The executor may launch the parsed call on that event while
  generation continues.
* **mode** -- ``"span_accept"``, ``"ready_signal"`` or ``"both"`` (the default,
  and what the arm runs).  The decomposition exists so the ablation table can
  attribute the effect.

There is deliberately **no** convergence override anywhere in this module or in
the patch.  ``converged`` ends denoising, at which point DiffusionGemma commits
the whole argmax canvas -- including out-of-span positions that are still
unaccepted and unstable.  Forcing it early would write garbage into the
authoritative assistant turn and truncate exactly the reasoning P-ReAct is
designed to let continue.  Scheduling may change; final action semantics may
not.  ``tests/test_preact.py`` mechanises that guarantee.

Everything is off unless ``CRYSTALCALL_PREACT`` is truthy.  With it off every
function here is the identity on its inputs, which is the guarantee that the
CrystalCall and standard arms run untouched decoding out of the same file.
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "MODES",
    "DEFAULT_CONFIG_PATH",
    "PreactConfigError",
    "PreactConfig",
    "ReadyLatch",
    "load_preact_config_file",
    "find_tool_call_span",
    "priority_accept_mask",
    "span_is_accepted",
    "span_is_stable",
    "should_signal_ready",
]

MODES: tuple[str, ...] = ("span_accept", "ready_signal", "both")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs/preact.yml"

_TRUTHY = frozenset({"1", "true", "yes", "on"})
_FALSY = frozenset({"", "0", "false", "no", "off"})


class PreactConfigError(RuntimeError):
    """The priority mechanism is misconfigured.

    Always fatal on purpose: a wrong (or defaulted) delimiter id would make the
    span never match, the mechanism silently inert, and the whole baseline
    vacuous while still *looking* like a real measurement.
    """


def _flag(raw: str | None, *, name: str, default: bool = False) -> bool:
    if raw is None:
        return default
    v = raw.strip().lower()
    if v in _TRUTHY:
        return True
    if v in _FALSY:
        return False
    raise PreactConfigError(f"{name}={raw!r} is not a boolean (use 0/1)")


def load_preact_config_file(path: str | os.PathLike[str] | None = None) -> Mapping[str, Any]:
    """Parse ``configs/preact.yml``.  Missing or unparseable is fatal."""
    p = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not p.is_file():
        raise PreactConfigError(
            f"missing {p}: the P-ReAct-style tool-call delimiter ids are data, "
            "not a default -- run `uv run scripts/resolve_gemma_tokens.py --write`"
        )
    import yaml  # local import: keeps the module cheap for stdlib-only callers

    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, Mapping):
        raise PreactConfigError(f"{p} does not parse to a mapping")
    return doc


def _tokens_from_config(doc: Mapping[str, Any], path: Path) -> Mapping[str, Any]:
    tokens = doc.get("tokens")
    if not isinstance(tokens, Mapping):
        raise PreactConfigError(f"{path} has no `tokens:` mapping")
    return tokens


def _require_id(tokens: Mapping[str, Any], key: str, where: str) -> int:
    val = tokens.get(key)
    if val is None:
        raise PreactConfigError(f"{where} does not define `{key}`")
    try:
        return int(val)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise PreactConfigError(f"{where}: `{key}` is not an integer: {val!r}") from exc


@dataclass(frozen=True, slots=True)
class PreactConfig:
    """Resolved settings for the tool-priority mechanism."""

    enabled: bool
    mode: str
    open_token_id: int
    close_token_id: int
    span_entropy_bound: float
    require_stable_steps: int

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise PreactConfigError(f"CRYSTALCALL_PREACT_MODE={self.mode!r} not in {MODES}")
        if self.require_stable_steps < 1:
            raise PreactConfigError(
                f"require_stable_steps must be >= 1, got {self.require_stable_steps}"
            )
        if self.span_entropy_bound < 0.0:
            raise PreactConfigError(
                f"span_entropy_bound must be >= 0, got {self.span_entropy_bound}"
            )
        if self.open_token_id == self.close_token_id:
            raise PreactConfigError(
                f"open_token_id and close_token_id must differ (both {self.open_token_id})"
            )

    # -- mode decomposition -------------------------------------------------
    @property
    def accepts_span(self) -> bool:
        """True when in-span priority acceptance is active."""
        return self.enabled and self.mode in ("span_accept", "both")

    @property
    def signals_ready(self) -> bool:
        """True when the ``tool_call_ready`` side-channel event is active."""
        return self.enabled and self.mode in ("ready_signal", "both")

    # -- construction -------------------------------------------------------
    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
        *,
        tokens: Mapping[str, Any] | str | os.PathLike[str] | None = None,
    ) -> "PreactConfig":
        """Build from the environment, with defaults from ``configs/preact.yml``.

        ``tokens`` may be an explicit mapping (tests), a path to an alternative
        config file, or ``None`` for the repo default.  Precedence for the
        delimiter ids is: explicit mapping > ``CRYSTALCALL_PREACT_{OPEN,CLOSE}_ID``
        > the config file.  There is deliberately no hardcoded fallback: a
        missing id raises rather than defaulting to a plausible number.
        """
        e = os.environ if env is None else env

        doc: Mapping[str, Any] | None = None
        path = DEFAULT_CONFIG_PATH
        if isinstance(tokens, Mapping):
            token_map: Mapping[str, Any] = tokens
            where = "the supplied tokens mapping"
            open_id = _require_id(token_map, "open_token_id", where)
            close_id = _require_id(token_map, "close_token_id", where)
        else:
            if tokens is not None:
                path = Path(tokens)
            doc = load_preact_config_file(path)
            token_map = _tokens_from_config(doc, path)
            where = str(path)
            open_env = (e.get("CRYSTALCALL_PREACT_OPEN_ID") or "").strip()
            close_env = (e.get("CRYSTALCALL_PREACT_CLOSE_ID") or "").strip()
            open_id = int(open_env) if open_env else _require_id(token_map, "open_token_id", where)
            close_id = (
                int(close_env) if close_env else _require_id(token_map, "close_token_id", where)
            )

        defaults: Mapping[str, Any] = {}
        if doc is not None and isinstance(doc.get("defaults"), Mapping):
            defaults = doc["defaults"]

        mode = (e.get("CRYSTALCALL_PREACT_MODE") or "").strip() or str(
            defaults.get("mode", "both")
        )
        bound_raw = (e.get("CRYSTALCALL_PREACT_BOUND") or "").strip()
        bound = float(bound_raw) if bound_raw else float(defaults.get("span_entropy_bound", 0.6))
        stable_raw = (e.get("CRYSTALCALL_PREACT_STABLE") or "").strip()
        stable = int(stable_raw) if stable_raw else int(defaults.get("require_stable_steps", 1))

        return cls(
            enabled=_flag(
                e.get("CRYSTALCALL_PREACT"),
                name="CRYSTALCALL_PREACT",
                default=bool(defaults.get("enabled", False)),
            ),
            mode=mode,
            open_token_id=open_id,
            close_token_id=close_id,
            span_entropy_bound=bound,
            require_stable_steps=stable,
        )


# ---------------------------------------------------------------------------
# The mechanism
# ---------------------------------------------------------------------------


def find_tool_call_span(token_ids: Sequence[int], cfg: PreactConfig) -> tuple[int, int] | None:
    """First tool-call span in a predicted-clean canvas, or ``None``.

    Returns the half-open ``[lo, hi)`` covering the opening delimiter through
    the closing delimiter inclusive.  Inert (``None``) when the mechanism is
    disabled, when there is no opener, or when no closer follows the opener --
    a closer that appears *before* any opener is ignored, as is a second
    opener.
    """
    if not cfg.enabled:
        return None
    lo = -1
    for i, tok in enumerate(token_ids):
        if tok == cfg.open_token_id:
            lo = i
            break
    if lo < 0:
        return None
    for j in range(lo + 1, len(token_ids)):
        if token_ids[j] == cfg.close_token_id:
            return (lo, j + 1)
    return None


def priority_accept_mask(
    base_mask: Sequence[bool],
    token_entropy: Sequence[float],
    span: tuple[int, int] | None,
    cfg: PreactConfig,
) -> list[bool]:
    """Ordinary acceptance mask unioned with in-span priority acceptance.

    Never un-accepts: the result is ``base_mask`` OR the priority mask, and the
    priority mask is empty outside ``span``.
    """
    out = [bool(x) for x in base_mask]
    if span is None or not cfg.accepts_span:
        return out
    lo, hi = span
    for i in range(max(lo, 0), min(hi, len(out), len(token_entropy))):
        if token_entropy[i] <= cfg.span_entropy_bound:
            out[i] = True
    return out


def span_is_accepted(
    token_entropy: Sequence[float],
    span: tuple[int, int] | None,
    cfg: PreactConfig,
) -> bool:
    """True when every in-span position clears the in-span entropy bound."""
    if span is None or not cfg.enabled:
        return False
    lo, hi = span
    if lo < 0 or hi > len(token_entropy):
        return False
    return all(token_entropy[i] <= cfg.span_entropy_bound for i in range(lo, hi))


def span_is_stable(
    history: Sequence[Sequence[int]],
    span: tuple[int, int] | None,
    cfg: PreactConfig,
) -> bool:
    """True when the span content is byte-identical over the last N canvases.

    ``history`` is oldest-first and its last entry is the current canvas; only
    the last ``require_stable_steps`` entries are examined, and there must be
    at least that many.
    """
    if span is None or not cfg.enabled:
        return False
    n = cfg.require_stable_steps
    if len(history) < n:
        return False
    lo, hi = span
    window = history[-n:]
    ref = tuple(window[-1][lo:hi])
    if len(ref) != hi - lo:
        return False
    return all(tuple(canvas[lo:hi]) == ref for canvas in window)


def should_signal_ready(
    span: tuple[int, int] | None,
    history: Sequence[Sequence[int]],
    token_entropy: Sequence[float],
    cfg: PreactConfig,
) -> bool:
    """True when the tool call is ready to launch: stable AND fully accepted.

    Readiness is a *signal*, not a decision: it changes nothing about decoding.
    Convergence is deliberately not an input and not an output -- the request
    ends when the ordinary rule says it does, exactly as in the standard arm.
    """
    if span is None or not cfg.signals_ready:
        return False
    return span_is_stable(history, span, cfg) and span_is_accepted(token_entropy, span, cfg)


class ReadyLatch:
    """One ``tool_call_ready`` event per request (per canvas).

    :meth:`observe` returns True exactly once -- on the first step where
    :func:`should_signal_ready` holds -- and False forever after, until
    :meth:`reset` starts a new canvas.  The sampler keeps the same latch as a
    per-slot bool tensor.
    """

    __slots__ = ("_fired",)

    def __init__(self) -> None:
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def reset(self) -> None:
        self._fired = False

    def observe(
        self,
        span: tuple[int, int] | None,
        history: Sequence[Sequence[int]],
        token_entropy: Sequence[float],
        cfg: PreactConfig,
    ) -> bool:
        if self._fired:
            return False
        if not should_signal_ready(span, history, token_entropy, cfg):
            return False
        self._fired = True
        return True
