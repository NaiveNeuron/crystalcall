"""P-ReAct-style tool-priority decoding for the patched DiffusionGemma sampler.

Mounted into the pinned vLLM container as ``crystalcall_preact`` (see
``scripts/serve_diffusiongemma.sh``), next to ``crystalcall_trace``.  Stdlib +
torch only: no new runtime dependency enters the image.

This is the tensor form of ``src/crystalcall/preact.py``; that module is the
unit-tested reference and ``docs/preact.md`` states the semantics and every
deviation from the paper.  The mechanism has exactly two effects:

  span detection  first ``open_id`` in the argmax (predicted-clean) canvas,
                  first ``close_id`` strictly after it, half-open ``[lo, hi)``
                  covering both delimiters.
  priority accept in-span positions with ``token_entropy <= bound`` are added
                  to the ordinary entropy-bound accept mask (union only).
  ready signal    when the span has been byte-identical for ``stable_steps``
                  consecutive denoising steps AND every in-span position
                  clears ``bound``, latch once and let the sampler emit one
                  ``tool_call_ready`` side-channel event per (request, canvas).
  mode            ``span_accept`` | ``ready_signal`` | ``both``.

**Nothing here touches convergence.**  ``converged`` ends denoising and makes
DiffusionGemma commit the WHOLE argmax canvas, including out-of-span positions
that are still unaccepted and unstable; forcing it early would corrupt the
assistant turn and truncate the reasoning P-ReAct exists to let continue.
:func:`priority_step` returns an acceptance mask and writes only its own
buffers -- it never reads or writes ``is_encoder_phase`` and never produces a
convergence override.

EVERYTHING is off unless ``CRYSTALCALL_PREACT`` is truthy.  When off,
:func:`enabled` returns False, the sampler allocates no buffers, passes ``None``
for them, and not one of the ops below enters the compiled graph -- which is
what makes the CrystalCall / standard arm identical to the trace-only build out
of the same file.

The delimiter ids are NOT defaulted.  They arrive as
``CRYSTALCALL_PREACT_OPEN_ID`` / ``CRYSTALCALL_PREACT_CLOSE_ID``, exported by
the serve script from ``configs/preact.yml``; a missing or unparseable id
raises at configuration time.  A silently wrong id would make the span never
match, the mechanism inert, and the baseline vacuous while still looking like a
measurement.
"""

from __future__ import annotations

import os

try:  # the container always has torch; the host may not (docs / lint only)
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

__all__ = [
    "MODES",
    "PreactError",
    "PreactSettings",
    "enabled",
    "config",
    "priority_step",
    "emit_tool_call_ready",
    "self_test",
]

MODES = ("span_accept", "ready_signal", "both")

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("", "0", "false", "no", "off")


class PreactError(RuntimeError):
    """Fatal misconfiguration of the tool-priority mechanism."""


def _flag(name: str, default: str = "0") -> bool:
    raw = os.environ.get(name, default).strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    raise PreactError(f"{name}={raw!r} is not a boolean (use 0/1)")


ENABLED: bool = _flag("CRYSTALCALL_PREACT")


def enabled() -> bool:
    """True when the priority mechanism is on.  Check BEFORE any GPU work."""
    return ENABLED


def _req_int(name: str) -> int:
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        raise PreactError(
            f"{name} is unset or empty while CRYSTALCALL_PREACT=1. The tool-call "
            "delimiter ids are data from configs/preact.yml, never a default; "
            "scripts/serve_diffusiongemma.sh must export them."
        )
    try:
        return int(raw)
    except ValueError as exc:
        raise PreactError(f"{name}={raw!r} is not an integer") from exc


class PreactSettings:
    """Immutable resolved settings; built once, at the first :func:`config`."""

    __slots__ = ("mode", "open_id", "close_id", "bound", "stable_steps")

    def __init__(self) -> None:
        mode = (os.environ.get("CRYSTALCALL_PREACT_MODE") or "both").strip()
        if mode not in MODES:
            raise PreactError(f"CRYSTALCALL_PREACT_MODE={mode!r} not in {MODES}")
        self.mode = mode
        self.open_id = _req_int("CRYSTALCALL_PREACT_OPEN_ID")
        self.close_id = _req_int("CRYSTALCALL_PREACT_CLOSE_ID")
        if self.open_id == self.close_id:
            raise PreactError(
                f"CRYSTALCALL_PREACT_OPEN_ID == CLOSE_ID == {self.open_id}: "
                "the tool-call delimiters must differ"
            )
        raw_bound = (os.environ.get("CRYSTALCALL_PREACT_BOUND") or "0.6").strip()
        try:
            self.bound = float(raw_bound)
        except ValueError as exc:
            raise PreactError(f"CRYSTALCALL_PREACT_BOUND={raw_bound!r} is not a float") from exc
        if self.bound < 0.0:
            raise PreactError(f"CRYSTALCALL_PREACT_BOUND={self.bound} must be >= 0")
        raw_stable = (os.environ.get("CRYSTALCALL_PREACT_STABLE") or "1").strip()
        try:
            self.stable_steps = int(raw_stable)
        except ValueError as exc:
            raise PreactError(f"CRYSTALCALL_PREACT_STABLE={raw_stable!r} is not an int") from exc
        if self.stable_steps < 1:
            raise PreactError(f"CRYSTALCALL_PREACT_STABLE={self.stable_steps} must be >= 1")

    @property
    def span_accept(self) -> bool:
        return self.mode in ("span_accept", "both")

    @property
    def ready_signal(self) -> bool:
        return self.mode in ("ready_signal", "both")

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "open_token_id": self.open_id,
            "close_token_id": self.close_id,
            "span_entropy_bound": self.bound,
            "require_stable_steps": self.stable_steps,
        }


_SETTINGS: PreactSettings | None = None


def config() -> PreactSettings:
    """Resolved settings.  Raises unless :func:`enabled`."""
    global _SETTINGS
    if not ENABLED:
        raise PreactError("crystalcall_preact.config() called with CRYSTALCALL_PREACT off")
    if _SETTINGS is None:
        _SETTINGS = PreactSettings()
    return _SETTINGS


# ---------------------------------------------------------------------------
# The device-side mechanism
# ---------------------------------------------------------------------------


def priority_step(
    argmax_tokens,  # [nd, CL] int64  predicted-clean canvas for this step
    token_entropy,  # [nd, CL] float  per-position entropy, already computed
    eb_mask,  # [nd, CL] bool   ordinary entropy-bound accept mask
    is_denoise,  # [nd] bool
    decode_slots,  # [nd] int64
    span_hist,  # [max_reqs, S, CL] int64  persistent ring of argmax canvases
    span_hist_len,  # [max_reqs]       int32   persistent
    fired,  # [max_reqs]       bool    persistent latch (one event per canvas)
    span_out,  # [max_reqs, 3]    int32   output: lo, hi, fired-this-step
    open_id: int,
    close_id: int,
    bound: float,
    stable_steps: int,
    span_accept: bool,
    ready_signal: bool,
    CL: int,
):
    """One priority step, entirely inside the compiled region.

    Returns ``accept_mask`` -- ``eb_mask`` unioned with the in-span priority
    mask, or ``eb_mask`` itself when the mode excludes acceptance priority.
    The readiness latch is written to ``span_out[:, 2]``; the caller ships that
    to the emitter after the step.  No convergence value is computed, returned
    or written.

    Reads only tensors the compiled step already holds: no logits copy, no
    probability transfer, no host synchronisation.  Span detection is a pair of
    ``==`` comparisons plus a masked ``amin`` over the position index -- the
    vectorised form of "first opener, first closer after it".  The stability
    ring is written with the same unrolled ``for i in range(S)`` pattern the
    upstream sampler already uses for ``accepted_canvas_history``.
    """
    nd = argmax_tokens.shape[0]
    device = argmax_tokens.device
    S = stable_steps

    pos = torch.arange(CL, device=device, dtype=torch.int64).unsqueeze(0).expand(nd, CL)
    sentinel = torch.full_like(pos, CL)

    # first opener; CL when absent
    lo = torch.where(argmax_tokens == open_id, pos, sentinel).amin(dim=-1)  # [nd]
    # first closer strictly after the opener; CL when absent
    close_after = (argmax_tokens == close_id) & (pos > lo.unsqueeze(1))
    hi_tok = torch.where(close_after, pos, sentinel).amin(dim=-1)  # [nd]
    valid = (lo < CL) & (hi_tok < CL)
    hi = (hi_tok + 1).clamp(max=CL)  # half-open, covers the closer

    in_span = valid.unsqueeze(1) & (pos >= lo.unsqueeze(1)) & (pos < hi.unsqueeze(1))
    span_ok = token_entropy <= bound

    # -- priority acceptance (union only) -----------------------------------
    accept = (eb_mask | (in_span & span_ok)) if span_accept else eb_mask

    # -- span stability: the last S argmax canvases agree on this window ----
    hist_len = span_hist_len[decode_slots]
    write_pos = hist_len % S
    for i in range(S):
        write_here = ((write_pos == i) & is_denoise).unsqueeze(1)
        span_hist[decode_slots, i] = torch.where(
            write_here, argmax_tokens, span_hist[decode_slots, i]
        )
    new_hist_len = torch.where(is_denoise, hist_len + 1, torch.zeros_like(hist_len))
    span_hist_len[decode_slots] = new_hist_len

    stable = valid & (new_hist_len >= S)
    for i in range(S):
        stable = stable & ((span_hist[decode_slots, i] == argmax_tokens) | ~in_span).all(dim=-1)

    # -- readiness (signal only: nothing about decoding changes) ------------
    fully_accepted = ((~in_span) | span_ok).all(dim=-1)
    ready = (valid & stable & fully_accepted & is_denoise) if ready_signal else None

    prev_fired = fired[decode_slots]
    now = ready if ready is not None else torch.zeros_like(valid)
    fire_now = now & ~prev_fired
    # Latch for the lifetime of this canvas; a commit step starts a new canvas
    # and clears it, so at most one event per (request_id, canvas_id).
    fired[decode_slots] = torch.where(is_denoise, prev_fired | now, torch.zeros_like(prev_fired))

    neg = torch.full_like(lo, -1)
    span_out[decode_slots, 0] = torch.where(valid, lo, neg).to(span_out.dtype)
    span_out[decode_slots, 1] = torch.where(valid, hi, neg).to(span_out.dtype)
    span_out[decode_slots, 2] = fire_now.to(span_out.dtype)

    return accept


# ---------------------------------------------------------------------------
# Side-channel emission
# ---------------------------------------------------------------------------


def emit_tool_call_ready(request_id: str, canvas_id: int, step: int, lo: int, hi: int) -> None:
    """Forward one ``tool_call_ready`` event to the CrystalCall emitter.

    Routed through ``crystalcall_trace`` so the event lands in the SAME NDJSON
    stream and schema family (``crystalcall-revision/2``) as ``revision`` and
    ``canvas_done``.  A no-op when the emitter is absent or tracing is off.
    """
    try:
        import crystalcall_trace
    except ImportError:
        return
    fn = getattr(crystalcall_trace, "tool_call_ready", None)
    if fn is None:
        return
    fn(request_id, canvas_id, step, lo, hi)


# ---------------------------------------------------------------------------
# Differential self-test (run inside the pinned image; CPU only, no GPU)
# ---------------------------------------------------------------------------


def self_test(trials: int = 600, cl: int = 24, seed: int = 12345) -> dict:
    """Check :func:`priority_step` against the pure reference implementation.

    The reference is ``crystalcall.preact`` from the repo (mount it on
    PYTHONPATH).  Random canvases and entropies are pushed through both, over
    every mode and two stability depths, and the accept mask, the detected span
    and the latched readiness flag must agree exactly.  This is the only way to
    exercise the tensor form without a GPU: the container has torch on CPU, the
    host does not.
    """
    import random

    from crystalcall.preact import (
        PreactConfig,
        ReadyLatch,
        find_tool_call_span,
        priority_accept_mask,
    )

    rng = random.Random(seed)
    open_id, close_id = 48, 49
    bound = 0.6
    checked = {"mask": 0, "ready": 0, "span": 0, "fires": 0}
    per_case = max(1, trials // (len(MODES) * 2))
    for mode in MODES:
        for stable_steps in (1, 2):
            cfg = PreactConfig(
                enabled=True,
                mode=mode,
                open_token_id=open_id,
                close_token_id=close_id,
                span_entropy_bound=bound,
                require_stable_steps=stable_steps,
            )
            span_hist = torch.zeros(1, stable_steps, cl, dtype=torch.int64)
            span_hist_len = torch.zeros(1, dtype=torch.int32)
            fired = torch.zeros(1, dtype=torch.bool)
            span_out = torch.full((1, 3), -1, dtype=torch.int32)
            slots = torch.zeros(1, dtype=torch.int64)
            latch = ReadyLatch()
            history: list[list[int]] = []
            for _ in range(per_case):
                # A commit step starts a new canvas: the latch and the
                # stability ring must both reset, so the mechanism can fire
                # again for the next canvas of the same request.
                is_denoise = rng.random() >= 0.12
                canvas = [rng.randrange(100, 120) for _ in range(cl)]
                if rng.random() < 0.85:
                    a = rng.randrange(0, cl - 3)
                    b = rng.randrange(a + 1, cl)
                    canvas[a] = open_id
                    canvas[b] = close_id
                if rng.random() < 0.35 and history:
                    canvas = list(history[-1])
                ent = [rng.choice([0.01, 0.3, 0.9, 2.0]) for _ in range(cl)]
                base = [e <= 0.05 for e in ent]

                span = find_tool_call_span(canvas, cfg)
                want_mask = priority_accept_mask(base, ent, span, cfg)
                if is_denoise:
                    history.append(canvas)
                    want_fire = latch.observe(span, history, ent, cfg)
                else:
                    history.clear()
                    latch.reset()
                    want_fire = False

                got_mask = priority_step(
                    torch.tensor([canvas], dtype=torch.int64),
                    torch.tensor([ent], dtype=torch.float32),
                    torch.tensor([base], dtype=torch.bool),
                    torch.tensor([is_denoise], dtype=torch.bool),
                    slots,
                    span_hist,
                    span_hist_len,
                    fired,
                    span_out,
                    open_id,
                    close_id,
                    bound,
                    stable_steps,
                    cfg.accepts_span,
                    cfg.signals_ready,
                    cl,
                )
                if got_mask[0].tolist() != want_mask:
                    raise AssertionError(f"mask mismatch mode={mode} span={span}")
                checked["mask"] += 1

                got_span = tuple(int(x) for x in span_out[0, :2].tolist())
                want_span = tuple(span) if span is not None else (-1, -1)
                if got_span != want_span:
                    raise AssertionError(f"span mismatch {got_span} != {want_span}")
                checked["span"] += 1

                got_fire = bool(span_out[0, 2].item())
                if got_fire != want_fire:
                    raise AssertionError(
                        f"ready mismatch mode={mode} stable={stable_steps} "
                        f"{got_fire} != {want_fire} span={span}"
                    )
                checked["ready"] += 1
                checked["fires"] += int(got_fire)
    if checked["fires"] == 0:
        raise AssertionError("self-test never fired tool_call_ready: vacuous")
    return {"ok": True, **checked, "modes": list(MODES)}


def main(argv: list[str]) -> int:
    if "--self-test" not in argv:
        print(__doc__)
        print("usage: python crystalcall_preact.py --self-test")
        return 0
    if torch is None:
        print("preact self-test: torch unavailable", flush=True)
        return 2
    import json

    print(json.dumps(self_test(), sort_keys=True))
    print("PREACT SELF-TEST PASS")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main(sys.argv[1:]))
