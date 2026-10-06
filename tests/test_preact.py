"""P-ReAct-style tool-priority decoding: the mechanism and its guarantees.

Everything here is GPU-free.  The tensor form in ``patches/preact_priority.py``
is checked differentially against this reference inside the pinned image with
``python3 patches/preact_priority.py --self-test`` (torch on CPU); see
``docs/preact.md``.

Three of these tests are load-bearing for the validity of the whole baseline,
not just for the code:

* the delimiter ids are re-resolved from the live checkpoint tokenizer, so a
  checkpoint swap fails the suite instead of silently producing a span that
  never matches (a vacuous arm that still looks like a measurement);
* nothing in the mechanism can override convergence, asserted against the
  generated patch and the generated sampler, not merely documented;
* the rendered container argv carries every ``CRYSTALCALL_PREACT*`` variable,
  because the serve script does not inherit host env into the container.
"""

from __future__ import annotations

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from crystalcall import preact
from crystalcall.preact import (
    MODES,
    PreactConfig,
    PreactConfigError,
    ReadyLatch,
    find_tool_call_span,
    priority_accept_mask,
    should_signal_ready,
    span_is_accepted,
    span_is_stable,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/preact.yml"
PREACT_PATCH = ROOT / "patches/vllm-preact.patch"
BUILT_SAMPLER = ROOT / "patches/build/diffusion_gemma.py"
SERVE = ROOT / "scripts/serve_diffusiongemma.sh"

OPEN_ID = 48
CLOSE_ID = 49


def cfg(**over) -> PreactConfig:
    base = dict(
        enabled=True,
        mode="both",
        open_token_id=OPEN_ID,
        close_token_id=CLOSE_ID,
        span_entropy_bound=0.6,
        require_stable_steps=1,
    )
    base.update(over)
    return PreactConfig(**base)


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


def test_from_env_is_off_by_default():
    c = PreactConfig.from_env({})
    assert c.enabled is False
    assert c.accepts_span is False
    assert c.signals_ready is False


def test_from_env_enables_and_reads_mode_and_bound():
    c = PreactConfig.from_env(
        {
            "CRYSTALCALL_PREACT": "1",
            "CRYSTALCALL_PREACT_MODE": "span_accept",
            "CRYSTALCALL_PREACT_BOUND": "0.25",
            "CRYSTALCALL_PREACT_STABLE": "3",
        }
    )
    assert c.enabled is True
    assert c.mode == "span_accept"
    assert c.span_entropy_bound == 0.25
    assert c.require_stable_steps == 3
    assert c.accepts_span is True
    assert c.signals_ready is False


def test_token_ids_come_from_the_config_file():
    doc = yaml.safe_load(CONFIG.read_text())
    c = PreactConfig.from_env({"CRYSTALCALL_PREACT": "1"})
    assert c.open_token_id == doc["tokens"]["open_token_id"]
    assert c.close_token_id == doc["tokens"]["close_token_id"]
    assert (c.open_token_id, c.close_token_id) == (OPEN_ID, CLOSE_ID)


def test_env_ids_override_the_config_file():
    c = PreactConfig.from_env(
        {
            "CRYSTALCALL_PREACT": "1",
            "CRYSTALCALL_PREACT_OPEN_ID": "7",
            "CRYSTALCALL_PREACT_CLOSE_ID": "8",
        }
    )
    assert (c.open_token_id, c.close_token_id) == (7, 8)


def test_missing_config_is_fatal_not_defaulted(tmp_path):
    with pytest.raises(PreactConfigError) as exc:
        PreactConfig.from_env({"CRYSTALCALL_PREACT": "1"}, tokens=tmp_path / "nope.yml")
    assert "not a default" in str(exc.value)


def test_bad_mode_and_bad_stability_are_rejected():
    with pytest.raises(PreactConfigError):
        cfg(mode="early_commit")
    with pytest.raises(PreactConfigError):
        cfg(require_stable_steps=0)
    with pytest.raises(PreactConfigError):
        cfg(close_token_id=OPEN_ID)


def test_mode_vocabulary_is_exactly_the_three_documented_modes():
    assert MODES == ("span_accept", "ready_signal", "both")


def test_config_ids_still_match_the_live_checkpoint_tokenizer():
    """A checkpoint swap must fail here, not silently make the arm vacuous."""
    script = ROOT / "scripts/resolve_gemma_tokens.py"
    proc = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, cwd=ROOT
    )
    if proc.returncode == 2:
        pytest.skip("checkpoint tokenizer not present on this machine")
    assert proc.returncode == 0, proc.stderr
    resolved = json.loads(proc.stdout)
    doc = yaml.safe_load(CONFIG.read_text())["tokens"]
    assert resolved["open_token_id"] == doc["open_token_id"]
    assert resolved["close_token_id"] == doc["close_token_id"]
    assert resolved["tokenizer_sha256"] == doc["tokenizer_sha256"]


# ---------------------------------------------------------------------------
# span detection
# ---------------------------------------------------------------------------


def test_find_span_covers_both_delimiters():
    c = cfg()
    canvas = [1, 2, OPEN_ID, 5, 6, CLOSE_ID, 7]
    assert find_tool_call_span(canvas, c) == (2, 6)


def test_find_span_picks_first_opener_and_first_closer_after_it():
    c = cfg()
    canvas = [OPEN_ID, 3, CLOSE_ID, OPEN_ID, 4, CLOSE_ID]
    assert find_tool_call_span(canvas, c) == (0, 3)


def test_find_span_ignores_a_closer_before_any_opener():
    c = cfg()
    canvas = [CLOSE_ID, 1, OPEN_ID, 2, CLOSE_ID]
    assert find_tool_call_span(canvas, c) == (2, 5)


def test_find_span_none_without_opener_or_closer():
    c = cfg()
    assert find_tool_call_span([1, 2, 3], c) is None
    assert find_tool_call_span([1, OPEN_ID, 2, 3], c) is None
    assert find_tool_call_span([CLOSE_ID, 1, 2], c) is None
    # a closer at the opener's own position does not close it
    assert find_tool_call_span([OPEN_ID], c) is None


def test_find_span_inert_when_disabled():
    canvas = [OPEN_ID, 1, CLOSE_ID]
    assert find_tool_call_span(canvas, cfg(enabled=False)) is None


# ---------------------------------------------------------------------------
# priority acceptance
# ---------------------------------------------------------------------------


def test_priority_mask_accepts_low_entropy_in_span_positions():
    c = cfg()
    canvas = [9, OPEN_ID, 3, 4, CLOSE_ID, 9]
    span = find_tool_call_span(canvas, c)
    ent = [2.0, 0.1, 0.2, 0.3, 0.1, 2.0]
    base = [False] * 6
    got = priority_accept_mask(base, ent, span, c)
    assert got == [False, True, True, True, True, False]


def test_priority_mask_never_unaccepts_and_leaves_outside_untouched():
    c = cfg()
    span = (1, 5)
    ent = [0.01, 0.9, 0.1, 5.0, 0.1, 0.01]
    base = [True, True, False, True, False, True]
    got = priority_accept_mask(base, ent, span, c)
    # outside the span: byte-identical to base, even where entropy is tiny
    assert got[0] == base[0] and got[5] == base[5]
    # inside: union only
    assert all(g or not b for g, b in zip(got, base, strict=True))
    assert got == [True, True, True, True, True, True]


def test_priority_mask_leaves_high_entropy_in_span_positions_alone():
    c = cfg(span_entropy_bound=0.2)
    span = (0, 3)
    ent = [0.1, 0.9, 0.1]
    got = priority_accept_mask([False, False, False], ent, span, c)
    assert got == [True, False, True]


def test_priority_mask_is_identity_without_a_span_or_when_disabled():
    base = [True, False, True, False]
    ent = [0.0, 0.0, 0.0, 0.0]
    assert priority_accept_mask(base, ent, None, cfg()) == base
    assert priority_accept_mask(base, ent, (0, 4), cfg(enabled=False)) == base


# ---------------------------------------------------------------------------
# stability
# ---------------------------------------------------------------------------


def test_span_is_stable_requires_n_identical_consecutive_spans():
    c = cfg(require_stable_steps=3)
    span = (1, 4)
    same = [0, 7, 8, 9, 0]
    assert span_is_stable([same], span, c) is False
    assert span_is_stable([same, same], span, c) is False
    assert span_is_stable([same, same, same], span, c) is True
    # only the last N matter; older canvases may differ freely
    other = [5, 1, 1, 1, 5]
    assert span_is_stable([other, same, same, same], span, c) is True


def test_span_is_stable_false_when_the_span_content_changes():
    c = cfg(require_stable_steps=2)
    span = (1, 4)
    a = [0, 7, 8, 9, 0]
    b = [0, 7, 8, 1, 0]
    assert span_is_stable([a, b], span, c) is False
    # a change strictly outside the span does not break stability
    d = [4, 7, 8, 9, 4]
    assert span_is_stable([a, d], span, c) is True


def test_span_is_stable_inert_when_disabled_or_spanless():
    same = [0, 7, 8, 9, 0]
    assert span_is_stable([same], (1, 4), cfg(enabled=False)) is False
    assert span_is_stable([same], None, cfg()) is False


# ---------------------------------------------------------------------------
# readiness
# ---------------------------------------------------------------------------


def test_ready_requires_stable_and_fully_accepted():
    c = cfg(require_stable_steps=2)
    span = (1, 4)
    canvas = [0, OPEN_ID, 8, CLOSE_ID, 0]
    good_ent = [9.0, 0.1, 0.1, 0.1, 9.0]
    assert should_signal_ready(span, [canvas, canvas], good_ent, c) is True
    # not stable yet
    assert should_signal_ready(span, [canvas], good_ent, c) is False
    # stable but one in-span position is over the bound
    bad_ent = [9.0, 0.1, 0.9, 0.1, 9.0]
    assert should_signal_ready(span, [canvas, canvas], bad_ent, c) is False
    assert span_is_accepted(bad_ent, span, c) is False


def test_ready_is_false_without_a_span_and_when_disabled():
    canvas = [0, OPEN_ID, CLOSE_ID]
    ent = [0.0, 0.0, 0.0]
    assert should_signal_ready(None, [canvas], ent, cfg()) is False
    assert should_signal_ready((1, 3), [canvas], ent, cfg(enabled=False)) is False


def test_ready_latch_fires_exactly_once_per_canvas():
    c = cfg()
    canvas = [OPEN_ID, 8, CLOSE_ID]
    ent = [0.1, 0.1, 0.1]
    latch = ReadyLatch()
    span = find_tool_call_span(canvas, c)
    assert latch.observe(span, [canvas], ent, c) is True
    assert latch.fired is True
    for _ in range(5):
        assert latch.observe(span, [canvas, canvas], ent, c) is False
    # a commit starts a new canvas: the latch re-arms
    latch.reset()
    assert latch.observe(span, [canvas], ent, c) is True


def test_ready_latch_never_fires_when_disabled():
    latch = ReadyLatch()
    canvas = [OPEN_ID, 8, CLOSE_ID]
    for _ in range(4):
        assert latch.observe((0, 3), [canvas], [0.0] * 3, cfg(enabled=False)) is False
    assert latch.fired is False


# ---------------------------------------------------------------------------
# inertness and mode decomposition
# ---------------------------------------------------------------------------


def test_disabled_is_completely_inert():
    """The CrystalCall / standard arms must see byte-identical inputs back."""
    c = cfg(enabled=False)
    canvas = [OPEN_ID, 3, CLOSE_ID, 9]
    base = [True, False, False, True]
    ent = [0.0, 0.0, 0.0, 0.0]
    assert find_tool_call_span(canvas, c) is None
    assert priority_accept_mask(base, ent, (0, 3), c) == base
    assert priority_accept_mask(base, ent, None, c) == base
    assert span_is_stable([canvas, canvas], (0, 3), c) is False
    assert span_is_accepted(ent, (0, 3), c) is False
    assert should_signal_ready((0, 3), [canvas, canvas], ent, c) is False


def test_mode_span_accept_changes_the_mask_but_not_readiness():
    c = cfg(mode="span_accept")
    canvas = [OPEN_ID, 3, CLOSE_ID]
    span = find_tool_call_span(canvas, c)
    ent = [0.1, 0.1, 0.1]
    base = [False, False, False]
    assert priority_accept_mask(base, ent, span, c) == [True, True, True]
    assert should_signal_ready(span, [canvas], ent, c) is False


def test_mode_ready_signal_changes_readiness_but_not_the_mask():
    c = cfg(mode="ready_signal")
    canvas = [OPEN_ID, 3, CLOSE_ID]
    span = find_tool_call_span(canvas, c)
    ent = [0.1, 0.1, 0.1]
    base = [False, False, False]
    assert priority_accept_mask(base, ent, span, c) == base
    assert should_signal_ready(span, [canvas], ent, c) is True


def test_mode_both_does_both():
    c = cfg(mode="both")
    canvas = [OPEN_ID, 3, CLOSE_ID]
    span = find_tool_call_span(canvas, c)
    ent = [0.1, 0.1, 0.1]
    assert priority_accept_mask([False] * 3, ent, span, c) == [True, True, True]
    assert should_signal_ready(span, [canvas], ent, c) is True


# ---------------------------------------------------------------------------
# the mechanism cannot override convergence  (mechanised, not documented)
# ---------------------------------------------------------------------------

_CONVERGENCE_WORDS = re.compile(r"converged|convergence|is_encoder_phase|commit", re.I)


def test_no_public_function_takes_or_returns_a_convergence_flag():
    for name in preact.__all__:
        obj = getattr(preact, name)
        if not callable(obj) or isinstance(obj, type):
            continue
        sig = inspect.signature(obj)
        for param in sig.parameters:
            assert not _CONVERGENCE_WORDS.search(param), f"{name}({param})"


def test_preact_patch_never_writes_the_convergence_tensor():
    added = [
        ln[1:]
        for ln in PREACT_PATCH.read_text().splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    assert added, "empty patch"
    code = [ln for ln in added if not ln.lstrip().startswith("#")]
    offenders = [
        ln
        for ln in code
        if "is_encoder_phase" in ln
        or re.search(r"\bconverged\b\s*=", ln)
        or re.search(r"\bconverged\b\s*\|", ln)
    ]
    assert offenders == [], offenders


def test_generated_sampler_keeps_the_upstream_convergence_expression():
    """The one place convergence is decided must be byte-identical upstream."""
    text = BUILT_SAMPLER.read_text()
    upstream = (
        "    converged = (stable & confident_tensor[decode_slots] & "
        "(new_hist_len >= ST)) | (\n"
        "        step_after >= max_denoising_steps\n"
        "    )\n"
        "    # Commit done → denoise next (False); denoise converged → commit next (True)\n"
        "    is_encoder_phase[decode_slots] = torch.where(\n"
        "        is_commit, is_commit.new_zeros(num_decode), converged\n"
        "    )\n"
    )
    assert upstream in text
    # exactly one assignment to `converged`, and no preact line near it
    assigns = [ln for ln in text.splitlines() if re.match(r"\s*converged\s*=", ln)]
    assert len(assigns) == 1, assigns
    for ln in text.splitlines():
        if "preact" in ln.lower():
            assert "is_encoder_phase" not in ln
            assert not re.search(r"\bconverged\b", ln)


def test_container_module_returns_only_an_accept_mask():
    src = (ROOT / "patches/preact_priority.py").read_text()
    body = src.split("def priority_step(", 1)[1].split("\ndef ", 1)[0]
    assert "is_encoder_phase" not in body
    assert "converged" not in body
    returns = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("return ")]
    assert returns == ["return accept"], returns


# ---------------------------------------------------------------------------
# the container actually receives the variables
# ---------------------------------------------------------------------------

_PREACT_VARS = (
    "CRYSTALCALL_PREACT",
    "CRYSTALCALL_PREACT_MODE",
    "CRYSTALCALL_PREACT_BOUND",
    "CRYSTALCALL_PREACT_STABLE",
    "CRYSTALCALL_PREACT_OPEN_ID",
    "CRYSTALCALL_PREACT_CLOSE_ID",
)


def _rendered(env_extra: dict[str, str]) -> list[str]:
    import os

    env = {**os.environ, **env_extra}
    proc = subprocess.run(
        ["bash", str(SERVE), "--print-docker-args"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.splitlines()


def test_rendered_container_argv_carries_every_preact_variable():
    lines = _rendered({"CRYSTALCALL_PREACT": "1"})
    got = {}
    for ln in lines:
        for var in _PREACT_VARS:
            if ln.startswith(var + "="):
                got[var] = ln.split("=", 1)[1]
    assert set(got) == set(_PREACT_VARS), sorted(set(_PREACT_VARS) - set(got))
    assert got["CRYSTALCALL_PREACT"] == "1"
    assert got["CRYSTALCALL_PREACT_MODE"] == "both"
    assert got["CRYSTALCALL_PREACT_OPEN_ID"] == str(OPEN_ID)
    assert got["CRYSTALCALL_PREACT_CLOSE_ID"] == str(CLOSE_ID)
    assert got["CRYSTALCALL_PREACT_BOUND"] not in ("", None)
    assert got["CRYSTALCALL_PREACT_STABLE"] not in ("", None)
    # the priority module is mounted where the patch imports it from
    assert any("crystalcall_preact.py" in ln for ln in lines)


def test_rendered_container_argv_defaults_preact_off():
    lines = _rendered({})
    assert "CRYSTALCALL_PREACT=0" in lines
    assert "CRYSTALCALL_PREACT=1" not in lines


def test_serve_script_rejects_an_unknown_mode():
    import os

    proc = subprocess.run(
        ["bash", str(SERVE), "--print-docker-args"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={**os.environ, "CRYSTALCALL_PREACT": "1", "CRYSTALCALL_PREACT_MODE": "early_commit"},
    )
    assert proc.returncode != 0
    assert "span_accept|ready_signal|both" in proc.stderr
