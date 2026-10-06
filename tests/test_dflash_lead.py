"""Guards on the drafter-rung measurement script.

The one error this script must never make is presenting the pre-commit lead as
measured: SGLang streams only post-verified tokens, so the script can measure
acceptance granularity and bound the lead, nothing more.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def mod():
    pytest.importorskip("jinja2", reason="script renders the target's chat template")
    spec = importlib.util.spec_from_file_location(
        "dflash_lead", REPO / "scripts" / "dflash_lead.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_percentiles_are_order_statistics(mod):
    xs = [10.0, 20.0, 30.0, 40.0]
    assert mod.pct(xs, 0.0) == 10.0
    assert mod.pct(xs, 1.0) == 40.0
    assert mod.pct([], 0.5) is None


def test_committed_artifacts_label_the_lead_as_a_bound():
    """Both rungs must expose the lead under `derived_bound`, never as a
    measured field: the draft time of a call is not observable through the
    streaming API."""
    for name in ("lead_block4.json", "lead_block8.json"):
        path = REPO / "experiments" / "sweeps" / "dflash2" / name
        blob = json.loads(path.read_text())
        summary = blob["summary"]
        assert "derived_bound" in summary
        assert "pre_commit_lead_upper_bound_ms" in summary["derived_bound"]
        assert "bound, not a measurement" in summary["derived_bound"]["why"]
        # Nothing under `measured` may be named as a lead.
        assert not [k for k in summary["measured"] if "lead" in k]


def test_committed_artifacts_carry_acceptance_granularity():
    """The measured quantities are granularity and the post-commit tail; both
    rungs must have them for every prompt, or the rung is not comparable."""
    for name, expected_tokens in (("lead_block4.json", 4.0), ("lead_block8.json", 8.0)):
        blob = json.loads((REPO / "experiments/sweeps/dflash2" / name).read_text())
        rows = blob["rows"]
        assert len(rows) == 12
        assert all(r["tokens_per_event"] for r in rows)
        assert all(r["verify_rounds"] > 0 for r in rows)
        assert all(r["ms_per_event"] for r in rows)
        # `spec_accept_length` counts the target's own bonus token alongside the
        # accepted drafts, so a block of N can commit N+1 tokens in one round.
        assert max(r["tokens_per_event"] for r in rows) <= expected_tokens + 1.0


def test_event_price_is_flat_within_a_rung():
    """One acceptance event costs one target forward pass,
    so `ms_per_event` is near-constant across prompts within a rung. A wide
    spread would mean the per-round mean is dominated by something else."""
    blob = json.loads(
        (REPO / "experiments/sweeps/dflash2/lead_block8.json").read_text()
    )
    ms = [r["ms_per_event"] for r in blob["rows"] if r["ms_per_event"]]
    assert max(ms) - min(ms) < 5.0, f"event price not flat: {min(ms)}..{max(ms)}"


def test_tail_is_an_integer_number_of_events():
    """The tail is quantised in acceptance events on this decoder too."""
    blob = json.loads(
        (REPO / "experiments/sweeps/dflash2/lead_block8.json").read_text()
    )
    ratios = [
        r["call_to_done_ms"] / r["ms_per_event"]
        for r in blob["rows"]
        if r["call_to_done_ms"] and r["ms_per_event"]
    ]
    assert ratios
    for ratio in ratios:
        assert abs(ratio - round(ratio)) < 0.05, f"tail not quantised: {ratio}"


def test_tail_tokens_are_measured_not_derived():
    """The tail in tokens comes from the stream's cumulative `output_ids`, so it
    must be an integer and the counter must agree with the server's own
    `completion_tokens`. A float here would mean someone reinstated the
    tail-ms / ms-per-token division."""
    for name in ("lead_block4.json", "lead_block8.json"):
        rows = json.loads(
            (REPO / "experiments/sweeps/dflash2" / name).read_text()
        )["rows"]
        for row in rows:
            assert isinstance(row["tail_tokens"], int)
            assert 0 < row["tail_tokens"] <= row["tokens_emitted"]
            assert row["tokens_emitted"] == row["completion_tokens"]
            assert row["tokens_at_commit"] + row["tail_tokens"] == row["tokens_emitted"]


def test_granularity_knob_bit_on_both_rungs():
    """The positive control: `spec_accept_histogram` must be as wide as the
    requested block, or the rung varied the verify window rather than the
    drafted block and the block-size comparison is not the intended one."""
    for name, width in (("lead_block4.json", 4), ("lead_block8.json", 8)):
        rows = json.loads(
            (REPO / "experiments/sweeps/dflash2" / name).read_text()
        )["rows"]
        assert {len(r["spec_accept_histogram"]) for r in rows} == {width}
