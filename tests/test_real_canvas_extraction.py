"""Regression tests for the two silent-zero measurement bugs.

Both produced plausible-looking zeros rather than errors. A fixture captured verbatim from a
real recorded canvas is the only thing that keeps them from returning.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crystalcall.parse_ptc import PtcExtractor, locate_code_regions
from crystalcall.replay import Trajectory, analyze_trajectory, build_views
from crystalcall.trace import CanvasRevision

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/real_canvas_multiple_133.json"


def _fixture() -> dict:
    if not FIXTURE.is_file():
        pytest.skip("real-canvas fixture not captured")
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# The canvas payload is Gemma wire format, not JSON
# --------------------------------------------------------------------------- #


def test_the_fixture_is_the_native_wire_format_not_json():
    """Pin the shape, so a change in the model's surface is visible here."""
    text = _fixture()["decoded_text"]
    assert "<|tool_call>" in text
    assert "call:eval{" in text
    assert 'code:<|"|>' in text
    # The thing the old locator looked for is absent, which is why it found
    # nothing on every revision of every trajectory.
    assert '"code":' not in text
    assert "```" not in text


def test_the_gemma_locator_finds_the_program_in_a_real_canvas():
    text = _fixture()["decoded_text"]
    regions = locate_code_regions(text)
    by_locator = {r.locator: r for r in regions}
    assert "gemma_tool_call" in by_locator, sorted(by_locator)
    body = by_locator["gemma_tool_call"].text
    assert "calculate_mutual_fund_balance" in body
    # The region stops at the closing delimiter and excludes the envelope.
    assert "<|tool_call>" not in body
    assert "<eos>" not in body


def test_the_region_index_map_points_back_into_the_canvas():
    """Token spans are checked against the revision's accepted mask, so the
    map from region offsets to canvas offsets has to be exact."""
    text = _fixture()["decoded_text"]
    region = next(r for r in locate_code_regions(text) if r.locator == "gemma_tool_call")
    assert len(region.index_map) == len(region.text)
    for i, ch in enumerate(region.text):
        assert text[region.index_map[i]] == ch


def test_extract_recovers_a_canonical_candidate_from_the_real_canvas(registry_133):
    text = _fixture()["decoded_text"]
    result = PtcExtractor(registry=registry_133).extract(text)
    tools = {c.call.tool for c in result.candidates}
    assert "calculate_mutual_fund_balance" in tools, (
        f"parse_mode={result.parse_mode} "
        f"rejections={[(r.reason, r.detail) for r in result.rejections]}"
    )
    cand = next(c for c in result.candidates if c.call.tool == "calculate_mutual_fund_balance")
    assert cand.call.args_dict == {
        "annual_yield": 0.05,
        "investment_amount": 50000,
        "years": 3,
    }


def test_an_unterminated_payload_still_yields_the_tail():
    """A provisional canvas is routinely cut off mid-string; that half-formed
    hypothesis is exactly what this project exists to observe."""
    text = '<|tool_call>call:eval{code:<|"|>r0 = tool.f(a=1)\nr1 = tool.f(a='
    region = next(r for r in locate_code_regions(text) if r.locator == "gemma_tool_call")
    assert region.text.startswith("r0 = tool.f(a=1)")
    assert region.text.endswith("tool.f(a=")


# --------------------------------------------------------------------------- #
# The invariant milestone is scoped to the call's own canvas
# --------------------------------------------------------------------------- #


@pytest.fixture
def registry_133():
    from crystalcall.manifest import load_manifest, registry_for

    recs = {r.prompt_id: r for r in load_manifest(ROOT / "experiments/manifests/dev.jsonl")}
    rec = recs.get("multiple_133")
    if rec is None:
        pytest.skip("multiple_133 not in the development manifest")
    return registry_for(rec)


class _ByteDecoder:
    """Byte-per-token decoder with exact offsets."""

    def decode(self, ids):  # noqa: ANN001, ANN201
        from crystalcall.tokens import DecodedCanvas

        ids = [int(i) for i in ids]
        text = bytes(ids).decode("utf-8", "replace")
        return DecodedCanvas(
            text=text, offsets=tuple((i, i + 1) for i in range(len(ids))), exact=True
        )


def _rev(step: int, ns: int, text: str, canvas_id: int) -> CanvasRevision:
    ids = tuple(text.encode("utf-8"))
    return CanvasRevision(
        request_id="t0",
        canvas_id=canvas_id,
        step=step,
        monotonic_ns=ns,
        predicted_token_ids=ids,
        accepted_mask=tuple(1 for _ in ids),
        token_confidence=tuple(0.99 for _ in ids),
        mean_entropy=0.01,
        converged=False,
        phase="denoise",
        prefix_len=0,
    )


def test_invariant_is_scoped_to_the_calls_own_canvas(registry):
    """The closing canvas carries no call, so scanning across canvases made the
    milestone unreachable and every pre-commit lead `None`."""
    from crystalcall.canonicalize import canonicalize

    prog = 'r0 = tool.calc_sum(a=1.0, b=2.0)\ndisplay({"r0": r0})\n'
    revs = [
        _rev(0, 1_000_000, "nothing here yet\n", 0),
        _rev(1, 2_000_000, prog, 0),
        _rev(2, 3_000_000, prog, 0),
        # The continuation canvas holds no call at all.
        _rev(0, 4_000_000, "done.\n", 1),
    ]
    call = canonicalize(registry, "calc_sum", {"a": 1.0, "b": 2.0})
    traj = Trajectory(
        request_id="t0",
        prompt_id="p0",
        category="simple",
        seed=0,
        arm="diffusion_standard",
        revisions=revs,
        final_text="done.",
        final_program=prog,
        request_start_ns=0,
        response_done_ns=5_000_000,
        canvas_done_ns={0: 3_500_000, 1: 4_500_000},
        authoritative_calls=[call.to_json()],
        valid_final_parse=True,
        benchmark_correct=True,
    )
    views = build_views(traj, PtcExtractor(registry=registry), _ByteDecoder())
    analysis = analyze_trajectory(traj, views, registry)
    assert analysis.final_calls, "the authoritative call was not analysed"
    fc = analysis.final_calls[0]
    assert fc.canvas_id == 0
    assert fc.invariant_step == 1, "the milestone must be the first stable revision"
    assert fc.lead_ms() is not None, "a measurable pre-commit lead must exist"
    assert fc.lead_ms() >= 0.0
