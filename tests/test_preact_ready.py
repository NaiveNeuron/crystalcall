"""The host half of the P-ReAct-style baseline: `tool_call_ready` -> launch.

The sampler-side priority mechanism lives in `patches/` and `crystalcall.preact`;
these tests pin the *consumer* side, which is where a silent failure would
be most damaging: an ignored event makes the baseline look slow, and an
uncounted false launch makes it look perfect.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crystalcall.canonicalize import canonicalize
from crystalcall.effects import EffectRegistry
from crystalcall.parse_ptc import PtcExtractor
from crystalcall.replay import (
    PREACT_ARM,
    PREACT_POLICIES,
    Trajectory,
    _policy_launches,
    analyze_trajectory,
    build_views,
    load_trajectories,
)
from crystalcall.scheduler import CostModel, RemainingDecodeModel
from crystalcall.sidechannel import REVISION_SCHEMA, RequestStream, SamplerEvent
from crystalcall.trace import CanvasRevision, TraceWriter

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# side channel
# --------------------------------------------------------------------------- #


def test_schema_is_v2_and_ready_is_additive():
    assert REVISION_SCHEMA == "crystalcall-revision/2"


def test_request_stream_routes_tool_call_ready():
    s = RequestStream(request_id="r0")
    s.apply(
        SamplerEvent(
            kind="tool_call_ready",
            request_id="r0",
            monotonic_ns=1_500,
            payload={
                "kind": "tool_call_ready",
                "request_id": "r0",
                "canvas_id": 0,
                "step": 7,
                "monotonic_ns": 1_500,
                "span": [12, 41],
                "span_stable": True,
            },
        )
    )
    assert s.ready == [
        {"canvas_id": 0, "step": 7, "monotonic_ns": 1_500, "span_lo": 12, "span_hi": 41}
    ]
    # It is a readiness signal only: it must not close the request or be
    # mistaken for a canvas commit.
    assert not s.closed.is_set()
    assert s.canvas_done_ns == {}
    assert s.revisions == []


def test_request_stream_tolerates_a_missing_span():
    s = RequestStream(request_id="r0")
    s.apply(
        SamplerEvent(
            kind="tool_call_ready",
            request_id="r0",
            monotonic_ns=9,
            payload={"kind": "tool_call_ready", "request_id": "r0", "step": 1},
        )
    )
    assert s.ready[0]["span_lo"] == -1
    assert s.ready[0]["span_hi"] == -1


def test_a_v1_producer_is_still_understood():
    """The four frozen kinds keep working; `/2` only adds one."""
    s = RequestStream(request_id="r0")
    for kind, payload in (
        ("request_begin", {"prompt_tokens": 3, "sampler": {"canvas_length": 4}}),
        ("canvas_done", {"canvas_id": 0}),
        ("request_end", {"output_tokens": 4, "finish_reason": "stop"}),
    ):
        s.apply(
            SamplerEvent(
                kind=kind,
                request_id="r0",
                monotonic_ns=1,
                payload={"kind": kind, "request_id": "r0", "monotonic_ns": 1, **payload},
            )
        )
    assert s.prompt_tokens == 3
    assert s.canvas_done_ns == {0: 1}
    assert s.closed.is_set()
    assert s.ready == []


# --------------------------------------------------------------------------- #
# offline policy
# --------------------------------------------------------------------------- #


def _tokenizer():
    class _T:
        def encode(self, text: str) -> list[int]:
            return list(text.encode("utf-8"))

        def decode(self, ids, **_kw) -> str:
            return bytes(int(i) for i in ids).decode("utf-8", "replace")

        def convert_ids_to_tokens(self, ids) -> list[str]:
            return [chr(int(i)) for i in ids]

    return _T()


class _Decoder:
    """Byte-per-token decoder with exact offsets, as `CanvasDecoder` provides."""

    def __init__(self) -> None:
        self._t = _tokenizer()

    def decode(self, ids):  # noqa: ANN001, ANN201
        from crystalcall.tokens import DecodedCanvas

        ids = [int(i) for i in ids]
        text = self._t.decode(ids)
        offsets = tuple((i, i + 1) for i in range(len(ids)))
        return DecodedCanvas(text=text, offsets=offsets, exact=True)


def _rev(
    request_id: str,
    step: int,
    ns: int,
    program: str,
    *,
    accepted: bool,
    canvas_id: int = 0,
) -> CanvasRevision:
    ids = tuple(program.encode("utf-8"))
    return CanvasRevision(
        request_id=request_id,
        canvas_id=canvas_id,
        step=step,
        monotonic_ns=ns,
        predicted_token_ids=ids,
        accepted_mask=tuple(1 if accepted else 0 for _ in ids),
        token_confidence=tuple(0.99 for _ in ids),
        mean_entropy=0.02,
        converged=False,
        phase="denoise",
        prefix_len=0,
    )


FINAL = 'r0 = tool.calc_sum(a=1.0, b=2.0)\ndisplay({"r0": r0})\n'
WRONG = 'r0 = tool.calc_sum(a=9.0, b=2.0)\ndisplay({"r0": r0})\n'


def _trajectory(registry: EffectRegistry, arm: str, ready: list[dict[str, int]]) -> Trajectory:
    revs = [
        _rev("t0", 0, 1_000_000, WRONG, accepted=False),
        _rev("t0", 1, 2_000_000, FINAL, accepted=False),
        _rev("t0", 2, 3_000_000, FINAL, accepted=True),
    ]
    call = canonicalize(registry, "calc_sum", {"a": 1.0, "b": 2.0})
    return Trajectory(
        request_id="t0",
        prompt_id="p0",
        category="simple",
        seed=0,
        arm=arm,
        revisions=revs,
        final_text="ok",
        final_program=FINAL,
        request_start_ns=0,
        response_done_ns=9_000_000,
        canvas_done_ns={0: 4_000_000},
        ready=ready,
        authoritative_calls=[call.to_json()],
        valid_final_parse=True,
        benchmark_correct=True,
    )


def _launches(registry: EffectRegistry, traj: Trajectory):
    views = build_views(traj, PtcExtractor(registry=registry), _Decoder())
    analysis = analyze_trajectory(traj, views, registry)
    return _policy_launches(
        traj,
        views,
        analysis,
        registry,
        "preact_ready",
        remaining_model=RemainingDecodeModel(),
        costs=CostModel(),
    )


def test_preact_ready_launches_at_the_readiness_timestamp(registry):
    traj = _trajectory(
        registry,
        PREACT_ARM,
        [{"canvas_id": 0, "step": 1, "monotonic_ns": 2_000_000, "span_lo": 5, "span_hi": 38}],
    )
    launches = _launches(registry, traj)
    assert len(launches) == 1
    lr = launches[0]
    assert lr.launch_ns == 2_000_000
    assert lr.launch_step == 1
    assert lr.matched_final is True
    assert lr.reason == "preact_ready[c0:5:38]"
    # Strictly earlier than the response, which is the whole point of the arm.
    assert lr.launch_ns < traj.response_done_ns


def test_a_readiness_signal_on_a_call_the_final_program_drops_is_a_false_launch(registry):
    """The span can settle to a call that later changes.  If those were
    silently discarded the baseline would show 100% precision by construction."""
    traj = _trajectory(
        registry,
        PREACT_ARM,
        [
            # Step 0 carries the WRONG argument; the final program does not.
            {"canvas_id": 0, "step": 0, "monotonic_ns": 1_000_000, "span_lo": 5, "span_hi": 38},
            {"canvas_id": 0, "step": 2, "monotonic_ns": 3_000_000, "span_lo": 5, "span_hi": 38},
        ],
    )
    launches = _launches(registry, traj)
    matched = [lr for lr in launches if lr.matched_final]
    false = [lr for lr in launches if not lr.matched_final]
    assert len(false) == 1, launches
    assert false[0].launch_ns == 1_000_000
    assert len(matched) == 1
    assert matched[0].launch_ns == 3_000_000


def test_a_call_with_no_readiness_signal_falls_back_to_response_completion(registry):
    traj = _trajectory(registry, PREACT_ARM, [])
    launches = _launches(registry, traj)
    assert [lr.reason for lr in launches] == ["preact_no_ready_signal"]
    assert launches[0].launch_ns == traj.response_done_ns


def test_preact_ready_refuses_a_non_preact_arm(registry):
    traj = _trajectory(registry, "diffusion_standard", [])
    with pytest.raises(ValueError, match="only applies to the 'preact_style' arm"):
        _launches(registry, traj)


def test_a_forged_readiness_event_on_another_arm_is_refused(registry):
    """The arm check is unconditional.  A readiness event appearing on a
    standard trace is a corrupt or forged shard, not a faster baseline."""
    traj = _trajectory(
        registry,
        "diffusion_standard",
        [{"canvas_id": 0, "step": 1, "monotonic_ns": 1, "span_lo": 5, "span_hi": 38}],
    )
    with pytest.raises(ValueError, match="only applies to the 'preact_style' arm"):
        _launches(registry, traj)


def test_the_event_joins_on_canvas_and_step_not_step_alone(registry):
    """Step indices restart per canvas.  Joining on `step` alone would credit a
    readiness event to a different canvas's revision."""
    reg = registry
    revs = [
        # canvas 0, step 1: the WRONG call
        _rev("t0", 1, 1_000_000, WRONG, accepted=False, canvas_id=0),
        # canvas 1, step 1: the final call.  Same step number.
        _rev("t0", 1, 2_000_000, FINAL, accepted=True, canvas_id=1),
    ]
    call = canonicalize(reg, "calc_sum", {"a": 1.0, "b": 2.0})
    traj = Trajectory(
        request_id="t0",
        prompt_id="p0",
        category="simple",
        seed=0,
        arm=PREACT_ARM,
        revisions=revs,
        final_text="ok",
        final_program=FINAL,
        request_start_ns=0,
        response_done_ns=9_000_000,
        canvas_done_ns={0: 1_500_000, 1: 3_000_000},
        ready=[{"canvas_id": 1, "step": 1, "monotonic_ns": 2_000_000, "span_lo": 5, "span_hi": 38}],
        authoritative_calls=[call.to_json()],
        valid_final_parse=True,
        benchmark_correct=True,
    )
    launches = _launches(reg, traj)
    assert len(launches) == 1
    assert launches[0].matched_final is True
    assert launches[0].launch_ns == 2_000_000
    assert launches[0].reason.startswith("preact_ready[c1:")


def test_a_candidate_outside_the_signalled_span_is_not_launched(registry):
    """One event names one span.  A second call elsewhere in the canvas was
    never announced and must not inherit the readiness timestamp."""
    reg = registry
    two = (
        'r0 = tool.calc_sum(a=1.0, b=2.0)\n'
        'r1 = tool.calc_sum(a=3.0, b=4.0)\n'
        'display({"r0": r0, "r1": r1})\n'
    )
    revs = [_rev("t0", 0, 1_000_000, two, accepted=True)]
    c0 = canonicalize(reg, "calc_sum", {"a": 1.0, "b": 2.0})
    c1 = canonicalize(reg, "calc_sum", {"a": 3.0, "b": 4.0})
    traj = Trajectory(
        request_id="t0",
        prompt_id="p0",
        category="parallel",
        seed=0,
        arm=PREACT_ARM,
        revisions=revs,
        final_text="ok",
        final_program=two,
        request_start_ns=0,
        response_done_ns=9_000_000,
        canvas_done_ns={0: 4_000_000},
        # The span covers only the FIRST statement.
        ready=[{"canvas_id": 0, "step": 0, "monotonic_ns": 1_000_000, "span_lo": 5, "span_hi": 32}],
        authoritative_calls=[c0.to_json(), c1.to_json()],
        valid_final_parse=True,
        benchmark_correct=True,
    )
    launches = _launches(reg, traj)
    by_reason = {lr.reason: lr for lr in launches}
    assert len(launches) == 2, launches
    signalled = [lr for lr in launches if lr.reason.startswith("preact_ready[")]
    fell_back = [lr for lr in launches if lr.reason == "preact_no_ready_signal"]
    assert len(signalled) == 1, by_reason
    assert len(fell_back) == 1, by_reason
    assert signalled[0].identity == c0.identity
    assert fell_back[0].identity == c1.identity
    assert fell_back[0].launch_ns == traj.response_done_ns


def test_preact_ready_is_not_offered_to_other_arms():
    from crystalcall.replay import BASELINE_POLICIES

    assert PREACT_POLICIES == ("preact_ready",)
    assert not set(PREACT_POLICIES) & set(BASELINE_POLICIES)


def test_the_main_table_row_d_uses_the_readiness_policy():
    import yaml

    spec = yaml.safe_load((ROOT / "configs/experiments/main_table.yaml").read_text())
    row = next(r for r in spec["rows"] if r["id"] == "preact_style")
    assert row["arm"] == PREACT_ARM
    assert row["policy"] == "preact_ready"
    # It launches before the program is authoritative, so it is speculative and
    # must be held to the same accounting as CrystalCall.
    assert row["speculative"] is True
    assert row["source"] == "generated"


# --------------------------------------------------------------------------- #
# trace round trip
# --------------------------------------------------------------------------- #


def test_ready_records_survive_the_shard_round_trip(tmp_path, registry):
    shard = tmp_path / "preact-seed0.jsonl"
    writer = TraceWriter(shard, meta={"arm": PREACT_ARM})
    writer.append(
        "request",
        {
            "request_id": "t0",
            "prompt_id": "p0",
            "category": "simple",
            "seed": 0,
            "arm": PREACT_ARM,
            "request_start_ns": 0,
        },
    )
    for rev in _trajectory(registry, PREACT_ARM, []).revisions:
        writer.revision(rev)
    writer.append(
        "tool_call_ready",
        {
            "request_id": "t0",
            "canvas_id": 0,
            "step": 1,
            "monotonic_ns": 2_000_000,
            "span_lo": 5,
            "span_hi": 38,
        },
    )
    writer.append(
        "final",
        {
            "request_id": "t0",
            "final_text": "ok",
            "final_program": FINAL,
            "response_done_ns": 9_000_000,
            "valid_final_parse": True,
        },
    )
    writer.close()

    (traj,) = load_trajectories([shard], verify=True)
    assert traj.ready == [
        {"canvas_id": 0, "step": 1, "monotonic_ns": 2_000_000, "span_lo": 5, "span_hi": 38}
    ]
    assert traj.to_json()["n_ready"] == 1
    # And the record is inside the hash chain like every other record.
    lines = shard.read_text().splitlines()
    assert any(json.loads(ln)["kind"] == "tool_call_ready" for ln in lines)
