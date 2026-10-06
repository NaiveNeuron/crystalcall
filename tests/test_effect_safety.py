"""Effect safety: only Pure and SnapshotRead are ever speculated, and the
launch gate abstains for everything else.

Also proves the negative safety property that matters most: a speculative call
that *runs to completion after invalidation* leaves no observable effect,
because no write or live read is ever launched in the first place.
"""

from __future__ import annotations

import threading
import time

import pytest

from crystalcall.canonicalize import canonicalize
from crystalcall.effects import EffectKind, ResourceClass
from crystalcall.futures import FutureKey, FutureTable, QuarantineViolation
from crystalcall.parse_ptc import PtcExtractor
from crystalcall.runtime import DenoisePtcRuntime, Instrumentation
from crystalcall.scheduler import CostModel, GatePolicy, RemainingDecodeModel, RevisionTracker
from crystalcall.sidechannel import RequestStream
from crystalcall.tokens import CanvasDecoder

from conftest import make_trajectory


def test_effect_eligibility_table(registry):
    assert registry.is_speculation_eligible("calc_sum")
    assert registry.is_speculation_eligible("search_repo")
    assert registry.is_speculation_eligible("sqlite_query")
    assert not registry.is_speculation_eligible("read_live_clock")
    assert not registry.is_speculation_eligible("write_note")
    assert set(registry.eligible_names()) == {
        "calc_sum",
        "lookup_symbol",
        "search_repo",
        "sqlite_query",
        "tag_set",
    }


def test_snapshot_read_requires_a_resolver():
    from crystalcall.effects import EffectKind, ParamKind, ParamSpec, ResourceClass, ToolSpec

    with pytest.raises(ValueError, match="requires a snapshot_resolver"):
        ToolSpec(
            name="bad",
            effect=EffectKind.SNAPSHOT_READ,
            resource=ResourceClass.STORAGE,
            params=(ParamSpec("q", ParamKind.STR),),
            expected_duration_ms=1.0,
        )


def test_future_table_refuses_write_and_live_read(registry, tools):
    table = FutureTable()
    try:
        for name, args in (
            ("write_note", {"path": "a.txt", "text": "x"}),
            ("read_live_clock", {}),
        ):
            call = canonicalize(registry, name, args)
            key = FutureKey.of("r1", call)
            with pytest.raises(QuarantineViolation):
                table.launch(key, call, ResourceClass.CPU, lambda _c: None)
            assert not table.has(key)
        assert tools.side_effects == []
    finally:
        table.close()


def _run_stream(registry, tools, canvas_texts, policy, instrumentation=Instrumentation.FULL):
    from crystalcall.sidechannel import RevisionSink

    decoder = CanvasDecoder(tokenizer=__import__("conftest").CharTokenizer())
    sink = RevisionSink()
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=decoder,
        policy=policy,
        costs=CostModel(),
        remaining_model=RemainingDecodeModel(expected_steps_per_canvas=48.0),
        instrumentation=instrumentation,
    )
    stream = RequestStream(request_id="srv-1")
    rt = runtime.begin("r1", registry, tools.impls(), stream)
    for rev in make_trajectory(canvas_texts):
        rt.on_revision(rev)
    if runtime.futures is not None:
        runtime.futures.drain()
    return runtime, rt


def test_gate_never_launches_write_or_live_read(registry, tools):
    code = (
        "```python\n"
        'write_note(path="a.txt", text="x")\n'
        "read_live_clock()\n"
        'lookup_symbol(name="A")\n'
        "```"
    )
    policy = GatePolicy(name="t", k=1, tau=0.5, min_overlap_ms=0.0)
    runtime, rt = _run_stream(registry, tools, [code] * 6, policy)
    try:
        launched_tools = {k.canonical_tool for k in rt.launched.values()}
        assert launched_tools == {"lookup_symbol"}
        assert tools.side_effects == []
        assert [n for n, _ in tools.calls] == ["lookup_symbol"]
    finally:
        runtime.close()


def test_resource_gate_can_abstain(registry, tools):
    code = '```python\nsearch_repo(query="q", path="src")\n```'
    # search_repo is a STORAGE tool; disallow storage and the gate must abstain.
    policy = GatePolicy(
        name="cpu_only",
        k=1,
        tau=0.5,
        min_overlap_ms=0.0,
        allowed_resources=frozenset({ResourceClass.CPU}),
    )
    runtime, rt = _run_stream(registry, tools, [code] * 4, policy)
    try:
        assert rt.launched == {}
        assert tools.calls == []
    finally:
        runtime.close()


def test_overlap_gate_abstains_when_no_time_remains(registry, tools):
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(name="strict_overlap", k=1, tau=0.5, min_overlap_ms=10_000.0)
    runtime, rt = _run_stream(registry, tools, [code] * 4, policy)
    try:
        assert rt.launched == {}
    finally:
        runtime.close()


def test_live_gate_judges_overlap_against_the_injected_tool_duration(registry, tools):
    """A manifest may declare a 0 ms tool while the harness injects a real
    delay (`--tool-sleep-ms`).  The live gate must judge overlap against the
    injected delay, or the overlap check rejects every candidate and the
    online arm launches nothing (the 0-of-50 outcome of the first live run)."""
    from crystalcall.sidechannel import RevisionSink

    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(name="overlap", k=1, tau=0.5, min_overlap_ms=50.0)
    from dataclasses import replace

    zero = registry.__class__(
        replace(registry.get(n), expected_duration_ms=0.0) if n == "lookup_symbol" else registry.get(n)
        for n in registry.names()
    )
    for override, expect_launch in ((None, False), (500.0, True)):
        runtime = DenoisePtcRuntime(
            sink=RevisionSink(),
            decoder=CanvasDecoder(tokenizer=__import__("conftest").CharTokenizer()),
            policy=policy,
            costs=CostModel(),
            remaining_model=RemainingDecodeModel(expected_steps_per_canvas=48.0),
            tool_duration_override_ms=override,
        )
        rt = runtime.begin("r1", zero, tools.impls(), RequestStream(request_id="srv-1"))
        for rev in make_trajectory([code] * 4):
            rt.on_revision(rev)
        runtime.futures.drain()
        try:
            assert bool(rt.launched) is expect_launch, (override, rt.launched)
        finally:
            runtime.close()


def test_persistence_gate_delays_launch(registry, tools):
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(name="k3", k=3, tau=0.5, min_overlap_ms=0.0)
    runtime, rt = _run_stream(registry, tools, [code] * 5, policy)
    try:
        state = next(iter(rt.tracker.states.values()))
        assert state.launched_step == 2  # third consecutive revision
    finally:
        runtime.close()


def test_confidence_gate_blocks_low_confidence(registry, tools):
    from crystalcall.sidechannel import RevisionSink
    from crystalcall.trace import CanvasRevision

    import conftest

    decoder = CanvasDecoder(tokenizer=conftest.CharTokenizer())
    sink = RevisionSink()
    policy = GatePolicy(name="tau_high", k=1, tau=0.95, min_overlap_ms=0.0)
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=decoder,
        policy=policy,
        instrumentation=Instrumentation.FULL,
    )
    stream = RequestStream(request_id="srv-1")
    rt = runtime.begin("r1", registry, tools.impls(), stream)
    try:
        for rev in make_trajectory(['```python\nlookup_symbol(name="A")\n```'] * 4, confidence=0.5):
            rt.on_revision(rev)
        runtime.futures.drain()
        assert rt.launched == {}
    finally:
        runtime.close()


def test_parse_only_instrumentation_never_launches(registry, tools):
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(name="t", k=1, tau=0.5, min_overlap_ms=0.0)
    runtime, rt = _run_stream(registry, tools, [code] * 4, policy, Instrumentation.PARSE)
    try:
        assert rt.stats.parses == 4
        assert rt.stats.launches == 0
        assert tools.calls == []
    finally:
        runtime.close()


def _run_accepted(registry, tools, canvas_texts, policy, accepted_from_step):
    """`_run_stream` with entropy-bound acceptance modelled explicitly."""
    from crystalcall.sidechannel import RevisionSink

    decoder = CanvasDecoder(tokenizer=__import__("conftest").CharTokenizer())
    sink = RevisionSink()
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=decoder,
        policy=policy,
        costs=CostModel(),
        remaining_model=RemainingDecodeModel(expected_steps_per_canvas=48.0),
        instrumentation=Instrumentation.FULL,
    )
    stream = RequestStream(request_id="srv-acc")
    rt = runtime.begin("r-acc", registry, tools.impls(), stream)
    for rev in make_trajectory(canvas_texts, accepted_from_step=accepted_from_step):
        rt.on_revision(rev)
    if runtime.futures is not None:
        runtime.futures.drain()
    return runtime, rt


def test_acceptance_gate_waits_for_the_accepted_mask(registry, tools):
    """`use_acceptance` is the forward reading of replay's retrospective
    `diffusion_earliest_committed` reference: launch when every token of the
    span lies inside the entropy-bound accepted set."""
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(
        name="acc",
        k=1,
        tau=0.5,
        min_overlap_ms=0.0,
        use_persistence=False,
        use_confidence=False,
        use_acceptance=True,
    )
    runtime, rt = _run_accepted(registry, tools, [code] * 5, policy, accepted_from_step=2)
    try:
        state = next(iter(rt.tracker.states.values()))
        assert state.first_committed_step == 2
        assert state.launched_step == 2
    finally:
        runtime.close()


def test_acceptance_gate_abstains_while_nothing_is_accepted(registry, tools):
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(
        name="acc",
        k=1,
        tau=0.5,
        min_overlap_ms=0.0,
        use_persistence=False,
        use_confidence=False,
        use_acceptance=True,
    )
    runtime, rt = _run_accepted(registry, tools, [code] * 5, policy, accepted_from_step=None)
    try:
        assert rt.launched == {}
        assert tools.calls == []
    finally:
        runtime.close()


def test_accepted_span_can_still_be_renoised_into_a_false_launch(registry, tools):
    """This sampler renoises non-accepted positions, so acceptance is NOT
    monotone (`accepted_count` decreases in 84 of 100 development
    trajectories). An acceptance-gated launch is therefore a
    real bet, not a certainty: the launched identity need not be the final one.
    """
    early = '```python\nlookup_symbol(name="A")\n```'
    final = '```python\nlookup_symbol(name="B")\n```'
    policy = GatePolicy(
        name="acc",
        k=1,
        tau=0.5,
        min_overlap_ms=0.0,
        use_persistence=False,
        use_confidence=False,
        use_acceptance=True,
    )
    runtime, rt = _run_accepted(
        registry, tools, [early, early, final, final], policy, accepted_from_step=1
    )
    try:
        a_digest = canonicalize(registry, "lookup_symbol", {"name": "A"}).args_digest
        b_digest = canonicalize(registry, "lookup_symbol", {"name": "B"}).args_digest
        launched = {k.canonical_arguments for k in rt.launched.values()}
        assert launched, "the accepted span must have been launched"
        # The final program asks for B, but A was accepted first and launched.
        # That launch is wasted work the exact-key claim discards, and it is
        # invisible to the retrospective reference, which only ever launches
        # calls that appear in the final program.
        assert a_digest in launched
        assert a_digest != b_digest
    finally:
        runtime.close()


def test_trace_only_instrumentation_never_parses(registry, tools):
    code = '```python\nlookup_symbol(name="A")\n```'
    policy = GatePolicy(name="t", k=1, tau=0.5, min_overlap_ms=0.0)
    runtime, rt = _run_stream(registry, tools, [code] * 4, policy, Instrumentation.TRACE)
    try:
        assert rt.stats.revisions == 4
        assert rt.stats.parses == 0
        assert tools.calls == []
    finally:
        runtime.close()


def test_cancellation_failure_creates_no_semantic_effect(registry, tools):
    """A speculation that ignores cancellation and runs to completion after
    invalidation must still leave nothing observable."""
    table = FutureTable()
    started = threading.Event()
    finished = threading.Event()

    call = canonicalize(registry, "lookup_symbol", {"name": "A"})
    key = FutureKey.of("r1", call)

    def stubborn(cancel: threading.Event):
        started.set()
        # Deliberately ignore `cancel`.
        time.sleep(0.05)
        finished.set()
        return {"symbol": "A"}

    try:
        table.launch(key, call, ResourceClass.CPU, stubborn)
        assert started.wait(2.0)
        table.cancel(key, reason="mutated")
        table.discard_unmatched("r1")
        assert finished.wait(2.0)
        table.drain()
        # The result is unreachable even though the work completed.
        claimed, value = table.claim(key, timeout=1.0)
        assert claimed is False
        assert value is None
        fut = table.get(key)
        assert fut is not None and fut.state.value in ("discarded", "cancelled")
        assert tools.side_effects == []
    finally:
        table.close()


def test_synthetic_benchmark_tools_are_stubbed_but_never_speculation_eligible():
    """Every manifest prompt must be runnable, and a synthetic `Write` must
    still be permanently ineligible.

    Regression: `build_impls` used to raise for any non-Pure declaration
    without a snapshot root, so a single BFCL prompt declaring `update_user_info`
    aborted the whole 100-prompt development trace. No real implementation
    exists for a synthetic benchmark tool whatever effect the manifest rules
    assign it; the effect governs speculation eligibility, not executability.
    """
    from pathlib import Path

    from crystalcall.manifest import load_manifest, registry_for
    from crystalcall.tools import build_impls

    root = Path(__file__).resolve().parents[1]
    seen_write = 0
    for name in ("dev.jsonl", "test.jsonl"):
        path = root / "experiments/manifests" / name
        if not path.is_file():
            continue
        for rec in load_manifest(path):
            impls = build_impls(rec.tools)
            registry = registry_for(rec)
            for decl in rec.tools:
                assert str(decl.name) in impls, (rec.prompt_id, decl.name)
                spec = registry.get(str(decl.name))
                if spec is None:
                    continue
                if spec.effect is EffectKind.WRITE:
                    seen_write += 1
                    assert not spec.speculation_eligible
                if spec.effect is EffectKind.LIVE_READ:
                    assert not spec.speculation_eligible
    assert seen_write > 0, "the manifests declare no Write tool; the guard proves nothing"
