"""The central systems invariant:

    Obs(R(P)) == Obs(R_s(P))

For every eligible episode, the authoritative tool results, the authoritative
tool trace and the persistent state must be identical under ordinary and
speculative scheduling.  Only timing and resource consumption may differ.
"""

from __future__ import annotations

import threading

import pytest

from crystalcall.effects import ResourceClass
from crystalcall.futures import FutureTable
from crystalcall.omp_bridge import HostToolRouter
from crystalcall.runtime import DenoisePtcRuntime, Instrumentation
from crystalcall.scheduler import CostModel, GatePolicy, RemainingDecodeModel
from crystalcall.sidechannel import RequestStream, RevisionSink
from crystalcall.tokens import CanvasDecoder

import conftest
from conftest import make_trajectory

POLICY = GatePolicy(
    name="test_full_gate",
    k=2,
    tau=0.9,
    min_overlap_ms=0.0,
    max_inflight=8,
    allowed_resources=frozenset({ResourceClass.CPU, ResourceClass.STORAGE, ResourceClass.GPU}),
)

# A representative authoritative program exercising every accepted shape.
FINAL_PROGRAM = """
a = search_repo(query="retry timeout", path="src/")
b = sqlite_query(database="frozen.db", sql="SELECT id FROM tickets")
vs = [lookup_symbol(name=x) for x in ["A", "B"]]
c = calc_sum(a=1, b=2)
write_note(path="out.txt", text="done")
"""

FINAL_CANVAS = f"```python\n{FINAL_PROGRAM}```"

# Provisional canvases: the call mutates twice before settling on the final form.
MUTATING_TRAJECTORY = [
    '```python\na = search_repo(query="retry", path="src/")\n```',
    '```python\na = search_repo(query="retry timeo", path="src/")\n```',
    FINAL_CANVAS,
    FINAL_CANVAS,
    FINAL_CANVAS,
    FINAL_CANVAS,
]


def authoritative_calls(registry):
    """The calls the final program issues, in program order, as OMP would."""
    from crystalcall.parse_ptc import PtcExtractor

    result = PtcExtractor(registry=registry).parse_final_program(FINAL_PROGRAM)
    assert result.ok
    return [(c.call.tool, c.call.args_dict) for c in result.candidates]


def observable(router: HostToolRouter, tools) -> dict:
    """Everything an experiment is allowed to observe about an episode."""
    return {
        "trace": [
            {
                "tool": c.call.tool if c.call else c.tool_name,
                "args": c.call.args_dict if c.call else c.raw_arguments,
                "snapshot": c.call.snapshot_id if c.call else None,
                "error": c.error,
                "canonicalization_error": c.canonicalization_error,
                "result_digest": c.result_digest,
            }
            for c in router.calls()
        ],
        "state": list(tools.side_effects),
    }


def run_ordinary(registry, tools) -> dict:
    router = HostToolRouter(registry, tools.impls(), futures=None)
    for i, (name, args) in enumerate(authoritative_calls(registry)):
        router.handle(name, args, host_call_id=f"h{i}", tool_call_id=f"t{i}")
    return observable(router, tools)


def run_speculative(registry, tools, trajectory, *, request_id="r1"):
    futures = FutureTable(concurrency={rc: 4 for rc in ResourceClass})
    sink = RevisionSink()
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=CanvasDecoder(tokenizer=conftest.CharTokenizer()),
        policy=POLICY,
        costs=CostModel(),
        remaining_model=RemainingDecodeModel(expected_steps_per_canvas=48.0),
        instrumentation=Instrumentation.FULL,
        futures=futures,
    )
    router = HostToolRouter(
        registry,
        tools.impls(),
        futures=futures,
        request_id_provider=lambda: request_id,
        claim_timeout=10.0,
    )
    stream = RequestStream(request_id="srv-1")
    rt = runtime.begin(request_id, registry, tools.impls(), stream)
    try:
        for rev in make_trajectory(trajectory, request_id="srv-1"):
            rt.on_revision(rev)
        futures.drain()
        for i, (name, args) in enumerate(authoritative_calls(registry)):
            router.handle(name, args, host_call_id=f"h{i}", tool_call_id=f"t{i}")
        report = runtime.end(rt, stream)
        stats = futures.stats(request_id)
        return observable(router, tools), stats, report, router
    finally:
        runtime.close()


# --------------------------------------------------------------------------- #
# the invariant
# --------------------------------------------------------------------------- #


def test_observables_are_identical(registry, tools):
    ordinary = run_ordinary(registry, tools)
    tools2 = conftest.RecordingTools()
    speculative, stats, _report, router = run_speculative(registry, tools2, MUTATING_TRAJECTORY)
    assert speculative == ordinary
    # ...and the speculation actually did something.
    assert stats["launched"] >= 1
    assert stats["claimed"] >= 1
    assert any(c.claimed_speculative for c in router.calls())


def test_claimed_result_equals_normally_computed_result(registry, tools):
    ordinary = run_ordinary(registry, tools)
    tools2 = conftest.RecordingTools()
    speculative, _stats, _r, router = run_speculative(registry, tools2, MUTATING_TRAJECTORY)
    for a, b in zip(ordinary["trace"], speculative["trace"], strict=True):
        assert a["result_digest"] == b["result_digest"]
        assert a["tool"] == b["tool"] and a["args"] == b["args"]


def test_writes_are_always_executed_authoritatively(registry, tools):
    _speculative, _stats, _r, router = run_speculative(
        registry, tools, MUTATING_TRAJECTORY
    )
    writes = [c for c in router.calls() if c.call and c.call.tool == "write_note"]
    assert len(writes) == 1
    assert writes[0].claimed_speculative is False
    assert tools.side_effects == ["out.txt<-done"]


def test_mutated_provisional_call_result_is_never_exposed(registry, tools):
    """The first two canvases contain a *different* query; those speculations
    must be discarded, never claimed, and must not appear in the trace."""
    ordinary = run_ordinary(registry, conftest.RecordingTools())
    speculative, stats, _r, router = run_speculative(registry, tools, MUTATING_TRAJECTORY)
    assert speculative["trace"] == ordinary["trace"]
    queries = {
        args.get("query")
        for args in (c.call.args_dict for c in router.calls() if c.call)
        if "query" in args
    }
    assert queries == {"retry timeout"}
    assert stats["wasted"] >= 0  # discarded futures are accounted, not exposed


def test_removed_provisional_call_cannot_leak(registry, tools):
    """A call that appears provisionally and then disappears from the final
    program must not be observable at all."""
    trajectory = [
        '```python\nghost = lookup_symbol(name="GHOST")\n```',
        '```python\nghost = lookup_symbol(name="GHOST")\n```',
        '```python\nghost = lookup_symbol(name="GHOST")\n```',
        FINAL_CANVAS,
        FINAL_CANVAS,
        FINAL_CANVAS,
    ]
    speculative, stats, _r, router = run_speculative(registry, tools, trajectory)
    names = {c.call.args_dict.get("name") for c in router.calls() if c.call}
    assert "GHOST" not in names
    ordinary = run_ordinary(registry, conftest.RecordingTools())
    assert speculative["trace"] == ordinary["trace"]
    assert stats["wasted"] >= 1


def test_no_speculation_at_all_still_matches(registry, tools):
    """With an unsatisfiable gate nothing is launched and the observables are
    unchanged: the fallback path is exercised end to end."""
    futures = FutureTable()
    sink = RevisionSink()
    runtime = DenoisePtcRuntime(
        sink=sink,
        decoder=CanvasDecoder(tokenizer=conftest.CharTokenizer()),
        policy=GatePolicy(name="never", k=99, tau=1.1, min_overlap_ms=1e9),
        instrumentation=Instrumentation.FULL,
        futures=futures,
    )
    router = HostToolRouter(
        registry, tools.impls(), futures=futures, request_id_provider=lambda: "r1"
    )
    stream = RequestStream(request_id="srv-1")
    rt = runtime.begin("r1", registry, tools.impls(), stream)
    try:
        for rev in make_trajectory(MUTATING_TRAJECTORY, request_id="srv-1"):
            rt.on_revision(rev)
        for i, (name, args) in enumerate(authoritative_calls(registry)):
            router.handle(name, args, host_call_id=f"h{i}", tool_call_id=f"t{i}")
        assert futures.stats("r1")["launched"] == 0
        assert not any(c.claimed_speculative for c in router.calls())
        assert observable(router, tools) == run_ordinary(registry, conftest.RecordingTools())
    finally:
        runtime.close()


def test_repeated_speculative_runs_are_deterministic(registry):
    runs = []
    for _ in range(3):
        t = conftest.RecordingTools()
        obs, _s, _r, _router = run_speculative(registry, t, MUTATING_TRAJECTORY)
        runs.append(obs)
    assert runs[0] == runs[1] == runs[2]


def test_invalid_final_call_is_preserved_as_an_error(registry, tools):
    """A malformed authoritative call must surface as a tool error and must
    never be satisfied from a future."""
    router = HostToolRouter(
        registry, tools.impls(), futures=FutureTable(), request_id_provider=lambda: "r1"
    )
    value, error = router.handle(
        "lookup_symbol", {"bogus": 1}, host_call_id="h0", tool_call_id="t0"
    )
    assert value is None
    assert error is not None and "unknown_parameter" in error
    rec = router.calls()[0]
    assert rec.canonicalization_error is not None
    assert rec.claimed_speculative is False


def test_router_rejects_missing_implementations(registry, tools):
    impls = tools.impls()
    impls.pop("calc_sum")
    with pytest.raises(ValueError, match="calc_sum"):
        HostToolRouter(registry, impls)


def test_intent_field_is_stripped_before_canonicalisation(registry, tools):
    """OMP's Python bridge injects an ``i`` intent field; it is not a tool
    parameter and must not change identity or cause a rejection."""
    router = HostToolRouter(registry, tools.impls(), futures=None)
    v1, e1 = router.handle(
        "lookup_symbol", {"name": "A", "i": "py prelude"}, host_call_id="h0", tool_call_id="t0"
    )
    v2, e2 = router.handle(
        "lookup_symbol", {"name": "A"}, host_call_id="h1", tool_call_id="t1"
    )
    assert e1 is None and e2 is None and v1 == v2
    a, b = router.calls()
    assert a.call is not None and b.call is not None
    assert a.call.identity == b.call.identity


def test_concurrent_claims_of_one_future_yield_exactly_one_winner(registry, tools):
    """Two authoritative calls with the same canonical identity: the first
    claims the future, the second executes normally.  Both see the same value."""
    from crystalcall.canonicalize import canonicalize
    from crystalcall.futures import FutureKey

    futures = FutureTable()
    try:
        call = canonicalize(registry, "lookup_symbol", {"name": "A"})
        key = FutureKey.of("r1", call)
        futures.launch(
            key, call, ResourceClass.CPU, lambda c: tools.lookup_symbol(call.args_dict, c)
        )
        futures.drain()
        router = HostToolRouter(
            registry, tools.impls(), futures=futures, request_id_provider=lambda: "r1"
        )
        results = []
        errors = []

        def go(i: int) -> None:
            v, e = router.handle(
                "lookup_symbol", {"name": "A"}, host_call_id=f"h{i}", tool_call_id=f"t{i}"
            )
            results.append(v)
            errors.append(e)

        threads = [threading.Thread(target=go, args=(i,)) for i in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [None, None]
        assert results[0] == results[1]
        claimed = [c.claimed_speculative for c in router.calls()]
        assert sum(claimed) == 1
    finally:
        futures.close()


# --------------------------------------------------------------------------- #
# per-claim verification (the direct measurement behind the S6 gate)
# --------------------------------------------------------------------------- #


def test_claim_verification_reports_a_match(registry, tools):
    """Every claimed result is re-executed normally and compared, which holds
    the model fixed by construction: a mismatch can only come from scheduling."""
    from crystalcall.canonicalize import canonicalize
    from crystalcall.futures import FutureKey

    futures = FutureTable()
    try:
        call = canonicalize(registry, "lookup_symbol", {"name": "A"})
        key = FutureKey.of("r1", call)
        futures.launch(
            key,
            call,
            ResourceClass.CPU,
            lambda c: tools.lookup_symbol(call.args_dict, c),
        )
        futures.drain()
        router = HostToolRouter(
            registry,
            tools.impls(),
            futures=futures,
            request_id_provider=lambda: "r1",
            verify_claims=True,
        )
        value, error = router.handle(
            "lookup_symbol", {"name": "A"}, host_call_id="h0", tool_call_id="t0"
        )
        assert error is None
        rec = router.calls()[0]
        assert rec.claimed_speculative is True
        # Verification is deferred out of the live turn: unflushed, it is unset.
        assert rec.verification is None
        assert router.pending_verifications == 1
        tally = router.flush_verifications()
        assert tally == {"verified": 1, "match": 1, "divergent": 0, "error": 0}
        assert router.pending_verifications == 0
        assert rec.verification == "match"
        assert rec.reference_digest == rec.result_digest
        # Verification time is accounted separately and never folded into the
        # measured call latency.
        assert rec.verification_ms >= 0.0
    finally:
        futures.close()


def test_claim_verification_detects_a_divergent_result(registry, tools):
    """The check must have teeth: a future carrying a different value for the
    same canonical key is reported as divergent rather than passing silently."""
    from crystalcall.canonicalize import canonicalize
    from crystalcall.futures import FutureKey

    futures = FutureTable()
    try:
        call = canonicalize(registry, "lookup_symbol", {"name": "A"})
        key = FutureKey.of("r1", call)
        # A deliberately wrong speculative payload under the correct key.
        futures.launch(
            key, call, ResourceClass.CPU, lambda _c: {"symbol": "A", "kind": "WRONG"}
        )
        futures.drain()
        router = HostToolRouter(
            registry,
            tools.impls(),
            futures=futures,
            request_id_provider=lambda: "r1",
            verify_claims=True,
        )
        value, error = router.handle(
            "lookup_symbol", {"name": "A"}, host_call_id="h0", tool_call_id="t0"
        )
        assert error is None
        rec = router.calls()[0]
        assert rec.claimed_speculative is True
        assert rec.verification is None
        tally = router.flush_verifications()
        assert tally["divergent"] == 1
        assert rec.verification == "divergent"
        assert rec.reference_digest != rec.result_digest
        assert rec.verification_detail
    finally:
        futures.close()


def test_unclaimed_calls_are_not_verified(registry, tools):
    router = HostToolRouter(
        registry, tools.impls(), futures=None, verify_claims=True
    )
    router.handle("lookup_symbol", {"name": "A"}, host_call_id="h0", tool_call_id="t0")
    rec = router.calls()[0]
    assert rec.claimed_speculative is False
    assert router.pending_verifications == 0
    assert router.flush_verifications() == {
        "verified": 0,
        "match": 0,
        "divergent": 0,
        "error": 0,
    }
    assert rec.verification is None
    # The tool ran exactly once: verification must not double-execute a
    # normally-executed call.
    assert [n for n, _ in tools.calls] == ["lookup_symbol"]


def test_handle_return_is_not_blocked_by_verification(registry, tools):
    """The verifying re-execution must NOT run inside the live turn: it would
    delay the model's next request in the speculating arm only, contaminating
    exactly the treatment whose latency is being measured."""
    import time

    from crystalcall.canonicalize import canonicalize
    from crystalcall.futures import FutureKey

    futures = FutureTable()
    try:
        call = canonicalize(registry, "lookup_symbol", {"name": "A"})
        key = FutureKey.of("r1", call)
        futures.launch(
            key, call, ResourceClass.CPU, lambda c: tools.lookup_symbol(call.args_dict, c)
        )
        futures.drain()
        # A tool whose normal execution is slow: if verification ran inside
        # `handle`, the call would pay that cost.
        tools.delay_ms = 250.0
        router = HostToolRouter(
            registry,
            tools.impls(),
            futures=futures,
            request_id_provider=lambda: "r1",
            verify_claims=True,
        )
        t0 = time.monotonic()
        value, error = router.handle(
            "lookup_symbol", {"name": "A"}, host_call_id="h0", tool_call_id="t0"
        )
        handle_ms = (time.monotonic() - t0) * 1000.0
        assert error is None and value is not None
        assert handle_ms < 100.0, f"handle blocked for {handle_ms:.1f} ms"
        assert router.pending_verifications == 1
        # Exactly one execution so far: the claim, not the verification.
        assert len(tools.calls) == 1

        t1 = time.monotonic()
        router.flush_verifications()
        flush_ms = (time.monotonic() - t1) * 1000.0
        assert flush_ms >= 200.0, "the deferred verification did not actually execute"
        assert len(tools.calls) == 2
        assert router.calls()[0].verification == "match"
    finally:
        tools.delay_ms = 0.0
        futures.close()
