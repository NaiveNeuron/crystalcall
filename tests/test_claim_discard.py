"""Exact claim, await, normal-execute, cancel, discard and duplicate semantics."""

from __future__ import annotations

import threading
import time

import pytest

from crystalcall.canonicalize import canonicalize
from crystalcall.effects import ResourceClass
from crystalcall.futures import FutureKey, FutureState, FutureTable, QuarantineViolation


@pytest.fixture
def table():
    t = FutureTable()
    yield t
    t.close()


def key_for(registry, request_id, name, args):
    return FutureKey.of(request_id, canonicalize(registry, name, args))


def launch(table, registry, request_id, name, args, thunk=None, resource=ResourceClass.CPU):
    call = canonicalize(registry, name, args)
    key = FutureKey.of(request_id, call)
    fn = thunk or (lambda _c: {"ok": name})
    return key, table.launch(key, call, resource, fn)


# --------------------------------------------------------------------------- #
# claim
# --------------------------------------------------------------------------- #


def test_exact_key_claim_returns_the_value(table, registry):
    key, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    claimed, value = table.claim(key, timeout=5.0)
    assert claimed is True
    assert value == {"ok": "lookup_symbol"}
    assert table.get(key).state is FutureState.CLAIMED


def test_claim_awaits_a_running_future(table, registry):
    gate = threading.Event()

    def slow(_cancel):
        gate.wait(5.0)
        return {"ok": "slow"}

    key, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"}, slow)
    t = threading.Thread(target=lambda: (time.sleep(0.05), gate.set()))
    t.start()
    claimed, value = table.claim(key, timeout=5.0)
    t.join()
    assert claimed and value == {"ok": "slow"}


def test_second_claim_of_the_same_future_fails(table, registry):
    key, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    assert table.claim(key, timeout=5.0)[0] is True
    assert table.claim(key, timeout=5.0) == (False, None)


def test_missing_future_means_execute_normally(table, registry):
    key = key_for(registry, "r1", "lookup_symbol", {"name": "A"})
    assert table.claim(key, timeout=0.1) == (False, None)


@pytest.mark.parametrize(
    ("launched", "attempted"),
    [
        ({"name": "A"}, {"name": "B"}),  # different argument
        ({"name": "A"}, {"name": "a"}),  # case-sensitive argument
    ],
)
def test_mutated_arguments_cannot_claim(table, registry, launched, attempted):
    launch(table, registry, "r1", "lookup_symbol", launched)
    table.drain()
    other = key_for(registry, "r1", "lookup_symbol", attempted)
    assert table.claim(other, timeout=0.2) == (False, None)


def test_different_request_id_cannot_claim(table, registry):
    launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    table.drain()
    other = key_for(registry, "r2", "lookup_symbol", {"name": "A"})
    assert table.claim(other, timeout=0.2) == (False, None)


def test_different_snapshot_cannot_share_a_future(table, registry):
    launch(table, registry, "r1", "sqlite_query", {"database": "one.db", "sql": "SELECT 1"})
    table.drain()
    other = key_for(registry, "r1", "sqlite_query", {"database": "two.db", "sql": "SELECT 1"})
    assert other.immutable_snapshot_id is not None
    assert table.claim(other, timeout=0.2) == (False, None)


def test_equivalent_spellings_claim_the_same_future(table, registry):
    launch(table, registry, "r1", "search_repo", {"query": "q", "path": "src/"})
    table.drain()
    equivalent = key_for(
        registry, "r1", "search_repo", {"query": "q", "path": "./src", "max_results": 50}
    )
    claimed, value = table.claim(equivalent, timeout=5.0)
    assert claimed and value == {"ok": "search_repo"}


# --------------------------------------------------------------------------- #
# quarantine
# --------------------------------------------------------------------------- #


def test_result_is_unreachable_without_a_claim(table, registry):
    sentinel = "QUARANTINED_PAYLOAD_SENTINEL"
    key, fut = launch(
        table, registry, "r1", "lookup_symbol", {"name": "A"}, lambda _c: {"v": sentinel}
    )
    table.drain()
    wrong = FutureKey(
        request_id="r1",
        canonical_tool="lookup_symbol",
        canonical_arguments="deadbeef",
        immutable_snapshot_id=None,
    )
    with pytest.raises(QuarantineViolation):
        fut._payload.release(wrong)
    with pytest.raises(QuarantineViolation):
        str(fut._payload)
    with pytest.raises(QuarantineViolation):
        list(fut._payload)
    # Neither the public serialisation nor the audit trail ever carries the value.
    assert sentinel not in repr(fut.to_json())
    assert all(sentinel not in repr(e.to_json()) for e in table.events())


def test_discarded_future_drops_its_value(table, registry):
    key, fut = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    table.drain()
    assert table.discard_unmatched("r1") == 1
    assert fut.state is FutureState.DISCARDED
    assert table.claim(key, timeout=0.2) == (False, None)


def test_discard_leaves_claimed_futures_alone(table, registry):
    key, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    assert table.claim(key, timeout=5.0)[0] is True
    assert table.discard_unmatched("r1") == 0
    assert table.get(key).state is FutureState.CLAIMED


def test_discard_is_scoped_to_one_request(table, registry):
    k1, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    k2, _ = launch(table, registry, "r2", "lookup_symbol", {"name": "A"})
    table.drain()
    assert table.discard_unmatched("r1") == 1
    assert table.get(k2).state is FutureState.COMPLETED
    assert table.claim(k2, timeout=5.0)[0] is True


# --------------------------------------------------------------------------- #
# duplicates, failures, accounting
# --------------------------------------------------------------------------- #


def test_duplicate_launch_is_suppressed(table, registry):
    calls: list[int] = []

    def counting(_cancel):
        calls.append(1)
        return {"ok": "lookup_symbol"}

    k1, f1 = launch(table, registry, "r1", "lookup_symbol", {"name": "A"}, counting)
    k2, f2 = launch(table, registry, "r1", "lookup_symbol", {"name": "A"}, counting)
    table.drain()
    assert k1 == k2
    assert f1 is f2
    assert len(calls) == 1
    assert f1.duplicate_hits == 1
    assert table.stats("r1")["duplicate_hits"] == 1


def test_failed_speculation_falls_back_to_normal_execution(table, registry):
    def boom(_cancel):
        raise RuntimeError("tool exploded")

    key, fut = launch(table, registry, "r1", "lookup_symbol", {"name": "A"}, boom)
    table.drain()
    assert fut.state is FutureState.FAILED
    assert table.claim(key, timeout=1.0) == (False, None)
    assert "tool exploded" in (fut.error or "")


def test_stats_report_precision_and_waste(table, registry):
    k_good, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    launch(table, registry, "r1", "lookup_symbol", {"name": "WRONG"})
    table.drain()
    assert table.claim(k_good, timeout=5.0)[0] is True
    table.discard_unmatched("r1")
    st = table.stats("r1")
    assert st["launched"] == 2
    assert st["claimed"] == 1
    assert st["wasted"] == 1
    assert st["precision"] == pytest.approx(0.5)


def test_events_form_an_audit_trail(table, registry):
    key, _ = launch(table, registry, "r1", "lookup_symbol", {"name": "A"})
    table.drain()
    table.claim(key, timeout=5.0)
    kinds = [e.kind for e in table.events()]
    assert kinds[0] == "launch"
    assert "start" in kinds and "finish" in kinds and "claim" in kinds
    assert all(e.key.request_id == "r1" for e in table.events())
