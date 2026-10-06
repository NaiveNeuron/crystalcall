"""The trace transfer path: narrow dtypes, pinned staging, deferred emission.

Phase 0 measured the original `.cpu()` path at +16.5% end-to-end, all of it in
the device->host transfer. These tests pin the properties of
the replacement that make it cheap and, more importantly, the properties that
make it *correct*: the wire format is unchanged, the timestamp belongs to the
step the revision came from, and a dropped revision is counted rather than
emitted torn.

They run without a GPU: the generated sampler file is checked structurally, and
the emitter is exercised directly.
"""

from __future__ import annotations

import importlib.util
import json
import re
import time
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "patches/build/diffusion_gemma.py"
TRACE_PATCH = ROOT / "patches/vllm-trace.patch"


@pytest.fixture(scope="module")
def emitter():
    """The in-container emitter, imported directly (stdlib only)."""
    path = ROOT / "patches/crystalcall_trace.py"
    if not path.is_file():
        pytest.skip("emitter missing")
    spec = importlib.util.spec_from_file_location("_cc_trace", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_cc_trace"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built() -> str:
    if not BUILD.is_file():
        pytest.skip("patched sampler not built; run patches/apply_patch.sh")
    return BUILD.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# the copy must not stall the sampler
# --------------------------------------------------------------------------- #


def test_no_blocking_device_to_host_copy_remains(built: str):
    """`.cpu()` inside the emit path is the +15.8%."""
    emit = _emit_region(built)
    assert ".cpu()" not in emit, "a blocking device->host copy is back in the emit path"
    assert "non_blocking=True" in emit, "the staging copy must be asynchronous"
    assert "pin_memory=True" in emit, "async copies require pinned host memory"


def test_the_copy_is_awaited_by_an_event_on_the_stage_worker(built: str, emitter):
    emit = _emit_region(built)
    assert "torch.cuda.Event()" in emit
    assert "defer_staged" in emit, "the sampler must hand the event to the worker"
    assert "flush_staged" in built
    # Event synchronization and host conversion belong to the stdlib worker,
    # not to the sampler source.
    stage_worker = emitter._stage_worker.__doc__ or ""
    assert "event" in stage_worker.lower()
    assert "synchronize" in emitter._stage_worker.__code__.co_names
    assert "torch.cuda.synchronize" not in emit, "never synchronise the whole device"


def test_staging_buffers_are_allocated_once_and_reused(built: str):
    emit = _emit_region(built)
    # The ring is built only when it does not exist yet.
    assert "if self._trace_ring is None" in emit
    assert emit.count("torch.empty(") >= 6, "each field needs its own persistent buffer"
    # And acquisition never allocates.
    acquire = built[built.index("def _trace_stage_acquire") : built.index("def flush_trace")]
    assert acquire.count("torch.empty(") >= 6
    assert 'b["busy"] = True' in acquire


def test_payload_dtypes_are_narrowed(built: str):
    """int64 token ids and an int64 0/1 mask were 8 bytes each."""
    emit = _emit_region(built)
    assert "torch.int32" in emit, "token ids should be int32 (vocab 262144 fits)"
    assert "torch.uint8" in emit, "the accept mask is 0/1"
    # Confidence and entropy stay float32: the launch gate compares them to tau
    # directly, so they must be bit-for-bit what the sampler computed.
    assert "torch.float32" in emit
    assert "torch.float16" not in emit, (
        "float16 would change the tau=0.9 comparison in the launch gate"
    )


def test_a_full_ring_drops_and_counts_rather_than_stalling(built: str):
    emit = _emit_region(built)
    assert "note_drop()" in emit
    assert "if slot is None" in emit
    assert "rec.desynced = True" in emit
    # The drop path must return before staging anything.
    drop_at = emit.index("note_drop()")
    stage_at = emit.index('buf["event"].record()')
    assert drop_at < stage_at, "a dropped step must not stage a partial record"


# --------------------------------------------------------------------------- #
# correctness of the deferral
# --------------------------------------------------------------------------- #


def test_the_timestamp_is_conservative_and_diagnostic(built: str, emitter):
    """The sampler records enqueue time; the worker records availability."""
    emit = _emit_region(built)
    assert "sample_ns = time.monotonic_ns()" in emit
    assert 'buf["sample_ns"] = sample_ns' in emit
    assert "synchronize" in emitter._stage_worker.__code__.co_names
    assert {"time", "monotonic_ns"} <= set(emitter._stage_worker.__code__.co_names)


def test_staged_event_wait_and_materialisation_run_off_sampler(emitter, tmp_path):
    class HostRows:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, _slice):
            return self

        def tolist(self):
            return self.rows

    class Event:
        def __init__(self):
            self.synchronised = False

        def synchronize(self):
            self.synchronised = True
    class Rec:
        req_id = "staged-r0"
        canvas_id = 0
        prefix_len = 0
        desynced = False

    out = tmp_path / "staged.ndjson"
    emitter._reconfigure(path=str(out), level="full")
    event = Event()
    rec = Rec()
    item = {
        "event": event,
        "n": 1,
        "payload": True,
        "sample_ns": 10,
        "tokens": HostRows([[7, 8]]),
        "mask": HostRows([[1, 1]]),
        "conf": HostRows([[0.99, 0.98]]),
        "entropy": HostRows([[0.01, 0.02]]),
        "scalars": HostRows([[0, 1, 1]]),
        "mean_entropy": HostRows([0.2]),
        "recs": [rec],
        "valid_lens": [2],
        "busy": True,
    }
    try:
        assert emitter.defer_staged(item)
        assert emitter.flush_staged(5.0)
        assert emitter.flush(5.0)
    finally:
        emitter._shutdown()
    events = [json.loads(line) for line in out.read_text().splitlines()]
    revision = next(e for e in events if e["kind"] == "revision")
    assert event.synchronised
    assert revision["sample_monotonic_ns"] == 10
    assert revision["monotonic_ns"] > 0
    assert revision["phase"] == "commit"
    assert any(e["kind"] == "canvas_done" for e in events)
    assert rec.canvas_id == 1 and rec.prefix_len == 2
    assert not item["busy"]


def test_dropped_commit_suppresses_later_revisions(emitter, tmp_path):
    class HostRows:
        def __init__(self, rows):
            self.rows = rows

        def __getitem__(self, _slice):
            return self

        def tolist(self):
            return self.rows

    class Event:
        def synchronize(self):
            pass

    class Rec:
        req_id = "dropped-r0"
        canvas_id = 0
        prefix_len = 0
        desynced = True

    out = tmp_path / "dropped.ndjson"
    emitter._reconfigure(path=str(out), level="full")
    item = {
        "event": Event(),
        "n": 1,
        "payload": True,
        "sample_ns": 1,
        "tokens": HostRows([[7]]),
        "mask": HostRows([[1]]),
        "conf": HostRows([[0.99]]),
        "entropy": HostRows([[0.01]]),
        "scalars": HostRows([[0, 0, 0]]),
        "mean_entropy": HostRows([0.2]),
        "recs": [Rec()],
        "valid_lens": [1],
        "busy": True,
    }
    try:
        # This models the sampler's nonblocking drop marker after a dropped
        # commit; a later queued revision must not revive a corrupt trace.
        assert emitter.defer_staged(item)
        assert emitter.flush(5.0)
    finally:
        emitter._shutdown()
    assert not out.exists() or not out.read_text().strip()


def test_revision_replay_uses_sink_receive_time():
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    producer_ns = 7
    before = time.monotonic_ns()
    sink._ingest(
        {
            "kind": "revision",
            "request_id": "receive-r0",
            "monotonic_ns": producer_ns,
            "canvas_id": 0,
            "step": 0,
            "predicted_token_ids": [1],
            "accepted_mask": [1],
            "token_confidence": [0.99],
            "token_entropy": [0.01],
            "mean_entropy": 0.01,
            "converged": True,
            "phase": "denoise",
            "prefix_len": 0,
        }
    )
    after = time.monotonic_ns()
    stream = sink.stream("receive-r0")
    sink._server.server_close()
    assert stream is not None
    assert len(stream.revisions) == 1
    revision = stream.revisions[0]
    assert before <= revision.monotonic_ns <= after
    assert revision.producer_monotonic_ns == producer_ns


def test_raw_replay_prefers_persisted_receive_timestamp(tmp_path):
    from crystalcall.sidechannel import replay_events

    path = tmp_path / "events.ndjson"
    path.write_text(
        json.dumps(
            {
                "kind": "revision",
                "request_id": "replay-r0",
                "monotonic_ns": 10,
                "receive_monotonic_ns": 20,
            }
        )
        + "\n"
        + json.dumps(
            {
                "kind": "revision",
                "request_id": "replay-r0",
                "monotonic_ns": 30,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    events = list(replay_events(str(path)))
    assert [event.monotonic_ns for event in events] == [20, 30]
    revision = events[0].as_revision()
    assert revision is not None
    assert revision.monotonic_ns == 20
    assert revision.producer_monotonic_ns == 10


def _revision_event(request_id: str, step: int) -> dict:
    return {
        "kind": "revision",
        "request_id": request_id,
        "monotonic_ns": 100 + step,
        "canvas_id": 0,
        "step": step,
        "predicted_token_ids": [1],
        "accepted_mask": [1],
        "token_confidence": [0.99],
        "token_entropy": [0.01],
        "mean_entropy": 0.01,
        "converged": False,
        "phase": "denoise",
        "prefix_len": 0,
    }


def test_stream_is_open_until_request_end_arrives():
    """The overhead ladder must not score a rung on a prefix of its stream: the
    HTTP response and the trace sender are asynchronous, so `wait_closed` may
    only succeed once `request_end` has landed."""
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    try:
        sink._ingest(_revision_event("drain-r0", 0))
        stream = sink.stream("drain-r0")
        assert stream is not None
        assert stream.wait_closed(0.05) is False
        sink._ingest(_revision_event("drain-r0", 1))
        assert stream.wait_closed(0.05) is False
        sink._ingest(
            {
                "kind": "request_end",
                "request_id": "drain-r0",
                "monotonic_ns": 500,
                "output_tokens": 64,
                "finish_reason": "length",
            }
        )
        assert stream.wait_closed(1.0) is True
        assert len(stream.snapshot()) == 2
        assert stream.output_tokens == 64
    finally:
        sink._server.server_close()


def test_streams_since_marker_isolates_one_occurrence():
    """Each ladder occurrence counts only the streams it opened; the harness
    drops streams every iteration to bound memory, so per-occurrence markers
    must not resurrect earlier requests."""
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    try:
        sink._ingest(_revision_event("occ-0", 0))
        first = sink.streams_since(0)
        assert [s.request_id for s in first] == ["occ-0"]

        marker = sink.marker()
        assert sink.streams_since(marker) == []
        assert sink.wait_for_new_stream(marker, 0.05) is None

        sink._ingest(_revision_event("occ-1", 0))
        fresh = sink.wait_for_new_stream(marker, 1.0)
        assert fresh is not None and fresh.request_id == "occ-1"
        assert [s.request_id for s in sink.streams_since(marker)] == ["occ-1"]

        sink.drop("occ-0")
        assert [s.request_id for s in sink.streams_since(0)] == ["occ-1"]
    finally:
        sink._server.server_close()


def test_wait_for_stream_binds_exactly_one_request_id():
    """Overhead rows are attributed by the HTTP response id, so a late warm-up
    or another request's stream must never satisfy the wait."""
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    try:
        sink._ingest(_revision_event("warmup-tail", 0))
        # a foreign stream is present, but the measured id is not
        assert sink.wait_for_stream("measured-r0", 0.05) is None

        sink._ingest(_revision_event("measured-r0", 0))
        found = sink.wait_for_stream("measured-r0", 1.0)
        assert found is not None
        assert found.request_id == "measured-r0"
        assert found.wait_closed(0.02) is False
    finally:
        sink._server.server_close()


def test_wait_for_stream_joins_openai_id_to_engine_suffix():
    """The HTTP surface returns `chatcmpl-<id>`; the sampler tags events
    `chatcmpl-<id>-<engine suffix>`. The join is by prefix, and an ambiguous
    prefix is an error rather than a silent pick."""
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    try:
        sink._ingest(_revision_event("chatcmpl-abc-engine1", 0))
        found = sink.wait_for_stream("chatcmpl-abc", 1.0)
        assert found is not None
        assert found.request_id == "chatcmpl-abc-engine1"

        sink._ingest(_revision_event("chatcmpl-abc-engine2", 0))
        with pytest.raises(ValueError, match="ambiguous"):
            sink.resolve_stream("chatcmpl-abc")
    finally:
        sink._server.server_close()



def test_stream_snapshot_after_close_keeps_every_revision():
    """Snapshotting a turn's streams the moment the HTTP response returns would
    persist a PREFIX of the revision stream. This pins the property
    `run_trace_arm` relies on: a stream that has received `request_end`
    returns every revision it ever saw, and one that has not is detectable."""
    from crystalcall.sidechannel import RevisionSink

    sink = RevisionSink()
    try:
        marker = sink.marker()
        for step in range(5):
            sink._ingest(_revision_event("drain-late", step))
        fresh = sink.streams_since(marker)
        assert [s.request_id for s in fresh] == ["drain-late"]
        # Not yet closed: the snapshot is not yet trustworthy.
        assert fresh[0].wait_closed(0.02) is False
        sink._ingest(
            {
                "kind": "request_end",
                "request_id": "drain-late",
                "monotonic_ns": 900,
                "output_tokens": 64,
                "finish_reason": "length",
            }
        )
        assert fresh[0].wait_closed(1.0) is True
        assert len(fresh[0].snapshot()) == 5
    finally:
        sink._server.server_close()


def test_slot_records_are_resolved_at_stage_time(built: str):
    """The slot table changes as requests come and go, so the mapping from
    decode slot to request must be captured when the payload is staged."""
    emit = _emit_region(built)
    assert 'buf["recs"] = [_TRACE_SLOTS.get(int(x)) for x in decode_slots_np[:n]]' in emit


def test_in_flight_revisions_are_drained_before_the_slot_is_discarded(built: str):
    remove = built[built.index("def remove_request(self, req_id: str)") :][:1200]
    assert "_TRACE_SAMPLER.flush_trace()" in remove
    assert remove.index("flush_trace()") < remove.index("_TRACE_SLOTS.pop"), (
        "the drain must happen before the slot record is dropped"
    )


def test_the_sampler_is_registered_for_the_drain(built: str):
    """`remove_request` is on the model state, which holds no sampler
    reference; a `getattr(self, 'sampler', None)` would silently be None."""
    assert "_TRACE_SAMPLER = None" in built
    assert "global _TRACE_SAMPLER" in built
    assert "_TRACE_SAMPLER = self" in built


# --------------------------------------------------------------------------- #
# the wire format is unchanged
# --------------------------------------------------------------------------- #


def test_emit_revision_accepts_an_explicit_timestamp_and_keeps_the_schema(emitter, tmp_path):
    out = tmp_path / "trace.ndjson"
    emitter._reconfigure(sink="", path=str(out), level="full")
    try:
        emitter.emit_revision(
            "r0", 0, 7, [1, 2, 3], [1, 0, 1], [0.9, 0.8, 1.0], [0.01, 0.2, 0.0],
            0.05, False, "denoise", 0, 123456789,
        )
        emitter.flush()
    finally:
        emitter._shutdown()
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["kind"] == "revision"
    assert rec["monotonic_ns"] == 123456789, "an explicit timestamp must be honoured"
    assert rec["predicted_token_ids"] == [1, 2, 3]
    assert rec["accepted_mask"] == [1, 0, 1]
    assert rec["token_confidence"] == [0.9, 0.8, 1.0]
    assert rec["step"] == 7 and rec["phase"] == "denoise"
    assert set(rec) == {
        "kind", "request_id", "canvas_id", "step", "monotonic_ns",
        "predicted_token_ids", "accepted_mask", "token_confidence",
        "token_entropy", "mean_entropy", "converged", "phase", "prefix_len",
    }


def test_omitting_the_timestamp_still_stamps_now(emitter, tmp_path):
    out = tmp_path / "trace2.ndjson"
    emitter._reconfigure(sink="", path=str(out), level="full")
    try:
        emitter.emit_revision("r0", 0, 0, [1], [1], [1.0], [0.0], 0.0, False, "denoise", 0)
        emitter.flush()
    finally:
        emitter._shutdown()
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["monotonic_ns"] > 0


def test_nopayload_level_still_emits_scalars_with_empty_arrays(emitter, tmp_path):
    out = tmp_path / "trace3.ndjson"
    emitter._reconfigure(sink="", path=str(out), level="nopayload")
    try:
        emitter.emit_revision("r0", 0, 3, None, None, None, None, 0.25, True, "commit", 8, 42)
        emitter.flush()
    finally:
        emitter._shutdown()
    rec = json.loads(out.read_text(encoding="utf-8").splitlines()[-1])
    assert rec["predicted_token_ids"] == [] and rec["accepted_mask"] == []
    assert rec["mean_entropy"] == 0.25 and rec["converged"] is True
    assert rec["monotonic_ns"] == 42


def test_ring_depth_is_bounded_and_overridable(emitter, monkeypatch):
    assert 2 <= emitter.ring_depth() <= 8
    monkeypatch.setenv("CRYSTALCALL_TRACE_RING", "1")
    assert emitter.ring_depth() == 2, "depth must stay at least double-buffered"
    monkeypatch.setenv("CRYSTALCALL_TRACE_RING", "99")
    assert emitter.ring_depth() == 8, "depth must stay bounded"
    monkeypatch.setenv("CRYSTALCALL_TRACE_RING", "nonsense")
    assert emitter.ring_depth() == 3


def test_note_drop_is_counted_in_the_emitter_stats(emitter, tmp_path):
    emitter._reconfigure(sink="", path=str(tmp_path / "t.ndjson"), level="full")
    try:
        before = emitter.stats()["dropped"]
        emitter.note_drop()
        emitter.note_drop()
        assert emitter.stats()["dropped"] == before + 2
    finally:
        emitter._shutdown()


# --------------------------------------------------------------------------- #
# the optimisation must not touch decoding
# --------------------------------------------------------------------------- #


def test_the_patch_adds_nothing_to_the_sampling_maths():
    """Observation only: no added line may write a decoding tensor."""
    added = [
        ln[1:]
        for ln in TRACE_PATCH.read_text(encoding="utf-8").splitlines()
        if ln.startswith("+") and not ln.startswith("+++")
    ]
    forbidden = re.compile(
        r"\b(states\.(accepted_canvas|canvas|step|is_encoder_phase|stability)\w*)\s*(\[[^\]]*\])?\s*="
        r"|\bprobs\s*=|\blogits\s*=|\bscaled\s*="
    )
    offenders = [ln for ln in added if forbidden.search(ln)]
    assert not offenders, f"added lines mutate decoding state: {offenders[:3]}"


def _emit_region(built: str) -> str:
    """The sampler-side staging path; host conversion lives in the emitter."""
    start = built.index("def _emit_trace")
    end = built.index("def flush_trace")
    return built[start:end]
