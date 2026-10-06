"""Private CanvasRevision side channel.

The instrumented DiffusionGemma sampler inside the vLLM server pushes compact
NDJSON events to a loopback TCP socket owned by this module.  The channel is
*private*: nothing received here is ever written into the OMP transcript, the
model context, or a user-visible stream.  It exists only for the CrystalCall
parser, gate and trace.

Event kinds emitted by the patched sampler (schema ``crystalcall-revision/2``):

    {"kind":"request_begin","request_id":...,"monotonic_ns":...,"prompt_tokens":N,
     "sampler":{...}}
    {"kind":"revision","request_id":...,"canvas_id":C,"step":S,"monotonic_ns":...,
     "sample_monotonic_ns":...,
     "predicted_token_ids":[...],"accepted_mask":[...],
     "token_confidence":[...] | "token_entropy":[...],
     "mean_entropy":F,"converged":B,"phase":"denoise"|"commit","prefix_len":N}
    {"kind":"canvas_done","request_id":...,"canvas_id":C,"monotonic_ns":...}
    {"kind":"request_end","request_id":...,"monotonic_ns":...,"output_tokens":N,
     "finish_reason":"..."}

``/2`` adds one optional additive kind, emitted only when the server runs with
priority decoding on (``CRYSTALCALL_PREACT=1``, the P-ReAct-style baseline).
A ``/1`` producer is still a valid ``/2`` producer:

    {"kind":"tool_call_ready","request_id":...,"canvas_id":C,"step":S,
     "monotonic_ns":...,"span":[lo,hi],"span_stable":true}

It is a *readiness signal only*: it never ends denoising and never commits a
canvas -- hence ``span_stable`` rather than any word suggesting commitment.
The standard and CrystalCall arms never emit it.

``CLOCK_MONOTONIC`` is a single system-wide clock shared by the container and
the harness (Docker does not enter a time namespace).  The producer's
``monotonic_ns`` is retained as ``producer_monotonic_ns``; on live ingestion
the sink replaces ``monotonic_ns`` with the host receive timestamp.  Replay
and the online gate therefore cannot credit sender, socket, or parser backlog
as usable lead.  ``RevisionSink.clock_skew_ns`` reports the producer-to-host
transport delay.
"""

from __future__ import annotations

import json
import socket
import socketserver
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from .trace import CanvasRevision

__all__ = [
    "REVISION_SCHEMA",
    "RevisionSink",
    "RequestStream",
    "SamplerEvent",
]

REVISION_SCHEMA = "crystalcall-revision/2"


@dataclass(frozen=True, slots=True)
class SamplerEvent:
    kind: str
    request_id: str
    monotonic_ns: int
    payload: Mapping[str, Any]

    def as_revision(self) -> CanvasRevision | None:
        if self.kind != "revision":
            return None
        return CanvasRevision.from_json(self.payload)


@dataclass
class RequestStream:
    """Live, ordered view of one model request's sampler events."""

    request_id: str
    begin_ns: int = 0
    end_ns: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    revisions: list[CanvasRevision] = field(default_factory=list)
    canvas_done_ns: dict[int, int] = field(default_factory=dict)
    #: ``tool_call_ready`` events, in arrival order.  Emitted only by a server
    #: running with priority decoding on (the P-ReAct-style arm); the standard
    #: and CrystalCall arms never see one.  Each entry is
    #: ``{"canvas_id","step","monotonic_ns","span_lo","span_hi"}``.
    ready: list[dict[str, Any]] = field(default_factory=list)
    sampler: dict[str, Any] = field(default_factory=dict)
    closed: threading.Event = field(default_factory=threading.Event)
    #: Called for each revision as it arrives (the online gate hooks in here).
    on_revision: Callable[[CanvasRevision], None] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def apply(self, ev: SamplerEvent) -> None:
        if ev.kind == "request_begin":
            self.begin_ns = ev.monotonic_ns
            self.prompt_tokens = int(ev.payload.get("prompt_tokens", 0) or 0)
            self.sampler = dict(ev.payload.get("sampler") or {})
            return
        if ev.kind == "revision":
            rev = ev.as_revision()
            if rev is None:
                return
            with self._lock:
                self.revisions.append(rev)
            if self.on_revision is not None:
                self.on_revision(rev)
            return
        if ev.kind == "canvas_done":
            self.canvas_done_ns[int(ev.payload.get("canvas_id", 0))] = ev.monotonic_ns
            return
        if ev.kind == "tool_call_ready":
            span = ev.payload.get("span") or []
            entry = {
                "canvas_id": int(ev.payload.get("canvas_id", 0) or 0),
                "step": int(ev.payload.get("step", -1)),
                "monotonic_ns": ev.monotonic_ns,
                "span_lo": int(span[0]) if len(span) == 2 else -1,
                "span_hi": int(span[1]) if len(span) == 2 else -1,
            }
            with self._lock:
                self.ready.append(entry)
            return
        if ev.kind == "request_end":
            self.end_ns = ev.monotonic_ns
            self.output_tokens = int(ev.payload.get("output_tokens", 0) or 0)
            self.finish_reason = str(ev.payload.get("finish_reason", ""))
            self.closed.set()
            return

    def snapshot(self) -> list[CanvasRevision]:
        with self._lock:
            return list(self.revisions)

    @property
    def steps(self) -> int:
        return len(self.revisions)

    def wait_closed(self, timeout: float | None = None) -> bool:
        return self.closed.wait(timeout)


class _Handler(socketserver.StreamRequestHandler):
    #: Set by RevisionSink before serving.
    sink: RevisionSink

    def handle(self) -> None:  # noqa: D102
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for raw in self.rfile:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                self.sink._malformed += 1
                continue
            self.sink._ingest(obj)


class _Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


class RevisionSink:
    """Loopback receiver for the sampler side channel."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        self._streams: dict[str, RequestStream] = {}
        self._order: list[str] = []
        self._lock = threading.RLock()
        self._malformed = 0
        self._events = 0
        self._skew_samples: list[int] = []
        self._new_stream_hook: Callable[[RequestStream], None] | None = None
        handler = type("_BoundHandler", (_Handler,), {"sink": self})
        self._server = _Server((host, port), handler)
        self.host, self.port = self._server.server_address[:2]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    # ---- lifecycle ------------------------------------------------------ #

    def start(self) -> RevisionSink:
        self._thread.start()
        return self

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def __enter__(self) -> RevisionSink:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def endpoint(self) -> str:
        return f"{self.host}:{self.port}"

    # ---- ingest --------------------------------------------------------- #

    def _ingest(self, obj: Mapping[str, Any]) -> None:
        kind = str(obj.get("kind", ""))
        rid = str(obj.get("request_id", ""))
        if not kind or not rid:
            self._malformed += 1
            return
        try:
            producer_ns = int(obj.get("monotonic_ns", 0))
        except (TypeError, ValueError):
            self._malformed += 1
            return
        # The producer's timestamp marks event creation, but a launch cannot
        # act until this host has received the frame.  Keep both: replay and
        # the online gate consume the conservative receive timestamp, while
        # producer_ns remains a diagnostic in the persisted revision.
        received_ns = time.monotonic_ns()
        wire = dict(obj)
        wire["producer_monotonic_ns"] = producer_ns
        wire["monotonic_ns"] = received_ns
        self._events += 1
        if kind == "request_begin":
            self._skew_samples.append(received_ns - producer_ns)
        ev = SamplerEvent(
            kind=kind, request_id=rid, monotonic_ns=received_ns, payload=wire
        )
        with self._lock:
            stream = self._streams.get(rid)
            if stream is None:
                stream = RequestStream(request_id=rid)
                self._streams[rid] = stream
                self._order.append(rid)
                hook = self._new_stream_hook
            else:
                hook = None
        if hook is not None:
            try:
                hook(stream)
            except Exception:  # noqa: BLE001 - the sink must never die
                pass
        stream.apply(ev)

    # ---- access --------------------------------------------------------- #

    def set_new_stream_hook(self, hook: Callable[[RequestStream], None] | None) -> None:
        with self._lock:
            self._new_stream_hook = hook

    def stream(self, request_id: str) -> RequestStream | None:
        with self._lock:
            return self._streams.get(request_id)

    def request_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(self._order)

    def latest_stream(self) -> RequestStream | None:
        with self._lock:
            if not self._order:
                return None
            return self._streams[self._order[-1]]

    def streams_since(self, marker: int) -> list[RequestStream]:
        """Every stream opened at or after ``marker`` (an index into arrival order)."""
        with self._lock:
            return [self._streams[r] for r in self._order[marker:]]

    def marker(self) -> int:
        with self._lock:
            return len(self._order)

    def drop(self, request_id: str) -> None:
        with self._lock:
            self._streams.pop(request_id, None)
            if request_id in self._order:
                self._order.remove(request_id)

    def wait_for_new_stream(self, marker: int, timeout: float = 60.0) -> RequestStream | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            fresh = self.streams_since(marker)
            if fresh:
                return fresh[0]
            time.sleep(0.002)
        return None

    def resolve_stream(self, request_id: str) -> RequestStream | None:
        """Find the stream for an OpenAI response id.

        The HTTP surface returns ``chatcmpl-<id>`` while the sampler tags its
        side-channel events ``chatcmpl-<id>-<engine suffix>``, so the join is by
        prefix. An ambiguous prefix is an error, never a silent pick.
        """
        with self._lock:
            exact = self._streams.get(request_id)
            if exact is not None:
                return exact
            hits = [r for r in self._order if r.startswith(f"{request_id}-")]
            if len(hits) > 1:
                raise ValueError(
                    f"ambiguous side-channel streams for {request_id!r}: {hits}"
                )
            return self._streams[hits[0]] if hits else None

    def wait_for_stream(self, request_id: str, timeout: float = 60.0) -> RequestStream | None:
        """Wait for the stream belonging to exactly this response id.

        Arrival-order waits cannot attribute a stream to a specific HTTP
        response: a late warm-up or a previous request's tail satisfies them.
        Overhead accounting must bind the response id it measured.
        """
        deadline = time.monotonic() + timeout
        while True:
            stream = self.resolve_stream(request_id)
            if stream is not None:
                return stream
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.002)

    # ---- diagnostics ---------------------------------------------------- #

    def clock_skew_ns(self) -> dict[str, Any]:
        if not self._skew_samples:
            return {"samples": 0}
        xs = sorted(self._skew_samples)
        return {
            "samples": len(xs),
            "min_ns": xs[0],
            "median_ns": xs[len(xs) // 2],
            "max_ns": xs[-1],
        }

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "endpoint": self.endpoint,
                "requests": len(self._streams),
                "events": self._events,
                "malformed": self._malformed,
                "clock_skew": self.clock_skew_ns(),
            }


def replay_events(path: str) -> Iterator[SamplerEvent]:
    """Read captured events, preferring a persisted host-receive timestamp.

    Emitter NDJSON normally contains only producer/event-complete time. Such a
    file is suitable for structural replay but is not backlog-safe for lead or
    latency claims. A capture that persists ``receive_monotonic_ns`` (or a
    ``CanvasRevision`` JSON written after live sink ingestion) is safe to use
    for those timing claims.
    """
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            payload = dict(obj)
            raw_ns = int(obj.get("monotonic_ns", 0))
            receive_ns = int(obj.get("receive_monotonic_ns", raw_ns))
            if "receive_monotonic_ns" in obj:
                payload.setdefault("producer_monotonic_ns", raw_ns)
                payload["monotonic_ns"] = receive_ns
            yield SamplerEvent(
                kind=str(payload.get("kind", "")),
                request_id=str(payload.get("request_id", "")),
                monotonic_ns=receive_ns,
                payload=payload,
            )
