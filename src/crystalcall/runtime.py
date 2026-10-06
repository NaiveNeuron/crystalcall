"""The online DenoisePTC runtime.

Wires the private revision side channel to the parser, the launch gate and the
quarantined future table.  Decoding is never altered: this module only
*observes* the ordinary sampler.

Instrumentation levels (used by the Phase 0 overhead ladder):

===========  ============================================================
``off``      no side channel consumer at all
``trace``    revisions are received and recorded, nothing is parsed
``parse``    revisions are decoded, parsed, canonicalised and gated, but no
             tool is ever launched
``full``     the gate launches eligible candidates into the future table
===========  ============================================================
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonicalize import CanonicalCall
from .effects import EffectRegistry, ResourceClass
from .futures import FutureKey, FutureTable, assert_no_write_speculation
from .parse_ptc import ParseResult, PtcExtractor
from .scheduler import CostModel, GatePolicy, RemainingDecodeModel, RevisionTracker
from .sidechannel import RequestStream, RevisionSink
from .tokens import CanvasDecoder
from .trace import CanvasRevision

__all__ = ["Instrumentation", "RequestRuntime", "DenoisePtcRuntime", "RuntimeStats"]

ToolImpl = Callable[[Mapping[str, Any], threading.Event], Any]


class Instrumentation(enum.Enum):
    OFF = "off"
    TRACE = "trace"
    PARSE = "parse"
    FULL = "full"

    @property
    def parses(self) -> bool:
        return self in (Instrumentation.PARSE, Instrumentation.FULL)

    @property
    def launches(self) -> bool:
        return self is Instrumentation.FULL


@dataclass(slots=True)
class RuntimeStats:
    revisions: int = 0
    parses: int = 0
    parse_failures: int = 0
    candidates: int = 0
    launches: int = 0
    decode_ns: int = 0
    parse_ns: int = 0
    gate_ns: int = 0

    def to_json(self) -> dict[str, Any]:
        return {
            "revisions": self.revisions,
            "parses": self.parses,
            "parse_failures": self.parse_failures,
            "candidates": self.candidates,
            "launches": self.launches,
            "decode_ms": self.decode_ns / 1e6,
            "parse_ms": self.parse_ns / 1e6,
            "gate_ms": self.gate_ns / 1e6,
            "parse_ms_per_revision": (self.parse_ns / 1e6 / self.revisions) if self.revisions else 0.0,
            "decode_ms_per_revision": (
                (self.decode_ns / 1e6 / self.revisions) if self.revisions else 0.0
            ),
        }


@dataclass
class RequestRuntime:
    """Per-model-request speculative state."""

    request_id: str
    registry: EffectRegistry
    impls: Mapping[str, ToolImpl]
    extractor: PtcExtractor
    decoder: CanvasDecoder
    tracker: RevisionTracker
    futures: FutureTable | None
    instrumentation: Instrumentation
    stats: RuntimeStats = field(default_factory=RuntimeStats)
    parse_results: list[tuple[CanvasRevision, ParseResult]] = field(default_factory=list)
    launched: dict[str, FutureKey] = field(default_factory=dict)
    keep_parse_results: bool = True
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def on_revision(self, rev: CanvasRevision) -> None:
        self.stats.revisions += 1
        if not self.instrumentation.parses:
            return
        t0 = time.perf_counter_ns()
        decoded = self.decoder.decode(rev.predicted_token_ids)
        t1 = time.perf_counter_ns()
        parse = self.extractor.extract(decoded.text)
        t2 = time.perf_counter_ns()
        self.stats.decode_ns += t1 - t0
        self.stats.parse_ns += t2 - t1
        self.stats.parses += 1
        if not parse.ok:
            self.stats.parse_failures += 1
        self.stats.candidates += len(parse.candidates)
        if self.keep_parse_results:
            with self._lock:
                self.parse_results.append((rev, parse))

        decisions = self.tracker.observe(rev, parse, decoded)
        self.stats.gate_ns += time.perf_counter_ns() - t2
        if not self.instrumentation.launches or self.futures is None:
            return
        for state, decision in decisions:
            if not decision.launch:
                continue
            if state.identity in self.launched:
                continue
            self._launch(state.call, rev.step, decision.reason)

    def _launch(self, call: CanonicalCall, step: int, reason: str) -> None:
        assert self.futures is not None
        assert_no_write_speculation(call)
        spec = self.registry.get(call.tool)
        if spec is None or not spec.speculation_eligible:
            return
        impl = self.impls.get(call.tool)
        if impl is None:
            return
        key = FutureKey.of(self.request_id, call)
        args = call.args_dict

        def thunk(cancel: threading.Event, _impl: ToolImpl = impl, _args: dict[str, Any] = args) -> Any:
            return _impl(_args, cancel)

        self.futures.launch(
            key,
            call,
            spec.resource,
            thunk,
            revision=step,
            reason=reason,
        )
        self.launched[call.identity] = key
        self.stats.launches += 1

    def finish(self, *, discard: bool = True) -> dict[str, Any]:
        if self.futures is not None and discard:
            self.futures.discard_unmatched(self.request_id)
        return {
            "request_id": self.request_id,
            "stats": self.stats.to_json(),
            "tracker": self.tracker.to_json(),
            "futures": [
                f.to_json()
                for f in (self.futures.futures() if self.futures else ())
                if f.key.request_id == self.request_id
            ],
        }


class DenoisePtcRuntime:
    """Process-level owner of the side channel, decoder and future table."""

    def __init__(
        self,
        sink: RevisionSink,
        decoder: CanvasDecoder,
        *,
        policy: GatePolicy,
        costs: CostModel | None = None,
        remaining_model: RemainingDecodeModel | None = None,
        instrumentation: Instrumentation = Instrumentation.FULL,
        futures: FutureTable | None = None,
        concurrency: Mapping[ResourceClass, int] | None = None,
        tool_duration_override_ms: float | None = None,
    ) -> None:
        self.sink = sink
        self.decoder = decoder
        self.policy = policy
        #: The tool duration the gate judges overlap against.  Benchmark
        #: manifests declare `expected_duration_ms = 0.0` for their synthetic
        #: tools, and a gate that believes that rejects every candidate on the
        #: overlap check.  Replay already
        #: overrides it with the simulated duration; the live runtime must use
        #: the duration the harness actually injects (`--tool-sleep-ms`).
        self.tool_duration_override_ms = tool_duration_override_ms
        self.costs = costs or CostModel()
        self.remaining_model = remaining_model or RemainingDecodeModel()
        self.instrumentation = instrumentation
        self.futures = futures
        if self.futures is None and instrumentation.launches:
            self.futures = FutureTable(concurrency=concurrency)
        self._active: RequestRuntime | None = None
        self._lock = threading.Lock()
        self._history: list[dict[str, Any]] = []

    # ---- binding -------------------------------------------------------- #

    def begin(
        self,
        request_id: str,
        registry: EffectRegistry,
        impls: Mapping[str, ToolImpl],
        stream: RequestStream,
        *,
        keep_parse_results: bool = True,
    ) -> RequestRuntime:
        """Attach the runtime to one sampler stream.

        ``request_id`` is the *logical* CrystalCall request identity used in
        future keys; ``stream.request_id`` is the server-side id.
        """
        rt = RequestRuntime(
            request_id=request_id,
            registry=registry,
            impls=dict(impls),
            extractor=PtcExtractor(registry=registry),
            decoder=self.decoder,
            tracker=RevisionTracker(
                registry=registry,
                policy=self.policy,
                remaining_model=self.remaining_model,
                costs=self.costs,
                request_id=request_id,
                duration_override_ms=self.tool_duration_override_ms,
            ),
            futures=self.futures if self.instrumentation.launches else None,
            instrumentation=self.instrumentation,
            keep_parse_results=keep_parse_results,
        )
        with self._lock:
            self._active = rt
        # Replay anything that already arrived before binding, then subscribe.
        for rev in stream.snapshot():
            rt.on_revision(rev)
        stream.on_revision = rt.on_revision
        return rt

    def active_request_id(self) -> str:
        with self._lock:
            return self._active.request_id if self._active else ""

    def end(self, rt: RequestRuntime, stream: RequestStream | None = None) -> dict[str, Any]:
        if stream is not None:
            stream.on_revision = None
        report = rt.finish()
        with self._lock:
            self._history.append(report)
            if self._active is rt:
                self._active = None
        return report

    def history(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._history)

    def close(self) -> None:
        if self.futures is not None:
            self.futures.close()

    # ---- measured costs ------------------------------------------------- #

    def measured_costs(self) -> CostModel:
        """Recompute the cost model from observed per-revision timings."""
        parses = sum(h["stats"]["parses"] for h in self._history) or 0
        if parses == 0:
            return self.costs
        parse_ms = sum(h["stats"]["parse_ms"] for h in self._history)
        decode_ms = sum(h["stats"]["decode_ms"] for h in self._history)
        return CostModel(
            trace_ms_per_revision=decode_ms / parses,
            parse_ms_per_revision=parse_ms / parses,
            schedule_ms_per_launch=self.costs.schedule_ms_per_launch,
            contention_ms_per_launch=self.costs.contention_ms_per_launch,
        )
