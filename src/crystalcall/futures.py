"""Quarantined speculative future table.

A future is keyed by

    (request_id, canonical_tool, canonical_arguments, immutable_snapshot_id)

and its result is *quarantined*: the value is stored inside a wrapper that
refuses to yield it to anybody except an exact-key claim.  Nothing else in the
process can read a speculative result -- not the model context, not the OMP
transcript, not a downstream tool, not the persistent trace (the trace records
only a digest and timing).

Cancellation is an optimisation, never a safety property.  Because only
``Pure`` and ``SnapshotRead`` tools are ever launched, a speculative call that
runs to completion after invalidation is harmless by construction; the table
simply discards its value.
"""

from __future__ import annotations

import enum
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import Future as _Future
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .canonicalize import CanonicalCall
from .effects import EffectKind, ResourceClass

__all__ = [
    "FutureKey",
    "FutureState",
    "SpeculativeFuture",
    "FutureTable",
    "QuarantineViolation",
    "FutureEvent",
]


class QuarantineViolation(RuntimeError):
    """Raised when speculative output is read without an exact-key claim."""


class FutureState(enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CLAIMED = "claimed"
    DISCARDED = "discarded"


@dataclass(frozen=True, slots=True)
class FutureKey:
    """The one and only key by which a speculative result may be claimed."""

    request_id: str
    canonical_tool: str
    canonical_arguments: str  # digest of the canonical argument object
    immutable_snapshot_id: str | None

    @staticmethod
    def of(request_id: str, call: CanonicalCall) -> FutureKey:
        return FutureKey(
            request_id=request_id,
            canonical_tool=call.tool,
            canonical_arguments=call.args_digest,
            immutable_snapshot_id=call.snapshot_id,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "canonical_tool": self.canonical_tool,
            "canonical_arguments": self.canonical_arguments,
            "immutable_snapshot_id": self.immutable_snapshot_id,
        }

    def __str__(self) -> str:
        return (
            f"{self.request_id}/{self.canonical_tool}/"
            f"{self.canonical_arguments[:12]}/{self.immutable_snapshot_id or '-'}"
        )


class _Quarantined:
    """Opaque holder.  ``release`` requires the exact key."""

    __slots__ = ("_value", "_key", "_released", "_error")

    def __init__(self, key: FutureKey) -> None:
        self._key = key
        self._value: Any = None
        self._error: BaseException | None = None
        self._released = False

    def store(self, value: Any) -> None:
        self._value = value

    def store_error(self, exc: BaseException) -> None:
        self._error = exc

    def release(self, key: FutureKey) -> Any:
        if key != self._key:
            raise QuarantineViolation(f"key mismatch: {key} != {self._key}")
        if self._error is not None:
            raise self._error
        self._released = True
        return self._value

    def drop(self) -> None:
        self._value = None
        self._error = None

    @property
    def released(self) -> bool:
        return self._released

    # Deliberately hostile accessors: any accidental read is a loud failure.
    def __repr__(self) -> str:  # pragma: no cover - defensive
        return f"<Quarantined {self._key}>"

    def __str__(self) -> str:  # pragma: no cover - defensive
        raise QuarantineViolation("speculative result stringified without a claim")

    def __iter__(self) -> Iterator[Any]:  # pragma: no cover - defensive
        raise QuarantineViolation("speculative result iterated without a claim")


@dataclass
class SpeculativeFuture:
    """One launched speculation."""

    key: FutureKey
    call: CanonicalCall
    resource: ResourceClass
    state: FutureState = FutureState.PENDING
    launched_ns: int = 0
    started_ns: int = 0
    finished_ns: int = 0
    claimed_ns: int = 0
    cancel_requested_ns: int = 0
    discarded_ns: int = 0
    launch_revision: int = -1
    launch_reason: str = ""
    error: str | None = None
    duplicate_hits: int = 0
    _payload: _Quarantined = field(init=False, repr=False)
    _inner: _Future | None = field(default=None, repr=False)
    _cancel_flag: threading.Event = field(default_factory=threading.Event, repr=False)

    def __post_init__(self) -> None:
        self._payload = _Quarantined(self.key)

    # ---- lifecycle ------------------------------------------------------ #

    @property
    def terminal(self) -> bool:
        return self.state in (
            FutureState.COMPLETED,
            FutureState.FAILED,
            FutureState.CANCELLED,
            FutureState.CLAIMED,
            FutureState.DISCARDED,
        )

    @property
    def wall_ms(self) -> float:
        if self.finished_ns and self.started_ns:
            return (self.finished_ns - self.started_ns) / 1e6
        return 0.0

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_flag.is_set()

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key.to_json(),
            "call": self.call.to_json(),
            "resource": self.resource.value,
            "state": self.state.value,
            "launched_ns": self.launched_ns,
            "started_ns": self.started_ns,
            "finished_ns": self.finished_ns,
            "claimed_ns": self.claimed_ns,
            "cancel_requested_ns": self.cancel_requested_ns,
            "discarded_ns": self.discarded_ns,
            "launch_revision": self.launch_revision,
            "launch_reason": self.launch_reason,
            "error": self.error,
            "duplicate_hits": self.duplicate_hits,
            "wall_ms": self.wall_ms,
            "released": self._payload.released,
        }


@dataclass(frozen=True, slots=True)
class FutureEvent:
    """Append-only audit record for the trace."""

    monotonic_ns: int
    kind: str  # launch | start | finish | fail | cancel | claim | discard | duplicate
    key: FutureKey
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "monotonic_ns": self.monotonic_ns,
            "kind": self.kind,
            "key": self.key.to_json(),
            "detail": self.detail,
        }


class FutureTable:
    """Thread-safe table of quarantined speculative futures."""

    def __init__(
        self,
        *,
        max_workers: int = 8,
        concurrency: Mapping[ResourceClass, int] | None = None,
        clock: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self._lock = threading.RLock()
        self._futures: dict[FutureKey, SpeculativeFuture] = {}
        self._events: list[FutureEvent] = []
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="crystal-spec")
        limits = dict(concurrency or {})
        self._sem: dict[ResourceClass, threading.BoundedSemaphore] = {
            rc: threading.BoundedSemaphore(max(1, limits.get(rc, 2))) for rc in ResourceClass
        }
        self._clock = clock
        self._closed = False

    # ---- introspection -------------------------------------------------- #

    def __len__(self) -> int:
        with self._lock:
            return len(self._futures)

    def get(self, key: FutureKey) -> SpeculativeFuture | None:
        with self._lock:
            return self._futures.get(key)

    def has(self, key: FutureKey) -> bool:
        with self._lock:
            return key in self._futures

    def futures(self) -> tuple[SpeculativeFuture, ...]:
        with self._lock:
            return tuple(self._futures.values())

    def events(self) -> tuple[FutureEvent, ...]:
        with self._lock:
            return tuple(self._events)

    def _record(self, kind: str, key: FutureKey, detail: str = "") -> None:
        self._events.append(FutureEvent(self._clock(), kind, key, detail))

    # ---- launch --------------------------------------------------------- #

    def launch(
        self,
        key: FutureKey,
        call: CanonicalCall,
        resource: ResourceClass,
        thunk: Callable[[threading.Event], Any],
        *,
        revision: int = -1,
        reason: str = "",
    ) -> SpeculativeFuture:
        """Launch a speculation.  Refuses ineligible effects and duplicates."""
        if not call.effect.speculation_eligible:
            raise QuarantineViolation(
                f"refusing to speculate effect {call.effect.value} for {call.tool}"
            )
        with self._lock:
            if self._closed:
                raise RuntimeError("future table is closed")
            existing = self._futures.get(key)
            if existing is not None:
                existing.duplicate_hits += 1
                self._record("duplicate", key, reason)
                return existing
            fut = SpeculativeFuture(
                key=key,
                call=call,
                resource=resource,
                launched_ns=self._clock(),
                launch_revision=revision,
                launch_reason=reason,
            )
            self._futures[key] = fut
            self._record("launch", key, reason)

        def _run() -> Any:
            sem = self._sem[resource]
            sem.acquire()
            try:
                with self._lock:
                    if fut.state is FutureState.PENDING:
                        fut.state = FutureState.RUNNING
                        fut.started_ns = self._clock()
                        self._record("start", key)
                value = thunk(fut._cancel_flag)
                with self._lock:
                    fut.finished_ns = self._clock()
                    fut._payload.store(value)
                    if fut.state is FutureState.RUNNING:
                        fut.state = FutureState.COMPLETED
                    self._record("finish", key, f"{fut.wall_ms:.3f}ms")
                return None
            except BaseException as exc:  # noqa: BLE001 - recorded, never raised out
                with self._lock:
                    fut.finished_ns = self._clock()
                    fut.error = f"{type(exc).__name__}: {exc}"
                    fut._payload.store_error(exc)
                    if fut.state is not FutureState.DISCARDED:
                        fut.state = FutureState.FAILED
                    self._record("fail", key, fut.error)
                return None
            finally:
                sem.release()

        fut._inner = self._pool.submit(_run)
        return fut

    # ---- claim / await -------------------------------------------------- #

    def claim(self, key: FutureKey, *, timeout: float | None = None) -> tuple[bool, Any]:
        """Attempt an exact-key claim.

        Returns ``(claimed, value)``.  ``claimed`` is ``False`` when no future
        with this exact key exists, when it failed, or when it had already been
        claimed or discarded -- in which case the caller must execute normally.
        Blocks until the matching future settles.
        """
        with self._lock:
            fut = self._futures.get(key)
            if fut is None:
                return (False, None)
            if fut.state in (FutureState.CLAIMED, FutureState.DISCARDED, FutureState.CANCELLED):
                return (False, None)
            inner = fut._inner

        if inner is not None:
            try:
                inner.result(timeout=timeout)
            except TimeoutError:
                return (False, None)
            except BaseException:  # noqa: BLE001 - the worker never re-raises
                pass

        with self._lock:
            if fut.state is not FutureState.COMPLETED:
                return (False, None)
            try:
                value = fut._payload.release(key)
            except QuarantineViolation:
                return (False, None)
            except BaseException:  # noqa: BLE001 - failed speculation
                return (False, None)
            fut.state = FutureState.CLAIMED
            fut.claimed_ns = self._clock()
            self._record("claim", key)
            return (True, value)

    # ---- cancel / discard ----------------------------------------------- #

    def cancel(self, key: FutureKey, *, reason: str = "") -> bool:
        """Best-effort cancellation.  Never a safety property."""
        with self._lock:
            fut = self._futures.get(key)
            if fut is None or fut.terminal:
                return False
            fut._cancel_flag.set()
            fut.cancel_requested_ns = self._clock()
            self._record("cancel", key, reason)
            if fut._inner is not None and fut._inner.cancel():
                fut.state = FutureState.CANCELLED
                fut.finished_ns = self._clock()
                fut._payload.drop()
                return True
            return False

    def discard_unmatched(self, request_id: str, *, reason: str = "unmatched") -> int:
        """Discard every not-yet-claimed future for a finished request."""
        n = 0
        with self._lock:
            for key, fut in self._futures.items():
                if key.request_id != request_id:
                    continue
                if fut.state is FutureState.CLAIMED:
                    continue
                if fut.state is FutureState.DISCARDED:
                    continue
                fut._cancel_flag.set()
                if fut._inner is not None:
                    fut._inner.cancel()
                fut.state = FutureState.DISCARDED
                fut.discarded_ns = self._clock()
                fut._payload.drop()
                self._record("discard", key, reason)
                n += 1
        return n

    def drain(self, timeout: float = 30.0) -> None:
        """Wait for every in-flight speculation to settle (test hygiene)."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                pending = [f for f in self._futures.values() if f._inner and not f._inner.done()]
            if not pending:
                return
            time.sleep(0.005)

    def close(self) -> None:
        with self._lock:
            self._closed = True
        self._pool.shutdown(wait=False, cancel_futures=True)

    # ---- accounting ----------------------------------------------------- #

    def stats(self, request_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            fs = [
                f
                for f in self._futures.values()
                if request_id is None or f.key.request_id == request_id
            ]
        by_state: dict[str, int] = {}
        for f in fs:
            by_state[f.state.value] = by_state.get(f.state.value, 0) + 1
        claimed = [f for f in fs if f.state is FutureState.CLAIMED]
        wasted = [f for f in fs if f.state in (FutureState.DISCARDED, FutureState.CANCELLED)]
        return {
            "launched": len(fs),
            "by_state": by_state,
            "claimed": len(claimed),
            "wasted": len(wasted),
            "duplicate_hits": sum(f.duplicate_hits for f in fs),
            "wasted_tool_ms": sum(f.wall_ms for f in wasted),
            "claimed_tool_ms": sum(f.wall_ms for f in claimed),
            "precision": (len(claimed) / len(fs)) if fs else None,
        }


def assert_no_write_speculation(call: CanonicalCall) -> None:
    """Belt-and-braces guard used at every launch site."""
    if call.effect in (EffectKind.WRITE, EffectKind.LIVE_READ):
        raise QuarantineViolation(f"effect {call.effect.value} is never speculation-eligible")
