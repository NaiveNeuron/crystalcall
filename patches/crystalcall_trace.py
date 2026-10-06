"""CrystalCall sampler side-channel emitter (stdlib only, container-side).

Mounted into the vLLM container and imported by the patched DiffusionGemma
sampler.  Emits NDJSON events (schema ``crystalcall-revision/2``) to a plain
TCP loopback socket and/or an append-only file.

``/2`` adds one optional, purely additive event kind on top of ``/1``:
``tool_call_ready``, emitted by the P-ReAct-style arm when a tool-call span in
the canvas has settled.  It is readiness only -- it never ends denoising and
never commits a canvas -- and a producer that never emits it (every non-preact
run) is still a valid ``/2`` producer.  The version describes the producer's
capability, not the run.

Design constraints (these are correctness requirements, not preferences):

* **Never block the sampler thread.**  Staged submissions and ordinary events
  use bounded ``put_nowait`` operations; ``queue.Full`` increments a drop
  counter.  Only request teardown may wait for staged CUDA events.
* **Never raise into the sampler thread.**  Every hot-path entry point is
  wrapped so that any internal error is counted, not propagated.
* **No host conversion in the sampler thread.**  A staging worker waits for
  CUDA events and calls ``.tolist()``; a separate sender worker rounds and
  ``json.dumps`` serialises already-materialised plain Python values.
* **Zero cost when off.**  With ``CRYSTALCALL_TRACE`` unset the module
  allocates no thread, no queue and no socket, and every function returns on
  its first statement.

Environment:

* ``CRYSTALCALL_TRACE``        ``1`` to enable (anything else = off).
* ``CRYSTALCALL_TRACE_SINK``   ``host:port`` NDJSON TCP sink.
* ``CRYSTALCALL_TRACE_FILE``   append NDJSON to this path (in addition to, or
                               instead of, the socket).
* ``CRYSTALCALL_TRACE_QUEUE``  output queue depth, default 4096.
* ``CRYSTALCALL_TRACE_STAGE_QUEUE``  pinned-buffer queue depth, default 8.

``nopayload`` keeps the event stream and all scalar summaries but ships empty
per-token arrays; the *caller* must consult :func:`want_payload` and skip the
device->host copy entirely in that mode.
"""

from __future__ import annotations

import atexit
import json
import os
import queue
import socket
import sys
import threading
import time

__all__ = [
    "SCHEMA",
    "ENABLED",
    "enabled",
    "want_payload",
    "begin_request",
    "defer_staged",
    "emit_revision",
    "canvas_done",
    "tool_call_ready",
    "end_request",
    "flush",
    "flush_staged",
    "stats",
    "self_test",
]

SCHEMA = "crystalcall-revision/2"

_TRUTHY = ("1", "true", "yes", "on")


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in _TRUTHY


ENABLED: bool = _env_flag("CRYSTALCALL_TRACE")
SINK: str = os.environ.get("CRYSTALCALL_TRACE_SINK", "").strip()
FILE: str = os.environ.get("CRYSTALCALL_TRACE_FILE", "").strip()
STATS_FILE: str = os.environ.get("CRYSTALCALL_TRACE_STATS", "").strip()
LEVEL: str = (os.environ.get("CRYSTALCALL_TRACE_LEVEL", "full").strip().lower() or "full")
try:
    QUEUE_MAX: int = int(os.environ.get("CRYSTALCALL_TRACE_QUEUE", "") or 4096)
except ValueError:
    QUEUE_MAX = 4096

# A sink is required: with neither socket nor file there is nowhere to put
# events, so stay fully off rather than filling a queue nobody drains.
if ENABLED and not (SINK or FILE):
    ENABLED = False

_PAYLOAD: bool = LEVEL != "nopayload"

# ---- counters (plain ints; only the worker writes SENT/BYTES/ENCODE_NS) ----
_SENT = 0
_DROPPED = 0
_ERRORS = 0
_BYTES = 0
_RECONNECTS = 0
_ENCODE_NS = 0
_QUEUE_MAX_SEEN = 0

_Q: "queue.Queue[dict | None] | None" = None
_WORKER: threading.Thread | None = None
_START_LOCK = threading.Lock()
_SENTINEL: dict | None = None  # None is the shutdown marker

# The sampler hands pinned host buffers to this queue.  A second worker waits
# for the CUDA event and materialises ordinary Python lists; the sampler never
# calls `.tolist()` and never synchronises an event.  The queue is deliberately
# bounded: a slow host consumer drops a complete staged record rather than
# stalling denoising.
_STAGE_Q: "queue.Queue[dict | object] | None" = None
_STAGE_WORKER: threading.Thread | None = None
_STAGE_SENTINEL = object()
try:
    _STAGE_QUEUE_MAX = max(
        2, min(16, int(os.environ.get("CRYSTALCALL_TRACE_STAGE_QUEUE", "8")))
    )
except ValueError:
    _STAGE_QUEUE_MAX = 8

_BACKOFF_MIN = 0.05
_BACKOFF_MAX = 2.0


def enabled() -> bool:
    """True when tracing is on.  Callers MUST check this *before* any GPU work."""
    return ENABLED


def want_payload() -> bool:
    """True when per-token arrays should be materialised (i.e. level == full)."""
    return ENABLED and _PAYLOAD


# ---------------------------------------------------------------------------
# Background sender
# ---------------------------------------------------------------------------


class _Sender:
    """Socket + file writer.  Lives entirely on the worker thread."""

    def __init__(self, sink: str, path: str) -> None:
        self._host = ""
        self._port = 0
        if sink:
            host, _, port = sink.rpartition(":")
            if host and port.isdigit():
                self._host, self._port = host, int(port)
        self._path = path
        self._sock: socket.socket | None = None
        self._fh = None
        self._next_try = 0.0
        self._backoff = _BACKOFF_MIN

    # -- socket ---------------------------------------------------------- #

    def _connect(self) -> None:
        global _RECONNECTS
        now = time.monotonic()
        if now < self._next_try:
            return
        try:
            s = socket.create_connection((self._host, self._port), timeout=1.0)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(1.0)
            self._sock = s
            self._backoff = _BACKOFF_MIN
            _RECONNECTS += 1
        except OSError:
            self._sock = None
            self._next_try = now + self._backoff
            self._backoff = min(self._backoff * 2.0, _BACKOFF_MAX)

    def _open_file(self) -> None:
        if self._fh is not None or not self._path:
            return
        try:
            self._fh = open(self._path, "ab", buffering=1 << 16)
        except OSError:
            self._path = ""  # give up permanently
            self._fh = None

    def write(self, blob: bytes) -> None:
        global _SENT, _BYTES, _ERRORS
        delivered = False
        if self._host:
            if self._sock is None:
                self._connect()
            if self._sock is not None:
                try:
                    self._sock.sendall(blob)
                    delivered = True
                except OSError:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                    self._sock = None
                    self._next_try = time.monotonic() + self._backoff
                    self._backoff = min(self._backoff * 2.0, _BACKOFF_MAX)
        if self._path:
            self._open_file()
            if self._fh is not None:
                try:
                    self._fh.write(blob)
                    delivered = True
                except OSError:
                    _ERRORS += 1
        if delivered:
            _SENT += 1
            _BYTES += len(blob)
        else:
            # Sink down: the event is dropped on the floor, by design.
            global _DROPPED
            _DROPPED += 1

    def idle(self) -> None:
        if self._fh is not None:
            try:
                self._fh.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_WR)
            except OSError:
                pass
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._fh is not None:
            try:
                self._fh.flush()
                self._fh.close()
            except OSError:
                pass
            self._fh = None


_ROUND_LIST_KEYS = ("token_confidence", "token_entropy")
_ROUND_SCALAR_KEYS = ("mean_entropy",)


def _encode(ev: dict) -> bytes:
    """Round floats and serialise.  Runs on the worker thread only."""
    for k in _ROUND_LIST_KEYS:
        xs = ev.get(k)
        if xs:
            ev[k] = [round(x, 5) for x in xs]
    for k in _ROUND_SCALAR_KEYS:
        v = ev.get(k)
        if v is not None:
            ev[k] = round(v, 5)
    return (json.dumps(ev, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _dump_stats() -> None:
    """Snapshot counters to CRYSTALCALL_TRACE_STATS (worker thread only).

    Diagnostics path only: it lets a harness outside the container read the
    drop/encode counters without touching the wire schema. Written when the
    queue drains, so it never competes with event delivery.
    """
    if not STATS_FILE:
        return
    try:
        with open(STATS_FILE, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(stats(), sort_keys=True))
            fh.write("\n")
    except OSError:
        pass


def _worker(q: "queue.Queue[dict | None]", sink: str, path: str) -> None:
    global _ENCODE_NS, _ERRORS
    sender = _Sender(sink, path)
    try:
        while True:
            ev = q.get()
            if ev is _SENTINEL:
                break
            try:
                t0 = time.monotonic_ns()
                blob = _encode(ev)
                _ENCODE_NS += time.monotonic_ns() - t0
                sender.write(blob)
            except Exception:  # noqa: BLE001 - the emitter must never die
                _ERRORS += 1
            finally:
                q.task_done()
            if q.empty():
                sender.idle()
                _dump_stats()
    finally:
        sender.close()
        _dump_stats()


def _ensure_worker() -> "queue.Queue[dict | None] | None":
    global _Q, _WORKER
    q = _Q
    if q is not None:
        return q
    with _START_LOCK:
        if _Q is not None:
            return _Q
        _Q = q = queue.Queue(maxsize=max(1, QUEUE_MAX))
        _WORKER = threading.Thread(
            target=_worker,
            args=(q, SINK, FILE),
            name="crystalcall-trace",
            daemon=True,
        )
        _WORKER.start()
        return q


def _put(ev: dict) -> bool:
    global _DROPPED, _QUEUE_MAX_SEEN, _ERRORS
    try:
        q = _ensure_worker()
        if q is None:
            _DROPPED += 1
            return False
        n = q.qsize()
        if n > _QUEUE_MAX_SEEN:
            _QUEUE_MAX_SEEN = n
        q.put_nowait(ev)
        return True
    except queue.Full:
        _DROPPED += 1
        return False
    except Exception:  # noqa: BLE001 - never raise into the sampler
        _ERRORS += 1
        return False

def _release_stage(item: dict) -> None:
    """Return a sampler-owned pinned slot after host materialisation."""
    item["busy"] = False
    item["recs"] = None


def _stage_worker(q: "queue.Queue[dict | object]") -> None:
    """Wait for CUDA copies and enqueue plain wire events off the sampler."""
    global _DROPPED, _ERRORS
    while True:
        item = q.get()
        if item is _STAGE_SENTINEL:
            q.task_done()
            return
        try:
            # This is the only event wait in the process.  It is deliberately
            # on the staging worker, never on the sampler's denoise thread.
            item["event"].synchronize()
            available_ns = time.monotonic_ns()
            n = int(item["n"])
            payload = bool(item["payload"])
            scal = item["scalars"][:n].tolist()
            means = item["mean_entropy"][:n].tolist()
            toks = item["tokens"][:n].tolist() if payload else None
            masks = item["mask"][:n].tolist() if payload else None
            confs = item["conf"][:n].tolist() if payload else None
            ents = item["entropy"][:n].tolist() if payload else None
            recs = item["recs"]
            valid_lens = item["valid_lens"]
            for i in range(n):
                rec = recs[i]
                if rec is None or rec.desynced:
                    continue
                committing = bool(scal[i][2])
                canvas_id = rec.canvas_id
                prefix_len = rec.prefix_len
                revision = {
                    "kind": "revision",
                    "request_id": rec.req_id,
                    "canvas_id": int(canvas_id),
                    "step": int(scal[i][0]),
                    # A lead must not credit GPU/copy time that had not
                    # elapsed.  `monotonic_ns` is host availability after
                    # event completion; the step enqueue time is retained
                    # only as an additive diagnostic.
                    "monotonic_ns": available_ns,
                    "sample_monotonic_ns": int(item["sample_ns"]),
                    "predicted_token_ids": toks[i] if payload else [],
                    "accepted_mask": masks[i] if payload else [],
                    "token_confidence": confs[i] if payload else [],
                    "token_entropy": ents[i] if payload else [],
                    "mean_entropy": float(means[i]),
                    "converged": bool(scal[i][1]),
                    "phase": "commit" if committing else "denoise",
                    "prefix_len": int(prefix_len),
                }
                if not _put(revision):
                    rec.desynced = True
                    continue
                if committing:
                    if not _put(
                        {
                            "kind": "canvas_done",
                            "request_id": rec.req_id,
                            "canvas_id": int(canvas_id),
                            "monotonic_ns": available_ns,
                        }
                    ):
                        rec.desynced = True
                        continue
                    rec.canvas_id += 1
                    rec.prefix_len += int(valid_lens[i])
        except Exception:  # noqa: BLE001 - never kill the staging worker
            for rec in item.get("recs") or []:
                if rec is not None:
                    rec.desynced = True
            _ERRORS += 1
            _DROPPED += 1
        finally:
            _release_stage(item)
            q.task_done()


def _ensure_stage_worker() -> "queue.Queue[dict | object] | None":
    global _STAGE_Q, _STAGE_WORKER
    q = _STAGE_Q
    if q is not None:
        return q
    with _START_LOCK:
        if _STAGE_Q is not None:
            return _STAGE_Q
        _STAGE_Q = q = queue.Queue(maxsize=_STAGE_QUEUE_MAX)
        _STAGE_WORKER = threading.Thread(
            target=_stage_worker,
            args=(q,),
            name="crystalcall-trace-stage",
            daemon=True,
        )
        _STAGE_WORKER.start()
        return q


def defer_staged(item: dict) -> bool:
    """Hand one pinned staging buffer to the event-wait worker."""
    global _DROPPED, _ERRORS
    if not ENABLED:
        return False
    try:
        q = _ensure_stage_worker()
        if q is None:
            _DROPPED += 1
            return False
        q.put_nowait(item)
        return True
    except queue.Full:
        _DROPPED += 1
        return False
    except Exception:  # noqa: BLE001 - never raise into the sampler
        _ERRORS += 1
        return False


def flush_staged(timeout: float | None = None) -> bool:
    """Wait for all pinned buffers to become host-visible."""
    if not ENABLED:
        return True
    q = _STAGE_Q
    if q is None:
        return True
    if timeout is None:
        q.join()
        return True
    deadline = time.monotonic() + timeout
    while q.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    return q.unfinished_tasks == 0


# ---------------------------------------------------------------------------
# Public emitters.  Each guards on ENABLED as its first statement.
# ---------------------------------------------------------------------------


def begin_request(request_id: str, prompt_tokens: int, sampler: dict) -> None:
    if not ENABLED:
        return
    _put(
        {
            "kind": "request_begin",
            "schema": SCHEMA,
            "request_id": request_id,
            "monotonic_ns": time.monotonic_ns(),
            "prompt_tokens": int(prompt_tokens),
            "sampler": sampler,
        }
    )


def ring_depth() -> int:
    """Depth of the sampler's pinned staging ring.

    Deeper tolerates a slower consumer at the cost of pinned memory; 3 covers
    the observed one-step deferral with slack. Overridable for experiments.
    """
    try:
        d = int(os.environ.get("CRYSTALCALL_TRACE_RING", "3"))
    except ValueError:
        d = 3
    return max(2, min(d, 8))


def note_drop() -> None:
    """Record that a revision was dropped because the staging ring was full.

    Counted, never silent: a dropped revision is missing evidence, and the
    overhead ladder and `--self-test` both surface the count.
    """
    global _DROPPED
    _DROPPED += 1


def emit_revision(
    request_id: str,
    canvas_id: int,
    step: int,
    predicted_token_ids,
    accepted_mask,
    token_confidence,
    token_entropy,
    mean_entropy: float,
    converged: bool,
    phase: str,
    prefix_len: int,
    monotonic_ns: int | None = None,
    sample_monotonic_ns: int | None = None,
) -> None:
    """Queue one ``revision`` record.

    ``monotonic_ns`` is the host-availability timestamp.  A deferred CUDA
    transfer must not credit GPU/copy time that had not elapsed, so the
    staging worker supplies it after the event completes.  ``sample_monotonic_ns``
    is optional diagnostic timing for the sampler-side enqueue.
    """
    if not ENABLED:
        return
    ev = {
        "kind": "revision",
        "request_id": request_id,
        "canvas_id": int(canvas_id),
        "step": int(step),
        "monotonic_ns": int(monotonic_ns) if monotonic_ns is not None else time.monotonic_ns(),
        "predicted_token_ids": predicted_token_ids or [],
        "accepted_mask": accepted_mask or [],
        "token_confidence": token_confidence or [],
        "token_entropy": token_entropy or [],
        "mean_entropy": float(mean_entropy),
        "converged": bool(converged),
        "phase": phase,
        "prefix_len": int(prefix_len),
    }
    if sample_monotonic_ns is not None:
        ev["sample_monotonic_ns"] = int(sample_monotonic_ns)
    _put(ev)

def canvas_done(request_id: str, canvas_id: int) -> None:
    if not ENABLED:
        return
    _put(
        {
            "kind": "canvas_done",
            "request_id": request_id,
            "canvas_id": int(canvas_id),
            "monotonic_ns": time.monotonic_ns(),
        }
    )


def tool_call_ready(
    request_id: str, canvas_id: int, step: int, span_lo: int, span_hi: int
) -> None:
    """P-ReAct-style arm only: the tool-call span has settled and is launchable.

    Emitted at most ONCE per ``(request_id, canvas_id)``, at the step where the
    span first became stable and fully accepted.  Readiness only: denoising is
    not ended, no canvas is committed early, and the request finishes under the
    ordinary convergence rule exactly as in the standard arm.

    ``span`` is the half-open ``[lo, hi)`` token range over the canvas covering
    both delimiters -- the same range the offline replay reparses at ``step``.

    Additive over the frozen ``/1`` core: the pristine (no-preact) path never
    reaches this function, so a standard / CrystalCall run emits exactly
    ``request_begin``, ``revision``, ``canvas_done``, ``request_end``.
    """
    if not ENABLED:
        return
    _put(
        {
            "kind": "tool_call_ready",
            "request_id": request_id,
            "canvas_id": int(canvas_id),
            "step": int(step),
            "monotonic_ns": time.monotonic_ns(),
            "span": [int(span_lo), int(span_hi)],
            "span_stable": True,
        }
    )


def end_request(request_id: str, output_tokens: int = 0, finish_reason: str = "") -> None:
    if not ENABLED:
        return
    _put(
        {
            "kind": "request_end",
            "request_id": request_id,
            "monotonic_ns": time.monotonic_ns(),
            "output_tokens": int(output_tokens),
            "finish_reason": finish_reason,
        }
    )


def flush(timeout: float = 2.0) -> bool:
    """Best-effort drain of staged buffers and the output queue."""
    if not ENABLED:
        return True
    if not flush_staged(timeout):
        return False
    q = _Q
    if q is None:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if q.unfinished_tasks == 0:
            return True
        time.sleep(0.005)
    return q.unfinished_tasks == 0


def stats() -> dict:
    return {
        "sent": _SENT,
        "dropped": _DROPPED,
        "bytes": _BYTES,
        "reconnects": _RECONNECTS,
        "encode_ns": _ENCODE_NS,
        "queue_max": _QUEUE_MAX_SEEN,
        "errors": _ERRORS,
        "enabled": ENABLED,
        "level": LEVEL,
        "sink": SINK,
        "file": FILE,
    }


@atexit.register
def _shutdown() -> None:
    global _Q, _WORKER, _STAGE_Q, _STAGE_WORKER
    sq = _STAGE_Q
    if sq is not None:
        # Request teardown is not on the denoise hot path; preserve every
        # staged revision before closing the stage worker.
        flush_staged()
        try:
            sq.put_nowait(_STAGE_SENTINEL)
        except Exception:  # noqa: BLE001
            pass
        sw = _STAGE_WORKER
        if sw is not None:
            sw.join(5.0)
        _STAGE_Q = None
        _STAGE_WORKER = None

    q = _Q
    if q is None:
        return
    deadline = time.monotonic() + 5.0
    while q.unfinished_tasks and time.monotonic() < deadline:
        time.sleep(0.005)
    try:
        q.put_nowait(_SENTINEL)
    except Exception:  # noqa: BLE001
        return
    w = _WORKER
    if w is not None:
        w.join(5.0)
    _Q = None
    _WORKER = None


# ---------------------------------------------------------------------------
# Self test
# ---------------------------------------------------------------------------


def _reconfigure(sink: str = "", path: str = "", level: str = "full", qmax: int = 4096) -> None:
    """Test-only: point a fresh worker at a new sink."""
    global ENABLED, SINK, FILE, LEVEL, QUEUE_MAX, _PAYLOAD
    global _Q, _WORKER, _STAGE_Q, _STAGE_WORKER
    global _SENT, _DROPPED, _BYTES, _RECONNECTS, _ENCODE_NS
    global _QUEUE_MAX_SEEN, _ERRORS
    _shutdown()
    ENABLED, SINK, FILE, LEVEL, QUEUE_MAX = True, sink, path, level, qmax
    _PAYLOAD = level != "nopayload"
    _SENT = _DROPPED = _BYTES = _RECONNECTS = _ENCODE_NS = _QUEUE_MAX_SEEN = 0
    _ERRORS = 0


def self_test(n_revisions: int = 48, canvas: int = 256) -> dict:
    """Exercise the emitter against a throwaway local listener.

    Returns ``stats()`` augmented with what the listener actually decoded.
    """
    import tempfile

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    host, port = srv.getsockname()

    received: list[dict] = []
    malformed = 0
    done = threading.Event()

    def serve() -> None:
        nonlocal malformed
        try:
            conn, _ = srv.accept()
        except OSError:
            done.set()
            return
        with conn, conn.makefile("rb") as fh:
            for raw in fh:
                if not raw.strip():
                    continue
                try:
                    received.append(json.loads(raw))
                except json.JSONDecodeError:
                    malformed += 1
        done.set()

    t = threading.Thread(target=serve, daemon=True)
    t.start()

    tmp = tempfile.NamedTemporaryFile(prefix="cc-trace-", suffix=".ndjson", delete=False)
    tmp.close()
    _reconfigure(sink=f"{host}:{port}", path=tmp.name, level="full")

    rid = "cc-selftest-0"
    begin_request(rid, 17, {"sampler": "entropy_bound", "entropy_bound": 0.1, "canvas_length": canvas})
    ids = list(range(canvas))
    mask = [1 if i % 3 == 0 else 0 for i in range(canvas)]
    conf = [0.5 + i / (2.0 * canvas) for i in range(canvas)]
    ent = [i / float(canvas) for i in range(canvas)]
    for step in range(n_revisions):
        emit_revision(
            rid,
            0,
            step,
            ids,
            mask,
            conf,
            ent,
            sum(ent) / len(ent),
            step == n_revisions - 1,
            "commit" if step == n_revisions - 1 else "denoise",
            0,
        )
    canvas_done(rid, 0)
    end_request(rid, canvas, "stop")

    ok_flush = flush(10.0)
    out = stats()

    # Shut the worker down so the listener sees EOF and finishes its loop.
    q = _Q
    if q is not None:
        try:
            q.put_nowait(_SENTINEL)
        except Exception:  # noqa: BLE001
            pass
        if _WORKER is not None:
            _WORKER.join(5.0)
    done.wait(5.0)
    srv.close()

    n_expected = n_revisions + 3
    kinds: dict[str, int] = {}
    for ev in received:
        kinds[ev.get("kind", "?")] = kinds.get(ev.get("kind", "?"), 0) + 1
    file_lines = 0
    try:
        with open(tmp.name, "rb") as fh:
            file_lines = sum(1 for line in fh if line.strip())
    except OSError:
        pass

    revs = [e for e in received if e.get("kind") == "revision"]
    arrays_ok = all(
        len(e.get("predicted_token_ids", ())) == canvas
        and len(e.get("accepted_mask", ())) == canvas
        and len(e.get("token_confidence", ())) == canvas
        and len(e.get("token_entropy", ())) == canvas
        for e in revs
    )
    steps_ok = [e["step"] for e in revs] == list(range(n_revisions))

    out.update(
        {
            "flushed": ok_flush,
            "expected_events": n_expected,
            "received_events": len(received),
            "received_kinds": kinds,
            "malformed": malformed,
            "file_lines": file_lines,
            "file": tmp.name,
            "arrays_ok": arrays_ok,
            "steps_ok": steps_ok,
            "bytes_per_revision": (out["bytes"] // max(len(revs), 1)),
            "encode_ns_per_event": (out["encode_ns"] // max(out["sent"], 1)),
            "ok": (
                len(received) == n_expected
                and out["dropped"] == 0
                and out["errors"] == 0
                and malformed == 0
                and arrays_ok
                and steps_ok
                and file_lines == n_expected
            ),
        }
    )
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return out


def _nopayload_test(n: int = 8, canvas: int = 256) -> dict:
    """Level=nopayload must emit scalars with empty arrays."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(prefix="cc-trace-np-", suffix=".ndjson", delete=False)
    tmp.close()
    _reconfigure(path=tmp.name, level="nopayload")
    assert not want_payload(), "want_payload() must be False at nopayload"
    rid = "cc-selftest-np"
    begin_request(rid, 3, {"sampler": "entropy_bound"})
    for step in range(n):
        emit_revision(rid, 0, step, None, None, None, None, 0.25, False, "denoise", 0)
    end_request(rid, 0, "stop")
    flush(5.0)
    out = stats()
    with open(tmp.name, "rb") as fh:
        evs = [json.loads(line) for line in fh if line.strip()]
    revs = [e for e in evs if e["kind"] == "revision"]
    out.update(
        {
            "events": len(evs),
            "revisions": len(revs),
            "empty_arrays": all(not e["predicted_token_ids"] and not e["token_entropy"] for e in revs),
            "bytes_per_revision": out["bytes"] // max(len(revs), 1),
        }
    )
    out["ok"] = (
        len(evs) == n + 2
        and out["revisions"] == n
        and out["empty_arrays"]
        and out["dropped"] == 0
        and out["errors"] == 0
    )
    try:
        os.unlink(tmp.name)
    except OSError:
        pass
    return out


def _drop_test(n: int = 200) -> dict:
    """A dead sink must drop, not block, not raise."""
    _reconfigure(sink="127.0.0.1:9", level="full", qmax=8)
    t0 = time.monotonic_ns()
    for step in range(n):
        emit_revision("cc-drop", 0, step, [1, 2, 3], [1, 0, 1], [0.1] * 3, [0.2] * 3, 0.2, False, "denoise", 0)
    elapsed_ns = time.monotonic_ns() - t0
    time.sleep(0.2)
    out = stats()
    out["emit_ns_total"] = elapsed_ns
    out["emit_ns_per_call"] = elapsed_ns // n
    out["ok"] = out["sent"] == 0 and out["dropped"] > 0 and out["errors"] == 0
    return out


def _off_test() -> dict:
    """With tracing off every entry point must be a cheap no-op."""
    global ENABLED, _Q, _WORKER
    if _Q is not None:
        try:
            _Q.put_nowait(_SENTINEL)
        except Exception:  # noqa: BLE001
            pass
        if _WORKER is not None:
            _WORKER.join(1.0)
    _Q = None
    _WORKER = None
    ENABLED = False
    n = 20000
    t0 = time.monotonic_ns()
    for i in range(n):
        emit_revision("x", 0, i, None, None, None, None, 0.0, False, "denoise", 0)
    ns = time.monotonic_ns() - t0
    return {
        "calls": n,
        "ns_per_call": ns / n,
        "worker_started": _Q is not None,
        "ok": _Q is None and (ns / n) < 2000,
    }


def main(argv: list[str]) -> int:
    if "--self-test" not in argv:
        print(__doc__)
        print("usage: python -m crystalcall_trace --self-test")
        return 2
    full = self_test()
    nop = _nopayload_test()
    drop = _drop_test()
    off = _off_test()
    for name, res in (("full", full), ("nopayload", nop), ("drop", drop), ("off", off)):
        print(f"[{name}] {json.dumps(res, sort_keys=True)}")
    ok = all(r["ok"] for r in (full, nop, drop, off))
    print("SELF-TEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
