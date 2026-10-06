"""Append-only, hash-chained trace store and the ``CanvasRevision`` schema.

The trace is the primary scientific artifact.  Every record carries a sequence
number and the SHA-256 of ``prev_chain || canonical_json(record)``, so any
truncation, reordering or edit is detectable by
:meth:`TraceReader.verify`.  A sidecar ``<shard>.meta.json`` records the final
chain value, the record count and the file digest.

The revision schema mirrors the specification exactly:

    request_id, canvas_id, denoising_step, monotonic_ns,
    predicted_token_ids[N], accepted_mask[N],
    token_confidence[N] | token_entropy[N], mean_entropy, converged,
    phase in {denoise, commit}

Full vocabulary logits are never transferred or stored: only argmax token ids
and per-token scalar summaries.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "SCHEMA_VERSION",
    "CanvasRevision",
    "TraceWriter",
    "TraceReader",
    "TraceRecord",
    "TraceIntegrityError",
    "canonical_record_json",
]

SCHEMA_VERSION = "crystalcall-trace/1"

Phase = Literal["denoise", "commit"]


class TraceIntegrityError(RuntimeError):
    pass


def canonical_record_json(obj: Mapping[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _round_list(xs: Sequence[float] | None, ndigits: int = 5) -> list[float] | None:
    if xs is None:
        return None
    return [round(float(x), ndigits) for x in xs]


@dataclass(frozen=True, slots=True)
class CanvasRevision:
    """One predicted-clean canvas observation from the diffusion sampler.

    ``monotonic_ns`` is the conservative host-receive timestamp when the
    revision was constructed from a live ``RevisionSink``. Producer-side
    timestamps remain optional diagnostics; raw emitter captures without a
    persisted receive timestamp are structural-replay inputs only and must not
    be used to claim backlog-safe timing.

    """
    request_id: str
    canvas_id: int
    step: int
    monotonic_ns: int
    predicted_token_ids: tuple[int, ...]
    accepted_mask: tuple[bool, ...]
    mean_entropy: float
    converged: bool
    phase: Phase
    token_confidence: tuple[float, ...] | None = None
    token_entropy: tuple[float, ...] | None = None
    #: Number of tokens already committed in *earlier* canvases; lets the
    #: parser place this canvas inside the whole response.
    prefix_len: int = 0
    #: Server-side wall time spent producing this revision, if reported.
    step_ms: float | None = None
    #: Sampler-side enqueue timestamp, before device copies complete.
    sample_monotonic_ns: int | None = None
    #: Producer host timestamp after the CUDA event, before socket receive.
    producer_monotonic_ns: int | None = None

    @property
    def n(self) -> int:
        return len(self.predicted_token_ids)

    @property
    def accepted_count(self) -> int:
        return sum(1 for a in self.accepted_mask if a)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "request_id": self.request_id,
            "canvas_id": self.canvas_id,
            "step": self.step,
            "monotonic_ns": self.monotonic_ns,
            "predicted_token_ids": list(self.predicted_token_ids),
            "accepted_mask": [1 if a else 0 for a in self.accepted_mask],
            "mean_entropy": round(float(self.mean_entropy), 6),
            "converged": bool(self.converged),
            "phase": self.phase,
            "prefix_len": self.prefix_len,
        }
        if self.token_confidence is not None:
            out["token_confidence"] = _round_list(self.token_confidence)
        if self.token_entropy is not None:
            out["token_entropy"] = _round_list(self.token_entropy)
        if self.step_ms is not None:
            out["step_ms"] = round(float(self.step_ms), 4)
        if self.sample_monotonic_ns is not None:
            out["sample_monotonic_ns"] = int(self.sample_monotonic_ns)
        if self.producer_monotonic_ns is not None:
            out["producer_monotonic_ns"] = int(self.producer_monotonic_ns)
        return out

    @staticmethod
    def from_json(d: Mapping[str, Any]) -> CanvasRevision:
        mask = d.get("accepted_mask") or []
        return CanvasRevision(
            request_id=str(d["request_id"]),
            canvas_id=int(d.get("canvas_id", 0)),
            step=int(d.get("step", d.get("denoising_step", 0))),
            monotonic_ns=int(d["monotonic_ns"]),
            predicted_token_ids=tuple(int(t) for t in d.get("predicted_token_ids", ())),
            accepted_mask=tuple(bool(m) for m in mask),
            mean_entropy=float(d.get("mean_entropy", 0.0)),
            converged=bool(d.get("converged", False)),
            phase=d.get("phase", "denoise"),  # type: ignore[arg-type]
            token_confidence=(
                tuple(float(x) for x in d["token_confidence"])
                if d.get("token_confidence") is not None
                else None
            ),
            token_entropy=(
                tuple(float(x) for x in d["token_entropy"])
                if d.get("token_entropy") is not None
                else None
            ),
            prefix_len=int(d.get("prefix_len", 0)),
            step_ms=float(d["step_ms"]) if d.get("step_ms") is not None else None,
            sample_monotonic_ns=(
                int(d["sample_monotonic_ns"])
                if d.get("sample_monotonic_ns") is not None
                else None
            ),
            producer_monotonic_ns=(
                int(d["producer_monotonic_ns"])
                if d.get("producer_monotonic_ns") is not None
                else None
            ),
        )

    def span_confidence(self, lo: int, hi: int) -> tuple[float, float] | None:
        """``(min, mean)`` confidence over token positions ``[lo, hi)``.

        When only entropy is available it is converted to a monotone
        pseudo-confidence ``exp(-H)`` so a single threshold works for both.
        """
        lo = max(0, lo)
        hi = min(self.n, hi)
        if hi <= lo:
            return None
        if self.token_confidence is not None:
            vals = self.token_confidence[lo:hi]
        elif self.token_entropy is not None:
            import math

            vals = tuple(math.exp(-max(0.0, h)) for h in self.token_entropy[lo:hi])
        else:
            return None
        if not vals:
            return None
        return (min(vals), sum(vals) / len(vals))


@dataclass(frozen=True, slots=True)
class TraceRecord:
    seq: int
    kind: str
    chain: str
    payload: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {"seq": self.seq, "kind": self.kind, "chain": self.chain, **self.payload}


@dataclass
class TraceWriter:
    """Append-only hash-chained JSONL shard writer.

    Not thread-safe by itself; the online runtime funnels all writes through a
    single collector thread.
    """

    path: Path
    meta: Mapping[str, Any] = field(default_factory=dict)
    compress: bool | None = None
    _fh: Any = field(default=None, init=False, repr=False)
    _seq: int = field(default=0, init=False)
    _chain: str = field(default="", init=False)
    _bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.compress is None:
            self.compress = self.path.suffix == ".gz"
        if self.path.exists():
            raise TraceIntegrityError(
                f"refusing to overwrite existing trace shard {self.path}"
            )
        raw = open(self.path, "xb")  # noqa: SIM115 - lifetime is the writer's
        if self.compress:
            self._fh = io.TextIOWrapper(
                gzip.GzipFile(fileobj=raw, mode="wb", mtime=0), encoding="utf-8"
            )
        else:
            self._fh = io.TextIOWrapper(raw, encoding="utf-8")
        self._chain = hashlib.sha256(SCHEMA_VERSION.encode()).hexdigest()
        self.append(
            "header",
            {
                "schema_version": SCHEMA_VERSION,
                "created_unix_ns": time.time_ns(),
                "meta": dict(self.meta),
            },
        )

    # ---- writing -------------------------------------------------------- #

    def append(self, kind: str, payload: Mapping[str, Any]) -> int:
        if self._fh is None:
            raise TraceIntegrityError("writer is closed")
        body = dict(payload)
        seq = self._seq
        base = canonical_record_json({"seq": seq, "kind": kind, **body})
        chain = hashlib.sha256((self._chain + base).encode("utf-8")).hexdigest()
        line = canonical_record_json({"seq": seq, "kind": kind, "chain": chain, **body})
        self._fh.write(line)
        self._fh.write("\n")
        self._bytes += len(line) + 1
        self._chain = chain
        self._seq += 1
        return seq

    def revision(self, rev: CanvasRevision) -> int:
        return self.append("revision", rev.to_json())

    def revisions(self, revs: Iterable[CanvasRevision]) -> int:
        n = 0
        for r in revs:
            self.revision(r)
            n += 1
        return n

    def flush(self) -> None:
        if self._fh is not None:
            self._fh.flush()

    def close(self) -> Mapping[str, Any]:
        if self._fh is None:
            return self.read_meta()
        self.append("footer", {"records": self._seq, "closed_unix_ns": time.time_ns()})
        self._fh.flush()
        self._fh.close()
        self._fh = None
        digest = _file_sha256(self.path)
        meta = {
            "schema_version": SCHEMA_VERSION,
            "shard": self.path.name,
            "records": self._seq,
            "final_chain": self._chain,
            "sha256": digest,
            "bytes": self.path.stat().st_size,
            "meta": dict(self.meta),
        }
        self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        # Make the shard read-only: it is an append-only scientific artifact.
        try:
            os.chmod(self.path, 0o444)
        except OSError:
            pass
        return meta

    def __enter__(self) -> TraceWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def meta_path(self) -> Path:
        return self.path.with_suffix(self.path.suffix + ".meta.json")

    def read_meta(self) -> Mapping[str, Any]:
        if self.meta_path.exists():
            return json.loads(self.meta_path.read_text())
        return {}

    @property
    def records_written(self) -> int:
        return self._seq


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class TraceReader:
    """Reads and verifies a hash-chained shard."""

    def __init__(self, path: str | os.PathLike[str]) -> None:
        self.path = Path(path)

    def _open(self) -> Any:
        if self.path.suffix == ".gz":
            return io.TextIOWrapper(gzip.open(self.path, "rb"), encoding="utf-8")
        return open(self.path, encoding="utf-8")  # noqa: SIM115

    def records(self) -> Iterator[TraceRecord]:
        with self._open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                seq = int(d.pop("seq"))
                kind = str(d.pop("kind"))
                chain = str(d.pop("chain"))
                yield TraceRecord(seq=seq, kind=kind, chain=chain, payload=d)

    def verify(self) -> dict[str, Any]:
        chain = hashlib.sha256(SCHEMA_VERSION.encode()).hexdigest()
        n = 0
        for rec in self.records():
            if rec.seq != n:
                raise TraceIntegrityError(f"{self.path}: seq gap at {rec.seq} (expected {n})")
            base = canonical_record_json({"seq": rec.seq, "kind": rec.kind, **rec.payload})
            expect = hashlib.sha256((chain + base).encode("utf-8")).hexdigest()
            if expect != rec.chain:
                raise TraceIntegrityError(f"{self.path}: chain mismatch at seq {rec.seq}")
            chain = expect
            n += 1
        meta_path = self.path.with_suffix(self.path.suffix + ".meta.json")
        out = {"records": n, "final_chain": chain, "sha256": _file_sha256(self.path)}
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            if meta.get("final_chain") and meta["final_chain"] != chain:
                raise TraceIntegrityError(f"{self.path}: sidecar chain mismatch")
            if meta.get("sha256") and meta["sha256"] != out["sha256"]:
                raise TraceIntegrityError(f"{self.path}: sidecar digest mismatch")
            if meta.get("records") is not None and int(meta["records"]) != n:
                raise TraceIntegrityError(f"{self.path}: sidecar record count mismatch")
            out["meta"] = meta.get("meta", {})
        return out

    def by_kind(self, kind: str) -> Iterator[dict[str, Any]]:
        for rec in self.records():
            if rec.kind == kind:
                yield rec.payload

    def revisions(self) -> Iterator[CanvasRevision]:
        for p in self.by_kind("revision"):
            yield CanvasRevision.from_json(p)

    def header(self) -> dict[str, Any]:
        for rec in self.records():
            if rec.kind == "header":
                return rec.payload
        return {}


def shard_manifest(directory: str | os.PathLike[str]) -> dict[str, Any]:
    """Verify every shard in a directory and return a checksum manifest."""
    d = Path(directory)
    shards: dict[str, Any] = {}
    for p in sorted(d.glob("*.jsonl")) + sorted(d.glob("*.jsonl.gz")):
        shards[p.name] = TraceReader(p).verify()
    return {"schema_version": SCHEMA_VERSION, "directory": str(d), "shards": shards}
