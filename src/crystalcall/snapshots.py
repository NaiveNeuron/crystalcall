"""Immutable, content-addressed snapshots for CrystalCall / DenoisePTC.

Speculation is only *lossless* if a speculative execution and the later
authoritative execution observe byte-identical state.  This module supplies the
only mechanism CrystalCall trusts for that: a frozen, read-only, content-addressed
snapshot whose identifier is a pure function of its contents.

Snapshot id derivation (identical for every kind)::

    entries  = sorted (name, sha256_hex(content)) pairs, ordered by name
    manifest = canonical JSON {"kind": …, "entries": [[name, hash], …],
                               "extra": {…}}          (sorted keys, no spaces)
    digest   = sha256(manifest.encode("utf-8")).hexdigest()
    id       = "<prefix>@" + digest[:16]

``name`` is a POSIX relative path for repo snapshots, a document id for sqlite
and embedding snapshots.  ``extra`` carries the small amount of build metadata
that changes semantics (the embedding model id and dimension, the sqlite schema
version).  Nothing derived from wall-clock time, hostname, inode numbers, file
modes or mtimes ever enters the digest, so two builds of identical content on
different machines produce the same snapshot id.

On disk a snapshot is made read-only (files ``0444``, directories ``0555``) and
carries a ``SNAPSHOT.json`` sidecar describing itself.  ``verify_snapshot``
recomputes the content digest and raises :class:`SnapshotDriftError` on any
drift; a silently mutated snapshot would invalidate every equivalence claim in
the paper, so drift is always fatal rather than a warning.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "SnapshotSpec",
    "SnapshotDriftError",
    "SNAPSHOT_SIDECAR",
    "TEXT_SUFFIXES",
    "EMBED_DIM",
    "EMBED_WINDOW_TOKENS",
    "EMBED_TARGET_ROWS",
    "DEFAULT_EMBED_MODEL",
    "SQLITE_SCHEMA_VERSION",
    "load_embedding_matrix",
    "build_repo_snapshot",
    "build_sqlite_fts_snapshot",
    "build_embedding_snapshot",
    "build_corpus_snapshot",
    "load_snapshot",
    "verify_snapshot",
    "snapshot_permissions_ok",
    "sub_snapshot",
    "embed_text",
    "iter_corpus_files",
    "sha256_file",
    "sha256_bytes",
    "make_read_only",
    "make_writable",
]

#: Sidecar file name written into every snapshot directory.
SNAPSHOT_SIDECAR = "SNAPSHOT.json"

#: Files that participate in a corpus snapshot.  Everything else (binaries,
#: caches, compiled artefacts) is excluded so the corpus stays text-only and
#: therefore identically searchable by ripgrep and the pure-Python fallback.
TEXT_SUFFIXES: tuple[str, ...] = (".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yml", ".yaml")

#: Directory names never copied into a snapshot.
_SKIP_DIRS = frozenset({"__pycache__", ".git", ".venv", ".mypy_cache", ".pytest_cache", ".ruff_cache"})

#: Pseudo-embedding width.  Sized (together with :data:`EMBED_TARGET_ROWS`) so
#: the frozen matrix is a few hundred MB of float32 and one ``embed_search``
#: invocation is a real accelerator load rather than pure launch overhead.
EMBED_DIM = 768

#: Token window used to cut corpus documents into retrievable passages.
EMBED_WINDOW_TOKENS = 32

#: Target passage count.  The window stride is derived from the corpus token
#: count so this target is approached without ever depending on wall-clock,
#: machine or ordering state.
EMBED_TARGET_ROWS = 170_000

#: Guard rails on the produced matrix size, in bytes.
EMBED_MIN_BYTES = 350_000_000
EMBED_MAX_BYTES = 900_000_000

DEFAULT_EMBED_MODEL = "hashed-pseudo-embed-v1"

SQLITE_SCHEMA_VERSION = 1

_MAX_DOC_BYTES = 1 << 20  # 1 MiB: keeps the sqlite/embedding snapshots bounded


class SnapshotDriftError(RuntimeError):
    """A snapshot's on-disk content no longer matches its recorded digest."""


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    """An immutable, content-addressed snapshot.

    ``snapshot_id`` is the value bound into every ``SnapshotRead`` canonical
    call identity; ``digest`` is the full sha256 the id is truncated from.
    """

    snapshot_id: str
    root: Path
    kind: str
    digest: str
    meta: dict = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        """The kind-specific part of the id, without the ``@digest`` suffix."""
        return self.snapshot_id.rsplit("@", 1)[0]

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "kind": self.kind,
            "digest": self.digest,
            "meta": self.meta,
        }


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _digest_entries(
    kind: str,
    entries: Sequence[tuple[str, str]],
    extra: Mapping[str, Any] | None = None,
) -> str:
    """The one content-digest function.  See the module docstring."""
    ordered = sorted(entries, key=lambda kv: kv[0])
    names = [n for n, _ in ordered]
    if len(set(names)) != len(names):
        dupes = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"duplicate snapshot entry names: {dupes}")
    manifest = _canonical_json(
        {
            "kind": kind,
            "entries": [[n, h] for n, h in ordered],
            "extra": dict(extra or {}),
        }
    )
    return sha256_bytes(manifest.encode("utf-8"))


def _snapshot_id(prefix: str, digest: str) -> str:
    return f"{prefix}@{digest[:16]}"


# --------------------------------------------------------------------------- #
# read-only enforcement
# --------------------------------------------------------------------------- #


def make_read_only(root: Path) -> None:
    """Files ``0444``, directories ``0555``, deepest first."""
    paths = sorted((p for p in root.rglob("*")), key=lambda p: len(p.parts), reverse=True)
    for p in paths:
        if p.is_symlink():
            continue
        os.chmod(p, 0o555 if p.is_dir() else 0o444)
    os.chmod(root, 0o555)


def make_writable(root: Path, *, recursive: bool = True) -> None:
    """Undo :func:`make_read_only` so a snapshot directory can be rebuilt.

    ``recursive=False`` unlocks only ``root`` itself, which is what a *container*
    directory needs: unlocking it recursively would silently un-freeze every
    snapshot living inside it.
    """
    if not root.exists():
        return
    os.chmod(root, 0o755)
    if not recursive:
        return
    for p in sorted(root.rglob("*"), key=lambda p: len(p.parts)):
        if p.is_symlink():
            continue
        os.chmod(p, 0o755 if p.is_dir() else 0o644)


def _reset_dir(dest: Path) -> None:
    make_writable(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)


def _write_sidecar(spec: SnapshotSpec) -> None:
    path = spec.root / SNAPSHOT_SIDECAR
    if path.exists():
        os.chmod(path, 0o644)
    path.write_text(_canonical_json(spec.to_json()) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# corpus enumeration
# --------------------------------------------------------------------------- #


def iter_corpus_files(root: Path, suffixes: Iterable[str] = TEXT_SUFFIXES) -> Iterator[Path]:
    """Deterministic, sorted walk of the text files under ``root``.

    Symlinks, caches and non-text suffixes are skipped.  Yields absolute paths
    in sorted POSIX-relative order.
    """
    wanted = tuple(suffixes)
    root = root.resolve()
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in _SKIP_DIRS and not d.startswith("."))
        for name in sorted(filenames):
            if name == SNAPSHOT_SIDECAR:
                continue
            if not name.endswith(wanted):
                continue
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            out.append(p)
    out.sort(key=lambda p: p.relative_to(root).as_posix())
    yield from out


def _doc_id(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_doc(path: Path) -> str | None:
    """Read one corpus document as text, or ``None`` if it is not usable."""
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if len(raw) > _MAX_DOC_BYTES:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


# --------------------------------------------------------------------------- #
# repo snapshot
# --------------------------------------------------------------------------- #


def build_repo_snapshot(src: Path, dest: Path) -> SnapshotSpec:
    """Freeze the text files under ``src`` into a read-only snapshot at ``dest``."""
    src = Path(src).resolve()
    dest = Path(dest)
    if not src.is_dir():
        raise NotADirectoryError(f"repo snapshot source is not a directory: {src}")
    _reset_dir(dest)
    dest = dest.resolve()

    entries: list[tuple[str, str]] = []
    total_lines = 0
    total_bytes = 0
    for path in iter_corpus_files(src):
        text = _read_doc(path)
        if text is None:
            continue
        rel = _doc_id(path, src)
        out_path = dest / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        data = text.encode("utf-8")
        out_path.write_bytes(data)
        entries.append((rel, sha256_bytes(data)))
        total_lines += text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        total_bytes += len(data)

    if not entries:
        raise ValueError(f"repo snapshot source contained no text files: {src}")

    digest = _digest_entries("repo", entries)
    spec = SnapshotSpec(
        snapshot_id=_snapshot_id("repo", digest),
        root=dest,
        kind="repo",
        digest=digest,
        meta={
            "files": len(entries),
            "lines": total_lines,
            "bytes": total_bytes,
            "suffixes": list(TEXT_SUFFIXES),
        },
    )
    _write_sidecar(spec)
    make_read_only(dest)
    return spec


def _repo_entries(root: Path) -> list[tuple[str, str]]:
    return [(_doc_id(p, root), sha256_file(p)) for p in iter_corpus_files(root)]


# --------------------------------------------------------------------------- #
# sqlite FTS5 snapshot
# --------------------------------------------------------------------------- #

_FTS_SCHEMA = (
    "CREATE VIRTUAL TABLE docs USING fts5(path, body, tokenize='unicode61')",
    "CREATE TABLE doc_meta ("
    " path TEXT PRIMARY KEY NOT NULL,"
    " sha256 TEXT NOT NULL,"
    " lines INTEGER NOT NULL,"
    " bytes INTEGER NOT NULL"
    ")",
)


def build_sqlite_fts_snapshot(src_corpus: Path, dest_db: Path) -> SnapshotSpec:
    """Build a frozen, read-only SQLite FTS5 index over ``src_corpus``.

    ``dest_db`` is the database file; the ``SNAPSHOT.json`` sidecar is written
    into its parent directory, which becomes the snapshot root.
    """
    src_corpus = Path(src_corpus).resolve()
    dest_db = Path(dest_db)
    root = dest_db.parent
    _reset_dir(root)
    root = root.resolve()
    dest_db = root / dest_db.name

    docs: list[tuple[str, str, str, int, int]] = []
    for path in iter_corpus_files(src_corpus):
        text = _read_doc(path)
        if text is None:
            continue
        rel = _doc_id(path, src_corpus)
        data = text.encode("utf-8")
        lines = text.count("\n") + (0 if text.endswith("\n") or not text else 1)
        docs.append((rel, text, sha256_bytes(data), lines, len(data)))
    if not docs:
        raise ValueError(f"sqlite snapshot source contained no text files: {src_corpus}")
    docs.sort(key=lambda row: row[0])

    conn = sqlite3.connect(dest_db)
    try:
        conn.execute("PRAGMA journal_mode=DELETE")
        for ddl in _FTS_SCHEMA:
            conn.execute(ddl)
        conn.executemany(
            "INSERT INTO docs(rowid, path, body) VALUES (?, ?, ?)",
            [(i + 1, rel, text) for i, (rel, text, _, _, _) in enumerate(docs)],
        )
        conn.executemany(
            "INSERT INTO doc_meta(path, sha256, lines, bytes) VALUES (?, ?, ?, ?)",
            [(rel, sha, lines, nbytes) for rel, _, sha, lines, nbytes in docs],
        )
        conn.commit()
        conn.execute("INSERT INTO docs(docs) VALUES('optimize')")
        conn.commit()
        conn.execute("VACUUM")
        conn.commit()
    finally:
        conn.close()

    digest = _digest_entries(
        "sqlite",
        [(rel, sha) for rel, _, sha, _, _ in docs],
        extra={"schema_version": SQLITE_SCHEMA_VERSION, "database": dest_db.name},
    )
    spec = SnapshotSpec(
        snapshot_id=_snapshot_id(f"sqlite:{dest_db.stem}", digest),
        root=root,
        kind="sqlite",
        digest=digest,
        meta={
            "database": dest_db.name,
            "documents": len(docs),
            "lines": sum(row[3] for row in docs),
            "bytes": sum(row[4] for row in docs),
            "schema_version": SQLITE_SCHEMA_VERSION,
            "tables": ["docs", "doc_meta"],
        },
    )
    _write_sidecar(spec)
    make_read_only(root)
    return spec


def _sqlite_entries(spec: SnapshotSpec) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    db = spec.root / str(spec.meta["database"])
    uri = f"file:{db.as_posix()}?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute("SELECT path, sha256 FROM doc_meta ORDER BY path").fetchall()
        body = conn.execute("SELECT path, body FROM docs ORDER BY path").fetchall()
    finally:
        conn.close()
    recorded = {p: s for p, s in rows}
    actual = {p: sha256_bytes(b.encode("utf-8")) for p, b in body}
    if recorded != actual:
        bad = sorted(k for k in set(recorded) | set(actual) if recorded.get(k) != actual.get(k))
        raise SnapshotDriftError(
            f"{spec.snapshot_id}: FTS body no longer matches doc_meta sha256 for {bad[:8]}"
        )
    extra = {
        "schema_version": int(spec.meta["schema_version"]),
        "database": str(spec.meta["database"]),
    }
    return [(p, s) for p, s in rows], extra


# --------------------------------------------------------------------------- #
# embedding snapshot
# --------------------------------------------------------------------------- #

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def embed_text(text: str, *, dim: int = EMBED_DIM, model_id: str = DEFAULT_EMBED_MODEL) -> np.ndarray:
    """Deterministic hashed pseudo-embedding.

    This is a *reproducible pseudo-embedder*, not a learned model: each token is
    hashed with keyed BLAKE2b into a signed bucket (a signed random-projection /
    "hashing trick" sketch), the counts are L2-normalised and cast to float32.
    It has no semantic quality guarantees whatsoever.  CrystalCall uses it
    because this experiment measures *scheduling and resource contention* on a
    GPU-class retrieval op, not retrieval accuracy, and because it is exactly
    reproducible offline on any machine with no model download.
    """
    person = model_id.encode("utf-8")[:16]
    acc = np.zeros(dim, dtype=np.float64)
    for tok in _TOKEN_RE.findall(text.lower()):
        h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8, person=person).digest()
        n = int.from_bytes(h, "big")
        acc[n % dim] += 1.0 if (n >> 63) & 1 else -1.0
    norm = float(np.linalg.norm(acc))
    if norm > 0.0:
        acc /= norm
    return acc.astype(np.float32)


def _token_maps(tokens: Sequence[str], dim: int, model_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised (bucket, sign) lookup for a token list.

    Identical, by construction, to what :func:`embed_text` computes token by
    token: the unique tokens are hashed once and the per-occurrence bucket and
    sign are gathered.  Signed integer counts summed in ``float64`` are exact,
    so batching cannot change a single bit of the result.
    """
    person = model_id.encode("utf-8")[:16]
    uniq, inverse = np.unique(np.asarray(tokens, dtype=object), return_inverse=True)
    buckets = np.empty(uniq.size, dtype=np.int64)
    signs = np.empty(uniq.size, dtype=np.float64)
    for i, tok in enumerate(uniq):
        h = hashlib.blake2b(str(tok).encode("utf-8"), digest_size=8, person=person).digest()
        n = int.from_bytes(h, "big")
        buckets[i] = n % dim
        signs[i] = 1.0 if (n >> 63) & 1 else -1.0
    return buckets[inverse], signs[inverse]


def _passages(
    src_corpus: Path, window: int, target_rows: int
) -> tuple[list[str], list[tuple[int, int, int]], list[list[str]]]:
    """Cut the corpus into deterministic token windows.

    Returns ``(ids, spans, doc_tokens)`` where ``spans`` is a list of
    ``(doc_index, start, end)`` token ranges.  The stride is derived from the
    corpus token count, so it is a pure function of the corpus content.
    """
    doc_ids: list[str] = []
    doc_tokens: list[list[str]] = []
    for path in iter_corpus_files(src_corpus):
        text = _read_doc(path)
        if text is None:
            continue
        toks = _TOKEN_RE.findall(text.lower())
        if not toks:
            continue
        doc_ids.append(_doc_id(path, src_corpus))
        doc_tokens.append(toks)
    if not doc_ids:
        raise ValueError(f"embedding snapshot source contained no text files: {src_corpus}")
    order = sorted(range(len(doc_ids)), key=lambda i: doc_ids[i])
    doc_ids = [doc_ids[i] for i in order]
    doc_tokens = [doc_tokens[i] for i in order]

    total = sum(len(t) for t in doc_tokens)
    stride = max(1, (total + target_rows // 2) // target_rows)

    ids: list[str] = []
    spans: list[tuple[int, int, int]] = []
    for d, toks in enumerate(doc_tokens):
        for start in range(0, len(toks), stride):
            end = min(start + window, len(toks))
            ids.append(f"{doc_ids[d]}#{start:08d}")
            spans.append((d, start, end))
    return ids, spans, doc_tokens


def build_embedding_snapshot(
    src_corpus: Path,
    dest: Path,
    model_id: str = DEFAULT_EMBED_MODEL,
    *,
    dim: int = EMBED_DIM,
    window: int = EMBED_WINDOW_TOKENS,
    target_rows: int = EMBED_TARGET_ROWS,
    chunk_rows: int = 8192,
    min_bytes: int = EMBED_MIN_BYTES,
    max_bytes: int = EMBED_MAX_BYTES,
) -> SnapshotSpec:
    """Precompute passage pseudo-embeddings for ``src_corpus``.

    Writes ``embeddings.f32`` (a raw row-major ``float32`` blob of shape
    ``(rows, dim)`` with L2-normalised rows, in sorted passage-id order) and
    ``ids.json`` (the matching id list).  A raw blob rather than ``.npy`` so the
    CUDA ``embed_search`` kernel can ``mmap`` it with no header parsing and the
    numpy fallback can ``memmap`` the identical bytes.
    """
    src_corpus = Path(src_corpus).resolve()
    dest = Path(dest)
    _reset_dir(dest)
    dest = dest.resolve()

    ids, spans, doc_tokens = _passages(src_corpus, window, target_rows)
    rows = len(ids)
    nbytes = rows * dim * 4
    if not (min_bytes <= nbytes <= max_bytes):
        raise ValueError(
            f"embedding matrix would be {nbytes} bytes ({rows} rows x {dim} dim), outside "
            f"[{min_bytes}, {max_bytes}]; adjust EMBED_TARGET_ROWS/EMBED_DIM"
        )

    per_doc = [_token_maps(t, dim, model_id) for t in doc_tokens]

    matrix_path = dest / "embeddings.f32"
    out = np.memmap(matrix_path, dtype=np.float32, mode="w+", shape=(rows, dim))
    for lo in range(0, rows, chunk_rows):
        hi = min(lo + chunk_rows, rows)
        block = np.zeros((hi - lo, dim), dtype=np.float64)
        for i in range(lo, hi):
            d, start, end = spans[i]
            buckets, signs = per_doc[d]
            np.add.at(block[i - lo], buckets[start:end], signs[start:end])
        norms = np.linalg.norm(block, axis=1)
        np.divide(block, norms[:, None], out=block, where=norms[:, None] > 0.0)
        out[lo:hi] = block.astype(np.float32)
    out.flush()
    del out

    (dest / "ids.json").write_text(_canonical_json(ids) + "\n", encoding="utf-8")

    entries = [
        ("embeddings.f32", sha256_file(matrix_path)),
        ("ids.json", sha256_file(dest / "ids.json")),
    ]
    extra = {
        "model_id": model_id,
        "dim": dim,
        "rows": rows,
        "window": window,
        "dtype": "float32",
    }
    digest = _digest_entries("embed", entries, extra=extra)
    spec = SnapshotSpec(
        snapshot_id=_snapshot_id("emb:corpus", digest),
        root=dest,
        kind="embed",
        digest=digest,
        meta={
            "model_id": model_id,
            "dim": dim,
            "rows": rows,
            "window": window,
            "dtype": "float32",
            "bytes": nbytes,
            "documents": len(doc_tokens),
            "matrix": "embeddings.f32",
            "ids": "ids.json",
            "embedder": "hashed pseudo-embedding (signed BLAKE2b hashing trick) over "
            f"{window}-token passage windows; reproducible, not semantic",
        },
    )
    _write_sidecar(spec)
    make_read_only(dest)
    return spec


def load_embedding_matrix(spec: SnapshotSpec) -> np.ndarray:
    """Memory-map the frozen embedding matrix as a read-only float32 view."""
    rows = int(spec.meta["rows"])
    dim = int(spec.meta["dim"])
    return np.memmap(
        spec.root / str(spec.meta["matrix"]), dtype=np.float32, mode="r", shape=(rows, dim)
    )


def _embed_entries(spec: SnapshotSpec) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    matrix_path = spec.root / str(spec.meta["matrix"])
    ids_path = spec.root / str(spec.meta["ids"])
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    rows = int(spec.meta["rows"])
    dim = int(spec.meta["dim"])
    if len(ids) != rows:
        raise SnapshotDriftError(f"{spec.snapshot_id}: {len(ids)} ids for {rows} rows")
    if matrix_path.stat().st_size != rows * dim * 4:
        raise SnapshotDriftError(
            f"{spec.snapshot_id}: matrix is {matrix_path.stat().st_size} bytes, "
            f"expected {rows * dim * 4}"
        )
    entries = [
        ("embeddings.f32", sha256_file(matrix_path)),
        ("ids.json", sha256_file(ids_path)),
    ]
    extra = {
        "model_id": str(spec.meta["model_id"]),
        "dim": dim,
        "rows": rows,
        "window": int(spec.meta["window"]),
        "dtype": "float32",
    }
    return entries, extra


# --------------------------------------------------------------------------- #
# composite corpus snapshot
# --------------------------------------------------------------------------- #


def build_corpus_snapshot(
    root: Path,
    repo: SnapshotSpec,
    sqlite_spec: SnapshotSpec,
    embed: SnapshotSpec,
) -> SnapshotSpec:
    """Bind the three sub-snapshots into one composite corpus snapshot.

    The composite id is a pure function of the three sub-digests, so a change to
    any sub-snapshot changes the corpus id (and therefore every speculative call
    identity derived from it).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    subs = {"repo": repo, "sqlite": sqlite_spec, "embed": embed}
    entries = [(name, spec.digest) for name, spec in sorted(subs.items())]
    digest = _digest_entries("corpus", entries)
    root = root.resolve()
    spec = SnapshotSpec(
        snapshot_id=_snapshot_id("corpus", digest),
        root=root,
        kind="corpus",
        digest=digest,
        meta={
            "python_version": sys.version.split()[0],
            "sub": {
                name: {
                    "snapshot_id": sub.snapshot_id,
                    "kind": sub.kind,
                    "digest": sub.digest,
                    "path": sub.root.relative_to(root).as_posix()
                    if sub.root != root
                    else ".",
                    "meta": sub.meta,
                }
                for name, sub in sorted(subs.items())
            },
        },
    )
    _write_sidecar(spec)
    os.chmod(root, 0o555)
    return spec


def sub_snapshot(spec: SnapshotSpec, name: str) -> SnapshotSpec:
    """Reconstruct one sub-:class:`SnapshotSpec` of a composite corpus snapshot."""
    if spec.kind != "corpus":
        raise ValueError(f"{spec.snapshot_id}: not a composite corpus snapshot")
    sub = spec.meta["sub"][name]
    return SnapshotSpec(
        snapshot_id=str(sub["snapshot_id"]),
        root=(spec.root / str(sub["path"])).resolve(),
        kind=str(sub["kind"]),
        digest=str(sub["digest"]),
        meta=dict(sub["meta"]),
    )


# --------------------------------------------------------------------------- #
# load / verify
# --------------------------------------------------------------------------- #


def load_snapshot(path: Path) -> SnapshotSpec:
    """Read ``<path>/SNAPSHOT.json`` (or the sidecar itself) into a spec."""
    path = Path(path)
    sidecar = path if path.is_file() else path / SNAPSHOT_SIDECAR
    if not sidecar.is_file():
        raise FileNotFoundError(f"no {SNAPSHOT_SIDECAR} at {path}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    return SnapshotSpec(
        snapshot_id=str(payload["snapshot_id"]),
        root=sidecar.parent.resolve(),
        kind=str(payload["kind"]),
        digest=str(payload["digest"]),
        meta=dict(payload.get("meta") or {}),
    )


def verify_snapshot(spec: SnapshotSpec, *, strict: bool = True) -> bool:
    """Recompute the content digest and compare it with the recorded one.

    Returns ``True`` when the snapshot is intact.  On drift it raises
    :class:`SnapshotDriftError` (``strict=True``, the default) or returns
    ``False``.  A snapshot that has silently changed invalidates every
    speculative-equivalence claim built on it, so drift is fatal by default.
    """
    try:
        if spec.kind == "repo":
            digest = _digest_entries("repo", _repo_entries(spec.root))
        elif spec.kind == "sqlite":
            entries, extra = _sqlite_entries(spec)
            digest = _digest_entries("sqlite", entries, extra=extra)
        elif spec.kind == "embed":
            entries, extra = _embed_entries(spec)
            digest = _digest_entries("embed", entries, extra=extra)
        elif spec.kind == "corpus":
            subs = {name: sub_snapshot(spec, name) for name in sorted(spec.meta["sub"])}
            for name, sub in subs.items():
                if not verify_snapshot(sub, strict=strict):
                    return False
            digest = _digest_entries(
                "corpus", [(name, sub.digest) for name, sub in sorted(subs.items())]
            )
        else:
            raise ValueError(f"unknown snapshot kind {spec.kind!r}")
    except SnapshotDriftError:
        if strict:
            raise
        return False

    if digest == spec.digest and _snapshot_id(spec.prefix, digest) == spec.snapshot_id:
        return True
    if strict:
        raise SnapshotDriftError(
            f"{spec.snapshot_id} at {spec.root}: recomputed digest {digest} != recorded {spec.digest}"
        )
    return False


def snapshot_permissions_ok(spec: SnapshotSpec) -> bool:
    """True when every file is ``0444`` and every directory ``0555``."""
    if spec.kind == "corpus":
        return all(
            snapshot_permissions_ok(sub_snapshot(spec, name)) for name in sorted(spec.meta["sub"])
        )
    for p in [spec.root, *sorted(spec.root.rglob("*"))]:
        if p.is_symlink():
            continue
        mode = stat.S_IMODE(p.stat().st_mode)
        if mode != (0o555 if p.is_dir() else 0o444):
            return False
    return True
