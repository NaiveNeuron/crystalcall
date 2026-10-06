"""Tool construction: declarations -> ``EffectRegistry`` + executable impls.

Two families of tools live here.

**BFCL-style declarations.**  :func:`build_registry` turns serialisable
``ToolDecl`` records (see ``crystalcall.manifest``) into real
:class:`~crystalcall.effects.ToolSpec` objects, and :func:`build_impls` gives
each one a deterministic ``Pure`` stub whose value is a pure function of the
canonical arguments.  The stub never touches the filesystem, the clock or the
network, so a speculative execution and the authoritative execution of the same
canonical call are bit-identical by construction.  An optional interruptible
sleep (``CRYSTALCALL_TOOL_SLEEP_MS``) lets the end-to-end harness sweep the
100/500/2000/5000 ms latency conditions without perturbing tool *identity* or
tool *value*.

**The three real local tool families.**  ``search_repo``, ``sqlite_query``,
``analyze_module`` and ``embed_search`` read a frozen, content-addressed
snapshot (see :mod:`crystalcall.snapshots`) and are therefore ``SnapshotRead``:
their result is a pure function of (canonical arguments, snapshot id).  Two
deliberately ineligible tools, ``write_note`` (``Write``) and
``read_live_clock`` (``LiveRead``), exist so the launch gate's abstention path
is exercised by something real.

``build_registry`` duck-types its declarations on ``.name``, ``.effect``,
``.resource``, ``.expected_duration_ms``, ``.params`` and ``.description``, so
this module never imports ``crystalcall.manifest`` and there is no cycle.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .canonicalize import canonical_json
from .effects import (
    EffectKind,
    EffectRegistry,
    ParamKind,
    ParamSpec,
    ResourceClass,
    ToolSpec,
)
from .snapshots import (
    EMBED_DIM,
    SnapshotSpec,
    embed_text,
    load_embedding_matrix,
    sub_snapshot,
)

__all__ = [
    "ToolImpl",
    "ToolDeclLike",
    "build_registry",
    "build_impls",
    "local_registry",
    "local_impls",
    "LOCAL_TOOL_DECLS",
    "LOCAL_TOOL_NAMES",
    "SPECULATABLE_LOCAL_TOOLS",
    "INELIGIBLE_LOCAL_TOOLS",
    "SLEEP_ENV_VAR",
    "sleep_ms_from_env",
    "SqlRejected",
    "PathEscapesSnapshot",
    "analyze_source",
    "validate_read_only_sql",
    "resolve_ripgrep",
    "resolve_embed_binary",
    "RIPGREP_ENV_VAR",
    "VENDORED_RG",
    "CUDA_EMBED_ENV_VAR",
    "CUDA_EMBED_BINARY",
    "CUDA_EMBED_SOURCE",
    "ANALYZE_CHECKS",
]

#: A tool implementation: canonical arguments + a cancellation event -> value.
ToolImpl = Callable[[Mapping[str, Any], threading.Event], Any]

#: Anything with the six ``ToolDecl`` attributes.
ToolDeclLike = Any

#: Environment variable used by the deterministic-duration harness.
SLEEP_ENV_VAR = "CRYSTALCALL_TOOL_SLEEP_MS"

#: Where ``write_note`` is allowed to write (never inside a snapshot).
SCRATCH_ENV_VAR = "CRYSTALCALL_SCRATCH_DIR"

_PARAM_FIELDS = frozenset(
    {"name", "kind", "required", "default", "order_sensitive", "element_kind", "path_root"}
)


class SqlRejected(ValueError):
    """A SQL string failed the read-only single-``SELECT`` gate."""


class PathEscapesSnapshot(ValueError):
    """A path argument resolved outside the frozen snapshot root."""


# --------------------------------------------------------------------------- #
# declaration -> spec
# --------------------------------------------------------------------------- #


def _param_spec(raw: Mapping[str, Any] | ParamSpec) -> ParamSpec:
    if isinstance(raw, ParamSpec):
        return raw
    unknown = sorted(set(raw) - _PARAM_FIELDS)
    if unknown:
        raise ValueError(f"unknown ParamSpec field(s) {unknown} in {dict(raw)!r}")
    element = raw.get("element_kind")
    return ParamSpec(
        name=str(raw["name"]),
        kind=raw["kind"] if isinstance(raw["kind"], ParamKind) else ParamKind(raw["kind"]),
        required=bool(raw.get("required", True)),
        default=raw.get("default"),
        order_sensitive=bool(raw.get("order_sensitive", True)),
        # Per-parameter ``path_root`` wins over the registry-level root; see
        # ``canonicalize._norm_param`` (``spec.path_root or registry_root``).
        path_root=None if raw.get("path_root") is None else str(raw["path_root"]),
        element_kind=None
        if element is None
        else (element if isinstance(element, ParamKind) else ParamKind(element)),
    )


def _resolver_for(
    name: str, snapshot_id: str | Mapping[str, str] | None
) -> Callable[[Mapping[str, Any]], str]:
    """A snapshot resolver bound to a fixed id via closure.

    The resolver ignores the arguments: a frozen snapshot is bound once, for the
    whole run, so *every* call against it shares one immutable id.  It is
    deterministic and raises rather than inventing an id.
    """
    if snapshot_id is None:
        raise ValueError(
            f"tool {name!r} is SnapshotRead but no snapshot_id was supplied to build_registry"
        )
    if isinstance(snapshot_id, str):
        bound = snapshot_id
    else:
        if name not in snapshot_id:
            raise ValueError(f"tool {name!r} is SnapshotRead but snapshot_id map has no entry")
        bound = str(snapshot_id[name])
    if not bound:
        raise ValueError(f"tool {name!r}: empty snapshot id")

    def resolve(_args: Mapping[str, Any]) -> str:
        return bound

    return resolve


def build_registry(
    decls: Sequence[ToolDeclLike],
    *,
    path_root: str | None = None,
    snapshot_id: str | Mapping[str, str] | None = None,
    concurrency_limits: Mapping[str, int] | None = None,
) -> EffectRegistry:
    """Build an :class:`EffectRegistry` from serialisable tool declarations.

    ``snapshot_id`` binds ``SnapshotRead`` tools: either one id for all of them
    or a per-tool mapping.  ``Pure`` declarations (the BFCL surface) need none.
    """
    limits = dict(concurrency_limits or {})
    specs: list[ToolSpec] = []
    for decl in decls:
        name = str(decl.name)
        effect = decl.effect if isinstance(decl.effect, EffectKind) else EffectKind(decl.effect)
        resource = (
            decl.resource
            if isinstance(decl.resource, ResourceClass)
            else ResourceClass(decl.resource)
        )
        params = tuple(_param_spec(p) for p in decl.params)
        specs.append(
            ToolSpec(
                name=name,
                effect=effect,
                resource=resource,
                params=params,
                expected_duration_ms=float(decl.expected_duration_ms),
                concurrency_limit=int(limits.get(name, getattr(decl, "concurrency_limit", 1) or 1)),
                # BFCL tool names are dotted (``math.gcd``); the manifest layer
                # registers a sanitised identifier as the canonical name and the
                # original dotted spelling as an alias so both resolve to one
                # identity.
                aliases=frozenset(str(a) for a in (getattr(decl, "aliases", ()) or ())),
                snapshot_resolver=(
                    _resolver_for(name, snapshot_id)
                    if effect is EffectKind.SNAPSHOT_READ
                    else None
                ),
                description=str(getattr(decl, "description", "") or ""),
            )
        )
    return EffectRegistry(specs, path_root=path_root)


# --------------------------------------------------------------------------- #
# injected, interruptible, value-preserving latency
# --------------------------------------------------------------------------- #


def sleep_ms_from_env(env: Mapping[str, str] | None = None) -> float:
    """Read ``CRYSTALCALL_TOOL_SLEEP_MS``; 0.0 when unset/blank.

    Read at *call* time, never cached, so one process can sweep the latency
    conditions.  A malformed value is a hard error: a silently ignored latency
    condition would corrupt every timing result in the paper.
    """
    raw = (env if env is not None else os.environ).get(SLEEP_ENV_VAR, "")
    raw = raw.strip()
    if not raw:
        return 0.0
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{SLEEP_ENV_VAR}={raw!r} is not a number") from exc
    if value < 0.0:
        raise ValueError(f"{SLEEP_ENV_VAR}={raw!r} must be >= 0")
    return value


def _inject_sleep(cancel: threading.Event) -> None:
    """Interruptible sleep that never influences the returned value."""
    ms = sleep_ms_from_env()
    if ms > 0.0:
        cancel.wait(ms / 1000.0)


# --------------------------------------------------------------------------- #
# deterministic Pure stubs (BFCL surface)
# --------------------------------------------------------------------------- #


def _stub_impl(name: str) -> ToolImpl:
    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        # The payload must LOOK like a tool result, not like an echo of the
        # call.  DiffusionGemma reads an echo such as
        # `{"tool", "args", "ok", "canonical"}` as "the tool returned the call
        # itself, not a result", then issues a retry turn and recomputes the
        # answer by hand, which inflates the turn count and depresses BFCL
        # correctness for a purely harness-side reason.
        #
        # The value stays a deterministic function of the canonical call, so
        # result digests remain stable and order-independent, which is what the
        # claim/verify protocol needs.
        ordered = {k: args[k] for k in sorted(args)}
        canonical = canonical_json({"tool": name, "args": ordered})
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        _inject_sleep(cancel)
        # No "tool"/"args"/"canonical" keys and no tool name in the value: the
        # name-in-output pattern is part of what the model read as an echo.
        return {
            "ok": True,
            "result": {"value": digest[:12]},
            "deterministic": True,
        }

    return impl


def build_impls(
    decls: Sequence[ToolDeclLike],
    snapshot_root: Path | None = None,
) -> dict[str, ToolImpl]:
    """One implementation per declaration.

    A declaration naming one of the **local tool families** gets the real
    implementation, which needs ``snapshot_root`` (the composite corpus
    snapshot directory); asking for one without it raises rather than silently
    faking an effectful tool.

    Every other declaration is a **synthetic benchmark tool** -- BFCL's
    `update_user_info`, `math.gcd`, and so on. No real implementation exists or
    could exist for those, whatever effect the manifest rules assigned them, so
    they get the deterministic stub. The effect annotation still governs
    *speculation eligibility* through the registry, which is where it belongs:
    a synthetic `Write` remains permanently ineligible (constraint 9) and is
    simply executed by the stub when the authoritative program reaches it.
    """
    # Every local *family* name, not just the snapshot-backed four:
    # `write_note` and `read_live_clock` have real implementations too, and
    # stubbing them would be exactly the "silently faking an effectful tool"
    # this function refuses to do.
    local_names = {str(d.name) for d in LOCAL_TOOL_DECLS}
    wanted_local = sorted({str(d.name) for d in decls if str(d.name) in local_names})
    local: dict[str, ToolImpl] = {}
    if wanted_local:
        if snapshot_root is None:
            raise ValueError(
                f"local tools {wanted_local} need snapshot_root to bind a real "
                "implementation"
            )
        from .snapshots import load_snapshot

        local = local_impls(load_snapshot(Path(snapshot_root)))

    impls: dict[str, ToolImpl] = {}
    for decl in decls:
        name = str(decl.name)
        impls[name] = local[name] if name in local else _stub_impl(name)
    return impls


# --------------------------------------------------------------------------- #
# local tool declarations
# --------------------------------------------------------------------------- #


class _Decl:
    """Local stand-in with the same attribute surface as ``manifest.ToolDecl``.

    ``LOCAL_TOOL_DECLS`` is consumed through the duck-typed contract, so the
    local families stay usable before/without ``crystalcall.manifest``.
    """

    __slots__ = (
        "name",
        "effect",
        "resource",
        "expected_duration_ms",
        "params",
        "description",
        "concurrency_limit",
    )

    def __init__(
        self,
        name: str,
        effect: str,
        resource: str,
        expected_duration_ms: float,
        params: tuple[dict[str, Any], ...],
        description: str = "",
        concurrency_limit: int = 1,
    ) -> None:
        self.name = name
        self.effect = effect
        self.resource = resource
        self.expected_duration_ms = expected_duration_ms
        self.params = params
        self.description = description
        self.concurrency_limit = concurrency_limit

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "effect": self.effect,
            "resource": self.resource,
            "expected_duration_ms": self.expected_duration_ms,
            "params": [dict(p) for p in self.params],
            "description": self.description,
            "concurrency_limit": self.concurrency_limit,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Decl({self.name!r}, {self.effect!r})"


ANALYZE_CHECKS: tuple[str, ...] = ("complexity", "imports", "syntax")

#: Expected durations are placeholders here; the frozen, measured values live in
#: ``configs/tools.yml`` and are what the harness loads.  Keeping them in the
#: config (not the code) is what makes the numbers reproducible from data.
LOCAL_TOOL_DECLS: tuple[_Decl, ...] = (
    _Decl(
        name="search_repo",
        effect="SnapshotRead",
        resource="storage",
        expected_duration_ms=40.0,
        params=(
            {"name": "query", "kind": "str", "required": True},
            {"name": "path", "kind": "path", "required": False, "default": "."},
            {"name": "max_results", "kind": "int", "required": False, "default": 50},
        ),
        description=(
            "Literal (fixed-string) search for `query` in the frozen repository "
            "snapshot under `path`. Returns at most `max_results` matches sorted "
            "by (path, line)."
        ),
        concurrency_limit=4,
    ),
    _Decl(
        name="sqlite_query",
        effect="SnapshotRead",
        resource="storage",
        expected_duration_ms=15.0,
        params=(
            {"name": "database", "kind": "str", "required": True},
            {"name": "sql", "kind": "str", "required": True},
            {"name": "max_rows", "kind": "int", "required": False, "default": 100},
        ),
        description=(
            "Run one read-only SELECT against the frozen SQLite FTS5 snapshot "
            "(tables `docs(path, body)` and `doc_meta(path, sha256, lines, bytes)`). "
            "Writes, DDL, ATTACH, PRAGMA and multi-statement input are rejected."
        ),
        concurrency_limit=4,
    ),
    _Decl(
        name="analyze_module",
        effect="SnapshotRead",
        resource="cpu",
        expected_duration_ms=8.0,
        params=(
            {"name": "path", "kind": "path", "required": True},
            {
                "name": "checks",
                "kind": "list",
                "required": False,
                "default": list(ANALYZE_CHECKS),
                "order_sensitive": False,
                "element_kind": "str",
            },
        ),
        description=(
            "Static analysis of one file in the frozen repository snapshot using "
            "the stdlib `ast` module only: syntax validity, per-function branch "
            "count, sorted imports, def/class counts."
        ),
        concurrency_limit=8,
    ),
    _Decl(
        name="embed_search",
        effect="SnapshotRead",
        resource="gpu",
        expected_duration_ms=25.0,
        params=(
            {"name": "query", "kind": "str", "required": True},
            {"name": "top_k", "kind": "int", "required": False, "default": 5},
        ),
        description=(
            "Dense retrieval over the frozen embedding snapshot: embeds `query` "
            "with the snapshot's deterministic embedder and returns the `top_k` "
            "documents by cosine similarity."
        ),
        concurrency_limit=1,
    ),
    # --- deliberately speculation-INELIGIBLE ------------------------------- #
    _Decl(
        name="write_note",
        effect="Write",
        resource="storage",
        expected_duration_ms=5.0,
        params=(
            {"name": "path", "kind": "path", "required": True},
            {"name": "text", "kind": "str", "required": True},
        ),
        description=(
            "Append a note to the mutable scratch area. Externally visible side "
            "effect: never speculated, only ever executed authoritatively."
        ),
        concurrency_limit=1,
    ),
    _Decl(
        name="read_live_clock",
        effect="LiveRead",
        resource="cpu",
        expected_duration_ms=1.0,
        params=(),
        description=(
            "Read the host wall clock. Observes mutable live state, so its value "
            "is not a function of any snapshot: never speculated."
        ),
        concurrency_limit=1,
    ),
)

LOCAL_TOOL_NAMES: tuple[str, ...] = tuple(d.name for d in LOCAL_TOOL_DECLS)
SPECULATABLE_LOCAL_TOOLS: tuple[str, ...] = tuple(
    d.name for d in LOCAL_TOOL_DECLS if EffectKind(d.effect).speculation_eligible
)
INELIGIBLE_LOCAL_TOOLS: tuple[str, ...] = tuple(
    d.name for d in LOCAL_TOOL_DECLS if not EffectKind(d.effect).speculation_eligible
)

#: Which sub-snapshot each local ``SnapshotRead`` tool is bound to.
LOCAL_TOOL_SNAPSHOT: Mapping[str, str] = {
    "search_repo": "repo",
    "analyze_module": "repo",
    "sqlite_query": "sqlite",
    "embed_search": "embed",
}


# --------------------------------------------------------------------------- #
# local registry
# --------------------------------------------------------------------------- #


def _sub_specs(snapshot: SnapshotSpec) -> dict[str, SnapshotSpec]:
    if snapshot.kind == "corpus":
        return {name: sub_snapshot(snapshot, name) for name in ("repo", "sqlite", "embed")}
    return {snapshot.kind: snapshot}


def local_registry(
    snapshot: SnapshotSpec,
    *,
    concurrency_limits: Mapping[str, int] | None = None,
) -> EffectRegistry:
    """The frozen local tool registry, bound to ``snapshot``.

    Each ``SnapshotRead`` tool is bound to the id of the *sub*-snapshot it
    actually reads, so e.g. rebuilding only the embedding snapshot changes
    ``embed_search`` identities and leaves ``search_repo`` identities alone.
    """
    subs = _sub_specs(snapshot)
    ids: dict[str, str] = {}
    for tool, sub in LOCAL_TOOL_SNAPSHOT.items():
        spec = subs.get(sub)
        if spec is None:
            raise ValueError(
                f"snapshot {snapshot.snapshot_id} has no {sub!r} sub-snapshot needed by {tool!r}"
            )
        ids[tool] = spec.snapshot_id
    repo_root = subs["repo"].root if "repo" in subs else snapshot.root
    return build_registry(
        LOCAL_TOOL_DECLS,
        path_root=repo_root.as_posix(),
        snapshot_id=ids,
        concurrency_limits=concurrency_limits,
    )


# --------------------------------------------------------------------------- #
# (a) search_repo
# --------------------------------------------------------------------------- #

_MAX_LINE_CHARS = 400


def _resolve_in_root(root: Path, rel: str) -> Path:
    """Resolve a canonical relative path strictly inside ``root``."""
    if os.path.isabs(rel):
        raise PathEscapesSnapshot(f"absolute path not allowed: {rel!r}")
    target = (root / rel).resolve()
    root = root.resolve()
    if target != root and root not in target.parents:
        raise PathEscapesSnapshot(f"{rel!r} resolves outside the snapshot root {root}")
    return target


def _clip(text: str) -> str:
    text = text.rstrip("\r\n")
    return text if len(text) <= _MAX_LINE_CHARS else text[:_MAX_LINE_CHARS]


def _search_python(root: Path, target: Path, query: str) -> list[dict[str, Any]]:
    from .snapshots import iter_corpus_files

    files = [target] if target.is_file() else list(iter_corpus_files(target))
    out: list[dict[str, Any]] = []
    for path in files:
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        rel = path.relative_to(root).as_posix()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if query in line:
                out.append({"path": rel, "line": lineno, "text": _clip(line)})
    return out


#: Environment override for the ripgrep binary.
RIPGREP_ENV_VAR = "CRYSTALCALL_RG"

#: Vendored, pinned ripgrep used when the host has none on ``PATH``.  Provenance
#: (release URL + tarball sha256) is recorded in
#: ``experiments/vendor/bin/PROVENANCE.json`` and in ``configs/tools.yml``.
VENDORED_RG = Path(__file__).resolve().parents[2] / "experiments" / "vendor" / "bin" / "rg"


def resolve_ripgrep() -> Path | None:
    """``$CRYSTALCALL_RG`` -> ``PATH`` -> the pinned vendored binary -> ``None``.

    Returning ``None`` is not an error: ``search_repo`` then uses the
    pure-Python fallback and says so in its result.
    """
    raw = os.environ.get(RIPGREP_ENV_VAR, "").strip()
    if raw:
        cand = Path(raw)
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
        raise FileNotFoundError(f"{RIPGREP_ENV_VAR}={raw!r} is not an executable file")
    found = shutil.which("rg")
    if found:
        return Path(found).resolve()
    if VENDORED_RG.is_file() and os.access(VENDORED_RG, os.X_OK):
        return VENDORED_RG.resolve()
    return None


def _search_ripgrep(root: Path, target: Path, query: str) -> list[dict[str, Any]] | None:
    exe = resolve_ripgrep()
    if exe is None:
        return None
    globs: list[str] = []
    from .snapshots import TEXT_SUFFIXES

    for suffix in TEXT_SUFFIXES:
        globs += ["--glob", f"*{suffix}"]
    cmd = [
        str(exe),
        "--fixed-strings",
        "--line-number",
        "--with-filename",
        "--no-heading",
        "--color=never",
        "--no-messages",
        "--no-ignore",
        "--no-config",
        "--sort=path",
        *globs,
        "--",
        query,
        str(target),
    ]
    proc = subprocess.run(cmd, capture_output=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(
            f"rg failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace')[:200]}"
        )
    out: list[dict[str, Any]] = []
    # ``root`` is already resolved and every path ripgrep prints is under it, so
    # the relative path is a string slice.  Calling ``Path.resolve()`` per match
    # would cost a syscall each and dominate the tool's runtime.
    prefix = f"{root.as_posix()}/"
    for raw in proc.stdout.decode("utf-8", "replace").splitlines():
        head, sep, text = raw.partition(":")
        if not sep:
            continue
        lineno, sep2, text = text.partition(":")
        if not sep2 or not lineno.isdigit():
            continue
        rel = head[len(prefix) :] if head.startswith(prefix) else head
        out.append({"path": rel, "line": int(lineno), "text": _clip(text)})
    return out


def _make_search_repo(repo: SnapshotSpec) -> ToolImpl:
    root = repo.root.resolve()

    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        query = str(args["query"])
        rel = str(args.get("path", "."))
        max_results = int(args.get("max_results", 50))
        if not query:
            raise ValueError("search_repo: empty query")
        if max_results <= 0:
            raise ValueError(f"search_repo: max_results must be > 0, got {max_results}")
        target = _resolve_in_root(root, rel)
        if not target.exists():
            raise FileNotFoundError(f"search_repo: {rel!r} not found in snapshot")
        matches = _search_ripgrep(root, target, query)
        engine = "ripgrep"
        if matches is None:
            matches = _search_python(root, target, query)
            engine = "python"
        matches.sort(key=lambda m: (m["path"], m["line"]))
        truncated = len(matches) > max_results
        return {
            "matches": matches[:max_results],
            "engine": engine,
            "total_matches": len(matches),
            "truncated": truncated,
            "snapshot_id": repo.snapshot_id,
        }

    return impl


# --------------------------------------------------------------------------- #
# (b) sqlite_query
# --------------------------------------------------------------------------- #

#: Only these statements may ever reach SQLite.  ``WITH`` is *not* allowed:
#: SQLite accepts ``WITH … INSERT``, so a ``WITH`` prefix is not proof of
#: read-only intent.
_SQL_ALLOWED_HEADS = frozenset({"select"})

_SQL_FORBIDDEN = re.compile(
    r"\b("
    r"insert|update|delete|replace|drop|alter|create|attach|detach|pragma|vacuum|reindex|"
    r"begin|commit|rollback|savepoint|release|trigger|analyze|truncate"
    r")\b",
    re.IGNORECASE,
)

_SQL_MAX_SCAN_ROWS = 100_000

_ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def _strip_sql_literals(sql: str) -> str:
    """Blank out string/identifier literals and comments so token scans are sound."""
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]
        if ch in "'\"`":
            quote = ch
            out.append(" ")
            i += 1
            while i < n:
                if sql[i] == quote:
                    if i + 1 < n and sql[i + 1] == quote:
                        i += 2
                        continue
                    i += 1
                    break
                out.append(" " if sql[i] != "\n" else "\n")
                i += 1
            continue
        if ch == "[":
            out.append(" ")
            i += 1
            while i < n and sql[i] != "]":
                out.append(" ")
                i += 1
            i += 1
            continue
        if sql.startswith("--", i):
            while i < n and sql[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end < 0 else end + 2
            out.append(" " * (end - i))
            i = end
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def validate_read_only_sql(sql: str) -> str:
    """Return the single read-only ``SELECT`` in ``sql`` or raise :class:`SqlRejected`.

    Three independent gates, all of which must pass (a fourth, an ``sqlite3``
    authorizer, runs inside the connection):

    1. **Single statement.** ``sqlite3.complete_statement`` is applied to every
       prefix; if a complete statement ends before the end of the input, the
       string is multi-statement and rejected.
    2. **Allow-list.** The first keyword must be ``SELECT``.
    3. **Forbidden tokens.** A word-boundary scan over the literal-stripped SQL
       rejects every write/DDL/ATTACH/PRAGMA/transaction keyword.
    """
    if not isinstance(sql, str):
        raise SqlRejected(f"sql must be a string, got {type(sql).__name__}")
    text = sql.strip()
    if not text:
        raise SqlRejected("empty sql")
    if "\x00" in text:
        raise SqlRejected("NUL byte in sql")

    body = text[:-1].strip() if text.endswith(";") else text
    if not body:
        raise SqlRejected("empty sql")
    if not sqlite3.complete_statement(body + ";"):
        raise SqlRejected("incomplete statement")
    # Gate 1: any complete statement ending before the end => multi-statement.
    for cut in range(1, len(body)):
        if body[cut - 1] != ";":
            continue
        if sqlite3.complete_statement(body[:cut]) and body[cut:].strip():
            raise SqlRejected("multiple statements are not allowed")

    stripped = _strip_sql_literals(body)
    head = re.match(r"\s*([A-Za-z_]+)", stripped)
    if head is None or head.group(1).lower() not in _SQL_ALLOWED_HEADS:
        got = "?" if head is None else head.group(1).upper()
        raise SqlRejected(f"only a single SELECT is allowed, got {got}")
    hit = _SQL_FORBIDDEN.search(stripped)
    if hit is not None:
        raise SqlRejected(f"forbidden keyword {hit.group(1).upper()!r}")
    return body


_SQLITE_ALLOWED_ACTIONS = frozenset(
    {
        sqlite3.SQLITE_SELECT,
        sqlite3.SQLITE_READ,
        sqlite3.SQLITE_FUNCTION,
        sqlite3.SQLITE_RECURSIVE,
    }
)


def _deny_writes(action: int, *_rest: Any) -> int:
    """``sqlite3`` authorizer: read paths only, everything else denied."""
    return sqlite3.SQLITE_OK if action in _SQLITE_ALLOWED_ACTIONS else sqlite3.SQLITE_DENY


def _make_sqlite_query(snap: SnapshotSpec) -> ToolImpl:
    root = snap.root.resolve()
    allowed = {str(snap.meta["database"])}
    allowed.add(Path(str(snap.meta["database"])).stem)

    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        database = str(args["database"])
        max_rows = int(args.get("max_rows", 100))
        if max_rows <= 0:
            raise ValueError(f"sqlite_query: max_rows must be > 0, got {max_rows}")
        name = Path(database).name
        if name not in allowed and Path(name).stem not in allowed:
            raise ValueError(
                f"sqlite_query: unknown database {database!r}; snapshot exposes {sorted(allowed)}"
            )
        db_path = root / str(snap.meta["database"])
        body = validate_read_only_sql(str(args["sql"]))

        uri = f"file:{db_path.as_posix()}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True)
        try:
            conn.set_authorizer(_deny_writes)
            cur = conn.execute(body)
            columns = [d[0] for d in (cur.description or ())]
            ordered_by_sql = bool(_ORDER_BY.search(_strip_sql_literals(body)))
            if ordered_by_sql:
                rows = cur.fetchmany(max_rows + 1)
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
            else:
                rows = cur.fetchmany(_SQL_MAX_SCAN_ROWS + 1)
                if len(rows) > _SQL_MAX_SCAN_ROWS:
                    raise ValueError(
                        "sqlite_query: query without ORDER BY returned more than "
                        f"{_SQL_MAX_SCAN_ROWS} rows; add an ORDER BY"
                    )
                # No ORDER BY => SQLite's row order is not part of the contract.
                # Sort canonically so the value is a pure function of the query.
                rows = sorted(rows, key=lambda r: canonical_json(list(r)))
                truncated = len(rows) > max_rows
                rows = rows[:max_rows]
        except sqlite3.DatabaseError as exc:
            raise SqlRejected(f"sqlite rejected the query: {exc}") from exc
        finally:
            conn.set_authorizer(None)
            conn.close()
        return {
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
            "ordered_by_sql": ordered_by_sql,
            "snapshot_id": snap.snapshot_id,
        }

    return impl


# --------------------------------------------------------------------------- #
# (c) analyze_module
# --------------------------------------------------------------------------- #

#: Nodes that add one decision point to a function's branch count.  This is a
#: cyclomatic-complexity-*style* count: 1 + the number of these nodes lexically
#: inside the function (nested functions excluded, they are counted separately).
_BRANCH_NODES = (
    ast.If,
    ast.IfExp,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.ExceptHandler,
    ast.With,
    ast.AsyncWith,
    ast.Assert,
    ast.match_case,
)


def _branch_count(fn: ast.AST) -> int:
    count = 1
    stack: list[ast.AST] = list(ast.iter_child_nodes(fn))
    while stack:
        node = stack.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue  # counted as its own unit
        if isinstance(node, _BRANCH_NODES):
            count += 1
        if isinstance(node, ast.BoolOp):
            count += len(node.values) - 1
        if isinstance(node, (ast.comprehension,)):
            count += 1 + len(node.ifs)
        stack.extend(ast.iter_child_nodes(node))
    return count


def _qualified_functions(tree: ast.Module) -> list[tuple[str, ast.AST]]:
    out: list[tuple[str, ast.AST]] = []

    def walk(node: ast.AST, prefix: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}{child.name}"
                out.append((name, child))
                walk(child, f"{name}.")
            elif isinstance(child, ast.ClassDef):
                walk(child, f"{prefix}{child.name}.")
            else:
                walk(child, prefix)

    walk(tree, "")
    return out


def _import_names(tree: ast.Module) -> list[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            base = ("." * (node.level or 0)) + (node.module or "")
            for alias in node.names:
                names.add(f"{base}.{alias.name}" if base and not base.endswith(".") else f"{base}{alias.name}")
    return sorted(names)


def analyze_source(source: str, checks: Sequence[str], *, path: str = "<memory>") -> dict[str, Any]:
    """Pure static analysis of ``source``.  No filesystem, clock or network."""
    wanted = sorted(set(checks))
    unknown = [c for c in wanted if c not in ANALYZE_CHECKS]
    if unknown:
        raise ValueError(f"analyze_module: unknown check(s) {unknown}; known: {list(ANALYZE_CHECKS)}")
    result: dict[str, Any] = {"path": path, "checks": wanted}
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError as exc:
        result["syntax"] = {
            "valid": False,
            "error": f"{exc.msg} (line {exc.lineno}, col {exc.offset})",
        }
        if "complexity" in wanted:
            result["complexity"] = []
        if "imports" in wanted:
            result["imports"] = []
        result["counts"] = {"defs": 0, "classes": 0, "lines": source.count("\n")}
        return result

    if "syntax" in wanted:
        result["syntax"] = {"valid": True, "error": None}
    functions = _qualified_functions(tree)
    if "complexity" in wanted:
        result["complexity"] = [
            {"name": name, "lineno": int(getattr(fn, "lineno", 0)), "branches": _branch_count(fn)}
            for name, fn in sorted(functions, key=lambda kv: (kv[0], getattr(kv[1], "lineno", 0)))
        ]
    if "imports" in wanted:
        result["imports"] = _import_names(tree)
    classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
    result["counts"] = {
        "defs": len(functions),
        "classes": classes,
        "lines": source.count("\n"),
    }
    return result


def _make_analyze_module(repo: SnapshotSpec) -> ToolImpl:
    root = repo.root.resolve()

    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        rel = str(args["path"])
        checks = list(args.get("checks") or ANALYZE_CHECKS)
        target = _resolve_in_root(root, rel)
        if not target.is_file():
            raise FileNotFoundError(f"analyze_module: {rel!r} is not a file in the snapshot")
        source = target.read_bytes().decode("utf-8")
        out = analyze_source(source, checks, path=target.relative_to(root).as_posix())
        out["snapshot_id"] = repo.snapshot_id
        return out

    return impl


# --------------------------------------------------------------------------- #
# (d) embed_search
# --------------------------------------------------------------------------- #

#: Environment override for the compiled CUDA kernel.
CUDA_EMBED_ENV_VAR = "CRYSTALCALL_EMBED_SEARCH_BIN"

#: Name of the compiled kernel inside the corpus snapshot tree (``<corpus>/bin/``).
CUDA_EMBED_BINARY = "embed_search"

#: Source of the kernel, compiled by ``scripts/build_snapshots.py``.
CUDA_EMBED_SOURCE = Path(__file__).resolve().parent / "cuda" / "embed_search.cu"

_IDS_CACHE: dict[str, list[str]] = {}
_IDS_LOCK = threading.Lock()
#: Kept for the cold-cache benchmark: clearing it forces the ids re-read.
_EMBED_CACHE = _IDS_CACHE


def resolve_embed_binary(corpus_root: Path | None) -> Path | None:
    """``$CRYSTALCALL_EMBED_SEARCH_BIN`` -> ``<corpus>/bin/embed_search`` -> ``None``.

    ``None`` means the CUDA kernel was not built on this host; ``embed_search``
    then runs the deterministic numpy path and reports ``device="cpu"``.
    """
    raw = os.environ.get(CUDA_EMBED_ENV_VAR, "").strip()
    if raw:
        cand = Path(raw)
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
        raise FileNotFoundError(f"{CUDA_EMBED_ENV_VAR}={raw!r} is not an executable file")
    if corpus_root is not None:
        cand = corpus_root / "bin" / CUDA_EMBED_BINARY
        if cand.is_file() and os.access(cand, os.X_OK):
            return cand.resolve()
    return None


def _snapshot_ids(snap: SnapshotSpec) -> list[str]:
    with _IDS_LOCK:
        cached = _IDS_CACHE.get(snap.snapshot_id)
        if cached is not None:
            return cached
        ids = list(json.loads((snap.root / str(snap.meta["ids"])).read_text(encoding="utf-8")))
        _IDS_CACHE[snap.snapshot_id] = ids
        return ids


def _embed_search_cuda(
    binary: Path, snap: SnapshotSpec, vec: np.ndarray, top_k: int
) -> list[tuple[int, float]]:
    """Run the deterministic CUDA kernel and parse ``<row>\\t<score>`` lines."""
    rows = int(snap.meta["rows"])
    dim = int(snap.meta["dim"])
    matrix = snap.root / str(snap.meta["matrix"])
    with tempfile.NamedTemporaryFile(suffix=".f32", delete=False) as fh:
        fh.write(np.ascontiguousarray(vec, dtype=np.float32).tobytes())
        query_path = Path(fh.name)
    try:
        proc = subprocess.run(
            [str(binary), str(matrix), str(rows), str(dim), str(query_path), str(top_k)],
            capture_output=True,
            check=False,
        )
    finally:
        query_path.unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"embed_search CUDA kernel failed ({proc.returncode}): "
            f"{proc.stderr.decode('utf-8', 'replace').strip()[:300]}"
        )
    out: list[tuple[int, float]] = []
    for line in proc.stdout.decode("utf-8", "replace").splitlines():
        idx, sep, score = line.partition("\t")
        if not sep:
            continue
        out.append((int(idx), float(score)))
    return out


def _make_embed_search(snap: SnapshotSpec, corpus_root: Path | None = None) -> ToolImpl:
    model_id = str(snap.meta["model_id"])
    dim = int(snap.meta.get("dim", EMBED_DIM))

    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        query = str(args["query"])
        top_k = int(args.get("top_k", 5))
        if not query:
            raise ValueError("embed_search: empty query")
        if top_k <= 0:
            raise ValueError(f"embed_search: top_k must be > 0, got {top_k}")
        ids = _snapshot_ids(snap)
        vec = embed_text(query, dim=dim, model_id=model_id)
        binary = resolve_embed_binary(corpus_root)
        if binary is not None:
            device = "cuda"
            ranked = _embed_search_cuda(binary, snap, vec, min(top_k, len(ids)))
        else:
            # Deterministic CPU fallback: identical layout, identical ranking
            # rule.  Only reached when the CUDA kernel was not built.
            device = "cpu"
            matrix = load_embedding_matrix(snap)
            scores = np.asarray(matrix, dtype=np.float32) @ vec
            order = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), i))
            ranked = [(i, float(scores[i])) for i in order[: min(top_k, len(ids))]]
        # Round for reporting only; the ranking above is a total order on
        # (-score, row index) so rounding can never reorder hits.
        hits = [{"id": ids[i], "score": round(score, 6)} for i, score in ranked]
        return {
            "hits": hits,
            "device": device,
            "documents": len(ids),
            "dim": dim,
            "model_id": model_id,
            "snapshot_id": snap.snapshot_id,
        }

    return impl


# --------------------------------------------------------------------------- #
# ineligible tools
# --------------------------------------------------------------------------- #


def _scratch_root() -> Path:
    raw = os.environ.get(SCRATCH_ENV_VAR, "").strip()
    root = Path(raw) if raw else Path(os.environ.get("TMPDIR", "/tmp")) / "crystalcall-scratch"
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def _make_write_note() -> ToolImpl:
    lock = threading.Lock()

    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        rel = str(args["path"])
        text = str(args["text"])
        root = _scratch_root()
        target = _resolve_in_root(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        with lock:
            with target.open("a", encoding="utf-8") as fh:
                fh.write(text if text.endswith("\n") else text + "\n")
            size = target.stat().st_size
        return {"path": rel, "bytes_written": len(text.encode("utf-8")), "file_bytes": size}

    return impl


def _make_read_live_clock() -> ToolImpl:
    def impl(args: Mapping[str, Any], cancel: threading.Event) -> dict[str, Any]:
        if args:
            raise ValueError(f"read_live_clock takes no arguments, got {sorted(args)}")
        return {"unix_ns": time.time_ns(), "monotonic_ns": time.monotonic_ns()}

    return impl


# --------------------------------------------------------------------------- #
# local impls
# --------------------------------------------------------------------------- #


def local_impls(snapshot: SnapshotSpec) -> dict[str, ToolImpl]:
    """Real implementations for every tool in :data:`LOCAL_TOOL_DECLS`.

    ``snapshot`` is the composite corpus snapshot (kind ``"corpus"``); the three
    frozen sub-snapshots are resolved from it.
    """
    subs = _sub_specs(snapshot)
    missing = sorted({"repo", "sqlite", "embed"} - set(subs))
    if missing:
        raise ValueError(f"snapshot {snapshot.snapshot_id} lacks sub-snapshot(s) {missing}")
    return {
        "search_repo": _make_search_repo(subs["repo"]),
        "sqlite_query": _make_sqlite_query(subs["sqlite"]),
        "analyze_module": _make_analyze_module(subs["repo"]),
        "embed_search": _make_embed_search(
            subs["embed"], snapshot.root if snapshot.kind == "corpus" else None
        ),
        "write_note": _make_write_note(),
        "read_live_clock": _make_read_live_clock(),
    }
