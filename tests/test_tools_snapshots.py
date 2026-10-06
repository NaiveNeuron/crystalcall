"""Contract tests for the frozen snapshots and the local tool families.

Everything here defends a property the paper's lossless-speculation claim rests
on: snapshot ids are content-determined, the read-only SQLite gate cannot be
talked past, path arguments cannot escape the snapshot, static analysis is a
pure function of file content, ineligible effects are never speculatable, and
the injected latency used by the deterministic-duration harness cannot change a
returned value.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from crystalcall.effects import EffectKind
from crystalcall.snapshots import (
    SnapshotDriftError,
    build_corpus_snapshot,
    build_embedding_snapshot,
    build_repo_snapshot,
    build_sqlite_fts_snapshot,
    load_snapshot,
    make_writable,
    snapshot_permissions_ok,
    verify_snapshot,
)
from crystalcall.tools import (
    ANALYZE_CHECKS,
    INELIGIBLE_LOCAL_TOOLS,
    LOCAL_TOOL_DECLS,
    PathEscapesSnapshot,
    SLEEP_ENV_VAR,
    SPECULATABLE_LOCAL_TOOLS,
    SqlRejected,
    analyze_source,
    build_impls,
    build_registry,
    local_impls,
    local_registry,
    validate_read_only_sql,
)

# Tiny snapshots: the production guard rails on matrix size are relaxed here so
# the suite stays fast, never the *derivation* of the ids.
EMBED_KW = {"dim": 8, "window": 4, "target_rows": 8, "min_bytes": 0, "max_bytes": 1 << 20}

MODULE_A = '''\
"""Docstring."""
import os
from collections import deque


class Widget:
    def spin(self, n):
        total = 0
        for i in range(n):
            if i % 2 == 0 and i > 2:
                total += i
        return total


def helper(x):
    return [y for y in x if y]
'''

MODULE_B = '''\
import sys


def other(x):
    return x + 1
'''


def _write_corpus(root: Path, extra: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "pkg").mkdir(exist_ok=True)
    (root / "pkg" / "a.py").write_text(MODULE_A, encoding="utf-8")
    (root / "pkg" / "b.py").write_text(MODULE_B, encoding="utf-8")
    (root / "notes.md").write_text("# Notes\nalpha beta gamma\n", encoding="utf-8")
    if extra is not None:
        (root / "pkg" / "c.py").write_text(extra, encoding="utf-8")
    return root


@pytest.fixture
def unlock(request):
    """Return read-only snapshot trees to writable so pytest can clean up."""
    roots: list[Path] = []

    def _register(path: Path) -> Path:
        roots.append(path)
        return path

    yield _register
    for path in roots:
        make_writable(path)


@pytest.fixture
def corpus(tmp_path, unlock, monkeypatch):
    monkeypatch.delenv("CRYSTALCALL_EMBED_SEARCH_BIN", raising=False)
    monkeypatch.delenv(SLEEP_ENV_VAR, raising=False)
    src = _write_corpus(tmp_path / "src")
    out = unlock(tmp_path / "snap")
    out.mkdir()
    repo = build_repo_snapshot(src, out / "repo")
    sql = build_sqlite_fts_snapshot(repo.root, out / "sqlite" / "corpus.db")
    emb = build_embedding_snapshot(repo.root, out / "embed", **EMBED_KW)
    return build_corpus_snapshot(out, repo, sql, emb)


# --------------------------------------------------------------------------- #
# snapshot identity
# --------------------------------------------------------------------------- #


def test_snapshot_id_is_content_determined(tmp_path, unlock):
    src = _write_corpus(tmp_path / "src")
    spec = build_repo_snapshot(src, unlock(tmp_path / "a"))

    assert spec.snapshot_id.startswith("repo@")
    prefix, _, short = spec.snapshot_id.partition("@")
    assert short == spec.digest[:16]
    assert len(short) == 16
    assert verify_snapshot(spec) is True
    assert snapshot_permissions_ok(spec) is True
    # The sidecar round-trips to the same identity.
    assert load_snapshot(spec.root).snapshot_id == spec.snapshot_id


def test_same_content_different_location_gives_same_id(tmp_path, unlock):
    a = build_repo_snapshot(_write_corpus(tmp_path / "src_a"), unlock(tmp_path / "a"))
    b = build_repo_snapshot(_write_corpus(tmp_path / "src_b"), unlock(tmp_path / "b"))
    assert a.snapshot_id == b.snapshot_id
    assert a.digest == b.digest
    assert a.root != b.root


def test_different_content_gives_different_id(tmp_path, unlock):
    a = build_repo_snapshot(_write_corpus(tmp_path / "src_a"), unlock(tmp_path / "a"))
    b = build_repo_snapshot(
        _write_corpus(tmp_path / "src_b", extra="X = 1\n"), unlock(tmp_path / "b")
    )
    assert a.snapshot_id != b.snapshot_id


def test_renaming_a_file_changes_the_id(tmp_path, unlock):
    """The manifest hashes paths as well as bytes."""
    src_a = _write_corpus(tmp_path / "src_a")
    a = build_repo_snapshot(src_a, unlock(tmp_path / "a"))
    src_b = _write_corpus(tmp_path / "src_b")
    (src_b / "pkg" / "a.py").rename(src_b / "pkg" / "renamed.py")
    b = build_repo_snapshot(src_b, unlock(tmp_path / "b"))
    assert a.snapshot_id != b.snapshot_id


def test_verify_snapshot_detects_drift(tmp_path, unlock):
    spec = build_repo_snapshot(_write_corpus(tmp_path / "src"), unlock(tmp_path / "a"))
    victim = spec.root / "pkg" / "a.py"
    os.chmod(spec.root, 0o755)
    os.chmod(victim.parent, 0o755)
    os.chmod(victim, 0o644)
    victim.write_text(MODULE_A + "\n# tampered\n", encoding="utf-8")

    with pytest.raises(SnapshotDriftError):
        verify_snapshot(spec)
    assert verify_snapshot(spec, strict=False) is False


def test_corpus_snapshot_binds_all_three_sub_ids(corpus):
    assert corpus.kind == "corpus"
    assert corpus.snapshot_id.startswith("corpus@")
    ids = {name: sub["snapshot_id"] for name, sub in corpus.meta["sub"].items()}
    assert set(ids) == {"repo", "sqlite", "embed"}
    assert ids["repo"].startswith("repo@")
    assert ids["sqlite"].startswith("sqlite:corpus@")
    assert ids["embed"].startswith("emb:corpus@")
    assert verify_snapshot(corpus) is True


def test_registry_binds_each_tool_to_its_sub_snapshot(corpus):
    registry = local_registry(corpus)
    ids = {name: sub["snapshot_id"] for name, sub in corpus.meta["sub"].items()}
    from crystalcall.canonicalize import canonicalize

    assert canonicalize(registry, "search_repo", {"query": "x"}).snapshot_id == ids["repo"]
    assert canonicalize(registry, "analyze_module", {"path": "pkg/a.py"}).snapshot_id == ids["repo"]
    assert (
        canonicalize(registry, "sqlite_query", {"database": "corpus.db", "sql": "SELECT 1"}).snapshot_id
        == ids["sqlite"]
    )
    assert canonicalize(registry, "embed_search", {"query": "x"}).snapshot_id == ids["embed"]


# --------------------------------------------------------------------------- #
# sqlite read-only gate
# --------------------------------------------------------------------------- #

REJECTED_SQL = [
    "INSERT INTO doc_meta(path, sha256, lines, bytes) VALUES ('x', 'y', 1, 1)",
    "UPDATE doc_meta SET lines = 0",
    "DELETE FROM doc_meta",
    "DROP TABLE doc_meta",
    "CREATE TABLE evil (x INTEGER)",
    "ALTER TABLE doc_meta ADD COLUMN evil INTEGER",
    "CREATE VIRTUAL TABLE evil USING fts5(x)",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "DETACH DATABASE evil",
    "PRAGMA journal_mode=WAL",
    "PRAGMA writable_schema=ON",
    "VACUUM",
    "REINDEX",
    "BEGIN; SELECT 1",
    "REPLACE INTO doc_meta VALUES ('a','b',1,1)",
    "SELECT 1; DROP TABLE doc_meta",
    "SELECT 1; SELECT 2",
    "SELECT 1;;SELECT 2",
    "WITH x AS (SELECT 1) SELECT * FROM x",
    "",
    "   ",
]


@pytest.mark.parametrize("sql", REJECTED_SQL)
def test_sqlite_query_rejects_non_readonly_sql(corpus, sql):
    impl = local_impls(corpus)["sqlite_query"]
    with pytest.raises(SqlRejected):
        validate_read_only_sql(sql)
    with pytest.raises(SqlRejected):
        impl({"database": "corpus.db", "sql": sql, "max_rows": 10}, threading.Event())


def test_sqlite_query_rejects_write_hidden_in_a_string_lookalike(corpus):
    """A forbidden keyword inside a *literal* is fine; outside it is not."""
    impl = local_impls(corpus)["sqlite_query"]
    ok = impl(
        {
            "database": "corpus.db",
            "sql": "SELECT path FROM doc_meta WHERE path LIKE '%insert%' ORDER BY path",
            "max_rows": 5,
        },
        threading.Event(),
    )
    assert ok["columns"] == ["path"]
    with pytest.raises(SqlRejected):
        impl(
            {"database": "corpus.db", "sql": "SELECT 1 -- ok\n; DELETE FROM doc_meta"},
            threading.Event(),
        )


def test_sqlite_query_accepts_select_and_is_deterministic(corpus):
    impl = local_impls(corpus)["sqlite_query"]
    args = {
        "database": "corpus.db",
        "sql": "SELECT path, lines FROM doc_meta ORDER BY path",
        "max_rows": 10,
    }
    a = impl(args, threading.Event())
    b = impl(args, threading.Event())
    assert a == b
    assert a["ordered_by_sql"] is True
    assert a["columns"] == ["path", "lines"]
    assert [r[0] for r in a["rows"]] == sorted(r[0] for r in a["rows"])


def test_sqlite_query_sorts_rows_when_sql_has_no_order_by(corpus):
    impl = local_impls(corpus)["sqlite_query"]
    out = impl(
        {"database": "corpus.db", "sql": "SELECT path FROM doc_meta", "max_rows": 10},
        threading.Event(),
    )
    assert out["ordered_by_sql"] is False
    assert out["rows"] == sorted(out["rows"], key=lambda r: json.dumps(r))


def test_sqlite_query_rejects_unknown_database(corpus):
    impl = local_impls(corpus)["sqlite_query"]
    with pytest.raises(ValueError, match="unknown database"):
        impl({"database": "other.db", "sql": "SELECT 1"}, threading.Event())


# --------------------------------------------------------------------------- #
# search_repo containment
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["../..", "../../etc", "/etc", "pkg/../../..", "/etc/passwd"])
def test_search_repo_refuses_paths_outside_the_snapshot(corpus, bad):
    impl = local_impls(corpus)["search_repo"]
    with pytest.raises(PathEscapesSnapshot):
        impl({"query": "root", "path": bad, "max_results": 5}, threading.Event())


def test_analyze_module_refuses_paths_outside_the_snapshot(corpus):
    impl = local_impls(corpus)["analyze_module"]
    with pytest.raises(PathEscapesSnapshot):
        impl({"path": "../../etc/passwd", "checks": ["syntax"]}, threading.Event())


def test_search_repo_is_deterministic_and_sorted(corpus):
    impl = local_impls(corpus)["search_repo"]
    args = {"query": "def ", "path": ".", "max_results": 50}
    a = impl(args, threading.Event())
    b = impl(args, threading.Event())
    assert a == b
    assert a["engine"] in ("ripgrep", "python")
    assert a["matches"]
    keys = [(m["path"], m["line"]) for m in a["matches"]]
    assert keys == sorted(keys)


def test_search_repo_respects_max_results(corpus):
    impl = local_impls(corpus)["search_repo"]
    out = impl({"query": "def ", "path": ".", "max_results": 1}, threading.Event())
    assert len(out["matches"]) == 1
    assert out["truncated"] is True


# --------------------------------------------------------------------------- #
# analyze_module purity
# --------------------------------------------------------------------------- #


def test_analyze_module_is_a_pure_function_of_content(tmp_path, unlock, corpus):
    """Same bytes at a different path in a different snapshot -> same analysis."""
    other_src = tmp_path / "other"
    other_src.mkdir()
    (other_src / "elsewhere.py").write_text(MODULE_A, encoding="utf-8")
    other = build_repo_snapshot(other_src, unlock(tmp_path / "other_snap"))

    from crystalcall.tools import _make_analyze_module

    here = local_impls(corpus)["analyze_module"]
    there = _make_analyze_module(other)
    a = here({"path": "pkg/a.py", "checks": list(ANALYZE_CHECKS)}, threading.Event())
    b = there({"path": "elsewhere.py", "checks": list(ANALYZE_CHECKS)}, threading.Event())

    volatile = ("path", "snapshot_id")
    assert {k: v for k, v in a.items() if k not in volatile} == {
        k: v for k, v in b.items() if k not in volatile
    }
    # ...and equal to the in-memory analysis of the same source.
    direct = analyze_source(MODULE_A, list(ANALYZE_CHECKS))
    assert direct["counts"] == a["counts"]
    assert direct["imports"] == a["imports"] == ["collections.deque", "os"]
    assert direct["complexity"] == a["complexity"]


def test_analyze_module_reports_branches_and_syntax_errors():
    ok = analyze_source(MODULE_A, ["complexity", "syntax"])
    assert ok["syntax"] == {"valid": True, "error": None}
    by_name = {row["name"]: row["branches"] for row in ok["complexity"]}
    # 1 base + `for` + `if` + one extra `and` operand + the comprehension.
    assert by_name["Widget.spin"] == 4
    assert by_name["helper"] == 3

    bad = analyze_source("def f(:\n", ["syntax", "complexity", "imports"])
    assert bad["syntax"]["valid"] is False
    assert bad["complexity"] == [] and bad["imports"] == []


def test_analyze_module_rejects_unknown_checks():
    with pytest.raises(ValueError, match="unknown check"):
        analyze_source(MODULE_A, ["syntax", "vibes"])


# --------------------------------------------------------------------------- #
# effect eligibility
# --------------------------------------------------------------------------- #


def test_ineligible_tools_are_never_speculatable(corpus):
    registry = local_registry(corpus)
    assert INELIGIBLE_LOCAL_TOOLS == ("write_note", "read_live_clock")
    for name in INELIGIBLE_LOCAL_TOOLS:
        spec = registry.resolve(name)
        assert spec.effect in (EffectKind.WRITE, EffectKind.LIVE_READ)
        assert spec.snapshot_resolver is None
        assert spec.speculation_eligible is False
        assert registry.is_speculation_eligible(name) is False
        assert name not in registry.eligible_names()
    for name in SPECULATABLE_LOCAL_TOOLS:
        assert registry.is_speculation_eligible(name) is True


def test_ineligible_tools_still_execute_authoritatively(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("CRYSTALCALL_SCRATCH_DIR", str(tmp_path / "scratch"))
    impls = local_impls(corpus)
    out = impls["write_note"]({"path": "note.txt", "text": "hello"}, threading.Event())
    assert out["bytes_written"] == 5
    assert (tmp_path / "scratch" / "note.txt").read_text(encoding="utf-8") == "hello\n"

    clock = impls["read_live_clock"]({}, threading.Event())
    assert clock["unix_ns"] > 0
    with pytest.raises(ValueError):
        impls["read_live_clock"]({"tz": "UTC"}, threading.Event())


def test_write_note_cannot_escape_the_scratch_root(corpus, tmp_path, monkeypatch):
    monkeypatch.setenv("CRYSTALCALL_SCRATCH_DIR", str(tmp_path / "scratch"))
    impls = local_impls(corpus)
    with pytest.raises(PathEscapesSnapshot):
        impls["write_note"]({"path": "../escaped.txt", "text": "x"}, threading.Event())


def test_snapshot_read_tools_require_a_bound_snapshot_id():
    with pytest.raises(ValueError, match="no snapshot_id was supplied"):
        build_registry(LOCAL_TOOL_DECLS)


# --------------------------------------------------------------------------- #
# injected latency
# --------------------------------------------------------------------------- #

PURE_DECLS = tuple(
    d
    for d in (
        type(
            "D",
            (),
            {
                "name": "add_numbers",
                "effect": "Pure",
                "resource": "cpu",
                "expected_duration_ms": 1.0,
                "params": ({"name": "a", "kind": "int"}, {"name": "b", "kind": "int"}),
                "description": "add",
            },
        )(),
    )
)


def test_sleep_env_var_does_not_change_returned_values(monkeypatch):
    impls = build_impls(PURE_DECLS)
    impl = impls["add_numbers"]
    args = {"a": 1, "b": 2}

    monkeypatch.delenv(SLEEP_ENV_VAR, raising=False)
    baseline = impl(args, threading.Event())

    for value in ("0", "5", "37.5"):
        monkeypatch.setenv(SLEEP_ENV_VAR, value)
        assert impl(args, threading.Event()) == baseline


def test_sleep_env_var_does_not_change_local_tool_values(corpus, monkeypatch):
    impls = local_impls(corpus)
    cases = {
        "search_repo": {"query": "def ", "path": ".", "max_results": 5},
        "sqlite_query": {
            "database": "corpus.db",
            "sql": "SELECT path FROM doc_meta ORDER BY path",
            "max_rows": 5,
        },
        "analyze_module": {"path": "pkg/a.py", "checks": list(ANALYZE_CHECKS)},
        "embed_search": {"query": "alpha beta", "top_k": 3},
    }
    monkeypatch.delenv(SLEEP_ENV_VAR, raising=False)
    baseline = {name: impls[name](args, threading.Event()) for name, args in cases.items()}
    monkeypatch.setenv(SLEEP_ENV_VAR, "25")
    for name, args in cases.items():
        assert impls[name](args, threading.Event()) == baseline[name]


def test_injected_sleep_is_interruptible_and_actually_sleeps(monkeypatch):
    import time

    impls = build_impls(PURE_DECLS)
    impl = impls["add_numbers"]
    monkeypatch.setenv(SLEEP_ENV_VAR, "400")

    t0 = time.perf_counter()
    impl({"a": 1, "b": 2}, threading.Event())
    slept_ms = (time.perf_counter() - t0) * 1e3
    assert slept_ms >= 350

    cancel = threading.Event()
    cancel.set()
    t0 = time.perf_counter()
    impl({"a": 1, "b": 2}, cancel)
    assert (time.perf_counter() - t0) * 1e3 < 100


def test_malformed_sleep_env_var_is_a_hard_error(monkeypatch):
    impls = build_impls(PURE_DECLS)
    monkeypatch.setenv(SLEEP_ENV_VAR, "soon")
    with pytest.raises(ValueError, match=SLEEP_ENV_VAR):
        impls["add_numbers"]({"a": 1, "b": 2}, threading.Event())


# --------------------------------------------------------------------------- #
# pure stubs
# --------------------------------------------------------------------------- #


def test_pure_stub_is_a_pure_function_of_canonical_arguments():
    impls = build_impls(PURE_DECLS)
    impl = impls["add_numbers"]
    a = impl({"a": 1, "b": 2}, threading.Event())
    b = impl({"b": 2, "a": 1}, threading.Event())
    assert a == b
    # The payload must be RESULT-shaped, never an echo of the call: the model
    # read an echo as "the tool did not run" and retried by hand on 76 of 100
    # development trajectories.
    assert a["deterministic"] is True
    assert set(a) == {"ok", "result", "deterministic"}
    assert "tool" not in a and "args" not in a and "canonical" not in a
    # Not even the tool name appears in the value the model reads.
    assert "add_numbers" not in json.dumps(a)
    assert isinstance(a["result"]["value"], str)
    # Still a pure function of the canonical call, so result digests are stable.
    assert impl({"a": 1, "b": 3}, threading.Event()) != a


def test_build_impls_refuses_to_fake_an_effectful_tool():
    with pytest.raises(ValueError, match="snapshot_root"):
        build_impls(LOCAL_TOOL_DECLS)


def test_build_registry_honours_per_param_path_root():
    decl = type(
        "D",
        (),
        {
            "name": "reader",
            "effect": "Pure",
            "resource": "cpu",
            "expected_duration_ms": 1.0,
            "params": ({"name": "path", "kind": "path", "path_root": "/"},),
            "description": "",
            "aliases": ("pkg.reader",),
        },
    )()
    registry = build_registry([decl], path_root="/nowhere")
    spec = registry.resolve("reader")
    assert spec.param("path").path_root == "/"
    assert registry.resolve("pkg.reader") is spec

    from crystalcall.canonicalize import canonicalize

    call = canonicalize(registry, "reader", {"path": "/etc/hosts"})
    assert dict(call.args)["path"] == "etc/hosts"


def test_build_registry_rejects_unknown_param_fields():
    decl = type(
        "D",
        (),
        {
            "name": "x",
            "effect": "Pure",
            "resource": "cpu",
            "expected_duration_ms": 1.0,
            "params": ({"name": "a", "kind": "int", "bogus": 1},),
            "description": "",
        },
    )()
    with pytest.raises(ValueError, match="unknown ParamSpec field"):
        build_registry([decl])
