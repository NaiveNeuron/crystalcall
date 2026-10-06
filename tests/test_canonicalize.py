"""Canonical call identity: equivalent spellings collide, inequivalent ones never do."""

from __future__ import annotations

import pytest

from crystalcall.canonicalize import CanonicalizationError, canonicalize
from crystalcall.effects import EffectKind

from conftest import SNAPSHOT_A


def ident(reg, name, kwargs=None, positional=()):
    return canonicalize(reg, name, kwargs, positional).identity


# --------------------------------------------------------------------------- #
# equivalence
# --------------------------------------------------------------------------- #


def test_positional_and_keyword_agree(registry):
    assert ident(registry, "calc_sum", {"a": 1, "b": 2}) == ident(
        registry, "calc_sum", None, (1, 2)
    )


def test_omitted_optional_materialises_default(registry):
    assert ident(registry, "calc_sum", {"a": 1, "b": 2}) == ident(
        registry, "calc_sum", {"a": 1, "b": 2, "precision": 2}
    )
    assert ident(registry, "search_repo", {"query": "x"}) == ident(
        registry, "search_repo", {"query": "x", "path": ".", "max_results": 50}
    )


def test_numeric_normalisation(registry):
    assert ident(registry, "calc_sum", {"a": 1, "b": 2}) == ident(
        registry, "calc_sum", {"a": 1.0, "b": 2.0}
    )
    assert ident(registry, "calc_sum", {"a": 1.5, "b": 2}) != ident(
        registry, "calc_sum", {"a": 1, "b": 2}
    )


def test_alias_and_namespace_resolve(registry):
    base = ident(registry, "calc_sum", {"a": 1, "b": 2})
    assert ident(registry, "sum_numbers", {"a": 1, "b": 2}) == base
    assert ident(registry, "tool.calc_sum", {"a": 1, "b": 2}) == base
    assert ident(registry, "tools.sum_numbers", {"a": 1, "b": 2}) == base


def test_path_normalisation_inside_root(registry):
    a = ident(registry, "search_repo", {"query": "q", "path": "src/"})
    b = ident(registry, "search_repo", {"query": "q", "path": "./src"})
    c = ident(registry, "search_repo", {"query": "q", "path": "src/./"})
    d = ident(registry, "search_repo", {"query": "q", "path": "/srv/frozen/src"})
    assert a == b == c == d


def test_unicode_normalisation(registry):
    # U+00E9 vs e + U+0301 are the same string after NFC.
    assert ident(registry, "lookup_symbol", {"name": "caf\u00e9"}) == ident(
        registry, "lookup_symbol", {"name": "cafe\u0301"}
    )


def test_unordered_list_sorted_only_when_schema_permits(registry):
    assert ident(registry, "tag_set", {"tags": ["b", "a"]}) == ident(
        registry, "tag_set", {"tags": ["a", "b"]}
    )


def test_dict_key_order_irrelevant(registry):
    a = canonicalize(registry, "calc_sum", {"a": 1, "b": 2})
    b = canonicalize(registry, "calc_sum", {"b": 2, "a": 1})
    assert a.args == b.args
    assert a.args_digest == b.args_digest


# --------------------------------------------------------------------------- #
# non-collision
# --------------------------------------------------------------------------- #


def test_distinct_arguments_never_collide(registry):
    seen = {
        ident(registry, "lookup_symbol", {"name": n})
        for n in ("A", "B", "C", "a", "", " A", "A ")
    }
    assert len(seen) == 7


def test_distinct_tools_never_collide(registry):
    assert ident(registry, "lookup_symbol", {"name": "x"}) != ident(
        registry, "search_repo", {"query": "x"}
    )


def test_snapshot_is_part_of_identity(registry):
    a = canonicalize(registry, "sqlite_query", {"database": "one.db", "sql": "SELECT 1"})
    b = canonicalize(registry, "sqlite_query", {"database": "two.db", "sql": "SELECT 1"})
    assert a.snapshot_id != b.snapshot_id
    assert a.identity != b.identity


def test_snapshot_id_present_only_for_snapshot_reads(registry):
    assert canonicalize(registry, "calc_sum", {"a": 1, "b": 1}).snapshot_id is None
    sr = canonicalize(registry, "search_repo", {"query": "q"})
    assert sr.snapshot_id == SNAPSHOT_A
    assert sr.effect is EffectKind.SNAPSHOT_READ


# --------------------------------------------------------------------------- #
# rejection, never silent repair
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("name", "kwargs", "reason"),
    [
        ("calc_sum", {"a": 1}, "missing_required"),
        ("calc_sum", {"a": 1, "b": 2, "c": 3}, "unknown_parameter"),
        ("calc_sum", {"a": "1", "b": 2}, "type_mismatch"),
        ("calc_sum", {"a": True, "b": 2}, "type_mismatch"),
        ("calc_sum", {"a": float("nan"), "b": 2}, "non_finite_number"),
        ("calc_sum", {"a": 1, "b": 2, "precision": 1.5}, "type_mismatch"),
        ("no_such_tool", {}, "unknown_tool"),
        ("search_repo", {"query": "q", "path": "../etc"}, "path_escapes_root"),
        ("search_repo", {"query": "q", "path": "/etc/passwd"}, "path_escapes_root"),
        ("tag_set", {"tags": "notalist"}, "type_mismatch"),
    ],
)
def test_rejections(registry, name, kwargs, reason):
    with pytest.raises(CanonicalizationError) as exc:
        canonicalize(registry, name, kwargs)
    assert exc.value.reason == reason


def test_duplicate_keyword_is_rejected(registry):
    with pytest.raises(CanonicalizationError) as exc:
        canonicalize(registry, "calc_sum", {"a": 1, "b": 2}, duplicate_keys=("a",))
    assert exc.value.reason == "duplicate_keyword"


def test_positional_and_keyword_conflict_is_rejected(registry):
    with pytest.raises(CanonicalizationError) as exc:
        canonicalize(registry, "calc_sum", {"a": 5}, (1,))
    assert exc.value.reason == "duplicate_keyword"


def test_too_many_positional_is_rejected(registry):
    with pytest.raises(CanonicalizationError) as exc:
        canonicalize(registry, "calc_sum", None, (1, 2, 3, 4))
    assert exc.value.reason == "too_many_positional"


def test_registry_fingerprint_is_stable_and_content_sensitive(registry):
    from crystalcall.effects import EffectKind, ParamKind, ParamSpec, ResourceClass, ToolSpec

    fp = registry.fingerprint()
    assert fp == registry.fingerprint()
    registry.register(
        ToolSpec(
            name="extra_tool",
            effect=EffectKind.PURE,
            resource=ResourceClass.CPU,
            params=(ParamSpec("x", ParamKind.INT),),
            expected_duration_ms=1.0,
        )
    )
    assert registry.fingerprint() != fp
