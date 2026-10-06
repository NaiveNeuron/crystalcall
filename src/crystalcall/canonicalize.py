"""Canonical tool-call identity.

One canonicaliser is used by speculative extraction, authoritative OMP
execution, replay analysis, every baseline and the equivalence tests.  Byte
equality of source text is deliberately *not* the identity relation: the same
semantic call may be spelled with positional or keyword arguments, omitted
optional arguments, differently ordered dictionary keys, or a differently
spelled but equivalent path.

Canonicalisation is total-or-reject.  It never silently repairs a call: an
unknown parameter, a duplicate key, a wrong type, a path that escapes the
allow-listed root, or a non-finite number all raise
:class:`CanonicalizationError`.
"""

from __future__ import annotations

import hashlib
import json
import math
import posixpath
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .effects import (
    EffectKind,
    EffectRegistry,
    ParamKind,
    ParamSpec,
    ToolSpec,
    UnknownToolError,
)

__all__ = [
    "CanonicalCall",
    "CanonicalizationError",
    "canonicalize",
    "canonical_json",
    "digest_args",
]

_MAX_DEPTH = 8


class CanonicalizationError(ValueError):
    """A call could not be canonicalised.  Carries a machine-readable reason so
    rejection classes can be counted in the result data."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no ASCII escaping."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest_args(args: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(args)).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CanonicalCall:
    """A canonical, executable tool-call identity."""

    tool: str
    """Canonical tool name (aliases and namespaces resolved)."""

    args: tuple[tuple[str, Any], ...]
    """Canonical arguments, sorted by parameter name, defaults materialised."""

    snapshot_id: str | None
    """Immutable snapshot identifier for ``SnapshotRead``; ``None`` otherwise."""

    effect: EffectKind

    @property
    def args_dict(self) -> dict[str, Any]:
        return dict(self.args)

    @property
    def args_digest(self) -> str:
        return digest_args(self.args_dict)

    @property
    def identity(self) -> str:
        """Stable, printable identity string (excludes ``request_id``)."""
        return f"{self.tool}|{self.args_digest}|{self.snapshot_id or '-'}"

    def digest(self) -> str:
        return hashlib.sha256(self.identity.encode("utf-8")).hexdigest()

    def to_json(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": self.args_dict,
            "snapshot_id": self.snapshot_id,
            "effect": self.effect.value,
            "args_digest": self.args_digest,
            "identity": self.identity,
        }

    def source(self) -> str:
        """Canonical Python spelling; used in replay and in test fixtures."""
        rendered = ", ".join(f"{k}={v!r}" for k, v in self.args)
        return f"{self.tool}({rendered})"


# --------------------------------------------------------------------------- #
# value normalisation
# --------------------------------------------------------------------------- #


def _norm_str(v: Any) -> str:
    if not isinstance(v, str):
        raise CanonicalizationError("type_mismatch", f"expected str, got {type(v).__name__}")
    return unicodedata.normalize("NFC", v)


def _norm_number(v: Any) -> int | float:
    if isinstance(v, bool):
        raise CanonicalizationError("type_mismatch", "bool supplied for numeric parameter")
    if not isinstance(v, (int, float)):
        raise CanonicalizationError("type_mismatch", f"expected number, got {type(v).__name__}")
    if isinstance(v, float):
        if not math.isfinite(v):
            raise CanonicalizationError("non_finite_number", repr(v))
        if v.is_integer():
            return int(v)
        return v
    return int(v)


def _norm_path(v: Any, root: str | None) -> str:
    raw = _norm_str(v)
    if "\x00" in raw:
        raise CanonicalizationError("invalid_path", "NUL byte")
    # Purely lexical normalisation: no filesystem access, no symlink
    # resolution.  Speculation must not touch the filesystem to decide identity.
    cleaned = posixpath.normpath(raw)
    if cleaned == ".":
        cleaned = "."
    if root is None:
        if posixpath.isabs(cleaned):
            raise CanonicalizationError("path_escapes_root", raw)
        if cleaned.startswith(".."):
            raise CanonicalizationError("path_escapes_root", raw)
        return cleaned
    norm_root = posixpath.normpath(root)
    if posixpath.isabs(cleaned):
        if cleaned != norm_root and not cleaned.startswith(norm_root.rstrip("/") + "/"):
            raise CanonicalizationError("path_escapes_root", raw)
        rel = posixpath.relpath(cleaned, norm_root)
    else:
        rel = posixpath.normpath(cleaned)
    if rel == "." or rel == "":
        return "."
    if rel.startswith(".."):
        raise CanonicalizationError("path_escapes_root", raw)
    return rel


def _norm_literal(v: Any, depth: int = 0) -> Any:
    """Normalise an arbitrary statically-resolved literal value."""
    if depth > _MAX_DEPTH:
        raise CanonicalizationError("nesting_too_deep", str(depth))
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, str):
        return _norm_str(v)
    if isinstance(v, (int, float)):
        return _norm_number(v)
    if isinstance(v, (list, tuple)):
        return [_norm_literal(x, depth + 1) for x in v]
    if isinstance(v, Mapping):
        out: dict[str, Any] = {}
        for k, val in v.items():
            if not isinstance(k, str):
                raise CanonicalizationError("non_string_dict_key", repr(k))
            nk = _norm_str(k)
            if nk in out:
                raise CanonicalizationError("duplicate_dict_key", nk)
            out[nk] = _norm_literal(val, depth + 1)
        return out
    raise CanonicalizationError("unsupported_value", type(v).__name__)


def _norm_param(value: Any, spec: ParamSpec, registry_root: str | None) -> Any:
    kind = spec.kind
    if kind is ParamKind.STR:
        return _norm_str(value)
    if kind is ParamKind.PATH:
        return _norm_path(value, spec.path_root or registry_root)
    if kind is ParamKind.INT:
        n = _norm_number(value)
        if not isinstance(n, int):
            raise CanonicalizationError("type_mismatch", f"expected int, got {value!r}")
        return n
    if kind is ParamKind.FLOAT:
        return _norm_number(value)
    if kind is ParamKind.BOOL:
        if not isinstance(value, bool):
            raise CanonicalizationError("type_mismatch", f"expected bool, got {value!r}")
        return value
    if kind is ParamKind.LIST:
        if not isinstance(value, (list, tuple)):
            raise CanonicalizationError("type_mismatch", f"expected list, got {value!r}")
        if spec.element_kind is not None:
            elem_spec = ParamSpec(
                name=spec.name,
                kind=spec.element_kind,
                required=True,
                path_root=spec.path_root,
            )
            items = [_norm_param(x, elem_spec, registry_root) for x in value]
        else:
            items = [_norm_literal(x) for x in value]
        if not spec.order_sensitive:
            items = sorted(items, key=canonical_json)
        return items
    if kind is ParamKind.DICT:
        if not isinstance(value, Mapping):
            raise CanonicalizationError("type_mismatch", f"expected dict, got {value!r}")
        return _norm_literal(value)
    raise CanonicalizationError("unsupported_param_kind", kind.value)


# --------------------------------------------------------------------------- #
# the canonicaliser
# --------------------------------------------------------------------------- #


def canonicalize(
    registry: EffectRegistry,
    name: str,
    kwargs: Mapping[str, Any] | None = None,
    positional: Sequence[Any] = (),
    *,
    duplicate_keys: Sequence[str] = (),
) -> CanonicalCall:
    """Canonicalise one call against the registry.

    ``duplicate_keys`` lets a caller (the AST extractor) report keys that
    appeared more than once in the *source*, which a Python ``dict`` would
    already have collapsed.  Duplicates are always rejected, never repaired.
    """
    if duplicate_keys:
        raise CanonicalizationError("duplicate_keyword", ",".join(sorted(set(duplicate_keys))))

    try:
        spec: ToolSpec = registry.resolve(name)
    except UnknownToolError:
        raise CanonicalizationError("unknown_tool", name) from None

    supplied: dict[str, Any] = {}

    if len(positional) > len(spec.params):
        raise CanonicalizationError(
            "too_many_positional", f"{len(positional)} > {len(spec.params)}"
        )
    for p, value in zip(spec.params, positional, strict=False):
        supplied[p.name] = value

    for k, v in (kwargs or {}).items():
        if not isinstance(k, str):
            raise CanonicalizationError("non_string_keyword", repr(k))
        if k in supplied:
            raise CanonicalizationError("duplicate_keyword", k)
        supplied[k] = v

    declared = set(spec.param_names)
    unknown = sorted(set(supplied) - declared)
    if unknown:
        raise CanonicalizationError("unknown_parameter", ",".join(unknown))

    canon: dict[str, Any] = {}
    for p in spec.params:
        if p.name in supplied:
            canon[p.name] = _norm_param(supplied[p.name], p, registry.path_root)
        elif p.required:
            raise CanonicalizationError("missing_required", p.name)
        elif p.default is not None:
            # Materialise declared defaults so that ``f(a=1)`` and
            # ``f(a=1, b=<default>)`` are the same identity.
            canon[p.name] = _norm_param(p.default, p, registry.path_root)
        # An optional parameter whose default is None stays absent.

    snapshot_id: str | None = None
    if spec.effect is EffectKind.SNAPSHOT_READ:
        assert spec.snapshot_resolver is not None
        try:
            snapshot_id = spec.snapshot_resolver(canon)
        except CanonicalizationError:
            raise
        except Exception as exc:  # noqa: BLE001 - resolver failures reject the call
            raise CanonicalizationError("snapshot_unresolvable", f"{type(exc).__name__}: {exc}")
        if not isinstance(snapshot_id, str) or not snapshot_id:
            raise CanonicalizationError("snapshot_unresolvable", repr(snapshot_id))

    ordered = tuple(sorted(canon.items(), key=lambda kv: kv[0]))
    return CanonicalCall(
        tool=spec.name, args=ordered, snapshot_id=snapshot_id, effect=spec.effect
    )
