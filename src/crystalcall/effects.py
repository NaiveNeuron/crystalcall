"""Effect registry and tool schema declarations for CrystalCall / DenoisePTC.

The registry is the single source of truth for:

* the canonical tool name and its accepted aliases;
* the declared parameter list (name, order, type, default, normalisation);
* the static effect annotation;
* the resource class, expected duration and concurrency limit used by the
  launch gate's expected-overlap test;
* how an immutable snapshot identifier is derived for ``SnapshotRead`` tools.

Only ``Pure`` and ``SnapshotRead`` are speculation-eligible.  ``LiveRead`` and
``Write`` are never speculated, no matter what the gate says.
"""

from __future__ import annotations

import enum
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "EffectKind",
    "ResourceClass",
    "ParamKind",
    "ParamSpec",
    "ToolSpec",
    "EffectRegistry",
    "SnapshotResolver",
    "UnknownToolError",
]


class UnknownToolError(KeyError):
    """Raised when a call names a tool that is not in the allow-list."""


class EffectKind(enum.Enum):
    """The static effect annotation carried by every registered tool."""

    PURE = "Pure"
    SNAPSHOT_READ = "SnapshotRead"
    LIVE_READ = "LiveRead"
    WRITE = "Write"

    @property
    def speculation_eligible(self) -> bool:
        return self in (EffectKind.PURE, EffectKind.SNAPSHOT_READ)


class ResourceClass(enum.Enum):
    """Dominant resource consumed by a tool invocation."""

    CPU = "cpu"
    STORAGE = "storage"
    GPU = "gpu"
    NETWORK = "network"


class ParamKind(enum.Enum):
    """Declared parameter type.  Drives canonical normalisation."""

    STR = "str"
    PATH = "path"
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    LIST = "list"
    DICT = "dict"


#: A snapshot resolver maps canonical arguments to an immutable snapshot id.
#: It must be deterministic and must fail (raise) rather than invent an id.
SnapshotResolver = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True, slots=True)
class ParamSpec:
    """One declared parameter of a registered tool."""

    name: str
    kind: ParamKind
    required: bool = True
    default: Any = None
    #: For ``LIST`` parameters: whether element order is semantically
    #: meaningful.  When ``False`` the canonicaliser sorts elements, which is
    #: the only place unordered array equality is permitted.
    order_sensitive: bool = True
    #: For ``PATH`` parameters: the allow-listed root the path must resolve
    #: inside.  ``None`` means the registry-level root is used.
    path_root: str | None = None
    #: Element kind for ``LIST`` parameters (``None`` = heterogeneous literals).
    element_kind: ParamKind | None = None


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Complete static description of one allow-listed tool."""

    name: str
    effect: EffectKind
    resource: ResourceClass
    params: tuple[ParamSpec, ...]
    #: Table-based duration estimate (milliseconds) used by the overlap gate.
    expected_duration_ms: float
    concurrency_limit: int = 1
    aliases: frozenset[str] = field(default_factory=frozenset)
    #: Required for ``SNAPSHOT_READ``.  Returns the immutable snapshot id.
    snapshot_resolver: SnapshotResolver | None = None
    #: Human-readable description; part of the frozen tool representation.
    description: str = ""

    def __post_init__(self) -> None:
        if self.effect is EffectKind.SNAPSHOT_READ and self.snapshot_resolver is None:
            raise ValueError(f"tool {self.name!r}: SnapshotRead requires a snapshot_resolver")
        if self.effect is not EffectKind.SNAPSHOT_READ and self.snapshot_resolver is not None:
            raise ValueError(f"tool {self.name!r}: snapshot_resolver only valid for SnapshotRead")
        seen: set[str] = set()
        for p in self.params:
            if p.name in seen:
                raise ValueError(f"tool {self.name!r}: duplicate parameter {p.name!r}")
            seen.add(p.name)
        # A required parameter must not carry a default; that would make the
        # canonical form of an omitted argument ambiguous.
        for p in self.params:
            if p.required and p.default is not None:
                raise ValueError(f"tool {self.name!r}: required param {p.name!r} has a default")

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(p.name for p in self.params)

    def param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    @property
    def speculation_eligible(self) -> bool:
        return self.effect.speculation_eligible


class EffectRegistry:
    """The allow-list.  Speculation, authoritative execution, replay, baselines
    and equivalence tests all resolve tools through this one object."""

    def __init__(self, tools: Iterable[ToolSpec] = (), *, path_root: str | None = None) -> None:
        self._by_name: dict[str, ToolSpec] = {}
        self._alias_to_name: dict[str, str] = {}
        self.path_root = path_root
        for t in tools:
            self.register(t)

    def register(self, spec: ToolSpec) -> ToolSpec:
        if spec.name in self._by_name or spec.name in self._alias_to_name:
            raise ValueError(f"duplicate tool identity {spec.name!r}")
        for a in spec.aliases:
            if a in self._by_name or a in self._alias_to_name:
                raise ValueError(f"duplicate tool alias {a!r}")
        self._by_name[spec.name] = spec
        for a in spec.aliases:
            self._alias_to_name[a] = spec.name
        return spec

    def resolve(self, name: str) -> ToolSpec:
        """Resolve a (possibly aliased or namespaced) tool name.

        Namespace prefixes ``tool.``, ``tools.`` and ``omp.`` are stripped;
        anything else unknown raises.
        """
        candidate = name
        for prefix in ("tool.", "tools.", "omp."):
            if candidate.startswith(prefix):
                candidate = candidate[len(prefix) :]
                break
        if candidate in self._by_name:
            return self._by_name[candidate]
        if candidate in self._alias_to_name:
            return self._by_name[self._alias_to_name[candidate]]
        raise UnknownToolError(name)

    def get(self, name: str) -> ToolSpec | None:
        try:
            return self.resolve(name)
        except UnknownToolError:
            return None

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self.get(name) is not None

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_name))

    def all_names(self) -> tuple[str, ...]:
        """Canonical names plus every accepted alias and namespaced spelling."""
        out = set(self._by_name)
        out.update(self._alias_to_name)
        return tuple(sorted(out))

    def specs(self) -> tuple[ToolSpec, ...]:
        return tuple(self._by_name[n] for n in sorted(self._by_name))

    def eligible_names(self) -> tuple[str, ...]:
        return tuple(n for n in self.names() if self._by_name[n].speculation_eligible)

    def is_speculation_eligible(self, name: str) -> bool:
        spec = self.get(name)
        return spec is not None and spec.speculation_eligible

    def fingerprint(self) -> str:
        """Stable digest of the whole registry, recorded in every run's config
        hash so a drifting tool surface fails loudly."""
        import hashlib
        import json

        payload: list[Any] = []
        for spec in self.specs():
            payload.append(
                {
                    "name": spec.name,
                    "effect": spec.effect.value,
                    "resource": spec.resource.value,
                    "expected_duration_ms": spec.expected_duration_ms,
                    "concurrency_limit": spec.concurrency_limit,
                    "aliases": sorted(spec.aliases),
                    "params": [
                        {
                            "name": p.name,
                            "kind": p.kind.value,
                            "required": p.required,
                            "default": p.default,
                            "order_sensitive": p.order_sensitive,
                            "element_kind": p.element_kind.value if p.element_kind else None,
                            "path_root": p.path_root,
                        }
                        for p in spec.params
                    ],
                }
            )
        blob = json.dumps(
            {"path_root": self.path_root, "tools": payload},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def json_schemas(self) -> list[dict[str, Any]]:
        """The one fixed tool representation handed to every model arm."""
        kind_to_json = {
            ParamKind.STR: "string",
            ParamKind.PATH: "string",
            ParamKind.INT: "integer",
            ParamKind.FLOAT: "number",
            ParamKind.BOOL: "boolean",
            ParamKind.LIST: "array",
            ParamKind.DICT: "object",
        }
        out: list[dict[str, Any]] = []
        for spec in self.specs():
            props: dict[str, Any] = {}
            required: list[str] = []
            for p in spec.params:
                props[p.name] = {"type": kind_to_json[p.kind]}
                if p.required:
                    required.append(p.name)
                elif p.default is not None:
                    props[p.name]["default"] = p.default
            out.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "effect": spec.effect.value,
                    "parameters": {
                        "type": "object",
                        "properties": props,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            )
        return out


def param_list(*items: Sequence[Any]) -> tuple[ParamSpec, ...]:
    """Tiny helper for declarative registries in configuration modules."""
    return tuple(ParamSpec(*i) for i in items)  # type: ignore[arg-type]
