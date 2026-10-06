"""Prompt manifests: the frozen, model-independent input surface of every run.

A *manifest* is a JSONL file whose first line is a header record and whose
remaining lines are :class:`PromptRecord` objects.  A ``PromptRecord`` carries
everything an arm needs to run one prompt and everything replay needs to score
it: the user message, the declared tool surface (:class:`ToolDecl`, a
serialisable seed for :class:`~crystalcall.effects.ToolSpec`) and the upstream
ground-truth calls normalised into ``{"tool": …, "args": {…}}``.

Three hard rules hold for everything in this module:

* **Model-independent.**  Nothing here may read a model output.  Selection,
  effect classification and filtering are pure functions of the upstream data
  plus ``configs/manifest_rules.yml``, whose sha256 lands in the header record.
* **Deterministic.**  Sorted iteration order, an explicit seed, no ``set``
  iteration on any output path.  Rebuilding a manifest is byte-identical.
* **No fabrication.**  A prompt that cannot be converted losslessly is dropped
  with a machine-readable reason, never repaired.

The upstream converters (BFCL, ToolSandbox) live here rather than in the build
script so that they are importable and testable; ``scripts/build_manifests.py``
is a thin CLI over this module.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .canonicalize import CanonicalizationError, canonicalize
from .effects import (
    EffectKind,
    EffectRegistry,
    ParamKind,
    ParamSpec,
    ResourceClass,
    ToolSpec,
)

__all__ = [
    "MANIFEST_SCHEMA",
    "RULES_SCHEMA",
    "ManifestBuildError",
    "ManifestRules",
    "ToolDecl",
    "PromptRecord",
    "ScenarioRecord",
    "DropRecord",
    "load_rules",
    "load_manifest",
    "load_header",
    "load_scenarios",
    "write_manifest",
    "write_scenarios",
    "write_hash_sidecar",
    "read_hash_sidecar",
    "registry_for",
    "sanitize_tool_name",
    "tool_name_map",
    "alias_plan",
    "registry_for_decls",
    "manifest_hash",
    "sha256_file",
    "sha256_bytes",
    "classify_effect",
    "param_decl",
    "tool_decl_from_bfcl",
    "build_bfcl_candidates",
    "select_splits",
    "extract_toolsandbox_scenarios",
    "select_toolsandbox",
    "toolsandbox_family",
    "PARAM_DECL_KEYS",
]

MANIFEST_SCHEMA = "crystalcall-manifest/1"
RULES_SCHEMA = "crystalcall-manifest-rules/1"

#: Every parameter declaration carries exactly these keys, always.
PARAM_DECL_KEYS = (
    "default",
    "element_kind",
    "kind",
    "name",
    "order_sensitive",
    "path_root",
    "required",
)


class ManifestBuildError(ValueError):
    """A prompt could not be converted.  Carries a machine-readable reason."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# --------------------------------------------------------------------------- #
# hashing helpers
# --------------------------------------------------------------------------- #


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def manifest_hash(path: str | Path) -> str:
    """sha256 of a manifest file, byte for byte."""
    return sha256_file(path)


def write_hash_sidecar(path: str | Path) -> str:
    """Write ``<path>.sha256`` in ``sha256sum`` format; return the digest."""
    p = Path(path)
    digest = sha256_file(p)
    p.with_name(p.name + ".sha256").write_text(f"{digest}  {p.name}\n", encoding="utf-8")
    return digest


def read_hash_sidecar(path: str | Path) -> str | None:
    """Digest recorded in ``<path>.sha256``, or ``None`` when absent."""
    side = Path(path).with_name(Path(path).name + ".sha256")
    if not side.exists():
        return None
    return side.read_text(encoding="utf-8").split()[0]


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------- #
# data model
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolDecl:
    """Serialisable seed for one :class:`~crystalcall.effects.ToolSpec`.

    ``effect``, ``resource`` and ``params[*]["kind"]`` hold the *string values*
    of the corresponding enums, so a manifest line is plain JSON.

    ``name`` is always a valid Python identifier: the generated program calls
    tools through OMP's attribute-based proxy (``tool.<name>(…)``), so an
    upstream name like ``math.gcd`` is unreachable and is sanitised.
    ``original_name`` keeps the upstream spelling verbatim; it is registered as
    a registry alias so both spellings share one canonical identity.
    """

    name: str
    effect: str
    resource: str
    expected_duration_ms: float
    params: tuple[dict, ...]
    description: str = ""
    original_name: str = ""

    @property
    def upstream_name(self) -> str:
        """The upstream spelling, falling back to ``name``."""
        return self.original_name or self.name

    @property
    def alias_candidate(self) -> str | None:
        """The upstream spelling when it differs from the canonical name."""
        original = self.original_name
        return original if original and original != self.name else None

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "effect": self.effect,
            "resource": self.resource,
            "expected_duration_ms": self.expected_duration_ms,
            "params": [dict(p) for p in self.params],
            "description": self.description,
            "original_name": self.original_name,
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> ToolDecl:
        return cls(
            name=obj["name"],
            effect=obj["effect"],
            resource=obj["resource"],
            expected_duration_ms=float(obj["expected_duration_ms"]),
            params=tuple(dict(p) for p in obj.get("params", ())),
            description=obj.get("description", ""),
            original_name=obj.get("original_name", ""),
        )


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One prompt: user message, declared tool surface, ground-truth calls."""

    prompt_id: str
    category: str
    source: str
    source_id: str
    user_message: str
    tools: tuple[ToolDecl, ...]
    expected_calls: tuple[dict, ...]
    multi_turn: bool = False
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "record": "prompt",
            "prompt_id": self.prompt_id,
            "category": self.category,
            "source": self.source,
            "source_id": self.source_id,
            "user_message": self.user_message,
            "tools": [t.to_json() for t in self.tools],
            "expected_calls": [dict(c) for c in self.expected_calls],
            "multi_turn": self.multi_turn,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> PromptRecord:
        return cls(
            prompt_id=obj["prompt_id"],
            category=obj["category"],
            source=obj["source"],
            source_id=obj["source_id"],
            user_message=obj["user_message"],
            tools=tuple(ToolDecl.from_json(t) for t in obj.get("tools", ())),
            expected_calls=tuple(dict(c) for c in obj.get("expected_calls", ())),
            multi_turn=bool(obj.get("multi_turn", False)),
            meta=dict(obj.get("meta", {})),
        )


@dataclass(frozen=True, slots=True)
class ScenarioRecord:
    """One preselected ToolSandbox scenario.

    ToolSandbox scenarios are stateful sandbox programs, not static prompts, so
    they carry no ``PromptRecord`` tool surface; the sandbox owns the tool
    implementations.  What the manifest pins is *which* scenarios run, derived
    statically from the checkout.
    """

    scenario_id: str
    family: str
    base_scenario: str
    variant: str
    module: str
    categories: tuple[str, ...]
    tool_allow_list: tuple[str, ...]
    user_messages: tuple[str, ...]
    n_milestones: int
    source: str = "toolsandbox"
    meta: dict = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "record": "scenario",
            "scenario_id": self.scenario_id,
            "family": self.family,
            "base_scenario": self.base_scenario,
            "variant": self.variant,
            "module": self.module,
            "categories": list(self.categories),
            "tool_allow_list": list(self.tool_allow_list),
            "user_messages": list(self.user_messages),
            "n_milestones": self.n_milestones,
            "source": self.source,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, obj: Mapping[str, Any]) -> ScenarioRecord:
        return cls(
            scenario_id=obj["scenario_id"],
            family=obj["family"],
            base_scenario=obj["base_scenario"],
            variant=obj["variant"],
            module=obj["module"],
            categories=tuple(obj.get("categories", ())),
            tool_allow_list=tuple(obj.get("tool_allow_list", ())),
            user_messages=tuple(obj.get("user_messages", ())),
            n_milestones=int(obj.get("n_milestones", 0)),
            source=obj.get("source", "toolsandbox"),
            meta=dict(obj.get("meta", {})),
        )


@dataclass(frozen=True, slots=True)
class DropRecord:
    """One rejected upstream prompt, with the rule that rejected it."""

    prompt_id: str
    category: str
    reason: str
    detail: str = ""

    def to_json(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "category": self.category,
            "reason": self.reason,
            "detail": self.detail,
        }


# --------------------------------------------------------------------------- #
# manifest I/O
# --------------------------------------------------------------------------- #


def _iter_json_lines(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_header(path: str | Path) -> dict[str, Any] | None:
    """Header record of a manifest file, or ``None`` when it has none."""
    for obj in _iter_json_lines(path):
        return obj if obj.get("record") == "header" else None
    return None


def load_manifest(path: str | Path) -> list[PromptRecord]:
    """Every prompt record of a manifest file, in file order."""
    return [
        PromptRecord.from_json(obj)
        for obj in _iter_json_lines(path)
        if obj.get("record", "prompt") == "prompt"
    ]


def load_scenarios(path: str | Path) -> list[ScenarioRecord]:
    """Every scenario record of a scenario manifest, in file order."""
    return [
        ScenarioRecord.from_json(obj)
        for obj in _iter_json_lines(path)
        if obj.get("record", "scenario") == "scenario"
    ]


def _write_jsonl(
    path: str | Path,
    header: Mapping[str, Any] | None,
    rows: Iterable[Mapping[str, Any]],
) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[str] = []
    if header is not None:
        head = dict(header)
        head["record"] = "header"
        chunks.append(_dumps(head))
    chunks.extend(_dumps(r) for r in rows)
    p.write_text("".join(c + "\n" for c in chunks), encoding="utf-8")
    return sha256_file(p)


def write_manifest(
    path: str | Path,
    records: Sequence[PromptRecord],
    *,
    header: Mapping[str, Any] | None = None,
) -> str:
    """Write a prompt manifest; return the sha256 of the written file."""
    return _write_jsonl(path, header, (r.to_json() for r in records))


def write_scenarios(
    path: str | Path,
    records: Sequence[ScenarioRecord],
    *,
    header: Mapping[str, Any] | None = None,
) -> str:
    """Write a scenario manifest; return the sha256 of the written file."""
    return _write_jsonl(path, header, (r.to_json() for r in records))


# --------------------------------------------------------------------------- #
# tool-name sanitisation
# --------------------------------------------------------------------------- #

_NON_IDENTIFIER = re.compile(r"[^0-9A-Za-z_]")
_DIGIT_PREFIX = "t_"


def sanitize_tool_name(original: str) -> str:
    """Map an upstream tool name onto a valid Python identifier.

    The generated program reaches tools through OMP's attribute-based proxy
    (``tool.<name>(…)``), and the PTC extractor only accepts an identifier
    callee, so a dotted upstream name such as ``math.gcd`` is unreachable:
    ``tool.math`` would resolve first and fail.  Sanitisation is purely
    lexical and deterministic.
    """
    safe = _NON_IDENTIFIER.sub("_", original)
    if not safe:
        raise ManifestBuildError("unsanitisable_tool_name", original)
    if safe[0].isdigit():
        safe = f"{_DIGIT_PREFIX}{safe}"
    return safe


def tool_name_map(originals: Sequence[str]) -> dict[str, str]:
    """Upstream name -> sanitised name for one prompt's tool surface.

    Collisions introduced by sanitisation are resolved by appending ``_2``,
    ``_3``, … walking the *sorted* upstream names, so the map depends only on
    the set of upstream names.
    """
    taken: set[str] = set()
    mapping: dict[str, str] = {}
    for original in sorted(originals):
        base = sanitize_tool_name(original)
        safe = base
        index = 2
        while safe in taken:
            safe = f"{base}_{index}"
            index += 1
        taken.add(safe)
        mapping[original] = safe
    return mapping


# --------------------------------------------------------------------------- #
# registries
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _AliasedDecl:
    """Duck-typed declaration carrying the aliases ``build_registry`` registers."""

    name: str
    effect: str
    resource: str
    expected_duration_ms: float
    params: tuple[dict, ...]
    description: str
    aliases: tuple[str, ...]


def alias_plan(decls: Sequence[ToolDecl]) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
    """Which upstream spellings can be registered as aliases, and which cannot.

    An upstream spelling is registered as an alias of its sanitised name so a
    model that emits the dotted form still canonicalises to the same identity.
    :meth:`EffectRegistry.register` rejects a duplicate identity, so an alias
    that collides with any canonical name or any other alias in the same
    prompt is skipped rather than repaired.
    """
    canonical = {d.name for d in decls}
    aliases: dict[str, tuple[str, ...]] = {}
    skipped: list[str] = []
    claimed: set[str] = set()
    for decl in sorted(decls, key=lambda d: d.upstream_name):
        candidate = decl.alias_candidate
        if candidate is None:
            aliases[decl.name] = ()
            continue
        if candidate in canonical or candidate in claimed:
            skipped.append(candidate)
            aliases[decl.name] = ()
            continue
        claimed.add(candidate)
        aliases[decl.name] = (candidate,)
    return aliases, tuple(sorted(skipped))


def registry_for_decls(
    decls: Sequence[ToolDecl], *, path_root: str | None = None
) -> EffectRegistry:
    """Build an :class:`EffectRegistry` from serialised declarations.

    Each tool is registered under its sanitised identifier name, with the
    upstream spelling as an alias whenever that is collision-free.
    """
    from .tools import build_registry

    aliases, _skipped = alias_plan(decls)
    aliased = [
        _AliasedDecl(
            name=d.name,
            effect=d.effect,
            resource=d.resource,
            expected_duration_ms=d.expected_duration_ms,
            params=d.params,
            description=d.description,
            aliases=aliases.get(d.name, ()),
        )
        for d in decls
    ]
    return build_registry(aliased, path_root=path_root)


def registry_for(record: PromptRecord, *, path_root: str | None = None) -> EffectRegistry:
    """The per-prompt tool allow-list.

    Path parameters carry their own ``path_root`` (set by the manifest rules),
    which the canonicaliser prefers over the registry-level root, so the
    default ``path_root=None`` is the right choice for BFCL prompts.
    """
    return registry_for_decls(record.tools, path_root=path_root)


# --------------------------------------------------------------------------- #
# rules
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ManifestRules:
    """``configs/manifest_rules.yml``, parsed, plus its identity."""

    path: str
    sha256: str
    raw: Mapping[str, Any]

    # -- sections ---------------------------------------------------------- #
    @property
    def bfcl(self) -> Mapping[str, Any]:
        return self.raw["bfcl"]

    @property
    def param_kinds(self) -> Mapping[str, Any]:
        return self.raw["param_kinds"]

    @property
    def effects(self) -> Mapping[str, Any]:
        return self.raw["effects"]

    @property
    def tools(self) -> Mapping[str, Any]:
        return self.raw["tools"]

    @property
    def ground_truth(self) -> Mapping[str, Any]:
        return self.raw["ground_truth"]

    @property
    def selection(self) -> Mapping[str, Any]:
        return self.raw["selection"]

    @property
    def toolsandbox(self) -> Mapping[str, Any]:
        return self.raw["toolsandbox"]

    @property
    def tool_names(self) -> Mapping[str, Any]:
        return self.raw["tool_names"]

    # -- derived ----------------------------------------------------------- #
    @property
    def categories(self) -> Mapping[str, str]:
        return self.bfcl["categories"]

    @property
    def type_map(self) -> Mapping[str, str]:
        return self.param_kinds["type_map"]

    @property
    def path_root(self) -> str | None:
        return self.param_kinds.get("path_root")

    @property
    def path_name_re(self) -> re.Pattern[str]:
        return re.compile(self.param_kinds["path_name_regex"])

    @property
    def write_re(self) -> re.Pattern[str]:
        return re.compile(self.effects["write_regex"])

    @property
    def live_read_re(self) -> re.Pattern[str]:
        return re.compile(self.effects["live_read_regex"])

    @property
    def call_expression_re(self) -> re.Pattern[str]:
        return re.compile(self.ground_truth["call_expression_regex"])


def load_rules(path: str | Path) -> ManifestRules:
    """Load and validate the manifest rules file."""
    import yaml

    p = Path(path)
    blob = p.read_bytes()
    raw = yaml.safe_load(blob.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ManifestBuildError("rules_not_mapping", str(p))
    if raw.get("schema") != RULES_SCHEMA:
        raise ManifestBuildError("rules_schema_mismatch", str(raw.get("schema")))
    for section in (
        "bfcl",
        "param_kinds",
        "tool_names",
        "effects",
        "tools",
        "ground_truth",
        "selection",
        "toolsandbox",
    ):
        if section not in raw:
            raise ManifestBuildError("rules_missing_section", section)
    # The sanitiser is implemented once, in :func:`sanitize_tool_name`; the
    # rule file declares it, so a drifting declaration must fail rather than
    # silently disagree with the code.
    declared = raw["tool_names"]
    for key, expected in (
        ("sanitize_regex", _NON_IDENTIFIER.pattern),
        ("replacement", "_"),
        ("digit_prefix", _DIGIT_PREFIX),
        ("register_original_as_alias", True),
    ):
        if declared.get(key) != expected:
            raise ManifestBuildError(
                "unsupported_rule", f"tool_names.{key}={declared.get(key)!r} != {expected!r}"
            )
    if raw["effects"].get("classify_on") != "original_name":
        raise ManifestBuildError(
            "unsupported_rule", f"effects.classify_on={raw['effects'].get('classify_on')!r}"
        )
    return ManifestRules(path=str(p), sha256=sha256_bytes(blob), raw=raw)


# --------------------------------------------------------------------------- #
# effect classification
# --------------------------------------------------------------------------- #


def classify_effect(name: str, rules: ManifestRules) -> str:
    """Mechanically classify a tool name into an :class:`EffectKind` value.

    Declared before any model run, applied by regex search, first match wins in
    the order given by ``effects.order``.
    """
    patterns = {"Write": rules.write_re, "LiveRead": rules.live_read_re}
    for effect in rules.effects["order"]:
        pat = patterns.get(effect)
        if pat is None:
            raise ManifestBuildError("unknown_effect_in_rules", effect)
        if pat.search(name):
            return effect
    return rules.effects["fallback"]


# --------------------------------------------------------------------------- #
# BFCL -> ToolDecl
# --------------------------------------------------------------------------- #


def param_decl(
    name: str,
    kind: str,
    *,
    required: bool,
    default: Any = None,
    order_sensitive: bool = True,
    element_kind: str | None = None,
    path_root: str | None = None,
) -> dict[str, Any]:
    """One parameter declaration.  Always carries :data:`PARAM_DECL_KEYS`."""
    decl = {
        "name": name,
        "kind": kind,
        "required": required,
        "default": None if required else default,
        "order_sensitive": order_sensitive,
        "element_kind": element_kind,
        "path_root": path_root,
    }
    assert tuple(sorted(decl)) == PARAM_DECL_KEYS
    return decl


def _probe_default_ok(decl: Mapping[str, Any], registry_root: str | None) -> bool:
    """Does a declared default survive the frozen canonicaliser?

    Rather than re-implement value normalisation, build a one-parameter probe
    tool and let :func:`canonicalize` materialise the default exactly as it
    will in production.
    """
    spec = ToolSpec(
        name="_probe",
        effect=EffectKind.PURE,
        resource=ResourceClass.CPU,
        params=(_param_spec(decl),),
        expected_duration_ms=0.0,
    )
    registry = EffectRegistry([spec], path_root=registry_root)
    try:
        canonicalize(registry, "_probe", {})
    except CanonicalizationError:
        return False
    return True


def _param_spec(decl: Mapping[str, Any]) -> ParamSpec:
    element = decl.get("element_kind")
    return ParamSpec(
        name=decl["name"],
        kind=ParamKind(decl["kind"]),
        required=bool(decl["required"]),
        default=decl.get("default"),
        order_sensitive=bool(decl.get("order_sensitive", True)),
        path_root=decl.get("path_root"),
        element_kind=ParamKind(element) if element else None,
    )


def _kind_for(pname: str, schema: Mapping[str, Any], rules: ManifestRules) -> str:
    jtype = schema.get("type")
    if not isinstance(jtype, str):
        raise ManifestBuildError("unsupported_param_type", f"{pname}:{jtype!r}")
    kind = rules.type_map.get(jtype)
    if kind is None:
        raise ManifestBuildError("unsupported_param_type", f"{pname}:{jtype}")
    if kind == "str" and rules.path_name_re.search(pname):
        return "path"
    return kind


def tool_decl_from_bfcl(
    fn: Mapping[str, Any], rules: ManifestRules, *, canonical_name: str | None = None
) -> tuple[ToolDecl, tuple[str, ...]]:
    """Convert one BFCL function schema into a :class:`ToolDecl`.

    Returns the declaration plus the names of parameters whose upstream default
    was dropped because it does not type-check against the declared kind.
    Parameter order follows the schema's declaration order, which is the order
    the model sees and the order positional calls bind against.

    ``canonical_name`` is the sanitised identifier the tool is registered and
    called under; it defaults to sanitising the upstream name on its own, and a
    caller with a whole prompt's tool surface passes the collision-resolved
    name from :func:`tool_name_map`.  The effect rule is always applied to the
    *upstream* name, which is what it was declared against.
    """
    name = fn.get("name")
    if not isinstance(name, str) or not name:
        raise ManifestBuildError("missing_tool_name", repr(name))
    safe_name = canonical_name or sanitize_tool_name(name)
    parameters = fn.get("parameters") or {}
    props = parameters.get("properties") or {}
    if not isinstance(props, Mapping):
        raise ManifestBuildError("malformed_properties", name)
    required = parameters.get("required") or []
    if not isinstance(required, list):
        raise ManifestBuildError("malformed_required", name)
    required_set = set(required)
    unique_key = rules.param_kinds["unique_items_key"]
    items_key = rules.param_kinds["items_key"]
    check_defaults = bool(rules.param_kinds["keep_default_only_if_type_checks"])

    params: list[dict[str, Any]] = []
    dropped: list[str] = []
    for pname, pschema in props.items():
        pschema = pschema or {}
        kind = _kind_for(pname, pschema, rules)
        is_required = pname in required_set
        element_kind: str | None = None
        order_sensitive = True
        if kind == "list":
            items = pschema.get(items_key) or {}
            itype = items.get("type") if isinstance(items, Mapping) else None
            if isinstance(itype, str):
                element_kind = rules.type_map.get(itype)
            order_sensitive = not bool(pschema.get(unique_key))
        decl = param_decl(
            pname,
            kind,
            required=is_required,
            default=None if is_required else pschema.get("default"),
            order_sensitive=order_sensitive,
            element_kind=element_kind,
            path_root=rules.path_root if kind == "path" else None,
        )
        if check_defaults and decl["default"] is not None and not _probe_default_ok(decl, None):
            dropped.append(pname)
            decl = dict(decl, default=None)
        params.append(decl)

    return (
        ToolDecl(
            name=safe_name,
            effect=classify_effect(name, rules),
            resource=rules.tools["resource"],
            expected_duration_ms=float(rules.tools["expected_duration_ms"]),
            params=tuple(params),
            description=fn.get("description", "") or "",
            original_name=name,
        ),
        tuple(dropped),
    )


# --------------------------------------------------------------------------- #
# BFCL -> PromptRecord
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _render_user_message(question: Any) -> str:
    """Render a BFCL ``question`` field as one user message.

    The four single-turn categories carry exactly one turn.  Any other shape,
    or any non-user role, is a conversion failure rather than something to
    paper over.
    """
    if not isinstance(question, list) or len(question) != 1:
        raise ManifestBuildError("unexpected_question_shape", f"turns={len(question) if isinstance(question, list) else type(question).__name__}")
    turn = question[0]
    if not isinstance(turn, list) or not turn:
        raise ManifestBuildError("unexpected_question_shape", "empty turn")
    parts: list[str] = []
    for msg in turn:
        role = msg.get("role")
        if role != "user":
            raise ManifestBuildError("unexpected_role", str(role))
        content = msg.get("content")
        if not isinstance(content, str):
            raise ManifestBuildError("non_string_content", type(content).__name__)
        parts.append(content.strip())
    text = "\n\n".join(p for p in parts if p)
    if not text:
        raise ManifestBuildError("empty_user_message", "")
    return text


def _closed_literal(value: Any, rules: ManifestRules, depth: int = 0) -> None:
    """Reject anything that is not a statically closed literal."""
    if depth > 8:
        raise ManifestBuildError("non_literal_argument", "nesting_too_deep")
    if value is None or isinstance(value, bool) or isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestBuildError("non_literal_argument", "non_finite")
        return
    if isinstance(value, str):
        if rules.call_expression_re.match(value):
            raise ManifestBuildError("non_literal_argument", f"call_expression:{value}")
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _closed_literal(item, rules, depth + 1)
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ManifestBuildError("non_literal_argument", f"non_string_key:{key!r}")
            _closed_literal(item, rules, depth + 1)
        return
    raise ManifestBuildError("non_literal_argument", type(value).__name__)


def _expected_calls_from_bfcl(
    ground_truth: Sequence[Mapping[str, Any]],
    registry: EffectRegistry,
    rules: ManifestRules,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Normalise BFCL ``ground_truth`` into ``{"tool":…, "args":{…}}`` calls.

    Returns ``(calls, alternatives_per_call, omitted_params)``.  For each
    parameter the *first* alternative is kept; every alternative is preserved
    for the record's ``meta``.  A non-required parameter whose alternatives
    admit ``""`` is omitted, matching BFCL's own checker semantics.

    Tool names are resolved *through the registry*, which knows both the
    sanitised identifier and the upstream spelling (registered as an alias), so
    the emitted call always names the tool the model will actually call.
    """
    take_first = bool(rules.ground_truth["first_alternative"])
    if not take_first:
        raise ManifestBuildError("unsupported_rule", "ground_truth.first_alternative must be true")
    omit_empty = bool(rules.ground_truth["omit_optional_when_empty_alternative"])
    calls: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    omitted: list[str] = []
    for entry in ground_truth:
        if not isinstance(entry, Mapping) or len(entry) != 1:
            raise ManifestBuildError("malformed_ground_truth", _dumps(entry)[:120])
        (tool_name, argmap), = entry.items()
        if not isinstance(argmap, Mapping):
            raise ManifestBuildError("malformed_ground_truth", tool_name)
        spec = registry.get(tool_name)
        safe_name = spec.name if spec is not None else sanitize_tool_name(tool_name)
        args: dict[str, Any] = {}
        alts: dict[str, Any] = {}
        for pname, raw_alts in argmap.items():
            alt_list = list(raw_alts) if isinstance(raw_alts, list) else [raw_alts]
            alts[pname] = alt_list
            if not alt_list:
                raise ManifestBuildError("empty_alternatives", f"{tool_name}.{pname}")
            pspec = spec.param(pname) if spec is not None else None
            if omit_empty and pspec is not None and not pspec.required and "" in alt_list:
                omitted.append(f"{safe_name}.{pname}")
                continue
            value = alt_list[0]
            _closed_literal(value, rules)
            args[pname] = value
        calls.append({"tool": safe_name, "args": args})
        alternatives.append({"tool": safe_name, "arguments": alts})
    return calls, alternatives, omitted


def _prompt_record_from_bfcl(
    prompt: Mapping[str, Any],
    answer: Mapping[str, Any],
    category: str,
    rules: ManifestRules,
) -> PromptRecord:
    prompt_id = prompt.get("id")
    if not isinstance(prompt_id, str) or not prompt_id:
        raise ManifestBuildError("missing_prompt_id", repr(prompt_id))

    functions = prompt.get("function") or []
    if len(functions) < int(rules.ground_truth["require_min_tools"]):
        raise ManifestBuildError("no_declared_tools", str(len(functions)))

    names: list[str] = []
    for fn in functions:
        fn_name = fn.get("name")
        if not isinstance(fn_name, str) or not fn_name:
            raise ManifestBuildError("missing_tool_name", repr(fn_name))
        names.append(fn_name)
    name_map = tool_name_map(names)

    decls: list[ToolDecl] = []
    dropped_defaults: list[str] = []
    for fn in functions:
        decl, dropped = tool_decl_from_bfcl(
            fn, rules, canonical_name=name_map[fn["name"]]
        )
        decls.append(decl)
        dropped_defaults.extend(f"{decl.name}.{p}" for p in dropped)

    seen: set[str] = set()
    for decl in decls:
        if decl.name in seen:
            raise ManifestBuildError("duplicate_tool", decl.name)
        seen.add(decl.name)

    user_message = _render_user_message(prompt.get("question"))

    ground_truth = answer.get("ground_truth") or []
    if len(ground_truth) < int(rules.ground_truth["require_min_calls"]):
        raise ManifestBuildError("no_ground_truth_calls", str(len(ground_truth)))

    registry = registry_for_decls(decls)
    calls, alternatives, omitted = _expected_calls_from_bfcl(ground_truth, registry, rules)

    if bool(rules.ground_truth["require_canonicalizable"]):
        for call in calls:
            try:
                canonicalize(registry, call["tool"], call["args"])
            except CanonicalizationError as exc:
                raise ManifestBuildError(
                    f"canonicalize:{exc.reason}", f"{call['tool']}:{exc.detail}"
                ) from None

    effects = {decl.name: decl.effect for decl in decls}
    _aliases, skipped_aliases = alias_plan(decls)
    meta = {
        "answer_alternatives": alternatives,
        "bfcl_file": rules.categories[category],
        "call_effects": [effects.get(c["tool"], "") for c in calls],
        "dropped_defaults": sorted(dropped_defaults),
        "n_declared_tools": len(decls),
        "n_expected_calls": len(calls),
        "omitted_optional_params": sorted(omitted),
        "sanitised_tool_names": sorted(d.name for d in decls if d.alias_candidate),
        "skipped_aliases": list(skipped_aliases),
        "tool_effects": dict(sorted(effects.items())),
        "tool_name_map": {d.name: d.upstream_name for d in sorted(decls, key=lambda d: d.name)},
    }
    return PromptRecord(
        prompt_id=prompt_id,
        category=category,
        source=rules.bfcl["source"],
        source_id=prompt_id,
        user_message=user_message,
        tools=tuple(decls),
        expected_calls=tuple(calls),
        multi_turn=False,
        meta=meta,
    )


def build_bfcl_candidates(
    vendor_dir: str | Path, rules: ManifestRules
) -> tuple[list[PromptRecord], list[DropRecord]]:
    """Convert every BFCL prompt of the four single-turn categories.

    Returns the accepted candidates (sorted by category then prompt id) and one
    :class:`DropRecord` per rejected prompt.
    """
    root = Path(vendor_dir) / rules.bfcl["data_dir"]
    answers_dir = root / rules.bfcl["answer_subdir"]
    accepted: list[PromptRecord] = []
    drops: list[DropRecord] = []
    for category in sorted(rules.categories):
        filename = rules.categories[category]
        prompt_path = root / filename
        answer_path = answers_dir / filename
        for path in (prompt_path, answer_path):
            if not path.exists():
                raise ManifestBuildError("missing_vendor_file", str(path))
        prompts = _read_jsonl(prompt_path)
        answers = {a["id"]: a for a in _read_jsonl(answer_path)}
        for prompt in prompts:
            pid = str(prompt.get("id", ""))
            answer = answers.get(pid)
            if answer is None:
                drops.append(DropRecord(pid, category, "missing_ground_truth", filename))
                continue
            try:
                accepted.append(_prompt_record_from_bfcl(prompt, answer, category, rules))
            except ManifestBuildError as exc:
                drops.append(DropRecord(pid, category, exc.reason, exc.detail))
    accepted.sort(key=lambda r: (r.category, r.prompt_id))
    drops.sort(key=lambda d: (d.category, d.prompt_id))
    return accepted, drops


# --------------------------------------------------------------------------- #
# split selection
# --------------------------------------------------------------------------- #


def select_splits(
    candidates: Sequence[PromptRecord], rules: ManifestRules, seed: int
) -> dict[str, list[PromptRecord]]:
    """Deterministically split candidates into ``dev`` and ``test``.

    Per category, in sorted category order: sort candidate prompt ids
    lexicographically, draw ``dev + test`` of them with a single seeded
    :class:`random.Random`, then assign the first ``dev_per_category`` draws to
    dev and the rest to test.  Dev and test are disjoint by construction.
    """
    n_dev = int(rules.selection["dev_per_category"])
    n_test = int(rules.selection["test_per_category"])
    by_id = {r.prompt_id: r for r in candidates}
    by_category: dict[str, list[str]] = {}
    for record in candidates:
        by_category.setdefault(record.category, []).append(record.prompt_id)

    rng = random.Random(seed)
    splits: dict[str, list[PromptRecord]] = {"dev": [], "test": []}
    for category in sorted(rules.categories):
        ids = sorted(by_category.get(category, ()))
        need = n_dev + n_test
        if len(ids) < need:
            raise ManifestBuildError(
                "insufficient_candidates", f"{category}: {len(ids)} < {need}"
            )
        drawn = rng.sample(ids, need)
        splits["dev"].extend(by_id[i] for i in drawn[:n_dev])
        splits["test"].extend(by_id[i] for i in drawn[n_dev:need])
    for split in splits.values():
        split.sort(key=lambda r: (r.category, r.prompt_id))
    return splits


# --------------------------------------------------------------------------- #
# ToolSandbox scenario enumeration (static, never executed)
# --------------------------------------------------------------------------- #


def _ast_str_list(node: ast.AST | None) -> tuple[str, ...]:
    if not isinstance(node, ast.List):
        return ()
    return tuple(e.value for e in node.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))


def _ast_user_messages(node: ast.AST | None) -> tuple[str, ...]:
    """User-authored message contents of a ``ScenarioExtension(messages=[…])``."""
    if not isinstance(node, ast.List):
        return ()
    out: list[str] = []
    for element in node.elts:
        if not isinstance(element, ast.Dict):
            continue
        entry: dict[str, ast.AST] = {}
        for key, value in zip(element.keys, element.values, strict=False):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                entry[key.value] = value
        sender = entry.get("sender")
        content = entry.get("content")
        if (
            isinstance(sender, ast.Attribute)
            and sender.attr == "USER"
            and isinstance(content, ast.Constant)
            and isinstance(content.value, str)
        ):
            out.append(content.value)
    return tuple(out)


def extract_toolsandbox_scenarios(
    vendor_dir: str | Path, rules: ManifestRules
) -> list[dict[str, Any]]:
    """Enumerate ToolSandbox base scenarios by reading the registry source.

    Every scenario in ToolSandbox is a ``ScenarioExtension`` literal in one of
    the four modules aggregated by ``tool_sandbox.scenarios.named_scenarios``.
    We parse those literals with :mod:`ast`; the package is never imported and
    never run.  Declared categories are the extension's own ``categories=[…]``
    plus the defaults its module's ``named_*`` builder appends.
    """
    cfg = rules.toolsandbox
    scen_dir = Path(vendor_dir) / cfg["checkout_dir"] / cfg["scenarios_subdir"]
    modules: Mapping[str, Sequence[str]] = cfg["scenario_modules"]
    out: list[dict[str, Any]] = []
    names: set[str] = set()
    for module in sorted(modules):
        path = scen_dir / module
        if not path.exists():
            raise ManifestBuildError("missing_vendor_file", str(path))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        defaults = list(modules[module])
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ScenarioExtension"
            ):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords if kw.arg}
            name_node = kwargs.get("name")
            if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
                raise ManifestBuildError("unnamed_scenario", f"{module}:{node.lineno}")
            name = name_node.value
            if name in names:
                raise ManifestBuildError("duplicate_scenario", name)
            names.add(name)
            declared = [
                e.attr
                for e in getattr(kwargs.get("categories"), "elts", [])
                if isinstance(e, ast.Attribute)
            ]
            categories = declared + [c for c in defaults if c not in declared]
            milestones = kwargs.get("milestones")
            out.append(
                {
                    "base_scenario": name,
                    "module": module,
                    "categories": tuple(categories),
                    "tool_allow_list": _ast_str_list(kwargs.get("tool_allow_list")),
                    "user_messages": _ast_user_messages(kwargs.get("messages")),
                    "n_milestones": len(milestones.elts) if isinstance(milestones, ast.List) else 0,
                    "lineno": node.lineno,
                }
            )
    out.sort(key=lambda s: s["base_scenario"])
    return out


def toolsandbox_family(categories: Sequence[str], rules: ManifestRules) -> str | None:
    """First matching family of the declared category tags, else ``None``."""
    for row in rules.toolsandbox["family_precedence"]:
        if row["category"] in categories:
            return row["family"]
    return None


def select_toolsandbox(
    scenarios: Sequence[Mapping[str, Any]], rules: ManifestRules, seed: int
) -> list[ScenarioRecord]:
    """Preselect a balanced set of ToolSandbox scenarios.

    Per family: permute the family's sorted base scenario names with a seeded
    :class:`random.Random`, then fill the quota variant round by variant round,
    so distinct base scenarios are exhausted before a second variant of any
    base is used.  Nothing here depends on a model outcome.
    """
    cfg = rules.toolsandbox
    per_family = int(cfg["per_family"])
    variants = list(cfg["variants"])
    families = [row["family"] for row in cfg["family_precedence"]]

    by_family: dict[str, list[Mapping[str, Any]]] = {f: [] for f in families}
    for scenario in scenarios:
        family = toolsandbox_family(scenario["categories"], rules)
        if family is not None:
            by_family[family].append(scenario)

    rng = random.Random(seed)
    out: list[ScenarioRecord] = []
    for family in families:
        bases = sorted(by_family[family], key=lambda s: s["base_scenario"])
        if not bases:
            raise ManifestBuildError("empty_family", family)
        order = rng.sample(bases, len(bases))
        capacity = len(order) * len(variants)
        if capacity < per_family:
            raise ManifestBuildError(
                "insufficient_scenarios", f"{family}: {capacity} < {per_family}"
            )
        picked = 0
        for variant in variants:
            for base in order:
                if picked >= per_family:
                    break
                suffix = variant["suffix"]
                out.append(
                    ScenarioRecord(
                        scenario_id=f"{base['base_scenario']}{suffix}",
                        family=family,
                        base_scenario=base["base_scenario"],
                        variant=variant["tag"],
                        module=base["module"],
                        categories=tuple(base["categories"]) + (variant["tag"],),
                        tool_allow_list=tuple(base["tool_allow_list"]),
                        user_messages=tuple(base["user_messages"]),
                        n_milestones=int(base["n_milestones"]),
                        source=cfg["source"],
                        meta={
                            "variant_suffix": suffix,
                            "declared_at_line": int(base["lineno"]),
                        },
                    )
                )
                picked += 1
            if picked >= per_family:
                break
    out.sort(key=lambda r: (r.family, r.scenario_id))
    return out
