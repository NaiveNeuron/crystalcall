"""The stateful ToolSandbox validation arm.

This arm is the experiment's proof that the losslessness invariant holds for a
*stateful, multi-turn* workload in which writes and live reads are interleaved
with eligible snapshot reads.  Every episode is run twice at the same seed --
once with ordinary scheduling, once with speculative scheduling -- and the two
runs are compared on the final assistant text, the authoritative canonical call
trace with result digests, the milestone evaluation and the full world-state
digest.  Any difference is a semantic divergence and is recorded, never dropped.

ToolSandbox is *read*, never run:  ``experiments/vendor/toolsandbox/checkout`` is
parsed with :mod:`ast`.  The package is never imported and never installed -- it
needs ``polars``, ``attrs``, ``networkx``, ``rapidfuzz``, ``dill``, ``pint``,
``geopy`` and ``holidays``, none of which is a CrystalCall runtime dependency,
and its RapidAPI-backed tools are neither reachable nor reproducible.  What this
module reimplements, and the deterministic rule it substitutes for each
non-vendored dependency, is declared as data in ``configs/toolsandbox.yml`` and
documented in ``docs/toolsandbox.md``.

The crux is effect assignment, and it is a pure function of declared data:

* a scenario tool that mutates world state is ``Write``;
* a read that is externally backed is ``LiveRead`` unconditionally;
* a read whose namespaces intersect what *some write in the same scenario* can
  change is ``LiveRead``;
* a read over namespaces that no write in that scenario touches is
  ``SnapshotRead`` whose snapshot id is the content digest of exactly that
  immutable sub-view;
* a read of nothing is ``Pure``.

Because the snapshot id is a content digest resolved against the live world at
canonicalisation time, a future launched against one world version can never be
claimed against another.  Speculative work additionally executes against a
*frozen detached copy* of the immutable sub-view, so a speculative value is a
function of exactly the state its snapshot id names -- by construction, not by
timing luck.
"""

from __future__ import annotations

import ast
import copy
import datetime as dt
import difflib
import functools
import hashlib
import json
import re
import threading
import time
import unicodedata
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonicalize import CanonicalCall, CanonicalizationError, canonical_json, canonicalize
from .effects import EffectKind, EffectRegistry, ParamKind, ParamSpec, ResourceClass, ToolSpec
from .futures import FutureKey, FutureTable, assert_no_write_speculation
from .manifest import ToolDecl, load_scenarios as load_scenario_records
from .omp_bridge import HostToolRouter, ToolImpl, _digest_value as _digest_result

__all__ = [
    "CONFIG_PATH",
    "NAMESPACES",
    "SCHEMA_VERSION",
    "Scenario",
    "ScenarioError",
    "StubModelServer",
    "WorldFrozen",
    "WorldState",
    "compare_replay",
    "derive_tool_effects",
    "effect_for",
    "episode_context",
    "evaluate_milestones",
    "impls_for_scenario",
    "load_config",
    "load_scenarios",
    "registry_for_scenario",
    "replay_episode",
    "run_episode",
    "scenario_effects",
    "snapshot_resolver",
    "stub_programs",
    "verify_against_checkout",
]

SCHEMA_VERSION = "crystalcall-toolsandbox/1"

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/toolsandbox.yml"

#: The world namespaces this adapter models.  ``clock`` is ours (the frozen
#: epoch that replaces upstream's ``datetime.datetime.now()``); the other four
#: are ``DatabaseNamespace`` members that carry tool-visible state.  Upstream's
#: ``SANDBOX`` namespace is the conversation log, which OMP owns here.
NAMESPACES: tuple[str, ...] = ("clock", "setting", "contact", "messaging", "reminder")

_WRITER_METHODS = frozenset({"add_to_database", "remove_from_database", "update_database"})
_READER_METHODS = frozenset({"get_database"})


class ScenarioError(RuntimeError):
    """A scenario could not be loaded or its declaration could not be read."""


class WorldFrozen(RuntimeError):
    """A write was attempted against a frozen (speculative) world view."""


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=4)
def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate ``configs/toolsandbox.yml``.

    The validation is strict about identifier-shaped strings: YAML 1.1 turns a
    bare ``on`` / ``off`` / ``yes`` / ``no`` scalar into a boolean, and a tool
    parameter silently renamed to ``True`` would produce a wrong tool schema and
    a wrong canonical identity.  It has to fail here, loudly.
    """
    import yaml

    p = Path(path or CONFIG_PATH)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or int(raw.get("schema", 0)) != 1:
        raise ScenarioError(f"{p}: expected a mapping with schema: 1")
    for key in (
        "arms",
        "checkout",
        "effect_rule",
        "milestones",
        "snapshot",
        "speculation",
        "stub",
        "substitutions",
        "tools",
        "user_turns",
        "world",
    ):
        if key not in raw:
            raise ScenarioError(f"{p}: missing top-level key {key!r}")
    tools: dict[str, Any] = {}
    for decl in raw["tools"]:
        name = decl.get("name")
        if not isinstance(name, str) or not name.isidentifier():
            raise ScenarioError(f"{p}: tool name {name!r} is not an identifier (quote it in YAML)")
        if name in tools:
            raise ScenarioError(f"{p}: duplicate tool declaration {name!r}")
        if decl.get("declared_class") not in ("write", "read", "pure"):
            raise ScenarioError(f"{p}: {name}: bad declared_class {decl.get('declared_class')!r}")
        for ns in list(decl["reads"]) + list(decl["writes"]):
            if ns not in NAMESPACES:
                raise ScenarioError(f"{p}: {name}: unknown namespace {ns!r}")
        for param in decl["params"]:
            pname = param.get("name")
            if not isinstance(pname, str) or not pname.isidentifier():
                raise ScenarioError(
                    f"{p}: {name}: parameter name {pname!r} is not an identifier (quote it in YAML)"
                )
            ParamKind(param["kind"])
        tools[name] = decl
    for tool, fillers in raw["stub"]["required_args"].items():
        for key in fillers:
            if not isinstance(key, str):
                raise ScenarioError(
                    f"{p}: stub.required_args.{tool}: argument name {key!r} is not a string "
                    "(quote it in YAML)"
                )
    raw["_tools_by_name"] = tools
    return raw


def checkout_dir(cfg: Mapping[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return ROOT / str(cfg["checkout"]["path"])


# --------------------------------------------------------------------------- #
# static derivation of read/write footprints from the checkout
# --------------------------------------------------------------------------- #


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        n.name: n
        for n in tree.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _dotted(node: ast.AST) -> str:
    parts: list[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    return ".".join(reversed(parts))


def _direct_footprint(fn: ast.AST) -> tuple[set[str], set[str], set[str], set[str]]:
    """``(reads, writes, callees, markers)`` for one function body."""
    reads: set[str] = set()
    writes: set[str] = set()
    callees: set[str] = set()
    markers: set[str] = set()
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in _WRITER_METHODS | _READER_METHODS:
            found: set[str] = set()
            for arg in list(node.args) + [kw.value for kw in node.keywords]:
                for sub in ast.walk(arg):
                    if isinstance(sub, ast.Attribute) and _dotted(sub).startswith(
                        "DatabaseNamespace."
                    ):
                        found.add(sub.attr.lower())
            target = writes if func.attr in _WRITER_METHODS else reads
            # An unresolvable namespace is a hard failure, never a silent pass.
            target.update(found or {"__unresolved__"})
        if isinstance(func, ast.Name):
            callees.add(func.id)
        dotted = _dotted(func)
        if dotted.startswith("datetime.datetime.now"):
            markers.add("clock")
        if dotted.startswith("requests."):
            markers.add("external")
    return reads, writes, callees, markers


def derive_tool_effects(checkout: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Re-derive each registered tool's read/write footprint from the checkout.

    Walks the transitive call graph of every ``@register_as_tool`` function over
    the module-level functions of ``tool_sandbox/tools/*.py`` and records which
    ``ExecutionContext`` database methods it reaches, on which namespace.  This
    is the derivation that produced the ``tools:`` table in
    ``configs/toolsandbox.yml``; :func:`verify_against_checkout` compares them.
    """
    root = Path(checkout or checkout_dir()) / "tool_sandbox/tools"
    funcs: dict[str, ast.AST] = {}
    registered: set[str] = set()
    for path in sorted(root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for name, fn in _module_functions(tree).items():
            if name in funcs:
                raise ScenarioError(f"duplicate tool-module function {name!r}")
            funcs[name] = fn
            for deco in fn.decorator_list:
                if isinstance(deco, ast.Call) and getattr(deco.func, "id", "") == "register_as_tool":
                    registered.add(name)
    direct = {name: _direct_footprint(fn) for name, fn in funcs.items()}

    def closure(name: str, seen: frozenset[str] = frozenset()) -> tuple[set[str], set[str], set[str]]:
        if name not in direct or name in seen:
            return set(), set(), set()
        seen = seen | {name}
        reads, writes, callees, markers = direct[name]
        r, w, m = set(reads), set(writes), set(markers)
        for callee in sorted(callees):
            cr, cw, cm = closure(callee, seen)
            r |= cr
            w |= cw
            m |= cm
        return r, w, m

    out: dict[str, dict[str, Any]] = {}
    for name in sorted(registered):
        reads, writes, markers = closure(name)
        if "__unresolved__" in reads | writes:
            raise ScenarioError(f"{name}: unresolvable DatabaseNamespace in the checkout source")
        if "clock" in markers:
            reads.add("clock")
        writes.discard("sandbox")
        reads.discard("sandbox")
        out[name] = {
            "reads": sorted(reads),
            "writes": sorted(writes),
            "declared_class": "write" if writes else ("read" if reads else "pure"),
            "external": "external" in markers,
        }
    return out


def verify_against_checkout(
    cfg: Mapping[str, Any] | None = None, checkout: str | Path | None = None
) -> list[str]:
    """Mismatches between the declared ``tools:`` table and the checkout.

    An empty list means the declaration and its source agree.
    """
    cfg = cfg or load_config()
    derived = derive_tool_effects(checkout)
    problems: list[str] = []
    for decl in cfg["tools"]:
        name = decl["name"]
        got = derived.get(name)
        if got is None:
            problems.append(f"{name}: declared but not registered in the checkout")
            continue
        for key in ("reads", "writes", "declared_class", "external"):
            want = decl[key]
            want = sorted(want) if isinstance(want, list) else want
            if want != got[key]:
                problems.append(f"{name}.{key}: config {want!r} != checkout {got[key]!r}")
    return problems


# --------------------------------------------------------------------------- #
# static extraction of the scenario declaration
# --------------------------------------------------------------------------- #

_ROLE_VALUES = {
    "SYSTEM": "system",
    "USER": "user",
    "AGENT": "agent",
    "EXECUTION_ENVIRONMENT": "execution_environment",
}


def _deterministic_uuid(payload: str) -> str:
    """``tool_sandbox.common.utils.deterministic_uuid``, reimplemented."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, payload))


class _MilestoneExprEvaluator:
    """Closed evaluator for the handful of expressions milestone targets use.

    Every accepted form is listed here; anything else yields ``unresolved`` so
    the milestone is reported indeterminate instead of silently mis-evaluated.
    """

    def __init__(self, epoch: float) -> None:
        self.epoch = epoch
        self.now = dt.datetime.fromtimestamp(epoch, dt.UTC)

    # -- helpers named by upstream scenario modules -------------------------- #
    def _tomorrow(self) -> dt.datetime:
        return self.now + dt.timedelta(days=1)

    def _next_iso_weekday(self, weekday: int) -> dt.datetime:
        delta = (weekday - self.now.isoweekday()) % 7
        return self.now + dt.timedelta(days=delta or 7)

    def eval(self, node: ast.AST) -> Any:
        try:
            return ast.literal_eval(node)
        except (ValueError, SyntaxError, TypeError):
            pass
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node)
            if dotted.startswith("RoleType."):
                return _ROLE_VALUES.get(node.attr, node.attr.lower())
            base = self.eval(node.value)
            if isinstance(base, dt.datetime) and node.attr in (
                "year",
                "month",
                "day",
                "hour",
                "minute",
                "second",
            ):
                return getattr(base, node.attr)
            return _Unresolved(ast.unparse(node))
        if isinstance(node, ast.Call):
            return self._call(node)
        return _Unresolved(ast.unparse(node))

    def _call(self, node: ast.Call) -> Any:
        name = _dotted(node.func)
        args = [self.eval(a) for a in node.args]
        kwargs = {k.arg: self.eval(k.value) for k in node.keywords if k.arg}
        if any(isinstance(v, _Unresolved) for v in list(args) + list(kwargs.values())):
            return _Unresolved(ast.unparse(node))
        if name == "deterministic_uuid":
            payload = kwargs.get("payload", args[0] if args else None)
            return _deterministic_uuid(str(payload))
        if name == "json.dumps":
            return args[0] if args else None
        if name == "get_tomorrow_datetime":
            return self._tomorrow()
        if name == "get_next_iso_weekday_datetime":
            return self._next_iso_weekday(int(kwargs.get("next_iso_weekday", args[0] if args else 1)))
        if name == "datetime.datetime":
            try:
                return dt.datetime(*args, **kwargs, tzinfo=dt.UTC) if "tzinfo" not in kwargs else dt.datetime(*args, **kwargs)
            except (TypeError, ValueError):
                return _Unresolved(ast.unparse(node))
        if name.endswith(".timestamp"):
            base = self.eval(node.func.value) if isinstance(node.func, ast.Attribute) else None
            if isinstance(base, dt.datetime):
                return base.timestamp()
            return _Unresolved(ast.unparse(node))
        return _Unresolved(ast.unparse(node))


@dataclass(frozen=True, slots=True)
class _Unresolved:
    """A milestone target expression this adapter deliberately does not guess."""

    source: str


def _kwarg(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _target_dataframe(node: ast.AST, ev: _MilestoneExprEvaluator) -> dict[str, Any] | None:
    if not (isinstance(node, ast.Call) and _dotted(node.func).endswith("DataFrame")):
        return None
    if not node.args or not isinstance(node.args[0], ast.Dict):
        return {}
    out: dict[str, Any] = {}
    body = node.args[0]
    for key, value in zip(body.keys, body.values, strict=False):
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            out[key.value] = ev.eval(value)
    return out


def _constraints(node: ast.AST | None, ev: _MilestoneExprEvaluator) -> list[dict[str, Any]]:
    if not isinstance(node, ast.List):
        return []
    out: list[dict[str, Any]] = []
    for entry in node.elts:
        if not isinstance(entry, ast.Call):
            continue
        cons_node = _kwarg(entry, "snapshot_constraints")
        cons: list[dict[str, Any]] = []
        if isinstance(cons_node, ast.List):
            for c in cons_node.elts:
                if not isinstance(c, ast.Call):
                    continue
                ns = _kwarg(c, "database_namespace")
                measure = _kwarg(c, "snapshot_constraint")
                target = _kwarg(c, "target_dataframe")
                reference = _kwarg(c, "reference_milestone_node_index")
                raw_target = _target_dataframe(target, ev) if target is not None else None
                cons.append(
                    {
                        "namespace": ns.attr.lower() if isinstance(ns, ast.Attribute) else None,
                        "measure": ast.unparse(measure) if measure is not None else None,
                        "reference": (
                            reference.value
                            if isinstance(reference, ast.Constant)
                            else None
                        ),
                        "target": _serialise_target(raw_target),
                    }
                )
        out.append({"constraints": cons})
    return out


def _serialise_target(target: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if target is None:
        return None
    out: dict[str, Any] = {}
    for key, value in target.items():
        if isinstance(value, _Unresolved):
            out[key] = {"unresolved": value.source}
        elif isinstance(value, dt.datetime):
            out[key] = {"value": value.timestamp()}
        else:
            out[key] = {"value": value}
    return out


def _user_messages(node: ast.AST | None, cfg: Mapping[str, Any]) -> tuple[tuple[str, ...], str | None]:
    """``(USER -> AGENT contents, SYSTEM -> USER task text)``."""
    if not isinstance(node, ast.List):
        return (), None
    users: list[str] = []
    goal: str | None = None
    for element in node.elts:
        if not isinstance(element, ast.Dict):
            continue
        entry: dict[str, ast.AST] = {}
        for key, value in zip(element.keys, element.values, strict=False):
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                entry[key.value] = value
        sender = entry.get("sender")
        content = entry.get("content")
        role = sender.attr if isinstance(sender, ast.Attribute) else None
        if role == "USER" and isinstance(content, ast.Constant):
            users.append(content.value)
        elif role == "SYSTEM" and content is not None:
            goal = _goal_text(content)
    return tuple(users), goal


def _flatten_add(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _flatten_add(node.left) + _flatten_add(node.right)
    return [node]


def _goal_text(content: ast.AST) -> str:
    """The task text of a SYSTEM -> USER instruction, preamble dropped.

    The instruction is always ``USER_INSTRUCTION + <task literals>``; the name
    ``USER_INSTRUCTION`` is the simulator-facing role-play preamble and carries
    no user content, so it is dropped.  ``deterministic_uuid(...)`` operands are
    evaluated (they are pure ``uuid5``); f-string interpolations that depend on
    the wall clock are rendered as their source between braces so that nothing
    is silently invented.
    """
    parts: list[str] = []
    for node in _flatten_add(content):
        if isinstance(node, ast.Name) and node.id == "USER_INSTRUCTION":
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            parts.append(node.value)
        elif isinstance(node, ast.JoinedStr):
            for piece in node.values:
                if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                    parts.append(piece.value)
                else:
                    parts.append("{" + ast.unparse(piece) + "}")
        elif isinstance(node, ast.Call) and _dotted(node.func) == "deterministic_uuid":
            arg = node.keywords[0].value if node.keywords else node.args[0]
            parts.append(_deterministic_uuid(str(ast.literal_eval(arg))))
        else:
            parts.append("{" + ast.unparse(node) + "}")
    return "".join(parts)


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _scripted_follow_up(goal: str, cfg: Mapping[str, Any]) -> str:
    """One deterministic follow-up user turn derived from the task text."""
    rules = cfg["user_turns"]
    text = goal.strip()
    for prefix in rules["strip_prefixes"]:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    drops = tuple(rules["drop_sentences"])
    kept = [
        s.strip()
        for s in _SENTENCE_SPLIT.split(text)
        if s.strip() and not any(d.lower() in s.lower() for d in drops)
    ]
    out = " ".join(kept).strip()
    return out or text


# --------------------------------------------------------------------------- #
# the scenario
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Scenario:
    """One ToolSandbox scenario, resolved into everything an episode needs."""

    scenario_id: str
    family: str
    user_turns: tuple[str, ...]
    tools: tuple[ToolDecl, ...]
    milestones: tuple[dict, ...]
    initial_world: dict
    meta: dict

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(self.meta.get("categories", ()))

    @property
    def minefields(self) -> tuple[dict, ...]:
        return tuple(self.meta.get("minefields", ()))

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.tools)

    @property
    def surface_writes(self) -> frozenset[str]:
        """Every namespace some tool of *this* scenario's surface can change."""
        cfg = load_config()
        out: set[str] = set()
        for name in self.tool_names:
            out.update(cfg["_tools_by_name"][name]["writes"])
        return frozenset(out)


def _tool_decl(name: str, effect: EffectKind, cfg: Mapping[str, Any]) -> ToolDecl:
    decl = cfg["_tools_by_name"][name]
    params: list[dict[str, Any]] = []
    for p in decl["params"]:
        params.append(
            {
                "name": p["name"],
                "kind": p["kind"],
                "required": bool(p["required"]),
                "default": p.get("default"),
                "order_sensitive": True,
                "element_kind": None,
                "path_root": None,
            }
        )
    return ToolDecl(
        name=name,
        effect=effect.value,
        resource=decl["resource"],
        expected_duration_ms=float(decl["expected_duration_ms"]),
        params=tuple(params),
        description=_describe(name, decl, effect),
        original_name=name,
    )


def _describe(name: str, decl: Mapping[str, Any], effect: EffectKind) -> str:
    reads = ", ".join(decl["reads"]) or "nothing"
    writes = ", ".join(decl["writes"]) or "nothing"
    return (
        f"ToolSandbox `{name}`. Effect {effect.value}; reads {reads}; writes {writes}"
        f"{'; externally backed' if decl['external'] else ''}."
    )


def effect_for(name: str, surface_writes: Iterable[str], cfg: Mapping[str, Any] | None = None) -> EffectKind:
    """The declared effect of ``name`` inside a scenario that writes ``surface_writes``.

    This is the whole effect-assignment rule, and it is total: every branch
    returns, and the only branches that return an eligible effect require an
    empty read set or a read set provably disjoint from every in-scenario write.
    """
    cfg = cfg or load_config()
    decl = cfg["_tools_by_name"].get(name)
    if decl is None:
        raise ScenarioError(f"tool {name!r} is not declared in configs/toolsandbox.yml")
    if decl["declared_class"] == "write":
        return EffectKind.WRITE
    if decl["external"]:
        return EffectKind.LIVE_READ
    reads = frozenset(decl["reads"])
    if reads & frozenset(surface_writes):
        return EffectKind.LIVE_READ
    if reads:
        return EffectKind.SNAPSHOT_READ
    return EffectKind.PURE


def scenario_effects(sc: Scenario) -> dict[str, EffectKind]:
    """Effect per tool of this scenario's surface, in sorted tool order."""
    cfg = load_config()
    writes = sc.surface_writes
    return {name: effect_for(name, writes, cfg) for name in sorted(sc.tool_names)}


def load_scenarios(manifest: str | Path) -> list[Scenario]:
    """Every frozen scenario of ``experiments/manifests/toolsandbox.jsonl``.

    The manifest pins *which* scenarios run (and their families, tool surface
    and first user message).  Everything else -- the milestones, the minefields,
    the base world variant and the scripted follow-up turn -- is read here,
    statically, from the pinned checkout.
    """
    cfg = load_config()
    records = load_scenario_records(manifest)
    ev = _MilestoneExprEvaluator(float(cfg["world"]["epoch_unix"]))
    scen_dir = checkout_dir(cfg) / "tool_sandbox/scenarios"
    trees: dict[str, dict[int, ast.Call]] = {}
    out: list[Scenario] = []
    for rec in records:
        module = rec.module
        if module not in trees:
            path = scen_dir / module
            if not path.exists():
                raise ScenarioError(f"missing checkout module {path}")
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            trees[module] = {
                n.lineno: n
                for n in ast.walk(tree)
                if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name)
                and n.func.id == "ScenarioExtension"
            }
        lineno = int(rec.meta.get("declared_at_line", 0))
        node = trees[module].get(lineno)
        if node is None:
            raise ScenarioError(f"{module}:{lineno}: no ScenarioExtension for {rec.scenario_id}")
        name_node = _kwarg(node, "name")
        declared = name_node.value if isinstance(name_node, ast.Constant) else None
        if declared != rec.base_scenario:
            raise ScenarioError(
                f"{module}:{lineno}: declared {declared!r} != manifest {rec.base_scenario!r}"
            )
        base_node = _kwarg(node, "base_scenario")
        base_key = (
            base_node.slice.value
            if isinstance(base_node, ast.Subscript) and isinstance(base_node.slice, ast.Constant)
            else "base"
        )
        milestones = _constraints(_kwarg(node, "milestones"), ev)
        if len(milestones) != rec.n_milestones:
            raise ScenarioError(
                f"{rec.scenario_id}: extracted {len(milestones)} milestones, "
                f"manifest froze {rec.n_milestones}"
            )
        minefields = _constraints(_kwarg(node, "minefields"), ev)
        users, goal = _user_messages(_kwarg(node, "messages"), cfg)
        if not users:
            raise ScenarioError(f"{rec.scenario_id}: no USER -> AGENT message declared")
        turns = list(rec.user_messages or users)
        multi = tuple(cfg["user_turns"]["multi_turn_categories"])
        if goal and any(c in rec.categories for c in multi):
            follow_up = _scripted_follow_up(goal, cfg)
            if follow_up and follow_up not in turns:
                turns.append(follow_up)
        writes: set[str] = set()
        for name in rec.tool_allow_list:
            decl = cfg["_tools_by_name"].get(name)
            if decl is None:
                raise ScenarioError(
                    f"{rec.scenario_id}: tool {name!r} is not declared in configs/toolsandbox.yml"
                )
            writes.update(decl["writes"])
        tools = tuple(
            _tool_decl(name, effect_for(name, writes, cfg), cfg)
            for name in sorted(rec.tool_allow_list)
        )
        out.append(
            Scenario(
                scenario_id=rec.scenario_id,
                family=rec.family,
                user_turns=tuple(turns),
                tools=tools,
                milestones=tuple(milestones),
                initial_world={"base_variant": base_key},
                meta={
                    "categories": tuple(rec.categories),
                    "module": module,
                    "declared_at_line": lineno,
                    "base_scenario": rec.base_scenario,
                    "variant": rec.variant,
                    "minefields": tuple(minefields),
                    "user_goal": goal or "",
                    "scripted_follow_up": len(turns) > len(rec.user_messages or users),
                    "n_milestones": rec.n_milestones,
                },
            )
        )
    return out


# --------------------------------------------------------------------------- #
# world state
# --------------------------------------------------------------------------- #


def _canon_digest(payload: Any) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _fuzzy_score(query: str, choice: str) -> float:
    a, b = _norm_text(query), _norm_text(choice)
    if not a or not b:
        return 100.0 if a == b else 0.0
    return difflib.SequenceMatcher(None, a, b).ratio() * 100.0


def _partial_score(query: str, choice: str) -> float:
    """Best window match, the role ``fuzz.partial_ratio`` plays upstream."""
    a, b = _norm_text(query), _norm_text(choice)
    if not a or not b:
        return 100.0 if a == b else 0.0
    if len(a) > len(b):
        a, b = b, a
    best = 0.0
    for i in range(0, len(b) - len(a) + 1):
        best = max(best, difflib.SequenceMatcher(None, a, b[i : i + len(a)]).ratio() * 100.0)
    return best


class WorldState:
    """Mutable per-episode ToolSandbox world.

    Five namespaces (:data:`NAMESPACES`).  ``clock`` holds the frozen epoch and
    is never written, so a tool whose only state dependency is the clock is an
    eligible snapshot read.  Rows are plain dicts in upstream's column order and
    upstream's row order, including upstream's own duplicate ``message_4`` id.
    """

    __slots__ = (
        "_cfg",
        "_frozen",
        "_immutable",
        "_initial",
        "_lock",
        "_write_seq",
        "epoch",
        "rows",
        "scenario_id",
        "versions",
    )

    def __init__(
        self,
        *,
        scenario_id: str,
        base_variant: str = "base",
        immutable: Iterable[str] = (),
        cfg: Mapping[str, Any] | None = None,
    ) -> None:
        self._cfg = dict(cfg or load_config())
        world = self._cfg["world"]
        self.scenario_id = scenario_id
        self.epoch = float(world["epoch_unix"])
        self.rows: dict[str, list[dict[str, Any]]] = self._build_initial(base_variant)
        self._initial = copy.deepcopy(self.rows)
        self.versions: dict[str, int] = {ns: 0 for ns in NAMESPACES}
        self._write_seq = 0
        self._frozen = False
        self._immutable = frozenset(immutable) or frozenset(NAMESPACES)
        self._lock = threading.RLock()

    # -- construction ------------------------------------------------------- #

    @classmethod
    def from_scenario(cls, sc: Scenario, cfg: Mapping[str, Any] | None = None) -> WorldState:
        cfg = cfg or load_config()
        return cls(
            scenario_id=sc.scenario_id,
            base_variant=str(sc.initial_world.get("base_variant", "base")),
            immutable=frozenset(NAMESPACES) - sc.surface_writes,
            cfg=cfg,
        )

    def _build_initial(self, base_variant: str) -> dict[str, list[dict[str, Any]]]:
        world = self._cfg["world"]
        variants = world["base_variants"]
        if base_variant not in variants:
            raise ScenarioError(f"unknown base world variant {base_variant!r}")
        setting = dict(world["setting_defaults"])
        payload = setting.pop("device_id_payload")
        setting = {
            "device_id": _deterministic_uuid(payload),
            "cellular": bool(setting["cellular"]),
            "wifi": bool(setting["wifi"]),
            "location_service": bool(setting["location_service"]),
            "low_battery_mode": bool(setting["low_battery_mode"]),
            "latitude": float(setting["latitude"]),
            "longitude": float(setting["longitude"]),
        }
        setting.update({k: bool(v) for k, v in (variants[base_variant] or {}).items()})
        epoch = float(world["epoch_unix"])
        contacts = [
            {
                "person_id": _deterministic_uuid(c["payload"]),
                "name": c["name"],
                "phone_number": c["phone_number"],
                "relationship": c["relationship"],
                "is_self": bool(c["is_self"]),
            }
            for c in world["contacts"]
        ]
        by_name = {c["name"]: c["person_id"] for c in contacts}
        messages = [
            {
                "message_id": _deterministic_uuid(m["payload"]),
                "sender_person_id": by_name.get(m["sender"]) if m["sender"] else None,
                "sender_phone_number": m["sender_phone_number"],
                "recipient_person_id": by_name.get(m["recipient"]) if m["recipient"] else None,
                "recipient_phone_number": m["recipient_phone_number"],
                "content": m["content"],
                "creation_timestamp": epoch + float(m["offset_seconds"]),
            }
            for m in world["messages"]
        ]
        reminders = [
            {
                "reminder_id": _deterministic_uuid(r["payload"]),
                "content": r["content"],
                "creation_timestamp": epoch + float(r["creation_offset_seconds"]),
                "reminder_timestamp": epoch + float(r["reminder_offset_seconds"]),
                "latitude": r["latitude"],
                "longitude": r["longitude"],
            }
            for r in world["reminders"]
        ]
        return {
            "clock": [{"epoch": epoch}],
            "setting": [setting],
            "contact": contacts,
            "messaging": messages,
            "reminder": reminders,
        }

    # -- identity ----------------------------------------------------------- #

    def view(self, namespaces: Iterable[str] | None = None) -> dict[str, list[dict[str, Any]]]:
        names = sorted(namespaces if namespaces is not None else self._immutable)
        unknown = [n for n in names if n not in NAMESPACES]
        if unknown:
            raise ScenarioError(f"unknown namespace(s) {unknown}")
        with self._lock:
            return {n: copy.deepcopy(self.rows[n]) for n in names}

    def snapshot_id(self, namespaces: Iterable[str] | None = None) -> str:
        """Content digest of exactly the named immutable sub-view.

        With no argument this is the digest of the whole immutable-read view of
        the scenario this world belongs to.  The id changes whenever any row of
        any named namespace changes, which is what makes a stale future
        unclaimable.
        """
        snap = self._cfg["snapshot"]
        names = sorted(namespaces if namespaces is not None else self._immutable)
        digest = _canon_digest({n: self.rows[n] for n in names})
        return f"{snap['prefix']}{'+'.join(names)}@{digest[: int(snap['digest_chars'])]}"

    def digest(self) -> str:
        """Digest of the FULL state, used for cross-run equivalence checks."""
        with self._lock:
            return _canon_digest(
                {"rows": {n: self.rows[n] for n in NAMESPACES}, "versions": self.versions}
            )

    @property
    def frozen(self) -> bool:
        return self._frozen

    @property
    def immutable_namespaces(self) -> frozenset[str]:
        return self._immutable

    def _copy(self, *, frozen: bool, namespaces: Iterable[str] | None) -> WorldState:
        with self._lock:
            clone = object.__new__(WorldState)
            clone._cfg = self._cfg
            clone.scenario_id = self.scenario_id
            clone.epoch = self.epoch
            clone.rows = copy.deepcopy(self.rows)
            clone._initial = copy.deepcopy(self._initial)
            clone.versions = dict(self.versions)
            clone._write_seq = self._write_seq
            clone._frozen = frozen
            clone._immutable = (
                frozenset(namespaces) if namespaces is not None else self._immutable
            )
            clone._lock = threading.RLock()
            return clone

    def clone(self) -> WorldState:
        """An exact, independent, writable copy.

        Same rows, same versions, same write sequence, same immutable-view set,
        so ``clone().digest() == self.digest()`` and ``clone().snapshot_id(ns) ==
        self.snapshot_id(ns)`` for every ``ns`` -- and a write to one is invisible
        to the other.  This is what makes the ordinary-scheduling replay start
        from *exactly* the state the speculative generation started from.
        """
        return self._copy(frozen=False, namespaces=None)

    def freeze(self, namespaces: Iterable[str] | None = None) -> WorldState:
        """A detached, write-refusing copy carrying identical snapshot ids.

        Speculative work runs against this, so a speculative value is a function
        of exactly the state its snapshot id names -- never of a later write that
        landed while the speculation was in flight.
        """
        return self._copy(frozen=True, namespaces=namespaces)

    def changed_namespaces(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(n for n in NAMESPACES if self.rows[n] != self._initial[n])

    # -- dispatch ----------------------------------------------------------- #

    def _declared(self, tool: str) -> Mapping[str, Any]:
        decl = self._cfg["_tools_by_name"].get(tool)
        if decl is None:
            raise ScenarioError(f"tool {tool!r} is not declared in configs/toolsandbox.yml")
        return decl

    def apply_write(self, tool: str, args: Mapping[str, Any]) -> Any:
        """Execute a declared ``Write`` tool against the live world."""
        decl = self._declared(tool)
        if decl["declared_class"] != "write":
            raise ScenarioError(f"{tool!r} is not a write tool")
        if self._frozen:
            raise WorldFrozen(f"refusing to write {tool!r} against a frozen world view")
        with self._lock:
            before = {n: copy.deepcopy(self.rows[n]) for n in decl["writes"]}
            value = _IMPLS[tool](self, dict(args))
            for ns in decl["writes"]:
                if self.rows[ns] != before[ns]:
                    self.versions[ns] += 1
            return value

    def read(self, tool: str, args: Mapping[str, Any]) -> Any:
        """Execute a declared read (or pure) tool."""
        decl = self._declared(tool)
        if decl["declared_class"] == "write":
            raise ScenarioError(f"{tool!r} is a write tool; use apply_write")
        with self._lock:
            return _IMPLS[tool](self, dict(args))

    def call(self, tool: str, args: Mapping[str, Any]) -> Any:
        decl = self._declared(tool)
        if decl["declared_class"] == "write":
            return self.apply_write(tool, args)
        return self.read(tool, args)

    # -- internals used by the tool bodies ---------------------------------- #

    def _next_row_id(self, namespace: str) -> str:
        self._write_seq += 1
        return _deterministic_uuid(f"{namespace}:{self._write_seq}")

    @property
    def setting(self) -> dict[str, Any]:
        return self.rows["setting"][0]

    def _require(self, setting_name: str, message: str, exc: type[Exception]) -> None:
        if not bool(self.setting[setting_name]):
            raise exc(message)


# --------------------------------------------------------------------------- #
# tool implementations (faithful to the checkout, with declared substitutions)
# --------------------------------------------------------------------------- #

_SETTING_TOOL_COLUMN = {
    "wifi": "wifi",
    "cellular_service": "cellular",
    "location_service": "location_service",
    "low_battery_mode": "low_battery_mode",
}


def _get_boolean_setting(w: WorldState, column: str) -> bool:
    return bool(w.setting[column])


def _set_boolean_setting(w: WorldState, column: str, on: bool) -> None:
    if not isinstance(on, bool):
        raise TypeError(f"on must be bool, found {type(on).__name__}")
    if bool(w.setting[column]) == on:
        raise ValueError(f"{column} already {('disabled', 'enabled')[int(on)]}")
    w.setting[column] = on


def _impl_set_setting(column: str) -> Callable[[WorldState, dict[str, Any]], None]:
    def impl(w: WorldState, args: dict[str, Any]) -> None:
        on = args["on"]
        _set_boolean_setting(w, column, bool(on))
        if column == "low_battery_mode" and on:
            for dependent in ("cellular", "wifi", "location_service"):
                try:
                    _set_boolean_setting(w, dependent, False)
                except ValueError:
                    pass

    return impl


def _impl_get_setting(column: str) -> Callable[[WorldState, dict[str, Any]], bool]:
    def impl(w: WorldState, _args: dict[str, Any]) -> bool:
        return _get_boolean_setting(w, column)

    return impl


def _impl_get_current_location(w: WorldState, _args: dict[str, Any]) -> dict[str, float]:
    w._require("location_service", "Location service is not enabled.", PermissionError)
    return {"latitude": float(w.setting["latitude"]), "longitude": float(w.setting["longitude"])}


def _maybe_current_lat_lon(
    w: WorldState, latitude: float | None, longitude: float | None
) -> tuple[float, float]:
    if latitude is not None and longitude is not None:
        return float(latitude), float(longitude)
    if latitude is not None or longitude is not None:
        raise ValueError("Latitude and longitude must both be provided, or both omitted")
    here = _impl_get_current_location(w, {})
    return here["latitude"], here["longitude"]


def _impl_get_current_timestamp(w: WorldState, _args: dict[str, Any]) -> float:
    return float(w.rows["clock"][0]["epoch"])


def _impl_timestamp_to_datetime_info(_w: WorldState, args: dict[str, Any]) -> dict[str, int]:
    moment = dt.datetime.fromtimestamp(float(args["timestamp"]), dt.UTC)
    return {
        "year": moment.year,
        "month": moment.month,
        "day": moment.day,
        "hour": moment.hour,
        "minute": moment.minute,
        "second": moment.second,
        "isoweekday": moment.isoweekday(),
    }


def _impl_datetime_info_to_timestamp(_w: WorldState, args: dict[str, Any]) -> float:
    return dt.datetime(
        int(args["year"]),
        int(args["month"]),
        int(args["day"]),
        int(args["hour"]),
        int(args["minute"]),
        int(args["second"]),
        tzinfo=dt.UTC,
    ).timestamp()


def _impl_shift_timestamp(_w: WorldState, args: dict[str, Any]) -> float:
    delta = dt.timedelta(
        weeks=int(args.get("weeks", 0)),
        days=int(args.get("days", 0)),
        hours=int(args.get("hours", 0)),
        minutes=int(args.get("minutes", 0)),
        seconds=int(args.get("seconds", 0)),
    )
    return (dt.datetime.fromtimestamp(float(args["timestamp"]), dt.UTC) + delta).timestamp()


def _impl_timestamp_diff(_w: WorldState, args: dict[str, Any]) -> dict[str, int]:
    delta = dt.datetime.fromtimestamp(float(args["timestamp_1"]), dt.UTC) - dt.datetime.fromtimestamp(
        float(args["timestamp_0"]), dt.UTC
    )
    return {"days": delta.days, "seconds": delta.seconds}


def _impl_calculate_lat_lon_distance(w: WorldState, args: dict[str, Any]) -> float:
    import math

    radius = float(w._cfg["substitutions"]["distance"]["earth_radius_km"])
    lat0, lon0 = math.radians(float(args["latitude_0"])), math.radians(float(args["longitude_0"]))
    lat1, lon1 = math.radians(float(args["latitude_1"])), math.radians(float(args["longitude_1"]))
    dlat, dlon = lat1 - lat0, lon1 - lon0
    h = math.sin(dlat / 2) ** 2 + math.cos(lat0) * math.cos(lat1) * math.sin(dlon / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(h))


def _unit_entry(w: WorldState, name: str) -> Mapping[str, Any]:
    units = w._cfg["substitutions"]["units"]
    key = _norm_text(name).replace(" ", "_")
    key = units["aliases"].get(key, key)
    entry = units["table"].get(key)
    if entry is None:
        raise ValueError(f"undefined unit {name!r}")
    return entry


def _impl_unit_conversion(w: WorldState, args: dict[str, Any]) -> float:
    src = _unit_entry(w, str(args["from_unit"]))
    dst = _unit_entry(w, str(args["to_unit"]))
    if src["dimension"] != dst["dimension"]:
        raise ValueError(f"cannot convert {args['from_unit']!r} to {args['to_unit']!r}")
    base = float(args["amount"]) * float(src["scale"]) + float(src["offset"])
    return (base - float(dst["offset"])) / float(dst["scale"])


def _impl_search_holiday(w: WorldState, args: dict[str, Any]) -> float | None:
    w._require("wifi", "Wifi is not enabled", ConnectionError)
    table = w._cfg["substitutions"]["holidays"]["table"]
    year = args.get("year")
    if year is None:
        year = dt.datetime.fromtimestamp(w.epoch, dt.UTC).year
    entries = table.get(int(year))
    if not entries:
        return None
    scored = sorted(
        (
            (-_partial_score(str(args["holiday_name"]), str(e["name"])), str(e["name"]), e["date"])
            for e in entries
        )
    )
    best = scored[0]
    if -best[0] <= 90.0:
        return None
    date = best[2]
    return dt.datetime(date.year, date.month, date.day, tzinfo=dt.UTC).timestamp()


def _impl_search_lat_lon(w: WorldState, args: dict[str, Any]) -> str | None:
    w._require("wifi", "Wifi is not enabled", ConnectionError)
    geocode = w._cfg["substitutions"]["geocode"]
    lat, lon = float(args["latitude"]), float(args["longitude"])
    key = f"{lat:.6f},{lon:.6f}"
    if key in geocode["table"]:
        return str(geocode["table"][key])
    return str(geocode["default_address_template"]).format(latitude=lat, longitude=lon)


def _synthetic_place(w: WorldState, location: str, lat: float, lon: float) -> dict[str, Any]:
    seed = int(hashlib.sha256(_norm_text(location).encode("utf-8")).hexdigest()[:4], 16)
    offset = round(seed / 65535.0 - 0.5, 6)
    return {
        "name": location,
        "address": f"{location}, United States",
        "latitude": round(lat + offset, 6),
        "longitude": round(lon + offset, 6),
        "type": "point_of_interest",
        "phone_number": None,
    }


def _impl_search_location_around_lat_lon(w: WorldState, args: dict[str, Any]) -> list[dict[str, Any]]:
    lat, lon = _maybe_current_lat_lon(w, args.get("latitude"), args.get("longitude"))
    w._require("wifi", "Wifi is not enabled", ConnectionError)
    table = w._cfg["substitutions"]["places"]["table"]
    key = _norm_text(str(args["location"]))
    if key in table:
        return [dict(row) for row in table[key]]
    for candidate, rows in sorted(table.items()):
        if _partial_score(key, candidate) > 90.0:
            return [dict(row) for row in rows]
    return [_synthetic_place(w, str(args["location"]), lat, lon)]


def _impl_search_weather_around_lat_lon(w: WorldState, args: dict[str, Any]) -> dict[str, Any]:
    days = args.get("days", 0)
    if not isinstance(days, int) or isinstance(days, bool) or days < 0:
        raise ValueError(f"Days must be positive integer, found days={days!r}.")
    lat, lon = _maybe_current_lat_lon(w, args.get("latitude"), args.get("longitude"))
    w._require("wifi", "Wifi is not enabled", ConnectionError)
    spec = w._cfg["substitutions"]["weather"]
    base = dict(spec["base"])
    cell = float(spec["coordinate_cell_degrees"])
    cells = round(lat / cell) + round(lon / cell)
    shift = float(spec["per_day_delta_c"]) * days + float(spec["coordinate_delta_c"]) * (cells % 5)
    out: dict[str, Any] = {}
    for key, value in sorted(base.items()):
        if key.endswith("temperature"):
            out[key] = round(float(value) + shift, 2)
        else:
            out[key] = value
    out["latitude"] = round(lat, 6)
    out["longitude"] = round(lon, 6)
    out["forecast_offset_days"] = days
    if days == 0:
        out["current_temperature"] = out["average_temperature"]
    return out


# -- searches over state ---------------------------------------------------- #


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    criteria: Sequence[tuple[str, Any, str, float]],
) -> list[dict[str, Any]]:
    """Upstream ``filter_dataframe``: every supplied criterion must match."""
    if all(value is None for _, value, _, _ in criteria):
        raise ValueError(
            "No search criteria are given. At least one search criteria should be provided among "
            f"{tuple(name for name, _, _, _ in criteria)}"
        )
    out = [dict(r) for r in rows]
    for column, value, mode, threshold in criteria:
        if value is None:
            continue
        if mode == "exact":
            out = [r for r in out if r.get(column) == value]
        elif mode == "fuzzy":
            out = [
                r
                for r in out
                if r.get(column) is not None
                and _fuzzy_score(str(value), str(r[column])) >= threshold
            ]
        elif mode == "ge":
            out = [r for r in out if r.get(column) is not None and r[column] >= value]
        elif mode == "le":
            out = [r for r in out if r.get(column) is not None and r[column] <= value]
        else:  # pragma: no cover - defensive
            raise ScenarioError(f"unknown filter mode {mode!r}")
    return out


def _impl_search_contacts(w: WorldState, args: dict[str, Any]) -> list[dict[str, Any]]:
    return _filter_rows(
        w.rows["contact"],
        [
            ("person_id", args.get("person_id"), "exact", 0.0),
            ("name", args.get("name"), "fuzzy", 90.0),
            ("phone_number", args.get("phone_number"), "exact", 0.0),
            ("relationship", args.get("relationship"), "fuzzy", 90.0),
            ("is_self", args.get("is_self"), "exact", 0.0),
        ],
    )


def _impl_search_messages(w: WorldState, args: dict[str, Any]) -> list[dict[str, Any]]:
    return _filter_rows(
        w.rows["messaging"],
        [
            ("message_id", args.get("message_id"), "exact", 0.0),
            ("sender_person_id", args.get("sender_person_id"), "exact", 0.0),
            ("sender_phone_number", args.get("sender_phone_number"), "exact", 0.0),
            ("recipient_person_id", args.get("recipient_person_id"), "exact", 0.0),
            ("recipient_phone_number", args.get("recipient_phone_number"), "exact", 0.0),
            ("content", args.get("content"), "fuzzy", 50.0),
            ("creation_timestamp", args.get("creation_timestamp_lowerbound"), "ge", 0.0),
            ("creation_timestamp", args.get("creation_timestamp_upperbound"), "le", 0.0),
        ],
    )


def _impl_search_reminder(w: WorldState, args: dict[str, Any]) -> list[dict[str, Any]]:
    return _filter_rows(
        w.rows["reminder"],
        [
            ("reminder_id", args.get("reminder_id"), "exact", 0.0),
            ("content", args.get("content"), "fuzzy", 50.0),
            ("creation_timestamp", args.get("creation_timestamp_lowerbound"), "ge", 0.0),
            ("creation_timestamp", args.get("creation_timestamp_upperbound"), "le", 0.0),
            ("reminder_timestamp", args.get("reminder_timestamp_lowerbound"), "ge", 0.0),
            ("reminder_timestamp", args.get("reminder_timestamp_upperbound"), "le", 0.0),
            ("latitude", args.get("latitude"), "exact", 0.0),
            ("longitude", args.get("longitude"), "exact", 0.0),
        ],
    )


# -- writes ----------------------------------------------------------------- #


def _impl_add_reminder(w: WorldState, args: dict[str, Any]) -> str:
    reminder_id = w._next_row_id("reminder")
    w.rows["reminder"].append(
        {
            "reminder_id": reminder_id,
            "content": str(args["content"]),
            "creation_timestamp": float(w.epoch),
            "reminder_timestamp": float(args["reminder_timestamp"]),
            "latitude": args.get("latitude"),
            "longitude": args.get("longitude"),
        }
    )
    return reminder_id


def _impl_modify_reminder(w: WorldState, args: dict[str, Any]) -> None:
    fields = ("content", "reminder_timestamp", "latitude", "longitude")
    if all(args.get(f) is None for f in fields):
        raise ValueError(
            "No update information given. At least one new field should be provided among "
            "[content, reminder_timestamp, latitude, longitude] in order to modify reminder"
        )
    reminder_id = str(args["reminder_id"])
    hits = [r for r in w.rows["reminder"] if r["reminder_id"] == reminder_id]
    if not hits:
        raise LookupError(f"No db entry matching reminder_id={reminder_id!r} found")
    if len(hits) > 1:
        raise ValueError(f"More than 1 entry with reminder_id={reminder_id!r} found")
    row = dict(hits[0])
    row["creation_timestamp"] = float(w.epoch)
    for f in fields:
        if args.get(f) is not None:
            row[f] = args[f]
    # Upstream removes then re-adds, which moves the row to the end.
    w.rows["reminder"] = [r for r in w.rows["reminder"] if r["reminder_id"] != reminder_id]
    w.rows["reminder"].append(row)


def _impl_remove_reminder(w: WorldState, args: dict[str, Any]) -> None:
    reminder_id = str(args["reminder_id"])
    kept = [r for r in w.rows["reminder"] if r["reminder_id"] != reminder_id]
    if len(kept) == len(w.rows["reminder"]):
        raise LookupError(f"No db entry matching reminder_id={reminder_id!r} found")
    w.rows["reminder"] = kept


def _impl_modify_contact(w: WorldState, args: dict[str, Any]) -> None:
    fields = ("name", "phone_number", "relationship", "is_self")
    if all(args.get(f) is None for f in fields):
        raise ValueError(
            "No update information given. At least one new field should be provided among "
            "[name, phone_number, relationship, is_self] in order to modify contact"
        )
    person_id = str(args["person_id"])
    selves = [r for r in w.rows["contact"] if r["is_self"] is True]
    if selves and args.get("is_self") and selves[0]["person_id"] != person_id:
        raise ValueError("Self entry already exists. Cannot add another one.")
    hits = [r for r in w.rows["contact"] if r["person_id"] == person_id]
    if not hits:
        raise LookupError(f"No db entry matching person_id={person_id!r} found")
    if len(hits) > 1:
        raise ValueError(f"More than 1 entry with person_id={person_id!r} found")
    row = dict(hits[0])
    for f in fields:
        if args.get(f) is not None:
            row[f] = args[f]
    w.rows["contact"] = [r for r in w.rows["contact"] if r["person_id"] != person_id]
    w.rows["contact"].append(row)


def _impl_remove_contact(w: WorldState, args: dict[str, Any]) -> None:
    person_id = str(args["person_id"])
    kept = [r for r in w.rows["contact"] if r["person_id"] != person_id]
    if len(kept) == len(w.rows["contact"]):
        raise LookupError(f"No db entry matching person_id={person_id!r} found")
    w.rows["contact"] = kept


def _impl_send_message_with_phone_number(w: WorldState, args: dict[str, Any]) -> str:
    phone_number = str(args["phone_number"])
    if not phone_number.startswith("+"):
        raise ValueError(f"phone_number {phone_number!r} must be in E.164 format")
    w._require("cellular", "Cellular service is not enabled", ConnectionError)
    message_id = w._next_row_id("messaging")
    selves = _impl_search_contacts(w, {"is_self": True})
    if len(selves) != 1:
        raise ValueError(
            "1 and only 1 self entry should exist in contacts database, instead found "
            f"{len(selves)}"
        )
    recipients = _impl_search_contacts(w, {"phone_number": phone_number})
    w.rows["messaging"].append(
        {
            "message_id": message_id,
            "sender_person_id": selves[0]["person_id"],
            "sender_phone_number": selves[0]["phone_number"],
            "recipient_person_id": recipients[0]["person_id"] if recipients else None,
            "recipient_phone_number": phone_number,
            "content": str(args["content"]),
            "creation_timestamp": float(w.epoch),
        }
    )
    return message_id


_IMPLS: dict[str, Callable[[WorldState, dict[str, Any]], Any]] = {
    "add_reminder": _impl_add_reminder,
    "calculate_lat_lon_distance": _impl_calculate_lat_lon_distance,
    "datetime_info_to_timestamp": _impl_datetime_info_to_timestamp,
    "get_cellular_service_status": _impl_get_setting("cellular"),
    "get_current_location": _impl_get_current_location,
    "get_current_timestamp": _impl_get_current_timestamp,
    "get_location_service_status": _impl_get_setting("location_service"),
    "get_low_battery_mode_status": _impl_get_setting("low_battery_mode"),
    "get_wifi_status": _impl_get_setting("wifi"),
    "modify_contact": _impl_modify_contact,
    "modify_reminder": _impl_modify_reminder,
    "remove_contact": _impl_remove_contact,
    "remove_reminder": _impl_remove_reminder,
    "search_contacts": _impl_search_contacts,
    "search_holiday": _impl_search_holiday,
    "search_lat_lon": _impl_search_lat_lon,
    "search_location_around_lat_lon": _impl_search_location_around_lat_lon,
    "search_messages": _impl_search_messages,
    "search_reminder": _impl_search_reminder,
    "search_weather_around_lat_lon": _impl_search_weather_around_lat_lon,
    "send_message_with_phone_number": _impl_send_message_with_phone_number,
    "set_cellular_service_status": _impl_set_setting("cellular"),
    "set_location_service_status": _impl_set_setting("location_service"),
    "set_low_battery_mode_status": _impl_set_setting("low_battery_mode"),
    "set_wifi_status": _impl_set_setting("wifi"),
    "shift_timestamp": _impl_shift_timestamp,
    "timestamp_diff": _impl_timestamp_diff,
    "timestamp_to_datetime_info": _impl_timestamp_to_datetime_info,
    "unit_conversion": _impl_unit_conversion,
}


# --------------------------------------------------------------------------- #
# registry / impls
# --------------------------------------------------------------------------- #


def _param_spec(raw: Mapping[str, Any]) -> ParamSpec:
    return ParamSpec(
        name=str(raw["name"]),
        kind=ParamKind(raw["kind"]),
        required=bool(raw["required"]),
        default=raw.get("default"),
    )


def registry_for_scenario(sc: Scenario, world: WorldState | None = None) -> EffectRegistry:
    """The scenario's allow-list, with live snapshot resolvers.

    ``SnapshotRead`` tools resolve their snapshot id against ``world`` at
    canonicalisation time, over exactly the namespaces the tool reads.  When
    ``world`` is omitted a fresh initial world for ``sc`` is created and bound;
    use :func:`episode_context` when you need the bound world back.
    """
    cfg = load_config()
    bound = world if world is not None else WorldState.from_scenario(sc, cfg)
    specs: list[ToolSpec] = []
    for decl in sc.tools:
        effect = EffectKind(decl.effect)
        raw = cfg["_tools_by_name"][decl.name]
        resolver = None
        if effect is EffectKind.SNAPSHOT_READ:
            namespaces = tuple(sorted(raw["reads"]))
            resolver = snapshot_resolver(bound, namespaces)
        specs.append(
            ToolSpec(
                name=decl.name,
                effect=effect,
                resource=ResourceClass(decl.resource),
                params=tuple(_param_spec(p) for p in decl.params),
                expected_duration_ms=float(decl.expected_duration_ms),
                concurrency_limit=int(raw["concurrency_limit"]),
                snapshot_resolver=resolver,
                description=decl.description,
            )
        )
    return EffectRegistry(specs)


def snapshot_resolver(
    world: WorldState, namespaces: Sequence[str]
) -> Callable[[Mapping[str, Any]], str]:
    """A resolver over exactly ``namespaces`` of the live ``world``.

    Deterministic, argument-independent and re-read on every canonicalisation:
    the returned id is the content digest of that sub-view *now*, which is what
    makes a future launched against an older world version unclaimable.
    """
    names = tuple(sorted(namespaces))

    def resolve(_args: Mapping[str, Any]) -> str:
        return world.snapshot_id(names)

    return resolve


def impls_for_scenario(sc: Scenario, world: WorldState) -> dict[str, ToolImpl]:
    """One implementation per scenario tool, all bound to ``world``.

    Passing a frozen world (:meth:`WorldState.freeze`) yields the speculative
    implementations: they can read but structurally cannot write.
    """

    def make(name: str) -> ToolImpl:
        def impl(args: Mapping[str, Any], _cancel: threading.Event) -> Any:
            return world.call(name, args)

        return impl

    return {decl.name: make(decl.name) for decl in sc.tools}


def episode_context(
    sc: Scenario, world: WorldState | None = None
) -> tuple[WorldState, EffectRegistry, dict[str, ToolImpl]]:
    """A ``(world, registry, impls)`` triple, all bound together.

    ``world`` defaults to a fresh initial world for ``sc``; pass a
    :meth:`WorldState.clone` to bind a registry and impls to that exact state.
    """
    bound = world if world is not None else WorldState.from_scenario(sc)
    registry = registry_for_scenario(sc, bound)
    return bound, registry, impls_for_scenario(sc, bound)


# --------------------------------------------------------------------------- #
# milestone evaluation
# --------------------------------------------------------------------------- #


def _rouge_l(reference: str, candidate: str) -> float:
    ref, cand = _norm_text(reference).split(), _norm_text(candidate).split()
    if not ref or not cand:
        return 1.0 if ref == cand else 0.0
    matcher = difflib.SequenceMatcher(None, ref, cand)
    lcs = sum(block.size for block in matcher.get_matching_blocks())
    if not lcs:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)


def _tool_trace_entries(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        return [dict(value)]
    if isinstance(value, (list, tuple)):
        return [dict(v) for v in value if isinstance(v, Mapping)]
    return []


def _call_matches(entry: Mapping[str, Any], calls: Sequence[Mapping[str, Any]]) -> bool:
    want_tool = entry.get("tool_name")
    want_args = entry.get("arguments") or {}
    for call in calls:
        if call.get("tool") != want_tool:
            continue
        args = call.get("args") or {}
        ok = True
        for key, value in want_args.items():
            if value is None:
                if args.get(key) is not None:
                    ok = False
                    break
                continue
            got = args.get(key)
            if isinstance(value, float) and isinstance(got, (int, float)):
                if abs(float(got) - value) > 1e-6:
                    ok = False
                    break
            elif got != value:
                ok = False
                break
        if ok:
            return True
    return False


def _row_matches(
    row: Mapping[str, Any], target: Mapping[str, Any], *, numeric_tolerance: float
) -> bool:
    """Does ``row`` satisfy every declared column of ``target``?

    Booleans compare exactly -- ``bool`` is a subclass of ``int`` in Python, so
    folding them into the numeric branch would make ``False`` match ``True``
    under any tolerance >= 1.  Numeric columns are POSIX timestamps and compare
    within ``milestones.numeric_tolerance``; strings compare fuzzily, as
    upstream's own column similarity measures do.
    """
    for column, spec in target.items():
        if "unresolved" in spec:
            if column not in row:
                return False
            continue
        want = spec["value"]
        got = row.get(column)
        if isinstance(want, bool) or isinstance(got, bool):
            if got is not want:
                return False
        elif isinstance(want, str) and isinstance(got, str):
            if _fuzzy_score(want, got) < 90.0:
                return False
        elif isinstance(want, (int, float)) and isinstance(got, (int, float)):
            if abs(float(got) - float(want)) > numeric_tolerance:
                return False
        elif got != want:
            return False
    return True


def _literal_target(target: Mapping[str, Any] | None) -> dict[str, Any]:
    return {k: v for k, v in (target or {}).items()}


def _evaluate_constraint(
    con: Mapping[str, Any],
    world: WorldState,
    calls: Sequence[Mapping[str, Any]],
    assistant: str,
    cfg: Mapping[str, Any],
) -> tuple[bool | None, str]:
    """``(satisfied, kind)``; ``None`` means indeterminate."""
    measure = str(con.get("measure") or "")
    namespace = con.get("namespace")
    target = _literal_target(con.get("target"))
    if "tool_trace_dependant_similarity" in measure:
        return None, "tool_trace_dependant_similarity"
    if namespace == "sandbox":
        if "tool_trace" in target:
            spec = target["tool_trace"]
            if "unresolved" in spec:
                return None, "unresolved_tool_trace"
            entries = _tool_trace_entries(spec["value"])
            if not entries:
                return None, "unresolved_tool_trace"
            return all(_call_matches(e, calls) for e in entries), "tool_trace"
        if "content" in target:
            spec = target["content"]
            if "unresolved" in spec:
                return None, "unresolved_content"
            text = str(spec["value"])
            recipient = (target.get("recipient") or {}).get("value")
            if recipient == "execution_environment":
                # AGENT -> EXECUTION_ENVIRONMENT with a bare tool name: a
                # forbidden-call constraint, used by minefields.
                return any(c.get("tool") == text for c in calls), "forbidden_tool_call"
            threshold = float(cfg["milestones"]["rouge_l_threshold"])
            return _rouge_l(text, assistant) >= threshold, "assistant_text"
        return None, "unsupported_sandbox_constraint"
    if namespace in NAMESPACES:
        rows = world.rows[namespace]
        if not target:
            if measure.startswith("update_similarity"):
                return namespace in world.changed_namespaces(), "namespace_changed"
            return None, "unsupported_untargeted_constraint"
        tolerance = float(cfg["milestones"]["numeric_tolerance"])
        hit = any(_row_matches(row, target, numeric_tolerance=tolerance) for row in rows)
        if measure.startswith("removal_similarity"):
            return not hit, "removal_similarity"
        return hit, measure.split("(")[0] or "presence"
    return None, "unsupported_namespace"


def evaluate_milestones(
    sc: Scenario, world: WorldState, transcript: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Milestone / minefield evaluation: a pure function of its three inputs.

    Reads nothing but ``sc``, ``world`` and ``transcript``; touches no clock, no
    filesystem and no global state, and mutates none of them.
    """
    cfg = load_config()
    calls = [e for e in transcript if e.get("kind") == "call"]
    assistant = " ".join(str(e.get("text", "")) for e in transcript if e.get("kind") == "assistant")

    per_milestone: list[dict[str, Any]] = []
    hit = 0
    indeterminate = 0
    for index, milestone in enumerate(sc.milestones):
        results = [
            _evaluate_constraint(con, world, calls, assistant, cfg)
            for con in milestone.get("constraints", ())
        ]
        kinds = [k for _, k in results]
        if not results or any(ok is None for ok, _ in results):
            status = "indeterminate"
            indeterminate += 1
        elif all(ok for ok, _ in results):
            status = "hit"
            hit += 1
        else:
            status = "miss"
        per_milestone.append({"index": index, "status": status, "constraint_kinds": kinds})

    mine_hits: list[int] = []
    for index, minefield in enumerate(sc.minefields):
        results = [
            _evaluate_constraint(con, world, calls, assistant, cfg)
            for con in minefield.get("constraints", ())
        ]
        if results and all(ok is True for ok, _ in results):
            mine_hits.append(index)

    total = len(sc.milestones)
    return {
        "total": total,
        "hit": hit,
        "indeterminate": indeterminate,
        "minefields_total": len(sc.minefields),
        "minefields_hit": len(mine_hits),
        "minefields_hit_index": mine_hits,
        "success": bool(hit == total and not mine_hits),
        "order_enforced": bool(cfg["milestones"]["order_enforced"]),
        "per_milestone": per_milestone,
        "digest": _canon_digest(
            {
                "hit": hit,
                "total": total,
                "indeterminate": indeterminate,
                "minefields": mine_hits,
                "per_milestone": per_milestone,
            }
        ),
    }


# --------------------------------------------------------------------------- #
# the GPU-free reference program
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class _Operation:
    """One call the stub program will make, derived from a milestone."""

    tool: str
    args: dict[str, Any]
    id_param: str | None = None
    id_source: str | None = None


def _milestone_operations(sc: Scenario) -> list[_Operation]:
    """Every call the scenario's own milestones declare, in milestone order.

    A ``tool_trace`` target names the tool and arguments directly.  A
    state-namespace target names a *mutation*, which ``stub.state_tools`` (and
    ``stub.setting_column_tools`` for the boolean settings) maps back onto the
    tool that performs it.  Nothing outside those declarations is used.
    """
    cfg = load_config()
    state_tools = cfg["stub"]["state_tools"]
    setting_tools = cfg["stub"]["setting_column_tools"]
    surface = set(sc.tool_names)
    out: list[_Operation] = []
    for milestone in sc.milestones:
        for con in milestone.get("constraints", ()):
            target = _literal_target(con.get("target"))
            measure = str(con.get("measure") or "").split("(")[0]
            namespace = con.get("namespace")
            trace_spec = target.get("tool_trace")
            if trace_spec and "unresolved" not in trace_spec:
                for entry in _tool_trace_entries(trace_spec["value"]):
                    name = str(entry.get("tool_name"))
                    if name in surface:
                        args = {
                            k: v
                            for k, v in (entry.get("arguments") or {}).items()
                            if v is not None
                        }
                        out.append(_Operation(name, args))
                continue
            if namespace == "setting":
                for column, spec in sorted(target.items()):
                    tool = setting_tools.get(column)
                    if tool in surface and "unresolved" not in spec:
                        out.append(_Operation(tool, {"on": bool(spec["value"])}))
                continue
            mapping = (state_tools.get(namespace) or {}).get(measure)
            if not mapping or mapping["tool"] not in surface:
                continue
            args: dict[str, Any] = {}
            for column, param in (mapping["columns"] or {}).items():
                spec = target.get(column)
                if spec and "unresolved" not in spec and spec["value"] is not None:
                    args[param] = spec["value"]
            id_param = mapping.get("id_column")
            id_source = mapping.get("id_source")
            if id_param:
                spec = target.get(id_param)
                if spec and "unresolved" not in spec:
                    args[id_param] = spec["value"]
                    id_param, id_source = None, None
                elif id_source not in surface:
                    continue
            out.append(_Operation(mapping["tool"], args, id_param, id_source))
    return out


def _fill_args(
    registry: EffectRegistry,
    tool: str,
    args: Mapping[str, Any],
    cfg: Mapping[str, Any],
    *,
    skip: str | None = None,
) -> dict[str, Any] | None:
    """Complete ``args`` for ``tool`` from the declared stub fillers.

    ``None`` means the call cannot be completed from declared data and is
    dropped rather than guessed.  ``skip`` names a parameter supplied at run
    time inside the program (a row id resolved by a search).
    """
    spec = registry.get(tool)
    if spec is None:
        return None
    filled = dict(args)
    fillers = cfg["stub"]["required_args"].get(tool, {})
    for param in spec.params:
        if param.name in filled or param.name == skip:
            continue
        if param.name in fillers:
            filled[param.name] = fillers[param.name]
        elif param.required:
            return None
    if not filled and skip is None and tool.startswith("search_"):
        # Upstream searches refuse an empty criteria set.
        filled.update(fillers)
        if not filled:
            return None
    return filled


def _guarded(name: str, body: str) -> str:
    indented = "\n".join("    " + line for line in body.splitlines())
    return (
        f"try:\n{indented}\n"
        f"except Exception as _e:\n    _rec({name!r}, 'error:' + type(_e).__name__)"
    )


def stub_programs(sc: Scenario, registry: EffectRegistry) -> list[str]:
    """One deterministic ``eval`` program per user turn, for ``--stub``.

    Every call is derived from the scenario's own milestone declarations plus the
    declared ``stub`` fillers; the program brackets them with the scenario's
    argument-free eligible reads so that each turn genuinely interleaves writes,
    live reads and eligible snapshot reads.  Row ids the milestone leaves open
    are resolved at run time inside the OMP kernel from the scenario's own search
    tools, which is why this is a program and not a list of calls.  It ends by
    printing a digest over every observed tool result, so the stub's final
    assistant text is a function of the tool results and cannot be identical
    across schedulings unless the results are.
    """
    cfg = load_config()
    effects = scenario_effects(sc)
    free_reads = [
        name
        for name, effect in effects.items()
        if effect.speculation_eligible
        and not any(p.required for p in (registry.get(name).params if registry.get(name) else ()))
        and not name.startswith("search_")
    ]
    operations = _milestone_operations(sc)
    turns = max(1, len(sc.user_turns))
    marker = str(cfg["stub"]["digest_marker"])
    programs: list[str] = []
    for turn in range(turns):
        lines = [
            "import hashlib, json",
            "_obs = []",
            "def _rec(name, value):",
            "    _obs.append([name, value])",
            "    return value",
        ]
        reads = [_guarded(n, f"_rec({n!r}, tool.{n}())") for n in free_reads]
        lines += reads
        for op in operations[turn::turns]:
            filled = _fill_args(registry, op.tool, op.args, cfg, skip=op.id_param)
            if filled is None:
                continue
            if op.id_param and op.id_source:
                source_args = _fill_args(registry, op.id_source, {}, cfg)
                if source_args is None:
                    continue
                src = ", ".join(f"{k}={v!r}" for k, v in sorted(source_args.items()))
                rest = "".join(f", {k}={v!r}" for k, v in sorted(filled.items()))
                lines.append(
                    _guarded(
                        op.tool,
                        f"_hits = tool.{op.id_source}({src})\n"
                        f"if _hits:\n"
                        f"    _rec({op.tool!r}, tool.{op.tool}("
                        f"{op.id_param}=_hits[0][{op.id_param!r}]{rest}))",
                    )
                )
                continue
            rendered = ", ".join(f"{k}={v!r}" for k, v in sorted(filled.items()))
            lines.append(_guarded(op.tool, f"_rec({op.tool!r}, tool.{op.tool}({rendered}))"))
        lines += reads
        lines.append(
            "_blob = json.dumps(_obs, sort_keys=True, default=str).encode()\n"
            f"print({marker!r} + '=' + hashlib.sha256(_blob).hexdigest())"
        )
        programs.append("\n".join(lines) + "\n")
    return programs


# --------------------------------------------------------------------------- #
# the GPU-free stub model
# --------------------------------------------------------------------------- #


def _sse(obj: Mapping[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


class StubModelServer:
    """An in-process OpenAI-compatible server that replays fixed programs.

    The technique of ``scripts/smoke_omp_ptc.py``: OMP talks to this instead of
    a model, so the authoritative path (native tool call -> OMP Python kernel ->
    ``tool.<name>(...)`` loopback -> ``host_tool_call`` -> HostToolRouter) runs
    with no GPU and no checkpoint, and every episode is deterministic.

    Per user turn it answers once with an ``eval`` tool call carrying that
    turn's program, then once with a final text that *echoes the digest the
    program printed over every tool result it observed*.  The reply is therefore
    a function of the tool results, so a corrupted speculative value would
    change the assistant text and the losslessness comparison would see it.

    OMP resolves the ``vllm`` provider base URL from its installed profile, so
    ``port`` must be the port that profile names (8000 for OMP-Lite).
    """

    def __init__(self, cfg: Mapping[str, Any] | None = None, port: int = 8000) -> None:
        cfg = cfg or load_config()
        self.model_id = str(cfg["stub"]["model_id"])
        self.marker = str(cfg["stub"]["digest_marker"])
        self.port = int(port)
        self.requests = 0
        self.programs: list[str] = []
        self._lock = threading.Lock()
        self._server: Any = None

    # -- lifecycle --------------------------------------------------------- #

    @staticmethod
    def port_free(port: int) -> bool:
        import socket

        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", int(port)))
            except OSError:
                return False
        return True

    def start(self) -> str:
        from http.server import ThreadingHTTPServer

        if self._server is not None:
            raise RuntimeError("stub server is already running")
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self.base_url

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None

    def __enter__(self) -> StubModelServer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def load(self, programs: Sequence[str]) -> None:
        """Install the programs for the episode about to be driven."""
        with self._lock:
            self.programs = list(programs)

    def load_for(self, sc: Scenario, registry: EffectRegistry) -> list[str]:
        programs = stub_programs(sc, registry)
        self.load(programs)
        return programs

    # -- wire -------------------------------------------------------------- #

    def _final_text(self, messages: Sequence[Mapping[str, Any]], pattern: re.Pattern[str]) -> str:
        for message in reversed(messages):
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(
                    str(b.get("text", "")) for b in content if isinstance(b, Mapping)
                )
            match = pattern.search(str(content or ""))
            if match:
                return f"observed {match.group(1)}"
        return "observed none"

    def _handler(self) -> type:
        from http.server import BaseHTTPRequestHandler

        stub = self
        digest_re = re.compile(re.escape(self.marker) + r"=([0-9a-f]{64})")

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args: Any) -> None:
                return

            def _json(self, body: bytes) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._json(
                        json.dumps(
                            {
                                "object": "list",
                                "data": [
                                    {
                                        "id": stub.model_id,
                                        "object": "model",
                                        "owned_by": "crystalcall-toolsandbox-stub",
                                        "max_model_len": 32768,
                                    }
                                ],
                            }
                        ).encode()
                    )
                    return
                self.send_error(404)

            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length") or 0)
                try:
                    req = json.loads(self.rfile.read(length))
                except json.JSONDecodeError:
                    self.send_error(400)
                    return
                with stub._lock:
                    stub.requests += 1
                    programs = list(stub.programs)
                messages = req.get("messages") or []
                turn = max(0, sum(1 for m in messages if m.get("role") == "user") - 1)
                last_role = messages[-1].get("role") if messages else None
                base = {
                    "id": f"chatcmpl-tsbx-{turn}",
                    "object": "chat.completion.chunk",
                    "created": 0,
                    "model": stub.model_id,
                }
                chunks: list[dict[str, Any]] = []
                if last_role == "tool" or not programs:
                    text = stub._final_text(messages, digest_re)
                    for i in range(0, len(text), 24):
                        chunks.append(
                            {
                                **base,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text[i : i + 24]},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                        )
                    chunks.append(
                        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                    )
                else:
                    program = programs[min(turn, len(programs) - 1)]
                    chunks.append(
                        {
                            **base,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "role": "assistant",
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": f"call_tsbx_{turn}",
                                                "type": "function",
                                                "function": {
                                                    "name": "eval",
                                                    "arguments": json.dumps(
                                                        {"language": "py", "code": program}
                                                    ),
                                                },
                                            }
                                        ],
                                    },
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
                    chunks.append(
                        {
                            **base,
                            "choices": [
                                {"index": 0, "delta": {}, "finish_reason": "tool_calls"}
                            ],
                        }
                    )
                chunks.append(
                    {
                        **base,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": 64,
                            "completion_tokens": 16,
                            "total_tokens": 80,
                        },
                    }
                )
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for chunk in chunks:
                    payload = _sse(chunk)
                    self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                done = b"data: [DONE]\n\n"
                self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        return Handler

# --------------------------------------------------------------------------- #
# one episode
# --------------------------------------------------------------------------- #


def _call_row(call: Any) -> dict[str, Any]:
    """The comparable projection of one authoritative call."""
    canonical = call.call
    return {
        "kind": "call",
        "tool": canonical.tool if canonical else call.tool_name,
        "args": canonical.args_dict if canonical else dict(call.raw_arguments),
        "snapshot_id": canonical.snapshot_id if canonical else None,
        "effect": canonical.effect.value if canonical else None,
        "identity": canonical.identity if canonical else None,
        "result_digest": call.result_digest,
        "error": call.error,
        "canonicalization_error": call.canonicalization_error,
        "claimed_speculative": bool(call.claimed_speculative),
    }


def _comparable(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The trace projection that must be identical across schedulings.

    ``claimed_speculative`` is excluded by design: it is the one field the two
    schedulings are *expected* to differ on.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("kind") != "call":
            continue
        out.append(
            {
                "tool": row["tool"],
                "args": row["args"],
                "snapshot_id": row["snapshot_id"],
                "effect": row["effect"],
                "result_digest": row["result_digest"],
                "error": row["error"],
                "canonicalization_error": row["canonicalization_error"],
            }
        )
    return out


def _launch_candidates(
    sc: Scenario,
    registry: EffectRegistry,
    previous: Sequence[Mapping[str, Any]],
) -> list[CanonicalCall]:
    """The declared speculation policy, as canonical calls.

    (a) every eligible tool whose canonical form needs no supplied argument;
    (b) every eligible canonical call observed in the previous turn.
    Re-canonicalised now, so each carries the *current* snapshot id.
    """
    out: list[CanonicalCall] = []
    seen: set[str] = set()
    for spec in registry.specs():
        if not spec.speculation_eligible or any(p.required for p in spec.params):
            continue
        try:
            call = canonicalize(registry, spec.name, {})
        except CanonicalizationError:
            continue
        if call.identity not in seen:
            seen.add(call.identity)
            out.append(call)
    for row in previous:
        if row.get("kind") != "call" or row.get("error") or row.get("canonicalization_error"):
            continue
        spec = registry.get(str(row["tool"]))
        if spec is None or not spec.speculation_eligible:
            continue
        try:
            call = canonicalize(registry, spec.name, dict(row["args"]))
        except CanonicalizationError:
            continue
        if call.identity not in seen:
            seen.add(call.identity)
            out.append(call)
    return out


def run_episode(
    sc: Scenario,
    *,
    seed: int,
    arm: str,
    client_factory: Callable[[], Any],
    futures: FutureTable | None = None,
    speculate: bool = False,
    turn_timeout: float = 900.0,
    trace: Callable[[str, Mapping[str, Any]], Any] | None = None,
) -> dict[str, Any]:
    """Drive one scenario end to end through OMP-Lite, then replay it.

    ``client_factory`` returns something with ``.rebind(router)`` and
    ``.prompt(message, timeout=...)`` -- in practice a started
    :class:`~crystalcall.omp_bridge.OmpRpcClient`, one fresh OMP session per
    episode.  Each user turn is fed in order and OMP runs its own tool calls
    between turns.

    With ``speculate=True`` the declared turn-boundary policy launches eligible
    speculations against a frozen copy of the world before each turn; writes and
    live reads are never launched.  The ``request_id`` component of the claim key
    carries the turn index, so a future launched in one turn is unclaimable in
    any other turn -- the first of the four key components doing exactly the job
    the invariant asks of it.

    **One generation, then a replay.**  The model is sampled exactly once.  The
    losslessness check does *not* generate a second time: DiffusionGemma's
    sampler renoises the canvas with Gumbel noise and ignores the request
    temperature, so two sequential requests are not
    identical even at a pinned server seed, and comparing two generations would
    confound sampling noise with scheduling.  Instead the recorded authoritative
    canonical call sequence is re-executed, in order, with ``futures=None``,
    against :meth:`WorldState.clone` of the episode's *initial* world.  The model
    is held fixed by construction, so any mismatch can only come from
    scheduling, which is exactly the invariant under test.  The returned dict
    carries the generated observation, the ordinary-scheduling replay and the
    comparison verdict.
    """
    cfg = load_config()
    mode = "speculative" if speculate else "ordinary"
    initial = WorldState.from_scenario(sc)
    world, registry, impls = episode_context(sc, initial.clone())
    episode_id = f"{arm}/{sc.scenario_id}/s{seed}/{mode}"
    turn_index = 0
    request_ids: list[str] = []

    def request_id_for_turn() -> str:
        return f"{episode_id}/t{turn_index}"

    own_futures = False
    if speculate and futures is None:
        limit = int(cfg["speculation"]["concurrency_per_resource"])
        futures = FutureTable(concurrency={rc: limit for rc in ResourceClass})
        own_futures = True
    router = HostToolRouter(
        registry,
        impls,
        futures=futures if speculate else None,
        request_id_provider=request_id_for_turn,
        claim_timeout=float(cfg["speculation"]["claim_timeout_s"]),
    )

    started_ns = time.monotonic_ns()
    transcript: list[dict[str, Any]] = []
    turns: list[dict[str, Any]] = []
    launches = 0
    error: str | None = None
    max_launch = int(cfg["speculation"]["max_launches_per_turn"])

    if trace is not None:
        trace(
            "toolsandbox_episode_start",
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_id": sc.scenario_id,
                "family": sc.family,
                "seed": seed,
                "arm": arm,
                "mode": mode,
                "request_id": episode_id,
                "registry_fingerprint": registry.fingerprint(),
                "effects": {k: v.value for k, v in scenario_effects(sc).items()},
                "user_turns": list(sc.user_turns),
                "world_digest_initial": initial.digest(),
                "immutable_namespaces": sorted(world.immutable_namespaces),
            },
        )

    try:
        client = client_factory()
        client.rebind(router)
        previous: list[dict[str, Any]] = []
        for index, message in enumerate(sc.user_turns):
            turn_index = index
            request_id = request_id_for_turn()
            request_ids.append(request_id)
            if speculate and futures is not None:
                frozen = world.freeze()
                frozen_impls = impls_for_scenario(sc, frozen)
                for call in _launch_candidates(sc, registry, previous)[:max_launch]:
                    assert_no_write_speculation(call)
                    spec = registry.resolve(call.tool)
                    impl = frozen_impls[call.tool]
                    futures.launch(
                        FutureKey.of(request_id, call),
                        call,
                        spec.resource,
                        lambda cancel, _i=impl, _a=call.args_dict: _i(_a, cancel),
                        reason=f"turn{index}:{cfg['speculation']['policy']}",
                    )
                    launches += 1
            transcript.append({"kind": "user", "text": message})
            router.reset()
            turn = client.prompt(message, timeout=turn_timeout)
            rows = [_call_row(c) for c in router.calls()]
            transcript.extend(rows)
            transcript.append({"kind": "assistant", "text": turn.assistant_text})
            digest = world.digest()
            turns.append(
                {
                    "turn_index": index,
                    "user": message,
                    "assistant": turn.assistant_text,
                    "calls": rows,
                    "eval_programs": list(turn.eval_programs),
                    "world_digest_after": digest,
                    "world_versions_after": dict(world.versions),
                    "wall_ms": turn.wall_ms,
                    "error": turn.error,
                }
            )
            if trace is not None:
                trace(
                    "toolsandbox_turn",
                    {
                        "schema_version": SCHEMA_VERSION,
                        "scenario_id": sc.scenario_id,
                        "seed": seed,
                        "arm": arm,
                        "mode": mode,
                        "turn_index": index,
                        "calls": rows,
                        "assistant": turn.assistant_text,
                        "world_digest_after": digest,
                        "wall_ms": turn.wall_ms,
                        "error": turn.error,
                    },
                )
            if turn.error:
                error = turn.error
                break
            previous = rows
    except Exception as exc:  # noqa: BLE001 - an episode failure is data
        error = f"{type(exc).__name__}: {exc}"

    stats = {"launched": 0, "claimed": 0, "wasted": 0}
    if speculate and futures is not None:
        launched = claimed = wasted = 0
        for rid in request_ids:
            raw = futures.stats(rid)
            futures.discard_unmatched(rid)
            launched += int(raw["launched"])
            claimed += int(raw["claimed"])
            wasted += int(futures.stats(rid)["wasted"])
        stats = {"launched": launched, "claimed": claimed, "wasted": wasted}
        if own_futures:
            futures.close()

    milestones = evaluate_milestones(sc, world, transcript)
    calls = [r for r in transcript if r.get("kind") == "call"]
    observed = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": sc.scenario_id,
        "family": sc.family,
        "seed": seed,
        "arm": arm,
        "mode": mode,
        "request_id": episode_id,
        "speculate": bool(speculate),
        "turns": turns,
        "transcript": transcript,
        "tool_trace": _comparable(transcript),
        "assistant_final": turns[-1]["assistant"] if turns else "",
        "world_state_digest_initial": initial.digest(),
        "world_state_digest_final": world.digest(),
        "world_versions_final": dict(world.versions),
        "milestones": milestones,
        "effects": {k: v.value for k, v in scenario_effects(sc).items()},
        "eligible_tools": sorted(registry.eligible_names()),
        "speculative_launches": stats["launched"] if speculate else 0,
        "speculative_claimed": stats["claimed"] if speculate else 0,
        "speculative_wasted": stats["wasted"] if speculate else 0,
        "authoritative_calls": len(calls),
        "claimed_calls": sum(1 for r in calls if r["claimed_speculative"]),
        "wall_ms": (time.monotonic_ns() - started_ns) / 1e6,
        "error": error,
    }

    replay = replay_episode(sc, observed, initial=initial)
    divergence = compare_replay(observed, replay)
    episode = {
        **observed,
        "replay": replay,
        "divergence": divergence,
        "semantic_divergence": int(divergence["divergent"]),
    }
    if trace is not None:
        trace(
            "toolsandbox_replay",
            {k: v for k, v in replay.items() if k != "transcript"},
        )
        trace("toolsandbox_divergence", divergence)
        trace(
            "toolsandbox_episode_end",
            {k: v for k, v in episode.items() if k not in ("transcript", "replay")},
        )
    return episode


def replay_episode(
    sc: Scenario,
    observed: Mapping[str, Any],
    *,
    initial: WorldState | None = None,
    corrupt: Callable[[int, dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Re-execute an observed canonical call sequence under ordinary scheduling.

    The model is not consulted: the recorded sequence *is* the generation.  The
    replay starts from :meth:`WorldState.clone` of the episode's initial world,
    dispatches every recorded call in execution order through the same tool
    implementations with no future table at all, and re-derives each result
    digest with the same function ``HostToolRouter`` uses.

    ``corrupt`` is a test hook: it may rewrite the ``(index, row)`` pair before
    dispatch, which is how the comparison is proven to have teeth.
    """
    base = initial if initial is not None else WorldState.from_scenario(sc)
    world, registry, impls = episode_context(sc, base.clone())
    rows: list[dict[str, Any]] = []
    transcript: list[dict[str, Any]] = []
    for entry in observed.get("transcript", ()):
        if entry.get("kind") != "call":
            transcript.append(dict(entry))
            continue
        row = dict(entry)
        index = len(rows)
        if corrupt is not None:
            row = corrupt(index, row)
        if row.get("canonicalization_error"):
            # The generation's call never reached a tool; nothing to re-execute.
            replayed = {
                "kind": "call",
                "tool": row["tool"],
                "args": row["args"],
                "snapshot_id": None,
                "effect": None,
                "result_digest": "",
                "error": None,
                "canonicalization_error": row["canonicalization_error"],
                "claimed_speculative": False,
            }
            rows.append(replayed)
            transcript.append(replayed)
            continue
        digest = ""
        error: str | None = None
        snapshot_id: str | None = None
        effect: str | None = None
        try:
            call = canonicalize(registry, str(row["tool"]), dict(row["args"]))
            snapshot_id, effect = call.snapshot_id, call.effect.value
            value = impls[call.tool](call.args_dict, threading.Event())
            digest = _digest_result(value)
        except CanonicalizationError as exc:
            error = f"invalid call: {exc.reason} ({exc.detail})"
        except Exception as exc:  # noqa: BLE001 - mirrors HostToolRouter.handle
            error = f"{type(exc).__name__}: {exc}"
        replayed = {
            "kind": "call",
            "tool": row["tool"],
            "args": row["args"],
            "snapshot_id": snapshot_id,
            "effect": effect,
            "result_digest": digest,
            "error": error,
            "canonicalization_error": None,
            "claimed_speculative": False,
        }
        rows.append(replayed)
        transcript.append(replayed)
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": sc.scenario_id,
        "seed": observed.get("seed"),
        "arm": observed.get("arm"),
        "mode": "ordinary_replay",
        "calls": len(rows),
        "transcript": transcript,
        "tool_trace": _comparable(transcript),
        "world_state_digest_initial": base.digest(),
        "world_state_digest_final": world.digest(),
        "world_versions_final": dict(world.versions),
        "milestones": evaluate_milestones(sc, world, transcript),
    }


def _replay_projection(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The per-call fields the replay must reproduce exactly.

    ``claimed_speculative`` is excluded by construction (the replay claims
    nothing) and so is ``snapshot_id``: a ``SnapshotRead``'s id is a digest of
    the world at the moment of the call, and reproducing it is implied by
    reproducing the world.  What must match is the *value*: tool, canonical
    arguments, result digest, error.
    """
    return [
        {
            "tool": r["tool"],
            "args": r["args"],
            "result_digest": r["result_digest"],
            "error": r["error"],
            "canonicalization_error": r["canonicalization_error"],
        }
        for r in rows
    ]


def compare_replay(observed: Mapping[str, Any], replay: Mapping[str, Any]) -> dict[str, Any]:
    """Compare one generation against its ordinary-scheduling replay.

    A difference in the per-call result digests (in execution order), the final
    :meth:`WorldState.digest`, or the milestone evaluation is a semantic
    divergence.  Because both sides consume the *same* generation, a divergence
    can only be caused by scheduling.
    """
    fields: list[str] = []
    detail: dict[str, Any] = {}

    def check(name: str, a: Any, b: Any) -> None:
        if canonical_json(a) != canonical_json(b):
            fields.append(name)
            detail[name] = {"observed": a, "replay": b}

    check(
        "tool_trace",
        _replay_projection(observed.get("tool_trace", ())),
        _replay_projection(replay.get("tool_trace", ())),
    )
    check(
        "world_state_digest_final",
        observed.get("world_state_digest_final"),
        replay.get("world_state_digest_final"),
    )
    check(
        "world_state_digest_initial",
        observed.get("world_state_digest_initial"),
        replay.get("world_state_digest_initial"),
    )
    check(
        "milestones",
        {k: v for k, v in (observed.get("milestones") or {}).items() if k != "per_milestone"},
        {k: v for k, v in (replay.get("milestones") or {}).items() if k != "per_milestone"},
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": observed.get("scenario_id"),
        "seed": observed.get("seed"),
        "arm": observed.get("arm"),
        "mode": observed.get("mode"),
        "method": "one_generation_replayed_against_a_cloned_world",
        "divergent": bool(fields),
        "fields": fields,
        "detail": detail,
    }
