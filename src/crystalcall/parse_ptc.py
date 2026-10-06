"""Python PTC extraction from a (possibly provisional) canvas.

Everything here is *parsing*, never execution.  A provisional canvas is treated
as untrusted text: it is located, decoded, handed to :mod:`ast`, and only
top-level, unconditional, statically-closed calls to allow-listed tools are
emitted as candidates.

Accepted MVP shapes (per the specification):

    result = tool_name(literal=...)
    tool_name(literal=...)
    [tool_name(x=v) for v in [literal, ...]]

Everything else is rejected with a machine-readable reason so that rejection
classes survive into the result data.
"""

from __future__ import annotations

import ast
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .canonicalize import CanonicalCall, CanonicalizationError, canonicalize
from .effects import EffectRegistry

__all__ = [
    "Candidate",
    "Rejection",
    "ParseResult",
    "PtcExtractor",
    "locate_code_regions",
]

_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\r?\n", re.IGNORECASE)
_JSON_CODE_KEY_RE = re.compile(r'"code"\s*:\s*"')

# DiffusionGemma's native tool-call wire format, as it actually appears in a
# predicted-clean canvas under `--tool-call-parser gemma4`:
#
#   <|tool_call>call:eval{language:<|"|>py<|"|>, code:<|"|><program><|"|>}<tool_call|>
#
# The arguments are NOT JSON: keys are bare, and string values are delimited by
# the single special token `<|"|>` with no escaping inside. A JSON-only parser
# would match no code region on any revision, so the extractor would produce
# zero candidates for every canvas and every crystallization measurement would
# read zero.
GEMMA_STRING_DELIM = '<|"|>'
_GEMMA_CODE_KEY_RE = re.compile(r"\bcode\s*:\s*" + re.escape(GEMMA_STRING_DELIM))
_MAX_PREFIX_TRIM_STEPS = 64


@dataclass(frozen=True, slots=True)
class Region:
    """A candidate code region inside a canvas, with an index map back to
    canvas character coordinates."""

    text: str
    index_map: tuple[int, ...]
    locator: str

    def canvas_span(self, start: int, end: int) -> tuple[int, int]:
        if not self.index_map:
            return (0, 0)
        lo = max(0, min(start, len(self.index_map) - 1))
        hi = max(0, min(end - 1, len(self.index_map) - 1))
        a, b = self.index_map[lo], self.index_map[hi]
        return (min(a, b), max(a, b) + 1)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One statically-closed, allow-listed, canonicalised provisional call."""

    call: CanonicalCall
    canvas_span: tuple[int, int]
    source_text: str
    locator: str
    form: str
    order: int
    comprehension_index: int | None = None

    @property
    def identity(self) -> str:
        return self.call.identity

    def to_json(self) -> dict[str, Any]:
        return {
            "call": self.call.to_json(),
            "canvas_span": list(self.canvas_span),
            "source_text": self.source_text,
            "locator": self.locator,
            "form": self.form,
            "order": self.order,
            "comprehension_index": self.comprehension_index,
        }


@dataclass(frozen=True, slots=True)
class Rejection:
    reason: str
    detail: str = ""
    canvas_span: tuple[int, int] | None = None


@dataclass(slots=True)
class ParseResult:
    """Outcome of extracting candidates from one canvas."""

    parse_mode: str  # "full" | "prefix" | "failed" | "no_region"
    candidates: tuple[Candidate, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    regions: int = 0
    trimmed_lines: int = 0

    @property
    def ok(self) -> bool:
        return self.parse_mode in ("full", "prefix")

    @property
    def rejection_counts(self) -> dict[str, int]:
        return dict(Counter(r.reason for r in self.rejections))

    def identities(self) -> tuple[str, ...]:
        return tuple(c.identity for c in self.candidates)

    def to_json(self) -> dict[str, Any]:
        return {
            "parse_mode": self.parse_mode,
            "regions": self.regions,
            "trimmed_lines": self.trimmed_lines,
            "candidates": [c.to_json() for c in self.candidates],
            "rejections": [
                {"reason": r.reason, "detail": r.detail, "canvas_span": r.canvas_span}
                for r in self.rejections
            ],
        }


# --------------------------------------------------------------------------- #
# region location
# --------------------------------------------------------------------------- #


def _identity_map(text: str, offset: int) -> tuple[int, ...]:
    return tuple(range(offset, offset + len(text)))


def _unescape_json_string(text: str, start: int) -> tuple[str, tuple[int, ...], int] | None:
    """Decode a JSON string body beginning at ``text[start]`` (just after the
    opening quote).  Returns the decoded text, a per-character map back into
    ``text`` coordinates, and the index just past the closing quote.

    Returns ``None`` when the string never terminates (a truncated provisional
    canvas); callers then fall back to the partial decode.
    """
    out: list[str] = []
    imap: list[int] = []
    i = start
    n = len(text)
    closed = False
    while i < n:
        ch = text[i]
        if ch == '"':
            closed = True
            i += 1
            break
        if ch == "\\":
            if i + 1 >= n:
                break
            esc = text[i + 1]
            simple = {
                '"': '"',
                "\\": "\\",
                "/": "/",
                "b": "\b",
                "f": "\f",
                "n": "\n",
                "r": "\r",
                "t": "\t",
            }
            if esc in simple:
                out.append(simple[esc])
                imap.append(i)
                i += 2
                continue
            if esc == "u" and i + 5 < n:
                try:
                    out.append(chr(int(text[i + 2 : i + 6], 16)))
                except ValueError:
                    return None
                imap.append(i)
                i += 6
                continue
            return None
        out.append(ch)
        imap.append(i)
        i += 1
    if not out:
        return None
    if not closed:
        # Truncated: still usable as a provisional region.
        return ("".join(out), tuple(imap), n)
    return ("".join(out), tuple(imap), i)


def locate_code_regions(canvas_text: str) -> list[Region]:
    """Locate every plausible Python PTC region in a canvas.

    Four locators are used and their results are unioned: DiffusionGemma's
    native ``code:<|"|>...<|"|>`` tool-call payload, a JSON ``"code": "..."``
    payload, a fenced block, and the whole canvas as a last resort.  Duplicate
    candidates collapse later because identity is canonical, so a redundant
    locator costs nothing but a spurious one costs measured precision.

    Every region carries an ``index_map`` back to canvas character offsets, so
    a candidate's span can be checked against the revision's accepted mask.
    """
    regions: list[Region] = []
    seen: set[tuple[str, int]] = set()

    # 0. DiffusionGemma's native tool-call payload: ``code:<|"|><python><|"|>``.
    #    A provisional canvas is routinely unterminated, so a missing closing
    #    delimiter yields the tail rather than nothing -- that is exactly the
    #    half-formed hypothesis this project exists to observe.
    for m in _GEMMA_CODE_KEY_RE.finditer(canvas_text):
        body_start = m.end()
        close = canvas_text.find(GEMMA_STRING_DELIM, body_start)
        body_end = close if close != -1 else len(canvas_text)
        text = canvas_text[body_start:body_end]
        if not text.strip():
            continue
        key = (text, body_start)
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            Region(
                text=text,
                index_map=_identity_map(text, body_start),
                locator="gemma_tool_call",
            )
        )

    # 1. Structured tool-call payloads: ``"code": "<escaped python>"``.
    for m in _JSON_CODE_KEY_RE.finditer(canvas_text):
        decoded = _unescape_json_string(canvas_text, m.end())
        if decoded is None:
            continue
        text, imap, _ = decoded
        key = (text, imap[0] if imap else -1)
        if key in seen:
            continue
        seen.add(key)
        regions.append(Region(text=text, index_map=imap, locator="tool_call_json"))

    # 2. Fenced code blocks.
    for m in _FENCE_RE.finditer(canvas_text):
        body_start = m.end()
        close = canvas_text.find("```", body_start)
        body_end = close if close != -1 else len(canvas_text)
        text = canvas_text[body_start:body_end]
        if not text.strip():
            continue
        key = (text, body_start)
        if key in seen:
            continue
        seen.add(key)
        regions.append(
            Region(
                text=text,
                index_map=_identity_map(text, body_start),
                locator="fenced_block",
            )
        )

    # 3. The whole canvas, as a last resort.
    if canvas_text.strip():
        key = (canvas_text, 0)
        if key not in seen:
            seen.add(key)
            regions.append(
                Region(
                    text=canvas_text,
                    index_map=_identity_map(canvas_text, 0),
                    locator="whole_canvas",
                )
            )
    return regions


# --------------------------------------------------------------------------- #
# static literal resolution
# --------------------------------------------------------------------------- #


class _StaticResolver:
    """Resolves AST expressions to Python literals, or fails.

    A name is resolvable only when it was bound earlier in the *same*
    provisional program by a top-level assignment whose value is itself
    statically resolvable and does not depend on any call.  This is what the
    specification calls "statically closed".
    """

    def __init__(self) -> None:
        self.env: dict[str, Any] = {}
        self.tainted: set[str] = set()

    def bind(self, name: str, value: Any, *, tainted: bool) -> None:
        if tainted:
            self.tainted.add(name)
            self.env.pop(name, None)
        else:
            self.tainted.discard(name)
            self.env[name] = value

    def resolve(self, node: ast.AST, depth: int = 0) -> Any:
        if depth > 8:
            raise _Unresolved("nesting_too_deep")
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, int, float, bool, type(None))):
                return node.value
            if isinstance(node.value, bytes):
                raise _Unresolved("bytes_literal")
            raise _Unresolved("unsupported_constant")
        if isinstance(node, ast.List):
            return [self.resolve(e, depth + 1) for e in node.elts]
        if isinstance(node, ast.Tuple):
            return [self.resolve(e, depth + 1) for e in node.elts]
        if isinstance(node, ast.Set):
            raise _Unresolved("set_literal")
        if isinstance(node, ast.Dict):
            out: dict[str, Any] = {}
            for k, v in zip(node.keys, node.values, strict=True):
                if k is None:
                    raise _Unresolved("dict_unpacking")
                key = self.resolve(k, depth + 1)
                if not isinstance(key, str):
                    raise _Unresolved("non_string_dict_key")
                if key in out:
                    raise _Unresolved("duplicate_dict_key")
                out[key] = self.resolve(v, depth + 1)
            return out
        if isinstance(node, ast.Name):
            if node.id in self.tainted:
                raise _Unresolved("depends_on_provisional_call")
            if node.id in self.env:
                return self.env[node.id]
            raise _Unresolved("unbound_name")
        if isinstance(node, ast.JoinedStr):
            parts: list[str] = []
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    parts.append(v.value)
                else:
                    raise _Unresolved("dynamic_fstring")
            return "".join(parts)
        if isinstance(node, ast.Attribute):
            raise _Unresolved("dynamic_attribute")
        if isinstance(node, ast.Call):
            raise _Unresolved("nested_call")
        if isinstance(node, (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare)):
            raise _Unresolved("computed_expression")
        if isinstance(node, ast.Subscript):
            raise _Unresolved("subscript")
        if isinstance(node, (ast.Lambda, ast.IfExp)):
            raise _Unresolved("dynamic_expression")
        raise _Unresolved(type(node).__name__.lower())


class _Unresolved(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# --------------------------------------------------------------------------- #
# the extractor
# --------------------------------------------------------------------------- #


def _call_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: list[str] = [func.attr]
        cur: ast.expr = func.value
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
            return ".".join(reversed(parts))
    return None


@dataclass(slots=True)
class PtcExtractor:
    """Extracts canonical candidates from provisional or final canvases."""

    registry: EffectRegistry
    allow_prefix_parse: bool = True
    max_candidates: int = 64
    _line_index_cache: dict[str, tuple[int, ...]] = field(default_factory=dict, repr=False)

    # ---- public API ---------------------------------------------------- #

    def extract(self, canvas_text: str) -> ParseResult:
        regions = locate_code_regions(canvas_text)
        if not regions:
            return ParseResult(parse_mode="no_region")

        candidates: list[Candidate] = []
        rejections: list[Rejection] = []
        seen: set[str] = set()
        best_mode = "failed"
        trimmed_total = 0

        for region in regions:
            tree, mode, trimmed = self._parse(region.text)
            trimmed_total += trimmed
            if tree is None:
                rejections.append(Rejection("parse_error", region.locator))
                continue
            if mode == "full" or best_mode == "failed":
                best_mode = "full" if mode == "full" else "prefix"
            if best_mode == "prefix" and mode == "full":
                best_mode = "full"
            cs, rs = self._walk(tree, region)
            for c in cs:
                if c.identity in seen:
                    continue
                seen.add(c.identity)
                candidates.append(c)
                if len(candidates) >= self.max_candidates:
                    break
            rejections.extend(rs)

        return ParseResult(
            parse_mode=best_mode,
            candidates=tuple(candidates),
            rejections=tuple(rejections),
            regions=len(regions),
            trimmed_lines=trimmed_total,
        )

    def parse_final_program(self, code: str) -> ParseResult:
        """Parse an authoritative program (already-extracted code) strictly."""
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return ParseResult(
                parse_mode="failed",
                rejections=(Rejection("parse_error", f"line {exc.lineno}: {exc.msg}"),),
                regions=1,
            )
        region = Region(text=code, index_map=_identity_map(code, 0), locator="authoritative")
        cs, rs = self._walk(tree, region)
        seen: set[str] = set()
        uniq: list[Candidate] = []
        for c in cs:
            if c.identity in seen:
                continue
            seen.add(c.identity)
            uniq.append(c)
        return ParseResult(
            parse_mode="full", candidates=tuple(uniq), rejections=tuple(rs), regions=1
        )

    # ---- internals ------------------------------------------------------ #

    def _parse(self, text: str) -> tuple[ast.Module | None, str, int]:
        try:
            return ast.parse(text), "full", 0
        except SyntaxError:
            pass
        except (ValueError, MemoryError, RecursionError):
            return None, "failed", 0
        if not self.allow_prefix_parse:
            return None, "failed", 0
        lines = text.splitlines()
        steps = min(len(lines) - 1, _MAX_PREFIX_TRIM_STEPS)
        for drop in range(1, steps + 1):
            prefix = "\n".join(lines[: len(lines) - drop])
            if not prefix.strip():
                break
            try:
                return ast.parse(prefix), "prefix", drop
            except SyntaxError:
                continue
            except (ValueError, MemoryError, RecursionError):
                return None, "failed", drop
        return None, "failed", 0

    def _span(self, region: Region, node: ast.AST) -> tuple[int, int]:
        line_starts = self._line_starts(region)
        try:
            lineno = getattr(node, "lineno", 1)
            col = getattr(node, "col_offset", 0)
            end_lineno = getattr(node, "end_lineno", lineno) or lineno
            end_col = getattr(node, "end_col_offset", col) or col
            start = line_starts[lineno - 1] + col
            end = line_starts[end_lineno - 1] + end_col
        except (IndexError, TypeError):
            return (0, 0)
        return region.canvas_span(start, end)

    def _line_starts(self, region: Region) -> tuple[int, ...]:
        # Keyed on the region *text*, never on ``id(region)``: regions are
        # transient, so an identity key silently returns another region's line
        # table once an address is reused, which shifts every candidate's
        # canvas span (and with it the confidence and acceptance features the
        # gate reads) in a way that varies run to run.
        key = region.text
        cached = self._line_index_cache.get(key)
        if cached is not None:
            return cached
        starts = [0]
        for i, ch in enumerate(region.text):
            if ch == "\n":
                starts.append(i + 1)
        out = tuple(starts)
        if len(self._line_index_cache) > 256:
            self._line_index_cache.clear()
        self._line_index_cache[key] = out
        return out

    def _source(self, region: Region, node: ast.AST) -> str:
        try:
            seg = ast.get_source_segment(region.text, node)  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            seg = None
        return seg or ""

    def _walk(
        self, tree: ast.Module, region: Region
    ) -> tuple[list[Candidate], list[Rejection]]:
        resolver = _StaticResolver()
        candidates: list[Candidate] = []
        rejections: list[Rejection] = []
        order = 0

        for stmt in tree.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                got, rej = self._from_call(stmt.value, region, resolver, order, "expr")
                order += 1
                candidates.extend(got)
                rejections.extend(rej)
                continue

            if isinstance(stmt, ast.Assign):
                if len(stmt.targets) != 1 or not isinstance(stmt.targets[0], ast.Name):
                    rejections.append(
                        Rejection("unsupported_assignment_target", canvas_span=self._span(region, stmt))
                    )
                    continue
                target = stmt.targets[0].id
                value = stmt.value
                if isinstance(value, ast.Call):
                    got, rej = self._from_call(value, region, resolver, order, "assign")
                    order += 1
                    candidates.extend(got)
                    rejections.extend(rej)
                    resolver.bind(target, None, tainted=True)
                    continue
                if isinstance(value, ast.ListComp):
                    got, rej = self._from_listcomp(value, region, resolver, order)
                    order += 1
                    candidates.extend(got)
                    rejections.extend(rej)
                    resolver.bind(target, None, tainted=True)
                    continue
                try:
                    resolver.bind(target, resolver.resolve(value), tainted=False)
                except _Unresolved:
                    resolver.bind(target, None, tainted=True)
                continue

            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.value is None:
                    continue
                if isinstance(stmt.value, ast.Call):
                    got, rej = self._from_call(stmt.value, region, resolver, order, "assign")
                    order += 1
                    candidates.extend(got)
                    rejections.extend(rej)
                resolver.bind(stmt.target.id, None, tainted=True)
                continue

            if isinstance(
                stmt,
                (
                    ast.If,
                    ast.For,
                    ast.While,
                    ast.Try,
                    ast.With,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                    ast.AsyncFor,
                    ast.AsyncWith,
                    ast.Match,
                ),
            ):
                rejections.append(
                    Rejection(
                        "non_toplevel_control_flow",
                        type(stmt).__name__,
                        self._span(region, stmt),
                    )
                )
                continue
            # import/pass/print-expr/etc.: ignored, not an error.

        return candidates, rejections

    def _from_call(
        self,
        node: ast.Call,
        region: Region,
        resolver: _StaticResolver,
        order: int,
        form: str,
    ) -> tuple[list[Candidate], list[Rejection]]:
        span = self._span(region, node)
        name = _call_name(node.func)
        if name is None:
            return [], [Rejection("dynamic_callee", canvas_span=span)]
        if self.registry.get(name) is None:
            return [], [Rejection("unknown_tool", name, span)]

        positional: list[Any] = []
        kwargs: dict[str, Any] = {}
        dup: list[str] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                return [], [Rejection("starred_argument", name, span)]
            try:
                positional.append(resolver.resolve(arg))
            except _Unresolved as exc:
                return [], [Rejection(exc.reason, name, span)]
        for kw in node.keywords:
            if kw.arg is None:
                return [], [Rejection("kwargs_unpacking", name, span)]
            if kw.arg in kwargs:
                dup.append(kw.arg)
            try:
                kwargs[kw.arg] = resolver.resolve(kw.value)
            except _Unresolved as exc:
                return [], [Rejection(exc.reason, name, span)]

        try:
            call = canonicalize(
                self.registry, name, kwargs, positional, duplicate_keys=tuple(dup)
            )
        except CanonicalizationError as exc:
            return [], [Rejection(exc.reason, f"{name}:{exc.detail}", span)]

        return (
            [
                Candidate(
                    call=call,
                    canvas_span=span,
                    source_text=self._source(region, node),
                    locator=region.locator,
                    form=form,
                    order=order,
                )
            ],
            [],
        )

    def _from_listcomp(
        self,
        node: ast.ListComp,
        region: Region,
        resolver: _StaticResolver,
        order: int,
    ) -> tuple[list[Candidate], list[Rejection]]:
        span = self._span(region, node)
        if len(node.generators) != 1:
            return [], [Rejection("multi_generator_comprehension", canvas_span=span)]
        gen = node.generators[0]
        if gen.ifs:
            return [], [Rejection("filtered_comprehension", canvas_span=span)]
        if gen.is_async:
            return [], [Rejection("async_comprehension", canvas_span=span)]
        if not isinstance(gen.target, ast.Name):
            return [], [Rejection("unsupported_comprehension_target", canvas_span=span)]
        if not isinstance(node.elt, ast.Call):
            return [], [Rejection("comprehension_body_not_call", canvas_span=span)]
        try:
            iterable = resolver.resolve(gen.iter)
        except _Unresolved as exc:
            return [], [Rejection(f"comprehension_{exc.reason}", canvas_span=span)]
        if not isinstance(iterable, list) or not iterable:
            return [], [Rejection("comprehension_iter_not_literal_list", canvas_span=span)]
        if len(iterable) > 32:
            return [], [Rejection("comprehension_too_wide", str(len(iterable)), span)]

        out: list[Candidate] = []
        rejections: list[Rejection] = []
        var = gen.target.id
        saved_env = dict(resolver.env)
        saved_taint = set(resolver.tainted)
        for i, value in enumerate(iterable):
            resolver.env = dict(saved_env)
            resolver.tainted = set(saved_taint)
            resolver.bind(var, value, tainted=False)
            got, rej = self._from_call(node.elt, region, resolver, order, "listcomp")
            for c in got:
                out.append(
                    Candidate(
                        call=c.call,
                        canvas_span=c.canvas_span,
                        source_text=c.source_text,
                        locator=c.locator,
                        form="listcomp",
                        order=order,
                        comprehension_index=i,
                    )
                )
            rejections.extend(rej)
        resolver.env = saved_env
        resolver.tainted = saved_taint
        return out, rejections


def extract_tool_call_code(payload: Mapping[str, Any] | str) -> str | None:
    """Pull the Python source out of an authoritative ``eval`` tool-call payload."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, Mapping):
        return None
    code = payload.get("code")
    return code if isinstance(code, str) else None
