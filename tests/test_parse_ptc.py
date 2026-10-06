"""PTC extraction: only top-level, unconditional, statically-closed, allow-listed calls."""

from __future__ import annotations

import json

import pytest

from crystalcall.parse_ptc import PtcExtractor, locate_code_regions


@pytest.fixture
def ex(registry) -> PtcExtractor:
    return PtcExtractor(registry=registry)


def ids(result) -> set[str]:
    return {c.call.identity for c in result.candidates}


def sources(result) -> set[str]:
    return {c.call.source() for c in result.candidates}


def reasons(result) -> set[str]:
    return set(result.rejection_counts)


# --------------------------------------------------------------------------- #
# accepted shapes
# --------------------------------------------------------------------------- #


def test_bare_call_expression(ex):
    r = ex.extract("```python\nlookup_symbol(name=\"A\")\n```")
    assert r.ok
    assert sources(r) == {"lookup_symbol(name='A')"}


def test_assigned_call(ex):
    r = ex.extract("```python\nres = lookup_symbol(name=\"A\")\n```")
    assert sources(r) == {"lookup_symbol(name='A')"}


def test_namespaced_bridge_call(ex):
    r = ex.extract("```python\nres = tool.lookup_symbol(name=\"A\")\n```")
    assert sources(r) == {"lookup_symbol(name='A')"}


def test_list_comprehension_over_literal_list(ex):
    r = ex.extract('```python\nvs = [lookup_symbol(name=x) for x in ["A", "B", "C"]]\n```')
    assert sources(r) == {
        "lookup_symbol(name='A')",
        "lookup_symbol(name='B')",
        "lookup_symbol(name='C')",
    }
    assert all(c.form == "listcomp" for c in r.candidates)


def test_literal_name_propagation(ex):
    r = ex.extract('```python\nq = "retry timeout"\nres = search_repo(query=q, path="src")\n```')
    assert sources(r) == {"search_repo(max_results=50, path='src', query='retry timeout')"}


def test_multiple_parallel_calls(ex):
    src = (
        "```python\n"
        'a = search_repo(query="x", path="src")\n'
        'b = search_repo(query="y", path="src")\n'
        "```"
    )
    r = ex.extract(src)
    assert len(r.candidates) == 2


def test_tool_call_json_payload_region(ex):
    payload = json.dumps({"language": "py", "code": 'lookup_symbol(name="A")\n'})
    canvas = f'<|tool_call>call:eval{payload}<tool_call|>'
    r = ex.extract(canvas)
    assert sources(r) == {"lookup_symbol(name='A')"}
    assert any(c.locator == "tool_call_json" for c in r.candidates)


def test_unterminated_fence_still_parses(ex):
    r = ex.extract('```python\nlookup_symbol(name="A")\n')
    assert sources(r) == {"lookup_symbol(name='A')"}


def test_prefix_parse_recovers_leading_statements(ex):
    canvas = '```python\nlookup_symbol(name="A")\nsearch_repo(query=\n```'
    r = ex.extract(canvas)
    assert "lookup_symbol(name='A')" in sources(r)


# --------------------------------------------------------------------------- #
# rejected shapes
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("code", "reason"),
    [
        ('if True:\n    lookup_symbol(name="A")\n', "non_toplevel_control_flow"),
        ('for x in y:\n    lookup_symbol(name=x)\n', "non_toplevel_control_flow"),
        ('try:\n    lookup_symbol(name="A")\nexcept Exception:\n    pass\n', "non_toplevel_control_flow"),
        ('def f():\n    lookup_symbol(name="A")\n', "non_toplevel_control_flow"),
        ('with open("f") as fh:\n    lookup_symbol(name="A")\n', "non_toplevel_control_flow"),
        ('lookup_symbol(name=undefined_name)\n', "unbound_name"),
        ('lookup_symbol(name=other())\n', "nested_call"),
        ('lookup_symbol(name=obj.attr)\n', "dynamic_attribute"),
        ('lookup_symbol(name=f"{x}")\n', "dynamic_fstring"),
        ('lookup_symbol(**kw)\n', "kwargs_unpacking"),
        ('lookup_symbol(*args)\n', "starred_argument"),
        ('lookup_symbol(name=data[0])\n', "subscript"),
        ('lookup_symbol(name="a" + "b")\n', "computed_expression"),
        ('getattr(tool, "lookup_symbol")(name="A")\n', "dynamic_callee"),
        ('unknown_tool(name="A")\n', "unknown_tool"),
        ('lookup_symbol(name="A", bogus=1)\n', "unknown_parameter"),
        ('lookup_symbol()\n', "missing_required"),
    ],
)
def test_rejected(ex, code, reason):
    r = ex.extract(f"```python\n{code}```")
    assert not r.candidates, f"unexpectedly accepted: {sources(r)}"
    assert reason in reasons(r), f"expected {reason}, got {reasons(r)}"


def test_call_depending_on_previous_call_result_is_rejected(ex):
    code = 'first = lookup_symbol(name="A")\nsecond = lookup_symbol(name=first)\n'
    r = ex.extract(f"```python\n{code}```")
    assert sources(r) == {"lookup_symbol(name='A')"}
    assert "depends_on_provisional_call" in reasons(r)


def test_comprehension_over_tool_output_is_rejected(ex):
    code = 'rows = search_repo(query="q")\nvs = [lookup_symbol(name=x) for x in rows]\n'
    r = ex.extract(f"```python\n{code}```")
    assert all(c.call.tool == "search_repo" for c in r.candidates)
    assert any(k.startswith("comprehension_") for k in reasons(r))


def test_filtered_comprehension_is_rejected(ex):
    code = 'vs = [lookup_symbol(name=x) for x in ["A"] if x]\n'
    r = ex.extract(f"```python\n{code}```")
    assert not r.candidates
    assert "filtered_comprehension" in reasons(r)


def test_write_and_live_read_are_extracted_but_not_eligible(ex, registry):
    code = 'write_note(path="a.txt", text="x")\nread_live_clock()\n'
    r = ex.extract(f"```python\n{code}```")
    tools = {c.call.tool for c in r.candidates}
    assert tools == {"write_note", "read_live_clock"}
    assert not any(registry.is_speculation_eligible(t) for t in tools)


def test_garbage_canvas_produces_no_candidates(ex):
    r = ex.extract("\u00ff\u00fe zz ((( ]] ??? \n ###")
    assert not r.candidates


def test_regions_are_located_in_priority_order():
    canvas = 'prefix {"code": "a()"} ```python\nb()\n```'
    regions = locate_code_regions(canvas)
    assert [r.locator for r in regions] == ["tool_call_json", "fenced_block", "whole_canvas"]


def test_canvas_span_maps_into_canvas_coordinates(ex):
    canvas = 'lead-in text\n```python\nlookup_symbol(name="A")\n```'
    r = ex.extract(canvas)
    c = next(c for c in r.candidates if c.locator == "fenced_block")
    lo, hi = c.canvas_span
    assert canvas[lo:hi] == 'lookup_symbol(name="A")'
