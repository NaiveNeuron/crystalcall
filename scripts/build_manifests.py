#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["pyyaml>=6.0", "numpy>=2.1"]
# ///
"""Build the CrystalCall prompt manifests from the pinned upstream benchmarks.

Everything is derived: the four BFCL single-turn categories become
``experiments/manifests/{dev,test}.jsonl``, the ToolSandbox scenario registry
becomes ``experiments/manifests/toolsandbox.jsonl``, and ``docs/manifests.md``
is regenerated from the same numbers.  All rules live in
``configs/manifest_rules.yml``; nothing here is conditioned on a model outcome.

The build is byte-deterministic: artefacts are rendered to bytes first, so
``--verify`` is a byte comparison against what is on disk, not a re-derivation
of a summary.

    uv run scripts/build_manifests.py                 # write artefacts
    uv run scripts/build_manifests.py --verify        # compare, exit 1 on drift
    uv run scripts/build_manifests.py --fetch         # (re)fetch pinned vendor data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from crystalcall.canonicalize import canonicalize  # noqa: E402
from crystalcall.manifest import (  # noqa: E402
    MANIFEST_SCHEMA,
    ManifestBuildError,
    ManifestRules,
    PromptRecord,
    build_bfcl_candidates,
    extract_toolsandbox_scenarios,
    load_rules,
    registry_for,
    select_splits,
    select_toolsandbox,
    sanitize_tool_name,
    sha256_bytes,
    toolsandbox_family,
)

DEFAULT_OUT_DIR = Path("experiments/manifests")
DEFAULT_VENDOR_DIR = Path("experiments/vendor")
DEFAULT_RULES = Path("configs/manifest_rules.yml")
DEFAULT_DOCS = Path("docs/manifests.md")
VENDOR_SOURCES = Path("experiments/manifests/vendor_sources.json")
EFFECT_ORDER = ("Pure", "LiveRead", "Write", "SnapshotRead")


# --------------------------------------------------------------------------- #
# rendering helpers (bytes in, bytes out: determinism by construction)
# --------------------------------------------------------------------------- #


def _dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def render_jsonl(header: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> bytes:
    head = dict(header)
    head["record"] = "header"
    lines = [_dumps(head)] + [_dumps(r) for r in rows]
    return "".join(line + "\n" for line in lines).encode("utf-8")


def render_sidecar(name: str, digest: str) -> bytes:
    return f"{digest}  {name}\n".encode("utf-8")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    out = ["| " + " | ".join(str(h) for h in headers) + " |"]
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# vendor fetch (pinned commit tarballs)
# --------------------------------------------------------------------------- #


#: Sources this build actually reads.  ``vendor_sources.json`` is shared with
#: other tasks, so the manifest header pins a digest of *these* entries rather
#: than the whole file: an unrelated source added by another task must not
#: invalidate the manifests, and any change to what we read must.
CONSUMED_SOURCES = ("bfcl", "toolsandbox")
PINNED_SOURCE_FIELDS = (
    "key",
    "repo",
    "commit",
    "archive_url",
    "archive_sha256",
    "archive_member",
    "install_path",
)


def provenance_digest(sources: Mapping[str, Any]) -> str:
    """sha256 over the pinned provenance of the sources this build reads."""
    consumed = [
        {field: s.get(field) for field in PINNED_SOURCE_FIELDS}
        for s in sorted(sources["sources"], key=lambda s: s["key"])
        if s["key"] in CONSUMED_SOURCES
    ]
    if len(consumed) != len(CONSUMED_SOURCES):
        raise SystemExit(
            f"vendor_sources.json must declare {CONSUMED_SOURCES}; found "
            f"{[s['key'] for s in consumed]}"
        )
    payload = {
        "sources": consumed,
        "input_files_sha256": dict(sorted(sources["input_files_sha256"].items())),
    }
    return sha256_bytes(_dumps(payload).encode("utf-8"))



def _archive_member(root: Path, source: Mapping[str, Any]) -> Path:
    """The path inside an extracted archive that gets installed.

    ``archive_member`` is optional: without it the archive's single top-level
    directory (GitHub's ``<repo>-<sha>/``) is installed whole.
    """
    declared = source.get("archive_member")
    if declared:
        return root / declared
    entries = sorted(p for p in root.iterdir())
    if len(entries) != 1:
        raise SystemExit(
            f"{source['key']}: archive has {len(entries)} top-level entries; "
            "declare archive_member in vendor_sources.json"
        )
    return entries[0]


def fetch_vendor(vendor_dir: Path, sources_path: Path) -> None:
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    for source in sources["sources"]:
        install = vendor_dir / source["install_path"]
        if not source["fetch_command"].startswith("curl "):
            # The harness binary is a per-architecture release asset, not a
            # source archive; it has its own fetch script.
            print(f"skip {source['key']}: run `{source['fetch_command']}`")
            continue
        print(f"fetch {source['key']}: {source['fetch_command']}")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "archive.tar.gz"
            with urllib.request.urlopen(source["archive_url"]) as response:  # noqa: S310
                archive.write_bytes(response.read())
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            if digest != source["archive_sha256"]:
                raise SystemExit(
                    f"{source['key']}: archive sha256 {digest} != pinned {source['archive_sha256']}"
                )
            extracted = Path(tmp) / "x"
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(extracted, filter="data")
            member = _archive_member(extracted, source)
            if not member.exists():
                raise SystemExit(
                    f"{source['key']}: archive member missing: {source.get('archive_member')}"
                )
            install.parent.mkdir(parents=True, exist_ok=True)
            if member.is_dir():
                if install.exists():
                    shutil.rmtree(install)
                shutil.copytree(member, install)
            else:
                if install.is_dir():
                    shutil.rmtree(install)
                shutil.copy2(member, install)
        print(f"  -> {install}")
    verify_vendor(vendor_dir, sources_path)


def verify_vendor(vendor_dir: Path, sources_path: Path) -> dict[str, str]:
    """Hash every declared input file; fail loudly on drift or absence."""
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    observed: dict[str, str] = {}
    problems: list[str] = []
    for rel, expected in sorted(sources["input_files_sha256"].items()):
        path = vendor_dir / rel
        if not path.exists():
            problems.append(f"missing vendor input: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        observed[rel] = digest
        if digest != expected:
            problems.append(f"{rel}: sha256 {digest} != pinned {expected}")
    if problems:
        commands = "\n".join(
            f"  {s['fetch_command']}   # {s['git_clone_command']}" for s in sources["sources"]
        )
        raise SystemExit(
            "vendor data is missing or has drifted:\n  "
            + "\n  ".join(problems)
            + "\n\nre-fetch with `--fetch`, or manually:\n"
            + commands
        )
    return observed


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #


def effect_distribution(records: Sequence[PromptRecord]) -> dict[str, dict[str, int]]:
    """Declared-tool and expected-call effect counts, per category."""
    tools: dict[str, Counter[str]] = {}
    calls: dict[str, Counter[str]] = {}
    for record in records:
        tools.setdefault(record.category, Counter()).update(t.effect for t in record.tools)
        calls.setdefault(record.category, Counter()).update(record.meta["call_effects"])
    out: dict[str, dict[str, int]] = {}
    for category in sorted(tools):
        out[category] = {
            f"tools_{effect}": tools[category].get(effect, 0) for effect in EFFECT_ORDER
        }
        out[category].update(
            {f"calls_{effect}": calls[category].get(effect, 0) for effect in EFFECT_ORDER}
        )
        out[category]["tools_total"] = sum(tools[category].values())
        out[category]["calls_total"] = sum(calls[category].values())
    return out


def check_canonicalizable(records: Sequence[PromptRecord]) -> list[str]:
    """Re-canonicalise every expected call of every record; return failures."""
    failures: list[str] = []
    for record in records:
        registry = registry_for(record)
        for call in record.expected_calls:
            try:
                canonicalize(registry, call["tool"], call["args"])
            except Exception as exc:  # noqa: BLE001 - any failure is a build error
                failures.append(f"{record.prompt_id}:{call['tool']}:{exc}")
    return failures


# --------------------------------------------------------------------------- #
# the build
# --------------------------------------------------------------------------- #


def build(
    *,
    vendor_dir: Path,
    out_dir: Path,
    rules_path: Path,
    docs_path: Path,
    seed: int,
) -> tuple[dict[Path, bytes], dict[str, Any]]:
    rules: ManifestRules = load_rules(rules_path)
    sources = json.loads((REPO_ROOT / VENDOR_SOURCES).read_text(encoding="utf-8"))
    vendor_observed = verify_vendor(vendor_dir, REPO_ROOT / VENDOR_SOURCES)
    commits = {s["key"]: s["commit"] for s in sources["sources"] if s["key"] in CONSUMED_SOURCES}
    vendor_provenance_sha256 = provenance_digest(sources)

    out_rel = _rel(out_dir)
    vendor_rel = _rel(vendor_dir)
    build_command = (
        f"uv run scripts/build_manifests.py --out-dir {out_rel} "
        f"--vendor-dir {vendor_rel} --seed {seed}"
    )

    # -- BFCL ------------------------------------------------------------- #
    candidates, drops = build_bfcl_candidates(vendor_dir, rules)
    upstream_totals = _upstream_totals(vendor_dir, rules)
    splits = select_splits(candidates, rules, seed)

    dev_ids = [r.prompt_id for r in splits["dev"]]
    test_ids = [r.prompt_id for r in splits["test"]]
    overlap = sorted(set(dev_ids) & set(test_ids))
    if overlap:
        raise SystemExit(f"dev/test overlap: {overlap[:5]}")
    for split_name, records in splits.items():
        failures = check_canonicalizable(records)
        if failures:
            raise SystemExit(f"{split_name}: non-canonicalisable calls: {failures[:5]}")

    # -- ToolSandbox ------------------------------------------------------ #
    scenarios = extract_toolsandbox_scenarios(vendor_dir, rules)
    selected_scenarios = select_toolsandbox(scenarios, rules, seed)

    # -- artefacts -------------------------------------------------------- #
    artefacts: dict[Path, bytes] = {}
    hashes: dict[str, str] = {}
    split_counts: dict[str, dict[str, int]] = {}
    for split_name in ("dev", "test"):
        records = splits[split_name]
        counts = Counter(r.category for r in records)
        split_counts[split_name] = {c: counts[c] for c in sorted(counts)}
        header = {
            "schema": MANIFEST_SCHEMA,
            "split": split_name,
            "source": rules.bfcl["source"],
            "bfcl_repo": next(s["repo"] for s in sources["sources"] if s["key"] == "bfcl"),
            "bfcl_commit": commits["bfcl"],
            "bfcl_files": {c: rules.categories[c] for c in sorted(rules.categories)},
            "rules_path": str(_rel(rules_path)),
            "rules_sha256": rules.sha256,
            "seed": seed,
            "counts": split_counts[split_name],
            "total": len(records),
            "build_command": build_command,
            "vendor_provenance_sha256": vendor_provenance_sha256,
            "generator": "scripts/build_manifests.py",
        }
        blob = render_jsonl(header, [r.to_json() for r in records])
        path = out_dir / f"{split_name}.jsonl"
        artefacts[path] = blob
        hashes[f"{split_name}.jsonl"] = sha256_bytes(blob)
        artefacts[path.with_name(path.name + ".sha256")] = render_sidecar(
            path.name, hashes[f"{split_name}.jsonl"]
        )

    family_counts = Counter(r.family for r in selected_scenarios)
    ts_header = {
        "schema": MANIFEST_SCHEMA,
        "split": "toolsandbox",
        "source": rules.toolsandbox["source"],
        "toolsandbox_repo": next(
            s["repo"] for s in sources["sources"] if s["key"] == "toolsandbox"
        ),
        "toolsandbox_commit": commits["toolsandbox"],
        "family_rule": {
            "precedence": [
                f"{row['category']} -> {row['family']}"
                for row in rules.toolsandbox["family_precedence"]
            ],
            "method": rules.toolsandbox["method"],
            "variants": [v["tag"] for v in rules.toolsandbox["variants"]],
        },
        "rules_path": str(_rel(rules_path)),
        "rules_sha256": rules.sha256,
        "seed": seed,
        "counts": {f: family_counts[f] for f in sorted(family_counts)},
        "total": len(selected_scenarios),
        "build_command": build_command,
        "vendor_provenance_sha256": vendor_provenance_sha256,
        "generator": "scripts/build_manifests.py",
    }
    ts_blob = render_jsonl(ts_header, [s.to_json() for s in selected_scenarios])
    ts_path = out_dir / "toolsandbox.jsonl"
    artefacts[ts_path] = ts_blob
    hashes["toolsandbox.jsonl"] = sha256_bytes(ts_blob)
    artefacts[ts_path.with_name(ts_path.name + ".sha256")] = render_sidecar(
        ts_path.name, hashes["toolsandbox.jsonl"]
    )
    hashes[str(_rel(rules_path))] = rules.sha256

    # -- stats ------------------------------------------------------------ #
    selected = splits["dev"] + splits["test"]
    drop_counts: dict[str, dict[str, int]] = {}
    for drop in drops:
        drop_counts.setdefault(drop.category, {}).setdefault(drop.reason, 0)
        drop_counts[drop.category][drop.reason] += 1
    stats = {
        "schema": "crystalcall-manifest-build-stats/1",
        "build_command": build_command,
        "seed": seed,
        "rules_path": str(_rel(rules_path)),
        "rules_sha256": rules.sha256,
        "commits": commits,
        "vendor_provenance_sha256": vendor_provenance_sha256,
        "vendor_input_files_sha256": vendor_observed,
        "bfcl": {
            "upstream_prompts": upstream_totals,
            "candidates_per_category": {
                c: sum(1 for r in candidates if r.category == c)
                for c in sorted(rules.categories)
            },
            "dropped_per_category": {
                c: sum(1 for d in drops if d.category == c) for c in sorted(rules.categories)
            },
            "drop_reasons": {c: dict(sorted(v.items())) for c, v in sorted(drop_counts.items())},
            "drops": [d.to_json() for d in drops],
            "effect_distribution_candidates": effect_distribution(candidates),
            "effect_distribution_selected": effect_distribution(selected),
            "split_counts": split_counts,
            "omitted_optional_params": sum(
                len(r.meta["omitted_optional_params"]) for r in selected
            ),
            "dropped_defaults": sum(len(r.meta["dropped_defaults"]) for r in selected),
            "expected_calls_selected": sum(len(r.expected_calls) for r in selected),
            "tool_declarations_selected": sum(len(r.tools) for r in selected),
            "tool_names_sanitised_selected": sum(
                len(r.meta["sanitised_tool_names"]) for r in selected
            ),
            "distinct_upstream_names_sanitised_selected": len(
                {
                    t.original_name
                    for r in selected
                    for t in r.tools
                    if t.alias_candidate is not None
                }
            ),
            "aliases_skipped_selected": sum(len(r.meta["skipped_aliases"]) for r in selected),
            "prompts_with_sanitised_names_selected": sum(
                1 for r in selected if r.meta["sanitised_tool_names"]
            ),
            "collision_suffixed_names_selected": sum(
                1
                for r in selected
                for t in r.tools
                if t.name != sanitize_tool_name(t.upstream_name)
            ),
        },
        "toolsandbox": {
            "base_scenarios_declared": len(scenarios),
            "base_scenarios_per_module": {
                m: sum(1 for s in scenarios if s["module"] == m)
                for m in sorted({s["module"] for s in scenarios})
            },
            "declared_category_counts": {
                cat: sum(1 for s in scenarios if cat in s["categories"])
                for cat in sorted({c for s in scenarios for c in s["categories"]})
            },
            "base_scenarios_per_family": {
                f: sum(1 for s in scenarios if toolsandbox_family(s["categories"], rules) == f)
                for f in sorted({row["family"] for row in rules.toolsandbox["family_precedence"]})
            },
            "unassigned_base_scenarios": sorted(
                s["base_scenario"]
                for s in scenarios
                if toolsandbox_family(s["categories"], rules) is None
            ),
            "selected_per_family": {f: family_counts[f] for f in sorted(family_counts)},
            "selected_per_variant": {
                v: sum(1 for s in selected_scenarios if s.variant == v)
                for v in sorted({s.variant for s in selected_scenarios})
            },
            "distinct_base_scenarios_selected": len(
                {s.base_scenario for s in selected_scenarios}
            ),
        },
        "hashes": dict(sorted(hashes.items())),
    }
    stats_blob = (_dumps_pretty(stats) + "\n").encode("utf-8")
    artefacts[out_dir / "build_stats.json"] = stats_blob

    docs_blob = render_docs(stats, rules, sources).encode("utf-8")
    artefacts[docs_path] = docs_blob
    return artefacts, stats


def _dumps_pretty(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=2, ensure_ascii=False)


def _rel(path: Path) -> Path:
    try:
        return path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return path


def _upstream_totals(vendor_dir: Path, rules: ManifestRules) -> dict[str, int]:
    root = vendor_dir / rules.bfcl["data_dir"]
    out: dict[str, int] = {}
    for category in sorted(rules.categories):
        path = root / rules.categories[category]
        out[category] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return out


# --------------------------------------------------------------------------- #
# docs
# --------------------------------------------------------------------------- #


def render_docs(
    stats: Mapping[str, Any], rules: ManifestRules, sources: Mapping[str, Any]
) -> str:
    bfcl = stats["bfcl"]
    ts = stats["toolsandbox"]
    cats = sorted(bfcl["candidates_per_category"])
    src = {s["key"]: s for s in sources["sources"]}

    fetch_lines = []
    for key in ("bfcl", "toolsandbox"):
        s = src[key]
        fetch_lines.append(
            f"# {key}: {s['repo']} @ {s['commit']}\n"
            f"{s['git_clone_command']}          # canonical form\n"
            f"{s['fetch_command']}\n"
            f"# archive sha256 {s['archive_sha256']}\n"
            f"# keep: {s['archive_member']} -> experiments/vendor/{s['install_path']}"
        )

    counts_rows = [
        [
            c,
            rules.categories[c],
            bfcl["upstream_prompts"][c],
            bfcl["candidates_per_category"][c],
            bfcl["dropped_per_category"][c],
            bfcl["split_counts"]["dev"].get(c, 0),
            bfcl["split_counts"]["test"].get(c, 0),
        ]
        for c in cats
    ]
    counts_rows.append(
        [
            "**total**",
            "",
            sum(bfcl["upstream_prompts"].values()),
            sum(bfcl["candidates_per_category"].values()),
            sum(bfcl["dropped_per_category"].values()),
            sum(bfcl["split_counts"]["dev"].values()),
            sum(bfcl["split_counts"]["test"].values()),
        ]
    )

    def effect_rows(dist: Mapping[str, Any], prefix: str) -> list[list[Any]]:
        rows: list[list[Any]] = []
        totals = Counter()
        for c in sorted(dist):
            row: list[Any] = [c]
            for effect in EFFECT_ORDER:
                value = dist[c][f"{prefix}_{effect}"]
                row.append(value)
                totals[effect] += value
            total = dist[c][f"{prefix}_total"]
            row.append(total)
            totals["total"] += total
            rows.append(row)
        rows.append(
            ["**total**"]
            + [totals[effect] for effect in EFFECT_ORDER]
            + [totals["total"]]
        )
        return rows

    reasons = sorted({r for v in bfcl["drop_reasons"].values() for r in v})
    drop_rows = [
        [reason]
        + [bfcl["drop_reasons"].get(c, {}).get(reason, 0) for c in cats]
        + [sum(bfcl["drop_reasons"].get(c, {}).get(reason, 0) for c in cats)]
        for reason in reasons
    ]
    drop_rows.append(
        ["**total**"]
        + [bfcl["dropped_per_category"][c] for c in cats]
        + [sum(bfcl["dropped_per_category"].values())]
    )

    drop_detail = "\n".join(
        f"- `{d['prompt_id']}` ({d['category']}): `{d['reason']}` — {d['detail']}"
        for d in bfcl["drops"]
    )

    family_rows = [
        [
            row["family"],
            f"`{row['category']}`",
            ts["base_scenarios_per_family"][row["family"]],
            ts["selected_per_family"].get(row["family"], 0),
        ]
        for row in rules.toolsandbox["family_precedence"]
    ]
    family_rows.append(
        [
            "**total**",
            "",
            sum(ts["base_scenarios_per_family"].values()),
            sum(ts["selected_per_family"].values()),
        ]
    )

    variant_rows = [[v, n] for v, n in sorted(ts["selected_per_variant"].items())]

    hash_rows = [[f"`{name}`", f"`{digest}`"] for name, digest in sorted(stats["hashes"].items())]

    return f"""<!-- GENERATED FILE.  Rebuild with `{stats['build_command']}`.  Do not edit by hand. -->
# CrystalCall prompt manifests

Every number in this document is computed by `scripts/build_manifests.py` from
the pinned upstream checkouts listed below; nothing is hand entered.  Rebuild
and verify with:

```sh
{stats['build_command']}
uv run scripts/build_manifests.py --verify
```

## 1. Upstream sources

`experiments/vendor/` is gitignored; `experiments/manifests/vendor_sources.json`
is the committed provenance record (repo, commit, archive sha256, per-file
sha256 of every input read).  This host's egress answers GitHub's
`POST /…/git-upload-pack` with HTTP 401, so anonymous `git clone` fails; the
build fetches the *same commit* as a codeload archive and verifies its sha256.

```sh
{chr(10).join(fetch_lines)}
```

| repo | commit | rule file | rules sha256 |
|---|---|---|---|
| {src['bfcl']['repo']} | `{stats['commits']['bfcl']}` | `{stats['rules_path']}` | `{stats['rules_sha256']}` |
| {src['toolsandbox']['repo']} | `{stats['commits']['toolsandbox']}` | `{stats['rules_path']}` | `{stats['rules_sha256']}` |

Every manifest header carries
`vendor_provenance_sha256: {stats['vendor_provenance_sha256']}`, a digest over
the pinned fields (`{', '.join(PINNED_SOURCE_FIELDS)}`) of the
`{', '.join(CONSUMED_SOURCES)}` entries plus every `input_files_sha256` entry —
not over the whole shared file, so a source another task vendors cannot
invalidate these manifests while any change to what this build reads does.
Before every build, each declared input file is re-hashed and compared against
`input_files_sha256`; a mismatch aborts with the fetch commands.

## 2. BFCL categories and counts

BFCL v4 splits `simple` by language; CrystalCall speculates Python programmatic
tool calls, so the Python split is used.  Ground truth comes from the matching
file under `possible_answer/`.

{md_table(["category", "upstream file", "upstream prompts", "candidates", "dropped", "dev", "test"], counts_rows)}

Selection (declared in `configs/manifest_rules.yml`, seed `{stats['seed']}`):
per category, in sorted category order, candidate prompt ids are sorted
lexicographically and `random.Random({stats['seed']})` draws
`dev_per_category + test_per_category` of them; the first
{bfcl['split_counts']['dev'][cats[0]]} draws go to dev, the remaining
{bfcl['split_counts']['test'][cats[0]]} to test.  Dev and test are disjoint by
construction, and records are written sorted by `(category, prompt_id)`.

Expected calls in the selected 400 prompts: **{bfcl['expected_calls_selected']}**.
Optional parameters omitted because BFCL's ground truth admits omission
(`""` among the alternatives, which is exactly how BFCL's own AST checker reads
it): **{bfcl['omitted_optional_params']}**.  Upstream schema defaults dropped
because they do not type-check against the declared `ParamKind`:
**{bfcl['dropped_defaults']}** (recorded per record in
`meta.dropped_defaults`).

## 3. Tool-name sanitisation

OMP's programmatic tool proxy is attribute based (`tool.<name>(…)`) and the PTC
extractor only accepts an identifier callee, so a dotted upstream name such as
`math.gcd` is unreachable: `tool.math` resolves first and fails.  Every tool
name is therefore sanitised to a Python identifier with
`re.sub(r"{rules.tool_names['sanitize_regex']}", "{rules.tool_names['replacement']}", name)`,
prefixed with `{rules.tool_names['digit_prefix']}` when it would start with a
digit, and disambiguated with `_2`, `_3`, … over the sorted upstream names when
sanitisation collides inside one prompt.

The upstream spelling is preserved verbatim in `ToolDecl.original_name` and in
`meta.tool_name_map` (`sanitised -> upstream`), and is registered as a registry
**alias** of the sanitised name, so a model that emits the dotted spelling
canonicalises to exactly the same identity.  An alias that would collide with
another tool's canonical name or alias in the same prompt is skipped
(`EffectRegistry.register` rejects duplicate identities) and counted in
`meta.skipped_aliases`.

| quantity | count |
|---|---|
| tool declarations in the selected 400 prompts | {bfcl['tool_declarations_selected']} |
| declarations whose name was sanitised | {bfcl['tool_names_sanitised_selected']} |
| distinct upstream names sanitised | {bfcl['distinct_upstream_names_sanitised_selected']} |
| prompts containing at least one sanitised name | {bfcl['prompts_with_sanitised_names_selected']} |
| names needing a `_2`/`_3` collision suffix | {bfcl['collision_suffixed_names_selected']} |
| upstream aliases skipped as colliding | {bfcl['aliases_skipped_selected']} |

`expected_calls[*]["tool"]` always uses the sanitised name, so
`canonicalize(registry_for(rec), c["tool"], c["args"])` names the tool the model
will actually call.

## 4. Effect classification

Declared before any model run, stored as data in `configs/manifest_rules.yml`,
applied as a case-insensitive regex search over the **original upstream**
function name (`effects.classify_on: {rules.effects['classify_on']}`, never the
sanitised identifier) with first-match-wins order `Write` -> `LiveRead` ->
`Pure`:

```yaml
write_regex: '{rules.effects['write_regex']}'
live_read_regex: '{rules.effects['live_read_regex']}'
fallback: {rules.effects['fallback']}
```

Distribution over the declared tool surface of the **selected** 400 prompts
(one count per prompt-tool declaration, so a tool declared in several prompts
counts several times):

{md_table(["category", *EFFECT_ORDER, "total"], effect_rows(bfcl['effect_distribution_selected'], 'tools'))}

Distribution over the **ground-truth calls** of the selected 400 prompts:

{md_table(["category", *EFFECT_ORDER, "total"], effect_rows(bfcl['effect_distribution_selected'], 'calls'))}

For reference, over the full candidate pool
({sum(bfcl['candidates_per_category'].values())} prompts), by declared tool:

{md_table(["category", *EFFECT_ORDER, "total"], effect_rows(bfcl['effect_distribution_candidates'], 'tools'))}

## 5. Drop reasons

Hard filters, all declared in `configs/manifest_rules.yml`: at least one
declared tool, a non-empty single user message, at least one ground-truth call,
every argument value a statically closed literal (a string that parses as a
call expression counts as a nested call), every declared parameter type
supported by `ParamKind`, and every expected call canonicalisable through
`canonicalize(registry_for(rec), call["tool"], call["args"])`.  A prompt that
fails any filter is excluded at build time.

{md_table(["reason", *cats, "total"], drop_rows)}

Reason glossary:

- `unsupported_param_type` — a declared parameter uses a JSON-schema type with
  no `ParamKind` (BFCL's `any`), so the tool surface cannot be typed.
- `canonicalize:type_mismatch` — a ground-truth value contradicts the declared
  schema type (e.g. the string `"false"` for a `boolean` parameter).
- `canonicalize:unknown_parameter` — the ground truth names a parameter the
  function schema does not declare.
- `non_literal_argument` — a ground-truth argument is a call expression rather
  than a closed literal.

Every excluded prompt:

{drop_detail}

## 6. ToolSandbox preselection

ToolSandbox scenarios are stateful sandbox programs, so the manifest pins
*which* scenarios run rather than a static tool surface.  The registry is read
statically with `ast` from the checkout
(`tool_sandbox/scenarios/*.py`); ToolSandbox is never imported or executed.
Every scenario is a `ScenarioExtension(name=…)` literal in one of the four
modules aggregated by `tool_sandbox.scenarios.named_scenarios`;
**{ts['base_scenarios_declared']}** base scenarios are declared
({', '.join(f"{m}: {n}" for m, n in sorted(ts['base_scenarios_per_module'].items()))}).

**Family rule.**  A scenario's family is derived from its own declared
`ScenarioCategories` tags — the `categories=[…]` literal of its
`ScenarioExtension` plus the defaults its module's `named_*` builder appends to
every extension it defines (e.g. `named_multiple_tool_call_scenarios` appends
`MULTIPLE_TOOL_CALL, SINGLE_USER_TURN`).  The first matching row of
`toolsandbox.family_precedence` wins, so families are disjoint and the rule is
most-specific-first:

{md_table(["family", "declared tag", "base scenarios", "selected"], family_rows)}

{len(ts['unassigned_base_scenarios'])} base scenarios carry none of the four
tags (single-tool, single-turn, no special phenomenon) and are therefore never
selected.

Because `canonicalization` and `multi_tool_workflows` overlap heavily upstream,
16 distinct *base* scenarios are not available for every family
(`multi_tool_workflows` has {ts['base_scenarios_per_family']['multi_tool_workflows']}
after precedence).  The quota is therefore filled from the tool-count variants
that `named_scenarios()` itself derives — base (`NO_DISTRACTION_TOOLS`),
`_3_distraction_tools`, `_10_distraction_tools`, `_all_tools` — walked
outermost, so every distinct base scenario of a family is used before any
second variant.  The four `*_scrambled` variants are excluded: they perturb the
tool representation, which must stay fixed across CrystalCall arms.  Selection
uses `random.Random({stats['seed']})` over the sorted base names and depends on
no model outcome.

{md_table(["variant", "selected"], variant_rows)}

Distinct base scenarios among the {sum(ts['selected_per_family'].values())}
selected: **{ts['distinct_base_scenarios_selected']}**.

## 7. Artefacts

| file | contents |
|---|---|
| `experiments/manifests/dev.jsonl` | header + {sum(bfcl['split_counts']['dev'].values())} prompt records |
| `experiments/manifests/test.jsonl` | header + {sum(bfcl['split_counts']['test'].values())} prompt records |
| `experiments/manifests/toolsandbox.jsonl` | header + {sum(ts['selected_per_family'].values())} scenario records |
| `experiments/manifests/*.sha256` | `sha256sum`-format sidecars |
| `experiments/manifests/build_stats.json` | every number in this document |
| `experiments/manifests/vendor_sources.json` | pinned commits, fetch commands, input hashes |

The first line of each manifest is a header record carrying schema
`{MANIFEST_SCHEMA}`, the upstream commit, the rule-file sha256, the seed, the
per-category counts and the build command.

{md_table(["artefact", "sha256"], hash_rows)}
"""


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / DEFAULT_OUT_DIR)
    parser.add_argument("--vendor-dir", type=Path, default=REPO_ROOT / DEFAULT_VENDOR_DIR)
    parser.add_argument("--rules", type=Path, default=REPO_ROOT / DEFAULT_RULES)
    parser.add_argument("--docs", type=Path, default=REPO_ROOT / DEFAULT_DOCS)
    parser.add_argument("--seed", type=int, default=None, help="default: selection.seed from the rules file")
    parser.add_argument("--verify", action="store_true", help="compare artefacts on disk, exit 1 on drift")
    parser.add_argument("--fetch", action="store_true", help="(re)fetch the pinned vendor archives first")
    args = parser.parse_args(argv)

    if args.fetch:
        fetch_vendor(args.vendor_dir, REPO_ROOT / VENDOR_SOURCES)

    seed = args.seed
    if seed is None:
        seed = int(load_rules(args.rules).selection["seed"])

    try:
        artefacts, stats = build(
            vendor_dir=args.vendor_dir,
            out_dir=args.out_dir,
            rules_path=args.rules,
            docs_path=args.docs,
            seed=seed,
        )
    except ManifestBuildError as exc:
        print(f"build failed: {exc.reason}: {exc.detail}", file=sys.stderr)
        return 2

    if args.verify:
        drift: list[str] = []
        for path, blob in sorted(artefacts.items()):
            if not path.exists():
                drift.append(f"missing: {_rel(path)}")
            elif path.read_bytes() != blob:
                drift.append(
                    f"drift: {_rel(path)} on-disk sha256 "
                    f"{hashlib.sha256(path.read_bytes()).hexdigest()} != rebuilt {sha256_bytes(blob)}"
                )
        for name, digest in sorted(stats["hashes"].items()):
            print(f"{digest}  {name}")
        if drift:
            print("\n".join(drift), file=sys.stderr)
            return 1
        print(f"verified {len(artefacts)} artefacts, no drift")
        return 0

    for path, blob in sorted(artefacts.items()):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(blob)
    for name, digest in sorted(stats["hashes"].items()):
        print(f"{digest}  {name}")
    print(f"wrote {len(artefacts)} artefacts to {_rel(args.out_dir)} and {_rel(args.docs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
