#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "numpy>=2.1",
#     "pyyaml>=6.0",
# ]
# ///
"""Build, verify and benchmark the frozen CrystalCall corpus snapshots.

The corpus is assembled from sources that are *always available offline* and
*deterministic for a given interpreter*, so no download and no network fetch is
involved:

1. this repository's own ``src/`` and ``docs/`` trees, and
2. a fixed, sorted allow-list of CPython standard-library module sources
   (including ``ast``, ``json`` and ``sqlite3``) located via ``sysconfig``.

Three snapshots are produced from that one staged corpus:

* ``repo/``    - the frozen text tree searched by ``search_repo`` / ``analyze_module``
* ``sqlite/``  - a read-only SQLite FTS5 index queried by ``sqlite_query``
* ``embed/``   - a precomputed float32 passage-embedding matrix used by ``embed_search``

plus ``bin/embed_search``, the deterministic CUDA kernel compiled from
``src/crystalcall/cuda/embed_search.cu``, and a composite ``corpus@<sha256-16>``
snapshot whose id is a pure function of the three sub-digests.

Usage::

    uv run scripts/build_snapshots.py --out-dir experiments/vendor/corpus
    uv run scripts/build_snapshots.py --out-dir experiments/vendor/corpus --verify
    uv run scripts/build_snapshots.py --bench \
        --emit-config configs/tools.yml --emit-docs docs/tools.md
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from crystalcall.snapshots import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    EMBED_DIM,
    EMBED_TARGET_ROWS,
    EMBED_WINDOW_TOKENS,
    SnapshotSpec,
    build_corpus_snapshot,
    build_embedding_snapshot,
    build_repo_snapshot,
    build_sqlite_fts_snapshot,
    iter_corpus_files,
    load_snapshot,
    make_writable,
    sha256_bytes,
    sha256_file,
    snapshot_permissions_ok,
    sub_snapshot,
    verify_snapshot,
)
from crystalcall.tools import (  # noqa: E402
    CUDA_EMBED_BINARY,
    CUDA_EMBED_SOURCE,
    LOCAL_TOOL_DECLS,
    local_impls,
    local_registry,
    resolve_embed_binary,
    resolve_ripgrep,
)

DEFAULT_OUT = Path("experiments/vendor/corpus")

#: Repository trees vendored into the corpus (relative to the repo root).
SELF_TREES: tuple[str, ...] = ("src", "docs")

#: Repo-relative paths excluded from the corpus because this script *generates*
#: them.  Including a generated file would make the corpus id depend on the
#: previous build's output, so `build -> bench -> emit-docs -> build` would never
#: converge.
SELF_EXCLUDE: frozenset[str] = frozenset({"docs/tools.md"})

#: Fixed, sorted allow-list of stdlib modules/packages vendored into the corpus.
#: Deterministic for a given CPython build and always present offline.
STDLIB_MODULES: tuple[str, ...] = (
    "argparse.py",
    "ast.py",
    "asyncio",
    "collections",
    "concurrent",
    "configparser.py",
    "csv.py",
    "ctypes",
    "dataclasses.py",
    "dis.py",
    "email",
    "encodings",
    "enum.py",
    "html",
    "http",
    "importlib",
    "inspect.py",
    "json",
    "lib2to3",
    "logging",
    "multiprocessing",
    "re",
    "sqlite3",
    "statistics.py",
    "tarfile.py",
    "tomllib",
    "typing.py",
    "unittest",
    "urllib",
    "wsgiref",
    "xml",
    "zipfile",
)

MIN_FILES = 300
MIN_LINES = 20_000

DB_NAME = "corpus.db"

#: The exact nvcc invocation; recorded verbatim in configs/tools.yml.
NVCC_ARCH = "sm_121"
NVCC_FLAGS: tuple[str, ...] = ("-O2", f"-arch={NVCC_ARCH}", "--fmad=false")

#: Pinned, vendored ripgrep provenance (the binary itself is gitignored).
RG_PROVENANCE: dict[str, Any] = {
    "tool": "ripgrep",
    "version": "14.1.1",
    "revision": "4649aa9700",
    "target": "aarch64-unknown-linux-gnu",
    "url": "https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/"
    "ripgrep-14.1.1-aarch64-unknown-linux-gnu.tar.gz",
    "tarball_sha256": "c827481c4ff4ea10c9dc7a4022c8de5db34a5737cb74484d62eb94a95841ab2f",
    "fetch_command": (
        "curl -fsSL https://github.com/BurntSushi/ripgrep/releases/download/14.1.1/"
        "ripgrep-14.1.1-aarch64-unknown-linux-gnu.tar.gz -o /tmp/rg.tar.gz && "
        "sha256sum /tmp/rg.tar.gz  # expect "
        "c827481c4ff4ea10c9dc7a4022c8de5db34a5737cb74484d62eb94a95841ab2f && "
        "tar -xzf /tmp/rg.tar.gz -C /tmp && "
        "install -Dm755 /tmp/ripgrep-14.1.1-aarch64-unknown-linux-gnu/rg "
        "experiments/vendor/bin/rg"
    ),
    "installed_path": "experiments/vendor/bin/rg",
    "note": "experiments/vendor/ is gitignored; this record is the committed "
    "provenance for the binary. Resolution order at runtime: $CRYSTALCALL_RG, "
    "then PATH, then this vendored binary.",
}

#: Fixed, deterministic benchmark workload per tool.
BENCH_CALLS: tuple[tuple[str, dict[str, Any]], ...] = (
    ("search_repo", {"query": "def ", "path": ".", "max_results": 50}),
    (
        "sqlite_query",
        {
            "database": DB_NAME,
            "sql": "SELECT path, lines FROM doc_meta ORDER BY path LIMIT 20",
            "max_rows": 100,
        },
    ),
    ("analyze_module", {"path": "stdlib/ast.py", "checks": ["complexity", "imports", "syntax"]}),
    ("embed_search", {"query": "frozen snapshot identity", "top_k": 5}),
    ("write_note", {"path": "bench.txt", "text": "benchmark"}),
    ("read_live_clock", {}),
)

GEN_BEGIN = "<!-- BEGIN GENERATED: durations -->"
GEN_END = "<!-- END GENERATED: durations -->"


# --------------------------------------------------------------------------- #
# staging
# --------------------------------------------------------------------------- #


def _copy_tree(src: Path, dest: Path, *, exclude_prefix: str = "") -> tuple[int, int]:
    files = lines = 0
    root = src.resolve()
    for path in iter_corpus_files(root):
        rel = path.relative_to(root).as_posix()
        if f"{exclude_prefix}{rel}" in SELF_EXCLUDE:
            continue
        out = dest / path.relative_to(root)
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            text = path.read_bytes().decode("utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        out.write_text(text, encoding="utf-8")
        files += 1
        lines += len(text.splitlines())
    return files, lines


def stage_corpus(staging: Path) -> dict[str, Any]:
    """Assemble the corpus under ``staging`` and describe what went in."""
    sources: list[dict[str, Any]] = []
    total_files = total_lines = 0

    for tree in SELF_TREES:
        src = REPO_ROOT / tree
        if not src.is_dir():
            sources.append(
                {"kind": "self", "path": tree, "files": 0, "lines": 0,
                 "note": "absent at build time"}
            )
            continue
        files, lines = _copy_tree(src, staging / "self" / tree, exclude_prefix=f"{tree}/")
        sources.append({"kind": "self", "path": tree, "files": files, "lines": lines})
        total_files += files
        total_lines += lines

    stdlib = Path(sysconfig.get_paths()["stdlib"]).resolve()
    for module in STDLIB_MODULES:
        origin = stdlib / module
        if not origin.exists():
            raise FileNotFoundError(f"stdlib module {module!r} not found under {stdlib}")
        if origin.is_file():
            text = origin.read_text(encoding="utf-8")
            out = staging / "stdlib" / module
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            files, lines = 1, len(text.splitlines())
        else:
            files, lines = _copy_tree(origin, staging / "stdlib" / module)
        sources.append({"kind": "stdlib", "path": module, "files": files, "lines": lines})
        total_files += files
        total_lines += lines

    if total_files < MIN_FILES or total_lines < MIN_LINES:
        raise SystemExit(
            f"corpus too small: {total_files} files / {total_lines} lines "
            f"(need >= {MIN_FILES} / >= {MIN_LINES}). Extend STDLIB_MODULES."
        )
    return {
        "sources": sources,
        "files": total_files,
        "lines": total_lines,
        "stdlib_root": str(stdlib),
    }


# --------------------------------------------------------------------------- #
# cuda kernel
# --------------------------------------------------------------------------- #


def compile_cuda_kernel(out_dir: Path) -> dict[str, Any]:
    """Compile the deterministic ``embed_search`` kernel into ``<out_dir>/bin``.

    Never fatal: on any failure the exact nvcc error is recorded and returned so
    the caller can report a loud, explicit degradation to the numpy path.
    """
    nvcc = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    record: dict[str, Any] = {
        "source": str(CUDA_EMBED_SOURCE.relative_to(REPO_ROOT)),
        "nvcc": nvcc if Path(nvcc).exists() else None,
        "flags": list(NVCC_FLAGS),
        "arch": NVCC_ARCH,
        "compiled": False,
        "binary": None,
        "binary_sha256": None,
        "nvcc_version": None,
        "command": None,
        "error": None,
    }
    if not Path(nvcc).exists():
        record["error"] = "nvcc not found on PATH or at /usr/local/cuda/bin/nvcc"
        return record

    version = subprocess.run([nvcc, "--version"], capture_output=True, check=False)
    record["nvcc_version"] = (
        version.stdout.decode("utf-8", "replace").strip().splitlines()[-1]
        if version.returncode == 0
        else None
    )

    bin_dir = out_dir / "bin"
    make_writable(out_dir, recursive=False)
    bin_dir.mkdir(parents=True, exist_ok=True)
    target = bin_dir / CUDA_EMBED_BINARY
    if target.exists():
        os.chmod(target, 0o755)
    cmd = [nvcc, *NVCC_FLAGS, "-o", str(target), str(CUDA_EMBED_SOURCE)]
    record["command"] = " ".join(cmd)
    proc = subprocess.run(cmd, capture_output=True, check=False, cwd=REPO_ROOT)
    if proc.returncode != 0:
        record["error"] = proc.stderr.decode("utf-8", "replace").strip()[:4000]
        return record
    os.chmod(target, 0o555)
    record["compiled"] = True
    record["binary"] = str(target.relative_to(out_dir))
    record["binary_sha256"] = sha256_file(target)
    return record


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #


def build(out_dir: Path) -> tuple[SnapshotSpec, dict[str, Any]]:
    out_dir = out_dir.resolve()
    make_writable(out_dir, recursive=False)
    out_dir.mkdir(parents=True, exist_ok=True)

    staging_parent = Path(tempfile.mkdtemp(prefix="crystalcall-staging-"))
    try:
        staging = staging_parent / "corpus"
        staging.mkdir()
        staged = stage_corpus(staging)

        repo = build_repo_snapshot(staging, out_dir / "repo")
        sqlite_spec = build_sqlite_fts_snapshot(repo.root, out_dir / "sqlite" / DB_NAME)
        embed = build_embedding_snapshot(repo.root, out_dir / "embed", DEFAULT_EMBED_MODEL)
        cuda = compile_cuda_kernel(out_dir)
        make_writable(out_dir, recursive=False)
        corpus = build_corpus_snapshot(out_dir, repo, sqlite_spec, embed)
    finally:
        shutil.rmtree(staging_parent, ignore_errors=True)

    manifest = {
        "corpus": corpus.to_json(),
        "sub": {
            name: sub_snapshot(corpus, name).to_json() for name in ("embed", "repo", "sqlite")
        },
        "staged": staged,
        "cuda": cuda,
        "ripgrep": _rg_record(),
        "build": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "embed_model_id": DEFAULT_EMBED_MODEL,
            "embed_dim": EMBED_DIM,
            "embed_window_tokens": EMBED_WINDOW_TOKENS,
            "embed_target_rows": EMBED_TARGET_ROWS,
            "command": f"uv run scripts/build_snapshots.py --out-dir {DEFAULT_OUT}",
            "network_fetch": None,
            "note": "All corpus inputs are local: this repository's own src/ and docs/ "
            "trees plus a fixed allow-list of CPython stdlib module sources located "
            "via sysconfig. No download is performed, so there is no fetch command "
            "or external sha256 to record for the corpus itself; the one vendored "
            "binary (ripgrep) has its provenance recorded separately.",
        },
        "read_only": snapshot_permissions_ok(corpus),
    }
    _write_json(out_dir, "CORPUS_MANIFEST.json", manifest)
    _write_committed_sources(manifest)
    _write_bin_provenance(manifest)
    return corpus, manifest


def _write_json(out_dir: Path, name: str, payload: Mapping[str, Any]) -> Path:
    make_writable(out_dir, recursive=False)
    path = out_dir / name
    if path.exists():
        os.chmod(path, 0o644)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o444)
    os.chmod(out_dir, 0o555)
    return path


def _rg_record() -> dict[str, Any]:
    resolved = resolve_ripgrep()
    record = dict(RG_PROVENANCE)
    record["resolved_path"] = str(resolved) if resolved else None
    record["resolved_via"] = (
        None
        if resolved is None
        else (
            "CRYSTALCALL_RG"
            if os.environ.get("CRYSTALCALL_RG", "").strip()
            else ("PATH" if shutil.which("rg") else "vendored")
        )
    )
    record["resolved_version"] = _rg_version()
    record["binary_sha256"] = sha256_file(resolved) if resolved and resolved.is_file() else None
    return record


def _write_committed_sources(manifest: Mapping[str, Any]) -> Path:
    """The small, committed provenance manifest for the (uncommitted) corpus."""
    payload = {
        "description": "Provenance of the frozen CrystalCall tool corpus. The corpus "
        "itself lives under experiments/vendor/ (gitignored) and is rebuilt with the "
        "command below; every corpus input is local, so no download is involved.",
        "build_command": manifest["build"]["command"],
        "verify_command": f"uv run scripts/build_snapshots.py --out-dir {DEFAULT_OUT} --verify",
        "self_trees": list(SELF_TREES),
        "stdlib_modules": list(STDLIB_MODULES),
        "min_files": MIN_FILES,
        "min_lines": MIN_LINES,
        "embed": {
            "model_id": DEFAULT_EMBED_MODEL,
            "dim": EMBED_DIM,
            "window_tokens": EMBED_WINDOW_TOKENS,
            "target_rows": EMBED_TARGET_ROWS,
        },
        "sqlite_database": DB_NAME,
        "vendored_binaries": [RG_PROVENANCE],
        "cuda": {
            "source": manifest["cuda"]["source"],
            "command": manifest["cuda"]["command"],
            "flags": manifest["cuda"]["flags"],
            "nvcc_version": manifest["cuda"]["nvcc_version"],
            "binary_sha256": manifest["cuda"]["binary_sha256"],
        },
        "id_derivation": "sha256 of canonical-JSON {kind, sorted [name, sha256] entries, "
        "extra} truncated to 16 hex chars; see src/crystalcall/snapshots.py",
        "note": "Snapshot ids are content-derived, so they change whenever this "
        "repository's src/ or docs/ change or the interpreter's stdlib changes. The "
        f"authoritative ids for a given build are in {DEFAULT_OUT}/CORPUS_MANIFEST.json.",
        "last_build": {
            "python_version": manifest["build"]["python_version"],
            "machine": manifest["build"]["machine"],
            "files": manifest["staged"]["files"],
            "lines": manifest["staged"]["lines"],
            "corpus_snapshot_id": manifest["corpus"]["snapshot_id"],
            "sub_snapshot_ids": {
                name: sub["snapshot_id"] for name, sub in sorted(manifest["sub"].items())
            },
        },
    }
    path = REPO_ROOT / "configs" / "corpus_sources.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(blob, encoding="utf-8")
    (path.parent / "corpus_sources.sha256").write_text(
        f"{sha256_bytes(blob.encode('utf-8'))}  configs/corpus_sources.json\n",
        encoding="utf-8",
    )
    return path


def _write_bin_provenance(manifest: Mapping[str, Any]) -> Path:
    """Provenance sidecar next to the vendored binaries (force-add to commit)."""
    path = REPO_ROOT / "experiments" / "vendor" / "bin" / "PROVENANCE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "binaries": [manifest["ripgrep"]],
        "note": "experiments/vendor/ is gitignored. Commit this file explicitly with "
        "`git add -f experiments/vendor/bin/PROVENANCE.json`.",
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


# --------------------------------------------------------------------------- #
# verify
# --------------------------------------------------------------------------- #


def verify(out_dir: Path) -> dict[str, Any]:
    corpus = load_snapshot(out_dir)
    ok = verify_snapshot(corpus)
    perms = snapshot_permissions_ok(corpus)
    report: dict[str, Any] = {
        "corpus_snapshot_id": corpus.snapshot_id,
        "digest_ok": bool(ok),
        "read_only_ok": bool(perms),
        "embed_search_binary": str(resolve_embed_binary(corpus.root) or ""),
        "ripgrep": str(resolve_ripgrep() or ""),
        "sub": {},
    }
    for name in ("embed", "repo", "sqlite"):
        sub = sub_snapshot(corpus, name)
        report["sub"][name] = {
            "snapshot_id": sub.snapshot_id,
            "digest_ok": bool(verify_snapshot(sub)),
            "meta": sub.meta,
        }
    if not perms:
        raise SystemExit("snapshot permissions are not read-only (expected 0444/0555)")
    return report


# --------------------------------------------------------------------------- #
# benchmark
# --------------------------------------------------------------------------- #


def _snapshot_files(corpus: SnapshotSpec) -> list[Path]:
    out: list[Path] = []
    for name in ("embed", "repo", "sqlite"):
        sub = sub_snapshot(corpus, name)
        out.extend(p for p in sorted(sub.root.rglob("*")) if p.is_file())
    return out


def _drop_page_cache(paths: Sequence[Path], ballast: Path | None) -> None:
    """Evict the snapshot's pages without root.

    ``sync`` flushes dirty pages, ``posix_fadvise(POSIX_FADV_DONTNEED)`` drops the
    snapshot's own page-cache residency (the only mechanism that actually works
    unprivileged, and the only one that is meaningful on a 128 GB host), and a
    streaming read of a large unrelated file adds cache pressure.  No ``sudo``
    and no write to ``/proc/sys/vm/drop_caches`` is used anywhere.
    """
    os.sync()
    for path in paths:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    if ballast is not None and ballast.is_file():
        try:
            with ballast.open("rb") as fh:
                while fh.read(1 << 22):
                    pass
            fd = os.open(ballast, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except OSError:
            pass


def _percentiles(samples: Sequence[float]) -> dict[str, float]:
    arr = np.asarray(sorted(samples), dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": round(float(arr.min()), 6),
        "p50": round(float(np.percentile(arr, 50, method="linear")), 6),
        "p95": round(float(np.percentile(arr, 95, method="linear")), 6),
        "max": round(float(arr.max()), 6),
        "mean": round(float(arr.mean()), 6),
    }


def _find_ballast() -> Path | None:
    """A large unrelated file used purely for cache pressure."""
    roots = [Path("/usr/lib/aarch64-linux-gnu"), Path("/usr/lib"), Path("/usr/local/cuda")]
    best: Path | None = None
    best_size = 0
    for root in roots:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("**/*.so*"))[:4000]:
            try:
                size = path.stat().st_size
            except OSError:
                continue
            if size > best_size:
                best, best_size = path, size
            if best_size > 64 << 20:
                return best
        if best is not None:
            return best
    return best


def benchmark(out_dir: Path, warm: int, cold: int) -> dict[str, Any]:
    corpus = load_snapshot(out_dir)
    registry = local_registry(corpus)
    impls = local_impls(corpus)
    files = _snapshot_files(corpus)
    ballast = _find_ballast()
    cancel = threading.Event()

    import crystalcall.tools as tools_mod

    scratch = Path(tempfile.mkdtemp(prefix="crystalcall-bench-scratch-"))
    os.environ["CRYSTALCALL_SCRATCH_DIR"] = str(scratch)
    os.environ.pop("CRYSTALCALL_TOOL_SLEEP_MS", None)

    results: dict[str, Any] = {}
    try:
        for name, args in BENCH_CALLS:
            impl = impls[name]
            spec = registry.resolve(name)

            cold_samples: list[float] = []
            for _ in range(cold):
                tools_mod._IDS_CACHE.clear()
                _drop_page_cache(files, ballast)
                t0 = time.perf_counter_ns()
                impl(args, cancel)
                cold_samples.append((time.perf_counter_ns() - t0) / 1e6)

            for _ in range(3):  # warm-up, discarded
                impl(args, cancel)
            warm_samples: list[float] = []
            for _ in range(warm):
                t0 = time.perf_counter_ns()
                impl(args, cancel)
                warm_samples.append((time.perf_counter_ns() - t0) / 1e6)

            value = impl(args, cancel)
            warm_stats = _percentiles(warm_samples)
            results[name] = {
                "effect": spec.effect.value,
                "resource": spec.resource.value,
                "concurrency_limit": spec.concurrency_limit,
                "args": args,
                "cold_ms": _percentiles(cold_samples),
                "warm_ms": warm_stats,
                "expected_duration_ms": warm_stats["p50"],
                "engine": value.get("engine") if isinstance(value, dict) else None,
                "device": value.get("device") if isinstance(value, dict) else None,
            }
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        os.environ.pop("CRYSTALCALL_SCRATCH_DIR", None)

    payload = {
        "corpus_snapshot_id": corpus.snapshot_id,
        "registry_fingerprint": registry.fingerprint(),
        "host": {
            "machine": platform.machine(),
            "python_version": platform.python_version(),
            "cpu_count": os.cpu_count(),
            "ripgrep": _rg_version(),
            "ripgrep_path": str(resolve_ripgrep() or ""),
            "torch": _torch_status(),
            "gpu": _gpu_status(),
            "embed_search_binary": str(resolve_embed_binary(corpus.root) or ""),
        },
        "cache_drop": {
            "method": "os.sync() + posix_fadvise(POSIX_FADV_DONTNEED) over all snapshot "
            "files + streaming read of a large unrelated file",
            "ballast": str(ballast) if ballast else None,
            "sudo": False,
        },
        "warm_reps": warm,
        "cold_reps": cold,
        "tools": results,
    }
    _write_json(out_dir, "BENCH.json", payload)
    return payload


def _rg_version() -> str | None:
    exe = resolve_ripgrep()
    if exe is None:
        return None
    proc = subprocess.run([str(exe), "--version"], capture_output=True, check=False)
    lines = proc.stdout.decode("utf-8", "replace").splitlines()
    return lines[0].strip() if lines else None


def _torch_status() -> dict[str, Any]:
    try:
        import torch
    except Exception as exc:
        return {
            "installed": False,
            "cuda_available": False,
            "reason": type(exc).__name__,
            "detail": str(exc)[:120],
        }
    try:
        return {
            "installed": True,
            "version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
    except Exception as exc:  # pragma: no cover
        return {
            "installed": True,
            "version": getattr(torch, "__version__", "?"),
            "cuda_available": False,
            "reason": str(exc)[:120],
        }


def _gpu_status() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        return {"available": False}
    proc = subprocess.run(
        [exe, "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return {"available": False}
    rows = [r.strip() for r in proc.stdout.decode("utf-8", "replace").splitlines() if r.strip()]
    return {"available": True, "devices": rows}


# --------------------------------------------------------------------------- #
# emitters
# --------------------------------------------------------------------------- #

_CONFIG_HEADER = """\
# CrystalCall: frozen declaration of the local (real) tool families.
#
# GENERATED FILE - do not hand-edit.  Regenerate with:
#   uv run scripts/build_snapshots.py --bench --emit-config configs/tools.yml
#
# Every `expected_duration_ms` below is the measured warm p50 of the fixed
# benchmark workload recorded in `{out}/BENCH.json`; nothing here is
# hand-entered.  Effects and resource classes are declarations, not
# measurements: they are the static annotations the launch gate reasons over,
# and they are what makes `write_note` / `read_live_clock` permanently
# ineligible for speculation.
"""


def emit_config(bench: Mapping[str, Any], path: Path, manifest: Mapping[str, Any]) -> Path:
    tools: list[dict[str, Any]] = []
    by_name = {d.name: d for d in LOCAL_TOOL_DECLS}
    for name in sorted(bench["tools"]):
        row = bench["tools"][name]
        decl = by_name[name]
        tools.append(
            {
                "name": name,
                "effect": row["effect"],
                "resource": row["resource"],
                "expected_duration_ms": row["expected_duration_ms"],
                "concurrency_limit": row["concurrency_limit"],
                "speculation_eligible": row["effect"] in ("Pure", "SnapshotRead"),
                "description": decl.description,
                "params": [dict(p) for p in decl.params],
                "measured": {
                    "benchmark_args": row["args"],
                    "warm_ms": row["warm_ms"],
                    "cold_ms": row["cold_ms"],
                    "engine": row["engine"],
                    "device": row["device"],
                },
            }
        )
    payload = {
        "schema": 1,
        "provenance": {
            "corpus_snapshot_id": bench["corpus_snapshot_id"],
            "registry_fingerprint": bench["registry_fingerprint"],
            "warm_reps": bench["warm_reps"],
            "cold_reps": bench["cold_reps"],
            "host": bench["host"],
            "cache_drop": bench["cache_drop"],
            "cuda": manifest.get("cuda"),
            "ripgrep": manifest.get("ripgrep"),
        },
        "tools": tools,
    }
    body = yaml.safe_dump(
        payload, sort_keys=False, default_flow_style=False, width=100, allow_unicode=True
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_CONFIG_HEADER.format(out=DEFAULT_OUT) + body, encoding="utf-8")
    return path


def _duration_table(bench: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    host = bench["host"]
    cuda = manifest.get("cuda") or {}
    lines = [
        f"Measured on `{host['machine']}`, CPython {host['python_version']}, "
        f"{host['cpu_count']} cores, GPU {', '.join((host.get('gpu') or {}).get('devices', [])) or 'n/a'}; "
        f"{bench['cold_reps']} cold and {bench['warm_reps']} warm invocations per tool "
        f"(3 discarded warm-ups). Corpus `{bench['corpus_snapshot_id']}`.",
        "",
        "| tool | effect | resource | cold p50 (ms) | cold p95 (ms) | warm p50 (ms) |"
        " warm p95 (ms) | frozen `expected_duration_ms` |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for name in sorted(bench["tools"]):
        row = bench["tools"][name]
        lines.append(
            f"| `{name}` | {row['effect']} | {row['resource']} | "
            f"{row['cold_ms']['p50']:.4g} | {row['cold_ms']['p95']:.4g} | "
            f"{row['warm_ms']['p50']:.4g} | {row['warm_ms']['p95']:.4g} | "
            f"{row['expected_duration_ms']:.4g} |"
        )
    torch = host["torch"]
    lines += [
        "",
        f"Cache-drop method for the cold rows: {bench['cache_drop']['method']} "
        f"(ballast file `{bench['cache_drop']['ballast']}`); `sudo` was not used.",
        "",
        f"- `search_repo` engine: **{bench['tools']['search_repo']['engine']}** "
        f"({host['ripgrep'] or 'ripgrep unavailable'} at `{host['ripgrep_path'] or 'n/a'}`).",
        f"- `embed_search` device: **{bench['tools']['embed_search']['device']}** "
        f"(binary `{host['embed_search_binary'] or 'not built'}`; "
        f"nvcc `{cuda.get('nvcc_version') or 'n/a'}`; compile command "
        f"`{cuda.get('command') or 'n/a'}`; binary sha256 "
        f"`{cuda.get('binary_sha256') or 'n/a'}`).",
        f"- `torch`: {'installed ' + str(torch.get('version')) if torch.get('installed') else '**not installed**'}"
        f", `torch.cuda.is_available()` = {bool(torch.get('cuda_available'))}. "
        "The GPU path does not use torch at all: it is the standalone CUDA kernel above.",
    ]
    return "\n".join(lines)


def emit_docs(bench: Mapping[str, Any], path: Path, manifest: Mapping[str, Any]) -> Path:
    block = f"{GEN_BEGIN}\n{_duration_table(bench, manifest)}\n{GEN_END}"
    if not path.is_file():
        raise SystemExit(f"{path} does not exist; create it with the generated block markers")
    text = path.read_text(encoding="utf-8")
    if GEN_BEGIN not in text or GEN_END not in text:
        raise SystemExit(f"{path} is missing the {GEN_BEGIN} / {GEN_END} markers")
    head, _, rest = text.partition(GEN_BEGIN)
    _, _, tail = rest.partition(GEN_END)
    path.write_text(head + block + tail, encoding="utf-8")
    return path


def _load_manifest(out_dir: Path) -> dict[str, Any]:
    path = out_dir / "CORPUS_MANIFEST.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--verify", action="store_true", help="verify an existing snapshot set")
    ap.add_argument("--bench", action="store_true", help="run the cold/warm micro-benchmark")
    ap.add_argument("--warm", type=int, default=20)
    ap.add_argument("--cold", type=int, default=5)
    ap.add_argument("--emit-config", type=Path, default=None)
    ap.add_argument("--emit-docs", type=Path, default=None)
    args = ap.parse_args(argv)

    out_dir = args.out_dir if args.out_dir.is_absolute() else (REPO_ROOT / args.out_dir)

    if args.verify:
        report = verify(out_dir)
        print(json.dumps(report, indent=2, sort_keys=True))
        print("VERIFY OK")
        return 0

    manifest: dict[str, Any] = {}
    if not args.bench:
        corpus, manifest = build(out_dir)
        print(
            json.dumps(
                {
                    "corpus": manifest["corpus"],
                    "sub": {k: v["snapshot_id"] for k, v in sorted(manifest["sub"].items())},
                    "files": manifest["staged"]["files"],
                    "lines": manifest["staged"]["lines"],
                    "read_only": manifest["read_only"],
                    "cuda_compiled": manifest["cuda"]["compiled"],
                    "cuda_error": manifest["cuda"]["error"],
                    "ripgrep": manifest["ripgrep"]["resolved_path"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        print("BUILD OK")

    if args.bench:
        if not manifest:
            manifest = _load_manifest(out_dir)
        bench = benchmark(out_dir, warm=args.warm, cold=args.cold)
        print(_duration_table(bench, manifest))
        for flag, emitter in (("emit_config", emit_config), ("emit_docs", emit_docs)):
            dest = getattr(args, flag)
            if dest is None:
                continue
            dest = dest if dest.is_absolute() else (REPO_ROOT / dest)
            print(f"wrote {emitter(bench, dest, manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
