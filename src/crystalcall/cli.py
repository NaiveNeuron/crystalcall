"""``crystalcall`` command surface.

Every experiment, figure and report is reproducible from one documented command
without editing source.  Commands fail loudly on configuration drift or
manifest-hash mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import SCHEMA_VERSIONS, __version__

ROOT_ENV = "CRYSTALCALL_ROOT"


def repo_root() -> Path:
    env = os.environ.get(ROOT_ENV)
    if env:
        return Path(env).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists() and (parent / "configs").exists():
            return parent
    return Path.cwd()


#: Trace subtrees that the BFCL policy replay must NOT pick up by default.
#: ToolSandbox episodes are stateful and have their own pipeline and manifest;
#: replaying them here would only produce `no_registry` skips.
# `toolsandbox` is a stateful episode tree, not a replayable trajectory.
# `_attempt1` is the first pilot attempt: it is the SAME 100 prompts under
# the pre-repair prompt, so replaying it alongside the current run would report
# 200 "episodes" from 100 prompts and double-count every rate and interval.
# It is still replayable by naming its shard explicitly.
# `heldout` holds the 300-prompt test-manifest runs; they are replayed into a
# separate derived tree (experiments/derived_heldout) and must never be pooled
# with the development shards under the same (arm, policy) key.
REPLAY_EXCLUDED_SUBTREES = ("toolsandbox", "_attempt1", "heldout")


def default_replay_shards(root: Path) -> list[Path]:
    raw = root / "experiments/raw"
    if not raw.exists():
        return []
    return sorted(
        p
        for p in raw.rglob("*.jsonl")
        if not any(part in REPLAY_EXCLUDED_SUBTREES for part in p.relative_to(raw).parts)
    )

def _write_json(path: Path, obj: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    return path


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


def cmd_doctor(args: argparse.Namespace) -> int:
    from . import doctor

    root = repo_root()
    doc = doctor.collect(
        root,
        image=args.image,
        probe_image=args.probe_image,
        probe_omp=not args.no_omp,
    )
    out = _write_json(Path(args.out or root / "reports/doctor.json"), doc)
    print(f"doctor digest {doc['digest'][:16]} -> {out}")
    hw = doc["hardware"]
    print(f"  host       {hw['hostname']} {hw['platform']}")
    print(f"  cpus       {hw['cpu_count']} parts={hw.get('cpu_parts')}")
    gpu = (doc["gpu"]["query_gpu"] or {}).get("stdout", "")
    print(f"  gpu        {gpu}")
    print(f"  power      readings={doc['gpu']['telemetry']['power_readings_supported']}")
    print(f"  omp        {doc['pins']['omp']['version']} settings={doc['omp_settings']['sha256'][:16] if doc['omp_settings']['sha256'] else None}")
    for name, m in doc["pins"]["models"].items():
        print(f"  model      {name} {m['status']} rev={m.get('revisions')} {round(m.get('total_bytes',0)/1e9,1)}GB")
    print(f"  image      {doc['pins']['image'].get('status')} {doc['pins']['image'].get('repo_digests')}")
    if "vllm" in doc["pins"]:
        print(f"  vllm       {doc['pins']['vllm']}")
    for name, v in sorted((doc["pins"]["vendor"] or {}).items()):
        if not isinstance(v, dict):
            continue
        print(f"  vendor     {name} {v.get('status')} {str(v.get('commit',''))[:12]}")
    if "omp_lite" in doc:
        ol = doc["omp_lite"]
        if ol.get("status") == "ok":
            print(
                f"  omp-lite   prompt={ol['system_prompt_sha256'][:16]} "
                f"tools={ol['tool_names']} repr={ol['tool_representation_sha256'][:16]}"
            )
        else:
            print(f"  omp-lite   ERROR {ol.get('error','')[:200]}")
    missing = [k for k, v in (doc.get("config_hashes") or {}).items() if v is None]
    if missing:
        print(f"  MISSING    {missing}")
    return 0


# --------------------------------------------------------------------------- #
# selftest (the safety core)
# --------------------------------------------------------------------------- #


SAFETY_TESTS = (
    "tests/test_canonicalize.py",
    "tests/test_parse_ptc.py",
    "tests/test_effect_safety.py",
    "tests/test_claim_discard.py",
    "tests/test_semantic_equivalence.py",
)


def cmd_selftest(args: argparse.Namespace) -> int:
    root = repo_root()
    targets = list(args.targets) or [t for t in SAFETY_TESTS if (root / t).exists()]
    cmd = [sys.executable, "-m", "pytest", "-q", *targets]
    print("+", " ".join(cmd))
    p = subprocess.run(cmd, cwd=root, check=False)  # noqa: S603
    record = {
        "schema_version": "crystalcall-selftest/1",
        "targets": targets,
        "returncode": p.returncode,
        "unix_ns": time.time_ns(),
    }
    _write_json(root / "reports/selftest.json", record)
    return p.returncode


# --------------------------------------------------------------------------- #
# serve
# --------------------------------------------------------------------------- #


def cmd_serve(args: argparse.Namespace) -> int:
    root = repo_root()
    script = root / ("scripts/serve_diffusiongemma.sh" if args.model == "diffusion" else "scripts/serve_gemma4.sh")
    if not script.exists():
        print(f"missing {script}", file=sys.stderr)
        return 2
    env = dict(os.environ)
    if args.trace_sink:
        env["CRYSTALCALL_TRACE"] = "1"
        env["CRYSTALCALL_TRACE_SINK"] = args.trace_sink
    if args.trace_level:
        env["CRYSTALCALL_TRACE_LEVEL"] = args.trace_level
    cmd = ["bash", str(script), *args.extra]
    print("+", " ".join(cmd))
    return subprocess.run(cmd, cwd=root, env=env, check=False).returncode  # noqa: S603


# --------------------------------------------------------------------------- #
# manifests
# --------------------------------------------------------------------------- #


def cmd_freeze_manifest(args: argparse.Namespace) -> int:
    root = repo_root()
    names = ("dev.jsonl", "test.jsonl", "toolsandbox.jsonl")
    frozen: dict[str, Any] = {
        "schema_version": SCHEMA_VERSIONS["manifest"],
        "frozen_unix_ns": time.time_ns(),
        "manifests": {},
        "configs": {},
        "policy": {},
    }
    missing = []
    for n in names:
        p = root / "experiments/manifests" / n
        digest = _sha256(p)
        if digest is None:
            missing.append(n)
        frozen["manifests"][n] = {"sha256": digest, "path": str(p.relative_to(root))}
    for n in (
        "configs/omp-lite.yml",
        "configs/models.yml",
        "configs/diffusiongemma.yaml",
        "configs/tools.yml",
        "configs/manifest_rules.yml",
        "configs/system_prompt.md",
        # The P-ReAct-style arm's delimiter token ids, in-span entropy bound and
        # stability depth are load-bearing for the S8 decisive comparison; a
        # post-freeze change here would silently redefine that baseline.
        "configs/preact.yml",
    ):
        frozen["configs"][n] = _sha256(root / n)
    policy_path = root / "configs/frozen_policy.json"
    if policy_path.exists():
        frozen["policy"] = json.loads(policy_path.read_text())
        frozen["policy_sha256"] = _sha256(policy_path)
    # The code-level surface that decides an arm's behaviour is frozen too, so
    # a silent edit to an arm's model/prompt/instrumentation or to the gate
    # library cannot pass unnoticed after the held-out runs.
    from .harness import ARMS
    from .scheduler import POLICY_LIBRARY

    frozen["arms"] = {
        name: {
            "model": spec.model,
            "system_prompt": spec.system_prompt,
            "instrumentation": spec.instrumentation,
            "autoregressive": spec.autoregressive,
            "sampler_env": dict(spec.sampler_env),
        }
        for name, spec in sorted(ARMS.items())
    }
    frozen["arms_digest"] = hashlib.sha256(
        json.dumps(frozen["arms"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    frozen["policy_library"] = {
        name: pol.fingerprint() for name, pol in sorted(POLICY_LIBRARY.items())
    }
    frozen["policy_library_digest"] = hashlib.sha256(
        json.dumps(frozen["policy_library"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out = _write_json(root / "configs/frozen_manifest.json", frozen)
    print(f"froze {out}")
    for n, v in frozen["manifests"].items():
        print(f"  {n:22s} {v['sha256']}")
    if missing:
        print(f"MISSING manifests: {missing}", file=sys.stderr)
        return 1
    return 0


def cmd_verify_freeze(args: argparse.Namespace) -> int:
    root = repo_root()
    path = root / "configs/frozen_manifest.json"
    if not path.exists():
        print("no frozen manifest; run `crystalcall freeze-manifest` first", file=sys.stderr)
        return 2
    frozen = json.loads(path.read_text())
    drift: list[str] = []

    # The manifest is tamper-evident against itself first: a hand-edited block
    # with a stale digest would otherwise slip past the recompute below, which
    # only compares the *current code* against the stored digest.
    for block, digest_key in (
        ("arms", "arms_digest"),
        ("policy_library", "policy_library_digest"),
    ):
        stored = frozen.get(digest_key)
        if not stored:
            continue
        recomputed = hashlib.sha256(
            json.dumps(frozen.get(block) or {}, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        if recomputed != stored:
            drift.append(
                f"frozen manifest {block!r} block does not match its own "
                f"{digest_key}: {stored} -> {recomputed} (the file was edited)"
            )
    for n, v in frozen.get("manifests", {}).items():
        cur = _sha256(root / "experiments/manifests" / n)
        if cur != v.get("sha256"):
            drift.append(f"manifest {n}: {v.get('sha256')} -> {cur}")
    for n, want in frozen.get("configs", {}).items():
        cur = _sha256(root / n)
        if cur != want:
            drift.append(f"config {n}: {want} -> {cur}")

    policy_path = root / "configs/frozen_policy.json"
    if frozen.get("policy_sha256"):
        cur = _sha256(policy_path)
        if cur != frozen["policy_sha256"]:
            drift.append(f"frozen policy: {frozen['policy_sha256']} -> {cur}")
    elif policy_path.exists():
        drift.append("frozen policy appeared after the freeze: re-freeze deliberately")

    from .harness import ARMS
    from .scheduler import POLICY_LIBRARY

    arms_now = {
        name: {
            "model": spec.model,
            "system_prompt": spec.system_prompt,
            "instrumentation": spec.instrumentation,
            "autoregressive": spec.autoregressive,
            "sampler_env": dict(spec.sampler_env),
        }
        for name, spec in sorted(ARMS.items())
    }
    arms_digest = hashlib.sha256(
        json.dumps(arms_now, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if frozen.get("arms_digest") and arms_digest != frozen["arms_digest"]:
        for name in sorted(set(arms_now) | set(frozen.get("arms", {}))):
            was = (frozen.get("arms") or {}).get(name)
            now = arms_now.get(name)
            if was != now:
                drift.append(f"arm {name}: {was} -> {now}")

    lib_now = {name: pol.fingerprint() for name, pol in sorted(POLICY_LIBRARY.items())}
    lib_digest = hashlib.sha256(
        json.dumps(lib_now, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if frozen.get("policy_library_digest") and lib_digest != frozen["policy_library_digest"]:
        for name in sorted(set(lib_now) | set(frozen.get("policy_library", {}))):
            was = (frozen.get("policy_library") or {}).get(name)
            now = lib_now.get(name)
            if was != now:
                drift.append(f"policy {name}: {was} -> {now}")
    if drift:
        print("CONFIGURATION DRIFT", file=sys.stderr)
        for d in drift:
            print("  " + d, file=sys.stderr)
        return 1
    print("frozen manifest and configuration verified")
    return 0


# --------------------------------------------------------------------------- #
# traces
# --------------------------------------------------------------------------- #


def cmd_trace(args: argparse.Namespace) -> int:
    from .harness import TraceRunConfig, run_trace_arm

    root = repo_root()
    cfg = TraceRunConfig(
        root=root,
        manifest=Path(args.manifest),
        arm=args.arm,
        seeds=tuple(int(s) for s in args.seed),
        model=args.model,
        out_dir=Path(args.out) if args.out else root / "experiments/raw" / args.arm,
        instrumentation=args.instrumentation,
        policy=args.policy,
        limit=args.limit,
        tool_sleep_ms=args.tool_sleep_ms,
        resume=args.resume,
        server_url=args.server_url,
        trace_sink_endpoint=args.sink,
    )
    report = run_trace_arm(cfg)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report.get("failures", 0) == 0 else 1


def cmd_replay(args: argparse.Namespace) -> int:
    from .harness import run_replay

    root = repo_root()
    report = run_replay(
        root=root,
        shards=[Path(s) for s in args.shard] or default_replay_shards(root),
        policies=tuple(args.policy),
        latencies=tuple(float(x) for x in args.latency_ms),
        out_dir=Path(args.out) if args.out else root / "experiments/derived",
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


def cmd_overhead(args: argparse.Namespace) -> int:
    from .harness import run_overhead

    root = repo_root()
    report = run_overhead(
        root=root,
        repeats=args.repeats,
        prompts=args.prompts,
        model=args.model,
        level=args.level,
        serve_script=args.serve_script,
        manage_server=not args.no_manage_server,
        server_url=args.server_url,
        sink_endpoint=args.sink,
    )
    slim = {k: v for k, v in report.items() if k != "latencies_ms"}
    print(json.dumps(slim, indent=2, sort_keys=True, default=str))
    return 0 if report.get("ok") else 1


def cmd_pilot(args: argparse.Namespace) -> int:
    from .pilot import run_pilot

    root = repo_root()
    report = run_pilot(root=root, attempt=args.attempt, shards=[Path(s) for s in args.shard])
    print(report["summary"])
    return 0 if report["decision"] == "continue" else 1


def cmd_analyze(args: argparse.Namespace) -> int:
    from .aggregate import aggregate
    from .harness import consolidate_overhead

    root = repo_root()
    n = consolidate_overhead(root)
    if n:
        print(f"consolidated {n} overhead rungs -> experiments/derived/overhead_microbench.jsonl")
    out = aggregate(
        root / "experiments/derived",
        root / "reports/results.json",
        root / "reports/results.md",
    )
    print(json.dumps({k: v for k, v in out.items() if k != "tables"}, indent=2, default=str))
    return 0


def cmd_figures(args: argparse.Namespace) -> int:
    from .figures import render_all

    root = repo_root()
    paths = render_all(root / "experiments/derived", root / "experiments/figures")
    for p in paths:
        print(p)
    return 0


def cmd_tools_validate(args: argparse.Namespace) -> int:
    """Validate the real frozen tool families end to end.

    Checks, in order: the corpus snapshot's content digest still matches; the
    registry builds and its effect annotations are what the config declares;
    every eligible tool is deterministic (two invocations agree byte-for-byte);
    the ineligible tools are refused by the speculation predicate; and
    `sqlite_query` rejects every non-read statement.
    """
    import threading

    from .canonicalize import canonicalize
    from .snapshots import load_snapshot, verify_snapshot
    from .tools import LOCAL_TOOL_DECLS, SqlRejected, local_impls, local_registry

    root = repo_root()
    corpus = Path(args.corpus or root / "experiments/vendor/corpus")
    report: dict[str, Any] = {
        "schema_version": "crystalcall-tools-validate/1",
        "corpus": str(corpus),
        "checks": {},
        "tools": {},
        "ok": True,
    }

    def fail(key: str, detail: Any) -> None:
        report["checks"][key] = {"ok": False, "detail": detail}
        report["ok"] = False

    def ok(key: str, detail: Any = None) -> None:
        report["checks"][key] = {"ok": True, "detail": detail}

    if not corpus.exists():
        fail("corpus_present", f"missing {corpus}; run scripts/build_snapshots.py")
        print(json.dumps(report, indent=2, default=str))
        return 1

    snapshot = load_snapshot(corpus)
    report["snapshot_id"] = snapshot.snapshot_id
    report["snapshot_digest"] = snapshot.digest
    if verify_snapshot(snapshot):
        ok("snapshot_content_digest", snapshot.snapshot_id)
    else:
        fail("snapshot_content_digest", "recomputed digest does not match SNAPSHOT.json")

    registry = local_registry(snapshot)
    impls = local_impls(snapshot)
    report["registry_fingerprint"] = registry.fingerprint()
    declared = {d.name: d.effect for d in LOCAL_TOOL_DECLS}
    mismatch = {
        n: (declared[n], registry.resolve(n).effect.value)
        for n in declared
        if registry.get(n) is None or registry.resolve(n).effect.value != declared[n]
    }
    (ok if not mismatch else fail)("effect_annotations", mismatch or declared)

    # Pick a real module deterministically from the frozen repo sub-snapshot
    # rather than hardcoding a path that content drift would invalidate.
    from .snapshots import iter_corpus_files, sub_snapshot

    repo_snap = sub_snapshot(snapshot, "repo") if snapshot.kind == "corpus" else snapshot
    py_files = sorted(
        p.relative_to(repo_snap.root).as_posix()
        for p in iter_corpus_files(repo_snap.root)
        if p.suffix == ".py"
    )
    report["probe_module"] = py_files[0] if py_files else None
    probes: dict[str, dict[str, Any]] = {
        "search_repo": {"query": "def ", "path": ".", "max_results": 5},
        "analyze_module": {"path": py_files[0]} if py_files else {},
        "sqlite_query": {
            "database": str(snapshot.meta.get("database", "corpus.db")),
            "sql": "SELECT path, lines FROM doc_meta ORDER BY path",
            "max_rows": 3,
        },
        "embed_search": {"query": "retry timeout", "top_k": 3},
    }
    cancel = threading.Event()
    for name in registry.names():
        spec = registry.resolve(name)
        entry: dict[str, Any] = {
            "effect": spec.effect.value,
            "resource": spec.resource.value,
            "expected_duration_ms": spec.expected_duration_ms,
            "speculation_eligible": spec.speculation_eligible,
        }
        if not spec.speculation_eligible:
            entry["status"] = "ineligible_not_invoked"
            report["tools"][name] = entry
            continue
        probe = probes.get(name)
        if probe is None:
            entry["status"] = "no_probe"
            report["tools"][name] = entry
            continue
        try:
            call = canonicalize(registry, name, probe)
            entry["snapshot_id"] = call.snapshot_id
            t0 = time.monotonic_ns()
            a = impls[name](call.args_dict, cancel)
            entry["wall_ms"] = round((time.monotonic_ns() - t0) / 1e6, 3)
            b = impls[name](call.args_dict, cancel)
            entry["deterministic"] = json.dumps(a, sort_keys=True, default=str) == json.dumps(
                b, sort_keys=True, default=str
            )
            entry["status"] = "ok" if entry["deterministic"] else "nondeterministic"
            if not entry["deterministic"]:
                report["ok"] = False
            entry["result_keys"] = sorted(a) if isinstance(a, dict) else type(a).__name__
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            entry["status"] = f"error: {type(exc).__name__}: {exc}"
            report["ok"] = False
        report["tools"][name] = entry

    ineligible = [n for n in registry.names() if not registry.is_speculation_eligible(n)]
    if set(ineligible) >= {"write_note", "read_live_clock"}:
        ok("ineligible_tools_present", ineligible)
    else:
        fail("ineligible_tools_present", ineligible)

    rejected: dict[str, str] = {}
    accepted: list[str] = []
    for sql in (
        "INSERT INTO docs VALUES (1)",
        "UPDATE docs SET body='x'",
        "DELETE FROM docs",
        "DROP TABLE docs",
        "ATTACH DATABASE 'x.db' AS x",
        "PRAGMA journal_mode=WAL",
        "SELECT 1; DROP TABLE docs",
        "CREATE TABLE t(x)",
        "WITH t AS (SELECT 1) INSERT INTO docs SELECT * FROM t",
    ):
        try:
            from .tools import validate_read_only_sql

            validate_read_only_sql(sql)
            accepted.append(sql)
        except SqlRejected as exc:
            rejected[sql] = str(exc)
        except Exception as exc:  # noqa: BLE001
            rejected[sql] = f"{type(exc).__name__}: {exc}"
    if accepted:
        fail("sql_write_rejection", {"accepted": accepted})
    else:
        ok("sql_write_rejection", {"rejected": len(rejected)})

    _write_json(root / "reports/tools_validate.json", report)
    print(json.dumps(report, indent=2, default=str))
    return 0 if report["ok"] else 1


def cmd_verify_traces(args: argparse.Namespace) -> int:
    from .trace import TraceReader, shard_manifest

    root = repo_root()
    raw = root / "experiments/raw"
    dirs = sorted({p.parent for p in raw.rglob("*.jsonl")})
    if not dirs:
        print("no trace shards found", file=sys.stderr)
        return 2
    bad = 0
    for d in dirs:
        try:
            man = shard_manifest(d)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {d}: {exc}", file=sys.stderr)
            bad += 1
            continue
        total = sum(s["records"] for s in man["shards"].values())
        print(f"OK   {d.relative_to(root)}  shards={len(man['shards'])} records={total}")
        _write_json(d / "MANIFEST.sha256.json", man)
    return 1 if bad else 0


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="crystalcall", description=__doc__)
    p.add_argument("--version", action="version", version=f"crystalcall {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="environment, pins, hashes, telemetry")
    d.add_argument("--out")
    d.add_argument("--image", default="vllm/vllm-openai:gemma")
    d.add_argument("--probe-image", action="store_true", help="run the image to read vLLM's version")
    d.add_argument("--no-omp", action="store_true", help="skip the OMP-Lite RPC probe")
    d.set_defaults(func=cmd_doctor)

    s = sub.add_parser("selftest", help="run the safety core test suite")
    s.add_argument("targets", nargs="*")
    s.set_defaults(func=cmd_selftest)

    sv = sub.add_parser("serve", help="start a model server")
    sv.add_argument("--model", choices=("diffusion", "gemma4"), default="diffusion")
    sv.add_argument("--trace-sink", help="host:port for the revision side channel")
    sv.add_argument("--trace-level", choices=("full", "nopayload"))
    sv.add_argument("extra", nargs=argparse.REMAINDER)
    sv.set_defaults(func=cmd_serve)

    fm = sub.add_parser("freeze-manifest", help="freeze manifests, configs and thresholds")
    fm.set_defaults(func=cmd_freeze_manifest)

    vf = sub.add_parser("verify-freeze", help="fail loudly on manifest/config drift")
    vf.set_defaults(func=cmd_verify_freeze)

    t = sub.add_parser("trace", help="generate model trajectories for a manifest")
    t.add_argument("--manifest", required=True)
    t.add_argument("--arm", default="diffusion_standard")
    t.add_argument("--seed", nargs="+", default=["0"])
    t.add_argument("--model", default=None, help="override the arm registry model")
    t.add_argument("--out")
    t.add_argument(
        "--instrumentation", choices=("off", "trace", "parse", "full"), default=None
    )
    t.add_argument("--policy", default="full_gate")
    t.add_argument("--limit", type=int)
    t.add_argument("--tool-sleep-ms", type=float, default=0.0)
    t.add_argument("--resume", action="store_true")
    t.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    t.add_argument("--sink", help="host:port for the revision side channel")
    t.set_defaults(func=cmd_trace)

    r = sub.add_parser("run", help="alias of trace with an explicit arm")
    r.add_argument("--manifest", required=True)
    r.add_argument("--arm", required=True)
    r.add_argument("--seed", nargs="+", default=["0"])
    r.add_argument("--model", default=None, help="override the arm registry model")
    r.add_argument("--out")
    r.add_argument(
        "--instrumentation", choices=("off", "trace", "parse", "full"), default=None
    )
    r.add_argument("--policy", default="full_gate")
    r.add_argument("--limit", type=int)
    r.add_argument("--tool-sleep-ms", type=float, default=0.0)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    r.add_argument("--sink", help="host:port for the revision side channel")
    r.set_defaults(func=cmd_trace)

    rp = sub.add_parser("replay", help="offline policy and latency replay")
    rp.add_argument("--shard", nargs="*", default=[])
    rp.add_argument("--policy", nargs="+", default=["all"])
    rp.add_argument("--latency-ms", nargs="+", default=["100", "500", "2000", "5000"])
    rp.add_argument("--out")
    rp.set_defaults(func=cmd_replay)

    ov = sub.add_parser("overhead", help="one rung of the instrumentation overhead ladder")
    ov.add_argument("--level", choices=("off", "nopayload", "full", "parse"), default="off")
    ov.add_argument("--repeats", type=int, default=30)
    ov.add_argument("--prompts", type=int, default=30)
    ov.add_argument("--model", default="diffusiongemma-26b-a4b-it")
    ov.add_argument("--sink", help="host:port for the revision side channel")
    ov.add_argument("--server-url")
    ov.add_argument("--serve-script", default="scripts/serve_diffusiongemma.sh")
    ov.add_argument(
        "--no-manage-server",
        action="store_true",
        help="assume a server is already running with the matching trace level",
    )
    ov.set_defaults(func=cmd_overhead)

    pl = sub.add_parser("pilot", help="Phase 1 crystallization pilot and go/no-go gate")
    pl.add_argument("--attempt", type=int, default=1)
    pl.add_argument("--shard", nargs="*", default=[])
    pl.set_defaults(func=cmd_pilot)

    an = sub.add_parser("analyze", help="aggregate derived data into results.json/md")
    an.set_defaults(func=cmd_analyze)

    fg = sub.add_parser("figures", help="render every figure")
    fg.set_defaults(func=cmd_figures)

    tv = sub.add_parser("tools", help="tool registry utilities")
    tvs = tv.add_subparsers(dest="tools_command", required=True)
    tvv = tvs.add_parser("validate", help="validate the tool registry and snapshots")
    tvv.add_argument("--corpus")
    tvv.set_defaults(func=cmd_tools_validate)

    vt = sub.add_parser("verify-traces", help="verify append-only trace hash chains")
    vt.set_defaults(func=cmd_verify_traces)

    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except ModuleNotFoundError as exc:
        print(f"component not implemented yet: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
