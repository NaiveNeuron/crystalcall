# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "pyyaml>=6.0", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""Drive the frozen ToolSandbox scenarios through OMP-Lite.

64 scenarios x 3 seeds = 192 episodes per arm, one fresh OMP session per episode,
multi-turn, with the world state living in this process and every tool call
arriving over the authoritative host-tool bridge.

The losslessness check is ONE generation, replayed.  Each episode samples the
model exactly once (with futures on for ``toolsandbox_crystalcall``, off for
``toolsandbox_baseline``); the recorded authoritative canonical call sequence is
then re-executed in order, under ordinary scheduling with no future table, from
a ``WorldState.clone()`` of the episode's initial world.  The generation and the
replay are compared on the per-call result digests, the final
``WorldState.digest()`` and the milestone evaluation.

This is deliberately NOT two generations.  DiffusionGemma's sampler renoises the
canvas with Gumbel noise and ignores the request temperature, so two sequential
requests are not identical even at a pinned server seed; comparing them would
confound sampling noise with scheduling.  Holding the
generation fixed means a mismatch can only come from scheduling.  Any mismatch
is a semantic divergence: it is written into the derived rows, counted in the
summary, and makes the process exit non-zero.  It is never dropped.

With ``--stub`` the model is replaced by a fixed in-process OpenAI-compatible
server (the technique of ``scripts/smoke_omp_ptc.py``), so the whole arm runs
with no GPU and no model server and is fully deterministic.  The code path is
identical: stub mode uses the same one-generation-plus-replay check.

    # GPU-free verification
    uv run scripts/run_toolsandbox.py --stub --limit 6 --seed 0

    # the real arm (needs a served model)
    uv run scripts/run_toolsandbox.py --arm toolsandbox_crystalcall --seed 0

Exit codes: 0 = every episode ran and semantic_divergence == 0; 1 = ran but at
least one episode errored or diverged; 2 = usage/config error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crystalcall.effects import ResourceClass  # noqa: E402
from crystalcall.futures import FutureTable  # noqa: E402
from crystalcall.omp_bridge import OmpLiteProfile, OmpRpcClient  # noqa: E402
from crystalcall.toolsandbox import (  # noqa: E402
    SCHEMA_VERSION,
    Scenario,
    StubModelServer,
    episode_context,
    load_config,
    load_scenarios,
    run_episode,
    scenario_effects,
)
from crystalcall.trace import TraceReader, TraceWriter  # noqa: E402

#: arm -> whether the single generation runs with the speculative launch gate on.
ARMS: dict[str, bool] = {
    "toolsandbox_baseline": False,
    "toolsandbox_crystalcall": True,
}

DERIVED_EPISODES = "toolsandbox_episodes.jsonl"
DERIVED_TURNS = "toolsandbox_turns.jsonl"


# --------------------------------------------------------------------------- #
# derived output
# --------------------------------------------------------------------------- #


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _merge_jsonl(path: Path, rows: list[dict[str, Any]], keys: tuple[str, ...]) -> None:
    """Idempotent per-key merge, written back in sorted key order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    merged: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in _read_jsonl(path) + rows:
        merged[tuple(row.get(k) for k in keys)] = row
    with path.open("w", encoding="utf-8") as fh:
        for key in sorted(merged, key=lambda k: tuple(str(x) for x in k)):
            fh.write(json.dumps(merged[key], sort_keys=True, ensure_ascii=False) + "\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_row(
    sc: Scenario,
    *,
    arm: str,
    seed: int,
    episode: dict[str, Any],
    stub: bool,
) -> dict[str, Any]:
    milestones = episode["milestones"]
    replay = episode["replay"]
    divergence = episode["divergence"]
    exclusion = ""
    if episode.get("error"):
        exclusion = f"episode_error:{episode['error']}"
    elif milestones["indeterminate"]:
        exclusion = f"indeterminate_milestones:{milestones['indeterminate']}"
    return {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": sc.scenario_id,
        "family": sc.family,
        "seed": seed,
        "arm": arm,
        "speculate": bool(ARMS[arm]),
        "stub": bool(stub),
        "milestones_total": milestones["total"],
        "milestones_hit": milestones["hit"],
        "milestones_indeterminate": milestones["indeterminate"],
        "milestone_order_enforced": milestones["order_enforced"],
        "minefields_total": milestones["minefields_total"],
        "minefields_hit": milestones["minefields_hit"],
        "success": milestones["success"],
        "turns": len(episode["turns"]),
        "wall_ms": episode["wall_ms"],
        "world_state_digest_final": episode["world_state_digest_final"],
        "world_state_digest_replay": replay["world_state_digest_final"],
        "world_state_digest_initial": episode["world_state_digest_initial"],
        "replay_calls": replay["calls"],
        "speculative_launches": episode["speculative_launches"],
        "speculative_claimed": episode["speculative_claimed"],
        "speculative_wasted": episode["speculative_wasted"],
        "authoritative_calls": episode["authoritative_calls"],
        "claimed_calls": episode["claimed_calls"],
        "eligible_tools": episode["eligible_tools"],
        "effects": episode["effects"],
        "semantic_divergence": int(divergence["divergent"]),
        "divergent_fields": divergence["fields"],
        "divergence_method": divergence["method"],
        "exclusion": exclusion,
        "notes": (
            "one generation per episode; the recorded canonical call sequence is replayed "
            "under ordinary scheduling against a clone of the initial world"
        ),
    }


def _turn_rows(
    sc: Scenario, *, arm: str, seed: int, episode: dict[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for turn in episode["turns"]:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "scenario_id": sc.scenario_id,
                "family": sc.family,
                "seed": seed,
                "arm": arm,
                "turn_index": turn["turn_index"],
                "user": turn["user"],
                "assistant": turn["assistant"],
                "calls": len(turn["calls"]),
                "claimed_calls": sum(1 for c in turn["calls"] if c["claimed_speculative"]),
                "write_calls": sum(1 for c in turn["calls"] if c["effect"] == "Write"),
                "live_read_calls": sum(1 for c in turn["calls"] if c["effect"] == "LiveRead"),
                "snapshot_read_calls": sum(
                    1 for c in turn["calls"] if c["effect"] == "SnapshotRead"
                ),
                "pure_calls": sum(1 for c in turn["calls"] if c["effect"] == "Pure"),
                "failed_calls": sum(
                    1 for c in turn["calls"] if c["error"] or c["canonicalization_error"]
                ),
                "tools": [c["tool"] for c in turn["calls"]],
                "snapshot_ids": [c["snapshot_id"] for c in turn["calls"]],
                "world_digest_after": turn["world_digest_after"],
                "wall_ms": turn["wall_ms"],
                "error": turn["error"] or "",
            }
        )
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _shard_path(out_dir: Path, arm: str, seed: int) -> Path:
    base = out_dir / f"{arm}-seed{seed}.jsonl"
    if not base.exists():
        return base
    index = 1
    while (candidate := out_dir / f"{arm}-seed{seed}.part{index}.jsonl").exists():
        index += 1
    return candidate


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=ROOT / "experiments/manifests/toolsandbox.jsonl")
    ap.add_argument("--arm", choices=sorted(ARMS), default="toolsandbox_crystalcall")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="0 = every scenario")
    ap.add_argument("--resume", action="store_true", help="skip episodes already in the derived rows")
    ap.add_argument("--stub", action="store_true", help="drive an in-process stub model (no GPU)")
    ap.add_argument("--server-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument(
        "--stub-port",
        type=int,
        default=8000,
        help="port the --stub server binds; must match the vllm provider base URL",
    )
    ap.add_argument("--model", default="vllm/diffusiongemma-26b-a4b-it")
    ap.add_argument("--out", type=Path, default=ROOT / "experiments/raw/toolsandbox")
    ap.add_argument("--derived", type=Path, default=ROOT / "experiments/derived")
    ap.add_argument("--turn-timeout", type=float, default=300.0)
    args = ap.parse_args()

    try:
        cfg = load_config()
        scenarios = load_scenarios(args.manifest)
    except Exception as exc:  # noqa: BLE001 - a config error is a usage error
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    if args.limit:
        scenarios = scenarios[: args.limit]

    done: set[str] = set()
    if args.resume:
        for row in _read_jsonl(args.derived / DERIVED_EPISODES):
            if row.get("arm") == args.arm and row.get("seed") == args.seed:
                done.add(str(row.get("scenario_id")))
        scenarios = [s for s in scenarios if s.scenario_id not in done]
    print(f"arm        {args.arm} (speculate={ARMS[args.arm]})")
    print(f"seed       {args.seed}")
    print(f"scenarios  {len(scenarios)}" + (f" ({len(done)} resumed)" if done else ""))

    stub: StubModelServer | None = None
    server_url = args.server_url
    if args.stub:
        # OMP resolves the `vllm` provider base URL from the installed profile's
        # models.yml, so the stub has to answer on exactly that host and port.
        # `--stub-port` exists for the case where the profile is repointed.
        if not StubModelServer.port_free(args.stub_port):
            print(
                f"error: 127.0.0.1:{args.stub_port} is busy. Either a real model server is "
                "already listening (in which case drop --stub) or another run is in flight.",
                file=sys.stderr,
            )
            return 2
        stub = StubModelServer(cfg, port=args.stub_port)
        server_url = stub.start()
        model = f"vllm/{stub.model_id}"
        print(f"stub       {server_url}")
    else:
        import httpx

        model = args.model
        try:
            ok = httpx.get(server_url.rstrip("/") + "/models", timeout=5.0).status_code == 200
        except httpx.HTTPError:
            ok = False
        if not ok:
            print(f"error: no model server at {server_url}; start one or pass --stub", file=sys.stderr)
            return 2

    workspace = ROOT / "experiments/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    profile = OmpLiteProfile(
        profile="crystalcall",
        config=ROOT / "configs/omp-lite.yml",
        system_prompt=ROOT / "configs/system_prompt.md",
        model=model,
        thinking="off",
        tools=("eval",),
        cwd=workspace,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    shard = _shard_path(args.out, args.arm, args.seed)
    writer = TraceWriter(
        shard,
        meta={
            "schema_version": SCHEMA_VERSION,
            "arm": args.arm,
            "speculate": ARMS[args.arm],
            "divergence_method": "one_generation_replayed_against_a_cloned_world",
            "seed": args.seed,
            "model": model,
            "stub": bool(args.stub),
            "manifest": str(args.manifest),
            "manifest_sha256": _sha256(args.manifest),
            "config": "configs/toolsandbox.yml",
            "config_sha256": _sha256(ROOT / "configs/toolsandbox.yml"),
            "checkout_commit": cfg["checkout"]["commit"],
            "speculation_policy": cfg["speculation"]["policy"],
            "profile_fingerprint": profile.fingerprint(),
            "scenarios": len(scenarios),
        },
    )

    limit = int(cfg["speculation"]["concurrency_per_resource"])
    futures = FutureTable(concurrency={rc: limit for rc in ResourceClass})
    client: OmpRpcClient | None = None
    episode_rows: list[dict[str, Any]] = []
    turn_rows: list[dict[str, Any]] = []
    divergences = 0
    failures = 0
    started = time.monotonic()

    try:
        # NOT in `args.out`: `trace.shard_manifest` verifies every `*.jsonl` it
        # finds there, and the RPC log is not a hash-chained shard.
        client = OmpRpcClient(
            profile, log_path=args.out / "logs" / f"{args.arm}-seed{args.seed}.rpc.jsonl"
        )
        client.start(timeout=240)

        def factory() -> OmpRpcClient:
            assert client is not None
            return client

        for index, sc in enumerate(scenarios):
            if stub is not None:
                _, registry, _ = episode_context(sc)
                stub.load_for(sc, registry)
            speculate = ARMS[args.arm]
            episode = run_episode(
                sc,
                seed=args.seed,
                arm=args.arm,
                client_factory=factory,
                futures=futures if speculate else None,
                speculate=speculate,
                turn_timeout=args.turn_timeout,
                trace=writer.append,
            )
            divergence = episode["divergence"]
            row = _episode_row(
                sc,
                arm=args.arm,
                seed=args.seed,
                episode=episode,
                stub=bool(args.stub),
            )
            episode_rows.append(row)
            turn_rows.extend(_turn_rows(sc, arm=args.arm, seed=args.seed, episode=episode))
            divergences += int(divergence["divergent"])
            if row["exclusion"].startswith("episode_error"):
                failures += 1
            effects = scenario_effects(sc)
            eligible = sum(1 for e in effects.values() if e.speculation_eligible)
            flag = "DIVERGENT" if divergence["divergent"] else "ok"
            print(
                f"[{index + 1:>3}/{len(scenarios)}] {sc.scenario_id[:58]:58s} "
                f"{sc.family[:22]:22s} turns={len(episode['turns'])} "
                f"calls={episode['authoritative_calls']:>2} "
                f"replay={episode['replay']['calls']:>2} "
                f"eligible={eligible}/{len(effects)} "
                f"spec={episode['speculative_launches']}/{episode['speculative_claimed']} "
                f"ms={episode['milestones']['hit']}/{episode['milestones']['total']} {flag}",
                flush=True,
            )
    finally:
        if client is not None:
            client.close()
        futures.close()
        meta = writer.close()
        if stub is not None:
            stub.stop()

    _merge_jsonl(args.derived / DERIVED_EPISODES, episode_rows, ("arm", "seed", "scenario_id"))
    _merge_jsonl(
        args.derived / DERIVED_TURNS, turn_rows, ("arm", "seed", "scenario_id", "turn_index")
    )
    verified = TraceReader(shard).verify()

    summary = {
        "schema_version": SCHEMA_VERSION,
        "arm": args.arm,
        "seed": args.seed,
        "stub": bool(args.stub),
        "episodes": len(episode_rows),
        "semantic_divergence": divergences,
        "divergence_method": "one_generation_replayed_against_a_cloned_world",
        "failures": failures,
        "shard": str(shard),
        "shard_records": verified["records"],
        "shard_sha256": verified["sha256"],
        "derived": [str(args.derived / DERIVED_EPISODES), str(args.derived / DERIVED_TURNS)],
        "wall_s": round(time.monotonic() - started, 3),
    }
    reports = ROOT / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"toolsandbox-{args.arm}-seed{args.seed}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print()
    print(f"shard      {shard} ({verified['records']} records, chain verified)")
    print(f"derived    {args.derived / DERIVED_EPISODES}, {args.derived / DERIVED_TURNS}")
    print(f"episodes   {len(episode_rows)}  failures {failures}")
    print(f"divergence {divergences}")
    print(f"meta       {meta['sha256'][:16]}")
    if divergences or failures:
        print("\nFAIL")
        return 1
    print(
        "\nPASS: zero semantic divergence between the generation and its "
        "ordinary-scheduling replay"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
