"""The ToolSandbox validation arm.

Every test here is GPU-free and model-free: where a model is needed, OMP talks
to :class:`crystalcall.toolsandbox.StubModelServer`, the same in-process
OpenAI-compatible stub the ``--stub`` runner uses, so the authoritative path
(native tool call -> OMP Python kernel -> ``tool.<name>(...)`` loopback ->
``host_tool_call`` -> ``HostToolRouter``) is exercised for real.

What is asserted, in order:

* the frozen manifest still resolves to 64 scenarios, 16 per family;
* the declared read/write footprints still agree with the pinned checkout;
* effect assignment is total, and every speculation-eligible tool reads only
  namespaces that no write in its own scenario can touch;
* a write moves ``WorldState.digest()`` and moves the snapshot id of every view
  it touches, while leaving disjoint views' ids alone;
* a future launched before a write cannot be claimed after it;
* writes and live reads are refused at the launch site;
* a whole stub episode is identical under ordinary and speculative scheduling
  for three scenarios spanning three families;
* milestone evaluation is a pure function of (scenario, world, transcript).
"""

from __future__ import annotations

import collections
import json
import subprocess
import threading
from pathlib import Path
from typing import Any

import pytest

from crystalcall.canonicalize import canonicalize
from crystalcall.effects import EffectKind, ResourceClass
from crystalcall.futures import FutureKey, FutureTable, QuarantineViolation
from crystalcall.omp_bridge import (
    HostToolRouter,
    OmpLiteProfile,
    OmpRpcClient,
    pinned_omp_version,
    resolve_omp_binary,
)
from crystalcall.toolsandbox import (
    NAMESPACES,
    Scenario,
    StubModelServer,
    WorldFrozen,
    WorldState,
    compare_replay,
    derive_tool_effects,
    effect_for,
    episode_context,
    evaluate_milestones,
    impls_for_scenario,
    load_config,
    load_scenarios,
    registry_for_scenario,
    replay_episode,
    run_episode,
    scenario_effects,
    snapshot_resolver,
    verify_against_checkout,
)
from crystalcall.trace import TraceReader, TraceWriter

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "experiments/manifests/toolsandbox.jsonl"
FAMILIES = (
    "canonicalization",
    "insufficient_information",
    "multi_tool_workflows",
    "state_dependencies",
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    return load_config()


@pytest.fixture(scope="module")
def scenarios() -> list[Scenario]:
    return load_scenarios(MANIFEST)


@pytest.fixture(scope="module")
def by_id(scenarios: list[Scenario]) -> dict[str, Scenario]:
    return {s.scenario_id: s for s in scenarios}


# --------------------------------------------------------------------------- #
# 1. the frozen manifest
# --------------------------------------------------------------------------- #


def test_manifest_resolves_to_the_frozen_64(scenarios: list[Scenario]) -> None:
    assert len(scenarios) == 64
    counts = collections.Counter(s.family for s in scenarios)
    assert dict(counts) == {family: 16 for family in FAMILIES}
    assert len({s.scenario_id for s in scenarios}) == 64
    # Loading is order-preserving and repeatable, so a run is reproducible from
    # the manifest alone.
    again = load_scenarios(MANIFEST)
    assert [s.scenario_id for s in again] == [s.scenario_id for s in scenarios]


def test_every_scenario_has_turns_tools_and_a_declared_world(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        assert sc.user_turns, sc.scenario_id
        assert all(t.strip() for t in sc.user_turns), sc.scenario_id
        assert sc.tools, sc.scenario_id
        assert sc.initial_world["base_variant"] in load_config()["world"]["base_variants"]
        assert len(sc.milestones) == sc.meta["n_milestones"]


def test_scripted_follow_up_only_for_declared_multi_user_turn(scenarios: list[Scenario]) -> None:
    cfg = load_config()
    multi = tuple(cfg["user_turns"]["multi_turn_categories"])
    for sc in scenarios:
        declared_multi = any(c in sc.categories for c in multi)
        assert bool(sc.meta["scripted_follow_up"]) is declared_multi, sc.scenario_id
        assert len(sc.user_turns) == (2 if declared_multi else 1), sc.scenario_id


# --------------------------------------------------------------------------- #
# 2. the declaration still matches its source
# --------------------------------------------------------------------------- #


def test_declared_footprints_match_the_checkout() -> None:
    assert verify_against_checkout() == []


def test_derivation_covers_every_declared_tool(cfg: dict[str, Any]) -> None:
    derived = derive_tool_effects()
    declared = set(cfg["_tools_by_name"])
    assert declared <= set(derived)
    # No declared tool may reach the SANDBOX conversation log: that is OMP's.
    for name in declared:
        assert "sandbox" not in derived[name]["reads"] + derived[name]["writes"]


# --------------------------------------------------------------------------- #
# 3. effect assignment is total and eligibility is immutable-view-only
# --------------------------------------------------------------------------- #


def test_effect_assignment_is_total(scenarios: list[Scenario]) -> None:
    for sc in scenarios:
        effects = scenario_effects(sc)
        assert set(effects) == set(sc.tool_names), sc.scenario_id
        assert all(isinstance(e, EffectKind) for e in effects.values())
        # The registry agrees with the Scenario's own declarations.
        registry = registry_for_scenario(sc)
        for name, effect in effects.items():
            assert registry.resolve(name).effect is effect


def test_every_eligible_tool_is_immutable_view_only(
    scenarios: list[Scenario], cfg: dict[str, Any]
) -> None:
    for sc in scenarios:
        writes = sc.surface_writes
        for name, effect in scenario_effects(sc).items():
            decl = cfg["_tools_by_name"][name]
            if not effect.speculation_eligible:
                continue
            assert decl["declared_class"] != "write", (sc.scenario_id, name)
            assert not decl["external"], (sc.scenario_id, name)
            assert not (set(decl["reads"]) & writes), (sc.scenario_id, name, decl["reads"])
            if effect is EffectKind.PURE:
                assert not decl["reads"], (sc.scenario_id, name)
            else:
                assert effect is EffectKind.SNAPSHOT_READ
                assert decl["reads"], (sc.scenario_id, name)


def test_external_and_write_tools_are_never_eligible(cfg: dict[str, Any]) -> None:
    every_namespace = set(NAMESPACES)
    for name, decl in sorted(cfg["_tools_by_name"].items()):
        # Even in a scenario that writes nothing, an external read stays live.
        effect = effect_for(name, (), cfg)
        if decl["declared_class"] == "write":
            assert effect is EffectKind.WRITE, name
        elif decl["external"]:
            assert effect is EffectKind.LIVE_READ, name
        # And in a scenario that writes everything, no read is eligible.
        if decl["declared_class"] != "write" and decl["reads"]:
            assert effect_for(name, every_namespace, cfg) is EffectKind.LIVE_READ, name
        # The only tool that stays eligible when every namespace is written is
        # one that reads nothing at all, i.e. a genuinely Pure tool.
        if effect_for(name, every_namespace, cfg).speculation_eligible:
            assert not decl["reads"], name
            assert effect_for(name, every_namespace, cfg) is EffectKind.PURE, name


def test_unknown_tool_is_rejected_rather_than_defaulted() -> None:
    from crystalcall.toolsandbox import ScenarioError

    with pytest.raises(ScenarioError):
        effect_for("definitely_not_a_toolsandbox_tool", ())


def test_eligible_snapshot_read_resolves_over_exactly_its_reads(
    scenarios: list[Scenario], cfg: dict[str, Any]
) -> None:
    for sc in scenarios:
        world, registry, _ = episode_context(sc)
        for name, effect in scenario_effects(sc).items():
            if effect is not EffectKind.SNAPSHOT_READ:
                continue
            reads = tuple(sorted(cfg["_tools_by_name"][name]["reads"]))
            resolver = registry.resolve(name).snapshot_resolver
            assert resolver is not None
            assert resolver({}) == world.snapshot_id(reads)
            assert "+".join(reads) in resolver({})


# --------------------------------------------------------------------------- #
# 4. a write moves the digest and the snapshot ids of the views it touches
# --------------------------------------------------------------------------- #


def _reminder_scenario(by_id: dict[str, Scenario]) -> Scenario:
    return by_id["add_reminder_content_and_week_delta_and_time"]


def test_write_moves_full_digest_and_touched_snapshot_ids(by_id: dict[str, Scenario]) -> None:
    sc = _reminder_scenario(by_id)
    world = WorldState.from_scenario(sc)

    before_digest = world.digest()
    before_reminder = world.snapshot_id(("reminder",))
    before_clock = world.snapshot_id(("clock",))
    before_multi = world.snapshot_id(("clock", "contact", "reminder"))

    world.apply_write(
        "add_reminder", {"content": "Buy chocolate milk", "reminder_timestamp": 1711152000.0}
    )

    assert world.digest() != before_digest
    assert world.snapshot_id(("reminder",)) != before_reminder
    assert world.snapshot_id(("clock", "contact", "reminder")) != before_multi
    # A view the write does not touch keeps its id: the id is the digest of
    # exactly the named sub-view and nothing wider.
    assert world.snapshot_id(("clock",)) == before_clock
    assert world.snapshot_id(("contact",)) == WorldState.from_scenario(sc).snapshot_id(
        ("contact",)
    )
    assert world.versions["reminder"] == 1
    assert world.changed_namespaces() == ("reminder",)


def test_setting_write_moves_only_the_setting_view(by_id: dict[str, Scenario]) -> None:
    sc = by_id["turn_on_wifi_low_battery_mode"]
    world = WorldState.from_scenario(sc)
    before = {ns: world.snapshot_id((ns,)) for ns in NAMESPACES}
    world.apply_write("set_low_battery_mode_status", {"on": False})
    after = {ns: world.snapshot_id((ns,)) for ns in NAMESPACES}
    assert after["setting"] != before["setting"]
    assert {ns: after[ns] for ns in NAMESPACES if ns != "setting"} == {
        ns: before[ns] for ns in NAMESPACES if ns != "setting"
    }


def test_identical_worlds_share_ids_and_differing_worlds_do_not(
    by_id: dict[str, Scenario],
) -> None:
    sc = _reminder_scenario(by_id)
    a, b = WorldState.from_scenario(sc), WorldState.from_scenario(sc)
    assert a.digest() == b.digest()
    assert a.snapshot_id() == b.snapshot_id()
    b.apply_write("add_reminder", {"content": "x", "reminder_timestamp": 1.0})
    assert a.digest() != b.digest()


def test_frozen_world_refuses_writes_and_pins_its_ids(by_id: dict[str, Scenario]) -> None:
    sc = _reminder_scenario(by_id)
    world = WorldState.from_scenario(sc)
    frozen = world.freeze()
    pinned = frozen.snapshot_id(("reminder",))

    with pytest.raises(WorldFrozen):
        frozen.apply_write("add_reminder", {"content": "x", "reminder_timestamp": 1.0})

    world.apply_write("add_reminder", {"content": "y", "reminder_timestamp": 2.0})
    # The frozen copy is detached: a later authoritative write cannot be seen by
    # in-flight speculative work, so a speculative value is a function of
    # exactly the state its snapshot id names.
    assert frozen.snapshot_id(("reminder",)) == pinned
    assert frozen.snapshot_id(("reminder",)) != world.snapshot_id(("reminder",))
    assert len(frozen.rows["reminder"]) == 3
    assert len(world.rows["reminder"]) == 4


def test_writes_are_deterministic_across_worlds(by_id: dict[str, Scenario]) -> None:
    sc = _reminder_scenario(by_id)
    digests = []
    for _ in range(3):
        world = WorldState.from_scenario(sc)
        world.apply_write("add_reminder", {"content": "milk", "reminder_timestamp": 1711152000.0})
        digests.append(world.digest())
    assert len(set(digests)) == 1


# --------------------------------------------------------------------------- #
# 5. the claim gate: stale futures, and no write speculation
# --------------------------------------------------------------------------- #


def _mutable_snapshot_registry(sc: Scenario, world: WorldState):
    """A registry that declares ``search_reminder`` a snapshot read of ``reminder``.

    The per-scenario rule would call this a live read, and rightly: ``reminder``
    is written in this scenario.  Forcing it here is how we exercise the
    *mechanism* the rule is the first line of defence for -- the exact-key claim
    -- against a world version that really does move underneath it.
    """
    from crystalcall.effects import EffectRegistry, ToolSpec

    base = registry_for_scenario(sc, world)
    specs = []
    for spec in base.specs():
        if spec.name == "search_reminder":
            spec = ToolSpec(
                name=spec.name,
                effect=EffectKind.SNAPSHOT_READ,
                resource=spec.resource,
                params=spec.params,
                expected_duration_ms=spec.expected_duration_ms,
                concurrency_limit=spec.concurrency_limit,
                snapshot_resolver=snapshot_resolver(world, ("reminder",)),
                description=spec.description,
            )
        specs.append(spec)
    return EffectRegistry(specs)


def test_future_launched_before_a_write_cannot_be_claimed_after_it(
    by_id: dict[str, Scenario],
) -> None:
    sc = by_id["remove_reminder_with_recency_latest"]
    assert "reminder" in sc.surface_writes
    world = WorldState.from_scenario(sc)
    registry = _mutable_snapshot_registry(sc, world)
    args = {"creation_timestamp_lowerbound": 0.0}

    before = canonicalize(registry, "search_reminder", args)
    assert before.effect is EffectKind.SNAPSHOT_READ
    assert before.snapshot_id is not None

    futures = FutureTable(concurrency={rc: 2 for rc in ResourceClass})
    try:
        frozen = world.freeze()
        frozen_impls = impls_for_scenario(sc, frozen)
        key = FutureKey.of("req", before)
        futures.launch(
            key,
            before,
            registry.resolve("search_reminder").resource,
            lambda cancel: frozen_impls["search_reminder"](args, cancel),
            reason="test",
        )
        futures.drain()

        # Before the write, the exact key claims.
        assert futures.claim(key, timeout=10.0)[0] is True

        # Re-launch, then move the world.
        key2 = FutureKey.of("req2", canonicalize(registry, "search_reminder", args))
        futures.launch(
            key2,
            before,
            registry.resolve("search_reminder").resource,
            lambda cancel: frozen_impls["search_reminder"](args, cancel),
            reason="test",
        )
        futures.drain()
        world.apply_write(
            "add_reminder", {"content": "later", "reminder_timestamp": 1711152000.0}
        )

        after = canonicalize(registry, "search_reminder", args)
        assert after.snapshot_id != before.snapshot_id
        assert after.args_digest == before.args_digest
        assert after.tool == before.tool

        stale = FutureKey.of("req2", after)
        assert stale != key2
        claimed, value = futures.claim(stale, timeout=5.0)
        assert claimed is False
        assert value is None
    finally:
        futures.close()


def test_router_executes_authoritatively_when_the_snapshot_moved(
    by_id: dict[str, Scenario],
) -> None:
    sc = by_id["remove_reminder_with_recency_latest"]
    world = WorldState.from_scenario(sc)
    registry = _mutable_snapshot_registry(sc, world)
    args = {"creation_timestamp_lowerbound": 0.0}
    futures = FutureTable(concurrency={rc: 2 for rc in ResourceClass})
    try:
        frozen = world.freeze()
        frozen_impls = impls_for_scenario(sc, frozen)
        call = canonicalize(registry, "search_reminder", args)
        futures.launch(
            FutureKey.of("req", call),
            call,
            registry.resolve("search_reminder").resource,
            lambda cancel: frozen_impls["search_reminder"](args, cancel),
            reason="test",
        )
        futures.drain()
        world.apply_write("add_reminder", {"content": "z", "reminder_timestamp": 3.0})

        router = HostToolRouter(
            registry,
            impls_for_scenario(sc, world),
            futures=futures,
            request_id_provider=lambda: "req",
            claim_timeout=5.0,
        )
        value, error = router.handle(
            "search_reminder", args, host_call_id="h", tool_call_id="t"
        )
        assert error is None
        record = router.calls()[0]
        assert record.claimed_speculative is False
        # The authoritative result reflects the *current* world, which the stale
        # speculation could not have known about.
        assert len(value) == 4
    finally:
        futures.close()


def test_writes_and_live_reads_are_refused_at_the_launch_site(
    scenarios: list[Scenario],
) -> None:
    checked = {"Write": 0, "LiveRead": 0}
    futures = FutureTable(concurrency={rc: 1 for rc in ResourceClass})
    try:
        for sc in scenarios:
            world, registry, impls = episode_context(sc)
            for name, effect in scenario_effects(sc).items():
                if effect.speculation_eligible:
                    continue
                spec = registry.resolve(name)
                args = _minimal_args(sc, name)
                if args is None:
                    continue
                call = canonicalize(registry, name, args)
                assert call.effect is effect
                assert call.snapshot_id is None
                with pytest.raises(QuarantineViolation):
                    futures.launch(
                        FutureKey.of("req", call),
                        call,
                        spec.resource,
                        lambda cancel: None,
                        reason="must-refuse",
                    )
                checked[effect.value] += 1
    finally:
        futures.close()
    assert checked["Write"] > 0
    assert checked["LiveRead"] > 0


def _minimal_args(sc: Scenario, name: str) -> dict[str, Any] | None:
    fillers = load_config()["stub"]["required_args"].get(name, {})
    registry = registry_for_scenario(sc)
    spec = registry.get(name)
    if spec is None:
        return None
    out: dict[str, Any] = {}
    for param in spec.params:
        if param.name in fillers:
            out[param.name] = fillers[param.name]
        elif param.required:
            return None
    return out


def test_a_frozen_world_cannot_be_used_to_speculate_a_write(
    by_id: dict[str, Scenario],
) -> None:
    sc = _reminder_scenario(by_id)
    world = WorldState.from_scenario(sc)
    frozen_impls = impls_for_scenario(sc, world.freeze())
    with pytest.raises(WorldFrozen):
        frozen_impls["add_reminder"]({"content": "x", "reminder_timestamp": 1.0}, threading.Event())
    # And the live world is untouched by the attempt.
    assert world.changed_namespaces() == ()


# --------------------------------------------------------------------------- #
# 6. milestone evaluation is a pure function
# --------------------------------------------------------------------------- #


def _transcript(sc: Scenario) -> list[dict[str, Any]]:
    return [
        {"kind": "user", "text": sc.user_turns[0]},
        {
            "kind": "call",
            "tool": "get_current_timestamp",
            "args": {},
            "snapshot_id": "tsbx:clock@deadbeefdeadbeef",
            "effect": "SnapshotRead",
            "identity": "get_current_timestamp|x|y",
            "result_digest": "d",
            "error": None,
            "canonicalization_error": None,
            "claimed_speculative": False,
        },
        {"kind": "assistant", "text": "Reminder added."},
    ]


def test_milestone_evaluation_is_pure(scenarios: list[Scenario]) -> None:
    for sc in scenarios[:24]:
        world = WorldState.from_scenario(sc)
        transcript = _transcript(sc)
        world_before = world.digest()
        transcript_snapshot = json.dumps(transcript, sort_keys=True)

        first = evaluate_milestones(sc, world, transcript)
        second = evaluate_milestones(sc, world, transcript)

        assert first == second, sc.scenario_id
        assert first["digest"] == second["digest"]
        # No input was mutated.
        assert world.digest() == world_before
        assert json.dumps(transcript, sort_keys=True) == transcript_snapshot
        # Independent equal inputs give the same answer.
        assert (
            evaluate_milestones(sc, WorldState.from_scenario(sc), _transcript(sc))["digest"]
            == first["digest"]
        )
        assert first["total"] == len(sc.milestones)
        assert 0 <= first["hit"] <= first["total"]


def test_milestone_evaluation_reacts_to_world_and_transcript(
    by_id: dict[str, Scenario],
) -> None:
    sc = _reminder_scenario(by_id)
    empty = evaluate_milestones(sc, WorldState.from_scenario(sc), [])
    world = WorldState.from_scenario(sc)
    world.apply_write(
        "add_reminder", {"content": "Buy chocolate milk", "reminder_timestamp": 1711152000.0}
    )
    with_write = evaluate_milestones(sc, world, _transcript(sc))
    assert with_write["hit"] > empty["hit"]
    assert with_write["digest"] != empty["digest"]


def test_minefield_is_hit_only_by_the_forbidden_call(by_id: dict[str, Scenario]) -> None:
    sc = by_id["find_days_till_holiday_insufficient_information"]
    assert sc.minefields
    clean = evaluate_milestones(sc, WorldState.from_scenario(sc), [])
    assert clean["minefields_hit"] == 0
    assert clean["success"] is True
    forbidden = dict(_transcript(sc)[1])
    forbidden["tool"] = "timestamp_diff"
    tripped = evaluate_milestones(sc, WorldState.from_scenario(sc), [forbidden])
    assert tripped["minefields_hit"] == 1
    assert tripped["success"] is False


# --------------------------------------------------------------------------- #
# 7. end to end: one generation, replayed under ordinary scheduling
# --------------------------------------------------------------------------- #

#: One scenario each from three families, chosen for coverage of the effect
#: mix: a write + eligible snapshot read, an eligible-read-only workflow, and a
#: state-dependent scenario whose reads are all live.
E2E_SCENARIOS = (
    "add_reminder_content_and_week_delta_and_time_multiple_user_turn",  # canonicalization
    "search_message_with_recency_oldest",  # multi_tool_workflows
    "turn_on_wifi_low_battery_mode",  # state_dependencies
)


def _pinned_omp_version() -> str | None:
    """The `omp` build the pinned corpus was produced with (`reports/doctor.json`)."""
    version = pinned_omp_version(ROOT)
    return f"omp/{version}" if version else None


def _resolved_omp_version() -> str | None:
    """`--version` of the binary the harness will actually launch.

    Resolution prefers the checksum-verified build vendored under
    `experiments/vendor/omp/<pinned>/`, so this reports the pinned build when it
    is present and the host build only when it is not.
    """
    try:
        out = subprocess.run(
            [resolve_omp_binary("omp", ROOT), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        # No binary / no exec is a skip, handled by the caller. A coding error
        # must NOT be swallowed here: that would disable the pin guard silently.
        return None
    return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else None


@pytest.fixture(scope="module")
def omp_stub(cfg: dict[str, Any]):
    """A stub model plus one started OMP-Lite session, shared by the e2e tests."""
    port = int(cfg["stub"].get("port", 8000))
    if not StubModelServer.port_free(port):
        pytest.skip(f"127.0.0.1:{port} is busy; a real model server is probably running")
    if not (ROOT / "configs/omp-lite.yml").exists():
        pytest.skip("configs/omp-lite.yml missing")
    # The stateful arm drives the *real* omp binary, so it is only meaningful on
    # the build the corpus was pinned to: the native tool-call path is not
    # comparable across builds. The pinned build is
    # vendored and preferred by `resolve_omp_binary`; if only a drifted host
    # build is available, announce the mismatch rather than asserting through it.
    pinned, resolved = _pinned_omp_version(), _resolved_omp_version()
    if pinned and resolved and pinned != resolved:
        pytest.skip(
            f"omp resolves to {resolved}, corpus pinned to {pinned}: vendor the "
            "pinned build with `scripts/fetch_omp.sh`"
        )
    stub = StubModelServer(cfg, port=port)
    stub.start()
    workspace = ROOT / "experiments/workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    profile = OmpLiteProfile(
        profile="crystalcall",
        config=ROOT / "configs/omp-lite.yml",
        system_prompt=ROOT / "configs/system_prompt.md",
        model=f"vllm/{stub.model_id}",
        thinking="off",
        tools=("eval",),
        cwd=workspace,
    )
    client = OmpRpcClient(profile)
    try:
        client.start(timeout=240)
    except Exception as exc:  # noqa: BLE001 - no omp binary is a skip, not a failure
        stub.stop()
        pytest.skip(f"could not start omp: {exc}")
    yield stub, client
    client.close()
    stub.stop()


def _episode(
    stub: StubModelServer,
    client: OmpRpcClient,
    sc: Scenario,
    *,
    speculate: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """One generation of ``sc`` plus its ordinary-scheduling replay."""
    _, registry, _ = episode_context(sc)
    stub.load_for(sc, registry)
    return run_episode(
        sc,
        seed=seed,
        arm="toolsandbox_crystalcall" if speculate else "toolsandbox_baseline",
        client_factory=lambda: client,
        speculate=speculate,
        turn_timeout=120.0,
    )


@pytest.mark.parametrize("scenario_id", E2E_SCENARIOS)
def test_speculative_generation_matches_its_ordinary_replay(
    omp_stub, by_id: dict[str, Scenario], scenario_id: str
) -> None:
    stub, client = omp_stub
    sc = by_id[scenario_id]
    episode = _episode(stub, client, sc)

    assert episode["error"] is None, episode["error"]
    assert episode["authoritative_calls"] > 0, "the stub program made no tool call"

    divergence = episode["divergence"]
    assert divergence["divergent"] is False, divergence["fields"]
    assert divergence["fields"] == []
    assert divergence["method"] == "one_generation_replayed_against_a_cloned_world"

    # The comparison is not vacuous: the replay really re-executed the sequence
    # and really reached the same world.
    replay = episode["replay"]
    assert replay["calls"] == episode["authoritative_calls"]
    assert replay["world_state_digest_initial"] == episode["world_state_digest_initial"]
    assert replay["world_state_digest_final"] == episode["world_state_digest_final"]
    assert [c["result_digest"] for c in replay["tool_trace"]] == [
        c["result_digest"] for c in episode["tool_trace"]
    ]
    assert replay["milestones"]["digest"] == episode["milestones"]["digest"]
    # The replay claims nothing, by construction.
    assert all(not c["claimed_speculative"] for c in replay["transcript"] if c.get("kind") == "call")
    # And the stub's reply is a function of the tool results, not a constant.
    assert episode["assistant_final"].startswith("observed ")
    assert episode["assistant_final"] != "observed none"


def test_baseline_arm_also_matches_its_replay(omp_stub, by_id: dict[str, Scenario]) -> None:
    stub, client = omp_stub
    episode = _episode(stub, client, by_id[E2E_SCENARIOS[0]], speculate=False)
    assert episode["speculative_launches"] == 0
    assert episode["claimed_calls"] == 0
    assert episode["divergence"]["divergent"] is False


def test_three_families_are_covered(by_id: dict[str, Scenario]) -> None:
    families = {by_id[s].family for s in E2E_SCENARIOS}
    assert len(families) == 3
    assert families <= set(FAMILIES)


def test_speculation_claims_without_changing_the_result(
    omp_stub, by_id: dict[str, Scenario]
) -> None:
    stub, client = omp_stub
    sc = by_id["add_reminder_content_and_week_delta_and_time_multiple_user_turn"]
    episode = _episode(stub, client, sc)
    assert episode["speculative_launches"] > 0
    assert episode["speculative_claimed"] > 0
    assert episode["claimed_calls"] > 0
    # Every claimed value equals the value ordinary scheduling computes.
    assert [c["result_digest"] for c in episode["tool_trace"]] == [
        c["result_digest"] for c in episode["replay"]["tool_trace"]
    ]
    for row in episode["transcript"]:
        if row.get("kind") == "call" and row.get("claimed_speculative"):
            assert row["effect"] in ("Pure", "SnapshotRead")


def test_writes_are_never_claimed_across_every_e2e_scenario(
    omp_stub, by_id: dict[str, Scenario]
) -> None:
    stub, client = omp_stub
    seen_write = False
    for scenario_id in E2E_SCENARIOS:
        episode = _episode(stub, client, by_id[scenario_id])
        for row in episode["transcript"]:
            if row.get("kind") != "call":
                continue
            if row["effect"] == "Write":
                seen_write = True
                assert row["claimed_speculative"] is False, (scenario_id, row["tool"])
            if row["effect"] == "LiveRead":
                assert row["claimed_speculative"] is False, (scenario_id, row["tool"])
    assert seen_write, "no Write call was exercised end to end"


# --------------------------------------------------------------------------- #
# 8. the comparison has teeth
# --------------------------------------------------------------------------- #


def _corrupt_first_write(index: int, row: dict[str, Any]) -> dict[str, Any]:
    """Rewrite one argument of the first write, as a wrong speculation would."""
    if row.get("effect") != "Write":
        return row
    out = dict(row)
    args = dict(out["args"])
    for key, value in sorted(args.items()):
        if isinstance(value, str):
            args[key] = value + "_CORRUPTED"
            break
        if isinstance(value, bool):
            args[key] = not value
            break
        if isinstance(value, (int, float)):
            args[key] = value + 1
            break
    out["args"] = args
    return out


@pytest.mark.parametrize(
    "scenario_id",
    ["add_reminder_content_and_week_delta_and_time_multiple_user_turn", "turn_on_wifi_low_battery_mode"],
)
def test_a_corrupted_replay_is_detected_as_divergent(
    omp_stub, by_id: dict[str, Scenario], scenario_id: str
) -> None:
    stub, client = omp_stub
    sc = by_id[scenario_id]
    episode = _episode(stub, client, sc)
    assert episode["divergence"]["divergent"] is False
    assert any(c["effect"] == "Write" for c in episode["tool_trace"]), scenario_id

    corrupted = replay_episode(sc, episode, corrupt=_corrupt_first_write)
    verdict = compare_replay(episode, corrupted)
    assert verdict["divergent"] is True, "a corrupted replay must not pass"
    assert "tool_trace" in verdict["fields"]
    assert "world_state_digest_final" in verdict["fields"]
    assert verdict["detail"]["world_state_digest_final"]["observed"] != (
        verdict["detail"]["world_state_digest_final"]["replay"]
    )


def test_a_corrupted_read_result_is_detected(omp_stub, by_id: dict[str, Scenario]) -> None:
    stub, client = omp_stub
    sc = by_id["search_message_with_recency_oldest"]
    episode = _episode(stub, client, sc)
    assert episode["divergence"]["divergent"] is False

    def corrupt(index: int, row: dict[str, Any]) -> dict[str, Any]:
        if index != 0:
            return row
        out = dict(row)
        out["args"] = {**out["args"], "content": "definitely not in the message store"}
        return out

    corrupted = replay_episode(sc, episode, corrupt=corrupt)
    verdict = compare_replay(episode, corrupted)
    assert verdict["divergent"] is True
    assert "tool_trace" in verdict["fields"]


def test_an_untouched_replay_is_bit_identical(omp_stub, by_id: dict[str, Scenario]) -> None:
    """Replaying twice with no corruption is deterministic, so the teeth tests
    above cannot be passing merely because the replay is noisy."""
    stub, client = omp_stub
    sc = by_id[E2E_SCENARIOS[0]]
    episode = _episode(stub, client, sc)
    a = replay_episode(sc, episode)
    b = replay_episode(sc, episode)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert compare_replay(episode, a)["divergent"] is False


def test_episode_trace_shard_verifies(
    omp_stub, by_id: dict[str, Scenario], tmp_path: Path
) -> None:
    stub, client = omp_stub
    sc = by_id["search_message_with_recency_oldest"]
    _, registry, _ = episode_context(sc)
    stub.load_for(sc, registry)
    shard = tmp_path / "episode.jsonl"
    writer = TraceWriter(shard, meta={"test": "toolsandbox"})
    try:
        run_episode(
            sc,
            seed=0,
            arm="toolsandbox_crystalcall",
            client_factory=lambda: client,
            speculate=True,
            turn_timeout=120.0,
            trace=writer.append,
        )
    finally:
        meta = writer.close()
    verified = TraceReader(shard).verify()
    assert verified["records"] == meta["records"]
    assert verified["final_chain"] == meta["final_chain"]
    kinds = collections.Counter(r.kind for r in TraceReader(shard).records())
    assert kinds["toolsandbox_episode_start"] == 1
    assert kinds["toolsandbox_episode_end"] == 1
    assert kinds["toolsandbox_replay"] == 1
    assert kinds["toolsandbox_divergence"] == 1
    assert kinds["toolsandbox_turn"] == len(sc.user_turns)
