"""The arm registry is the only thing allowed to differ between arms.

These tests pin each arm's resolved model, system prompt and instrumentation
level, and assert that the shared substrate (OMP-Lite overlay, tool surface,
canonicaliser, retry policy) is identical everywhere.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crystalcall.ar import AR_POLICIES, assert_monotonic, build_ar_revisions
from crystalcall.harness import ARMS, TraceRunConfig
from crystalcall.omp_bridge import OmpLiteProfile
from crystalcall.runtime import Instrumentation

ROOT = Path(__file__).resolve().parents[1]

EXPECTED = {
    "diffusion_standard": ("vllm/diffusiongemma-26b-a4b-it", "configs/system_prompt.md", "parse", False),
    "crystalcall": ("vllm/diffusiongemma-26b-a4b-it", "configs/system_prompt.md", "full", False),
    "preact_style": ("vllm/diffusiongemma-26b-a4b-it", "configs/system_prompt.md", "parse", False),
    "gemma4_ar": ("vllm/gemma-4-26b-a4b-it", "configs/system_prompt.md", "off", True),
}


def test_arm_registry_is_exactly_the_declared_set():
    assert set(ARMS) == set(EXPECTED)


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_arm_resolves_to_its_declared_configuration(name):
    spec = ARMS[name]
    model, prompt, instr, ar = EXPECTED[name]
    assert spec.model == model
    assert spec.system_prompt == prompt
    assert spec.instrumentation == instr
    assert spec.autoregressive is ar
    assert Instrumentation(spec.instrumentation)
    assert (ROOT / spec.system_prompt).exists()


def test_config_defaults_do_not_shadow_the_arm_registry():
    """Regression: a non-None default here silently overrode every arm, which
    would have run DiffusionGemma for the autoregressive control."""
    cfg = TraceRunConfig(root=ROOT, manifest=ROOT / "x.jsonl", arm="gemma4_ar")
    assert cfg.model is None
    assert cfg.instrumentation is None


def test_only_diffusion_arms_launch_speculatively():
    assert Instrumentation(ARMS["crystalcall"].instrumentation).launches
    for name in ("diffusion_standard", "preact_style", "gemma4_ar"):
        assert not Instrumentation(ARMS[name].instrumentation).launches


def test_ar_arm_is_the_only_autoregressive_one():
    ar = [n for n, s in ARMS.items() if s.autoregressive]
    assert ar == ["gemma4_ar"]
    assert AR_POLICIES == ("ar_streaming", "ar_sptc")


def test_every_arm_shares_the_same_omp_lite_substrate():
    """Only model and system prompt may differ in the launch configuration."""
    profiles = {
        name: OmpLiteProfile(
            profile="crystalcall",
            config=ROOT / "configs/omp-lite.yml",
            system_prompt=ROOT / spec.system_prompt,
            model=spec.model,
            thinking="off",
            tools=("eval",),
        )
        for name, spec in ARMS.items()
    }
    invariant = set()
    for p in profiles.values():
        argv = p.argv()
        # Drop the two fields that are allowed to vary.
        for flag in ("--model", "--system-prompt"):
            i = argv.index(flag)
            del argv[i : i + 2]
        invariant.add(tuple(argv[1:]))  # argv[0] is the resolved binary path
    assert len(invariant) == 1, invariant


def test_every_arm_uses_the_identical_system_prompt():
    """Non-negotiable constraint 3: the prompt is fixed across arms.

    The P-ReAct-style baseline used to swap in a `call-first` prompt.  That was
    both the wrong mechanism (the memo requires altered token-selection
    priority, not a prompt reorder) and a broken control.  It is now a sampler
    change; the prompt file is shared by every arm.
    """
    prompts = {s.system_prompt for s in ARMS.values()}
    assert prompts == {"configs/system_prompt.md"}
    assert not (ROOT / "configs/system_prompt_preact.md").exists()


def test_preact_is_a_sampler_change_and_the_only_one():
    """The arm is defined by server-side decoding env, nothing else."""
    env = dict(ARMS["preact_style"].sampler_env)
    assert env == {"CRYSTALCALL_PREACT": "1", "CRYSTALCALL_PREACT_MODE": "both"}
    for name, spec in ARMS.items():
        if name != "preact_style":
            assert spec.sampler_env == (), name
    # Everything else about the arm matches the standard diffusion arm.
    a, b = ARMS["preact_style"], ARMS["diffusion_standard"]
    assert (a.model, a.system_prompt, a.instrumentation, a.autoregressive) == (
        b.model,
        b.system_prompt,
        b.instrumentation,
        b.autoregressive,
    )


def test_arm_registry_digest_is_stable():
    """A change to any arm's configuration must be a deliberate, visible edit."""
    blob = "|".join(
        f"{n}:{s.model}:{s.system_prompt}:{s.instrumentation}:{int(s.autoregressive)}"
        f":{';'.join('='.join(kv) for kv in s.sampler_env)}"
        for n, s in sorted(ARMS.items())
    )
    assert blob == (
        "crystalcall:vllm/diffusiongemma-26b-a4b-it:configs/system_prompt.md:full:0:|"
        "diffusion_standard:vllm/diffusiongemma-26b-a4b-it:configs/system_prompt.md:parse:0:|"
        "gemma4_ar:vllm/gemma-4-26b-a4b-it:configs/system_prompt.md:off:1:|"
        "preact_style:vllm/diffusiongemma-26b-a4b-it:configs/system_prompt.md:parse:0"
        ":CRYSTALCALL_PREACT=1;CRYSTALCALL_PREACT_MODE=both"
    )


# --------------------------------------------------------------------------- #
# AR projection
# --------------------------------------------------------------------------- #


class _ByteTokenizer:
    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, ids, **_kw) -> str:
        return bytes(int(i) for i in ids).decode("utf-8", "replace")

    def convert_ids_to_tokens(self, ids) -> list[str]:
        return [chr(int(i)) for i in ids]


def test_ar_projection_is_monotonic_and_committed():
    from crystalcall.ar import TextDelta

    deltas = [
        TextDelta(monotonic_ns=1_000, text="r0 = tool."),
        TextDelta(monotonic_ns=2_000, text="get_x(a=1)"),
        TextDelta(monotonic_ns=3_000, text="\n"),
    ]
    revs = build_ar_revisions(deltas, request_id="r", tokenizer=_ByteTokenizer())
    assert len(revs) == 3
    assert all(all(m for m in r.accepted_mask) for r in revs)
    assert all(r.phase == "commit" for r in revs)
    assert revs[-1].converged
    check = assert_monotonic(revs)
    assert check["monotonic"] and not check["violations"]


def test_ar_monotonicity_violation_is_detected():
    from crystalcall.trace import CanvasRevision

    def rev(step, ids):
        return CanvasRevision(
            request_id="r",
            canvas_id=0,
            step=step,
            monotonic_ns=1000 * (step + 1),
            predicted_token_ids=tuple(ids),
            accepted_mask=(True,) * len(ids),
            mean_entropy=0.0,
            converged=False,
            phase="commit",
        )

    good = [rev(0, [1, 2]), rev(1, [1, 2, 3])]
    bad = [rev(0, [1, 2]), rev(1, [9, 9, 3])]
    assert assert_monotonic(good)["monotonic"]
    assert not assert_monotonic(bad)["monotonic"]


def test_ar_deltas_are_sliced_at_the_program_turn_boundary():
    """An OMP turn issues a program request and then, after the tool results
    return, a closing-prose request. Only the first belongs to the trajectory a
    launch policy observes; including the second would fold tool wait time and
    final prose into AR decode while diffusion stops at the program stream."""
    from crystalcall.ar import deltas_from_events
    from crystalcall.harness import program_turn_end_ns
    from crystalcall.omp_bridge import TurnResult

    def delta(ns: int, text: str) -> dict:
        return {
            "type": "message_update",
            "_recv_ns": ns,
            "assistantMessageEvent": {"type": "text_delta", "delta": text},
        }

    events = [
        delta(100, "r0 = tool.get_x("),
        delta(200, "a=1)\n"),
        {"type": "tool_execution_start", "_recv_ns": 250},
        {"type": "tool_execution_end", "_recv_ns": 900},
        delta(1000, "I ran "),
        delta(1100, "the program."),
        {"type": "agent_end", "_recv_ns": 1200, "isTerminal": True},
    ]
    turn = TurnResult(
        prompt="p",
        assistant_text="",
        started_ns=50,
        ended_ns=1200,
        events=events,
        tool_calls=[],
        eval_programs=["r0 = tool.get_x(a=1)\n"],
    )

    cutoff = program_turn_end_ns(turn)
    assert cutoff == 250

    all_deltas = deltas_from_events(events)
    assert len(all_deltas) == 4
    sliced = [d for d in all_deltas if not d.monotonic_ns or d.monotonic_ns <= cutoff]
    assert [d.monotonic_ns for d in sliced] == [100, 200]

    revs = build_ar_revisions(sliced, request_id="r", tokenizer=_ByteTokenizer())
    assert len(revs) == 2
    # The projected response end is the last program token, never the prose turn.
    assert revs[-1].monotonic_ns == 200
    assert revs[-1].monotonic_ns < 1000


def test_speculating_arm_requires_frozen_thresholds(tmp_path, monkeypatch):
    """`k`, `tau` and the minimum useful overlap are selected on development
    data.  A held-out arm that speculates must refuse to run against the
    library defaults, or it would report an untuned policy under the frozen
    policy's name."""
    import json

    from crystalcall.harness import load_frozen_gate
    from crystalcall.scheduler import policy_by_name

    base = policy_by_name("full_gate")

    pol, meta = load_frozen_gate(tmp_path, base)
    assert meta["status"] == "absent"
    assert (pol.k, pol.tau, pol.min_overlap_ms) == (base.k, base.tau, base.min_overlap_ms)

    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/frozen_policy.json").write_text(
        json.dumps(
            {
                "selected_on": "experiments/manifests/dev.jsonl",
                "gate": {
                    "k": 3,
                    "tau": 0.95,
                    "min_overlap_ms": 100.0,
                    "confidence_mode": "min",
                },
            }
        )
    )
    frozen, meta = load_frozen_gate(tmp_path, base)
    assert meta["status"] == "applied"
    assert meta["selected_on"] == "experiments/manifests/dev.jsonl"
    assert meta["sha256"]
    assert (frozen.k, frozen.tau, frozen.min_overlap_ms) == (3, 0.95, 100.0)
    # A different policy is a different fingerprint, so drift is detectable.
    assert frozen.fingerprint() != base.fingerprint()


def test_run_trace_arm_refuses_to_speculate_without_a_frozen_policy(tmp_path):
    """The guard must be in `run_trace_arm`, before any model request."""
    import inspect

    from crystalcall import harness

    src = inspect.getsource(harness.run_trace_arm)
    assert "frozen_policy.json is absent" in src
    assert "instrumentation.launches" in src
    # And it must be raised before the manifest is walked.
    assert src.index("frozen_policy.json is absent") < src.index("for seed in cfg.seeds")


# --------------------------------------------------------------------------- #
# freeze drift
# --------------------------------------------------------------------------- #


def test_frozen_arm_digest_covers_the_sampler_env(tmp_path, monkeypatch):
    """A change to the P-ReAct-style arm's decoding env is a configuration
    change and must trip `verify-freeze`; and a hand-edited manifest block must
    fail against its own digest."""
    import hashlib
    import json
    import subprocess

    arms = {
        name: {
            "model": s.model,
            "system_prompt": s.system_prompt,
            "instrumentation": s.instrumentation,
            "autoregressive": s.autoregressive,
            "sampler_env": dict(s.sampler_env),
        }
        for name, s in sorted(ARMS.items())
    }
    digest = hashlib.sha256(
        json.dumps(arms, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert arms["preact_style"]["sampler_env"], "the arm's env must be part of the digest"

    # Flipping the mode changes the digest: the arm is a different experiment.
    other = json.loads(json.dumps(arms))
    other["preact_style"]["sampler_env"]["CRYSTALCALL_PREACT_MODE"] = "span_accept"
    assert (
        hashlib.sha256(
            json.dumps(other, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        != digest
    )

    # And the live CLI detects a hand-edited block against its own stored digest.
    fm = ROOT / "configs/frozen_manifest.json"
    # Phase 6 has not run, so the tree normally has no frozen manifest.  Make
    # one, tamper with it, assert the CLI notices, then restore exactly what was
    # there before -- a stray manifest would mark a phase as completed that
    # has not.
    preexisting = fm.exists()
    backup = fm.read_text() if preexisting else None
    if not preexisting:
        made = subprocess.run(
            ["uv", "run", "--project", ".", "crystalcall", "freeze-manifest"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        assert made.returncode == 0, made.stderr
    try:
        doc = json.loads(fm.read_text())
        doc["arms"]["preact_style"]["sampler_env"] = {}
        fm.write_text(json.dumps(doc, indent=2))
        proc = subprocess.run(
            ["uv", "run", "--project", ".", "crystalcall", "verify-freeze"],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )
        assert proc.returncode == 1, proc.stdout
        assert "does not match its own arms_digest" in proc.stdout + proc.stderr
    finally:
        if backup is None:
            fm.unlink(missing_ok=True)
        else:
            fm.write_text(backup)


def test_freeze_covers_the_preact_config(tmp_path):
    """The P-ReAct-style arm's token ids, entropy bound and stability depth are
    load-bearing for the S8 decisive comparison; a post-freeze edit must trip
    `verify-freeze` rather than silently redefining the baseline."""
    import json
    import subprocess

    fm = ROOT / "configs/frozen_manifest.json"
    cfg = ROOT / "configs/preact.yml"
    assert cfg.exists()

    preexisting = fm.exists()
    fm_backup = fm.read_text() if preexisting else None
    cfg_backup = cfg.read_text()

    def _run(cmd):
        return subprocess.run(
            ["uv", "run", "--project", ".", "crystalcall", cmd],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            check=False,
        )

    try:
        if not preexisting:
            made = _run("freeze-manifest")
            assert made.returncode == 0, made.stderr
        doc = json.loads(fm.read_text())
        assert "configs/preact.yml" in doc["configs"], sorted(doc["configs"])
        assert doc["configs"]["configs/preact.yml"]

        clean = _run("verify-freeze")
        assert clean.returncode == 0, clean.stdout + clean.stderr

        cfg.write_text(cfg_backup + "\n# post-freeze edit\n")
        drifted = _run("verify-freeze")
        assert drifted.returncode == 1
        assert "configs/preact.yml" in drifted.stdout + drifted.stderr
    finally:
        cfg.write_text(cfg_backup)
        if fm_backup is None:
            fm.unlink(missing_ok=True)
        else:
            fm.write_text(fm_backup)
