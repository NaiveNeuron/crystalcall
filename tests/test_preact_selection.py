"""The P-ReAct-style arm may not run on an unvalidated priority setting.

`configs/preact.yml`'s `span_entropy_bound` is a development-tuned parameter,
not a guess: at the original guessed value of 0.6 nats the arm lost every tool
call on 4 of 6 development prompts and two final programs stopped parsing.
A crippled baseline would hand CrystalCall an artefactual
advantage on the S8 decisive comparison, so the configured value must be the
one the sweep selected, and the provenance must be on record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "preact.yml"
SWEEP = ROOT / "experiments" / "derived" / "preact_bound_sweep.json"


def _cfg() -> dict:
    return yaml.safe_load(CFG.read_text(encoding="utf-8"))


def test_config_exists_and_declares_the_mechanism_defaults():
    doc = _cfg()
    defaults = doc["defaults"]
    assert defaults["mode"] in ("span_accept", "ready_signal", "both")
    assert isinstance(defaults["span_entropy_bound"], (int, float))
    assert int(defaults["require_stable_steps"]) >= 1
    # Off by default: the CrystalCall and standard arms run the same file.
    assert defaults["enabled"] is False


def test_delimiter_ids_are_pinned_with_their_tokenizer_provenance():
    """A checkpoint swap must fail loudly, not silently produce a vacuous arm:
    a wrong delimiter id means no span is ever found and the mechanism becomes
    a no-op that still looks like a baseline."""
    tokens = _cfg()["tokens"]
    assert int(tokens["open_token_id"]) >= 0
    assert int(tokens["close_token_id"]) >= 0
    assert tokens["open_token_id"] != tokens["close_token_id"]
    assert len(str(tokens["tokenizer_sha256"])) == 64
    assert tokens["model_repo"] == "google/diffusiongemma-26B-A4B-it"


def test_the_configured_bound_is_the_swept_selection():
    """The bound must come from `scripts/preact_bound_sweep.py`, not from taste."""
    if not SWEEP.is_file():
        pytest.skip("bound sweep has not been run yet (GPU stage)")
    report = json.loads(SWEEP.read_text(encoding="utf-8"))
    selected = report.get("selected")
    doc = _cfg()
    configured = float(doc["defaults"]["span_entropy_bound"])

    if selected is None:
        # No bound survived: the arm must be running readiness-only, which
        # cannot change the trajectory, and that must be the declared mode.
        assert doc["defaults"]["mode"] == "ready_signal", (
            "the sweep found no acceptable bound, so priority acceptance must be "
            "disabled (mode=ready_signal); shipping span_accept would ship a "
            "knowingly crippled baseline"
        )
        return

    assert configured == pytest.approx(float(selected)), (
        f"configs/preact.yml has span_entropy_bound={configured} but the sweep "
        f"selected {selected}; the baseline would not be the tuned one"
    )
    sel = doc.get("selection") or {}
    assert sel.get("selected_on"), "record where the bound was selected"
    assert sel.get("sweep_report"), "record the sweep report path"
    assert sel.get("sweep_sha256"), "record the sweep report digest"


def test_the_selected_rung_actually_passed_its_own_criteria():
    if not SWEEP.is_file():
        pytest.skip("bound sweep has not been run yet (GPU stage)")
    report = json.loads(SWEEP.read_text(encoding="utf-8"))
    selected = report.get("selected")
    if selected is None:
        pytest.skip("no bound was selected; covered by the mode assertion above")
    rung = next(r for r in report["rungs"] if float(r["bound"]) == float(selected))
    score = rung["score"]
    ref, got = score["reference_rates"], score["rates"]
    # No degradation in any ground-truth rate.
    assert got["correct_rate"] >= ref["correct_rate"]
    assert got["parse_rate"] >= ref["parse_rate"]
    assert got["called_any_rate"] >= ref["called_any_rate"]
    assert got["ready_events"] > 0
    assert score["reject_reasons"] == []
    # And it is genuinely the largest acceptable rung.
    acceptable = [float(r["bound"]) for r in report["rungs"] if r["score"]["acceptable"]]
    assert float(selected) == max(acceptable)


def test_every_rejected_rung_is_preserved_with_its_reason():
    """Failed configurations stay in the result data."""
    if not SWEEP.is_file():
        pytest.skip("bound sweep has not been run yet (GPU stage)")
    report = json.loads(SWEEP.read_text(encoding="utf-8"))
    assert report["rungs"], "the sweep recorded no rungs"
    assert [float(r["bound"]) for r in report["rungs"]] == sorted(
        float(r["bound"]) for r in report["rungs"]
    )
    for rung in report["rungs"]:
        score = rung["score"]
        for key in ("rates", "reference_rates", "divergence", "reject_reasons", "acceptable"):
            assert key in score, f"rung {rung['bound']} is missing {key}"
        for key in ("correct_rate", "parse_rate", "called_any_rate", "ready_events"):
            assert key in score["rates"], f"rung {rung['bound']} rates missing {key}"
        # A rejected rung must say *why*, in words, and the reason must be real.
        assert bool(score["reject_reasons"]) == (not score["acceptable"])
        if not score["acceptable"]:
            ref, got = score["reference_rates"], score["rates"]
            assert (
                got["correct_rate"] < ref["correct_rate"]
                or got["parse_rate"] < ref["parse_rate"]
                or got["called_any_rate"] < ref["called_any_rate"]
                or got["ready_events"] == 0
            ), f"rung {rung['bound']} is marked unacceptable with no failing criterion"


def test_per_prompt_divergence_is_recorded_but_is_not_a_criterion():
    """The engine is not restart-reproducible, so per-prompt disagreement
    must never decide acceptance -- only be reported."""
    if not SWEEP.is_file():
        pytest.skip("bound sweep has not been run yet (GPU stage)")
    report = json.loads(SWEEP.read_text(encoding="utf-8"))
    for rung in report["rungs"]:
        div = rung["score"]["divergence"]
        assert "differing_calls" in div and "differ_rate" in div
        # A rung with divergence but no rate degradation must still be acceptable.
        ref, got = rung["score"]["reference_rates"], rung["score"]["rates"]
        no_degradation = (
            got["correct_rate"] >= ref["correct_rate"]
            and got["parse_rate"] >= ref["parse_rate"]
            and got["called_any_rate"] >= ref["called_any_rate"]
            and got["ready_events"] > 0
        )
        assert rung["score"]["acceptable"] == no_degradation, (
            f"rung {rung['bound']}: acceptance disagrees with the declared rule; "
            f"divergence was {div['differ_rate']:.0%}"
        )


# --------------------------------------------------------------------------- #
# the sweep records its own selection
# --------------------------------------------------------------------------- #


def _sweep_module():
    import importlib.util
    import sys

    path = ROOT / "scripts/preact_bound_sweep.py"
    if not path.is_file():
        pytest.skip("bound sweep script missing")
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_bs", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _redirected_sweep(tmp_path, monkeypatch):
    """The sweep module with its config and report paths pointed at `tmp_path`.

    `write_selection` edits files through module globals, so redirecting them
    gives identical coverage without mutating the repository -- no
    finally-restore to lose to a Ctrl-C, and safe under parallel test runs.
    """
    import shutil

    mod = _sweep_module()
    cfg = tmp_path / "preact.yml"
    shutil.copyfile(CFG, cfg)
    monkeypatch.setattr(mod, "PREACT_CFG", cfg)
    monkeypatch.setattr(mod, "OUT", tmp_path / "preact_bound_sweep.json")
    monkeypatch.setattr(mod, "REPO", tmp_path)
    return mod


def test_write_selection_records_provenance_on_success(tmp_path, monkeypatch):
    """`--write-selection` must satisfy the arm's guard mechanically.

    The guard exists because a knowingly-destructive bound would hand
    CrystalCall an artefactual S8 advantage, so the provenance must be
    written by the sweep, never by hand.
    """
    mod = _redirected_sweep(tmp_path, monkeypatch)
    report = {"selected": 0.2, "prompt_ids": ["a", "b"], "selection_rule": "rule"}
    mod.OUT.write_text(json.dumps(report), encoding="utf-8")
    mod.write_selection(report)
    doc = yaml.safe_load(mod.PREACT_CFG.read_text(encoding="utf-8"))
    sel = doc["selection"]
    assert sel["status"] == "selected"
    assert sel["selected_on"]
    assert len(str(sel["sweep_sha256"])) == 64
    # The digest must match the report as it sits on disk.
    assert sel["sweep_sha256"] == mod.sha256_file(mod.OUT)
    assert float(doc["defaults"]["span_entropy_bound"]) == 0.2


def test_write_selection_degrades_to_ready_signal_on_rejection(tmp_path, monkeypatch):
    """If no bound survives, priority acceptance is switched off rather than
    shipped broken: `ready_signal` cannot change the trajectory at all."""
    mod = _redirected_sweep(tmp_path, monkeypatch)
    report = {"selected": None, "prompt_ids": ["a"], "selection_rule": "rule"}
    mod.OUT.write_text(json.dumps(report), encoding="utf-8")
    mod.write_selection(report)
    doc = yaml.safe_load(mod.PREACT_CFG.read_text(encoding="utf-8"))
    assert doc["selection"]["status"] == "rejected"
    assert doc["defaults"]["mode"] == "ready_signal"


# --------------------------------------------------------------------------- #
# which pilot attempt is this?
# --------------------------------------------------------------------------- #


def _attempt_module():
    import importlib.util
    import sys

    path = ROOT / "scripts/pilot_attempt.py"
    if not path.is_file():
        pytest.skip("attempt resolver missing")
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("_pa", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path, prompt: str, attempts: dict[int, str]) -> Path:
    """A minimal repository shape for the resolver."""
    (tmp_path / "configs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "reports").mkdir(parents=True, exist_ok=True)
    (tmp_path / "configs/system_prompt.md").write_text(prompt, encoding="utf-8")
    for n, sha in attempts.items():
        (tmp_path / f"reports/pilot_attempt{n}.json").write_text(
            json.dumps({"attempt": n, "system_prompt_sha256": sha}), encoding="utf-8"
        )
    return tmp_path


def _resolve(mod, monkeypatch, tmp_path, capsys) -> tuple[int, str]:
    monkeypatch.setattr(mod, "REPO", tmp_path)
    code = mod.main()
    out = capsys.readouterr()
    return code, (out.out + out.err).strip()


def test_first_run_is_attempt_one(tmp_path, monkeypatch, capsys):
    mod = _attempt_module()
    _tree(tmp_path, "prompt A\n", {})
    code, out = _resolve(mod, monkeypatch, tmp_path, capsys)
    assert code == 0 and out == "1"


def test_a_changed_prompt_after_one_attempt_is_the_permitted_repair(
    tmp_path, monkeypatch, capsys
):
    mod = _attempt_module()
    _tree(tmp_path, "prompt B\n", {1: "a" * 64})
    code, out = _resolve(mod, monkeypatch, tmp_path, capsys)
    assert code == 0 and out == "2"


def test_a_fresh_clone_reproducing_attempt_two_is_allowed(tmp_path, monkeypatch, capsys):
    """The regression this resolver exists for: the reports are committed, so
    presence-based derivation refused a clone that was merely reproducing."""
    import hashlib

    mod = _attempt_module()
    prompt = "prompt B\n"
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    _tree(tmp_path, prompt, {1: "a" * 64, 2: sha})
    code, out = _resolve(mod, monkeypatch, tmp_path, capsys)
    assert code == 0 and out == "2", out


def test_reproducing_attempt_one_is_allowed(tmp_path, monkeypatch, capsys):
    import hashlib

    mod = _attempt_module()
    prompt = "prompt A\n"
    sha = hashlib.sha256(prompt.encode()).hexdigest()
    _tree(tmp_path, prompt, {1: sha, 2: "b" * 64})
    code, out = _resolve(mod, monkeypatch, tmp_path, capsys)
    assert code == 0 and out == "1", out


def test_a_third_distinct_prompt_is_refused(tmp_path, monkeypatch, capsys):
    mod = _attempt_module()
    _tree(tmp_path, "prompt C\n", {1: "a" * 64, 2: "b" * 64})
    code, out = _resolve(mod, monkeypatch, tmp_path, capsys)
    assert code == 3
    assert "refusing a third attempt" in out
    assert "retuning against the pilot outcome" in out


def test_committed_attempt_reports_carry_a_resolvable_prompt_digest():
    """Both recorded attempts must be reproducible from a clean clone.

    `system_prompt_sha256` was added after the two pilot runs, so attempt 1's
    value is backfilled from the tracked prompt it used and marked as such.
    Without it, checking out attempt 1's prompt to
    reproduce that run would be refused as a third attempt.
    """
    import hashlib

    for n, prompt_rel in (
        (1, "configs/system_prompt_attempt1.md"),
        (2, "configs/system_prompt.md"),
    ):
        report = ROOT / f"reports/pilot_attempt{n}.json"
        prompt = ROOT / prompt_rel
        if not report.is_file():
            pytest.skip(f"attempt {n} not recorded in this tree")
        assert prompt.is_file(), f"{prompt_rel} must stay tracked for reproducibility"
        doc = json.loads(report.read_text(encoding="utf-8"))
        want = hashlib.sha256(prompt.read_bytes()).hexdigest()
        assert doc.get("system_prompt_sha256") == want, (
            f"attempt {n}'s recorded prompt digest does not match {prompt_rel}"
        )
    # Only the pre-field run is marked backfilled; the other is genuine.
    a1 = json.loads((ROOT / "reports/pilot_attempt1.json").read_text(encoding="utf-8"))
    a2 = json.loads((ROOT / "reports/pilot_attempt2.json").read_text(encoding="utf-8"))
    assert a1.get("system_prompt_sha256_backfilled") is True
    assert a1.get("system_prompt_sha256_source") == "configs/system_prompt_attempt1.md"
    assert not a2.get("system_prompt_sha256_backfilled")
