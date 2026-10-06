"""Host-side guards in the three-arm fidelity verifier.

These tests cover the parts that decide whether evidence may be trusted and
whether a run leaves a GPU-resident server behind. Neither needs a GPU.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def verifier():
    spec = importlib.util.spec_from_file_location(
        "verify_instrumentation", REPO / "scripts" / "verify_instrumentation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _completion(content: str) -> dict:
    return {
        "content": content,
        "reasoning": "",
        "finish_reason": "stop",
        "completion_tokens": len(content),
        "tool_calls": [],
        "server_request_id": f"req-{content}",
    }


def test_provenance_matching_source_is_accepted(verifier):
    current = {"git_commit": "abc", "sampler_sha256": "111"}
    blob = {"source_provenance": dict(current)}
    assert verifier.provenance_errors("baseline", blob, current) == []


def test_provenance_missing_block_is_rejected(verifier):
    """Evidence written before provenance stamping must not validate a run."""
    errs = verifier.provenance_errors("control", {}, {"git_commit": "abc"})
    assert len(errs) == 1
    assert "no source_provenance" in errs[0]
    assert "control" in errs[0]


def test_provenance_mismatch_names_the_differing_field(verifier):
    current = {"git_commit": "abc", "sampler_sha256": "111"}
    blob = {"source_provenance": {"git_commit": "abc", "sampler_sha256": "222"}}
    errs = verifier.provenance_errors("baseline", blob, current)
    assert len(errs) == 1
    assert "sampler_sha256" in errs[0]
    assert "git_commit" not in errs[0]


def test_provenance_ignores_commit_only_drift(verifier):
    """Docs/host-test commits cannot change engine behaviour, so a differing
    git_commit alone must not invalidate GPU evidence."""
    current = {"git_commit": "new", "sampler_sha256": "111"}
    blob = {"source_provenance": {"git_commit": "old", "sampler_sha256": "111"}}
    assert verifier.provenance_errors("baseline", blob, current) == []


def _run_control_arm(verifier, monkeypatch, tmp_path, baseline_blob, argv_extra=()):
    calls: list[str] = []
    monkeypatch.setattr(verifier, "EVIDENCE", tmp_path)
    monkeypatch.setattr(verifier, "stop_server", lambda: calls.append("stop"))
    monkeypatch.setattr(
        verifier, "start_server", lambda **kw: calls.append("start")
    )
    monkeypatch.setattr(verifier, "server_up", lambda base: True)
    monkeypatch.setattr(verifier, "resolve_model", lambda base: "model-x")
    monkeypatch.setattr(
        verifier,
        "run_prompts",
        lambda base, model: [_completion(c) for c in ("Paris", "51", "Red", "long")],
    )
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_blob), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_instrumentation.py",
            "--arm",
            "control",
            "--baseline-json",
            str(baseline_path),
            *argv_extra,
        ],
    )
    return verifier.main(), calls


def test_control_arm_stops_its_server_on_success(verifier, monkeypatch, tmp_path):
    """The control arm starts a server itself; leaving it up pins ~99 GB of GPU
    memory and blocks every later arm."""
    baseline = {
        "results": [_completion(c) for c in ("Paris", "51", "Red", "long")],
        "source_provenance": verifier.source_provenance(),
    }
    rc, calls = _run_control_arm(verifier, monkeypatch, tmp_path, baseline)
    assert rc == 0
    # stop before start (clean slate) and stop again after the requests
    assert calls == ["stop", "start", "stop"]
    written = json.loads((tmp_path / "control.json").read_text())
    assert written["reproducible_indices"] == [0, 1, 2, 3]
    assert written["source_provenance"] == verifier.source_provenance()


def test_control_arm_rejects_stale_baseline_before_starting_a_server(
    verifier, monkeypatch, tmp_path
):
    """Prerequisite evidence is checked first: an 8-minute GPU run must not be
    spent only to discover the baseline came from other source bytes."""
    baseline = {
        "results": [_completion(c) for c in ("Paris", "51", "Red", "long")],
        "source_provenance": {"git_commit": "stale", "sampler_sha256": "stale"},
    }
    rc, calls = _run_control_arm(verifier, monkeypatch, tmp_path, baseline)
    assert rc == 1
    assert calls == []


def test_control_arm_stops_its_server_when_the_run_raises(
    verifier, monkeypatch, tmp_path
):
    """`resolve_model`, `run_prompts` and the evidence write can all raise; the
    server this arm started must still be released."""
    baseline = {
        "results": [_completion(c) for c in ("Paris", "51", "Red", "long")],
        "source_provenance": verifier.source_provenance(),
    }

    def _boom(base, model):
        raise RuntimeError("engine died mid-run")

    calls: list[str] = []
    monkeypatch.setattr(verifier, "EVIDENCE", tmp_path)
    monkeypatch.setattr(verifier, "stop_server", lambda: calls.append("stop"))
    monkeypatch.setattr(verifier, "start_server", lambda **kw: calls.append("start"))
    monkeypatch.setattr(verifier, "server_up", lambda base: True)
    monkeypatch.setattr(verifier, "resolve_model", lambda base: "model-x")
    monkeypatch.setattr(verifier, "run_prompts", _boom)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_instrumentation.py",
            "--arm",
            "control",
            "--baseline-json",
            str(baseline_path),
        ],
    )
    with pytest.raises(RuntimeError, match="engine died mid-run"):
        verifier.main()
    assert calls == ["stop", "start", "stop"]


def test_control_arm_keep_server_leaves_it_running(verifier, monkeypatch, tmp_path):
    baseline = {
        "results": [_completion(c) for c in ("Paris", "51", "Red", "long")],
        "source_provenance": verifier.source_provenance(),
    }
    rc, calls = _run_control_arm(
        verifier, monkeypatch, tmp_path, baseline, argv_extra=("--keep-server",)
    )
    assert rc == 0
    assert calls == ["stop", "start"]


def _wire(verifier, monkeypatch, tmp_path, calls, *, start=None, run=None):
    monkeypatch.setattr(verifier, "EVIDENCE", tmp_path)
    monkeypatch.setattr(verifier, "stop_server", lambda: calls.append("stop"))
    monkeypatch.setattr(
        verifier, "start_server", start or (lambda **kw: calls.append("start"))
    )
    monkeypatch.setattr(verifier, "server_up", lambda base: True)
    monkeypatch.setattr(verifier, "resolve_model", lambda base: "model-x")
    monkeypatch.setattr(
        verifier,
        "run_prompts",
        run
        or (lambda base, model: [_completion(c) for c in ("Paris", "51", "Red", "long")]),
    )


def _argv(monkeypatch, arm: str, baseline_path: Path, *extra: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_instrumentation.py",
            "--arm",
            arm,
            "--baseline-json",
            str(baseline_path),
            *extra,
        ],
    )


def _baseline_blob(verifier) -> dict:
    return {
        "results": [_completion(c) for c in ("Paris", "51", "Red", "long")],
        "source_provenance": verifier.source_provenance(),
    }


def test_control_arm_stops_a_container_whose_startup_raised(
    verifier, monkeypatch, tmp_path
):
    """The serve script can create the container and then fail while waiting for
    readiness; the partially started server must still be released."""
    calls: list[str] = []

    def _start(**kw):
        calls.append("start")
        raise RuntimeError("server never became ready")

    _wire(verifier, monkeypatch, tmp_path, calls, start=_start)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_baseline_blob(verifier)), encoding="utf-8")
    _argv(monkeypatch, "control", baseline_path)
    with pytest.raises(RuntimeError, match="never became ready"):
        verifier.main()
    assert calls == ["stop", "start", "stop"]


def test_baseline_arm_stops_a_container_whose_startup_raised(
    verifier, monkeypatch, tmp_path
):
    calls: list[str] = []

    def _start(**kw):
        calls.append("start")
        raise RuntimeError("server never became ready")

    _wire(verifier, monkeypatch, tmp_path, calls, start=_start)
    baseline_path = tmp_path / "baseline.json"
    _argv(monkeypatch, "baseline", baseline_path)
    with pytest.raises(RuntimeError, match="never became ready"):
        verifier.main()
    assert calls == ["stop", "start", "stop"]


def test_baseline_arm_stops_its_server_when_the_run_raises(
    verifier, monkeypatch, tmp_path
):
    calls: list[str] = []

    def _boom(base, model):
        raise RuntimeError("engine died mid-run")

    _wire(verifier, monkeypatch, tmp_path, calls, run=_boom)
    baseline_path = tmp_path / "baseline.json"
    _argv(monkeypatch, "baseline", baseline_path)
    with pytest.raises(RuntimeError, match="engine died mid-run"):
        verifier.main()
    assert calls == ["stop", "start", "stop"]


def test_baseline_arm_keep_server_leaves_it_running(verifier, monkeypatch, tmp_path):
    calls: list[str] = []
    _wire(verifier, monkeypatch, tmp_path, calls)
    baseline_path = tmp_path / "baseline.json"
    _argv(monkeypatch, "baseline", baseline_path, "--keep-server")
    assert verifier.main() == 0
    assert calls == ["stop", "start"]
    written = json.loads(baseline_path.read_text())
    assert written["source_provenance"] == verifier.source_provenance()
