"""The trace join: revisions arrive under the server's request id, but the
shard is keyed by the parent ``arm/prompt/seed`` identity.

Regression guard. Sampler events carry each vLLM request id, while
``request``/``final`` records use the parent identity that
``load_trajectories`` joins on. If ``_write_request`` did not re-tag the
revisions, every revision would be silently dropped at load time and every
crystallization milestone would come back empty -- with no error anywhere.
"""

from __future__ import annotations
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import pytest

from crystalcall.harness import _write_request
from crystalcall.omp_bridge import AuthoritativeCall, HostToolRouter, TurnResult
from crystalcall.replay import load_trajectories
from crystalcall.sidechannel import RequestStream
from crystalcall.trace import CanvasRevision, TraceReader, TraceWriter

PARENT_ID = "diffusion_standard/simple_1/s0"
SERVER_ID = "chatcmpl-93a0f1c2deadbeef"


class _Rec:
    prompt_id = "simple_1"
    category = "simple"
    source = "bfcl_v4"
    source_id = "simple_1"
    user_message = "Compute the gcd of 12 and 18."
    expected_calls = ({"tool": "lookup_symbol", "args": {"name": "A"}},)

    class _T:
        name = "lookup_symbol"

    tools = (_T(),)


def _revisions(request_id: str, n: int = 4) -> list[CanvasRevision]:
    return [
        CanvasRevision(
            request_id=request_id,
            canvas_id=0,
            step=i,
            monotonic_ns=1_000_000_000 + i * 40_000_000,
            predicted_token_ids=(1, 2, 3, 4),
            accepted_mask=(i >= n - 1,) * 4,
            token_confidence=(0.99,) * 4,
            mean_entropy=0.05,
            converged=(i == n - 1),
            phase="commit" if i == n - 1 else "denoise",
        )
        for i in range(n)
    ]


def _stream(server_id: str, *, begin: int, end: int, canvas_done: int, n: int = 4):
    s = RequestStream(request_id=server_id)
    s.begin_ns = begin
    s.end_ns = end
    s.prompt_tokens = 42
    s.output_tokens = 256
    s.finish_reason = "stop"
    for r in _revisions(server_id, n):
        s.revisions.append(dc_replace(r, monotonic_ns=begin + r.step * 40_000_000))
    s.canvas_done_ns[0] = canvas_done
    return s


def _write(tmp_path: Path, registry, tools, server_id: str, *, extra_streams=()) -> Path:
    program = _stream(
        server_id, begin=1_000_000_000, end=1_200_000_000, canvas_done=1_190_000_000
    )
    streams = [program, *extra_streams]

    router = HostToolRouter(registry, tools.impls(), futures=None)
    router.handle("lookup_symbol", {"name": "A"}, host_call_id="h0", tool_call_id="t0")

    turn = TurnResult(
        prompt=_Rec.user_message,
        assistant_text="Done.",
        started_ns=1_000_000_000,
        ended_ns=1_900_000_000,
        events=[
            {"type": "tool_execution_start", "_recv_ns": 1_205_000_000},
        ],
        tool_calls=[],
        eval_programs=['r0 = lookup_symbol(name="A")\n'],
    )

    shard = tmp_path / "diffusion_standard-seed0.jsonl"
    writer = TraceWriter(shard, meta={"arm": "diffusion_standard", "seed": 0})
    try:
        _write_request(
            writer,
            rec=_Rec(),
            arm="diffusion_standard",
            seed=0,
            request_id=PARENT_ID,
            streams=streams,
            program_stream=program,
            turn=turn,
            router=router,
            registry=registry,
            runtime_report={},
            error=None,
        )
    finally:
        writer.close()
    return shard

def test_prose_turn_revisions_do_not_contaminate_the_trajectory(tmp_path, registry, tools):
    """An OMP turn issues several model requests, and each server stream
    restarts canvas_id and step at 0.  Only the program-producing request may
    become the analysed trajectory."""
    prose = _stream(
        "chatcmpl-prose-0001",
        begin=1_300_000_000,
        end=1_900_000_000,
        canvas_done=1_880_000_000,
        n=6,
    )
    shard = _write(tmp_path, registry, tools, SERVER_ID, extra_streams=[prose])
    traj = load_trajectories([shard], verify=True)[0]

    # 4 program revisions, not 4 + 6.
    assert len(traj.revisions) == 4
    assert [r.step for r in traj.revisions] == [0, 1, 2, 3]
    assert set(traj.canvas_done_ns) == {0}
    assert traj.canvas_done_ns[0] == 1_190_000_000
    # The authoritative execution point is the end of the program request, not
    # the end of the whole turn.
    assert traj.response_done_ns == 1_200_000_000
    assert traj.meta["turn_end_ns"] == 1_900_000_000


def test_every_model_request_is_still_recorded(tmp_path, registry, tools):
    prose = _stream(
        "chatcmpl-prose-0001",
        begin=1_300_000_000,
        end=1_900_000_000,
        canvas_done=1_880_000_000,
        n=6,
    )
    shard = _write(tmp_path, registry, tools, SERVER_ID, extra_streams=[prose])
    meta = next(iter(TraceReader(shard).by_kind("request")))["meta"]
    assert meta["program_server_request_id"] == SERVER_ID
    assert meta["server_request_ids"] == [SERVER_ID, "chatcmpl-prose-0001"]
    flags = {m["server_request_id"]: m["is_program_turn"] for m in meta["model_requests"]}
    assert flags == {SERVER_ID: True, "chatcmpl-prose-0001": False}
    assert [m["revisions"] for m in meta["model_requests"]] == [4, 6]


def test_program_turn_end_falls_back_to_tool_execution_start():
    from crystalcall.harness import program_turn_end_ns

    turn = TurnResult(
        prompt="p",
        assistant_text="",
        started_ns=1,
        ended_ns=999,
        events=[
            {"type": "message_update", "_recv_ns": 10},
            {"type": "tool_execution_start", "_recv_ns": 42},
            {"type": "tool_execution_start", "_recv_ns": 77},
        ],
        tool_calls=[],
        eval_programs=[],
    )
    assert program_turn_end_ns(turn) == 42
    empty = TurnResult(
        prompt="p",
        assistant_text="",
        started_ns=1,
        ended_ns=999,
        events=[],
        tool_calls=[],
        eval_programs=[],
    )
    assert program_turn_end_ns(empty) == 999


def test_revisions_survive_a_differing_server_request_id(tmp_path, registry, tools):
    shard = _write(tmp_path, registry, tools, SERVER_ID)
    TraceReader(shard).verify()

    trajectories = load_trajectories([shard], verify=True)
    assert len(trajectories) == 1
    traj = trajectories[0]
    assert traj.request_id == PARENT_ID
    assert len(traj.revisions) == 4, "revisions were dropped by the id join"
    assert {r.request_id for r in traj.revisions} == {PARENT_ID}
    assert traj.canvas_done_ns == {0: 1_190_000_000}
    assert traj.response_done_ns == 1_200_000_000


def test_original_server_ids_are_preserved_in_meta(tmp_path, registry, tools):
    shard = _write(tmp_path, registry, tools, SERVER_ID)
    request_records = list(TraceReader(shard).by_kind("request"))
    assert len(request_records) == 1
    meta = request_records[0]["meta"]
    assert meta["server_request_ids"] == [SERVER_ID]
    assert [m["prompt_tokens"] for m in meta["model_requests"]] == [42]


def test_matching_ids_are_left_untouched(tmp_path, registry, tools):
    shard = _write(tmp_path, registry, tools, PARENT_ID)
    traj = load_trajectories([shard], verify=True)[0]
    assert len(traj.revisions) == 4
    assert {r.request_id for r in traj.revisions} == {PARENT_ID}


def test_authoritative_calls_and_score_round_trip(tmp_path, registry, tools):
    shard = _write(tmp_path, registry, tools, SERVER_ID)
    traj = load_trajectories([shard], verify=True)[0]
    assert traj.valid_final_parse is True
    assert traj.benchmark_correct is True
    assert [c["tool"] for c in traj.authoritative_calls] == ["lookup_symbol"]
    assert traj.authoritative_calls[0]["args"] == {"name": "A"}

def test_trace_chain_detects_tampering(tmp_path, registry, tools):
    from crystalcall.trace import TraceIntegrityError

    shard = _write(tmp_path, registry, tools, SERVER_ID)
    lines = shard.read_text().splitlines()
    # Tamper with a revision record's mean entropy: a value the chain covers.
    target = next(i for i, ln in enumerate(lines) if '"kind":"revision"' in ln)
    assert '"mean_entropy":0.05' in lines[target], lines[target][:200]
    lines[target] = lines[target].replace('"mean_entropy":0.05', '"mean_entropy":0.99')
    tampered = tmp_path / "tampered.jsonl"
    tampered.write_text("\n".join(lines) + "\n")
    with pytest.raises(TraceIntegrityError):
        TraceReader(tampered).verify()


def test_trace_chain_detects_truncation(tmp_path, registry, tools):
    from crystalcall.trace import TraceIntegrityError

    shard = _write(tmp_path, registry, tools, SERVER_ID)
    lines = shard.read_text().splitlines()
    truncated = tmp_path / "truncated.jsonl"
    truncated.write_text("\n".join(lines[:-1]) + "\n")
    (tmp_path / "truncated.jsonl.meta.json").write_text(
        (shard.parent / (shard.name + ".meta.json")).read_text()
    )
    with pytest.raises(TraceIntegrityError):
        TraceReader(truncated).verify()


# --------------------------------------------------------------------------- #
# atomic shard commit
# --------------------------------------------------------------------------- #


def test_an_aborted_run_leaves_no_committable_shard(tmp_path, monkeypatch):
    """A truncated shard must never look complete to `--resume`.

    Regression: the development trace aborted on prompt 73 of 100 and left
    `diffusion_standard-seed0.jsonl` on disk with 72 records, footer and digest
    sealed. `--resume` keys on the shard's existence, so the retry would have
    skipped it as completed and every downstream rate -- parse rate,
    correctness, crystallization lead -- would have been computed over 72
    records while the report said 100.

    This drives the real `run_trace_arm` path: no GPU and no server, because it
    fails while binding the second record's tools, which happens before any
    request is issued.
    """
    import crystalcall.harness as H
    import crystalcall.manifest as M

    root = Path(__file__).resolve().parents[1]
    out = tmp_path / "raw" / "arm"
    calls = {"n": 0}
    real = M.registry_for

    # The failure must land INSIDE the generation loop, after the pre-flight
    # has already validated every record: that is the shape of the real
    # abort, and it is what leaves a partial shard behind.
    armed = {"v": False}

    def flaky(rec):  # noqa: ANN001, ANN202
        if armed["v"]:
            calls["n"] += 1
            # Record 1, not 2: the writer is opened before the loop, so the
            # partial-only outcome is identical and the test does not spend ~33 s
            # waiting on the model client for a record it will never use.
            raise RuntimeError("injected failure inside the generation loop")
        return real(rec)

    def ready(*_a, **_k):  # noqa: ANN002, ANN003, ANN202
        armed["v"] = True
        return True

    monkeypatch.setattr(M, "registry_for", flaky)
    monkeypatch.setattr(H, "_server_ready", ready)

    # A private sink port: the default collides with a concurrently running
    # overhead ladder or trace run on this box.
    import socket as _socket

    with _socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        free_port = probe.getsockname()[1]

    cfg = H.TraceRunConfig(
        root=root,
        manifest=root / "experiments/manifests/dev.jsonl",
        arm="diffusion_standard",
        seeds=(0,),
        out_dir=out,
        limit=3,
        instrumentation="off",
        trace_sink_endpoint=f"127.0.0.1:{free_port}",
    )
    with pytest.raises(RuntimeError, match="injected failure inside the generation loop"):
        H.run_trace_arm(cfg)

    shard = out / "diffusion_standard-seed0.jsonl"
    partial = shard.with_name(shard.name + ".partial")
    assert partial.exists(), "the aborted run must leave its evidence"
    assert not shard.exists(), "an aborted run must never produce the committed name"

    # The sidecar follows the partial, never the committed name, so no reader
    # can see metadata for a shard that does not exist.
    assert (partial.parent / (partial.name + ".meta.json")).exists()
    assert not (shard.parent / (shard.name + ".meta.json")).exists()

    # `--resume` keys on the committed name (harness.py), which is absent, so a
    # partial can never be skipped as completed work.
    assert not shard.exists()
