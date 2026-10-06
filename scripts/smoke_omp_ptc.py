# /// script
# requires-python = ">=3.12"
# dependencies = ["httpx>=0.27", "crystalcall"]
# [tool.uv.sources]
# crystalcall = { path = "..", editable = true }
# ///
"""End-to-end smoke test of the authoritative OMP-Lite PTC path, with no GPU.

This exercises the whole authoritative chain that the experiment depends on,
using a stub OpenAI-compatible server instead of the model:

    OMP-Lite (omp --mode rpc)
      -> native tool call to `eval`
      -> OMP's Python kernel
      -> `tool.<name>(...)` loopback bridge
      -> host_tool_call over RPC
      -> HostToolRouter: canonicalise -> claim-or-execute
      -> host_tool_result
      -> second model turn -> final assistant text

It also verifies that a *pre-launched* speculative future with the exact
canonical key is claimed by the authoritative call, and that a future with a
mutated argument is not.

Run:  uv run scripts/smoke_omp_ptc.py [--port 8000] [--keep-server]
Exit: 0 on success; non-zero with a printed reason otherwise.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crystalcall.canonicalize import canonicalize  # noqa: E402
from crystalcall.effects import ResourceClass  # noqa: E402
from crystalcall.futures import FutureKey, FutureTable  # noqa: E402
from crystalcall.manifest import load_manifest, registry_for  # noqa: E402
from crystalcall.omp_bridge import HostToolRouter, OmpLiteProfile, OmpRpcClient  # noqa: E402
from crystalcall.tools import build_impls  # noqa: E402

MODEL_ID = "diffusiongemma-26b-a4b-it"


# --------------------------------------------------------------------------- #
# stub OpenAI-compatible server
# --------------------------------------------------------------------------- #


class StubState:
    def __init__(self, program: str) -> None:
        self.program = program
        self.turns = 0
        self.requests: list[dict[str, Any]] = []
        self.lock = threading.Lock()


def _sse(obj: dict[str, Any]) -> bytes:
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


def _make_handler(state: StubState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args: Any) -> None:  # silence
            return

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/").endswith("/models"):
                body = json.dumps(
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": MODEL_ID,
                                "object": "model",
                                "owned_by": "crystalcall-smoke",
                                "max_model_len": 32768,
                            }
                        ],
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_error(404)

        def do_POST(self) -> None:  # noqa: N802
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n)
            try:
                req = json.loads(raw)
            except json.JSONDecodeError:
                self.send_error(400)
                return
            with state.lock:
                state.requests.append(req)
                turn = state.turns
                state.turns += 1

            created = int(time.time())
            base = {
                "id": f"chatcmpl-smoke-{turn}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_ID,
            }
            chunks: list[dict[str, Any]] = []
            if turn == 0:
                args = json.dumps({"language": "py", "code": state.program})
                chunks.append(
                    {
                        **base,
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "role": "assistant",
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": "call_smoke_0",
                                            "type": "function",
                                            "function": {"name": "eval", "arguments": args},
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        ],
                    }
                )
                chunks.append(
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}
                )
            else:
                for piece in ("Ran ", "the ", "program."):
                    chunks.append(
                        {
                            **base,
                            "choices": [
                                {"index": 0, "delta": {"content": piece}, "finish_reason": None}
                            ],
                        }
                    )
                chunks.append(
                    {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
                )
            chunks.append(
                {
                    **base,
                    "choices": [],
                    "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
                }
            )

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for c in chunks:
                payload = _sse(c)
                self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
                self.wfile.flush()
            done = b"data: [DONE]\n\n"
            self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    return Handler


# --------------------------------------------------------------------------- #
# the smoke test
# --------------------------------------------------------------------------- #


def build_program(record: Any) -> tuple[str, list[dict[str, Any]]]:
    """A Python PTC program issuing exactly the manifest's expected calls."""
    lines: list[str] = []
    expected: list[dict[str, Any]] = []
    for i, call in enumerate(record.expected_calls):
        args = ", ".join(f"{k}={v!r}" for k, v in sorted(call["args"].items()))
        lines.append(f"r{i} = tool.{call['tool']}({args})")
        expected.append({"tool": call["tool"], "args": dict(call["args"])})
    lines.append("display({" + ", ".join(f'"r{i}": r{i}' for i in range(len(expected))) + "})")
    return "\n".join(lines) + "\n", expected


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt-index", type=int, default=0)
    ap.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()

    manifest = ROOT / "experiments/manifests/dev.jsonl"
    records = load_manifest(manifest)
    record = records[args.prompt_index]
    registry = registry_for(record)
    impls = build_impls(record.tools)
    program, expected = build_program(record)

    print(f"prompt      {record.prompt_id} ({record.category})")
    print(f"tools       {registry.names()}")
    print(f"program     {program.strip()!r}")

    state = StubState(program)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), _make_handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    failures: list[str] = []
    request_id = f"smoke/{record.prompt_id}"
    futures = FutureTable(concurrency={rc: 2 for rc in ResourceClass})

    # Pre-launch two speculations: one exactly matching the first expected call,
    # one with a mutated argument that must never be claimed.
    matching_key = None
    if expected:
        call = canonicalize(registry, expected[0]["tool"], expected[0]["args"])
        matching_key = FutureKey.of(request_id, call)
        impl = impls[call.tool]
        futures.launch(
            matching_key,
            call,
            registry.resolve(call.tool).resource,
            lambda cancel, _i=impl, _a=call.args_dict: _i(_a, cancel),
            reason="smoke_prelaunch",
        )
        mutated_args = dict(expected[0]["args"])
        for k, v in mutated_args.items():
            if isinstance(v, str):
                mutated_args[k] = v + "_MUTATED"
                break
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                mutated_args[k] = v + 1
                break
        try:
            mcall = canonicalize(registry, expected[0]["tool"], mutated_args)
            if mcall.identity != call.identity:
                futures.launch(
                    FutureKey.of(request_id, mcall),
                    mcall,
                    registry.resolve(mcall.tool).resource,
                    lambda cancel, _i=impl, _a=mcall.args_dict: _i(_a, cancel),
                    reason="smoke_prelaunch_mutated",
                )
        except Exception as exc:  # noqa: BLE001 - a prompt may have no mutable arg
            print(f"note        no mutated variant for this prompt: {exc}")
        futures.drain()

    router = HostToolRouter(
        registry,
        impls,
        futures=futures,
        request_id_provider=lambda: request_id,
        claim_timeout=60.0,
    )
    profile = OmpLiteProfile(
        profile="crystalcall",
        config=ROOT / "configs/omp-lite.yml",
        system_prompt=ROOT / "configs/system_prompt.md",
        model=f"vllm/{MODEL_ID}",
        thinking="off",
        tools=("eval",),
        cwd=ROOT / "experiments/workspace",
    )
    (ROOT / "experiments/workspace").mkdir(parents=True, exist_ok=True)

    client = OmpRpcClient(profile, router, log_path=ROOT / "reports/smoke_rpc.jsonl")
    try:
        client.start(timeout=180)
        st = client.get_state()
        print(f"omp model   {(st.get('model') or {}).get('id')}")
        print(f"dumpTools   {[t.get('name') for t in (st.get('dumpTools') or [])]}")
        turn = client.prompt(record.user_message, timeout=args.timeout)
    finally:
        client.close()
        server.shutdown()

    calls = router.calls()
    print(f"model turns {state.turns}")
    print(f"assistant   {turn.assistant_text!r}")
    print(f"programs    {len(turn.eval_programs)}")
    print(f"host calls  {len(calls)}")
    for c in calls:
        print(
            f"  {c.tool_name} claimed={c.claimed_speculative} "
            f"err={c.error or c.canonicalization_error} digest={c.result_digest[:12]}"
        )

    if turn.error:
        failures.append(f"turn error: {turn.error}")
    if state.turns < 1:
        failures.append("stub server was never called: OMP did not reach the provider")
    if not turn.eval_programs:
        failures.append("no eval program observed in the event stream")
    if len(calls) != len(expected):
        failures.append(f"expected {len(expected)} host tool calls, saw {len(calls)}")
    for c in calls:
        if c.error or c.canonicalization_error:
            failures.append(f"{c.tool_name}: {c.error or c.canonicalization_error}")
    if expected and calls:
        want = canonicalize(registry, expected[0]["tool"], expected[0]["args"]).identity
        got = {c.call.identity for c in calls if c.call}
        if want not in got:
            failures.append(f"canonical identity mismatch: {want} not in {got}")
        claimed = [c for c in calls if c.claimed_speculative]
        if not claimed:
            failures.append("the exactly-matching pre-launched future was not claimed")
        mutated = [
            f
            for f in futures.futures()
            if f.launch_reason == "smoke_prelaunch_mutated" and f.state.value == "claimed"
        ]
        if mutated:
            failures.append("a mutated speculation was claimed")

    stats = futures.stats(request_id)
    futures.discard_unmatched(request_id)
    print(f"futures     {stats}")
    futures.close()

    report = {
        "schema_version": "crystalcall-smoke/1",
        "prompt_id": record.prompt_id,
        "category": record.category,
        "model_turns": state.turns,
        "assistant_text": turn.assistant_text,
        "eval_programs": turn.eval_programs,
        "host_calls": [c.to_json() for c in calls],
        "future_stats": stats,
        "failures": failures,
        "ok": not failures,
    }
    out = ROOT / "reports/smoke_omp_ptc.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(f"report      {out}")

    if failures:
        print("\nFAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nPASS: authoritative OMP-Lite PTC path verified end to end")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
