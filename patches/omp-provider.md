# OMP integration reference (in place of `omp-provider.patch`)

This file stands in for an `omp-provider.patch`: **CrystalCall does not patch or fork Oh My
Pi.** The pinned upstream binary `omp/18.1.4` is used unmodified and every integration point
is a documented, supported interface. Nothing about the harness is bespoke: this file plus
`configs/omp-lite.yml` list every way OMP is configured, so OMP cannot contribute hidden
policies.

## What is used, and why no patch is needed

| Need | Interface used | Why not a patch |
| --- | --- | --- |
| Headless driving | `omp --mode rpc` (JSON-RPC over stdio, protocol v2 negotiated) | The only headless surface with a published, versioned wire contract; `--mode json` has no documented frame schema. |
| Register benchmark tools | RPC `set_host_tools` | Puts our tools in the session's *active* registry, which is the same gate OMP's Python PTC bridge checks. Verified: after registration `get_state.dumpTools` reports `["eval", "search_repo"]`. |
| Authoritative tool dispatch | inbound `host_tool_call` / outbound `host_tool_result` | Because the tools are host-owned, dispatch necessarily round-trips into our process. That makes `HostToolRouter` the single choke point for canonicalisation, the exact-key claim and per-call verification — exactly where a patch would otherwise have had to inject. |
| Programmatic tool calling | OMP's own `tool.<name>(args)` Python loopback bridge inside the `eval` kernel | Already present in the pinned binary (`POST $PI_TOOL_BRIDGE_URL/v1/tool`, `Authorization: Bearer $PI_TOOL_BRIDGE_TOKEN`, body `{session, run, name, args}`). Re-implementing it would replace the runtime under test with a toy loop. |
| Fresh, non-leaking context per prompt | RPC `new_session` + `set_host_tools` | Each benchmark prompt gets its own tool surface and its own empty conversation. |
| Model wiring | `~/.omp/profiles/crystalcall/agent/models.yml` via `scripts/install_omp_profile.sh` | Provider id `vllm` because membership in OMP's local OpenAI-compatible provider set — not the URL — enables the local-model stream-timeout floor and reasoning-content replay. |
| Strip every adaptive policy | `--config configs/omp-lite.yml` plus launch flags | `omp://rpc.md` states that project/global config, `--config` and isolated settings remain authoritative over RPC-mode host defaults, so an explicit overlay wins. Every key in the overlay was verified against `omp config list` on 18.1.4. |
| Prove the surface is fixed | RPC `get_state` | `crystalcall doctor` hashes `systemPrompt` and `dumpTools` every run; `crystalcall verify-freeze` fails on drift. Current values: prompt `08e13f53ad4572a7`, tool representation `9d00d75367dca991`, tools `['eval']`. |

## Exact launch

`OmpLiteProfile.argv()` in `src/crystalcall/omp_bridge.py` is the single source of truth:

```
omp --mode rpc
    --profile crystalcall
    --model <arm model>
    --thinking off
    --tools eval
    --no-extensions --no-skills --no-rules --no-lsp --no-pty
    --no-session --no-title --auto-approve
    --config        configs/omp-lite.yml
    --system-prompt <arm system prompt>
    --cwd           experiments/workspace
```

The environment is scrubbed of every `*_API_KEY` / `*_OAUTH_TOKEN` and of `PI_DIALECT`, so
the harness cannot reach a hosted provider and cannot have its tool wire format flipped from
outside. `OmpLiteProfile.fingerprint()` digests the argv plus the hashes of the config and
system-prompt files, and that fingerprint is recorded in every trace shard header.

Only two fields differ between arms — `--model` and `--system-prompt`. This is asserted, not
merely commented: `tests/test_arms.py::test_every_arm_shares_the_same_omp_lite_substrate`
builds each arm's argv, removes those two flags, and requires exactly one distinct remainder.

## The one accepted deviation

`--system-prompt` swaps OMP's default instruction template for the custom-prompt template, but
the project/environment footer and a first-turn `<system-reminder>` carrying the date and cwd
remain; byte-exact sole-system-message replacement is SDK-only
(`CreateAgentSessionOptions.systemPrompt`) and unreachable from RPC. Rather than fork OMP to
remove it, the residual is accepted, held identical across arms, and *proved* identical by the
`get_state.systemPrompt` digest. The cwd is pinned; the date component drifts across calendar
days and is recorded as a known, arm-invariant nuisance.

## Latency hygiene inside the harness

Two things that would otherwise silently bias the speculating arm are deliberately kept out of
the live turn:

* **Claim verification is deferred.** `HostToolRouter.handle()` returns the claimed value
  immediately and queues the verifying re-execution; `flush_verifications()` runs it only
  after `client.prompt()` has returned. Re-executing inside the turn would delay the model's
  next request for the CrystalCall arm alone. Guarded by
  `tests/test_semantic_equivalence.py::test_handle_return_is_not_blocked_by_verification`.
* **Only the program-producing model request is the analysed trajectory.** An OMP turn issues
  a program request and then a closing-prose request; the latter's revisions are excluded and
  its identity recorded in `meta.model_requests`. Guarded by `tests/test_trace_join.py`.

## Files that constitute the integration

```
configs/omp-lite.yml                 the frozen settings overlay
configs/models.yml                   provider + both models
configs/system_prompt.md             the one fixed system prompt
scripts/install_omp_profile.sh       installs models.yml, fails loudly if it does not parse
src/crystalcall/omp_bridge.py        RPC client, host-tool router, claim + deferred verification
tests/test_arms.py                   proves the substrate is identical across arms
tests/test_semantic_equivalence.py   proves the claim/execute paths agree and do not block
scripts/smoke_omp_ptc.py             GPU-free end-to-end proof of the whole chain
```
