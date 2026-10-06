# The ToolSandbox validation arm

This arm is the experiment's proof that the losslessness invariant holds for a
**stateful, multi-turn** workload in which writes and live reads are interleaved
with eligible snapshot reads.  Everything here runs against a world state that
this process owns, driven through the authoritative OMP-Lite programmatic
tool-calling path; the model never touches the world directly.

| artifact | path |
| --- | --- |
| adapter | `src/crystalcall/toolsandbox.py` |
| declaration (all of it, as data) | `configs/toolsandbox.yml` |
| runner | `scripts/run_toolsandbox.py` |
| tests | `tests/test_toolsandbox.py` |
| frozen scenario set | `experiments/manifests/toolsandbox.jsonl` |
| trace shards | `experiments/raw/toolsandbox/<arm>-seed<N>.jsonl` (+ `.meta.json`) |
| derived rows | `experiments/derived/toolsandbox_episodes.jsonl`, `…_turns.jsonl` |

Provenance of what this document describes:

| file | sha256 |
| --- | --- |
| `experiments/manifests/toolsandbox.jsonl` | `e5f0214f234bc123f0d711f0f54f5e9a07cc07c5ee04f49cf73c68f2e991ad90` |
| `configs/toolsandbox.yml` | `e0015f5e244ebc4d88d12b56bbacbc621bd4f7f1594a4843c0699ead9f015060` |

## 1. The checkout, and what is read from it

ToolSandbox is vendored at `experiments/vendor/toolsandbox/checkout`
(`apple/ToolSandbox` @ `165848b9a78cead7ca7fe7c89c688b58e6501219`, archive
sha256 `0019647ab9a52348680a1e33f99c5db9c32b9513bcd78fed1af9c4beb804427e`; see
`docs/manifests.md`).

**It is never imported and never installed, and its own harness is never run.**
Two reasons, both hard: it needs `polars`, `attrs`, `networkx`, `rapidfuzz`,
`strenum`, `dill`, `typeguard`, `pint`, `geopy` and `holidays`, none of which is
a CrystalCall runtime dependency; and its RapidAPI-backed tools require a
`RAPID_API_KEY` and live network, so they are neither reachable nor reproducible.
We drive OMP, not their agent loop.  The checkout is parsed with `ast`.

Every symbol this adapter reads or reimplements is listed under
`checkout.symbols` in `configs/toolsandbox.yml`.  The load-bearing ones:

### How a scenario is declared

* `tool_sandbox/common/scenario.py:ScenarioExtension` — the literal every
  scenario is.  Fields used: `name`, `base_scenario` (a subscript into the dict
  returned by `named_base_scenarios`), `messages`, `tool_allow_list`,
  `milestones`, `minefields`, `categories`.
* `tool_sandbox/common/scenario.py:Scenario` — the container
  `ScenarioExtension.get_extended_scenario()` produces (`starting_context`,
  `evaluation`, `max_messages`, `categories`).  Not used at run time; read to
  establish that `starting_context` is the only source of initial world state.
* `tool_sandbox/common/execution_context.py:ScenarioCategories` — the category
  tags (`MULTIPLE_USER_TURN`, `STATE_DEPENDENCY`, `INSUFFICIENT_INFORMATION`, …)
  the family split and the scripted-follow-up rule read.
* `tool_sandbox/common/execution_context.py:RoleType` — `SYSTEM`, `USER`,
  `AGENT`, `EXECUTION_ENVIRONMENT`; distinguishes the user-simulator
  instruction from the user's own turn, and names the sender/recipient pair in
  every sandbox milestone target.
* `tool_sandbox/scenarios/user_simulator_few_shot_examples.py:USER_INSTRUCTION`
  — the role-play preamble prepended to every `SYSTEM -> USER` instruction.
* Scenario modules read (the three the frozen 64 come from):
  `tool_sandbox/scenarios/multiple_tool_call_scenarios.py` (33 scenarios),
  `multiple_user_turn_scenarios.py` (15), `insufficient_information_scenarios.py`
  (16).

### How tools are declared

* `tool_sandbox/common/utils.py:register_as_tool` — the decorator that marks an
  agent-visible tool (`@register_as_tool(visible_to=(RoleType.AGENT,))`).
* `tool_sandbox/common/tool_discovery.py:find_tools_by_module` — confirms tool
  discovery is exactly "module-level functions carrying that decorator".
* Tool modules read: `tool_sandbox/tools/{contact,messaging,reminder,setting,utilities,rapid_api_search_tools,user_tools}.py`.
  34 tools carry the decorator; the frozen 64 scenarios' allow-lists use 29 of
  them.  `end_conversation` (`user_tools.py`) is upstream's always-allowed turn
  terminator; we own turn structure, so it is not in our surface.
* `tool_sandbox/common/utils.py:filter_dataframe` and the
  `{exact_match,fuzzy_match,gt_eq,lt_eq}_filter_dataframe` helpers — the exact
  search semantics each `search_*` tool has, including the `ValueError` when no
  criterion is supplied and the per-column fuzzy thresholds (90 by default, 50
  for `content`).
* `tool_sandbox/common/utils.py:deterministic_uuid` —
  `uuid5(NAMESPACE_URL, payload)`, which is how every seeded row id is derived.

### How world state is represented and mutated

* `tool_sandbox/common/execution_context.py:ExecutionContext` — "a full
  encapsulation of the sandbox world state".
* `ExecutionContext.dbs_schemas` — the authoritative column list per namespace,
  reproduced verbatim in our rows.
* `ExecutionContext:DatabaseNamespace` — `SANDBOX`, `SETTING`, `CONTACT`,
  `MESSAGING`, `REMINDER`.
* Reads go through `ExecutionContext.get_database`; mutations go through exactly
  three methods: `add_to_database`, `remove_from_database`, `update_database`.
  This is the fact the whole effect assignment rests on.
* `tool_sandbox/scenarios/base_scenarios.py:named_base_scenarios` — the initial
  world: the four contacts, the five messages, the three reminders, the default
  settings row, and the `base_<column>_<on|off>…` variants that flip
  `wifi`/`location_service`/`cellular`/`low_battery_mode`.

### How milestone evaluation works

* `tool_sandbox/common/evaluation.py:Milestone` — a list of
  `SnapshotConstraint`s that must all hold on databases of the same
  `sandbox_message_index`.
* `…:Minefield` — a `Milestone` subclass for state that must **not** occur.
* `…:SnapshotConstraint` — `database_namespace`, `snapshot_constraint`
  (a similarity measure), `reference_milestone_node_index`, `target_dataframe`.
* Measures seen in the frozen 64: `snapshot_similarity` (136 constraints),
  `update_similarity` (9), `addition_similarity` (7), `removal_similarity` (6),
  and 28 `functools.partial(tool_trace_dependant_similarity, …)`.
* `…:MilestoneMatcher` / `…:Evaluation` — the milestone DAG (a linked list when
  no edge list is given) and the matching driver.  We do **not** reimplement the
  DAG ordering; see §5.

## 2. World state, and the substitutions that make it reproducible

`WorldState` models five namespaces (`crystalcall.toolsandbox.NAMESPACES`):

| namespace | contents | written by |
| --- | --- | --- |
| `clock` | one row, `{"epoch": 1711065600.0}` | nothing, ever |
| `setting` | one row, upstream's `SETTING` columns | the four `set_*` tools |
| `contact` | upstream's four contacts | `modify_contact`, `remove_contact` |
| `messaging` | upstream's five messages | `send_message_with_phone_number` |
| `reminder` | upstream's three reminders | `add_reminder`, `modify_reminder`, `remove_reminder` |

Upstream's `SANDBOX` namespace is the conversation log; OMP owns the
conversation here, so it is not modelled — no tool of the 29 reads or writes it.
Rows are plain dicts in upstream's column order and upstream's row order,
including upstream's own duplicated `message_4` payload, and `modify_*` moves
the rewritten row to the end exactly as upstream's remove-then-add does.

Six things in the upstream tool bodies are not reproducible as written.  Each is
replaced by a rule declared in `configs/toolsandbox.yml` under `world` or
`substitutions`, and by nothing else:

| upstream | why | substitution |
| --- | --- | --- |
| `datetime.datetime.now()` | a wall clock makes every episode irreproducible, and makes `get_current_timestamp` a live read | the frozen `clock` namespace, `world.epoch_unix = 1711065600.0` = `2024-03-22T00:00:00Z`, all conversions in UTC |
| `uuid4()` for new row ids | nondeterministic | `uuid5(NAMESPACE_URL, "<namespace>:<write_seq>")`, a function of the authoritative call order |
| `rapidfuzz.fuzz.WRatio` / `partial_ratio` | not a dependency | `difflib.SequenceMatcher.ratio()*100` over casefolded, punctuation-stripped, space-normalised strings; upstream's thresholds kept (90 default, 50 for `content`) |
| `pint` (`unit_conversion`) | not a dependency | the declared affine `substitutions.units` table (scale + offset to a base unit per dimension) with an alias map |
| `geopy.distance` (`calculate_lat_lon_distance`) | not a dependency | haversine, `R = 6371.0088 km` |
| `holidays.country_holidays` (`search_holiday`) | not a dependency | the declared US table for 2024 and 2025 |
| `requests.get` to RapidAPI (`search_lat_lon`, `search_location_around_lat_lon`, `search_weather_around_lat_lon`) | no key, no network, not reproducible | declared fixture tables plus declared deterministic synthesis for unlisted queries |

The state-dependency behaviour is **kept**: the externally backed tools still
raise `ConnectionError` when `wifi` is off, `get_current_location` still raises
`PermissionError` when `location_service` is off,
`send_message_with_phone_number` still raises `ConnectionError` when `cellular`
is off, `set_low_battery_mode_status(on=True)` still turns the other three off,
and `set_*` still raises `ValueError` when the setting already has that value.
That is what the `state_dependencies` family is for, and it survives intact.

## 3. Snapshot-id derivation

```
snapshot_id(namespaces) = "tsbx:" + "+".join(sorted(namespaces))
                        + "@" + sha256(canonical_json({ns: rows[ns]}))[:16]
```

`canonical_json` is the frozen `crystalcall.canonicalize.canonical_json`
(sorted keys, no whitespace, no ASCII escaping).  Two properties matter:

1. **It is the content digest of exactly the named sub-view.** A write to
   `reminder` moves `snapshot_id(("reminder",))` and
   `snapshot_id(("clock","contact","reminder"))` and leaves
   `snapshot_id(("clock",))` and `snapshot_id(("contact",))` untouched.
2. **It is resolved against the live world at canonicalisation time.**
   `ToolSpec.snapshot_resolver` is called by `canonicalize()` on every call, so
   the same tool with the same arguments has a *different* canonical identity
   before and after a write to a namespace it reads.  A future launched against
   one world version therefore cannot be claimed against another: the
   `immutable_snapshot_id` component of `FutureKey` does not match.

Speculative work additionally runs against `WorldState.freeze()` — a detached,
write-refusing deep copy taken at launch time.  So a speculative value is a
function of exactly the state its snapshot id names *by construction*, not by
timing luck: a write that lands while a speculation is in flight cannot be seen
by it. `WorldState.clone()` is the same copy without the write refusal, and is
what the replay check starts from.

## 4. Effect assignment

Derived from the checkout, declared as data, evaluated per scenario.

**Derivation (mechanical, from source).** For every `@register_as_tool`
function, walk the transitive call graph over the module-level functions of
`tool_sandbox/tools/*.py` and record:

* every `DatabaseNamespace.X` reaching `add_to_database`,
  `remove_from_database` or `update_database` → `writes`;
* every `DatabaseNamespace.X` reaching `get_database` → `reads`;
* a `datetime.datetime.now(...)` call → `reads` gains `clock`;
* a `requests.*` call → `external: true`.

An unresolvable `DatabaseNamespace` argument raises; there is no default.
`declared_class` is `write` if `writes` is non-empty, else `read` if `reads` is
non-empty, else `pure`.  This is `crystalcall.toolsandbox.derive_tool_effects`;
it produced the `tools:` table in `configs/toolsandbox.yml`, and
`verify_against_checkout()` re-derives it and reports any drift.
`tests/test_toolsandbox.py::test_declared_footprints_match_the_checkout`
asserts the report is empty, so the declaration cannot silently disagree with
its source.

**Assignment rule (`effect_for`, first match wins, total).**

| # | condition | effect |
| --- | --- | --- |
| 1 | `declared_class == write` | `Write` |
| 2 | `external` | `LiveRead` |
| 3 | `reads ∩ (union of the writes of every tool in this scenario's allow-list)` non-empty | `LiveRead` |
| 4 | `reads` non-empty and disjoint from every in-scenario write | `SnapshotRead(snapshot_id(reads))` |
| 5 | `reads` empty | `Pure` |

The rows are mutually exclusive given `declared_class`; an unknown tool raises
rather than falling through.  Rule 2 is the explicit conservative choice: an
externally backed read is not a function of any snapshot we can name, so it is
never eligible even in a scenario that writes nothing — this is the "when in
doubt, `LiveRead`" instruction applied to the three RapidAPI tools, and it holds
even though our substitution makes them locally deterministic.

Rule 3 is why eligibility is per *scenario* and not per tool: `search_reminder`
is `SnapshotRead` in a scenario that only reads reminders and `LiveRead` in a
scenario that also calls `add_reminder`.  That is the whole point — and it means
that within a single scenario an eligible snapshot read can never be invalidated
by that scenario's own writes.  The snapshot-id guard is therefore the *second*
line of defence, and it is tested directly against a world version that really
does move (`test_future_launched_before_a_write_cannot_be_claimed_after_it`).

### Per-family distribution

Counted over tool *slots* — one entry per (scenario, tool) pair, so a tool that
appears in three scenarios is counted three times, which is what determines how
much of the workload is speculable.

| family | `Pure` | `SnapshotRead` | `LiveRead` | `Write` | total | eligible |
| --- | --- | --- | --- | --- | --- | --- |
| `canonicalization` | 20 | 18 | 11 | 19 | 68 | 38 (56%) |
| `insufficient_information` | 17 | 7 | 7 | 8 | 39 | 24 (62%) |
| `multi_tool_workflows` | 0 | 18 | 7 | 7 | 32 | 18 (56%) |
| `state_dependencies` | 6 | 5 | 42 | 31 | 84 | 11 (13%) |
| **all** | **43** | **48** | **67** | **65** | **223** | **91 (41%)** |

91 of 223 scenario tool slots are speculation-eligible.  22 of the 64 scenarios
have *no* eligible tool at all: 3 `canonicalization`, 4
`insufficient_information`, 7 `multi_tool_workflows`, 8 `state_dependencies`.
The gradient is the result: the more a scenario writes, the less of it is
speculable, and `state_dependencies` — whose scenarios flip settings that every
read in the scenario depends on — is almost entirely ineligible by rule 3.

Nothing was *inferred* beyond the mechanical derivation: `reads`, `writes`,
`external` and `declared_class` all come from the source.  The two places where
judgement was exercised are stated as rules, not as per-tool exceptions:
mapping `datetime.datetime.now()` onto a read of the frozen `clock` namespace
(rule for `get_current_timestamp`, `add_reminder`, `modify_reminder`,
`send_message_with_phone_number`, `search_holiday`), and rule 2 for the three
externally backed tools.

## 5. Milestones and minefields

Extracted statically from each `ScenarioExtension`'s `milestones=[…]` and
`minefields=[…]`.  The extracted milestone count is asserted equal to the
`n_milestones` the manifest froze, for all 64 scenarios, at load time.

Target-dataframe cell values are evaluated by a *closed* evaluator: literals,
`RoleType.X`, `deterministic_uuid(payload=…)`, `json.dumps(<literal>)`,
`datetime.datetime(…).timestamp()`, and the two upstream scenario helpers
`get_tomorrow_datetime()` / `get_next_iso_weekday_datetime(next_iso_weekday=N)`
(evaluated against the frozen epoch).  Anything else is recorded as
`{"unresolved": "<source>"}` and never guessed.

| constraint shape | our evaluation |
| --- | --- |
| `SANDBOX` target with `tool_trace` | satisfied iff an authoritative call to the named tool exists whose canonical arguments agree on every declared key (declared `None` means "absent or null") |
| `SANDBOX` `AGENT -> USER` target with `content` | ROUGE-L F1 of the declared text against the concatenated assistant text, threshold `milestones.rouge_l_threshold = 0.4` (ours, not upstream's) |
| `addition_similarity` / `update_similarity` with a target | a row matching every declared literal column exists in the final namespace (booleans exact, timestamps within `numeric_tolerance = 1.0 s`, strings fuzzy at 90) |
| `update_similarity` with no target | that namespace differs from the initial world |
| `removal_similarity` | no matching row exists in the final namespace |
| `partial(tool_trace_dependant_similarity, …)` | **indeterminate** — reproducing it needs upstream's `days_extractor` / `result_to_timestamp*_extractor` / `*_temperature_extractor` closures, which we do not reimplement |
| `SANDBOX` `AGENT -> EXECUTION_ENVIRONMENT` target with a bare tool name (minefields) | hit iff the transcript contains a call to that tool |

Two documented weakenings, both recorded per episode:

* **Milestone DAG ordering is not enforced.** Upstream matches milestones
  against snapshots in DAG order; we evaluate each milestone against the final
  world plus the full transcript. Recorded as
  `milestone_order_enforced: false` on every derived row
  (`milestones.order_enforced` in the config).
* **`tool_trace_dependant_similarity` milestones are indeterminate**, counted in
  `milestones_indeterminate` and surfaced in `exclusion`.

`evaluate_milestones(scenario, world, transcript)` is a pure function: it reads
nothing but its three arguments, mutates none of them, and touches no clock,
filesystem or global (asserted by `test_milestone_evaluation_is_pure`).

## 6. Multi-turn structure

`user_turns` are the `USER -> AGENT` message contents of the
`ScenarioExtension`, in declaration order.  All 64 scenarios declare exactly
one.  Upstream generates follow-up user turns with a **live user-simulator
LLM** reading the `SYSTEM -> USER` instruction; we drive OMP, not their agent
loop, so no simulator runs here.

For the 15 scenarios tagged `MULTIPLE_USER_TURN` we append exactly **one**
scripted follow-up, derived from the same source the simulator would read: the
task text of that `SYSTEM -> USER` instruction, with the `USER_INSTRUCTION`
preamble dropped (it is a role-play instruction, not user content), the declared
`user_turns.strip_prefixes` removed (`"Ask User B to "`, …), and any sentence
matching `user_turns.drop_sentences` removed (`"You do not have any more
information"`, `"Do not leak this information"`, …).  No other turn is ever
synthesised.  The rule and its pattern lists are in `configs/toolsandbox.yml`;
`test_scripted_follow_up_only_for_declared_multi_user_turn` asserts the
follow-up appears for exactly those 15 scenarios and nowhere else.

So: 49 scenarios run one turn, 15 run two.

## 7. The losslessness check — one generation, replayed

**It is deliberately *not* two generations.**
DiffusionGemma's sampler renoises the canvas with Gumbel noise and ignores the
request temperature, so a pinned server `--seed` does not
make two sequential requests identical.  Generating twice and diffing would
confound model stochasticity with scheduling: it could report a "semantic
divergence" that is really sampling noise, or mask a real one behind it.

The check therefore holds the generation fixed:

1. **Generate once.** The episode is driven through OMP exactly once — with the
   speculative launch gate on for `toolsandbox_crystalcall`, off for
   `toolsandbox_baseline`.  `HostToolRouter.calls()` yields the authoritative
   canonical call sequence in execution order, with
   `AuthoritativeCall.call.identity`, `.args_dict`, `.result_digest` and
   `.claimed_speculative`, plus the full transcript.
2. **Replay.** `replay_episode()` takes `WorldState.clone()` of the episode's
   *initial* world and re-executes exactly that recorded canonical call
   sequence, in order, through the same tool implementations, with
   `futures=None` — no future table, no claims, ordinary scheduling — and
   re-derives each result digest with the same function `HostToolRouter` uses.
3. **Compare** (`compare_replay()`): per-call `result_digest` in execution order
   together with tool and canonical arguments, per-call `error` and
   `canonicalization_error`, the initial and final `WorldState.digest()`, and
   `evaluate_milestones(...)`.  `claimed_speculative` is excluded (the replay
   claims nothing by construction) and so is `snapshot_id` (a content digest of
   the world at the moment of the call, implied by reproducing the world).

Because both sides consume the *same* generation, a mismatch can only come from
scheduling — which is exactly the invariant under test.  Any mismatch is
recorded in `semantic_divergence` / `divergent_fields` on the derived row, in a
`toolsandbox_divergence` trace record with the full detail, counted in the run
summary, and makes the process exit non-zero.  It is never dropped.

`--stub` uses the identical code path, so the code under test is the same on GPU
and off.

**The check has teeth.** `replay_episode(..., corrupt=…)` is a test hook that
rewrites one recorded call before dispatch.
`test_a_corrupted_replay_is_detected_as_divergent` flips one argument of the
first `Write` in two scenarios and asserts the verdict is divergent in both
`tool_trace` and `world_state_digest_final`;
`test_a_corrupted_read_result_is_detected` perturbs a read's argument and
asserts `tool_trace` diverges; `test_an_untouched_replay_is_bit_identical`
asserts two uncorrupted replays are byte-identical, so those tests cannot be
passing merely because the replay is noisy.

## 8. Speculation policy

Declared in `configs/toolsandbox.yml` under `speculation`; policy id
`turn_boundary_eligible_prefetch`.  At each turn boundary, against the world
version current at that instant, the arm launches

* (a) every speculation-eligible tool of the scenario surface whose canonical
  form needs no supplied argument, and
* (b) every speculation-eligible canonical call observed authoritatively in the
  previous turn,

capped at `max_launches_per_turn = 16`, all against `world.freeze()`.  Writes and
live reads are never launched: `FutureTable.launch` refuses them and
`assert_no_write_speculation(call)` is asserted at the launch site.

The `request_id` component of the claim key carries the **turn index**
(`<arm>/<scenario>/s<seed>/<mode>/t<turn>`), so a future launched in one turn is
structurally unclaimable in any other turn — the first of the four key
components doing exactly the job the invariant asks of it.  Unclaimed futures
are discarded per turn.

## 9. Exact commands

GPU-free verification (no model, no server, fully deterministic):

```bash
uv run --project . python -m pytest tests/test_toolsandbox.py -q
uv run scripts/run_toolsandbox.py --stub --limit 6 --seed 0
uv run scripts/run_toolsandbox.py --stub --arm toolsandbox_baseline    --seed 0
uv run scripts/run_toolsandbox.py --stub --arm toolsandbox_crystalcall --seed 0
```

The real arm, **which needs the GPU**.  It does not start or stop the model
server; start the server first, then run the six invocations:

```bash
# 1. serve DiffusionGemma (needs the GPU free)
bash scripts/serve_diffusiongemma.sh

# 2. the arm: 2 arms x 3 seeds x 64 scenarios = 384 episodes
for arm in toolsandbox_baseline toolsandbox_crystalcall; do
  for seed in 0 1 2; do
    uv run scripts/run_toolsandbox.py --arm "$arm" --seed "$seed" || exit 1
  done
done
```

Flags: `--manifest` (default `experiments/manifests/toolsandbox.jsonl`), `--arm`
(`toolsandbox_baseline` | `toolsandbox_crystalcall`, default the latter),
`--seed` (default 0), `--limit` (0 = all 64), `--resume` (skip episodes already
present in the derived rows; shards are append-only, so a resumed run writes a
new `…partN.jsonl` shard), `--stub`, `--stub-port` (default 8000, must match the
`vllm` provider base URL in the installed OMP profile), `--server-url` (default
`http://127.0.0.1:8000/v1`), `--model`, `--out`, `--derived`, `--turn-timeout`.

Exit codes: `0` = every episode ran and `semantic_divergence == 0`; `1` = ran but
at least one episode errored or diverged; `2` = usage/config error.

## 10. Stub-mode evidence

`--stub` replaces the model with `crystalcall.toolsandbox.StubModelServer`, an
in-process OpenAI-compatible server (the technique of
`scripts/smoke_omp_ptc.py`).  Per user turn it answers once with an `eval` tool
call carrying that turn's program and once with a final text.

The program is derived from the scenario's own milestone declarations, not
invented: each `tool_trace` target names a tool and its arguments; each
state-namespace constraint names a mutation, which `stub.state_tools` (and
`stub.setting_column_tools` for the boolean settings) maps back onto the tool
that performs it; any required argument a milestone leaves open is filled from
`stub.required_args`; row ids a milestone leaves open are resolved at run time
inside the OMP kernel by the scenario's own search tools.  Each turn's program
is bracketed by the scenario's argument-free eligible reads, so every turn
genuinely interleaves writes, live reads and eligible snapshot reads.  The
program ends by printing `RESULTDIGEST=<sha256 over every observed tool
result>`, and the stub's final assistant text echoes that digest — so the
assistant text is a function of the tool results, not a constant.

Full sweep, 2 arms x 3 seeds x 64 scenarios:

| metric | value |
| --- | --- |
| episodes | 384 (192 per arm) |
| turn rows | 474 |
| **semantic divergences** | **0** |
| episode errors | 0 |
| authoritative tool calls | 1158 (258 `Write`, 186 `LiveRead`, 618 `SnapshotRead`, 96 `Pure`, 18 raising a declared state-dependency error) |
| speculative launches / claimed / wasted (`toolsandbox_crystalcall`) | 195 / 105 / 90 |
| claimed authoritative calls (`toolsandbox_crystalcall`) | 105 |
| claimed authoritative calls (`toolsandbox_baseline`) | 0 |
| `Write` or `LiveRead` calls ever claimed | 0 |
| milestones hit | 600 / 978 (126 indeterminate) |
| minefields hit | 0 |
| episodes marked `success` | 132 / 384 |
| trace shards | 6, 2022 hash-chained records, all verified |

`shard_manifest("experiments/raw/toolsandbox")` verifies all six shards; the
runner also verifies its own shard before exiting and records the digest in
`reports/toolsandbox-<arm>-seed<N>.json`.  The OMP RPC frame log is written to
`experiments/raw/toolsandbox/logs/` and deliberately *not* alongside the shards,
because `trace.shard_manifest` verifies every `*.jsonl` it finds in that
directory and the frame log is not a hash-chained shard.

Milestone hit rate is *not* a model result here: the stub agent is a fixed
program derived from the milestones, so hits measure whether our milestone
evaluation can see the mutation the milestone declares, and misses are mostly
`AGENT -> USER` text milestones that the stub's `observed <digest>` reply cannot
satisfy.  The number this arm reports is the divergence count.

## 11. Exclusions

No scenario is excluded from the run.  10 of the 64 carry an `exclusion` tag on
every derived row because at least one of their milestones is *indeterminate*
under our evaluation — they use
`partial(tool_trace_dependant_similarity, …)`, whose upstream extractor
closures we do not reimplement.  Their divergence check is unaffected (the
milestone evaluation is compared between the generation and its replay either
way); only their milestone score is not comparable to upstream's.

| scenario | family | indeterminate / total milestones |
| --- | --- | --- |
| `find_days_till_holiday_alt` | canonicalization | 3 / 4 |
| `search_reminder_with_recency_upcoming_implicit` | canonicalization | 1 / 3 |
| `find_current_city_low_battery_mode_alt` | state_dependencies | 1 / 6 |
| `find_days_till_holiday_wifi_off` | state_dependencies | 3 / 5 |
| `find_days_till_holiday_wifi_off_alt` | state_dependencies | 3 / 5 |
| `find_days_till_holiday_wifi_off_multiple_user_turn` | state_dependencies | 3 / 5 |
| `find_temperature_f_with_location_and_time_diff_low_battery_mode_multiple_user_turn` | state_dependencies | 2 / 5 |
| `find_temperature_f_with_location_wifi_off` | state_dependencies | 2 / 4 |
| `find_temperature_f_with_location_wifi_off_alt` | state_dependencies | 2 / 4 |
| `find_temperature_low_battery_mode_alt` | state_dependencies | 1 / 5 |

Other things deliberately left out, each recorded rather than hidden:

* the 5 upstream tools no frozen scenario's allow-list uses
  (`add_contact`, `convert_currency`, `search_stock`,
  `seconds_to_hours_minutes_seconds`, `end_conversation`) — not declared, not
  registered;
* upstream's user-simulator LLM (§6) and its milestone DAG ordering (§5);
* upstream's own harness, roles and `Scenario.play` loop — we drive OMP.

## 12. Derived record schemas

Both files carry `schema_version: "crystalcall-toolsandbox/1"`.

`toolsandbox_episodes.jsonl`, keyed `(arm, seed, scenario_id)`:
`schema_version`, `scenario_id`, `family`, `seed`, `arm`, `speculate`, `stub`,
`milestones_total`, `milestones_hit`, `milestones_indeterminate`,
`milestone_order_enforced`, `minefields_total`, `minefields_hit`, `success`,
`turns`, `wall_ms`, `world_state_digest_final`, `world_state_digest_replay`,
`world_state_digest_initial`, `replay_calls`, `speculative_launches`,
`speculative_claimed`, `speculative_wasted`, `authoritative_calls`,
`claimed_calls`, `eligible_tools`, `effects`, `semantic_divergence`,
`divergent_fields`, `divergence_method`, `exclusion`, `notes`.

`toolsandbox_turns.jsonl`, keyed `(arm, seed, scenario_id, turn_index)`:
`schema_version`, `scenario_id`, `family`, `seed`, `arm`, `turn_index`, `user`,
`assistant`, `calls`, `claimed_calls`, `write_calls`, `live_read_calls`,
`snapshot_read_calls`, `pure_calls`, `failed_calls`, `tools`, `snapshot_ids`,
`world_digest_after`, `wall_ms`, `error`.

`crystalcall.aggregate.DERIVED_SOURCES` lists `toolsandbox_episodes.jsonl`
(it feeds S6); `toolsandbox_turns.jsonl` is not aggregated.

Trace shard record kinds, in order per episode:
`toolsandbox_episode_start`, one `toolsandbox_turn` per user turn,
`toolsandbox_replay`, `toolsandbox_divergence`, `toolsandbox_episode_end`.
