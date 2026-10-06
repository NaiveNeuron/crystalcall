<!-- GENERATED FILE.  Rebuild with `uv run scripts/build_manifests.py --out-dir experiments/manifests --vendor-dir experiments/vendor --seed 20260903`.  Do not edit by hand. -->
# CrystalCall prompt manifests

Every number in this document is computed by `scripts/build_manifests.py` from
the pinned upstream checkouts listed below; nothing is hand entered.  Rebuild
and verify with:

```sh
uv run scripts/build_manifests.py --out-dir experiments/manifests --vendor-dir experiments/vendor --seed 20260903
uv run scripts/build_manifests.py --verify
```

## 1. Upstream sources

`experiments/vendor/` is gitignored; `experiments/manifests/vendor_sources.json`
is the committed provenance record (repo, commit, archive sha256, per-file
sha256 of every input read).  This host's egress answers GitHub's
`POST /…/git-upload-pack` with HTTP 401, so anonymous `git clone` fails; the
build fetches the *same commit* as a codeload archive and verifies its sha256.

```sh
# bfcl: https://github.com/ShishirPatil/gorilla @ 6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
git clone --depth 1 https://github.com/ShishirPatil/gorilla          # canonical form
curl -sSL -o gorilla.tar.gz https://codeload.github.com/ShishirPatil/gorilla/tar.gz/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8
# archive sha256 c6b5081337cabd317b56c6eead2ec735f7a3cb86ac4fd664e3f9c8c02d1f1f1e
# keep: gorilla-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard/bfcl_eval/data -> experiments/vendor/bfcl/data
# toolsandbox: https://github.com/apple/ToolSandbox @ 165848b9a78cead7ca7fe7c89c688b58e6501219
git clone --depth 1 https://github.com/apple/ToolSandbox          # canonical form
curl -sSL -o ToolSandbox.tar.gz https://codeload.github.com/apple/ToolSandbox/tar.gz/165848b9a78cead7ca7fe7c89c688b58e6501219
# archive sha256 0019647ab9a52348680a1e33f99c5db9c32b9513bcd78fed1af9c4beb804427e
# keep: ToolSandbox-165848b9a78cead7ca7fe7c89c688b58e6501219 -> experiments/vendor/toolsandbox/checkout
```

| repo | commit | rule file | rules sha256 |
|---|---|---|---|
| https://github.com/ShishirPatil/gorilla | `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8` | `configs/manifest_rules.yml` | `90e531c1bc1a46bd754e7e9bae817867aeacdfed53d173a47dd983108f4878ef` |
| https://github.com/apple/ToolSandbox | `165848b9a78cead7ca7fe7c89c688b58e6501219` | `configs/manifest_rules.yml` | `90e531c1bc1a46bd754e7e9bae817867aeacdfed53d173a47dd983108f4878ef` |

Every manifest header carries
`vendor_provenance_sha256: a4a02e66a2d07ddee194789372fcc0d7acbc39fb9301ee2cba4e12e252d779ec`, a digest over
the pinned fields (`key, repo, commit, archive_url, archive_sha256, archive_member, install_path`) of the
`bfcl, toolsandbox` entries plus every `input_files_sha256` entry —
not over the whole shared file, so a source another task vendors cannot
invalidate these manifests while any change to what this build reads does.
Before every build, each declared input file is re-hashed and compared against
`input_files_sha256`; a mismatch aborts with the fetch commands.

## 2. BFCL categories and counts

BFCL v4 splits `simple` by language; CrystalCall speculates Python programmatic
tool calls, so the Python split is used.  Ground truth comes from the matching
file under `possible_answer/`.

| category | upstream file | upstream prompts | candidates | dropped | dev | test |
|---|---|---|---|---|---|---|
| multiple | BFCL_v4_multiple.json | 200 | 199 | 1 | 25 | 75 |
| parallel | BFCL_v4_parallel.json | 200 | 200 | 0 | 25 | 75 |
| parallel_multiple | BFCL_v4_parallel_multiple.json | 200 | 193 | 7 | 25 | 75 |
| simple | BFCL_v4_simple_python.json | 400 | 398 | 2 | 25 | 75 |
| **total** |  | 1000 | 990 | 10 | 100 | 300 |

Selection (declared in `configs/manifest_rules.yml`, seed `20260903`):
per category, in sorted category order, candidate prompt ids are sorted
lexicographically and `random.Random(20260903)` draws
`dev_per_category + test_per_category` of them; the first
25 draws go to dev, the remaining
75 to test.  Dev and test are disjoint by
construction, and records are written sorted by `(category, prompt_id)`.

Expected calls in the selected 400 prompts: **779**.
Optional parameters omitted because BFCL's ground truth admits omission
(`""` among the alternatives, which is exactly how BFCL's own AST checker reads
it): **275**.  Upstream schema defaults dropped
because they do not type-check against the declared `ParamKind`:
**8** (recorded per record in
`meta.dropped_defaults`).

## 3. Tool-name sanitisation

OMP's programmatic tool proxy is attribute based (`tool.<name>(…)`) and the PTC
extractor only accepts an identifier callee, so a dotted upstream name such as
`math.gcd` is unreachable: `tool.math` resolves first and fails.  Every tool
name is therefore sanitised to a Python identifier with
`re.sub(r"[^0-9A-Za-z_]", "_", name)`,
prefixed with `t_` when it would start with a
digit, and disambiguated with `_2`, `_3`, … over the sorted upstream names when
sanitisation collides inside one prompt.

The upstream spelling is preserved verbatim in `ToolDecl.original_name` and in
`meta.tool_name_map` (`sanitised -> upstream`), and is registered as a registry
**alias** of the sanitised name, so a model that emits the dotted spelling
canonicalises to exactly the same identity.  An alias that would collide with
another tool's canonical name or alias in the same prompt is skipped
(`EffectRegistry.register` rejects duplicate identities) and counted in
`meta.skipped_aliases`.

| quantity | count |
|---|---|
| tool declarations in the selected 400 prompts | 739 |
| declarations whose name was sanitised | 409 |
| distinct upstream names sanitised | 290 |
| prompts containing at least one sanitised name | 253 |
| names needing a `_2`/`_3` collision suffix | 0 |
| upstream aliases skipped as colliding | 0 |

`expected_calls[*]["tool"]` always uses the sanitised name, so
`canonicalize(registry_for(rec), c["tool"], c["args"])` names the tool the model
will actually call.

## 4. Effect classification

Declared before any model run, stored as data in `configs/manifest_rules.yml`,
applied as a case-insensitive regex search over the **original upstream**
function name (`effects.classify_on: original_name`, never the
sanitised identifier) with first-match-wins order `Write` -> `LiveRead` ->
`Pure`:

```yaml
write_regex: '(?i)(send|post|create|delete|remove|update|set_|book|order|purchase|buy|pay|transfer|cancel|schedule|reserve|add_|insert|write|upload|submit|register|reset|move|rename|start|stop|launch|execute|run_|install|subscribe|unsubscribe|login|logout|sign|toggle|enable|disable|mute|archive|restore|share|invite|assign|apply|save)'
live_read_regex: '(?i)(current|live|now|realtime|real_time|today|latest|weather|traffic|stock_price|price|quote|exchange_rate|news|trending|status|available|nearby|location)'
fallback: Pure
```

Distribution over the declared tool surface of the **selected** 400 prompts
(one count per prompt-tool declaration, so a tool declared in several prompts
counts several times):

| category | Pure | LiveRead | Write | SnapshotRead | total |
|---|---|---|---|---|---|
| multiple | 236 | 26 | 18 | 0 | 280 |
| parallel | 85 | 7 | 8 | 0 | 100 |
| parallel_multiple | 219 | 19 | 21 | 0 | 259 |
| simple | 86 | 10 | 4 | 0 | 100 |
| **total** | 626 | 62 | 51 | 0 | 739 |

Distribution over the **ground-truth calls** of the selected 400 prompts:

| category | Pure | LiveRead | Write | SnapshotRead | total |
|---|---|---|---|---|---|
| multiple | 86 | 7 | 7 | 0 | 100 |
| parallel | 231 | 28 | 18 | 0 | 277 |
| parallel_multiple | 258 | 22 | 22 | 0 | 302 |
| simple | 86 | 10 | 4 | 0 | 100 |
| **total** | 661 | 67 | 51 | 0 | 779 |

For reference, over the full candidate pool
(990 prompts), by declared tool:

| category | Pure | LiveRead | Write | SnapshotRead | total |
|---|---|---|---|---|---|
| multiple | 462 | 52 | 39 | 0 | 553 |
| parallel | 167 | 20 | 13 | 0 | 200 |
| parallel_multiple | 415 | 42 | 46 | 0 | 503 |
| simple | 340 | 35 | 23 | 0 | 398 |
| **total** | 1384 | 149 | 121 | 0 | 1654 |

## 5. Drop reasons

Hard filters, all declared in `configs/manifest_rules.yml`: at least one
declared tool, a non-empty single user message, at least one ground-truth call,
every argument value a statically closed literal (a string that parses as a
call expression counts as a nested call), every declared parameter type
supported by `ParamKind`, and every expected call canonicalisable through
`canonicalize(registry_for(rec), call["tool"], call["args"])`.  A prompt that
fails any filter is excluded at build time.

| reason | multiple | parallel | parallel_multiple | simple | total |
|---|---|---|---|---|---|
| canonicalize:type_mismatch | 0 | 0 | 1 | 1 | 2 |
| canonicalize:unknown_parameter | 0 | 0 | 2 | 0 | 2 |
| non_literal_argument | 0 | 0 | 2 | 0 | 2 |
| unsupported_param_type | 1 | 0 | 2 | 1 | 4 |
| **total** | 1 | 0 | 7 | 2 | 10 |

Reason glossary:

- `unsupported_param_type` — a declared parameter uses a JSON-schema type with
  no `ParamKind` (BFCL's `any`), so the tool surface cannot be typed.
- `canonicalize:type_mismatch` — a ground-truth value contradicts the declared
  schema type (e.g. the string `"false"` for a `boolean` parameter).
- `canonicalize:unknown_parameter` — the ground truth names a parameter the
  function schema does not declare.
- `non_literal_argument` — a ground-truth argument is a call expression rather
  than a closed literal.

Every excluded prompt:

- `multiple_181` (multiple): `unsupported_param_type` — data:any
- `parallel_multiple_12` (parallel_multiple): `canonicalize:unknown_parameter` — calculate_voltage_difference:permeability
- `parallel_multiple_19` (parallel_multiple): `non_literal_argument` — call_expression:exp(-x**2)
- `parallel_multiple_194` (parallel_multiple): `unsupported_param_type` — data:any
- `parallel_multiple_21` (parallel_multiple): `canonicalize:type_mismatch` — linear_regression_fit:expected list, got "data['sales']"
- `parallel_multiple_26` (parallel_multiple): `canonicalize:unknown_parameter` — bank_calculate_balance:type
- `parallel_multiple_57` (parallel_multiple): `unsupported_param_type` — date:any
- `parallel_multiple_94` (parallel_multiple): `non_literal_argument` — call_expression:startswith(b)
- `simple_python_109` (simple): `unsupported_param_type` — data:any
- `simple_python_200` (simple): `canonicalize:type_mismatch` — calculate_emissions:expected number, got str

## 6. ToolSandbox preselection

ToolSandbox scenarios are stateful sandbox programs, so the manifest pins
*which* scenarios run rather than a static tool surface.  The registry is read
statically with `ast` from the checkout
(`tool_sandbox/scenarios/*.py`); ToolSandbox is never imported or executed.
Every scenario is a `ScenarioExtension(name=…)` literal in one of the four
modules aggregated by `tool_sandbox.scenarios.named_scenarios`;
**129** base scenarios are declared
(insufficient_information_scenarios.py: 28, multiple_tool_call_scenarios.py: 54, multiple_user_turn_scenarios.py: 28, single_tool_call_scenarios.py: 19).

**Family rule.**  A scenario's family is derived from its own declared
`ScenarioCategories` tags — the `categories=[…]` literal of its
`ScenarioExtension` plus the defaults its module's `named_*` builder appends to
every extension it defines (e.g. `named_multiple_tool_call_scenarios` appends
`MULTIPLE_TOOL_CALL, SINGLE_USER_TURN`).  The first matching row of
`toolsandbox.family_precedence` wins, so families are disjoint and the rule is
most-specific-first:

| family | declared tag | base scenarios | selected |
|---|---|---|---|
| insufficient_information | `INSUFFICIENT_INFORMATION` | 28 | 16 |
| state_dependencies | `STATE_DEPENDENCY` | 24 | 16 |
| canonicalization | `CANONICALIZATION` | 48 | 16 |
| multi_tool_workflows | `MULTIPLE_TOOL_CALL` | 14 | 16 |
| **total** |  | 114 | 64 |

15 base scenarios carry none of the four
tags (single-tool, single-turn, no special phenomenon) and are therefore never
selected.

Because `canonicalization` and `multi_tool_workflows` overlap heavily upstream,
16 distinct *base* scenarios are not available for every family
(`multi_tool_workflows` has 14
after precedence).  The quota is therefore filled from the tool-count variants
that `named_scenarios()` itself derives — base (`NO_DISTRACTION_TOOLS`),
`_3_distraction_tools`, `_10_distraction_tools`, `_all_tools` — walked
outermost, so every distinct base scenario of a family is used before any
second variant.  The four `*_scrambled` variants are excluded: they perturb the
tool representation, which must stay fixed across CrystalCall arms.  Selection
uses `random.Random(20260903)` over the sorted base names and depends on
no model outcome.

| variant | selected |
|---|---|
| NO_DISTRACTION_TOOLS | 62 |
| THREE_DISTRACTION_TOOLS | 2 |

Distinct base scenarios among the 64
selected: **62**.

## 7. Artefacts

| file | contents |
|---|---|
| `experiments/manifests/dev.jsonl` | header + 100 prompt records |
| `experiments/manifests/test.jsonl` | header + 300 prompt records |
| `experiments/manifests/toolsandbox.jsonl` | header + 64 scenario records |
| `experiments/manifests/*.sha256` | `sha256sum`-format sidecars |
| `experiments/manifests/build_stats.json` | every number in this document |
| `experiments/manifests/vendor_sources.json` | pinned commits, fetch commands, input hashes |

The first line of each manifest is a header record carrying schema
`crystalcall-manifest/1`, the upstream commit, the rule-file sha256, the seed, the
per-category counts and the build command.

| artefact | sha256 |
|---|---|
| `configs/manifest_rules.yml` | `90e531c1bc1a46bd754e7e9bae817867aeacdfed53d173a47dd983108f4878ef` |
| `dev.jsonl` | `bcac80d20775dddbbf667b29e81a5c8765545b6f4ff19abdcbab10d36f4a4188` |
| `test.jsonl` | `16b0c799263b711486afcaa2b1a0d7929fd1cc5bcd8d0b3e162caf02834e7574` |
| `toolsandbox.jsonl` | `e5f0214f234bc123f0d711f0f54f5e9a07cc07c5ee04f49cf73c68f2e991ad90` |
