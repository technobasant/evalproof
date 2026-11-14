# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Breaking

- The case fields `turns`, `state` and `expect` now have a meaning (the user turns that follow
  `input`, the initial session state, and the inline checks of the `checks` scorer) and are
  validated when the dataset loads: `turns` must be a list of strings or objects, `state` an
  object, `expect` a list of valid checks. A dataset that uses one of these names for something
  else fails to load with a message that says to rename the field.

### Added

- A `checks` scorer asserts on what an agent did: the tools it called and with which arguments
  (`tool_called`, `tool_not_called`, `tool_args_contain`, `forbidden_arg_values`, `tool_order`,
  `tool_succeeded`, `tool_call_count`, `rounds`), and what it said (`output_contains`,
  `output_omits`, `output_regex`, `output_cites_tool_result`), with `any_of`, `not` and `python`
  checks to combine them. Checks come from the suite (`params.checks`) and from each case
  (`expect`), are validated when the suite or the dataset loads, and are deterministic, so
  `evalproof verify` re-scores them.
- A `python` scorer calls a function of your own, `fn: module:function`, which returns a bool, a
  number from 0 to 1, a `(bool, reason)` pair or a `Score`; an exception or any other return
  value leaves the case undecided. `verify` re-scores it only when the scorer sets
  `deterministic: true`.
- `evalproof.adapters` turns what Google ADK, LangChain and LangGraph, the Claude Agent SDK and
  the OpenAI Agents SDK return into the trace a python system hands back, and `TraceRecorder`
  with `mcp_call` builds one by hand. No SDK is imported; tokens and cost roll up from the trace.
- Suite `variants` run the same cases, scorers and gate against another model, prompt or flag:
  each has a `name`, optional `vars` and an optional `system` merged over the suite's.
  `evalproof run --variant NAME` (repeatable) or `--all-variants` writes `runs/<variant>.json`,
  and the run file records the variant. A python system whose function has a `ctx` parameter gets
  a `RunContext` (`case_id`, `input`, `turns`, `state`, `variant`, `vars`, `trial`).
- Multi-turn cases run on every target that can hold a conversation: a python system gets the
  later turns as `ctx.turns`, an http system gets `turns`, `state`, `variant` and `vars` in the
  POST body, and a direct model target plays the turns in order, each after its previous reply.
  A coding-agent CLI takes one prompt per call, so a suite with `turns` cases on one is refused
  before any case runs.
- `evalproof readiness SUITE [--run FILE] [--json] [--require dim,...]` reads a suite and its
  dataset, and optionally a run file, and reports for each dimension of quality (correctness,
  behaviour, quality, regression gate, reliability, cost and latency, safety, robustness,
  multi-turn, load, production monitoring) whether it is covered, partly covered, missing or out
  of scope, with the evidence. It makes no model call. It exits 1 when a `--require`d dimension
  is not covered, and `--json` is for CI.
- `examples/agent-tools`, an offline agent example: a plain-Python agent with two fake tools and
  no model, 35 cases with inline `expect` checks (three multi-turn, five tagged `safety`, five
  `robustness`), and two variants. `v1` plants a bug that skips a tool for one intent, and only
  the `checks` scorer sees it, since the reply still reads well. The committed run files in
  `runs/` gate `v1` to `v2` with exit 0 and `v2` to `v1` with exit 1, and `evalproof readiness`
  reports its gaps. CI runs both directions and the readiness `--require` on every commit.
- The README has a "Testing agents" section: the trace contract (what a python system returns,
  the `ToolCall` fields and `ctx`), a table of every check with what it needs and when it passes,
  multi-turn cases (`turns`, `state`), variants (`--variant`, `--all-variants`), the example, and
  what `readiness` reports. "Agent SDK adapters" is now a subsection of it.
- Credentials are stated plainly. The README has a "What it needs to run" section, a table of
  each target, the variable it reads, whether it works locally and in CI, and where it is allowed.
  `evalproof init` prints what the chosen judge needs, locally and in CI. `evalproof init --ci
  api-key --cli-auth oauth-token` writes a workflow that passes `CLAUDE_CODE_OAUTH_TOKEN` from a
  repository secret, for Claude Code only and for your own repository and CI only (a usage error
  for Codex or any other target). `evalproof doctor` prints, for each target, the credential it
  would use and whether that works in CI; it checks variable names only and never reads a value.

## [0.5.2] — 2026-11-14

No behaviour changes: documentation, and `evalproof init` now pins 0.5.2 in the workflows it writes.

### Documentation

- MEASUREMENTS §9 calibrates the support-qa judge against reviewed labels for a random 30
  baseline cases (`examples/support-qa/human-review/`), next to the answer key on the same
  cases; `measure.py` prints the tables (#4). The README install section and a new use-cases
  section were rewritten, and PyPI shows the new README from this release.
- `CITATION.cff` gives the current version again; it had stayed at 0.3.0.

## [0.5.1] — 2026-11-13

Upgrading: run files written by 0.5.0 keep working. They have no tag-free hash, so `compare`
pairs them on the per-case hash as before, tags included; a run written by 0.5.1 pairs on the
tag-free hash. The first time both the baseline and the candidate come from 0.5.1, adding a tag
stops counting as an edit.

### Fixed

- Adding or changing a case's tags no longer counts as an edit in `compare`. Run files carry a
  new per-case `content_hash`, the case hash without its tags, and cases are paired on it when
  both runs have one (otherwise on `case_hash`, as in 0.5.0). Before, adding a `must_pass` tag
  excluded that case from the comparison until the baseline was refreshed. `verify` still counts
  a tag change, because `must_pass` reads tags from the run file, and its message now says so.

### Added

- `evalproof verify` scores the stored outputs again with the suite's scorers that need no
  model (`exact_match`, `contains`, `regex`, `json_schema`, `numeric_tolerance`, `recall_at_k`,
  `mrr`, `ndcg`) and fails, naming the cases, when a stored score differs: a score edited by
  hand, or made by another scorer version. It says which scorers it re-scored, or that none
  could be re-scored without a model call. Model-judged scores and the stored outputs
  themselves still cannot be checked.
- `evalproof rehash RUN SUITE [--out FILE]`: adds per-case hashes to a run file written before
  0.5, with no model call, so it can be gated against a changed dataset. It works when the
  cases the run scored are still the first ones in the suite's dataset, unchanged, which the
  `dataset_hash` stored in the run file proves: the dataset the run scored, the same with cases
  appended, or the dataset behind a `--limit` run. Otherwise it exits 4 and writes nothing. Scores, outputs and the suite hash
  are untouched, so `verify` still wants a run made with 0.5 or newer. The gate's exit-4 message
  for a baseline without hashes names the command.

## [0.5.0] — 2026-11-13

Breaking, in the order a user is likely to hit them:

- Unknown keys in a suite are now errors (`extra="forbid"` on every config model). A typo such
  as `gate.max_drops` used to be ignored and now fails, naming the key and the closest valid one.
- `evalproof run` exits 2 when every case errored. It still writes the run file and prints the
  report first. Before, it exited 0.
- The cache keys of agent-CLI calls (`claude-code`, `codex`, `opencode`, `cursor`, `copilot`)
  change once: the CLI version is no longer part of the key, so a CLI that updates itself keeps
  its cached replies. Cached replies from 0.4.x are not reused; `--no-cache` always asks again.
- `evalproof compare --metric NAME` exits 2 when neither run has a scorer called `NAME`.
  Before, it exited 4.
- `compare` no longer exits 4 because the dataset changed, when both runs carry per-case hashes
  (runs written by 0.5 and later do). Edited cases are excluded and listed instead.
- The action, given a `suite`, runs `evalproof verify` on the candidate first and fails with
  exit 4 when the committed run does not match the suite and its cases. A run file written by
  0.4.x has no suite fingerprint and fails it: re-run the suite with 0.5 and commit the file,
  or set `verify-candidate: "false"`.
- A cost budget can trip where it did not: a `claude-code` system or judge with no `pricing`
  now records the CLI's reported `total_cost_usd`, and the judge's cost counts toward
  `max_cost_usd`, so a suite with `gate.max_cost_usd` can newly exit 3.

Upgrading: run the suite with 0.5 on the base branch and commit `evals/runs/latest.json` there
first. A 0.4.x baseline has no per-case hashes, so a pull request that also changes the dataset
cannot be gated (exit 4) until the base branch has a 0.5 run file. Tags are part of a case's
hash, so adding a `must_pass` tag to a case counts as an edit until the baseline is refreshed.

### Added

- Per-case `case_hash` in run files (runner and importers). `compare` pairs cases by id and
  hash: a new case is not compared, an edited case is excluded and listed, so adding or editing
  cases no longer blocks the gate. A `--limit` run pairs with a full baseline.
- `evalproof verify RUN SUITE [--markdown FILE]`: no model call; checks that a run file was
  made with the suite's system, judge, scorers and seed (`RunInfo.suite_hash`) and covers
  exactly the dataset's cases. Exit 0 on a match, 4 on a mismatch. The action runs it before
  comparing (input `verify-candidate`, default `"true"`).
- Gate checks that need no baseline: `gate.min_pass_rate` (a number or a dict by scorer),
  `gate.must_pass` (case tags) and `gate.max_new_failures`, with `compare --min-pass-rate`,
  `--must-pass` and `--max-new-failures`. `evalproof run --gate` applies the first two to a
  single run.
- `run` and `report` end with a "Failed cases" list (`--show N`, default 10). `compare` lists
  the regressed ids of every compared metric and, with `--show N` (default 5), prints both
  answers and the candidate scorer's reason for that many regressed cases of the gated metrics.
- `ChatResponse.cost_usd`: the `claude-code` provider reports the CLI's `total_cost_usd`, which
  the run records as the call's cost when the suite gives no pricing for that target. Under a
  subscription login it is an API-equivalent estimate, not a bill. The judge's cost is now
  computed from its `pricing` and counts toward `max_cost_usd`.
- On a continuous gated metric, `min_pass_rate` cannot gate (exit 4, naming the fix),
  `max_new_failures` is skipped with a warning, and `must_pass` fails only an errored or
  undecided case, with a warning.
- `import --quiet` keeps the report off stdout. `compare` and `run --gate` print a warning
  when a `must_pass` tag matches no case.
- `docs/specs/0.5.0-dev-workflow.md`.

### Changed

- A new judge CLI version alone only warns (`judge_cli_version`); the gate still refuses when
  the resolved model, target, CLI or isolation changes.
- The report says "cannot gate" for exit 4 instead of "fail", and lists up to ten regressed ids
  per metric (it listed twenty, for gated metrics only).
- `run --system-prompt-file F` without `--label` labels the run `<system> + F` so two prompt
  files are told apart.
- `evalproof init --ci api-key` writes a workflow with no `paths:` filter: a first step runs the
  suite only when `WATCH` (default `evals/`) changed, forks are skipped with a notice, and
  responses are cached in `evals/.evalproof/cache` with `actions/cache`. The keyless workflow
  names what `verify` checks and what it cannot prove.
- An unexpected exception prints one line and a hint (`EVALPROOF_DEBUG=1` shows the traceback)
  instead of a traceback. Ctrl-C exits 130. Label file errors in `calibrate` carry `file:line`.
- The agent skill and the `evals/README.md` that `init` writes explain adding cases, reading a
  result, `verify`, `run --gate`, `must_pass` and `max_new_failures`.

### Fixed

- A `--limit` run, a stale run file and an edited case no longer pass the keyless gate
  unnoticed: `verify` fails the job.
- A run whose `--out` could not be written failed after every case had run; the path is now
  checked before the first case.

## [0.4.1] — 2026-11-13

### Fixed

- `cursor:` ask mode is not tool-free: it has a read tool that runs without approval and reads
  files in the workspace and outside it. The workspace now carries a `.cursor/cli.json` that
  allows nothing and denies reads (verified inside and outside the workspace), writes and shell
  commands (ask mode has no write or shell tool; those entries are a second guard), and any
  `tool_call` event in the output fails the call as an isolation break.
- `cursor:` wrote every call's chat to `~/.cursor/chats` and read the user's
  `~/.cursor/cli-config.json`. `CURSOR_CONFIG_DIR` and `CURSOR_DATA_DIR` now point into the
  call's temp root, which keeps the login (verified) and starts from a config that allows
  nothing.
- `copilot:` kept every call's session in `~/.copilot` and read the user's Copilot config.
  `COPILOT_HOME` now points into the call's temp root, which keeps the login (verified).
- `copilot:` passed `--available-tools=`, which copilot 1.0.110 ignores, so the model could use
  its tools (it read a file). It now passes an allow-list with one unknown name, which disables
  every tool (verified), and any tool event or tool request in the output fails the call.

### Changed

- `cursor:` uses `--output-format stream-json` (the json result joined the model's preamble
  with the answer).
- `copilot:` takes the prompt on stdin, prints JSONL, runs with `--no-custom-instructions` and
  `--disable-builtin-mcps`, reports real token counts from `--usage-output-file`, and records
  the model that answered each call: with `copilot:auto` Copilot routes every call, so the run
  artifact shows the routed model, not `auto`.
- Isolation tags are now `cursor/2` and `copilot/2`, so earlier cached replies for these two
  are not reused.
- `evalproof doctor` warns that `cursor-agent` sends the names and descriptions of the user's
  skills (`~/.claude/skills`, `~/.cursor/skills` and others) in every prompt, which no option
  turns off, and that `copilot:auto` as the judge makes compare and the gate refuse runs as a
  judge change (Copilot picks the model per call).

### Documentation

- The experimental providers section now says what was verified and what was not: Cursor
  reads instruction files from the directories above the workspace; Cursor and Copilot keep
  their own system prompt; Copilot's `--no-custom-instructions` was verified with canary
  files; the Copilot tool guard has not fired live; named-model availability and the
  premium-request rate depend on plan and policy; with tools off Copilot may write a made-up
  tool call as text.

## [0.4.0] — 2026-11-13

### Added

- `claude-code:<model>` and `codex:<model>` providers: run your own logged-in Claude Code or
  Codex CLI as a judge or a system under test, with no API key. Each call runs in a fresh
  empty directory with the agent's tools, settings, MCP servers, skills and memory switched
  off and no permission-bypass flag; API-key variables are removed from the child's
  environment unless `api_key_env` names one or `EVALPROOF_CLI_AUTH=api-key` keeps them.
  Neither CLI takes a temperature, seed or token limit, so those params are ignored (Codex
  maps `think` to its reasoning effort). Claude Code runs with no tools; Codex still offers a
  JavaScript isolate, sub-agents and a question to the user, so any item in its output other
  than a message, reasoning, an error or a plan fails the call as an isolation break, and a
  codex too old for the flags evalproof sets gets a message to upgrade. ADR 0004 has the
  reasoning.
- `EVALPROOF_CLI_AUTH` (`login`, the default, or `api-key`) chooses whether a CLI provider
  uses the subscription login or its own API key variable; `evalproof doctor` prints the mode
  in effect and warns when `api-key` has no key to use. `api_key_env` still wins.
- `evalproof doctor [suite.yaml]`: reports each CLI's path, version, login mode and the
  isolation flags it will use, without a model call, and exits 1 when a target of the suite
  is not usable. Without a suite it surveys the installed CLIs and exits 1 only when none is
  usable. It prints no account details.
- Run facts for CLI targets in `runner.environment`: `{judge,system}_cli`,
  `{judge,system}_cli_version`, `{judge,system}_isolation` and `{judge,system}_model_resolved`
  (the model ids that answered, sorted, cached replies included). The response cache keys on
  the CLI fingerprint, so a CLI upgrade is a cache miss.
- `compare` reports `same_judge`, `judge_changes` and `system_changes`, and the gate refuses
  to decide (exit 4) when the judge changed between the runs. `--allow-judge-change` or
  `gate.allow_judge_change: true` accepts the change and keeps a warning.
- Experimental `opencode:<provider>/<model>`, `cursor:<model>` and `copilot:<model>`
  providers, on the same footing as the CLI providers above (fresh empty directory, the CLI's
  key variable removed unless `api_key_env` names one, bounded runtime, CLI fingerprint with the
  isolation tags `opencode/0`, `cursor/0`, `copilot/0`). Each prints one warning on stderr the
  first time it is used. Only OpenCode was exercised for real, and only against a local ollama
  model; Cursor (not logged in) and Copilot (not installed) follow their documentation, and the
  README's "Experimental providers" says what was and was not verified. `evalproof doctor`
  without a suite now also lists whichever of these CLIs are on `PATH`.
- `evalproof.providers.TransientError`, a `ProviderError` that `with_retries` retries; the CLI
  providers raise it for rate limits and server errors.
- `tests/test_live_agent_cli.py`: one real call per installed CLI, skipped unless
  `EVALPROOF_LIVE=1`.
- `evalproof init [--dir PATH] [--judge TARGET] [--ci keyless|api-key|none] [--no-skill]
  [--force]`: writes `evals/suite.yaml`, three sample cases in `evals/cases.jsonl`, an
  `evals/README.md`, a GitHub Actions workflow at `.github/workflows/evalproof.yml` and an
  agent skill at `.agents/skills/evalproof/SKILL.md` and `.claude/skills/evalproof/SKILL.md`.
  The judge is the first of `claude`, `codex`, `ollama` found on `PATH`, else a hosted
  Anthropic model. The keyless workflow makes no model calls and gates the committed
  `evals/runs/latest.json` against the base branch; the api-key workflow runs the suite in CI
  (with `EVALPROOF_CLI_AUTH: api-key` for a CLI judge, so no second suite file is written)
  and pins the judge CLI to the version found at init time. Existing files are kept unless
  `--force` is passed, nothing is written outside `--dir`, and no model is called.

### Changed

- A gate that passed in 0.3.0 can now exit 4 when the judge's target, CLI version, isolation
  or resolved model differs between two runs that both carry these keys. Runs written before
  this release carry none and compare as before.
- A changed system target or CLI version between two runs now adds a warning, so the status
  is `warn` instead of `pass` (the exit code stays 0).
- `runner.environment` records `judge_target` and `system_target` for every model target (not
  only CLI ones), and the system target it records is the one in effect, with any override
  applied.
- The gate's "too few paired cases" reason now says how to fix it (add cases, or lower
  `gate.min_cases`).

## [0.3.0] — 2026-11-12

### Added

- `evalproof import` and `evalproof.import_run`: per-case results from promptfoo (JSON and
  JSONL), Inspect (`.json` and `.eval` logs), DeepEval, lm-evaluation-harness
  (`samples_*.jsonl`) and generic long or wide `jsonl` and `csv` files become a run that
  `compare` and the gate read. Case ids are stable across runs, system errors score as
  failures, repeats collapse by `--epochs`, and NaN or infinite scores are refused with the
  case and metric named. Raises `evalproof.ImportFormatError` (a `ValueError`; exit 2 on the
  command line).
- Gating several metrics: `gate.metrics` in the suite or `--metric` repeated on
  `compare`, with Holm-adjusted p-values (`evalproof.holm`). The decision carries
  `adjusted_p`, the report gains a Holm-adjusted p column, and a single metric behaves as
  before. ADR 0003 has the reasoning.
- The composite action's `metric` input takes several names, separated by newlines or commas.
- `examples/retrieval`: an offline retrieval example for recall@k, MRR and nDCG, with a
  regression that recall@5 cannot see and MRR and nDCG@5 can. Contributed by
  @HarshRajSinghania in #5. CI asserts its gate exits 1 on the regression and 0 on no change.
- MEASUREMENTS §8: the support-qa judge with thinking on (#3), with its runs, suite and
  analysis in `measure.py`. Against labels audited case by case it passes no wrong answer
  and κ rises, at 6.7× the judge time; the gate's decision does not change.
- A provider profile table (`providers/profiles.py`) behind one OpenAI-compatible adapter:
  `openai`, `azure`, `gemini`, `mistral`, `groq`, `together`, `fireworks`, `deepseek`, `xai`,
  `openrouter`, `cerebras`, the local servers `vllm`, `lmstudio` and `llamacpp`, and
  `openai-compatible` for any other server (set `base_url`). Each profile carries its base
  URL, key variable, auth header, token-limit field and a link to the provider's docs.
  `OpenAIProvider` stays importable.
- `TargetConfig.api_key_env` names the environment variable that holds the key, and
  `TargetConfig.headers` adds request headers to every HTTP provider; suite headers win over
  a provider's own, and neither is part of the cache key or written to a run. A named key
  variable that is not set is an error rather than a 401.
- Anthropic: a `refusal` stop reason is recorded as a case error naming its category, and so
  is `max_tokens` reached before any text, with a hint to raise `params.max_tokens`. The
  OpenAI-compatible adapter does the same for `finish_reason` `length` with no text, and
  for `content_filter`, and reads replies that arrive as a list of typed content parts.
- A test pins the cache key of an existing Ollama call, so a change to it cannot slip in.

### Changed

- Hosted providers (everything except Ollama and the local servers) send `temperature` and
  `seed` only when the suite sets them. Before, `openai` sent `temperature: 0` and
  `seed: 0` on every call, which reasoning models reject; now an unset value is left to the
  provider's default. A hosted run that relied on the implicit `temperature: 0` should set
  it in `params`. Ollama and the local servers still send both, so local runs repeat as
  before and keep their cache keys; `GenerationParams` defaults are unchanged. A hosted call
  that now leaves a value out gets a new cache key, so a reply cached at the old implicit
  `temperature: 0` is not reused for it. To point `openai` at an older local server through
  `base_url`, use `openai-compatible` or a local profile instead, which send both values.
- `openai` sends the token limit as `max_completion_tokens`, which OpenAI's reasoning models
  require (they reject `max_tokens`), and `mistral` sends its seed as `random_seed`. The
  default limit is still 512.
- The llama.cpp profile defaults to port 9931, the one its server documents now; for an older
  build on 8080, set `base_url`.

### Fixed

- Anthropic calls to current Claude models failed with HTTP 400, because `temperature` was
  sent on every request. It goes out only when the suite sets it, and no seed is sent, since
  the Messages API has none.
- A gate built without validation and given no metric passed by gating nothing; it now
  cannot gate (exit 4), and `compare --metric ""` is a usage error (exit 2).
- Found in review before release, in the new importers: two promptfoo prompts whose first 60
  characters matched merged into one system (the label is no longer truncated, and two
  prompts that share a label are an error), promptfoo output with repeated tests but no
  `repeatIndex` now warns that it reads as separate cases, Inspect message ids no longer
  change the dataset hash, a `null` promptfoo `testIdx` no longer crashes, and under
  `--epochs any` or `mean` a case with one good repeat is no longer marked as an error.
- `examples/support-qa/cases.jsonl`: the q21 and q22 references named Enterprise, which the
  product facts do not give SSO or an admin console (#7). The original set is kept as
  `cases.v1.jsonl` so the recorded runs stay reproducible; the runs judged on the revised set
  are in `runs/v2/`, and MEASUREMENTS §8 reports both.

## [0.2.1] — 2026-10-16

### Fixed

- The PyPI and Python badges on the 0.2.0 PyPI page read "package or version not found".
  PyPI's image proxy fetched them about a minute after the first upload, before the badge
  service could see the package, and keeps that answer for 12 to 24 hours. The PyPI page
  now leaves those two badges out, since it shows the version and supported Pythons
  itself, and the README's copies use new URLs so no cache serves the old images. CI fails
  if a PyPI-data badge reaches the PyPI description again.

## [0.2.0] — 2026-10-16

The first release on PyPI: `pip install evalproof`.

### Added

- A supported Python API: `from evalproof import compare_runs, decide, calibrate, ...`.
  The names in `evalproof.__all__` are the public surface; other modules are internal.
- Python 3.14 support, tested in CI alongside 3.11 to 3.13.
- Releases publish to PyPI from GitHub Actions with trusted publishing, so no token is
  stored anywhere. Each file on PyPI carries a PEP 740 attestation, and the GitHub release
  carries the same files with a build-provenance attestation.
- `SECURITY.md` with private vulnerability reporting and the trust boundaries, backed by a
  test that runs both hosted providers with a placeholder key and finds it in no file written.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), issue and pull-request templates,
  `CITATION.cff`, `CODEOWNERS`, Dependabot for actions and Python dependencies with a
  three-day cooldown, and a pre-commit configuration.

### Changed

- The version is written once, in `src/evalproof/_version.py`; the build reads it.
- The README published to PyPI rewrites relative links to GitHub and replaces the Mermaid
  diagram with a link, so the PyPI page has no broken links. CI checks this on every
  commit, along with `twine check --strict` and a clean install of the built wheel.
- CI pins every action to a commit SHA, checks out without persisted credentials, sets
  timeouts and cancels superseded pull-request runs, and audits the workflows with zizmor.
  The composite action pins `setup-uv` the same way.
- Package metadata: documentation, source, changelog and live-demo links; classifiers for
  typing, Python 3.14 and beta status.

## [0.1.2] — 2026-10-15

### Added

- `app/`: an interactive Streamlit app, live at https://evalproof.streamlit.app.
  *Check a drop* starts from four scenarios (the recorded run, the same rates on 4× the
  cases, a small but certain drop, a twenty-case suite) or from sliders, and shows the
  gate's verdict, every paired case as a grid, the coin-flip distribution behind the exact
  McNemar p-value, and the suite size at which the same drop would become significant.
  *The recorded run* replays the committed `examples/support-qa` runs: select any of the
  twelve changed answers to read both versions side by side, add a latency budget, and
  calibrate the judge on a gold sample of chosen size. It runs the library from `src/`
  through the same `compare_runs` and `decide` calls as the CLI; CI installs the app's own
  requirements and drives every scenario.

### Fixed

- `evalproof compare` raised OverflowError when a metric had more than about 1,030
  discordant pairs: the exact McNemar test summed `comb(n, i) * 0.5**n` in floating
  point, and `comb(n, i)` outgrows the largest float. The binomial tail is now summed
  as integers and divided once, so it stays exact at any size.

## [0.1.1] — 2026-10-13

### Fixed

- The Wilson interval now returns exactly 0 at k=0 and exactly 1 at k=n. Computing
  the boundaries by subtraction left platform-dependent rounding (2.8e-17 on x86_64
  Linux, 0.0 on arm64 macOS).

### Changed

- The three-day `exclude-newer` resolution guard is declared in `[tool.uv]`, so
  `uv.lock` means the same thing on every machine and in CI.
- CI pins uv to the version that wrote `uv.lock`.

## [0.1.0] — 2026-10-13

First release.

### Added

- Suites in YAML: a JSONL golden set, a system under test, scorers, an optional judge,
  calibration settings and a gate.
- Systems under test: Ollama, Anthropic and OpenAI models with a prompt template; Python
  functions; HTTP endpoints. System failures are recorded per case, never fatal.
- Scorers: exact match, contains, regex with accept/reject answer keys, a JSON Schema
  subset, numeric tolerance, recall@k, MRR, nDCG, and an LLM judge with a strict JSON
  verdict, one parse retry and an explicit undecided state.
- Pairwise judging with order swap; position-inconsistent verdicts become ties.
- Judge calibration: confusion matrix, TPR, FPR, Cohen's κ, and prevalence-corrected pass
  rates by adjusted classify-and-count (bootstrap interval) and prediction-powered
  inference (normal interval).
- Paired comparison of two runs: exact McNemar for pass/fail, paired bootstrap for scores.
- A CI gate that fails only on a drop that is both beyond `max_drop` and significant at
  `alpha`, plus p95-latency and cost budgets, with distinct exit codes.
- Content-addressed response cache; runs record the Ollama version and model digests.
- Markdown reports for GitHub step summaries, JSON reports, and a composite GitHub Action.
- Two examples: an offline ticket router (CI's gate contract) and a product support
  assistant judged by a local model, with measurements.

[Unreleased]: https://github.com/technobasant/evalproof/compare/v0.5.2...HEAD
[0.5.2]: https://github.com/technobasant/evalproof/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/technobasant/evalproof/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/technobasant/evalproof/compare/v0.4.1...v0.5.0
[0.4.1]: https://github.com/technobasant/evalproof/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/technobasant/evalproof/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/technobasant/evalproof/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/technobasant/evalproof/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/technobasant/evalproof/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/technobasant/evalproof/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/technobasant/evalproof/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/technobasant/evalproof/releases/tag/v0.1.0
