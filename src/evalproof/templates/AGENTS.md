<!-- evalproof:begin -->
## evalproof

This repository measures LLM changes with evalproof. Use this section when a task touches a
prompt, a model, a retrieval step, an agent's tools, or the eval suite itself. Do not edit the
text between the `evalproof:begin` and `evalproof:end` markers: `evalproof init --agent-files-only`
replaces it. Write your own notes outside them. Full docs: https://basantbhattarai.com.np/evalproof/

### Where things are

| Path | What it is |
|---|---|
| `evals/suite.yaml` | the suite: the system under test, the judge, the scorers, the gate |
| `evals/cases.jsonl` | the golden set, one JSON case per line |
| `evals/runs/latest.json` | the committed result of the last accepted run, the baseline for the next change |
| `evals/.evalproof/` | local scratch and the response cache, ignored by git |
| `evals/README.md` | the commands for this repo |
| `.github/workflows/evalproof.yml` or `.gitlab-ci.evalproof.yml` | the CI gate, if `init` wrote one |

If a path is missing, the repo may keep its suite elsewhere: search for `suite.yaml` before
creating one.

### How to run it

Use `evalproof` if `evalproof --version` works, otherwise `uvx evalproof`. `npx evalproof` also
works where uv or pipx is on `PATH`: the npm package is a launcher for the PyPI release. Run
every command from the repository root.

### What to do for a task

| Task | Do this |
|---|---|
| Check a change before committing | "Check a change" below |
| First run, no baseline yet | `evalproof run evals/suite.yaml --gate --out evals/runs/latest.json`. `--gate` applies `gate.min_pass_rate` and `gate.must_pass` to this run alone: exit 1 when a floor is missed, 4 when the gate cannot apply one. If the suite sets neither, ask the human for the floor; do not choose one to make the run pass |
| Add a case | "Add a case" below |
| Build or extend the suite | "Build or extend the suite" below |
| Choose between models or providers | One run per candidate with `--system TARGET`, or named `variants:` with `--variant NAME`. Write each to its own file under `evals/.evalproof/`. Then `evalproof compare baseline.json a.json b.json --suite evals/suite.yaml --gate` prints one table and exits with the worst code. Do not commit these runs as `latest.json` |
| Calibrate an LLM judge | `evalproof calibrate RUN --judge-scorer NAME --labels labels.jsonl` (lines of `{"id", "label"}` written by a human) or `--gold-scorer NAME` (a trusted scorer in the same run). The suite can hold a `calibration:` block. It makes no model call |
| Use a coding-agent binary as the judge | `judge: {target: claude-code:haiku}` or `codex:<model>`, plus `concurrency: 4`. Run `evalproof doctor evals/suite.yaml` first. Changing the judge is the human's decision, because runs from different judges cannot be compared |
| Test an agent | `system: {target: python:module:function}` where the function returns `{"output": ..., "trace": ...}`. Score the trace with a `checks` scorer and per-case `expect` checks. SDK adapters in `evalproof.adapters` build the trace: `adk.from_events` (Google ADK), `langchain.from_messages` (LangChain and LangGraph), `claude_agent_sdk.from_messages`, `openai_agents.from_result`, `otel.from_spans` (experimental), and `TraceRecorder` and `mcp_call` for plain Python and MCP |
| Score retrieval or RAG | Scorers `recall_at_k`, `mrr` and `ndcg` (`params: {k: 5, pass_threshold: 0.8}`). The case holds the relevant ids in `relevant` (the field is the scorer's `field` param); the system returns what it retrieved, ranked, as `retrieved`. Use `facts` for numbers and phrases an answer must contain |
| Hold latency or cost | Set `gate.max_p95_latency_ms` and `gate.max_cost_usd`. A breach exits 3. Cost needs `pricing` on the target (`input_per_mtok`, `output_per_mtok`, US dollars) or a system that returns `cost_usd` |
| Handle flaky output | `trials: 5` in the suite or `run --trials 5` (1 to 100). Both runs must use the same number of trials. `gate.aggregate` or `compare --aggregate` picks `mean`, `all` or `any`. The report lists flaky cases and a power line saying what drop the suite can detect; the power line never changes the exit code |
| Compare several builds | `variants:` in the suite, then `evalproof run evals/suite.yaml --all-variants`. Each variant writes `runs/<name>.json` |
| Gate results from another tool | `evalproof import FORMAT FILE --out PATH` with FORMAT one of `promptfoo`, `inspect`, `deepeval`, `lm-eval`, `garak`, `adk`, `claude-plugin-eval`, `jsonl`, `csv`. |
| Start from test cases another tool holds | `evalproof import-dataset FORMAT FILE --out evals/cases.jsonl --suite-out evals/suite.yaml` with FORMAT `auto` or one of `promptfoo`, `deepeval`, `ragas`, `langsmith`, `openai-evals`, `inspect`, `braintrust`, `langfuse`, `phoenix`, `adk`, `jsonl`, `csv`. In a new repository: `evalproof init --from FILE`. No model call. Assertions it cannot express stay on the case under `unmapped` and are counted in one stderr line: read it and tell the human. `promptfoo`, `adk`, `jsonl` and `csv` are stable; the rest are experimental |
| See what the suite leaves untested | `evalproof readiness evals/suite.yaml [--run FILE] [--require behaviour,regression-gate]`. No model call |
| Check a committed run file | `evalproof verify evals/runs/latest.json evals/suite.yaml`. No model call. Exit 4 means stale: run the suite again. For a baseline written before evalproof 0.5: `evalproof rehash RUN evals/suite.yaml` |
| Check that a CLI target is usable | `evalproof doctor evals/suite.yaml`. No model call |
| Set up CI | `evalproof init --ci keyless` (default: CI compares committed run files, no model calls) or `--ci api-key` (CI runs the suite with a secret), with `--platform github` or `gitlab` (`gitlab` is experimental). For Jenkins, Azure DevOps or anything that reads JUnit: `evalproof run SUITE --junit FILE` and `evalproof compare A B --suite SUITE --gate --junit FILE` (experimental) |
| Test a program in another language | `target: "command:node eval/system.js"` (experimental): one run of the program per case, the case as JSON on stdin, the answer on stdout, a non-zero exit is an error. Or an HTTP endpoint: `target: http://localhost:3000/eval`, which gets `{"id", "input"}` by POST |
| Run in a container | `docker run --rm -v "$PWD:/work" ghcr.io/technobasant/evalproof:latest run evals/suite.yaml --out evals/runs/latest.json`. Tags are `<version>`, `<major>.<minor>` and `latest`: pin one in CI. Pass a key with `-e NAME` and no value. The image has no coding-agent binary and no packages beyond evalproof's own. To reach Ollama on the host, set `base_url: http://host.docker.internal:11434` in the suite (the image's contents are experimental) |
| Refresh this section | `evalproof init --agent-files-only` (experimental) |

### Check a change

Run after making the change and before committing it:

```bash
mkdir -p evals/.evalproof
git show HEAD:evals/runs/latest.json > evals/.evalproof/baseline.json
evalproof run evals/suite.yaml --out evals/.evalproof/candidate.json
evalproof compare evals/.evalproof/baseline.json evals/.evalproof/candidate.json \
  --suite evals/suite.yaml --gate
echo "exit $?"
```

If the gate passes and the change stays, replace the committed run and check it as CI will:

```bash
evalproof run evals/suite.yaml --out evals/runs/latest.json
evalproof verify evals/runs/latest.json evals/suite.yaml
```

Read the result:

- `run` prints a pass rate per scorer with a 95% interval and latency, then the failed cases (id,
  scorer, reason; `--show N` changes how many). Errored cases count as failures. A run where
  every case errored exits 2.
- `compare` prints the change per scorer with its interval and p-value, the ids that regressed,
  and for a few of them both answers and the scorer's reason. "Cannot gate" (exit 4) means it did
  not decide. It is not a regression.
- The warning "dropped but not distinguishable from noise" means too few cases to tell. It does
  not mean the change is safe.
- The gate also fails on `gate.max_new_failures`, `gate.min_pass_rate` and `gate.must_pass`.
  Those settings belong to the owner. Do not change them.

### Add a case

Append one JSON object per line to `evals/cases.jsonl`:

```json
{"id": "refund-window", "input": {"question": "..."}, "reference": "the correct answer", "accept": ["a string a correct answer contains"], "tags": ["refunds"]}
```

- `id` is unique and never changes for the life of the case.
- Which fields a case needs depends on the suite's scorers: read their `params` in `suite.yaml`.
  By default `reference` is read by `exact_match`, `numeric_tolerance` and the judge, `facts` and
  `must_mention` by `facts`, `relevant` by `recall_at_k`, `mrr` and `ndcg`, and `expect` by
  `checks`. `contains` and `regex` read the fields their `any_of_field`, `all_of_field` and
  `none_of_field` params name (the starter suite uses `accept`). `turns` and `state` make a
  conversation. `rubrics` feeds a rubric-set judge.
- Add cases for real failures: a bug report, a bad answer from production, a regression.
- Cases are paired by id and content. A new case has no baseline score. An edited case (same id,
  new content) is left out of the comparison and listed in the report until the baseline is
  refreshed by merging. Adding or changing a tag is not an edit, but `verify` and the CI check
  still count a tag change, so run the suite again and commit the new `evals/runs/latest.json`
  after any change to the cases.

### Build or extend the suite

Use this when asked to create the suite, add coverage, or find what is untested. It reads the
repository, so do it before drafting anything.

1. Find where the repo calls a model: imports of the model SDKs listed in its dependency files,
   prompt files and template strings, system prompts, retrieval code, and agent tool definitions
   (name, arguments, what each returns). Note each call site and what its output feeds.
2. Read what already says what correct looks like: unit tests and fixtures, example inputs and
   outputs in docs, other tools' eval files (the formats `import` and `import-dataset` read),
   and bug reports, issue titles and fix commits. Every fixed bug is a candidate case.
   If the repo has a promptfoo config or a DeepEval, Ragas, LangSmith, Inspect, OpenAI Evals,
   Braintrust, Langfuse or Phoenix dataset, import it before drafting anything:
   `evalproof import-dataset auto FILE --out evals/cases.jsonl --suite-out evals/suite.yaml`.
   Keep the `unmapped` fields it leaves on cases and report what was not imported.
3. Run `evalproof readiness evals/suite.yaml`. It lists, per dimension, what is covered, partial,
   missing or out of scope. Work through the gaps that matter for the call sites you found.
4. Draft cases per gap. Choose the scorer first, then write the case fields it reads.
5. Check the draft: `evalproof doctor evals/suite.yaml`, then
   `evalproof run evals/suite.yaml --limit 5 --out evals/.evalproof/scratch.json`. The scratch
   file is ignored by git and is never committed. Then one full run to
   `evals/runs/latest.json` and `evalproof verify evals/runs/latest.json evals/suite.yaml`.

Rules for drafting:

- Expected answers (`reference`, `accept`, `reject`, `facts`, `must_mention`, `relevant`, and
  `expect` checks) come only from a source of truth: the code, the docs, existing tests, or the
  human. Never copy them from a model's output, and never from the system under test's own
  answer. If there is no source for a case, leave it out and ask.
- Tag every drafted case `draft`, and list each one to the human with the source of its expected
  answer. Never put `draft` in `gate.must_pass`. The human decides when a case stops being a
  draft. Removing the tag is a tag change: run the suite again and commit `latest.json`.
- Prefer deterministic scorers (`exact_match`, `contains`, `regex`, `json_schema`,
  `numeric_tolerance`, `facts`, `checks`) over `llm_judge`. They cost no judge call and `verify`
  can re-score them offline. Use the judge only for what they cannot express, and calibrate it.
- Do not add cases a correct system would fail, and do not tune a case until the current system
  passes it.
- The gate needs `gate.min_cases` paired cases (20 in the starter) before it will decide.

### Providers and targets

`system` and `judge` take `target: <provider>:<model>`. The model is everything after the first
colon, so `ollama:qwen3:8b` works.

| Target | Key variable | Notes |
|---|---|---|
| `ollama:<model>` | none | local. Receives `temperature` and `seed` on every call, so a run repeats |
| `vllm:<model>`, `llamacpp:<model>` | `VLLM_API_KEY`, `LLAMA_API_KEY`, both optional | local, same sampling as Ollama |
| `lmstudio:<model>` | optional, through `api_key_env` | local, same sampling as Ollama |
| `anthropic:<model>` | `ANTHROPIC_API_KEY` | native adapter. Never receives a seed; gets `temperature` only when `params` sets it |
| `openai`, `gemini`, `mistral`, `groq`, `together`, `fireworks`, `deepseek`, `xai`, `openrouter`, `cerebras` | `OPENAI_API_KEY`, `GEMINI_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `TOGETHER_API_KEY`, `FIREWORKS_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`, `OPENROUTER_API_KEY`, `CEREBRAS_API_KEY` | OpenAI-compatible profiles. Hosted: they get `temperature` and `seed` only when `params` sets them |
| `azure:<deployment>` | `AZURE_OPENAI_API_KEY` | needs `base_url` |
| `openai-compatible:<model>` | optional, through `api_key_env` | any other server (a gateway, LiteLLM, TGI, SGLang). Needs `base_url` |
| `python:<module>:<function>` | none | runs the function from the repo with the case input; it returns `{"output": ...}` and may add `trace`, `retrieved`, `tokens_in`, `tokens_out` and `cost_usd`. Declare a `ctx` parameter to get the case id, later turns, state, variant, vars and trial |
| `http://...`, `https://...` | none | POSTs `{"id", "input"}` (plus `turns`, `state`, `variant`, `vars` and `trial` when they apply) and reads `{"output", ...}` back |
| `command:<program> [args]` | none | experimental. Words split like a POSIX shell, no shell runs. Runs in the suite's directory. Print plain text or a JSON object with `output` on stdout, logs on stderr |
| `claude-code:<model>`, `codex:<model>` | none for the suite | run the installed coding-agent binary as a subprocess, isolated in an empty directory. Ignore `temperature`, `seed` and `max_tokens`. A suite with `turns` cases is refused. Run `evalproof doctor` first. Not in the Docker image, and `--platform gitlab --ci api-key` refuses them |
| `opencode:<provider>/<model>`, `cursor:<model>`, `copilot:<model>` | none for the suite | experimental, same rules as the two above. Each prints a note on stderr the first time it is used. `copilot:auto` can resolve to a different model on each call, so it makes a poor judge: the gate refuses runs whose judge's resolved model differs (exit 4) |

Settings that apply to every target:

- `base_url` points a target at a gateway, a regional endpoint (Azure needs it) or a self-hosted
  server. It overrides the profile's default.
- `api_key_env` names the environment variable that holds the key. If it is named, it must be
  set. The key never goes in the suite.
- `headers` are extra HTTP headers on every call. They are not part of the cache key and the
  suite is committed, so never put a secret in them.
- `params` holds `temperature` (default 0.0), `seed` (default 0), `max_tokens` (default 512),
  `think` (default false) and `extra` (merged into the request body). A thinking model spends
  `max_tokens` on thinking, so raise `params.max_tokens` for one. A reply cut off before its
  first word, a refusal or a content-filter block is recorded as that case's error.
- `pricing` (`input_per_mtok`, `output_per_mtok`) is needed for `gate.max_cost_usd` on a target
  that does not report its own cost. Local models cost nothing unless priced.
- `concurrency` (default 1) runs that many cases at once. Raise it for slow targets such as
  coding-agent binaries.
- `timeout_s` (default 120) bounds one call.
- AWS Bedrock and Google Vertex AI need cloud signing, which evalproof does not do. Use a
  gateway's OpenAI-compatible endpoint through `openai-compatible`.
- The judge is its own block with the same keys as `system`. A judge change (target, binary,
  isolation or resolved model) makes two runs incomparable: `compare` exits 4 unless
  `--allow-judge-change` or `gate.allow_judge_change` is set. That is the human's decision.
- The fingerprint that `verify` and the gate check covers the system and judge `target`,
  `system_prompt`, `prompt_template`, `params` and `base_url`, every scorer and the `seed`.
  Changing any of them means a new run, not an edit to `latest.json`.

### Cost, secrets and safety

- `run` makes real model calls for every case that is not in the response cache, and they cost
  quota or money. Run the suite once per change, not in a loop, and do not re-run to see whether
  a failure goes away. An unchanged case is free the second time. `--no-cache` asks again and
  costs again. `--cache-only` never calls a model: a miss is an error.
- These make no model call: `compare`, `calibrate`, `verify`, `rehash`, `report`, `import`,
  `import-dataset`, `init`, `doctor` (except `--cursor-canary`) and `readiness`. `pairwise` calls
  the judge.
- Keys come from environment variables only: the provider's own variable or the one
  `api_key_env` names. Never put a key in `suite.yaml`, in `headers`, in a case or in any
  committed file. Never read, print or commit a `.env` file or a key. If a key is missing, tell
  the human which variable to set.
- Cases and outputs are stored in the run file, which is committed. Keep secrets and personal
  data out of cases.
- A `python:` or `command:` target runs code from this repository. Read what it does before the
  first run.

### Rules

- Never edit `evals/runs/latest.json` by hand. Regenerate it with `evalproof run`.
- Never weaken a gate to make it pass: do not raise `max_drop` or `alpha`, lower `min_cases`,
  change `gate.metric`, or loosen the rubric.
- Never delete or edit cases, references or `accept` lists to make a run pass.
- On exit 1, 3 or 4, stop and report to the human: the exit code, the `compare` output and what
  you changed. Do not work around it by switching the judge, running a subset (`--limit`) or
  re-running until it passes.
- Do not commit a run made with `--limit` or `--system` as `latest.json`.
- A different judge model makes runs incomparable (a new judge CLI version alone only warns).
  Changing the judge is the human's decision.
- Do not change `gate.*`: those settings belong to the owner. Propose a change and let the
  human make it.

### Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | Pass, possibly with warnings | Read the warnings |
| 1 | Regression: a large, significant drop, or a failed `min_pass_rate`, `must_pass` or `max_new_failures` check. Also `readiness --require` with a dimension not covered, and `doctor` with a target not ready | Report it to the human |
| 2 | Usage error: bad suite, path or target. Also `run` where every case errored | Fix the command or the suite |
| 3 | Budget breach: p95 latency or cost over the limit | Report it to the human |
| 4 | Cannot gate: too few paired cases, a metric with no scores, the judge changed, runs with different trial counts, or `verify` found the run file does not match the suite and cases | Report it to the human; for a stale run file, run the suite again. A baseline written before evalproof 0.5 has no per-case hashes: `evalproof rehash BASELINE evals/suite.yaml` fixes it with no model call, while the cases it scored are unchanged (appended cases are fine) |

With several candidates, `compare --gate` exits with the worst code in the order 2, 4, 1, 3, 0.

### Command reference

Add `--help` to any command for its flags.

- `evalproof init [--dir DIR] [--judge TARGET] [--ci keyless|api-key|none] [--platform github|gitlab] [--cli-auth api-key|oauth-token] [--from FILE] [--from-format FORMAT] [--no-skill] [--no-agents-md] [--agent-files-only] [--force]`:
  write the starter suite, CI workflow, agent skill and this section. Existing files are kept
  unless `--force`; `AGENTS.md` is always merged, never overwritten. `--agent-files-only`
  refreshes only this section and the skill (experimental). `--from FILE` writes
  `evals/cases.jsonl` and the suite from another tool's test cases instead of the sample cases
  (`--from-format` defaults to `auto`).
- `evalproof run SUITE [--out FILE] [--limit N] [--trials K] [--system TARGET] [--variant NAME]... [--all-variants] [--gate] [--no-cache] [--cache-only] [--junit FILE] [--markdown FILE] [--quiet] [--show N]`:
  run the suite and write a run file. `--system-prompt-file` and `--label` also override the
  system.
- `evalproof compare BASELINE CANDIDATE... [--suite SUITE] [--gate] [--metric NAME]... [--max-drop F] [--alpha F] [--aggregate mean|all|any] [--min-cases N] [--max-p95-ms MS] [--max-cost USD] [--min-pass-rate F] [--must-pass TAG]... [--max-new-failures N] [--allow-judge-change] [--show N] [--seed N] [--markdown FILE] [--json FILE] [--junit FILE]`:
  compare runs case by case. Flags override the suite's `gate` block. `--junit` needs `--gate`
  (experimental).
- `evalproof calibrate RUN --judge-scorer NAME (--labels FILE | --gold-scorer NAME) [--gold-sample N] [--seed N] [--alpha F] [--markdown FILE] [--json FILE]`:
  measure the judge against gold labels and correct its rate.
- `evalproof report RUN [--markdown FILE] [--show N]`: print the summary of one run.
- `evalproof verify RUN SUITE [--markdown FILE]`: check, with no model call, that a run matches
  the suite and covers exactly the dataset's cases. Exit 4 when it does not.
- `evalproof rehash RUN SUITE [--out FILE]`: add per-case hashes to a run file from before
  evalproof 0.5.
- `evalproof doctor [SUITE]`: check the coding-agent binary targets and a `command:` target.
  `doctor --cursor-canary [--model MODEL]` makes one real model call, so do not run it unasked.
- `evalproof readiness SUITE [--run FILE] [--json] [--require DIM,...]`: report which of eleven
  dimensions the suite covers. Names: `correctness`, `behaviour`, `quality`, `regression-gate`,
  `reliability`, `cost-latency`, `safety`, `robustness`, `multi-turn`, `load`,
  `production-monitoring`. `load` and `production-monitoring` are out of scope: they need other
  tools.
- `evalproof import FORMAT FILE --out PATH [--system NAME] [--epochs mean|first|all|any|trials] [--threshold N] [--wide] [--markdown FILE] [--quiet]`:
  turn another tool's results into a run file. `adk` and `claude-plugin-eval` are experimental.
  For `jsonl` and `csv`, `--id-field`, `--metric-field`, `--score-field`, `--passed-field` and
  `--system-field` name the columns.
- `evalproof import-dataset FORMAT FILE --out PATH [--suite-out PATH] [--force] [--id-field COLUMN] [--input-field COLUMN] [--reference-field COLUMN] [--tags-field COLUMN]`:
  turn another tool's test cases into a dataset, and with `--suite-out` into a suite with the
  scorers they need. FORMAT is `auto` (read off the file) or a tool name. The four field
  options name the columns of a `jsonl` or `csv` table. `--suite-out` refuses to overwrite a
  file without `--force`.
- `evalproof pairwise BASELINE CANDIDATE --suite SUITE [--limit N] [--no-swap] [--no-cache] [--cache-only] [--markdown FILE] [--json FILE]`:
  ask the judge which of two outputs is better, in both orders unless `--no-swap`. Makes judge
  calls.
<!-- evalproof:end -->
