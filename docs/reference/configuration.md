---
description: Every key of the suite file with its type and default, the scorer parameters, an annotated example and what the suite fingerprint covers.
---

# Configuration

A suite is one YAML file that names the dataset, the system under test, the scorers and the gate. This page lists every key it accepts, for anyone writing or reviewing a suite, and a test checks it against the code that loads the file.

Unknown keys are errors, not warnings. The message names the file and the key, and suggests the nearest valid one: `suite.yaml: gate.max_dorp: unknown key (did you mean 'max_drop'?)`.

## An annotated suite

This is the suite of the support-assistant example, with its comments explained. Paths are relative to the directory of the suite file.

```yaml title="suite.yaml"
name: support-qa                  # the suite's name, shown in reports
dataset: cases.jsonl              # one JSON case per line
seed: 0

system:                           # the model or program under test
  target: ollama:qwen3:1.7b
  label: "qwen3:1.7b + grounded prompt"
  system_prompt_file: prompt.baseline.md    # the prompt text lives in a file
  prompt_template: "{question}"             # {field} reads a field of the case
  params: {temperature: 0, seed: 0, max_tokens: 200, think: false}
  timeout_s: 180

judge:                            # the model behind llm_judge scorers
  target: ollama:qwen3:8b
  params: {temperature: 0, seed: 0, max_tokens: 256, think: false}
  timeout_s: 180

scorers:
  - name: answer_key              # a regex per question, kept in the case
    type: regex
    params: {any_of_field: accept, none_of_field: reject}
  - name: judge
    type: llm_judge
    params: {rubric_file: rubric.md}

calibration:                      # measure the judge against the answer key
  judge_scorer: judge
  gold_scorer: answer_key
  gold_sample: 30

gate:                             # fail on a drop that is large and significant
  metric: judge
  max_drop: 0.03
  alpha: 0.05
  min_cases: 40

cache_dir: .evalproof/cache
```

## Suite keys

The top level of the file. `name`, `dataset`, `system` and `scorers` are required.

| Key | Type | Default | What it does |
|---|---|---|---|
| `name` | string | required | the suite's name, shown in reports and recorded in every run |
| `dataset` | string | required | path to the JSONL dataset, relative to the suite file |
| `system` | object | required | the target under test; see [System and judge](#system-and-judge) |
| `scorers` | list | required | one entry per scorer; names must be unique; see [Scorers](#scorers) |
| `judge` | object | none | the target behind `llm_judge` scorers and `pairwise`; required when a scorer has type `llm_judge`; same keys as `system` |
| `calibration` | object | none | how to calibrate the judge; see [Calibration](#calibration) |
| `gate` | object | none | what `compare --gate` decides on; see [Gate](#gate) |
| `variants` | list | none | named builds of the same suite; see [Variants](#variants) |
| `seed` | integer | `0` | recorded in the run file and part of the suite fingerprint |
| `cache_dir` | string | `.evalproof/cache` | where model responses are cached, relative to the suite file |
| `concurrency` | integer | `1` | how many cases run at once; at least 1 |
| `trials` | integer | `1` | how many times each case runs, 1 to 100; `evalproof run --trials` overrides it |
| `base_dir` | string | `.` | set by the loader to the directory of the suite file; leave it out |

## System and judge

`system` and `judge` take the same keys. `target` says what to call:

| Target | Calls |
|---|---|
| `ollama:<model>` or `anthropic:<model>` | the two native adapters |
| `<profile>:<model>` | an OpenAI-compatible service: `openai`, `azure`, `gemini`, `mistral`, `groq`, `together`, `fireworks`, `deepseek`, `xai`, `openrouter`, `cerebras`, `vllm`, `lmstudio` or `llamacpp`; the model name is everything after the first colon |
| `openai-compatible:<model>` | any other OpenAI-compatible server; needs `base_url` |
| `python:<module>:<function>` | a Python function in your project; see [Testing agents](../guides/testing-agents.md#testing-agents) |
| `http://...` or `https://...` | an HTTP endpoint |
| `claude-code:<model>`, `codex:<model>` | a coding agent you have logged in; see [Providers](../guides/providers.md#use-your-claude-code-or-codex-login) |

| Key | Type | Default | What it does |
|---|---|---|---|
| `target` | string | required | what to call, as in the table above |
| `label` | string | none | how reports name this system; the target when unset |
| `system_prompt` | string | none | the system prompt |
| `prompt_template` | string | `{input}` | the user message; `{field}` reads a field of the case |
| `system_prompt_file` | string | none | a file holding the system prompt, relative to the suite file; replaces `system_prompt` |
| `prompt_template_file` | string | none | a file holding the prompt template; replaces `prompt_template` |
| `params` | object | see below | generation parameters; see [Generation parameters](#generation-parameters) |
| `timeout_s` | number | `120.0` | seconds to wait for one call |
| `base_url` | string | none | the endpoint, for a gateway or a regional or self-hosted server |
| `api_key_env` | string | none | the environment variable that holds the key, instead of the provider's usual one |
| `headers` | object | none | extra HTTP headers on every call; never written to a run file or the cache |
| `pricing` | object | see below | cost per million tokens; see [Pricing](#pricing) |

!!! warning "Keep keys out of the suite file"

    A suite file is usually committed. Put the key in an environment variable and name it with `api_key_env`; a secret placed in `headers` would be committed too.

### Generation parameters

| Key | Type | Default | What it does |
|---|---|---|---|
| `temperature` | number | `0.0` | sampling temperature |
| `seed` | integer | `0` | sampling seed; each trial of a case adds its trial number, so repeated trials of a local model differ |
| `max_tokens` | integer | `512` | the most tokens to generate |
| `think` | boolean | `false` | for reasoning models: off by default for speed and determinism; `null` leaves the model's own setting |
| `extra` | object | none | extra fields merged into the request body (into `options` for Ollama) |

### Pricing

| Key | Type | Default | What it does |
|---|---|---|---|
| `input_per_mtok` | number | `0.0` | US dollars per million input tokens |
| `output_per_mtok` | number | `0.0` | US dollars per million output tokens |

Local models cost nothing unless you set a price.

## Scorers

Each entry of `scorers` has a `name`, a `type` and a `params` object. The name is what the gate, `calibrate` and the reports refer to. A `checks` or `python` scorer has its params checked when the suite loads; the others are checked when the run starts.

| Key | Type | Default | What it does |
|---|---|---|---|
| `name` | string | required | unique within the suite |
| `type` | string | required | one of the types below |
| `params` | object | none | the parameters of that type |

| Type | What it scores | Pass/fail or continuous |
|---|---|---|
| `exact_match` | the output equals a case field | pass/fail |
| `contains` | the output contains given strings | pass/fail |
| `regex` | the output matches a pattern | pass/fail |
| `json_schema` | the output is JSON that matches a schema | pass/fail |
| `numeric_tolerance` | a number in the output is close to the expected one | pass/fail |
| `recall_at_k`, `mrr`, `ndcg` | the ranked ids the system retrieved against the relevant ids | continuous, pass/fail with `pass_threshold` |
| `llm_judge` | a model grades the output against a rubric | pass/fail |
| `checks` | checks on the trace: tool calls, order, arguments, replies | pass/fail, or the fraction of checks passed |
| `python` | your function | a `Score`, a bool, a number from 0 to 1 or a `(bool, reason)` pair |

The parameters of each type:

| Type | Parameter | Default | What it does |
|---|---|---|---|
| `exact_match` | `field` | `reference` | the case field to compare with |
| `exact_match` | `normalize` | `true` | compare after lower-casing, collapsing whitespace and trimming surrounding punctuation |
| `contains` | `any_of` | none | at least one of these strings must appear |
| `contains` | `all_of` | none | every one of these strings must appear |
| `contains` | `any_of_field` | none | a case field holding the `any_of` list |
| `contains` | `all_of_field` | none | a case field holding the `all_of` list |
| `contains` | `case_sensitive` | `false` | match case |
| `regex` | `pattern` | none | the pattern that must match |
| `regex` | `any_of_field` | none | a case field holding patterns, one of which must match |
| `regex` | `none_of` | none | patterns that must not match |
| `regex` | `none_of_field` | none | a case field holding patterns that must not match |
| `regex` | `flags` | `i` | `i` ignores case, `m` makes `^` and `$` match at each line |
| `json_schema` | `schema` | none | the JSON schema |
| `json_schema` | `schema_field` | none | a case field holding the schema |
| `numeric_tolerance` | `field` | `reference` | the case field holding the expected number |
| `numeric_tolerance` | `abs_tol` | `0.0` | absolute tolerance |
| `numeric_tolerance` | `rel_tol` | `0.0` | relative tolerance |
| `numeric_tolerance` | `pick` | `first` | which number in the output counts: `first`, `last` or `any` |
| `recall_at_k`, `mrr`, `ndcg` | `k` | `10` | how many ranked ids to look at; at least 1 |
| `recall_at_k`, `mrr`, `ndcg` | `field` | `relevant` | the case field holding the relevant ids |
| `recall_at_k`, `mrr`, `ndcg` | `pass_threshold` | none | the value at or above which a case passes; with none the score is continuous only |
| `llm_judge` | `rubric` | built in | the grading instructions |
| `llm_judge` | `rubric_file` | none | a file holding the rubric, relative to the suite file |
| `llm_judge` | `max_retries` | `1` | retries when the judge's reply cannot be read |
| `checks` | `checks` | none | checks applied to every case, in the vocabulary of [Testing agents](../guides/testing-agents.md#testing-agents) |
| `checks` | `field` | `expect` | the case field holding per-case checks |
| `checks` | `on_missing` | `pass` | what a case with no checks scores: `pass` or `skip` |
| `checks` | `mode` | `all` | `all` needs every check to pass; `fraction` scores the share that passed |
| `python` | `fn` | required | the function, as `module:function` |
| `python` | `deterministic` | `false` | declare that the function always returns the same score for the same output, so `verify` can re-score it offline |
| `python` | `threshold` | `0.5` | a number from 0 to 1: a function that returns a number passes at or above it |

## Calibration

How to calibrate a judge, with `evalproof calibrate` or from Python. Give exactly one of `labels` and `gold_scorer`.

| Key | Type | Default | What it does |
|---|---|---|---|
| `judge_scorer` | string | required | the name of the `llm_judge` scorer to measure |
| `labels` | string | none | a JSONL file of human labels, `{id, label}` per line |
| `gold_scorer` | string | none | the name of a trusted scorer to use as gold labels |
| `gold_sample` | integer | none | draw this many cases at random as the gold set, to mimic a small hand-reviewed set |

## Gate

The gate fails only when a drop is both larger than `max_drop` and statistically significant. Name at least one scorer with `metric`, `metrics` or both.

| Key | Type | Default | What it does |
|---|---|---|---|
| `metric` | string | none | the scorer to gate on |
| `metrics` | list of strings | none | more scorers to gate on; with several, p-values are Holm-corrected so the chance of a false failure stays at `alpha` |
| `max_drop` | number | `0.02` | the largest drop that is tolerated |
| `alpha` | number | `0.05` | the significance level |
| `min_cases` | integer | `20` | the fewest paired cases the gate accepts; fewer is exit 4 |
| `max_p95_latency_ms` | number | none | the highest candidate p95 latency; over it the gate exits 3 |
| `max_cost_usd` | number | none | the highest candidate cost; over it the gate exits 3 |
| `n_boot` | integer | `5000` | bootstrap resamples |
| `allow_judge_change` | boolean | `false` | gate even when the judge differs between the runs; refused otherwise |
| `min_pass_rate` | number or object | none | the lowest candidate pass rate per gated metric, between 0 and 1; a number applies to every gated metric and an object maps scorer names to floors; errored cases count as failures |
| `must_pass` | list of strings | none | case tags: a case with one of them must pass every gated metric |
| `max_new_failures` | integer | none | the most cases that may go from pass to fail on a pass/fail metric, whatever the p-value; 0 suits deterministic scorers |
| `aggregate` | string | `mean` | with repeated trials, how a case's trials become one value on a pass/fail metric: `mean`, `all` or `any`; ignored at one trial per case |

`min_pass_rate` and `must_pass` are the absolute checks: they need no baseline, and `evalproof run --gate` applies just these. The [exit codes](exit-codes.md) say what each failure returns.

## Variants

A variant is one way to run the same cases, scorers and gate with another model, prompt or flag. See [Testing agents](../guides/testing-agents.md#variants-one-suite-several-builds).

| Key | Type | Default | What it does |
|---|---|---|---|
| `name` | string | required | unique; only `a-z`, `0-9`, `.`, `_` and `-`; the run file is `runs/<name>.json` |
| `vars` | object | none | what differs, as plain JSON values; a python or http system receives them, a model prompt reads them as template fields |
| `system` | object | none | a partial `system` block merged over the suite's; `params` and `pricing` merge key by key, every other key (`headers` included) is replaced |

## Rules checked when the suite loads

* Scorer names are unique, and so are variant names.
* An `llm_judge` scorer needs a `judge` target.
* Every name in `gate.metric`, `gate.metrics`, `gate.min_pass_rate` and `calibration` (`judge_scorer`, `gold_scorer`) is a scorer name.
* A gate names at least one scorer.
* `calibration` has exactly one of `labels` and `gold_scorer`.
* `concurrency` is at least 1, and `trials` is 1 to 100.
* `gate.min_pass_rate` is between 0 and 1, and `gate.max_new_failures` is at least 0.
* A variant's `system` override must produce a valid target.

## What the suite fingerprint covers

Every run stores a short hash of what makes its scores, `suite_hash`, and `evalproof verify` and the gate compare it. It covers the system's `target`, `system_prompt`, `prompt_template`, `params` and `base_url`; the same five for the judge; every scorer; the `seed`; and, for a variant, its name and `vars`.

It does not cover the dataset (`verify` checks that case by case), nor anything that only names or paces the run: `label`, `timeout_s`, `headers`, `api_key_env`, `pricing`, the `gate`, `cache_dir`, `concurrency`, `trials` and the suite `name`.

## Next steps

* Run the suite: [Commands](commands.md#evalproof-run).
* How the gate reads these settings: [Concepts](concepts.md#compare-and-gate).
* What a run file records: [Run file format](run-file.md).
