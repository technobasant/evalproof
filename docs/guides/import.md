---
description: Turn promptfoo, Inspect, DeepEval, lm-eval, garak, Google ADK, Claude Code plugin eval, CSV and JSONL results into run files that compare and the gate read, and Google ADK eval sets into datasets.
---

# Import results from other tools

This page is for anyone who already has results from another eval tool. You do not have to run your evals with evalproof to gate them. `evalproof import` turns the
per-case results another tool wrote into a run file that `compare` and the gate read:

```bash title="Terminal"
evalproof import promptfoo output.json --system openai:gpt-5 --out runs/candidate.json
evalproof import inspect logs/task.eval --out runs/candidate.json
evalproof import deepeval .deepeval/.latest_test_run.json --out runs/candidate.json
evalproof import lm-eval results/gpt2/samples_hellaswag_2026-11-12T10-20-30.jsonl --out runs/gpt2.json
evalproof import garak garak.report.jsonl --threshold 0.6 --out runs/scan.json
evalproof import adk .adk/eval_history/app_evalset_1788817991.2.evalset_result.json --out runs/adk.json
evalproof import claude-plugin-eval aggregate-result.json --system with --out runs/with.json
evalproof import csv results.csv --system a --out runs/a.json       # id,metric,passed,score,...
evalproof import jsonl results.jsonl --wide --id-field case --out runs/a.json  # one column per metric
```

Import each side with the same importer, then gate as usual, for example
`evalproof compare runs/baseline.json runs/candidate.json --gate --metric pass`.

A separate command, `evalproof import-dataset`, turns a test set (not results) into a dataset you can run. It reads
[Google ADK eval sets](#google-adk-eval-sets-as-a-dataset).

## What every importer does

* Case ids are stable between runs of the same dataset (they come from the test's inputs, or
  the tool's own sample id, never from its position), which is what lets two files pair.
  Each case also carries a hash of its id and input as the tool recorded them, so runs imported
  by different importers pair no cases and the gate refuses (exit 4) instead of pairing on a
  coincidence.
* A system error counts as a failure on every metric, as in a native run. The judge cost the
  file reports for that run stays on the metrics.
* Repeats of one case (epochs, repeated tests, generations) collapse by `--epochs`: `mean` (the
  default), `first`, `all` or `any`, or are kept as the case's
  [trials](repeated-trials.md#repeated-trials) with `--epochs trials`, which needs every case to
  have the same number of repeats. Under `mean`, `first` and `any`, a case with at least one
  repeat that ran is not counted as a system error; its errored repeats still score as failures.
  Collapsing sums the judge cost of every repeat. The repeats are ordered by the source's own
  repeat index (promptfoo `repeatIndex`, Inspect `epoch`, ADK `repetition_index`), so trial `n`
  is the `n`-th run, whatever order the file was written in.
* A collapsed case keeps the trace of its first repeat that did not error (of the first repeat,
  with `--epochs first`). When that drops the trace of another repeat, the run records
  `environment.trace_from` as `first repeat`. With `--epochs trials` every trial keeps its own.
* Imported cases carry the tags the importer gives them (garak, promptfoo red-team and DeepEval
  tags are below), in lower case. Tags label a case and are not part of its hash, so adding one
  does not make the case a different case. [`readiness`](readiness.md) counts them with
  `--run`.
* A NaN or infinite score is refused at import, naming the case and metric, because it would
  make the bootstrap report a significant difference that is not there.
* The `adk` and `claude-plugin-eval` importers are experimental: they were written from the
  tool's source or schema and no real result file was available. Using one prints
  `evalproof: the <format> importer is experimental` once and sets
  `environment.import_experimental` to `true` in the run, so a reader of the file can tell.
  Everything else here was read from real files, or says where it follows the source instead.
* `--threshold` applies to `garak` only; any other format refuses it.

## promptfoo

`evalproof import promptfoo output.json` reads the JSON that `promptfoo eval -o output.json`
writes, and the JSONL variant. promptfoo evaluates every test against every provider and prompt in
one file, so a file can hold several systems and `--system` picks one.

* A case id is a hash of the test's `vars` and `description`, never its position, so inserting a
  test or re-ordering the file keeps the ids. Two identical tests get `#2`, `#3`.
* Scores: `pass` (the test's verdict), one continuous score for each `namedScores` key, and a
  binary `assert:<metric or type>` for each assertion. A system error (`failureReason` 2) is an
  error row; a failed assertion is an ordinary failure.
* The rows of one test run with `--repeat` are one case with several repeats. promptfoo writes
  `repeatGroupId` and `repeatIndex` only when `--repeat` is above 1. The repeats are sorted by
  `repeatIndex`, so `--epochs trials` numbers trial `n` as the `n`-th repeat promptfoo ran, even
  when the rows were written in completion order.
* Red-team tests are tagged from their metadata. A `pluginId` gives `safety`, `redteam` and
  `plugin:<id>`; a `strategyId` gives `strategy:<id>`; a `severity` gives `severity:<level>`. Each is
  read from the row's `metadata`, else from `testCase.metadata`. `harmCategory` is a label for people and
  is not used.

The repeat and red-team handling follows promptfoo's source (0.125.1): no real export with a repeat
or a red-team row was available, so those two parts are tested on a file written to that source.

```bash title="Terminal"
evalproof import promptfoo redteam-output.json --system openai:gpt-5 --epochs trials --out runs/redteam.json
```

## Inspect AI

`evalproof import inspect` reads `.json` logs and the default `.eval` zip. One log is one model on
one task, so the system is `eval.model` and there is nothing to choose.

* A sample is a case (its id is `str(sample.id)`, which the task's dataset keeps stable) and each
  epoch is a repeat. Epochs are 1-based in Inspect, so trial `i` is epoch `i + 1`.
* Inspect scorers return their verdict in `value`. `C` is a pass, and `I` or `N` (no answer) a
  fail, all binary. `P` (partial) is 0.5 and has no pass or fail, so the metric compares as
  continuous. A bool is binary and a number is continuous. A dict gives one score per key, named
  `scorer.key`. A list, or a string that is none of those letters, is skipped with a warning.
* A log whose status is not `success` (cancelled, errored) is still imported, with a warning, and
  the status is recorded in `environment.source_status`. An incomplete log can hide a regression
  behind samples that never ran.
* `.eval` logs compressed with zstd need Python 3.14 or newer; otherwise run
  `inspect log convert --to json <file>` first.

### Tool trace

The sample's `events` give each case a trace, so [`tool_called`, `tool_order` and the other
agent checks](testing-agents.md) can run on an imported log.

* A tool event is a tool call: its function, arguments, result, error and working time. A call is
  `ok` when it has neither an `error` nor `failed`.
* A model event is a round. Its token counts are added up, and the reply of the last one is the
  trace's response.
* Inspect stores event strings of more than 100 characters as `attachment://<hash>` in the
  sample's `attachments`. They are resolved on import. A reference that is not in `attachments`
  stays as it is, with one warning.
* A log written in the newer condensed form (`events_data` or `input_refs`, and no `events`) has
  no trace, and one warning says so.
* A log does not say which user turn a call answered, so every call has `turn` 0.

## DeepEval, lm-eval, CSV and JSONL

* DeepEval test runs (`.deepeval/.latest_test_run.json`) give each metric two scores: `<name>` is
  the verdict against DeepEval's threshold (binary) and `<name>:score` the raw score (continuous).
  A metric that raised is undecided, not failed. A test case's own `tags` are kept. Conversational
  cases are skipped with a warning.
* lm-eval `samples_<task>_<time>.jsonl` files (`--log_samples`) are one task for one model; the
  system is the directory the file is in, or `--system`. A document that appears twice under one
  filter is an error, because the harness has no repeats.
* `csv` and `jsonl` read your own long files (one row per case and metric) or, with `--wide`, one
  column per metric. `--id-field`, `--metric-field`, `--score-field`, `--passed-field` and
  `--system-field` name the columns when they are not the defaults.

## garak

`evalproof import garak` reads `garak.<run>.report.jsonl`, the report a garak scan writes (0.9
and later). It imports one case per attempt.

* The id is `<probe>:<12 hex of the prompt's hash>`, numbered `#2`, `#3` when a probe sent the
  same prompt twice. The tags are `safety`, `garak` and `probe:<name>`.
* The generations of a prompt are its repeats, in output order. `--epochs mean` is then the share
  of generations that passed, and `--epochs trials` keeps each generation as a trial.
* There is one metric per detector, named as garak names it. An output passes when its detector
  score is below the threshold, and the metric is 1.0 for a pass and 0.0 for a hit. The raw score
  is kept in the metric's reason. An output a detector could not score (garak writes `null`)
  stays undecided.
* garak writes each attempt twice, and the second row holds the scores. An attempt with no scored
  row is skipped, and one warning counts them.
* The `eval` rows (garak's own totals) are not read, and no trace is imported: a report holds
  prompts and replies, not tool calls.

### The threshold

The threshold is the detector score at which an output counts as a hit. It is `--threshold` when
you give it, else the `run.eval_threshold` the scan was evaluated at, else 0.5, garak's default.
The run records the one it used and where it came from, in `environment.threshold` and
`environment.threshold_source`, which is `cli`, `eval_threshold` or `default`.

```bash title="Terminal"
evalproof import garak garak.report.jsonl --threshold 0.6 --out runs/scan.json
```

Import the baseline and the candidate with the same threshold, or leave it unset for both, so the
two runs score outputs the same way.

## Google ADK eval history

`evalproof import adk` reads `<app>_<eval set>_<time>.evalset_result.json`, the result `adk eval`
saves under `<agents_dir>/<app>/.adk/eval_history/`. This importer is experimental (see above).

* One case per `eval_id`, whose repeats are its runs in `repetition_index` order. The case's input
  is the first invocation's user text, and its output the last invocation's final response.
* One metric per ADK metric, scored as ADK scored it, with no rescaling. ADK's status decides the
  verdict: passed and failed are what they say, `NOT_EVALUATED` is undecided, and `INFORMATIONAL`
  (the efficiency metrics) has a value and no verdict. A `null` score is undecided. A metric on a
  1 to 5 scale (`response_evaluation_score`), the efficiency metrics and any score outside 0 to 1
  are named in a warning, because `mean` and `compare` treat scores as being on one scale.
* The trace comes from the session's events when the file has them, else from each invocation's
  `tool_uses` and `tool_responses` (or its `invocation_events`), each call at the turn of its
  invocation. A tool response with an `error` makes the call not `ok`.
* The suite and, unless `--system` gives a label, the system are the eval set's id. There are no
  tags. Latency, tokens and the judge's `rubric_scores` are not imported.
* Both encodings ADK has written are read: a JSON object, and the JSON string holding that
  object's text that ADK wrote before v1.22.0. Keys may be snake_case or camelCase.

The importer follows ADK's source (v2.14.1), because no real result file was available. A
`*.evalset.json` is a dataset, not a result: `import` refuses it and points to
[`import-dataset`](#google-adk-eval-sets-as-a-dataset).

## Claude Code plugin evals

`evalproof import claude-plugin-eval` reads `aggregate-result.json`, the result `claude plugin
eval` writes (`schemaVersion: 1`). This importer is experimental (see above): the docs show no sample file,
so it follows the writer's schema and the docs' description of it.

* A suite run with `--ablation with-without` holds two arms, `with` and `without` the plugin. Each
  arm is a system. `--system with` or `--system without` picks one; a file that holds both needs
  it, and the error lists the arms the file has. To compare the arms, import the file twice:

  ```bash title="Terminal"
  evalproof import claude-plugin-eval aggregate-result.json --system without --out runs/without.json
  evalproof import claude-plugin-eval aggregate-result.json --system with --out runs/with.json
  evalproof compare runs/without.json runs/with.json
  ```

* One case per `cases[].name` (a repeated name gets `#2`), whose input is `promptMarkdown`. The
  runs of the chosen arm, in list order, are its repeats (`runsPerCase`). A case with no runs in
  the chosen arm is skipped with a warning.
* Each run has a `score` metric, the weighted fraction of its graders that passed, which passes
  when the file says the run passed (a score of 1), and one metric per grader, 1.0 or 0.0 with the grader's explanation as the reason. A grader
  with `scored: false` (a `withOnly` indicator such as "the Skill tool was called") is shown and
  has no verdict, and `withOnly` graders are left out of the `without` arm.
* The file's `costUsd` includes the judge's cost. The run's cost is `costUsd` minus
  `judgeCostUsd`, and the judge's share is on the `score` metric, where `compare` adds it back, so
  the total in `compare` is the file's `costUsd`.
* A run with an `error`, or one a mock `aborted`, counts as a failure on every metric. The docs
  say some errored runs are still graded on what they produced, and the import does not use
  that grade.
* A suite that did not finish (`partial`) is imported with a warning that gives the reason
  (`cost_ceiling`, `interrupted` or `auth_failed`), because the cases it never reached are
  missing. The run records `environment.partial` and `environment.partial_reason`.
* When the cost ceiling skipped a run's judge graders, its score is not comparable with the
  others. The import names how many runs that is and records `environment.skipped_paid_graders`.
* The file holds no reply text, no tokens and no trace.

## Google ADK eval sets as a dataset

`evalproof import-dataset adk FILE --out cases.jsonl` turns a test set, which has no results,
into a dataset for [`evalproof run`](../reference/commands.md#evalproof-run). It is stable. It reads
exactly two shapes:

* an `EvalSet` (`*.evalset.json`, an object with `eval_set_id` and `eval_cases`);
* the flat legacy list from `*.test.json` files: `[{"query", "expected_tool_use": [{"tool_name",
  "tool_input"}], "reference"}]`.

Any other shape is refused, including ADK's other legacy shape (`{name, data, initial_session}`)
and an `*.evalset_result.json` (import that with `evalproof import adk`).

| ADK | evalproof case |
|---|---|
| `eval_id` | `id` (the legacy list has none: the file name without its last extension, a dash and the item's 0-based position) |
| the first invocation's `user_content` text | `input` |
| the later invocations' text | `turns` |
| the last invocation's `final_response` text | `reference` |
| `session_input.state` | `state`, when it is not empty |
| each invocation's `intermediate_data.tool_uses` | one `tool_order` check in `expect`, `match: exact`, with that invocation's `turn` and its calls as `{name, args}` |
| `rubrics` on the eval case and on its invocations | `rubrics`, as `{id, text, type}` |

The `tool_order` checks run with the [`checks` scorer](testing-agents.md) on a case's trace. The
rubrics are read by an `llm_judge` rubric set with `case_rubrics: true`, together with the
rubrics you put in the suite; a case rubric with the same id as a suite rubric replaces it.
`rubric_types` filters on the `type`.

```bash title="Terminal"
evalproof import-dataset adk evals/home_automation.evalset.json --out datasets/home.jsonl
```

```yaml title="suite.yaml (excerpt)"
dataset: datasets/home.jsonl
scorers:
  - name: tools
    type: checks               # runs each case's imported tool_order checks
  - name: quality
    type: llm_judge
    params:
      case_rubrics: true       # reads each case's imported rubrics
      rubrics: ["The reply is polite."]
```

* A case that uses `conversation_scenario` (user simulation) has no fixed turns to replay and is
  skipped. One warning names every skipped case, and a file with no case left is an error.
* `app_name`, `user_id`, `final_session_state` and the text of tool responses are not read.
  Rubrics kept in an `eval_config.json` are another file and are not read.
* `--out` is required, and an existing file is replaced, as with `import --out`. The output loads
  with `load_dataset`; the command checks each case before it writes.
* Rubrics with the same id and text on the case and an invocation are kept once; two texts for
  one id are an error, so that no rubric is lost without notice.

## Next steps

* Gate the imported runs in CI: [In CI](ci.md#in-ci).
* How runs are paired: [Concepts](../reference/concepts.md#when-the-cases-change).
