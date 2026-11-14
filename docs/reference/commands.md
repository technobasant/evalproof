---
description: Every evalproof command and flag, with its type and default, and the exit codes each command returns.
---

# Commands

Every `evalproof` subcommand, its arguments and every flag with its type and default. It is for anyone writing a CI step or a script around the tool, and a test checks it against the command line so it cannot drift.

`evalproof --version` prints the version and exits. A flag marked "flag" takes no value. A flag marked "repeatable" may be given several times. Where a default is "none" and the suite sets the value, the suite's value applies.

| Command | What it does |
|---|---|
| [`evalproof init`](#evalproof-init) | write a starter suite, a CI workflow and an agent skill into a repository |
| [`evalproof run`](#evalproof-run) | run a suite and write a run artifact |
| [`evalproof import`](#evalproof-import) | turn another tool's results into a run artifact you can compare and gate |
| [`evalproof compare`](#evalproof-compare) | compare two runs, or several candidates with one baseline; `--gate` turns the result into an exit code |
| [`evalproof calibrate`](#evalproof-calibrate) | measure a judge against gold labels and correct its rate |
| [`evalproof report`](#evalproof-report) | summarise one run |
| [`evalproof verify`](#evalproof-verify) | check that a run file matches the suite and dataset, without calling a model |
| [`evalproof rehash`](#evalproof-rehash) | add per-case hashes to a run file written before evalproof 0.5, without calling a model |
| [`evalproof doctor`](#evalproof-doctor) | check the coding-agent CLI targets, without calling a model |
| [`evalproof readiness`](#evalproof-readiness) | report which parts of quality a suite covers and which it leaves untested |
| [`evalproof pairwise`](#evalproof-pairwise) | judge baseline vs candidate outputs head to head |

## evalproof init

Writes a starter suite, a CI workflow and an agent skill into a repository. It makes no model call. `--ci keyless` (the default) writes a workflow that compares committed run files; `--ci api-key` writes one that runs the suite with a secret; `--ci none` writes no workflow. Files that already exist are kept unless you pass `--force`.

```bash title="Terminal"
evalproof init [--dir DIR] [--judge TARGET] [--ci {keyless,api-key,none}] [--cli-auth {api-key,oauth-token}] [--no-skill] [--force]
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--dir` | string | `.` | where to write; the default is the current directory |
| `--judge` | string | none | model for the judge and the system, for example `claude-code:haiku` |
| `--ci` | one of `keyless`, `api-key`, `none` | `keyless` | `keyless`: CI compares committed run files, with no model calls; `api-key`: CI runs the suite with a secret; `none`: no workflow |
| `--cli-auth` | one of `api-key`, `oauth-token` | `api-key` | with `--ci api-key` and a coding-agent CLI judge, what the workflow passes the CLI: its API key (`api-key`), or `oauth-token`: `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, for `claude-code` in your own repository and CI only |
| `--no-skill` | flag | off | do not write the agent skill |
| `--force` | flag | off | overwrite files that already exist |

```bash title="Example"
evalproof init --judge claude-code:haiku
```

Exit 0, or 2 for a usage or configuration error.

## evalproof run

Runs a suite: every case goes through the system, every scorer scores the output, and the result is written as a [run file](run-file.md). With several `--variant` options or `--all-variants`, each variant writes its own file, `runs/<variant>.json`. Without `--out` a single run is written to `runs/<suite>-<start time>.json`.

```bash title="Terminal"
evalproof run SUITE [options]
```

| Argument | Required | What it is |
|---|---|---|
| `SUITE` | yes | the suite file (`suite.yaml`) |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--out` | string | none | where to write the run (with `--variant`, only when exactly one variant runs) |
| `--limit` | int | none | run only the first N cases of the dataset |
| `--no-cache` | flag | off | do not read or write the response cache |
| `--trials` | int, 1 to 100 | none | run each case K times (1 to 100), overriding the suite's trials |
| `--system` | string | none | override the system target |
| `--system-prompt-file` | string | none | override the system prompt |
| `--label` | string | none | name this system in reports |
| `--variant` | string, repeatable | none | run one of the suite's variants and write `runs/<variant>.json`; repeat to run several, one run file each |
| `--all-variants` | flag | off | run every variant the suite defines, one run file each |
| `--markdown` | string | none | also append the report to this file |
| `--quiet` | flag | off | print no progress line and no report on stdout (`--markdown` still writes it) |
| `--show` | int | `10` | list up to N failed cases in the report (default 10; 0 hides the list) |
| `--gate` | flag | off | apply the suite's absolute checks (`gate.min_pass_rate`, `gate.must_pass`) to this run, with no baseline; exit 1 when one fails |

```bash title="Example"
evalproof run evals/suite.yaml --out runs/candidate.json --trials 3
```

Exit 0. With `--gate`: 1 when an absolute check fails, 4 when the gate cannot apply one. Exit 2 for a configuration error and when every case errored. See [Exit codes](exit-codes.md). See also: [Concepts](concepts.md#run), [Repeated trials](../guides/repeated-trials.md#repeated-trials), [Testing agents](../guides/testing-agents.md#variants-one-suite-several-builds).

## evalproof import

Turns another tool's results into a run file that `compare` and the gate can read. See [Import results from other tools](../guides/import.md#import-results-from-other-tools).

```bash title="Terminal"
evalproof import FORMAT FILE --out PATH [options]
```

| Argument | Required | What it is |
|---|---|---|
| `FORMAT` | yes | the tool or table layout the file is from: `promptfoo`, `inspect`, `deepeval`, `lm-eval`, `jsonl`, `csv` |
| `FILE` | yes | the results file to import |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--out` | string, required | none | where to write the run artifact |
| `--system` | string | none | pick one system out of a file that holds several; otherwise the run's label |
| `--epochs` | one of `mean`, `first`, `all`, `any`, `trials` | `mean` | how repeats of one case collapse: mean (default), first, all (every repeat must pass) or any; or trials, which keeps each repeat as its own trial of the case |
| `--markdown` | string | none | also append the report to this file |
| `--quiet` | flag | off | do not print the report to stdout (`--markdown` still writes it) |
| `--id-field` | string | none | column holding the case id (default: id) |
| `--metric-field` | string | none | column holding the metric name (default: metric) |
| `--score-field` | string | none | column holding the continuous score (default: score) |
| `--passed-field` | string | none | column holding pass/fail (default: passed) |
| `--system-field` | string | none | column holding the system (default: system) |
| `--wide` | flag | off | one row per case, one column per metric |

```bash title="Example"
evalproof import promptfoo results.json --out runs/promptfoo.json
```

Exit 0, or 2 for a usage error such as an unreadable file.

## evalproof compare

Compares each candidate with the baseline on the cases both runs share. With `--gate` the result becomes an exit code. Several candidates give one table, and `--gate` then corrects the p-values for all candidates and gated metrics at once and exits with the worst code (2, then 4, 1, 3, 0). Gate settings come from `--suite`; every flag below overrides the suite's `gate` block. Without `--gate` a comparison of two runs that cannot be compared is a usage error (exit 2).

```bash title="Terminal"
evalproof compare BASELINE CANDIDATE [CANDIDATE ...] [options]
```

| Argument | Required | What it is |
|---|---|---|
| `BASELINE` | yes | the baseline run file |
| `CANDIDATE` | yes | one run file, or several for a matrix comparison |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--suite` | string | none | take gate settings from this suite |
| `--metric` | string, repeatable | none | scorer to gate on; repeat to gate several (Holm-corrected) |
| `--max-drop` | float | none | largest drop in the gated metric the gate tolerates (default: the suite's `gate.max_drop`, else 0.02) |
| `--alpha` | float | none | significance level (default: the suite's `gate.alpha`, else 0.05) |
| `--aggregate` | one of `mean`, `all`, `any` | none | when the runs repeat each case (trials > 1), how a case's trials become one value on a pass/fail metric: mean (the pass rate, default), all (passed every trial) or any (passed at least once); overrides `gate.aggregate` |
| `--min-cases` | int | none | fewest paired cases the gate accepts (default: the suite's `gate.min_cases`, else 20) |
| `--max-p95-ms` | float | none | highest p95 latency of the candidate, in milliseconds; over it the gate exits 3 |
| `--max-cost` | float | none | highest total cost of the candidate, in US dollars; over it the gate exits 3 |
| `--min-pass-rate` | float | none | lowest candidate pass rate on each gated metric; errored cases count as failures |
| `--must-pass` | string, repeatable | none | a case tag whose cases must all pass every gated metric; repeat for several |
| `--max-new-failures` | int | none | most cases that may go from pass to fail on a binary gated metric, whatever the p-value; 0 suits deterministic scorers |
| `--allow-judge-change` | flag | off | gate even if the judge (target, CLI, isolation or resolved model) differs between the runs; it is refused otherwise. A new CLI version alone only warns |
| `--gate` | flag | off | exit non-zero when the gate fails |
| `--show` | int | `5` | show both outputs for up to N regressed cases of the gated metrics (default 5; 0 hides them) |
| `--seed` | int | `0` | seed of the bootstrap and the sign-flip test |
| `--markdown` | string | none | also append the report to this file (for example `$GITHUB_STEP_SUMMARY`) |
| `--json` | string | none | write the comparison as JSON to this file |

```bash title="Example"
evalproof compare runs/baseline.json runs/candidate.json --suite evals/suite.yaml --gate
```

Exit 0 to 4 under `--gate`; see [Exit codes](exit-codes.md). Without `--gate`, 0 or 2. See also: [Concepts](concepts.md#compare-and-gate), [In CI](../guides/ci.md#in-ci), [Repeated trials](../guides/repeated-trials.md#several-candidates).

## evalproof calibrate

Measures a judge against gold labels and corrects its pass rate. Give exactly one gold source: `--labels` or `--gold-scorer`. See [Concepts](concepts.md#calibrate).

```bash title="Terminal"
evalproof calibrate RUN --judge-scorer NAME (--labels FILE | --gold-scorer NAME) [options]
```

| Argument | Required | What it is |
|---|---|---|
| `RUN` | yes | the run file whose judge to calibrate |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--judge-scorer` | string, required | none | name of the `llm_judge` scorer in the run whose judgements to measure |
| `--labels` | string | none | JSONL of `{id, label}` human labels |
| `--gold-scorer` | string | none | a trusted scorer in the same run |
| `--gold-sample` | int | none | use a random sample of this size as the gold set |
| `--seed` | int | `0` | seed of the bootstrap and of the `--gold-sample` draw |
| `--alpha` | float | `0.05` | one minus the confidence level of the intervals (0.05 gives 95%) |
| `--markdown` | string | none | also append the report to this file |
| `--json` | string | none | write the calibration as JSON to this file |

```bash title="Example"
evalproof calibrate runs/candidate.json --judge-scorer judge --labels human.jsonl
```

Exit 0, or 2 for a usage error.

## evalproof report

Prints the summary of one run file.

```bash title="Terminal"
evalproof report RUN [--markdown FILE] [--show N]
```

| Argument | Required | What it is |
|---|---|---|
| `RUN` | yes | the run file to summarise |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--markdown` | string | none | also append the report to this file |
| `--show` | int | `10` | list up to N failed cases (default 10; 0 hides the list) |

```bash title="Example"
evalproof report runs/candidate.json --show 20
```

Exit 0, or 2 for a usage error.

## evalproof verify

Checks, with no model call, that a run file was made with this suite, covers exactly the dataset's cases as they are now, and has the scores its deterministic scorers give the stored outputs.

```bash title="Terminal"
evalproof verify RUN SUITE [--markdown FILE]
```

| Argument | Required | What it is |
|---|---|---|
| `RUN` | yes | the run file to check |
| `SUITE` | yes | the suite file it should match |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--markdown` | string | none | also append the verdict to this file |

```bash title="Example"
evalproof verify runs/baseline.json evals/suite.yaml
```

Exit 0 when the run matches, 4 when it does not, 2 for a usage error. See also: [Concepts](concepts.md#when-the-cases-change).

## evalproof rehash

Adds per-case hashes to a run file written before evalproof 0.5, from the dataset it scored, so it can be gated against a changed dataset. It makes no model call and is safe to run again on a file that already has hashes.

```bash title="Terminal"
evalproof rehash RUN SUITE [--out FILE]
```

| Argument | Required | What it is |
|---|---|---|
| `RUN` | yes | the run file to add hashes to |
| `SUITE` | yes | the suite file whose dataset the run scored |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--out` | string | none | write the result here instead of overwriting RUN |

```bash title="Example"
evalproof rehash runs/baseline.json evals/suite.yaml
```

Exit 0 on success, 4 when the dataset does not match the run, 2 for a usage error.

## evalproof doctor

Checks each coding-agent CLI target (`claude-code`, `codex`, and the experimental `opencode`, `cursor` and `copilot`): binary, version, login and the isolation flags. It makes no model call. With a suite it checks the targets that suite uses; without one it surveys every CLI on the machine.

```bash title="Terminal"
evalproof doctor [SUITE]
```

| Argument | Required | What it is |
|---|---|---|
| `SUITE` | no | optional: check the targets this suite uses (default: every CLI found) |

```bash title="Example"
evalproof doctor evals/suite.yaml
```

Exit 0 when every checked target is ready (or, without a suite, when at least one CLI is usable), 1 otherwise. See also: [Providers](../guides/providers.md#use-your-claude-code-or-codex-login).

## evalproof readiness

Reads a suite and its dataset (and a run, if given) and reports, for each dimension of AI quality, whether the suite covers it, covers it in part, leaves it untested or leaves it to another kind of tool, with the evidence. It makes no model call. See [Readiness](../guides/readiness.md).

```bash title="Terminal"
evalproof readiness SUITE [--run FILE] [--json] [--require DIM,...]
```

| Argument | Required | What it is |
|---|---|---|
| `SUITE` | yes | the suite file to assess |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--run` | string | none | add what this run file shows (traces, errors) |
| `--json` | flag | off | print JSON instead of a table |
| `--require` | string, repeatable | none | exit 1 unless these dimensions are covered (comma-separated; may repeat) |

```bash title="Example"
evalproof readiness evals/suite.yaml --require correctness,gate
```

Exit 0, or 1 when a `--require`d dimension is not covered, 2 for a usage error.

## evalproof pairwise

Asks the suite's judge which of two outputs is better for each case, in both orders unless `--no-swap` is set.

```bash title="Terminal"
evalproof pairwise BASELINE CANDIDATE --suite SUITE [options]
```

| Argument | Required | What it is |
|---|---|---|
| `BASELINE` | yes | the baseline run file |
| `CANDIDATE` | yes | the candidate run file |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--suite` | string, required | none | the suite whose `judge` compares the outputs |
| `--limit` | int | none | compare only the first N shared cases, in case-id order |
| `--no-swap` | flag | off | judge each pair in one order only (by default each pair is judged in both orders) |
| `--no-cache` | flag | off | do not read or write the response cache |
| `--markdown` | string | none | also append the report to this file |
| `--json` | string | none | write one row per case as JSON to this file |

```bash title="Example"
evalproof pairwise runs/baseline.json runs/candidate.json --suite evals/suite.yaml
```

Exit 0, or 2 for a usage error.

## Next steps

* What the exit codes mean for a pipeline: [Exit codes](exit-codes.md).
* The suite file the commands read: [Configuration](configuration.md).
* Put the commands in a pipeline: [In CI](../guides/ci.md#in-ci).
