---
description: Every evalproof command and flag, with its type and default, and the exit codes each command returns.
---

# Commands

Every `evalproof` subcommand, its arguments and every flag with its type and default. It is for anyone writing a CI step or a script around the tool, and a test checks it against the command line so it cannot drift.

`evalproof --version` prints the version and exits. A flag marked "flag" takes no value. A flag marked "repeatable" may be given several times. Where a default is "none" and the suite sets the value, the suite's value applies. [Upgrading](../about/upgrading.md) lists the flags and behaviour that changed between releases.

| Command | What it does |
|---|---|
| [`evalproof init`](#evalproof-init) | write a starter suite, a CI workflow and an agent skill into a repository |
| [`evalproof run`](#evalproof-run) | run a suite and write a run artifact |
| [`evalproof import`](#evalproof-import) | turn another tool's results into a run artifact you can compare and gate |
| [`evalproof import-dataset`](#evalproof-import-dataset) | turn another tool's test set into a dataset you can run |
| [`evalproof compare`](#evalproof-compare) | compare two runs, or several candidates with one baseline; `--gate` turns the result into an exit code |
| [`evalproof calibrate`](#evalproof-calibrate) | measure a judge against gold labels and correct its rate |
| [`evalproof report`](#evalproof-report) | summarise one run |
| [`evalproof verify`](#evalproof-verify) | check that a run file matches the suite and dataset, without calling a model |
| [`evalproof rehash`](#evalproof-rehash) | add per-case hashes to a run file written before evalproof 0.5, without calling a model |
| [`evalproof doctor`](#evalproof-doctor) | check the coding-agent CLI targets and a command target, without calling a model |
| [`evalproof readiness`](#evalproof-readiness) | report which parts of quality a suite covers and which it leaves untested |
| [`evalproof hook`](#evalproof-hook) | install or remove a git pre-push hook that runs `evalproof verify` |
| [`evalproof pairwise`](#evalproof-pairwise) | judge baseline vs candidate outputs head to head |

## evalproof init

Writes a starter suite, a CI workflow and an agent skill into a repository. It makes no model call. `--ci keyless` (the default) writes a workflow that compares committed run files; `--ci api-key` writes one that runs the suite with a secret; `--ci none` writes no workflow. `--platform github` (the default) writes `.github/workflows/evalproof.yml`; `--platform gitlab` writes `.gitlab-ci.evalproof.yml` and prints the two `include:` lines to add to your `.gitlab-ci.yml`, which `init` never edits. `--platform bitbucket` (experimental) writes `bitbucket-pipelines.yml`, or, when that file already exists and was not written by `init`, `evals/bitbucket-pipelines.snippet.yml` for you to merge by hand. Files that already exist are kept unless you pass `--force`.

It also adds an evalproof section to the repository's `AGENTS.md`, so any coding agent that reads that file knows how to use evalproof. That file is merged, never overwritten, with or without `--force`: a missing file is created, a file without the section gets it appended after one blank line, and a file that has it gets only the text between the `<!-- evalproof:begin -->` and `<!-- evalproof:end -->` markers replaced. Every byte outside the markers stays as it was, including the file's line endings. If the markers are damaged (one without the other, repeated, or in the wrong order), `init` stops with exit 2 and writes nothing. `--agent-files-only` writes the section and the skill and nothing else. [Coding agents](../guides/coding-agents.md) has the details.

`--from FILE` starts from test cases you already have in another tool: `evals/cases.jsonl` is those cases and `evals/suite.yaml` has the scorers they need, in place of the sample cases. Everything else `init` writes is unchanged. The file is read before anything is written, so a file that cannot be imported (exit 2) leaves the directory as it was. See [Import test cases from other tools](../guides/import-test-cases.md#start-a-repository-from-existing-cases).

```bash title="Terminal"
evalproof init [--dir DIR] [--judge TARGET] [--ci {keyless,api-key,none}] [--platform {github,gitlab,bitbucket}] [--cli-auth {api-key,oauth-token}] [--from FILE] [--from-format FORMAT] [--no-skill] [--no-agents-md] [--agent-files-only] [--force]
```

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--dir` | string | `.` | where to write; the default is the current directory |
| `--judge` | string | none | model for the judge and the system, for example `claude-code:haiku` |
| `--ci` | one of `keyless`, `api-key`, `none` | `keyless` | `keyless`: CI compares committed run files, with no model calls; `api-key`: CI runs the suite with a secret; `none`: no workflow |
| `--platform` | one of `github`, `gitlab`, `bitbucket` | `github` | where CI runs: `github` writes `.github/workflows/evalproof.yml`; `gitlab` writes `.gitlab-ci.evalproof.yml`, which you include from `.gitlab-ci.yml` (never edited); `bitbucket` (experimental) writes `bitbucket-pipelines.yml`, or `evals/bitbucket-pipelines.snippet.yml` when a `bitbucket-pipelines.yml` that `init` did not write exists (not even `--force` replaces it). `gitlab` and `bitbucket` with `--ci api-key` refuse a coding-agent CLI judge ([In CI](../guides/ci.md#gitlab-ci)) |
| `--cli-auth` | one of `api-key`, `oauth-token` | `api-key` | with `--ci api-key` and a coding-agent CLI judge, what the workflow passes the CLI: its API key (`api-key`), or `oauth-token`: `CLAUDE_CODE_OAUTH_TOKEN` from `claude setup-token`, for `claude-code` in your own repository and CI only |
| `--from` | string | none | a file of test cases from another tool, imported into `evals/cases.jsonl` and a suite with the scorers they need, instead of the sample cases. It cannot be combined with `--agent-files-only` (exit 2) |
| `--from-format` | one of `auto`, `promptfoo`, `deepeval`, `ragas`, `langsmith`, `openai-evals`, `inspect`, `braintrust`, `langfuse`, `phoenix`, `adk`, `jsonl`, `csv` | `auto` | the tool the `--from` file is from; `auto` reads it off the file. `jsonl` and `csv` need an `input` column here, as `init` has no field options: use `import-dataset` for other columns |
| `--no-skill` | flag | off | do not write the agent skill |
| `--no-agents-md` | flag | off | do not add the evalproof section to `AGENTS.md` |
| `--agent-files-only` | flag | off | experimental. Write only the agent files, the evalproof section of `AGENTS.md` and the skill: no suite and no CI workflow. It cannot be combined with `--judge`, `--ci`, `--platform` or `--cli-auth`, or with both `--no-agents-md` and `--no-skill` (exit 2). An existing skill copy is kept unless you pass `--force`, and a note says when a kept copy differs from this version |
| `--force` | flag | off | overwrite files that already exist; `AGENTS.md` is the exception, see above |

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
| `--limit` | int | none | run only the first N cases of the dataset (after `--tags` and `--exclude-tags` have chosen them) |
| `--tags` | string, repeatable | none | run only the cases that carry at least one of these tags; comma-separated, and the option may repeat (`--tags smoke,edge` is `--tags smoke --tags edge`); tags match exactly, with case. Chosen before `--limit`. The filter is recorded in the run file as `info.tag_filter`, and `verify` applies it again; `compare` warns when two runs used different filters. A filter that selects no case is an error (exit 2) that lists the dataset's tags. No suite key sets it |
| `--exclude-tags` | string, repeatable | none | run no case that carries any of these tags, with the same syntax; wins over `--tags` for a case that carries both |
| `--no-cache` | flag | off | do not read or write the response cache |
| `--cache-only` | flag | off | re-score or rebuild a run from the cache: every model call, the judge's included, is answered from the response cache, and a miss is an error (exit 2), never a model call, which stops the run (with several variants, the runs finished before the miss are written); the run records `cache_only: "true"` in its `environment`; not with `--no-cache`, and a `python`, `http` or `command` system is refused because it is not cached |
| `--trials` | int, 1 to 100 | none | run each case K times (1 to 100), overriding the suite's trials |
| `--system` | string | none | override the system target |
| `--system-prompt-file` | string | none | override the system prompt |
| `--label` | string | none | name this system in reports |
| `--variant` | string, repeatable | none | run one of the suite's variants and write `runs/<variant>.json`; repeat to run several, one run file each |
| `--all-variants` | flag | off | run every variant the suite defines, one run file each |
| `--markdown` | string | none | also append the report to this file |
| `--junit` | string | none | write a [JUnit XML report](junit.md) to this file: one testsuite per run written, one testcase per case (per trial when the run repeats its cases). With several variants the one file holds every run |
| `--quiet` | flag | off | print no progress line and no report on stdout (`--markdown` still writes it) |
| `--show` | int | `10` | list up to N failed cases in the report (default 10; 0 hides the list) |
| `--gate` | flag | off | apply the suite's absolute checks (`gate.min_pass_rate`, `gate.must_pass`) to this run, with no baseline; exit 1 when one fails |

```bash title="Example"
evalproof run evals/suite.yaml --out runs/candidate.json --trials 3
```

To run a smoke subset while you iterate: `evalproof run evals/suite.yaml --tags smoke --exclude-tags slow --out runs/smoke.json`.

Exit 0. With `--gate`: 1 when an absolute check fails, 4 when the gate cannot apply one. Exit 2 for a configuration error, when every case errored, and when `--tags` and `--exclude-tags` select no case. See [Exit codes](exit-codes.md). See also: [Concepts](concepts.md#run), [Repeated trials](../guides/repeated-trials.md#repeated-trials), [Testing agents](../guides/testing-agents.md#variants-one-suite-several-builds).

## evalproof import

Turns another tool's results into a run file that `compare` and the gate can read. See [Import results from other tools](../guides/import.md#import-results-from-other-tools).

```bash title="Terminal"
evalproof import FORMAT FILE --out PATH [options]
```

| Argument | Required | What it is |
|---|---|---|
| `FORMAT` | yes | the tool or table layout the file is from: `promptfoo`, `inspect`, `deepeval`, `lm-eval`, `garak`, `adk`, `claude-plugin-eval`, `jsonl`, `csv` |
| `FILE` | yes | the results file to import |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--out` | string, required | none | where to write the run artifact |
| `--system` | string | none | pick one system out of a file that holds several; otherwise the run's label |
| `--epochs` | one of `mean`, `first`, `all`, `any`, `trials` | `mean` | how repeats of one case collapse: mean (default), first, all (every repeat must pass) or any; or trials, which keeps each repeat as its own trial of the case |
| `--markdown` | string | none | also append the report to this file |
| `--quiet` | flag | off | do not print the report to stdout (`--markdown` still writes it) |
| `--threshold` | number | none | garak only (any other format refuses it): the detector score at which an output counts as a hit; without it the scan's `eval_threshold`, else 0.5. The one used and its source (`cli`, `eval_threshold` or `default`) are recorded in the run's `environment.threshold` and `environment.threshold_source` |
| `--id-field` | string | none | column holding the case id (default: id) |
| `--metric-field` | string | none | column holding the metric name (default: metric) |
| `--score-field` | string | none | column holding the continuous score (default: score) |
| `--passed-field` | string | none | column holding pass/fail (default: passed) |
| `--system-field` | string | none | column holding the system (default: system) |
| `--wide` | flag | off | one row per case, one column per metric |

```bash title="Example"
evalproof import promptfoo results.json --out runs/promptfoo.json
```

Exit 0, or 2 for a usage error such as an unreadable file. `adk` and `claude-plugin-eval` are experimental: they print a note once and set `environment.import_experimental` in the run.

## evalproof import-dataset

Turns another tool's test cases, which hold no results, into a JSONL dataset that `evalproof run` reads, and optionally into a suite with the scorers those cases need. It makes no model call. `auto` reads the format off the file, prints which one it chose and refuses a file that fits several or none. `adk` reads a Google ADK eval set (`*.evalset.json`) or the flat `*.test.json` list; any other shape is refused. See [Import test cases from other tools](../guides/import-test-cases.md) for every format and what each assertion becomes, and [Google ADK eval sets as a dataset](../guides/import.md#google-adk-eval-sets-as-a-dataset).

```bash title="Terminal"
evalproof import-dataset FORMAT FILE --out PATH [--suite-out PATH] [--force] [--id-field COLUMN] [--input-field COLUMN] [--reference-field COLUMN] [--tags-field COLUMN]
```

| Argument | Required | What it is |
|---|---|---|
| `FORMAT` | yes | the tool the file is from: one of `auto`, `promptfoo`, `deepeval`, `ragas`, `langsmith`, `openai-evals`, `inspect`, `braintrust`, `langfuse`, `phoenix`, `adk`, `jsonl`, `csv`. All but `promptfoo`, `jsonl`, `csv` and `adk` are experimental and print a note once |
| `FILE` | yes | the test set to import |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--out` | string, required | none | where to write the dataset; an existing file is replaced |
| `--suite-out` | string | none | also write a suite with exactly the scorers the cases need, and a `judge:` block only when one needs it. It refuses (exit 2) when nothing in the cases can be checked |
| `--force` | flag | off | overwrite an existing `--suite-out` file; without it the command stops (exit 2) before writing anything |
| `--id-field` | string | none | `jsonl` and `csv` only: the column that holds the case id |
| `--input-field` | string, repeatable | none | `jsonl` and `csv` only: the column that holds the input. Repeat it or give a comma list for several, which make the input an object; with none, the `input` column is used |
| `--reference-field` | string | none | `jsonl` and `csv` only: the column that holds the expected answer, which becomes the `reference` |
| `--tags-field` | string | none | `jsonl` and `csv` only: the column that holds tags, a list or a comma-separated string |

```bash title="Example"
evalproof import-dataset auto promptfooconfig.yaml --out evals/cases.jsonl --suite-out evals/suite.yaml
```

The cases that cannot be replayed (ADK user-simulation cases) are skipped with one warning on stderr. Assertions that no scorer can express stay on the case under `unmapped` and are counted in one summary line. Exit 0, or 2 for a usage error, an unreadable file, an unsupported or ambiguous shape, or a file with no case to import.

## evalproof compare

Compares each candidate with the baseline on the cases both runs share. With `--gate` the result becomes an exit code. Several candidates give one table, and `--gate` then corrects the p-values for all candidates and gated metrics at once and exits with the worst code (2, then 4, 1, 3, 0). Gate settings come from `--suite`; every flag below overrides the suite's `gate` block. Without `--gate` a comparison of two runs that cannot be compared is a usage error (exit 2).

Runs made with different tag filters (`run --tags`, `--exclude-tags`) are compared on the cases both hold, and the report warns that the filters differ; with `--gate` they cannot be gated (exit 4), because a filter does not show in the diff and the gate would decide on a subset of the cases.

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
| `--junit` | string | none | write a [JUnit XML report](junit.md) of the gate's checks to this file, whatever the verdict; one testcase per check. It needs `--gate`: without it the command exits 2 |

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

A run made with `--tags` or `--exclude-tags` records that filter, and `verify` checks it against the cases the same filter selects from the dataset now: a case that matches but is missing from the run, a case in the run that the filter does not select, and a case that gained or lost a tag all fail it.

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

Checks each coding-agent CLI target (`claude-code`, `codex`, and the experimental `opencode`, `cursor` and `copilot`): binary, version, login and the isolation flags. It makes no model call, except with `--cursor-canary`. With a suite it checks the targets that suite uses; without one it surveys every CLI on the machine. A suite's `command:` system is checked too: doctor prints the file its program resolves to and the directory it runs in, says that no credential is used, and reports a program that cannot be found.

```bash title="Terminal"
evalproof doctor [SUITE]
evalproof doctor --cursor-canary [--model MODEL]
```

| Argument | Required | What it is |
|---|---|---|
| `SUITE` | no | optional: check the targets this suite uses (default: every CLI found) |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--cursor-canary` | flag | off | instead of the free check, make **one real `cursor-agent` call** (it uses your Cursor quota) to see which skills and user rules reach the prompt; takes no `SUITE` |
| `--model` | string | `composer-2.5-fast` | the model for `--cursor-canary`; only with `--cursor-canary` |

```bash title="Example"
evalproof doctor evals/suite.yaml
evalproof doctor --cursor-canary
```

Exit 0 when every checked target is ready (or, without a suite, when at least one CLI is usable), 1 otherwise.

`--cursor-canary` runs the call through the cursor provider's normal path (the same flags, the macOS sandbox if it is on, and the tool guard) with a canary skill named `evalproof-canary` in the call's workspace. It asks the model which skills and Cursor User Rules it can see, and prints whether the canary was seen, which other skills were seen, and how many user rules were named (a count only; rule titles and contents are never printed). What it prints is the model's own account of its context, so it is best effort. Exit 0 when the canary was seen and no other skill was, 1 when the canary was not seen or another skill was, 4 when the call failed and there is no verdict, 2 for a usage error (`--model` without `--cursor-canary`, a `SUITE` with it, or an invalid `EVALPROOF_CURSOR_SANDBOX`). See also: [Providers](../guides/providers.md#experimental-providers), [Providers: login](../guides/providers.md#use-your-claude-code-or-codex-login).

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

## evalproof hook

Installs, or removes, a git `pre-push` hook that runs [`evalproof verify`](#evalproof-verify) on a committed run file before every push. A run file that no longer matches the suite or its dataset stops the push; no model is called. The hook is a POSIX `sh` script, which Git for Windows runs with its own `sh`.

`install` finds the hooks folder with `git rev-parse --git-path hooks`, so it follows `core.hooksPath` and, in a linked worktree, writes to the folder the worktrees share. The script starts with the line `# evalproof pre-push hook` and the evalproof version. Running `install` again updates a hook that has that line. A `pre-push` hook without it is never replaced unless you pass `--force`, and `uninstall` removes only a hook that has it.

The hook runs `evalproof` from the `PATH`, and otherwise `uvx evalproof@VERSION` for the version that wrote it. It skips with a message, and does not stop the push, when the suite or the run file does not exist, or when neither `evalproof` nor `uvx` is available. Otherwise it passes `verify`'s exit code through and adds one line on how to refresh the run or push with `git push --no-verify`. Git runs a hook from the repository root, so `--suite` and `--run` are stored relative to it.

```bash title="Terminal"
evalproof hook {install,uninstall} [--suite SUITE] [--run RUN] [--force]
```

| Argument | Required | What it is |
|---|---|---|
| `ACTION` | yes | `install` writes the hook; `uninstall` removes it |

| Flag | Type | Default | What it does |
|---|---|---|---|
| `--suite` | string | `evals/suite.yaml` | the suite the hook verifies against; `install` only |
| `--run` | string | `evals/runs/latest.json` | the committed run file the hook verifies; `install` only |
| `--force` | flag | off | install over a `pre-push` hook that evalproof did not write; `install` only |

```bash title="Example"
evalproof hook install
```

Exit 0, or 2 for a usage error: outside a git repository, a `pre-push` hook that evalproof did not write (`install` without `--force`, and `uninstall`), or git missing. `uninstall` with no hook in place says so and exits 0. See also: [Check before every push](../guides/ci.md#check-before-every-push).

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
| `--limit` | int | none | compare only the first N shared cases, in case-id order (after `--tags` and `--exclude-tags` have chosen them) |
| `--tags` | string, repeatable | none | judge only the cases that carry at least one of these tags, read from the suite's dataset; comma-separated, and the option may repeat |
| `--exclude-tags` | string, repeatable | none | judge no case that carries any of these tags; wins over `--tags`. A filter that selects no case in the dataset is an error (exit 2) |
| `--no-swap` | flag | off | judge each pair in one order only (by default each pair is judged in both orders) |
| `--no-cache` | flag | off | do not read or write the response cache |
| `--cache-only` | flag | off | answer every judge call from the response cache; a miss is an error (exit 2), never a model call; not with `--no-cache` |
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
