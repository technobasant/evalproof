---
description: Run the evalproof gate in GitHub Actions with the composite action, or in GitLab CI with a generated include file, in keyless, api-key or CLI-auth mode.
---

# In CI

The composite action runs `compare --gate`, appends the report to the job summary, and
fails the job on exit codes 1, 3 or 4. This page is for anyone wiring the gate into GitHub Actions
or GitLab CI.

## The action

```yaml title=".github/workflows/evalproof.yml"
- uses: technobasant/evalproof@v1.2.0
  with:
    baseline: eval/baseline.json     # produced on main, e.g. as a build artifact
    candidate: eval/candidate.json   # produced on this branch
    suite: eval/suite.yaml
    # metric: "judge, answer_key"    # optional: gate several metrics, newline- or comma-separated
```

!!! note "What the action checks first"

    With a `suite`, the action first runs `evalproof verify` on the candidate (set
    `verify-candidate: "false"` to skip it): the run file must have been made with that suite's
    system, judge, scorers and seed and cover exactly the dataset's cases as they are now (a tag
    change counts, because `must_pass` reads tags from the run file). A stale file, a `--limit` run
    and an edited case fail the job with exit 4 before any comparison. It also scores the stored
    outputs again with the suite's scorers that need no model (`exact_match`, `contains`, `regex`,
    `json_schema`, `numeric_tolerance` and the retrieval scorers) and fails when a stored score
    differs, so a hand-edited deterministic score is caught. It cannot check model-judged scores or
    the stored outputs themselves; for pull requests from people you do not trust, review the run
    file or use the api-key mode.

## Action inputs

| Input | Required | Default | What it does |
|---|---|---|---|
| `baseline` | yes | none | path to the baseline run file, for example the one produced on the base branch |
| `candidate` | yes | none | path to the candidate run file, produced on this branch |
| `suite` | no | none | the suite file; its `gate` block sets the metric and thresholds |
| `verify-candidate` | no | `"true"` | with a `suite`, run `evalproof verify` on the candidate first; `"false"` skips it |
| `metric` | no | the suite's | scorer to gate on; several, separated by newlines or commas, are gated with a Holm correction |
| `max-drop` | no | the suite's | largest tolerated drop, as a fraction (`0.02` is two points) |
| `alpha` | no | the suite's | significance level |
| `pr-comment` | no | `"false"` | `"true"` posts the report as one comment on the pull request and updates it on later runs |
| `github-token` | no | `${{ github.token }}` | the token that posts the comment when `pr-comment` is `"true"` |

The action has one output, `exit-code`: 0 pass, 1 regression, 3 budget breach, 4 cannot gate,
2 usage error.

## Comment on the pull request

Set `pr-comment: "true"` and the action posts the report as a comment on the pull request. It
posts one comment, marked with `<!-- evalproof-report -->`, and updates that comment on later runs
instead of adding a new one. The job summary still gets the whole report.

```yaml title=".github/workflows/evalproof.yml"
permissions:
  contents: read
  pull-requests: write

jobs:
  gate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v7
      - uses: technobasant/evalproof@v1.2.0
        with:
          baseline: eval/baseline.json
          candidate: eval/candidate.json
          suite: eval/suite.yaml
          pr-comment: "true"
```

The comment needs `pull-requests: write` on the token. A comment is cut at 65,000 bytes, with a note
that says where to find the rest. The comment step runs after the gate, so a failed gate is still
posted.

!!! warning "Fork pull requests"

    A pull request from a fork gets a read-only token, so the comment cannot be posted. The action
    prints a `::warning::` and keeps the gate's result. The comment never changes whether the job
    passes.

## Choose a workflow

`evalproof init` writes one of two workflows:

=== "Keyless (default)"

    **`--ci keyless`** makes no model call in CI: developers run the suite on their own login and
    commit `evals/runs/latest.json`; the job takes the baseline from the base branch, verifies the
    committed run, and gates. It also warns when files you watch (`WATCH`, default `evals/`)
    changed but the run file did not.

=== "API key"

    **`--ci api-key`** makes the run in CI with a repository secret. It starts on every pull
    request (a path filter would leave a required check pending), and its first step compares
    against the base branch: when nothing in `WATCH` changed it says so and runs nothing. Pull
    requests from forks get no secrets, so for them it runs nothing and says so. Responses are
    cached in `evals/.evalproof/cache` between runs, keyed on the contents of `evals/`, so a push
    that changes no prompt or case is answered from the cache. The cache is saved right after the
    run, before the gate, so a pull request that fails the gate keeps it too.

=== "CLI auth"

    If the judge is a Claude Code or Codex CLI, `evalproof init --ci api-key` writes a workflow that
    installs the CLI at the version found at init time and runs `evals/suite.yaml` as it is. The run
    step sets `EVALPROOF_CLI_AUTH: api-key` and the repository secret (`ANTHROPIC_API_KEY` for
    Claude Code, `OPENAI_API_KEY` for Codex), so the CLI uses that key instead of a login. The
    suite names no key and there is no second suite file; a suite with an `api_key_env` still uses
    that variable. The pinned CLI version is advisory: a different version than the baseline's only
    warns, and the gate still refuses when the resolved model changes.

## GitLab CI

`evalproof init --platform gitlab` writes `.gitlab-ci.evalproof.yml`. It never edits your
`.gitlab-ci.yml`. It prints the two lines to add there:

```yaml title=".gitlab-ci.yml"
include:
  - local: .gitlab-ci.evalproof.yml
```

If you have no `.gitlab-ci.yml`, create one with those two lines. `--ci keyless` (the default) and
`--ci api-key` work as on GitHub:

```bash title="Terminal"
evalproof init --platform gitlab                    # keyless
evalproof init --platform gitlab --ci api-key --judge anthropic:claude-haiku-5-5
```

The file defines one job, `evalproof`, in the `test` stage. Change the stage if your pipeline has
no `test` stage. The job runs in merge request pipelines only, because the baseline is the file
`evals/runs/latest.json` on the target branch. It runs in `python:3.13-slim` and installs git when
the image has none.

If your other jobs also run on every branch push, add a `workflow: rules:` block so a push to a
branch with an open merge request does not start two pipelines.

* **Keyless.** The job needs no variable and makes no model call. It verifies the committed run
  file, then gates it against the target branch, and warns when files in `EVALPROOF_WATCH` changed
  but the run file did not.
* **API key.** The job runs the suite with a key kept in a CI/CD variable. Add a variable named
  after the judge's key (`ANTHROPIC_API_KEY` for `anthropic:`) under Settings, CI/CD, Variables.
  Mark it Masked. Do not mark it Protected: GitLab passes a protected variable only to pipelines
  on protected branches, and a merge request from a feature branch is not one. Responses are
  cached in `evals/.evalproof/cache`, and the cache is kept when the gate fails.
* **Fork merge requests.** A merge request from a fork gets no CI/CD variables. In api-key mode the
  job runs nothing and says so.
* **Coding-agent CLI judges.** `--platform gitlab --ci api-key` refuses a judge that is a Claude
  Code or Codex CLI, because the image has no Node to install the CLI with. Use a hosted API
  target, or `--ci keyless`.

The job writes a JUnit report, `evalproof-junit.xml`, and GitLab shows it in the merge request's
test summary. The artifact is kept for a week, and it is saved when the gate fails, which is the
report you want to read. What the report holds, and how to use it in Jenkins or Azure DevOps:
[JUnit reports](../reference/junit.md).

## This repository's own CI

This repository's own CI runs the offline example on every commit and asserts that the
gate returns 1 for the real regression and 0 for the noisy and refactored candidates.

## Next steps

* Which secret each target needs: [What it needs to run](../getting-started/credentials.md).
* How the gate decides: [Concepts](../reference/concepts.md#compare-and-gate).
* What each exit code means: [Exit codes](../reference/exit-codes.md).
* Failed cases and gate checks as test results: [JUnit reports](../reference/junit.md).
* A gate that exits 4, or a verify failure: [Troubleshooting](troubleshooting.md#comparing-and-gating).
