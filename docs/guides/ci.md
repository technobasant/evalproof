# In CI

The composite action runs `compare --gate`, appends the report to the job summary, and
fails the job on exit codes 1, 3 or 4:

```yaml
- uses: technobasant/evalproof@v0.6.0
  with:
    baseline: eval/baseline.json     # produced on main, e.g. as a build artifact
    candidate: eval/candidate.json   # produced on this branch
    suite: eval/suite.yaml
    # metric: "judge, answer_key"    # optional: gate several metrics, newline- or comma-separated
```

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

This repository's own CI runs the offline example on every commit and asserts that the
gate returns 1 for the real regression and 0 for the noisy and refactored candidates.

`evalproof init` writes one of two workflows. **`--ci keyless`** (the default) makes no model
call in CI: developers run the suite on their own login and commit `evals/runs/latest.json`;
the job takes the baseline from the base branch, verifies the committed run, and gates. It also
warns when files you watch (`WATCH`, default `evals/`) changed but the run file did not.
**`--ci api-key`** makes the run in CI with a repository secret. It starts on every pull
request (a path filter would leave a required check pending), and its first step compares
against the base branch: when nothing in `WATCH` changed it says so and runs nothing. Pull
requests from forks get no secrets, so for them it runs nothing and says so. Responses are
cached in `evals/.evalproof/cache` between runs, keyed on the contents of `evals/`, so a push
that changes no prompt or case is answered from the cache. The cache is saved right after the
run, before the gate, so a pull request that fails the gate keeps it too.

If the judge is a Claude Code or Codex CLI, `evalproof init --ci api-key` writes a workflow that
installs the CLI at the version found at init time and runs `evals/suite.yaml` as it is. The run
step sets `EVALPROOF_CLI_AUTH: api-key` and the repository secret (`ANTHROPIC_API_KEY` for
Claude Code, `OPENAI_API_KEY` for Codex), so the CLI uses that key instead of a login. The
suite names no key and there is no second suite file; a suite with an `api_key_env` still uses
that variable. The pinned CLI version is advisory: a different version than the baseline's only
warns, and the gate still refuses when the resolved model changes.
