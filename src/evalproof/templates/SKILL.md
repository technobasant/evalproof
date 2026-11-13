---
name: evalproof
description: Use when changing a prompt, a model or a retrieval step, or when checking an LLM change for regressions, to run this repo's evalproof suite in evals/ and read its gate result.
---

# evalproof

This repo measures LLM changes with evalproof. `evals/cases.jsonl` is the golden set,
`evals/suite.yaml` names the system under test, the scorers, the judge and the gate, and
`evals/runs/latest.json` is the committed result of the last accepted run. The gate compares
two runs case by case and fails only on a drop that is both large enough to matter
(`max_drop`) and unlikely to be chance (`alpha`). A difference of two pass rates is not
enough on its own.

Use `evalproof` if it is installed, otherwise `uvx evalproof`.

## Check a change

Run these from the repository root, after making the change and before committing it:

```bash
mkdir -p evals/.evalproof
git show HEAD:evals/runs/latest.json > evals/.evalproof/baseline.json
evalproof run evals/suite.yaml --out evals/.evalproof/candidate.json
evalproof compare evals/.evalproof/baseline.json evals/.evalproof/candidate.json \
  --suite evals/suite.yaml --gate
echo "exit $?"
```

If the gate passes and the change is meant to stay, replace the committed run with the new
one, so the pull request carries it, and check it the way CI will (no model calls):

```bash
evalproof run evals/suite.yaml --out evals/runs/latest.json
evalproof verify evals/runs/latest.json evals/suite.yaml
```

`verify` exits 4 when the run file does not match the suite (system, judge, scorers, seed) or
does not cover exactly the cases in `cases.jsonl`: a stale file, a `--limit` run, an edited case.
With no baseline to compare with (a first run), set `gate.min_pass_rate` or `gate.must_pass` in
the suite and add `--gate` to `run`: it checks the run on its own and exits 1 when a floor is missed.

The model calls use the developer's own login or API key and cost quota or money. Run the
suite once per change, not in a loop. Responses are cached, so an unchanged case is free
the second time.

## Read the result

- `run` prints a pass rate per scorer with a 95% interval and latency, then a "Failed cases"
  list (id, scorer, reason; `--show N` changes how many). Errored cases count as failures, and
  a run where every case errored exits 2.
- `compare` prints the change per scorer with its interval and p-value, the ids that regressed
  on each metric, and for a few of them (`--show N`) both answers and the scorer's reason. Then
  the gate verdict: "cannot gate" (exit 4) means it did not decide, which is not a regression.
  Read the warnings: "dropped but not distinguishable from noise" means there are too few
  cases to tell, not that the change is safe.
- The gate also fails on `gate.max_new_failures` (cases that passed before and fail now, even
  when the statistics would pass), `gate.min_pass_rate` (a floor, per scorer or for all) and
  `gate.must_pass` (tags whose cases must all pass). Those are the owner's; do not change them.

## Add a case

Append one JSON object per line to `evals/cases.jsonl`:

```json
{"id": "refund-window", "input": {"question": "..."}, "reference": "the correct answer", "accept": ["a string a correct answer contains"], "tags": ["refunds"]}
```

- `id` is unique and stays the same for the life of the case.
- `reference` is what the judge compares against. `accept` feeds the `contains` scorer.
- Add cases for real failures: a bug report, a bad answer from production, a regression.
- Adding or editing a case does not block the gate. Cases are paired by id and content: a new
  case has no baseline score, and an edited case (same id, new content) is left out of the
  comparison and listed in the report until the baseline is refreshed, which happens when the
  change is merged. After changing cases, run the suite again and commit the new
  `evals/runs/latest.json`, so `verify` finds it covers exactly the cases that are there.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | Pass, possibly with warnings | Read the warnings |
| 1 | Regression: a large, significant drop | Report it to the human |
| 2 | Usage error: bad suite, path or target | Fix the command or the suite |
| 3 | Budget breach: p95 latency or cost over the limit | Report it to the human |
| 4 | Cannot gate: too few paired cases, a metric with no scores, the judge changed, or `verify` found the run file does not match the suite and cases | Report it to the human; for a stale run file, run the suite again |

## Rules

- Never edit `evals/runs/latest.json` by hand. Regenerate it with `evalproof run`.
- Never weaken a gate to make it pass: do not raise `max_drop` or `alpha`, lower
  `min_cases`, change `gate.metric`, or loosen the rubric.
- Never delete or edit cases, references or `accept` lists to make a run pass.
- On exit 1, 3 or 4, stop and report to the human: the exit code, the `compare` output and
  what you changed. Do not work around it by switching the judge, running a subset
  (`--limit`) or re-running until it passes.
- Do not commit a run made with `--limit` or `--system` as `latest.json`.
- A different judge model makes runs incomparable (a new judge CLI version alone only warns).
  Changing the judge is the human's decision.
