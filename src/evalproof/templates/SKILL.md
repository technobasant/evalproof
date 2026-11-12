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
one, so the pull request carries it:

```bash
evalproof run evals/suite.yaml --out evals/runs/latest.json
```

The model calls use the developer's own login or API key and cost quota or money. Run the
suite once per change, not in a loop. Responses are cached, so an unchanged case is free
the second time.

## Read the result

- `run` prints a pass rate per scorer with a 95% interval, the cases that errored or were
  undecided, and latency. Errored cases count as failures.
- `compare` prints the change per scorer with its interval and p-value, then the gate
  verdict. Read the warnings: "dropped but not distinguishable from noise" means there are
  too few cases to tell, not that the change is safe.
- A case that fails is worth reading: open the run file, find its `output` and `scores`.

## Add a case

Append one JSON object per line to `evals/cases.jsonl`:

```json
{"id": "refund-window", "input": {"question": "..."}, "reference": "the correct answer", "accept": ["a string a correct answer contains"], "tags": ["refunds"]}
```

- `id` is unique and stays the same for the life of the case.
- `reference` is what the judge compares against. `accept` feeds the `contains` scorer.
- Add cases for real failures: a bug report, a bad answer from production, a regression.
- Changing the cases changes the dataset hash. Runs on the old and the new set cannot be
  compared (exit 4), so after adding cases run the suite again and commit the new
  `evals/runs/latest.json`.

## Exit codes

| Code | Meaning | What to do |
|---|---|---|
| 0 | Pass, possibly with warnings | Read the warnings |
| 1 | Regression: a large, significant drop | Report it to the human |
| 2 | Usage error: bad suite, path or target | Fix the command or the suite |
| 3 | Budget breach: p95 latency or cost over the limit | Report it to the human |
| 4 | Cannot gate: too few cases, different data, a metric with no scores, or the judge changed | Report it to the human |

## Rules

- Never edit `evals/runs/latest.json` by hand. Regenerate it with `evalproof run`.
- Never weaken a gate to make it pass: do not raise `max_drop` or `alpha`, lower
  `min_cases`, change `gate.metric`, or loosen the rubric.
- Never delete or edit cases, references or `accept` lists to make a run pass.
- On exit 1, 3 or 4, stop and report to the human: the exit code, the `compare` output and
  what you changed. Do not work around it by switching the judge, running a subset
  (`--limit`) or re-running until it passes.
- Do not commit a run made with `--limit` or `--system` as `latest.json`.
- A different judge model or CLI version makes runs incomparable. Changing it is the
  human's decision.
