# Evals

Run the suite and write the run file (this is the command that writes `evals/runs/latest.json`):

    uvx evalproof run evals/suite.yaml --out evals/runs/latest.json

@@CI_TEXT@@

The three cases in `cases.jsonl` are samples. Replace them with real ones; the gate needs 20.
Adding or editing cases does not block the gate: cases are paired by id and content, a new case
has no baseline score, and an edited case is left out of the comparison (and listed in the
report) until the baseline is refreshed by merging. A tag is not content, so adding or changing
a tag is not an edit. Run again and commit the new `latest.json`;
`uvx evalproof verify evals/runs/latest.json evals/suite.yaml` checks, without a model call,
that it matches the suite, covers exactly the cases in `cases.jsonl`, and holds the scores the
suite's deterministic scorers give its stored outputs.

To read a result: `run` lists the cases that failed with the scorer's reason, and `compare`
lists the ids that regressed on each metric and shows both answers for a few (`--show N`).
Exit 4 means the gate could not decide, which is not a regression.
