# Evals

Run the suite and write the run file (this is the command that writes `evals/runs/latest.json`):

    uvx evalproof run evals/suite.yaml --out evals/runs/latest.json

@@CI_TEXT@@

The three cases in `cases.jsonl` are samples. Replace them with real ones; the gate needs 20.
Changing the cases changes the dataset hash, so a run made on the old cases cannot be compared
with one made on the new cases (exit 4): run again and commit the new `latest.json`.
