# Import results from other tools

You do not have to run your evals with evalproof to gate them. `evalproof import` turns the
per-case results another tool wrote into a run file that `compare` and the gate read:

```bash
evalproof import promptfoo output.json --system openai:gpt-5 --out runs/candidate.json
evalproof import inspect logs/task.eval --out runs/candidate.json
evalproof import deepeval .deepeval/.latest_test_run.json --out runs/candidate.json
evalproof import lm-eval results/gpt2/samples_hellaswag_2026-11-12T10-20-30.jsonl --out runs/gpt2.json
evalproof import csv results.csv --system a --out runs/a.json       # id,metric,passed,score,...
evalproof import jsonl results.jsonl --wide --id-field case --out runs/a.json  # one column per metric
```

Import each side with the same importer, then gate as usual, for example
`evalproof compare runs/baseline.json runs/candidate.json --gate --metric pass`. What to know:

* Case ids are stable between runs of the same dataset (they come from the test's inputs, or
  the tool's own sample id, never from its position), which is what lets two files pair.
  Each case also carries a hash of its id and input as the tool recorded them, so runs imported
  by different importers pair no cases and the gate refuses (exit 4) instead of pairing on a
  coincidence.
* A system error counts as a failure on every metric, as in a native run. Repeats of one case
  (epochs, repeated tests) collapse by `--epochs`: `mean` (the default), `first`, `all`
  or `any`, or are kept as the case's [trials](repeated-trials.md#repeated-trials) with `--epochs trials`. Under `mean`, `first` and `any`, a case with at least one repeat that ran is not
  counted as a system error; its errored repeats still score as failures.
* A NaN or infinite score is refused at import, naming the case and metric, because it would
  make the bootstrap report a significant difference that is not there.
* promptfoo files can hold several providers and `--system` picks one. Inspect `.eval` logs
  compressed with zstd need Python 3.14 or newer; otherwise run
  `inspect log convert --to json <file>` first. DeepEval's conversational cases are skipped
  with a warning.
