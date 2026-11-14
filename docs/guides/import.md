---
description: Turn promptfoo, Inspect, DeepEval, lm-eval, garak, Google ADK, CSV and JSONL results into run files that compare and the gate read.
---

# Import results from other tools

This page is for anyone who already has results from another eval tool. You do not have to run your evals with evalproof to gate them. `evalproof import` turns the
per-case results another tool wrote into a run file that `compare` and the gate read:

```bash title="Terminal"
evalproof import promptfoo output.json --system openai:gpt-5 --out runs/candidate.json
evalproof import inspect logs/task.eval --out runs/candidate.json
evalproof import deepeval .deepeval/.latest_test_run.json --out runs/candidate.json
evalproof import lm-eval results/gpt2/samples_hellaswag_2026-11-12T10-20-30.jsonl --out runs/gpt2.json
evalproof import garak garak.report.jsonl --out runs/scan.json
evalproof import adk .adk/eval_history/app_evalset_1788817991.2.evalset_result.json --out runs/adk.json
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
* garak reports (`garak.<run>.report.jsonl`, 0.9 and later) import one case per attempt, with the
  tags `safety`, `garak` and `probe:<name>`. Each generation of a prompt is a repeat, so
  `--epochs mean` is the share of generations that passed. One metric per detector: an output
  passes when its detector score is below the threshold, which is `--threshold`, else the
  scan's `eval_threshold`, else 0.5. The raw score is kept in the metric's reason, and an output
  a detector could not score stays undecided. Attempts garak never scored (no status-2 entry)
  are skipped with a warning.
* Google ADK eval results (`*.evalset_result.json`, from `.adk/eval_history`) import one case per
  eval case, with its runs (`repetition_index` order) as repeats. One metric per ADK metric,
  scored as ADK scored it (a 1 to 5 metric stays 1 to 5, and the import names it); passed and
  failed follow ADK's status, and a metric ADK did not evaluate is undecided. The tool calls come
  from the session's events, or from each invocation's intermediate data. This importer is
  experimental: it was written from ADK's source and no real result file was available.

## Next steps

* Gate the imported runs in CI: [In CI](ci.md#in-ci).
* How runs are paired: [Concepts](../reference/concepts.md#when-the-cases-change).
