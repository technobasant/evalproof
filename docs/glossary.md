---
description: Plain definitions of the terms evalproof uses, from case, run and suite to ACC, PPI, McNemar, pass@k and the readiness report.
---

# Glossary

The terms evalproof uses, in alphabetical order, each with a short definition and a link to where it is used. It is for anyone reading the reports or the documentation who meets a word they do not know.

| Term | Meaning |
|---|---|
| ACC | Adjusted classify-and-count. A correction to the judge's pass rate that uses the judge's true and false positive rates measured on gold labels: `(q − FPR) / (TPR − FPR)`, where `q` is the rate the judge reports. Its interval is a bootstrap over both the judged set and the gold sample. See [Concepts](reference/concepts.md#calibrate). |
| Calibration | Measuring a judge against gold labels and correcting its pass rate for its own errors. `evalproof calibrate` reports the raw judge rate, the gold labels alone, ACC and PPI. See [Concepts](reference/concepts.md#calibrate). |
| Case | One entry of the dataset (a line of the JSONL file) with an `id`, an `input` and optional tags, `turns`, `state` and `expect` checks. Runs are paired by case id and hash. |
| Flaky case | A case with both a pass and a fail among its decided trials. See [Repeated trials](guides/repeated-trials.md). |
| Gate | The check that turns a comparison or a single run into an exit code: a drop that is both large and significant, an absolute check, or a budget. See [Concepts](reference/concepts.md#compare-and-gate) and [Exit codes](reference/exit-codes.md). |
| Gold label | A trusted verdict for a case, from a person (`--labels`) or from a trusted scorer in the same run (`--gold-scorer`). Calibration compares the judge against them. |
| Judge | The model that scores a case in an `llm_judge` scorer, set by the suite's `judge` block. Its verdict is a pass or a fail, or undecided. See [Configuration](reference/configuration.md#system-and-judge). |
| MDE | Minimum detectable effect: the smallest drop in a gated metric that the suite can detect at 80% power. The compare report prints it beside the gate decision, with the number of cases needed to detect `max_drop`. It is information only and never changes the exit code. |
| Majority flip | A case the baseline passed in more than half of its trials and the candidate did not. `max_new_failures` counts majority flips when the runs repeat each case. See [Repeated trials](guides/repeated-trials.md). |
| McNemar | The exact paired test evalproof uses on pass/fail results: it looks only at the cases where the baseline and the candidate disagree. It is used for single-trial runs and for the `all` and `any` aggregates. |
| pass@k | The share of cases that passed at least once in their k trials. See [Repeated trials](guides/repeated-trials.md). |
| pass^k | The share of cases that passed in every one of their k trials. Written `pass^K` in the run report. |
| PPI | Prediction-powered inference. The judge's mean on the unlabelled cases plus the mean of (label − judge) on the gold sample, with a normal interval. See [Concepts](reference/concepts.md#calibrate). |
| Readiness | The report from `evalproof readiness` of which dimensions of quality a suite covers (correctness, behaviour, reliability, safety and the rest), with the evidence. It makes no model call. See [Readiness](guides/readiness.md). |
| Run | The JSON file `evalproof run` writes: every case's output, latency, tokens, cost and scores, with the hashes and facts needed to compare and verify it. See [Run file format](reference/run-file.md). |
| Sign-flip test | The paired permutation test evalproof uses on the `mean` aggregate and on continuous scores: it flips the sign of each case's paired difference and counts how often the result is as extreme as the observed one. It is computed exactly where that is practical and otherwise from 20 000 seeded draws. |
| Suite | The YAML file that names the dataset, the system under test, the scorers, the judge and the gate. See [Configuration](reference/configuration.md). |
| Trial | One execution of a case. With `trials: K`, each case runs K times and each trial is scored on its own. |
| Variant | A named change to the suite (another model, prompt or flag) that runs the same cases, scorers and gate. `evalproof run --variant NAME` writes one run file per variant. See [Testing agents](guides/testing-agents.md). |

## Next steps

* How the pieces fit together: [Concepts](reference/concepts.md).
* Every command and flag: [Commands](reference/commands.md).
