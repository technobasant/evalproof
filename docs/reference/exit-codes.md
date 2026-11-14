---
description: The exit codes of evalproof, 0 pass, 1 regression, 2 usage error, 3 budget breach, 4 cannot gate, 130 interrupted, and what produces each.
---

# Exit codes

evalproof ends with a distinct code for each way a run can finish, so a pipeline can tell a quality regression from a budget breach from a setup problem. This page is for anyone writing a CI step that reacts to them.

| Code | Name | Meaning | Returned by |
|---|---|---|---|
| 0 | pass | nothing failed; there may be warnings | every command |
| 1 | regression | a quality check failed | `compare --gate`, `run --gate`, `readiness --require`, `doctor` |
| 2 | usage error | the command, a file or the configuration is wrong | every command |
| 3 | budget breach | the candidate is over a latency or cost limit | `compare --gate` |
| 4 | cannot gate | the runs cannot be compared, so no verdict is possible | `compare --gate`, `run --gate`, `verify`, `rehash` |
| 130 | interrupted | the process was stopped with Ctrl-C | every command |

## Exit 1: a quality check failed

* The gated metric dropped by at least `max_drop` and the p-value is below `alpha`. With several gated metrics the p-value is Holm-adjusted.
* More cases than `max_new_failures` went from pass to fail.
* An absolute check failed: a pass rate below `min_pass_rate`, or a `must_pass` case that did not pass (in every trial, when the runs repeat each case).
* `evalproof readiness --require` named a dimension the suite does not cover.
* `evalproof doctor` found a coding-agent CLI target that is not ready.

## Exit 2: a usage or configuration error

evalproof prints `evalproof: error: <message>` to stderr and exits 2. The causes include:

* a suite or dataset file that is missing, empty or invalid, or an unknown key in the suite (the message names the key);
* a provider error, such as a missing API key;
* an unknown `--metric`, `--variant` or `--require` dimension;
* `evalproof run` where every case errored, with the first error in the message;
* `evalproof compare` without `--gate` on two runs that cannot be compared.

Any other exception also exits 2, with its type and message and a hint to set `EVALPROOF_DEBUG=1` for the traceback.

## Exit 3: a budget breach

The candidate's p95 latency is over `max_p95_latency_ms` (or `--max-p95-ms`), or its total cost is over `max_cost_usd` (or `--max-cost`). When the quality check also fails, the code is 1: a regression takes precedence over a budget breach.

## Exit 4: cannot gate

The gate could not reach a verdict, and a pass would be wrong. The causes include:

* too few paired cases: fewer than `min_cases`, or no paired scores for the metric;
* the dataset changed and a run predates per-case hashes (`evalproof rehash` fixes that);
* `min_pass_rate` on a metric that has only continuous scores, or `aggregate` set to `all` or `any` on one;
* the judge's target, CLI, isolation or resolved model changed between the runs, unless `allow_judge_change` is set (a new CLI version alone only warns);
* the runs repeat each case a different number of times;
* `evalproof verify` found that the run does not match the suite or the dataset;
* `evalproof rehash` found that the dataset does not match the run.

## Several candidates

`evalproof compare` with several candidates exits with the worst code of the set, in this order: 2, then 4, then 1, then 3, then 0.

## Use the code in a script

```bash title="Terminal"
evalproof compare runs/baseline.json runs/candidate.json --suite evals/suite.yaml --gate
code=$?
case "$code" in
  0) echo "gate passed" ;;
  1) echo "quality regression" ;;
  3) echo "over the latency or cost budget" ;;
  4) echo "the runs cannot be compared" ;;
  *) echo "setup problem (exit $code)" ;;
esac
exit "$code"
```

In GitHub Actions a step fails on any non-zero code, so the plain command is usually enough; read the codes when a pipeline should treat a budget breach differently from a regression.

## Next steps

* The gate's settings: [Configuration](configuration.md#gate).
* Every command and flag: [Commands](commands.md).
