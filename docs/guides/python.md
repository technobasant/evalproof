# From Python

The CLI is a thin layer over the same functions:

```python
from evalproof import GateConfig, calibrate, compare_runs, decide, load_run

baseline = load_run("runs/baseline.json")
candidate = load_run("runs/candidate.json")

cmp = compare_runs(baseline, candidate)  # paired cases: exact McNemar, bootstrap intervals
decision = decide(cmp, GateConfig(metric="judge", max_drop=0.03))
print(decision.status, decision.exit_code, decision.warnings)

report = calibrate(baseline, "judge", gold_scorer="answer_key", gold_sample=30)
for est in report.estimates:  # raw judge rate, gold only, ACC, PPI
    print(est.method, est.point, est.low, est.high)
```

The names in `evalproof.__all__` are the supported API. Other modules are internal and may
change between minor versions; see the [changelog](../about/changelog.md).
