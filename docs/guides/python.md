---
description: Call load_run, compare_runs, decide and calibrate from Python, the same functions the CLI uses.
---

# From Python

This page is for anyone calling evalproof from Python instead of the command line. The CLI is a thin layer over the same functions:

```python title="example.py"
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

## Next steps

* Every object, from the docstrings: [API reference](../reference/api.md).
