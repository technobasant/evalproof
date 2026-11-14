# What a suite covers: `evalproof readiness`

`evalproof readiness` reports which parts of quality a suite covers and which it leaves untested.

```bash
evalproof readiness suite.yaml [--run FILE] [--json] [--require behaviour,regression-gate]
```

It reads the suite and the dataset (and a run file, if you pass one) and reports, with no model
call, which of eleven dimensions the suite covers: correctness, behaviour (tool use), quality,
regression gate, reliability, cost and latency, safety, robustness, multi-turn, load and
production monitoring. Each is `covered`, `partial`, `missing` or `out of scope`, with the evidence.

* **correctness** is covered by a deterministic scorer that grades answers against a key
  (`contains`, `regex`, and so on); **behaviour** by a `checks` scorer, when at least half the
  cases have a check on the tool calls or rounds; **quality** by an `llm_judge` scorer, calibrated.
* **regression gate** needs a `gate` block, and `min_cases` no larger than the dataset;
  **cost and latency** need `max_cost_usd` or `max_p95_latency_ms` in it.
* **safety** and **robustness** are covered when a case is tagged `safety` or `robustness`;
  **multi-turn** is at most partial, since evalproof cannot say how many multi-turn cases are
  enough.
* **reliability** is covered with 3 or more `trials` per case, partial with 2 (two trials show
  disagreement but not a rate) and missing with 1; with `--run` it reads the run's own `trials`.
  **load** and **production monitoring** are out of scope: use a load-testing tool and an
  observability platform.

The report says what the suite measures, not that the measurement is good enough. `--require`
turns it into a gate: the exit code is 1 unless every named dimension is covered, and 2 for a
usage error. The names are `correctness`, `behaviour`, `quality`, `regression-gate`,
`reliability`, `cost-latency`, `safety`, `robustness`, `multi-turn`, `load` and
`production-monitoring`.
