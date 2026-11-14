# Repeated trials

A model, or an agent built on one, can answer the same input differently on another run, so one
run per case measures one draw of what the system does. Run each case several times:

```yaml
trials: 5          # in the suite; `evalproof run --trials 5` overrides it (1 to 100)
```

The run file then holds `cases x trials` results. Each trial is run and scored on its own. A
python system gets `ctx.trial` (0 to `trials - 1`), an http system a `trial` field in the request
body (only when `trials` is above 1), and a local model that honours a seed gets `seed + trial`,
so each trial can draw a different sample. The response cache keeps trial 0 under the key it always
had and keys every later trial separately, so a re-run is still free and a trial never replays
another's answer. Nothing new goes to any provider. A run file from before trials loads as one
trial per case.

**What the report shows.** A run of several trials reports the pooled pass rate (passes over
decided trials, with an interval that resamples cases, since a case's trials move together),
`pass@k` (the share of cases that passed at least once), `pass^k` (the share that passed every
time), the mean (and, for a continuous score, the within-case spread), and the **flaky** cases: those with a pass and a fail
among their decided trials. An errored trial counts as a failure; a trial the scorer did not decide
is left out. When every case gave the same output on every trial, the report says so, because the
system may be deterministic at these settings and the trials then add nothing.

**How compare reads them.** Both runs need the same number of trials per case; `compare` refuses
otherwise (exit 4 with `--gate`, 2 without). Each case's trials become one value per run, chosen by
`--aggregate` or the gate's `aggregate`:

| Aggregate | The case's value | Test | Use it when |
|---|---|---|---|
| `mean` (default) | its pass rate over the decided trials | paired sign-flip permutation test on the per-case rates | you care how often it works |
| `all` | 1 if it passed every trial | exact McNemar on the cases | a single failure is a failure (a safety rule) |
| `any` | 1 if it passed at least once | exact McNemar on the cases | a retry is fine |

`all` and `any` need a pass/fail metric, and `compare` refuses them on a gated continuous one. A
continuous score compares by its per-case mean, with the same permutation test. One trial per case
compares exactly as before, with McNemar's test and the paired bootstrap.

The permutation test flips the sign of each case's difference at random and asks how often a total
as large as the observed one comes up. It is exact (the report says `sign-flip permutation
(exact)`) when every case has the same number of decided trials, however many cases there are, and
when there are at most 16 cases that differ at all. Otherwise it is a Monte Carlo estimate from
20 000 seeded draws, labelled so, with p = (hits + 1) / (draws + 1), which is never 0 and the same
on every run. The interval on the change is a paired bootstrap over cases.

**What the gate does.** `max_drop` and `alpha` apply to the aggregated values, as before.
`must_pass: [tag]` needs every trial of every case with that tag to pass, and the failure names the
case once, with "in at least one trial". `min_pass_rate` is the pooled rate over trials, `min_cases`
counts cases, and the latency and cost budgets cover every trial. `max_new_failures` counts
**majority flips**: a case counts when the baseline passed more than half of its trials and the
candidate did not. A case that merely became flakier is listed ("Flaky in candidate") but is not a
new failure unless its rate fell from above one half to one half or below, so one unlucky trial
cannot fail the build by itself.

**Power.** Next to the decision the report says what the suite could have detected: "With 35 cases
this gate reliably detects drops of about 8.7 points (80% power); about 663 cases are needed to
detect max_drop (2.0 points)." The estimate uses the spread of the per-case differences, divides
`alpha` across the candidates and gated metrics, and says so plainly when the baseline is uniform
(all pass or all fail) and the spread cannot be estimated. It is information only: it never changes
the status, a warning or the exit code. A suite too small to see a 2-point drop passes a 2-point
drop, and now the report says why.

**The example.** `examples/agent-tools/suite-flaky.yaml` has a `flaky` build that forgets the
calendar on a fixed 40% of the outing trials, decided by `random.Random(f"{case_id}:{trial}")` so
that it is the same on every machine:

```bash
for v in v2 flaky; do
  uv run evalproof run examples/agent-tools/suite-flaky.yaml --variant $v --trials 5 \
    --out /tmp/agent-k5-$v.json --quiet
done
uv run evalproof compare /tmp/agent-k5-v2.json /tmp/agent-k5-flaky.json \
  --suite examples/agent-tools/suite-flaky.yaml --gate; echo $?   # 1
```

behaviour falls from 100.0% to 90.9% (exact p = 0.0078), `o01` and `o05` regress by majority, and
all eight outing cases are flaky. One run per case misses most of it: the same build reads as three
failed cases and a McNemar p of 0.25. CI runs both directions, and `readiness --run` on the k=5 run
reports reliability as covered.

## Several candidates

```bash
evalproof compare baseline.json a.json b.json c.json --suite suite.yaml --gate
```

One table has a row per candidate and gated metric (`delta (CI) · adjusted p · verdict`), then the
status of each candidate and its details in a collapsed block. Every candidate whose comparison
was made, times every gated metric, is in one Holm family, so that adding a candidate cannot make a
false failure more likely. The exit code is the worst of the candidates in the order usage (2),
cannot gate (4), regression (1), budget (3), pass (0); without `--gate` it is 0. A candidate that
cannot be compared (a different number of trials, say) is a cannot-gate entry under `--gate` and
does not stop the others; without `--gate` it is a usage error. `--json FILE` writes
`{"baseline", "candidates": [{"label", "path", "comparison", "gate"}]}`, with `comparison` null
for a candidate that was refused. One candidate behaves, and prints, as it always did.
