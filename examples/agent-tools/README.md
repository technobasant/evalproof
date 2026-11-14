# Offline example: an agent with two tools

No model, no network. This is what CI runs on every commit to prove that the agent checks do what
the docs say. The system under test is a plain-Python "agent": a keyword planner that reads a
message, calls one or both of two fake tools (`get_weather`, `get_events`), and writes a reply from
what they returned. `TraceRecorder` records every call, which is what the `checks` scorer reads.

The golden set is 35 cases in `cases.jsonl`. Each one carries its own `expect` list: which tools
must be called, with which arguments, in which order, and what the reply must cite.

| Group | Cases | What it covers |
|---|---:|---|
| weather, calendar, briefing | 14 | the right tool with the right arguments, and a reply that cites the result; one unknown city (a failed tool call), one empty day, one home city from `state` |
| `outing` | 8 | "can I have a picnic on Saturday?" needs the forecast **and** the calendar, in that order |
| `safety` | 4 | "delete all my calendar events" and "erase my calendar" are refused with no tool call; a request for a manager's home address is declined |
| `scope` | 1 | a joke gets "I can only help with the weather and your calendar" and no tool call |
| `robustness` | 5 | typos and odd casing: `wether in paris tomorow`, `WHAT'S ON MY CALENDER TODAY??` |
| multi-turn | 3 | `turns`: a clarifying question answered on the next turn, a city and a day that carry over, a "delete all of it" refused on the second turn (this one is tagged `safety` too) |

## The planted bug

The two variants differ by one variable, which the agent reads from `ctx.vars`:

| Variant | `calendar_for_outings` | Behaviour |
|---|---|---|
| `v1` | `false` | a flag was left off: asked about a picnic or a hike, the agent answers from the forecast alone and **never calls `get_events`** |
| `v2` | `true` | the fix: it calls `get_weather`, then `get_events` |

The bug is in the tool calls, and the reply still reads well, which is why the suite carries two
scorers:

* `correct` (a `contains` scorer) checks the words. It passes **100%** for both variants.
* `behaviour` (a `checks` scorer) checks what the agent did. v1 scores **77.1%** (27 of 35) and v2
  scores **100%**. The eight outing cases `o01` to `o08` fail on v1: `get_events` was never called.

`behaviour` is the gate metric. The gate also sets `min_cases: 30`, `must_pass: [safety]` (a case
tagged `safety` must pass, whatever the statistics say) and `max_new_failures: 0` (the scorer is
deterministic, so one case flipping is a real change).

## Run it

```bash
for v in v1 v2; do
  uv run evalproof run examples/agent-tools/suite.yaml --variant $v \
    --out examples/agent-tools/runs/$v.json --quiet
done

# v1 -> v2: the fix passes the gate
uv run evalproof compare examples/agent-tools/runs/v1.json examples/agent-tools/runs/v2.json \
  --suite examples/agent-tools/suite.yaml --gate
echo $?   # 0

# v2 -> v1: the bug fails it
uv run evalproof compare examples/agent-tools/runs/v2.json examples/agent-tools/runs/v1.json \
  --suite examples/agent-tools/suite.yaml --gate
echo $?   # 1
```

| Compare | behaviour | Paired outcome | Gate |
|---|---|---|---|
| v1 baseline, v2 candidate | 77.1% to 100% | 8 better, 0 worse (exact McNemar p = 0.0078) | **pass, exit 0** |
| v2 baseline, v1 candidate | 100% to 77.1% | 0 better, 8 worse (exact McNemar p = 0.0078) | **fail, exit 1**, and `o01` to `o08` are named as newly failing |

The two run files in `runs/` are the committed output of exactly those `run` commands. Running both
variants at once works too: `evalproof run examples/agent-tools/suite.yaml --all-variants` writes
`runs/v1.json` and `runs/v2.json` in the current directory.

## A flaky build, and repeated trials

`suite.yaml` runs every case once, and its agent is deterministic, so one run says everything there
is to say. A real agent is not like that: the same message can take a different path on another
run. `suite-flaky.yaml` has the same dataset, scorers and gate, and a second pair of variants that
show what one run per case can miss:

| Variant | `calendar_for_outings` | `flaky_calendar` | Behaviour |
|---|---|---|---|
| `v2` | `true` | none | the fixed build: it calls `get_events` on an outing question, every trial |
| `flaky` | `true` | `0.4` | the same build, except that on an outing question it forgets the calendar in 40% of the trials |

Which trials it forgets is decided by `random.Random(f"{case_id}:{trial}")`, where `ctx.case_id`
and `ctx.trial` are what the runner hands to the system. The agent stays deterministic: the same
trials fail on every machine, which is what lets CI assert on the result. A real agent would let
the model's own variation do this.

```bash
for v in v2 flaky; do
  uv run evalproof run examples/agent-tools/suite-flaky.yaml --variant $v --trials 5 \
    --out /tmp/agent-k5-$v.json --quiet
done

uv run evalproof compare /tmp/agent-k5-v2.json /tmp/agent-k5-flaky.json \
  --suite examples/agent-tools/suite-flaky.yaml --gate
echo $?   # 1
```

The run file is 35 cases times 5 trials. The report:

* **behaviour** goes from 100.0% to 90.9% of trials passing, 9.1 points down (95% CI 15.4 to 3.4
  down), with a sign-flip permutation test on the 35 paired pass rates: exact p = 0.0078.
* Two cases, `o01` and `o05`, **regressed by majority**: they passed more than half of their trials
  on v2 and no longer do on flaky. That is more than the 0 the gate allows, so it fails with exit 1.
* **Flaky in candidate (8)**: all eight outing cases, `o01` to `o08`, pass some trials and fail
  others. Of the five trials each passes 2 (`o01`, `o05`), 3 (`o02`, `o04`, `o07`, `o08`) or 4
  (`o03`, `o06`).
* **correct** is unchanged (100% to 100%, p = 1.0): the reply still reads well, as it did for v1.
* A **power** line: with 35 cases this gate reliably detects drops of about 8.7 points (80% power),
  and about 663 cases would be needed to detect the 2.0 points `max_drop` allows. It is
  information only and never changes the decision.

Run once per case, the same build looks different. Trial 0 alone fails three of the eight outing
cases, the paired test (exact McNemar p = 0.25) cannot tell that from noise, and the gate exits 1
only because `max_new_failures: 0` sees the three newly failing cases. Five trials is what turns
"three cases failed this time" into "these eight cases are unreliable, and the drop is
significant".

| Compare | behaviour | Test | Gate |
|---|---|---|---|
| v2 baseline, flaky candidate | 100.0% to 90.9% | sign-flip permutation, exact, p = 0.0078 | **fail, exit 1**; `o01` and `o05` regress by majority |
| v2 baseline, v2 again | 100.0% to 100.0% | p = 1.0 | **pass, exit 0** |
| flaky baseline, v2 candidate | 90.9% to 100.0% | sign-flip permutation, exact | **pass, exit 0**: the fix is an improvement |

Several candidates against one baseline work too: `evalproof compare base.json a.json b.json
--suite ... --gate` prints one table and exits with the worst of the candidates' codes.

The `readiness` check reads the trials from a run file:

```bash
uv run evalproof readiness examples/agent-tools/suite-flaky.yaml --require reliability
echo $?   # 1: the config says one trial per case

uv run evalproof readiness examples/agent-tools/suite-flaky.yaml \
  --run /tmp/agent-k5-flaky.json --require reliability
echo $?   # 0: "5 trials per case"
```

The flaky run files are not committed: they take a second to regenerate, and CI makes them fresh.

## How ready is this suite

`evalproof readiness` reads the suite and the dataset and says which test dimensions they cover. It
calls no model.

```bash
uv run evalproof readiness examples/agent-tools/suite.yaml --require behaviour,regression-gate
echo $?   # 0
```

```
## Readiness: weather-calendar-agent

35 cases. A static reading of the suite and the dataset; no model was called.

| Dimension | Status | Evidence |
|---|---|---|
| correctness | covered | contains `correct` |
| behaviour (tool use) | covered | 35 of 35 cases have a check on the tool calls or rounds (checks `behaviour`) |
| quality | missing | no llm_judge scorer |
| regression gate | covered | gate on `behaviour`, max_drop 0.02, alpha 0.05, min_cases 30, with checks that need no baseline |
| reliability | missing | one trial per case: set `trials: 3` or more |
| cost & latency | missing | no max_cost_usd or max_p95_latency_ms in the gate |
| safety | covered | 5 of 35 cases tagged `safety` |
| robustness | covered | 5 of 35 cases tagged `robustness` |
| multi-turn | partial | 3 of 35 cases have turns; informational, since evalproof cannot tell how many multi-turn cases are enough |
| load | out of scope | use a load-testing tool such as k6 or Locust |
| production monitoring | out of scope | use an observability platform; evalproof tests a fixed set of cases before release |

Required: behaviour, regression-gate; all covered.
```

The gaps are real, and the example does not hide them:

* **quality** is missing because there is no model to judge a reply, and this example has none.
* **reliability** is missing because the suite runs every case once. The next section shows what
  repeated trials add, and `suite-flaky.yaml` runs five per case.
* **cost and latency** are missing because the gate sets no budget. A deterministic agent that
  costs nothing and answers in under a millisecond has nothing to budget.
* **multi-turn** is partial because three cases is a start, not a verdict.

`--require` turns coverage into an exit code: 0 when every named dimension is covered, 1 when one is
not. CI uses it, so a change that drops the checks from the suite fails the build.

## The files

| File | What it is |
|---|---|
| `weather_agent.py` | the agent: tool tables, keyword planner, `TraceRecorder` calls, the `calendar_for_outings` flag and the `flaky_calendar` chance |
| `cases.jsonl` | the 35 cases, each with `expect`, `tags`, a `mentions` list for the `correct` scorer, and (for 3 of them) `turns` |
| `suite.yaml` | the target, the two variants, the two scorers and the gate |
| `suite-flaky.yaml` | the same dataset, scorers and gate with the `v2` and `flaky` variants, for `--trials 5` |
| `runs/v1.json`, `runs/v2.json` | the committed run files |
