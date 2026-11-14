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
| reliability | missing | repeated trials are not in this release |
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
* **reliability** is missing because the suite runs every case once. Repeated trials are not in
  this release.
* **cost and latency** are missing because the gate sets no budget. A deterministic agent that
  costs nothing and answers in under a millisecond has nothing to budget.
* **multi-turn** is partial because three cases is a start, not a verdict.

`--require` turns coverage into an exit code: 0 when every named dimension is covered, 1 when one is
not. CI uses it, so a change that drops the checks from the suite fails the build.

## The files

| File | What it is |
|---|---|
| `weather_agent.py` | the agent: tool tables, keyword planner, `TraceRecorder` calls, and the `calendar_for_outings` flag |
| `cases.jsonl` | the 35 cases, each with `expect`, `tags`, a `mentions` list for the `correct` scorer, and (for 3 of them) `turns` |
| `suite.yaml` | the target, the two variants, the two scorers and the gate |
| `runs/v1.json`, `runs/v2.json` | the committed run files |
