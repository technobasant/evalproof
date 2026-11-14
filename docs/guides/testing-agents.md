# Testing agents

An agent can give a fluent answer for the wrong reason: it skipped a tool, called one with the
wrong argument, or made up a number. A text-only scorer cannot see that. evalproof tests what an
agent *did* as well as what it said, and none of it needs a model: the `checks` scorer reads a
trace, a record of the tool calls and replies, and the gate treats the result like any other
pass rate.

## What the system returns: the trace

A python system (`target: python:agent:run`) returns a dict, and the `trace` in it is what the
checks read.

| Key | What it is |
|---|---|
| `output` | the final reply, a string (required) |
| `trace` | a dict or an `evalproof.models.Trace` |
| `retrieved`, `tokens_in`, `tokens_out`, `cost_usd` | optional, as for any python system |

A `Trace` has `responses` (the reply to each user turn; the last is `output`), `tool_calls`,
`rounds` (model calls, when known), `usage` (tokens and cost per model call) and `meta` (anything
else). Each tool call is a `ToolCall`:

| Field | What it is |
|---|---|
| `name`, `args`, `result` | the tool, the arguments it got and what it returned |
| `ok`, `error` | `ok` is `false` for a call that failed, with the `error` text |
| `turn` | the 0-based user turn the call belongs to |
| `id`, `latency_ms` | optional |

`TraceRecorder` builds the dict by hand, for an agent in plain Python:

```python
# agent.py
from evalproof import RunContext
from evalproof.adapters import TraceRecorder


def run(question: str, ctx: RunContext) -> dict:
    rec = TraceRecorder()
    rec.round()
    rec.tool("get_weather", {"city": "Paris"}, result={"temp_c": 20})
    rec.reply("It is 20 degrees in Paris.")
    return rec.result()
```

A tool call belongs to the turn under way (the number of replies recorded so far) unless `turn=`
says otherwise; pass `ok=False` and an `error` for a call that failed. For an agent built on an
SDK, the [adapters](adapters.md#agent-sdk-adapters) build the same dict from what the SDK returns.

**`ctx`.** Declare a parameter named `ctx` and the function is called as `run(input, ctx=ctx)`.
A `RunContext` has `case_id`, `input`, `turns` (the user turns after the input), `state` (the
case's initial session state), `variant` and `vars` (the suite variant being run and its
variables), and `trial` (the repeat index, 0 to `trials - 1`; always 0 unless the suite sets
`trials`). Each call gets its own copies of the mutable parts.

## The `checks` scorer

```yaml
scorers:
  - name: behaviour
    type: checks
    params:
      on_missing: skip           # a case with no checks is left out; `pass` (the default) passes it
      checks:                    # applied to every case
        - {type: tool_call_count, max: 6}
gate: {metric: behaviour, max_drop: 0.02, min_cases: 30}
```

Each case lists its own checks in `expect` (or in the field `params.field` names), and the scorer
applies those after the suite-wide ones:

```json
{"id": "w01", "input": "What's the weather in Paris today?",
 "expect": [
   {"type": "tool_called", "name": "get_weather", "args": {"city": "Paris", "day": "today"}},
   {"type": "tool_not_called", "name": "get_events"},
   {"type": "output_cites_tool_result", "name": "get_weather", "min_matches": 2}]}
```

A case passes when every check passes (`mode: all`, the default); `mode: fraction` scores the
share that passed, and the case still counts as passed only when all did. A typo in a check, an
unknown key or a bad value fails when the suite or the dataset loads, with the path to it. A case
whose system errored fails every check.

| Check | Needs | Optional | It passes when |
|---|---|---|---|
| `tool_called` | `name` | `args`, `args_match`, `min`, `max`, `turn` | the tool was called (once or more, or as many times as `min` and `max` allow, counting only calls whose arguments match `args`); `max: 0` alone means never |
| `tool_not_called` | `name` | `args`, `args_match`, `turn` | no call of the tool (with matching `args`, if given) |
| `tool_args_contain` | `name`, `arg`, `value` | `turn` | some call of the tool has an argument `arg` that contains `value`: a substring of a string, a member of a list, otherwise equal |
| `forbidden_arg_values` | `arg`, `values` | `name`, `turn` | no call (of the tool `name`, if given) has `arg` equal to, or containing, one of `values` |
| `tool_order` | `calls` | `match`, `turn` | the calls (tool names, or objects with `name`, `args`, `args_match`) happened in this order; see `match` below |
| `tool_succeeded` | | `name`, `turn` | no call (of the tool `name`, if given) failed (`ok` is true on all) |
| `tool_call_count` | `min` or `max` | `name`, `turn` | the number of calls (of the tool `name`, if given) is within `min` and `max` |
| `rounds` | `min` or `max` | | the trace's `rounds` is within `min` and `max`; fails when the trace reports none |
| `output_contains` | `text` | `all`, `case_sensitive`, `turn` | the reply contains `text`, a string or a list of them (all of them unless `all: false`); not case sensitive unless `case_sensitive: true` |
| `output_omits` | `text` | `case_sensitive`, `turn` | the reply contains none of `text` |
| `output_regex` | `pattern` | `flags`, `turn` | `pattern` is found in the reply; `flags` are any of `i`, `m`, `s` |
| `output_cites_tool_result` | `name` | `min_matches`, `rel_tol`, `strict`, `turn` | the reply states at least `min_matches` (default 1) of the numbers in the tool's results, each equal within `rel_tol` (default 1e-6) or equal to the result rounded to the digits the reply wrote; 0, 1 and years are ignored unless `strict: true`; fails when the tool was not called |
| `any_of` | `checks` | | at least one of the listed checks passes |
| `not` | `check` | | the inner check fails |
| `python` | `fn` | | `fn` (`module:function`, importable from the suite's directory) returns true when called as `fn(case, output)` with the case and its `SystemOutput`; it may return a `bool`, a `(bool, reason)` pair or a `Score`; a function that raises leaves the case undecided |

How the keys work:

* **`args` and `args_match`.** Every key in `args` must be in the call's arguments with the same
  value (numbers compare as numbers, `1 == 1.0`). The default, `args_match: exact`, wants strings
  and lists to be equal; `args_match: contains` lets a string be a substring and a list hold at
  least the listed elements. Nested objects are compared the same way, key by key.
* **`match`** (in `tool_order`). `subsequence` (the default) wants the listed calls in that order,
  with others allowed in between. `exact` wants exactly those calls and no others, in that order.
  `any_order` wants each listed call matched by a call of its own, whatever the order.
* **`turn`.** A tool check looks only at the calls of that user turn; a text check reads the reply
  to that turn instead of the final output. Turns count from 0, and 0 is the case's `input`.
* **Missing evidence.** A check on the tool calls, rounds or a cited result fails with "no trace"
  when the output has no trace (`rounds` also fails when the trace reports no round count), and
  `not` does not turn that into a pass: "was not called" cannot be concluded from a missing trace.
  A text check on the reply of a turn that has none fails the same way.

## Several turns

A case with `turns` is a conversation: `input` is turn 0 and each later user turn follows. A case
may also carry `state`, an object holding the initial session state the system starts from.

```json
{"id": "m01", "input": "What's the weather like?", "turns": ["Paris"],
 "expect": [
   {"type": "tool_not_called", "name": "get_weather", "turn": 0},
   {"type": "output_contains", "text": "which city", "turn": 0},
   {"type": "tool_called", "name": "get_weather", "args": {"city": "Paris", "day": "today"}, "turn": 1}]}
```

A python system gets the later turns as `ctx.turns` and the state as `ctx.state`. An http system
gets `turns`, `state`, `variant` and `vars` in the POST body when they are set. A direct model
target plays the turns in order, each after the model's previous reply, and its output is the last
reply. A coding-agent CLI takes one prompt per call, so a suite with `turns` cases on one is refused
up front, before any case runs.

## Variants: one suite, several builds

A variant is another way to run the same cases, scorers and gate: a different model, prompt or
flag. Name them in the suite:

```yaml
variants:
  - name: v1
    vars: {calendar_for_outings: false}
  - name: v2
    vars: {calendar_for_outings: true}
    system: {target: "python:agent:run_v2"}   # optional: merged over the suite's `system`
```

```bash
evalproof run suite.yaml --variant v1 --variant v2     # or --all-variants
evalproof compare runs/v1.json runs/v2.json --suite suite.yaml --gate
```

Each variant writes `runs/<variant>.json` (`--out` names a file only when one variant runs), and
the run file records the variant, so `compare` names it in its report. A variant's `vars` reach a
python system as `ctx.vars`, an http system in the request body, and a model prompt as template
fields. A variant name uses only `a-z`, `0-9`, `.`, `_` and `-`.

## Example: a planted bug that only the tool calls show

[`examples/agent-tools`](https://github.com/technobasant/evalproof/blob/main/examples/agent-tools) is an agent in plain Python (two fake tools, a
keyword planner, no model and no network) with 35 cases, including `safety`, `robustness` and
three multi-turn cases, and two variants. `v1` has a planted bug: asked whether to go on a picnic,
the agent answers from the forecast alone and never looks at the calendar. Its reply still reads
well, so a text scorer passes it; the `behaviour` scorer fails 8 cases, and the gate fails v1
against v2 with exit 1. CI runs both directions on every commit. A second suite there,
`suite-flaky.yaml`, shows [repeated trials](repeated-trials.md#repeated-trials) on a build that is right on some runs
and wrong on others.

```bash
uv run evalproof run examples/agent-tools/suite.yaml --all-variants
uv run evalproof compare runs/v1.json runs/v2.json \
  --suite examples/agent-tools/suite.yaml --gate; echo $?   # 0: v2 fixes it
uv run evalproof compare runs/v2.json runs/v1.json \
  --suite examples/agent-tools/suite.yaml --gate; echo $?   # 1: v1 is the regression
```
