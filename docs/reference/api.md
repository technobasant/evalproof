---
description: The Python objects evalproof exports, generated from the docstrings: load, compare, gate, calibrate, traces and adapters.
---

# API reference

The objects the [From Python](../guides/python.md) and [Agent SDK adapters](../guides/adapters.md)
guides use, generated from the docstrings in the package. Import them from `evalproof` and
`evalproof.adapters`; the names in `evalproof.__all__` are the supported API.

## Load, compare and gate

Read run files, compare two of them and turn the comparison into a gate decision.

::: evalproof.load_run

::: evalproof.compare_runs

::: evalproof.Comparison

::: evalproof.GateConfig

::: evalproof.decide

::: evalproof.GateDecision

## Calibrate

Measure a judge against gold labels and correct its pass rate.

::: evalproof.calibrate

::: evalproof.CalibrationReport

::: evalproof.Estimate

## Systems and traces

The objects a python system receives and returns.

::: evalproof.RunContext

::: evalproof.Trace

::: evalproof.ToolCall

## Adapters

Build a trace from what an agent SDK returns.

::: evalproof.adapters.TraceRecorder

::: evalproof.adapters.mcp_call

::: evalproof.adapters.adk.from_events

::: evalproof.adapters.langchain.from_messages

::: evalproof.adapters.claude_agent_sdk.from_messages

::: evalproof.adapters.openai_agents.from_result

::: evalproof.adapters.openai_agents.DEFAULT_TOOL_ERROR

::: evalproof.adapters.AgentResult
