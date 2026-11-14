# API reference

The objects the [From Python](../guides/python.md) and [Agent SDK adapters](../guides/adapters.md)
guides use, generated from the docstrings in the package. Import them from `evalproof` and
`evalproof.adapters`; the names in `evalproof.__all__` are the supported API.

## Load, compare and gate

::: evalproof.load_run

::: evalproof.compare_runs

::: evalproof.Comparison

::: evalproof.GateConfig

::: evalproof.decide

::: evalproof.GateDecision

## Calibrate

::: evalproof.calibrate

::: evalproof.CalibrationReport

::: evalproof.Estimate

## Systems and traces

::: evalproof.RunContext

::: evalproof.Trace

::: evalproof.ToolCall

## Adapters

::: evalproof.adapters.TraceRecorder

::: evalproof.adapters.mcp_call

::: evalproof.adapters.adk.from_events

::: evalproof.adapters.langchain.from_messages

::: evalproof.adapters.claude_agent_sdk.from_messages

::: evalproof.adapters.openai_agents.from_result

::: evalproof.adapters.openai_agents.DEFAULT_TOOL_ERROR

::: evalproof.adapters.AgentResult
