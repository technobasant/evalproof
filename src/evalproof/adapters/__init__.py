"""Adapters that turn what popular agent SDKs return into evalproof's trace contract.

A python system returns ``{"output", "trace", "tokens_in"?, "tokens_out"?, "cost_usd"?}``
(``AgentResult``); each adapter builds that dict from the SDK's own objects, or from the same
objects as plain dicts. None of them imports an SDK, so evalproof gains no dependency: the SDK is
yours, the adapter only reads what it returned.

* ``recorder.TraceRecorder`` and ``recorder.mcp_call``: build a trace by hand in plain Python.
* ``adk.from_events``: Google ADK events.
* ``langchain.from_messages``: LangChain and LangGraph messages.
* ``claude_agent_sdk.from_messages``: the messages the Claude Agent SDK yields.
* ``openai_agents.from_result``: an OpenAI Agents SDK ``RunResult``.
* ``otel.from_spans``: OpenTelemetry spans as OTLP/JSON or the Python SDK's JSON (experimental).
* Experimental, each read from the SDK source and not yet run against a real agent:
  ``llama_index.from_events`` (LlamaIndex agent events), ``crewai.from_output`` (a CrewAI
  ``CrewOutput``), ``pydantic_ai.from_messages`` (a Pydantic AI run), ``dspy.from_prediction`` (a
  ``dspy.ReAct`` prediction), ``haystack.from_messages`` (a Haystack ``Agent`` run),
  ``autogen.from_messages`` (an AutoGen AgentChat ``TaskResult``) and
  ``semantic_kernel.from_history`` (a Semantic Kernel ``ChatHistory``).
"""

from __future__ import annotations

from evalproof.adapters import (
    adk,
    autogen,
    claude_agent_sdk,
    crewai,
    dspy,
    haystack,
    langchain,
    llama_index,
    openai_agents,
    otel,
    pydantic_ai,
    semantic_kernel,
)
from evalproof.adapters._common import AgentResult
from evalproof.adapters.recorder import TraceRecorder, mcp_call

__all__ = [
    "AgentResult",
    "TraceRecorder",
    "adk",
    "autogen",
    "claude_agent_sdk",
    "crewai",
    "dspy",
    "haystack",
    "langchain",
    "llama_index",
    "mcp_call",
    "openai_agents",
    "otel",
    "pydantic_ai",
    "semantic_kernel",
]
