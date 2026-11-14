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
"""

from __future__ import annotations

from evalproof.adapters import adk, claude_agent_sdk, langchain, openai_agents
from evalproof.adapters._common import AgentResult
from evalproof.adapters.recorder import TraceRecorder, mcp_call

__all__ = [
    "AgentResult",
    "TraceRecorder",
    "adk",
    "claude_agent_sdk",
    "langchain",
    "mcp_call",
    "openai_agents",
]
