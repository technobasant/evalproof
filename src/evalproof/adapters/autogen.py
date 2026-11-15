"""AutoGen AgentChat (experimental): ``from_messages(result_or_messages)`` turns a run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real AutoGen agent or team.

Nothing here imports ``autogen_agentchat``. It reads message objects and the dicts that
``message.dump()`` produces the same way.

SDK fields read (names confirmed in the SDK source, not from memory):

* autogen-agentchat==0.7.5 (tag python-v0.7.5) autogen_agentchat/base/_task.py:9 ``TaskResult``
  (``messages``, ``stop_reason``).
* autogen_agentchat/messages.py: :86 ``source`` and :89 ``models_usage`` on chat messages, :161 and
  :164 the same on events; :367 ``TextMessage`` (``type == "TextMessage"``, ``content`` a string),
  :178 ``StructuredMessage`` (``content`` a model), :373 ``MultiModalMessage`` (``content`` a list
  of strings and images), :415 ``StopMessage``, :421 ``HandoffMessage``, :433
  ``ToolCallSummaryMessage``, :445 ``ToolCallRequestEvent`` (``content``: a list of
  ``FunctionCall``) and :490 ``ToolCallExecutionEvent`` (``content``: a list of
  ``FunctionExecutionResult``); :55 ``dump()`` is ``model_dump(mode="json")``.
* autogen-core==0.7.5 autogen_core/_types.py:7 ``FunctionCall`` (``id``, ``name``, ``arguments``: a
  JSON string); autogen_core/models/_types.py:56 ``FunctionExecutionResult`` (``content``: a
  string, ``name``, ``call_id``, ``is_error``: a bool or None) and :86 ``RequestUsage``
  (``prompt_tokens``, ``completion_tokens``).
* autogen_agentchat/agents/_base_chat_agent.py:130 and teams/_group_chat/_base_group_chat.py:460:
  the task given to ``run`` is the first message, a ``TextMessage`` whose ``source`` is
  ``"user"``.

A message whose ``source`` is ``"user"`` starts a new turn once an agent has answered, so a
transcript of several turns gets turns 0, 1, ...; with none, everything is turn 0. The reply of a
turn is the text of the last agent message of a text kind (``TextMessage``, ``StructuredMessage``
as JSON, ``MultiModalMessage``, ``HandoffMessage``, ``StopMessage``, ``ToolCallSummaryMessage``).
A result whose ``is_error`` is true is a failed call. ``rounds`` counts the agent messages and
tool-call requests that report ``models_usage``. Left out: the ``tool_calls`` and ``results`` of
a ``ToolCallSummaryMessage`` (they repeat the request and execution events), thoughts, memory
and streaming-chunk events, the model name (``RequestUsage`` has none, so usage has no model),
cached tokens and cost (no confirmed field).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from evalproof.adapters._common import (
    AgentResult,
    Ledger,
    add_usage,
    final_text,
    finish,
    get,
    text_of,
    warn_experimental,
)
from evalproof.models import ModelUsage

_TEXT_KINDS = frozenset(
    {
        "TextMessage",
        "StructuredMessage",
        "MultiModalMessage",
        "HandoffMessage",
        "StopMessage",
        "ToolCallSummaryMessage",
    }
)


def _list(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


def _text(kind: str, content: Any) -> str:
    if kind == "MultiModalMessage":
        return text_of(_list(content))
    if isinstance(content, str):
        return content
    return final_text(content)


def from_messages(source: Any) -> AgentResult:
    """Build ``{"output", "trace"}`` from an AutoGen AgentChat run (objects or dicts).

    ``source`` is the ``TaskResult`` of ``agent.run(...)`` or ``team.run(...)``, its list of
    ``messages``, or a dict with ``messages``. Each ``FunctionCall`` of a
    ``ToolCallRequestEvent`` is a call; a ``FunctionExecutionResult`` answers the call with its
    ``call_id``.
    """
    warn_experimental(
        "autogen",
        "its message shapes were read from the SDK source and not yet run against a real agent",
    )
    messages = get(source, "messages", default=source) if _has_messages(source) else source
    ledger = Ledger()
    turn = 0
    active = False
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for message in _list(messages):
        kind = str(get(message, "type", default=""))
        user = get(message, "source") == "user"
        if user and kind in _TEXT_KINDS:
            if active:
                turn += 1
                active = False
            continue
        stats = get(message, "models_usage")
        if kind == "ToolCallRequestEvent":
            active = True
            for call in _list(get(message, "content")):
                ledger.call(get(call, "name"), get(call, "arguments"), get(call, "id"), turn)
        elif kind == "ToolCallExecutionEvent":
            active = True
            for item in _list(get(message, "content")):
                failed = bool(get(item, "is_error", default=False))
                content = get(item, "content")
                ledger.result(
                    get(item, "call_id"),
                    get(item, "name"),
                    content,
                    ok=not failed,
                    error=(final_text(content) or "tool error") if failed else None,
                    turn=turn,
                )
            continue
        elif kind in _TEXT_KINDS:
            active = True
            text = _text(kind, get(message, "content"))
            if text:
                replies[turn] = text
        else:
            continue
        if stats is not None:
            rounds += 1
            add_usage(usage, None, get(stats, "prompt_tokens"), get(stats, "completion_tokens"))
    return finish(replies, ledger, rounds, usage)


def _has_messages(source: Any) -> bool:
    if isinstance(source, Mapping):
        return "messages" in source
    return not isinstance(source, str | list | tuple) and hasattr(source, "messages")


__all__ = ["from_messages"]
