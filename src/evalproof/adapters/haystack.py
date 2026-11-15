"""Haystack (experimental): ``from_messages(messages)`` turns an ``Agent`` run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real Haystack agent.

Nothing here imports ``haystack``. It reads ``ChatMessage`` objects and the dicts that
``ChatMessage.to_dict()`` produces the same way.

SDK fields read (names confirmed in the SDK source, not from memory):

* haystack-ai==3.5.0 haystack/dataclasses/chat_message.py:25 ``ChatRole`` (``"user"``,
  ``"system"``, ``"assistant"``, ``"tool"``; a ``str`` enum, so ``.value`` is read), :82
  ``ToolCall`` (``tool_name``, ``arguments``, ``id``), :124 ``ToolCallResult`` (``result``: a
  string or a list of text and image parts, ``origin``: the ``ToolCall``, ``error``: a bool),
  :306 ``ChatMessage`` with ``role`` :322, ``meta`` :329, ``name`` :336, ``texts`` :343,
  ``tool_calls`` :359 and ``tool_call_results`` :375.
* The dict form, ``ChatMessage.to_dict()`` (chat_message.py:284-287): ``role``, ``meta``,
  ``name`` and ``content``, a list of ``{"text": ...}``, ``{"tool_call": {...}}`` and
  ``{"tool_call_result": {"result", "origin", "error"}}`` parts (the keys at :228-235).
* haystack/components/agents/agent.py:911-928: ``Agent.run`` returns a dict whose ``messages`` is
  every message of the run (the input messages included).
* haystack/components/generators/chat/openai.py:748-751 and :98: an assistant message's
  ``meta["model"]`` and ``meta["usage"]`` (``prompt_tokens``, ``completion_tokens``);
  haystack/components/agents/utils.py:15-17 the other generator's names for them
  (``input_tokens``, ``output_tokens``).

A user message starts a new turn once the agent has answered, so a transcript of several turns
gets turns 0, 1, ...; with no user message everything is turn 0. The reply of a turn is the first
text of its last assistant message that calls no tool. A tool result with ``error`` true is a
failed call. Left out: cached-token counts and cost (no confirmed field), and ``Agent`` runs
that only return ``last_message`` (pass ``messages``).
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
    jsonable,
    text_of,
    warn_experimental,
)
from evalproof.models import ModelUsage


def _role(message: Any) -> str:
    role = get(message, "role", "_role")
    return str(getattr(role, "value", role) or "").lower()


def _content(message: Any) -> tuple[list[str], list[Any], list[Any]]:
    """The texts, tool calls and tool results of a message: from its ``content`` parts when it is
    a dict, from the ``texts``, ``tool_calls`` and ``tool_call_results`` properties otherwise."""
    texts: list[str] = []
    calls: list[Any] = []
    results: list[Any] = []
    if isinstance(message, Mapping):
        for part in _list(message.get("content")):
            if isinstance(part, Mapping):
                if isinstance(part.get("text"), str):
                    texts.append(part["text"])
                if part.get("tool_call") is not None:
                    calls.append(part["tool_call"])
                if part.get("tool_call_result") is not None:
                    results.append(part["tool_call_result"])
        return texts, calls, results
    for text in _list(get(message, "texts")):
        if isinstance(text, str):
            texts.append(text)
    if not texts and isinstance(get(message, "text"), str):
        texts.append(message.text)
    return texts, _list(get(message, "tool_calls")), _list(get(message, "tool_call_results"))


def _list(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


def _value(result: Any) -> Any:
    """A tool result's content: the text when it is text, else the parts as JSON."""
    if isinstance(result, str) or result is None:
        return result
    text = text_of(result)
    return text if text else jsonable(result)


def from_messages(messages: Iterable[Any] | Mapping[str, Any] | None) -> AgentResult:
    """Build ``{"output", "trace"}`` from Haystack chat messages (objects or dicts).

    Accepts the list, or the dict ``Agent.run`` returns (its ``messages``). Each ``ToolCall`` of
    an assistant message is a call; a ``ToolCallResult`` answers the call it names as its
    ``origin`` (by ``id``, else by name and order). ``rounds`` counts the assistant messages.
    """
    warn_experimental(
        "haystack",
        "its message shapes were read from the SDK source and not yet run against a real agent",
    )
    if isinstance(messages, Mapping):
        messages = get(messages, "messages", default=[])
    ledger = Ledger()
    turn = 0
    active = False
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for message in _list(messages):
        role = _role(message)
        texts, calls, results = _content(message)
        if role == "user":
            if active:
                turn += 1
                active = False
        elif role == "assistant":
            active = True
            rounds += 1
            for call in calls:
                ledger.call(get(call, "tool_name"), get(call, "arguments"), get(call, "id"), turn)
            meta = get(message, "meta", "_meta")
            stats = get(meta, "usage")
            if stats is not None:
                add_usage(
                    usage,
                    get(meta, "model"),
                    get(stats, "prompt_tokens", "input_tokens"),
                    get(stats, "completion_tokens", "output_tokens"),
                )
            if texts and texts[0] and not calls:
                replies[turn] = texts[0]
        elif role == "tool" or results:
            active = True
            for item in results:
                origin = get(item, "origin")
                failed = bool(get(item, "error", default=False))
                content = get(item, "result")
                ledger.result(
                    get(origin, "id"),
                    get(origin, "tool_name"),
                    _value(content),
                    ok=not failed,
                    error=(final_text(_value(content)) or "tool error") if failed else None,
                    turn=turn,
                )
    return finish(replies, ledger, rounds, usage)


__all__ = ["from_messages"]
