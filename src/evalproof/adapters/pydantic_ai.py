"""Pydantic AI (experimental): ``from_messages(result_or_messages)`` turns a run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real Pydantic AI agent.

Nothing here imports ``pydantic_ai``. It reads objects and the dicts the SDK's own message
serialisation produces the same way.

SDK fields read (names confirmed in the SDK source, not from memory):

* pydantic-ai==2.68.0 pydantic_ai_slim/pydantic_ai/run.py:962 ``AgentRunResult``, :965 ``output``,
  :1103 ``all_messages()`` (the whole conversation, including ``message_history``).
* pydantic_ai/messages.py:3759 ``ModelRequest`` (:3762 ``parts``, :3781 ``kind == "request"``) and
  :4914 ``ModelResponse`` (``parts``, :4928 ``usage``, :4937 ``model_name``, :4963
  ``kind == "response"``).
* Request parts, told apart by ``part_kind``: :1655 ``UserPromptPart`` (``"user-prompt"``, :1670),
  :2708 ``ToolReturnPart`` (``"tool-return"``, :2727) with :2373 ``tool_name``, :2376 ``content``,
  :2379 ``tool_call_id`` and :2403 ``outcome`` (``"success"``, ``"failed"``, ``"denied"``,
  ``"interrupted"`` or ``"retried"``), and :2851 ``RetryPromptPart`` (``"retry-prompt"``, :2889)
  with :2868 ``content`` (a string, or a list of validation errors), :2877 ``tool_name`` and :2880
  ``tool_call_id``.
* Response parts: :3944 ``TextPart`` (``"text"``, :3982) ``content``, and :4422 ``ToolCallPart``
  (``"tool-call"``, :4427) with ``tool_name``, ``tool_call_id`` and :4301 ``args`` (a JSON string,
  a dict or None) from :4295 ``BaseToolCallPart``.
* pydantic_ai/usage.py:126-153 ``RequestUsage.input_tokens``, ``output_tokens`` and
  ``cache_read_tokens``.

A user prompt starts a new turn once the agent has answered, so a transcript of several turns gets
turns 0, 1, ...; with no user prompt everything is turn 0. The reply of a turn is the text of its
last response that calls no tool. A tool call answered by a ``tool-return`` part whose ``outcome``
is not ``"success"``, or by a ``retry-prompt`` part that names a tool, is a failed call. Left out:
provider-side tool calls (``"builtin-tool-call"``, which the model provider ran, not your code),
retry prompts with no tool name (a retry of the final output, not a tool), and cost, which
Pydantic AI does not put on the messages. An agent with structured output also shows its output
tool as a call. Passing the run result, not only the messages, makes ``output`` the final output
(JSON when it is not a string).
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


def from_messages(source: Any) -> AgentResult:
    """Build ``{"output", "trace"}`` from a Pydantic AI run (objects or dicts).

    ``source`` is the run result (``agent.run_sync(...)``), the list of messages
    (``result.all_messages()``), or a dict with ``messages`` (and optionally ``output``). Each
    ``ToolCallPart`` is a call; a ``ToolReturnPart`` answers the call with its ``tool_call_id``.
    ``rounds`` counts the model responses.
    """
    warn_experimental(
        "pydantic_ai",
        "its message shapes were read from the SDK source and not yet run against a real agent",
    )
    output = None
    messages: Any = source
    if callable(get(source, "all_messages")):
        output = get(source, "output")
        messages = source.all_messages()
    elif isinstance(source, Mapping):
        output = get(source, "output")
        messages = get(source, "messages", default=[])
    ledger = Ledger()
    turn = 0
    active = False
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for message in _items(messages):
        parts = _items(get(message, "parts"))
        if get(message, "kind") == "response":
            active = True
            rounds += 1
            texts: list[str] = []
            called = False
            for part in parts:
                kind = get(part, "part_kind")
                if kind == "tool-call":
                    called = True
                    ledger.call(
                        get(part, "tool_name"),
                        get(part, "args"),
                        get(part, "tool_call_id"),
                        turn,
                    )
                elif kind == "text":
                    texts.append(text_of(get(part, "content")))
            meta = get(message, "usage")
            if meta is not None:
                add_usage(
                    usage,
                    get(message, "model_name"),
                    get(meta, "input_tokens"),
                    get(meta, "output_tokens"),
                    get(meta, "cache_read_tokens"),
                )
            text = "".join(texts)
            if text and not called:
                replies[turn] = text
            continue
        for part in parts:
            kind = get(part, "part_kind")
            if kind == "user-prompt":
                if active:
                    turn += 1
                    active = False
            elif kind == "tool-return":
                active = True
                failed = get(part, "outcome", default="success") != "success"
                content = get(part, "content")
                ledger.result(
                    get(part, "tool_call_id"),
                    get(part, "tool_name"),
                    content,
                    ok=not failed,
                    error=(final_text(content) or "tool error") if failed else None,
                    turn=turn,
                )
            elif kind == "retry-prompt" and get(part, "tool_name") is not None:
                active = True
                content = get(part, "content")
                ledger.result(
                    get(part, "tool_call_id"),
                    get(part, "tool_name"),
                    jsonable(content),
                    ok=False,
                    error=final_text(content) or "tool error",
                    turn=turn,
                )
    if output is not None and (replies or rounds):
        replies[turn] = final_text(output, replies.get(turn, ""))
    return finish(replies, ledger, rounds, usage)


def _items(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


__all__ = ["from_messages"]
