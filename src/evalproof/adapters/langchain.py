"""LangChain / LangGraph: ``from_messages(messages)`` turns a message list into a trace.

SDK fields read (names confirmed in the SDK source, not from memory):

* langchain-core==1.8.0 langchain_core/messages/ai.py:170 ``AIMessage.tool_calls``, :176
  ``usage_metadata``; :138 ``UsageMetadata.input_tokens``, :141 ``output_tokens``, :147
  ``input_token_details``; :66 ``InputTokenDetails.cache_read``.
* langchain-core==1.8.0 langchain_core/messages/tool.py:225-231 ``ToolCall.name``, ``.args``,
  ``.id``; :67 ``ToolMessage.tool_call_id``, :81 ``status`` (``"success"`` or ``"error"``).
* langchain-core==1.8.0 langchain_core/messages/base.py:102 ``content`` (a string or a list of
  strings and blocks), :113 ``response_metadata``, :116 ``type``, :124 ``name``; :262 ``text``
  (only blocks with ``type: "text"`` count).
* langchain-core==1.8.0 langchain_core/messages/utils.py:664-668 the type names ``human``/``user``
  and ``ai``/``assistant``; langchain_core/language_models/_compat_bridge.py:468
  ``response_metadata["model_name"]``.
* langchain==1.5.0 langchain/agents/middleware/types.py:354 ``AgentState.messages``: a
  ``create_agent`` run returns a dict whose ``messages`` is the list this reads.

A turn boundary is a human message, so a transcript of several turns gets turns 0, 1, ...; a
list with no human message is turn 0 throughout. LangChain reports no cost, so none is set.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from evalproof.adapters._common import AgentResult, Ledger, count, get, jsonable, text_of
from evalproof.models import ModelUsage, Trace


def _kind(message: Any) -> str:
    name = str(get(message, "type", "role", default="")).lower()
    if name in ("human", "user"):
        return "human"
    if name.startswith("ai") or name == "assistant":
        return "ai"
    if name.startswith("tool"):
        return "tool"
    return name


def _value(content: Any) -> Any:
    """A tool message's content as a result: the text when it is text, else the blocks."""
    if isinstance(content, str) or content is None:
        return content
    text = text_of(content)
    return text if text else jsonable(content)


def from_messages(messages: Iterable[Any] | Mapping[str, Any]) -> AgentResult:
    """Build ``{"output", "trace"}`` from LangChain messages (objects or dicts).

    Accepts the list, or the state dict a ``create_agent`` run returns (its ``messages``). Each
    ``AIMessage.tool_calls`` entry is a call; a ``ToolMessage`` answers the call with its
    ``tool_call_id``, and ``status == "error"`` makes it fail. The reply of a turn is the text of
    its last AI message that calls no tool; ``rounds`` counts the AI messages.
    """
    if isinstance(messages, Mapping):
        messages = get(messages, "messages", default=[])
    ledger = Ledger()
    turn = 0
    active = False
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for message in messages:
        kind = _kind(message)
        if kind == "human":
            if active:
                turn += 1
                active = False
            continue
        if kind == "ai":
            active = True
            rounds += 1
            calls = get(message, "tool_calls", default=[])
            for call in calls:
                ledger.call(get(call, "name"), get(call, "args"), get(call, "id"), turn)
            meta = get(message, "usage_metadata")
            if meta is not None:
                model = get(get(message, "response_metadata"), "model_name")
                entry = usage.setdefault(model, ModelUsage(role="agent", model=model))
                entry.tokens_in += count(get(meta, "input_tokens"))
                entry.tokens_out += count(get(meta, "output_tokens"))
                entry.cached_tokens += count(get(get(meta, "input_token_details"), "cache_read"))
            text = text_of(get(message, "content"))
            if text and not calls:
                replies[turn] = text
        elif kind == "tool":
            active = True
            failed = get(message, "status") == "error"
            content = get(message, "content")
            ledger.result(
                get(message, "tool_call_id"),
                get(message, "name"),
                _value(content),
                ok=not failed,
                error=(text_of(content) or "tool error") if failed else None,
                turn=turn,
            )
    responses = [replies.get(i, "") for i in range(max(replies) + 1)] if replies else []
    trace = Trace(
        responses=responses,
        tool_calls=ledger.calls,
        rounds=rounds or None,
        usage=list(usage.values()),
    )
    return {"output": responses[-1] if responses else "", "trace": trace}


__all__ = ["from_messages"]
