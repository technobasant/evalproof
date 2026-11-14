"""Claude Agent SDK: ``from_messages(messages)`` turns the messages ``query()`` yields into a trace.

SDK fields read (names confirmed in the SDK source, not from memory), all
claude-agent-sdk==0.2.182 claude_agent_sdk/types.py:

* :1225-1227 ``ToolUseBlock.id``, ``.name``, ``.input``; :1234-1236 ``ToolResultBlock.tool_use_id``,
  ``.content`` (a string, a list of dicts or None), ``.is_error``; text blocks have ``.text`` and
  thinking blocks ``.thinking`` instead.
* :1379 ``UserMessage.content`` (a string or a list of blocks; tool results arrive here);
  :1394-1399 ``AssistantMessage.content``, ``.model``, ``.message_id``.
* :1711-1722 ``ResultMessage.subtype``, ``.is_error``, ``.num_turns``, ``.total_cost_usd``,
  ``.result``, ``.model_usage``; :1689-1694 the per-model ``ModelUsage`` keys ``inputTokens``,
  ``outputTokens``, ``cacheReadInputTokens`` and ``costUSD`` (camelCase, passed through from the
  CLI: _internal/message_parser.py:323).
* :1813-1815 the totals on a ``ResultMessage`` (``total_cost_usd``) are running totals of the
  session, so the last result's totals are used, not a sum.

``ResultMessage.usage`` is an untyped ``dict[str, Any]`` and its keys are not defined anywhere in
the SDK source, so it is not read; ``model_usage`` carries the same tokens with confirmed names.
A turn ends at each ``ResultMessage``; with none, every call is turn 0.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from evalproof.adapters._common import (
    AgentResult,
    Ledger,
    count,
    get,
    has,
    jsonable,
    number,
    text_of,
)
from evalproof.models import ModelUsage, Trace


def _is_result(message: Any) -> bool:
    return any(has(message, f) for f in ("num_turns", "total_cost_usd", "is_error", "model_usage"))


def _blocks(message: Any) -> list[Any]:
    content = get(message, "content")
    return list(content) if isinstance(content, list | tuple) else []


def _value(content: Any) -> Any:
    if isinstance(content, str) or content is None:
        return content
    text = text_of(content)
    return text if text else jsonable(content)


def from_messages(messages: Iterable[Any]) -> AgentResult:
    """Build ``{"output", "trace", "cost_usd"?}`` from Claude Agent SDK messages (objects or dicts).

    ``ToolUseBlock`` entries are calls and ``ToolResultBlock`` entries (in user messages) answer
    them by id, ``is_error`` making the call fail. The reply of a turn is ``ResultMessage.result``,
    else the text of its last assistant message. ``cost_usd`` is ``total_cost_usd``, the only
    cost the SDK reports for the run; usage comes from ``model_usage``. ``rounds`` counts the
    assistant responses (messages with the same ``message_id`` are one response).
    """
    ledger = Ledger()
    turn = 0
    texts: dict[int, str] = {}
    replies: dict[int, str] = {}
    seen_ids: set[str] = set()
    rounds = 0
    cost: float | None = None
    model_usage: Any = None
    meta: dict[str, Any] = {}
    for message in messages:
        if _is_result(message):
            result = get(message, "result")
            reply = result if isinstance(result, str) else texts.get(turn)
            if reply is not None:
                replies[turn] = reply
            total = number(get(message, "total_cost_usd"))
            if total is not None:
                cost = total
            if get(message, "model_usage") is not None:
                model_usage = get(message, "model_usage")
            if get(message, "num_turns") is not None:
                meta["num_turns"] = get(message, "num_turns")
            if get(message, "is_error", default=False) is True:
                meta["is_error"] = True
            turn += 1
            continue
        blocks = _blocks(message)
        if has(message, "model"):  # an assistant message
            message_id = get(message, "message_id")
            if message_id is None or str(message_id) not in seen_ids:
                rounds += 1
                if message_id is not None:
                    seen_ids.add(str(message_id))
            for block in blocks:
                if has(block, "tool_use_id"):
                    continue
                if has(block, "name") and has(block, "input"):
                    ledger.call(get(block, "name"), get(block, "input"), get(block, "id"), turn)
            text = text_of(blocks)
            if text and get(message, "parent_tool_use_id") is None:
                texts[turn] = text
        else:  # a user message: tool results arrive here
            for block in blocks:
                if not has(block, "tool_use_id"):
                    continue
                failed = get(block, "is_error", default=False) is True
                content = get(block, "content")
                ledger.result(
                    get(block, "tool_use_id"),
                    None,
                    _value(content),
                    ok=not failed,
                    error=(text_of(content) or "tool error") if failed else None,
                    turn=turn,
                )
    usage = [
        ModelUsage(
            role="agent",
            model=str(name),
            tokens_in=count(get(entry, "inputTokens")),
            tokens_out=count(get(entry, "outputTokens")),
            cached_tokens=count(get(entry, "cacheReadInputTokens")),
            cost_usd=number(get(entry, "costUSD")),
        )
        for name, entry in (model_usage.items() if hasattr(model_usage, "items") else [])
    ]
    for turn_index, text in texts.items():  # a turn with no ResultMessage keeps its last text
        replies.setdefault(turn_index, text)
    responses = [replies.get(i, "") for i in range(max(replies) + 1)] if replies else []
    trace = Trace(
        responses=responses,
        tool_calls=ledger.calls,
        rounds=rounds or None,
        usage=usage,
        meta=meta,
    )
    out: AgentResult = {"output": responses[-1] if responses else "", "trace": trace}
    if cost is not None:
        out["cost_usd"] = cost
    return out


__all__ = ["from_messages"]
