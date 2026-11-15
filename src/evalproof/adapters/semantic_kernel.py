"""Semantic Kernel (experimental): ``from_history(history)`` turns a chat history into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real Semantic Kernel agent.

Nothing here imports ``semantic_kernel``. It reads ``ChatHistory`` and ``ChatMessageContent``
objects and the dicts that ``model_dump()`` produces the same way.

SDK fields read (names confirmed in the SDK source, not from memory):

* semantic-kernel==1.47.0 semantic_kernel/contents/chat_history.py:30 ``ChatHistory`` and :44
  ``messages``.
* contents/chat_message_content.py:75 ``ChatMessageContent`` with :97 ``role`` (an ``AuthorRole``),
  :99 ``items`` and :200 ``content`` (the text of the first ``TextContent`` item, else ``""``);
  contents/utils/author_role.py:5 ``AuthorRole`` (a ``str`` enum: ``"system"``, ``"user"``,
  ``"assistant"``, ``"tool"``, ``"developer"``; ``.value`` is read).
* contents/const.py:8-17 the ``content_type`` tags of the items: ``"text"``, ``"function_call"``
  and ``"function_result"``.
* contents/text_content.py:33 ``TextContent.text``.
* contents/function_call_content.py:34 ``FunctionCallContent``: ``id``, ``call_id``, ``name`` (the
  plugin and function name together), ``function_name``, ``plugin_name`` and ``arguments`` (a JSON
  string, a mapping or None).
* contents/function_result_content.py:30 ``FunctionResultContent``: ``id`` (the id of the function
  call it answers), ``call_id``, ``name``, ``function_name``, ``plugin_name`` and ``result``.

A user message starts a new turn once the agent has answered, so a history of several turns gets
turns 0, 1, ...; with no user message everything is turn 0. Each ``FunctionCallContent`` is a
call, named ``name`` (or ``function_name``); a ``FunctionResultContent`` answers the call with the
same ``id`` (or ``call_id``). The reply of a turn is the text of its last assistant message that
calls no function; ``rounds`` counts the assistant messages. Left out, because the SDK source
shows no field for them: failed calls (a result carries no error flag, so every result is ``ok``),
token usage and cost (they live in connector-specific ``metadata``).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from evalproof.adapters._common import AgentResult, Ledger, finish, get, warn_experimental


def _list(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "").lower()


def _name(item: Any) -> Any:
    return get(item, "name", "function_name")


def from_history(history: Any) -> AgentResult:
    """Build ``{"output", "trace"}`` from a Semantic Kernel chat history (objects or dicts).

    ``history`` is a ``ChatHistory``, the list of its ``messages``, or a dict with ``messages``.
    Function calls are read from ``FunctionCallContent`` items, function results from
    ``FunctionResultContent`` items.
    """
    warn_experimental(
        "semantic_kernel",
        "its content shapes were read from the SDK source and not yet run against a real agent",
    )
    if isinstance(history, Mapping) or (
        not isinstance(history, str | list | tuple) and hasattr(history, "messages")
    ):
        history = get(history, "messages", default=[])
    ledger = Ledger()
    turn = 0
    active = False
    replies: dict[int, str] = {}
    rounds = 0
    for message in _list(history):
        role = _enum(get(message, "role"))
        if role == "user":
            if active:
                turn += 1
                active = False
            continue
        if role not in ("assistant", "tool"):
            continue
        active = True
        texts: list[str] = []
        called = False
        for item in _list(get(message, "items")):
            kind = _enum(get(item, "content_type"))
            if kind == "function_call":
                called = True
                ledger.call(_name(item), get(item, "arguments"), get(item, "id", "call_id"), turn)
            elif kind == "function_result":
                ledger.result(
                    get(item, "id", "call_id"),
                    _name(item),
                    get(item, "result"),
                    turn=turn,
                )
            elif kind == "text" and isinstance(get(item, "text"), str):
                texts.append(get(item, "text"))
        if role != "assistant":
            continue
        rounds += 1
        text = texts[0] if texts else ""
        if not texts and isinstance(get(message, "content"), str):
            text = get(message, "content")
        if text and not called:
            replies[turn] = text
    return finish(replies, ledger, rounds, {})


__all__ = ["from_history"]
