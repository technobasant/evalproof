"""LlamaIndex (experimental): ``from_events(events)`` turns an agent run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real LlamaIndex agent.

Nothing here imports ``llama_index``. It reads event objects, ``AgentChatResponse`` objects and
the dicts ``model_dump()`` produces the same way. The events carry no type tag, so an event is
told apart by its fields.

SDK fields read (names confirmed in the SDK source, not from memory):

* llama-index-core==0.14.26 llama_index/core/agent/workflow/workflow_events.py: :71
  ``AgentOutput`` (``response``: a ``ChatMessage``, ``structured_response``, ``tool_calls``: a
  list of ``ToolSelection``, one per model response), :98 ``ToolCall`` (``tool_name``,
  ``tool_kwargs``, ``tool_id``), :106 ``ToolCallResult`` (the same three, and ``tool_output``).
  Also :24 ``AgentInput`` and :38 ``AgentStream``, which are read for nothing: a stream chunk
  repeats what ``AgentOutput`` holds.
* llama_index/core/llms/llm.py:70 ``ToolSelection`` (``tool_id``, ``tool_name``, ``tool_kwargs``).
* llama_index/core/tools/types.py:106 ``ToolOutput`` (``blocks``, ``tool_name``, ``raw_input``,
  ``raw_output``, ``is_error``) and :147 ``content`` (the text of its text blocks).
* llama_index/core/base/llms/types.py:1159 ``ChatMessage`` (``blocks``) and :1201 ``content`` (the
  text of its text blocks, or None).
* llama_index/core/chat_engine/types.py:59 ``AgentChatResponse`` (``response``: a string,
  ``sources``: a list of ``ToolOutput``), what the older agents and chat engines return.

``events`` is what ``handler.stream_events()`` yielded. The events carry no turn, so a list of
lists is one run per user turn (each ``agent.run(...)`` call), and a flat list is turn 0. A tool
call is seen twice, as a ``ToolCall`` event and in the ``tool_calls`` of
the ``AgentOutput`` that made it, so calls are counted once per ``tool_id``. The reply of a run is
the text of its last ``AgentOutput`` that makes no tool call, or its JSON ``structured_response``;
``rounds`` counts the ``AgentOutput`` events. A ``ToolOutput`` with ``is_error`` true is a failed
call. An ``AgentChatResponse`` is one turn with one call per entry of ``sources``, which carry no
ids (so they pair by order) and no round count. Left out, because the source shows no stable field:
token usage and cost (``AgentOutput.raw`` is the provider's own object, excluded from dumps).
"""

from __future__ import annotations

from typing import Any

from evalproof.adapters._common import (
    AgentResult,
    Ledger,
    final_text,
    finish,
    get,
    has,
    jsonable,
    runs,
    text_of,
    warn_experimental,
)


def _is_chat_response(item: Any) -> bool:
    return has(item, "sources") and isinstance(get(item, "response"), str)


def _text(message: Any) -> str:
    content = get(message, "content")
    if isinstance(content, str):
        return content
    return text_of(get(message, "blocks"))


def _output_text(output: Any) -> str:
    """A tool output as a result: its text, else its raw output."""
    if output is None:
        return ""
    return _text(output)


def _runs(source: Any) -> list[list[Any]]:
    if isinstance(source, list | tuple):
        items = list(source)
        if items and all(_is_chat_response(item) for item in items):
            return [[item] for item in items]
        return runs(items)
    return [[source]] if source is not None else []


def from_events(events: Any) -> AgentResult:
    """Build ``{"output", "trace"}`` from LlamaIndex agent events or an ``AgentChatResponse``.

    ``events`` is the list of events a workflow agent emitted (``ToolCall``, ``ToolCallResult``,
    ``AgentOutput``; others are ignored), a list of such lists (one per turn), an
    ``AgentChatResponse``, or a list of them (one per turn).
    """
    warn_experimental(
        "llama_index",
        "its event shapes were read from the SDK source and not yet run against a real agent",
    )
    ledger = Ledger()
    seen: set[str] = set()
    replies: dict[int, str] = {}
    rounds = 0

    def call(name: Any, kwargs: Any, id: Any, turn: int) -> None:
        if id is not None:
            if str(id) in seen:
                return
            seen.add(str(id))
        ledger.call(name, kwargs, id, turn)

    for turn, run in enumerate(_runs(events)):
        for item in run:
            if _is_chat_response(item):
                for source in get(item, "sources", default=[]) or []:
                    name = get(source, "tool_name")
                    failed = bool(get(source, "is_error", default=False))
                    text = _output_text(source)
                    ledger.call(name, get(source, "raw_input"), None, turn)
                    ledger.result(
                        None,
                        name,
                        text or jsonable(get(source, "raw_output")),
                        ok=not failed,
                        error=(text or "tool error") if failed else None,
                        turn=turn,
                    )
                if get(item, "response"):
                    replies[turn] = get(item, "response")
            elif has(item, "tool_output"):
                out = get(item, "tool_output")
                failed = bool(get(out, "is_error", default=False))
                text = _output_text(out)
                tool_id = get(item, "tool_id")
                name = get(item, "tool_name") or get(out, "tool_name")
                call(name, get(item, "tool_kwargs"), tool_id, turn)
                ledger.result(
                    tool_id,
                    name,
                    text or jsonable(get(out, "raw_output")),
                    ok=not failed,
                    error=(text or "tool error") if failed else None,
                    turn=turn,
                )
            elif has(item, "tool_kwargs"):
                call(get(item, "tool_name"), get(item, "tool_kwargs"), get(item, "tool_id"), turn)
            elif get(item, "response") is not None and not has(item, "delta"):
                rounds += 1
                selections = get(item, "tool_calls", default=[]) or []
                for selection in selections:
                    call(
                        get(selection, "tool_name"),
                        get(selection, "tool_kwargs"),
                        get(selection, "tool_id"),
                        turn,
                    )
                text = _text(get(item, "response"))
                structured = get(item, "structured_response")
                if not text and structured is not None:
                    text = final_text(structured)
                if text and not selections:
                    replies[turn] = text
    return finish(replies, ledger, rounds, {})


__all__ = ["from_events"]
