"""CrewAI (experimental): ``from_output(output)`` turns a crew run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real crew.

Nothing here imports ``crewai``. It reads ``CrewOutput`` and ``TaskOutput`` objects and the dicts
``model_dump()`` produces the same way.

SDK fields read (names confirmed in the SDK source, not from memory):

* crewai==1.15.41 crewai/crews/crew_output.py:14 ``CrewOutput`` (``raw``, ``tasks_output``,
  ``token_usage``); crewai/tasks/task_output.py:15 ``TaskOutput`` (``raw``, ``agent``,
  ``messages``: the ``LLMMessage`` list of the agent that ran the task, ``tool_failures``).
* crewai/types/usage_metrics.py:32 ``UsageMetrics`` (``prompt_tokens``, ``completion_tokens``,
  ``cached_prompt_tokens``).
* crewai/utilities/types.py:16 ``LLMMessage`` (``role``: ``"user"``, ``"assistant"``,
  ``"system"`` or ``"tool"``; ``content``, ``tool_calls``, ``tool_call_id``, ``name``).
* crewai/utilities/agent_utils.py:1440-1486 the assistant message of a native tool call: its
  ``tool_calls`` are ``{"id", "type": "function", "function": {"name", "arguments"}}`` with
  ``arguments`` a JSON string; :1591 and :1766 the tool message: ``{"role": "tool",
  "tool_call_id", "name", "content"}``.
* crewai/tools/tool_failure.py:74 ``ToolFailure.message`` and :114 ``ToolFailureRecord``
  (``tool_name``, ``failure``, ``tool_args``).

A crew run is one turn, and a list of outputs (one ``kickoff`` per user turn) is one turn each.
The reply of a turn is the ``raw`` of its output (the last task's, when the crew output has
none), else the text of the last assistant message that calls no tool. A tool message answers
the call with its ``tool_call_id``; it is a failed call when the task holds a
``ToolFailureRecord`` for that tool that no earlier tool message has taken. ``rounds`` counts the
assistant messages.
Left out: the text-protocol (ReAct) tool path, whose calls are only text in the messages, not
structured data; the model name (``UsageMetrics`` is a crew total and names none); and cost.
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


def _list(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


def _turns(output: Any) -> list[Any]:
    if isinstance(output, list | tuple):
        return list(output)
    return [output] if output is not None else []


def from_output(output: Any) -> AgentResult:
    """Build ``{"output", "trace"}`` from a CrewAI run (objects or dicts).

    ``output`` is the ``CrewOutput`` of ``crew.kickoff(...)``, a ``TaskOutput``, or a list of
    them (one per turn). Each entry of an assistant message's ``tool_calls`` is a call; a message
    with role ``"tool"`` answers the call with its ``tool_call_id``.
    """
    warn_experimental(
        "crewai",
        "its output shapes were read from the SDK source and not yet run against a real crew",
    )
    ledger = Ledger()
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for turn, item in enumerate(_turns(output)):
        tasks = _list(get(item, "tasks_output")) or [item]
        last_text = ""
        for task in tasks:
            failures = [
                failure
                for failure in _list(get(task, "tool_failures"))
                if get(failure, "tool_name") is not None
            ]
            for message in _list(get(task, "messages")):
                role = str(get(message, "role", default="")).lower()
                if role == "assistant":
                    rounds += 1
                    calls = _list(get(message, "tool_calls"))
                    for call in calls:
                        function = get(call, "function")
                        ledger.call(
                            get(function, "name"),
                            get(function, "arguments"),
                            get(call, "id"),
                            turn,
                        )
                    text = text_of(get(message, "content"))
                    if text and not calls:
                        last_text = text
                elif role == "tool":
                    name = get(message, "name")
                    failure = _take(failures, name)
                    content = get(message, "content")
                    ledger.result(
                        get(message, "tool_call_id"),
                        name,
                        content if isinstance(content, str) else text_of(content),
                        ok=failure is None,
                        error=(_why(failure, content)) if failure is not None else None,
                        turn=turn,
                    )
        raw = get(item, "raw") or get(tasks[-1], "raw")
        if raw or last_text:
            replies[turn] = final_text(raw) if raw else last_text
        meter = get(item, "token_usage")
        if meter is not None:
            add_usage(
                usage,
                None,
                get(meter, "prompt_tokens"),
                get(meter, "completion_tokens"),
                get(meter, "cached_prompt_tokens"),
            )
    return finish(replies, ledger, rounds, usage)


def _take(failures: list[Any], name: Any) -> Any:
    """The first unused failure record for the tool ``name``, removed from ``failures``."""
    for index, failure in enumerate(failures):
        if get(failure, "tool_name") == name:
            return failures.pop(index)
    return None


def _why(failure: Any, content: Any) -> str:
    message = get(get(failure, "failure"), "message")
    if isinstance(message, str) and message:
        return message
    return text_of(content) or "tool error"


__all__ = ["from_output"]
