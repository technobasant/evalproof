"""OpenAI Agents SDK: ``from_result(result)`` turns a ``RunResult`` into an evalproof trace.

SDK fields read (names confirmed in the SDK source, not from memory), all openai-agents==0.24.0
under agents/:

* result.py:375 ``RunResultBase.new_items``, :380 ``raw_responses`` (one entry per model
  response), :383 ``final_output``, :398 ``context_wrapper``.
* run_context.py:187 ``RunContextWrapper.usage``; usage.py:197 ``Usage.requests``, :200
  ``input_tokens``, :203 ``input_tokens_details`` (its ``cached_tokens``: usage.py:85-87, :178),
  :207 ``output_tokens``.
* items.py:176 ``MessageOutputItem.type == "message_output_item"`` and its ``raw_item.content``
  (``ResponseOutputText.text``: items.py:836-842); :389 ``ToolCallItem.type ==
  "tool_call_item"``, :399-413 ``tool_name`` and ``call_id`` (from ``raw_item``, whose
  ``arguments`` is a JSON string: items.py:645-666); :442 ``ToolCallOutputItem.type ==
  "tool_call_output_item"``, :437 ``output`` (whatever the tool returned), :448-454 ``call_id``.

The SDK has no field that says a tool failed. A function tool that raises is turned into a
plain string by its ``failure_error_function`` (tool.py:697-711) and arrives as an ordinary
``output``. By default that string is the fixed ``DEFAULT_TOOL_ERROR`` (tool.py:1997-2002), so an
output exactly equal to it marks the call failed. A tool with a custom failure text needs
``error_if``, a function of the output that returns True for a failure; it replaces the default
test (``error_if=lambda output: False`` turns failure detection off). The SDK reports no
cost, and its usage is not split by model, so the usage entry has no model and no cost.
A turn is a ``RunResult``: pass a list of results (one per user turn) for a multi-turn case.
Results that share one ``Usage`` object (the same ``RunContextWrapper`` passed to every run)
have their tokens counted once.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from evalproof.adapters._common import AgentResult, Ledger, count, get, jsonable, text_of
from evalproof.models import ModelUsage, Trace

DEFAULT_TOOL_ERROR = "An error occurred while running the tool. Please try again."
"""What the SDK's ``default_tool_error_function`` returns for a tool that raised (tool.py:2002)."""


def _is_default_failure(output: Any) -> bool:
    return isinstance(output, str) and output == DEFAULT_TOOL_ERROR


def _message_text(item: Any) -> str:
    return text_of(get(get(item, "raw_item"), "content"))


def _final_text(value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    dump = getattr(value, "model_dump_json", None)
    if callable(dump):
        try:
            return str(dump())
        except Exception:  # an object that cannot dump itself is serialised below
            pass
    return json.dumps(jsonable(value), default=str)


def from_result(result: Any, *, error_if: Callable[[Any], bool] | None = None) -> AgentResult:
    """Build ``{"output", "trace"}`` from the result of ``Runner.run`` / ``Runner.run_sync``.

    Tool calls come from the ``tool_call_item`` entries of ``new_items`` and are paired with their
    ``tool_call_output_item`` by ``call_id``, else by order. ``output`` is ``final_output`` (JSON
    when it is not a string), else the text of the last message. Tokens come from
    ``context_wrapper.usage`` and ``rounds`` is the number of ``raw_responses``.
    """
    failed_if = error_if if error_if is not None else _is_default_failure
    results = list(result) if isinstance(result, list | tuple) else [result]
    ledger = Ledger()
    responses: list[str] = []
    usage = ModelUsage(role="agent")
    rounds = 0
    rounds_known = False
    saw_usage = False
    # Usage objects already counted. Runs given the same RunContextWrapper, or resumed from a
    # RunState, share one cumulative Usage (run_internal/agent_runner_helpers.py:377-383 keeps a
    # wrapper it is given), so each object is counted once, at its final value.
    counted: set[int] = set()
    for turn, one in enumerate(results):
        last_message = ""
        for item in get(one, "new_items", default=[]):
            kind = get(item, "type")
            raw = get(item, "raw_item")
            if kind == "tool_call_item":
                ledger.call(
                    get(item, "tool_name") or get(raw, "name"),
                    get(raw, "arguments"),
                    get(item, "call_id") or get(raw, "call_id") or get(raw, "id"),
                    turn,
                )
            elif kind == "tool_call_output_item":
                output = get(item, "output")
                if output is None:
                    output = get(raw, "output")
                failed = bool(failed_if(output))
                ledger.result(
                    get(item, "call_id") or get(raw, "call_id") or get(raw, "id"),
                    None,
                    output,
                    ok=not failed,
                    error=(output if isinstance(output, str) else json.dumps(jsonable(output)))
                    if failed
                    else None,
                    turn=turn,
                )
            elif kind == "message_output_item":
                last_message = _message_text(item) or last_message
        responses.append(_final_text(get(one, "final_output"), last_message))
        raw_responses = get(one, "raw_responses")
        stats = get(get(one, "context_wrapper"), "usage")
        fresh = stats is not None and id(stats) not in counted
        if fresh:
            counted.add(id(stats))
        if isinstance(raw_responses, list | tuple):
            rounds += len(raw_responses)
            rounds_known = True
        elif fresh and count(get(stats, "requests")):
            rounds += count(get(stats, "requests"))
            rounds_known = True
        if fresh:
            saw_usage = True
            usage.tokens_in += count(get(stats, "input_tokens"))
            usage.tokens_out += count(get(stats, "output_tokens"))
            usage.cached_tokens += count(get(get(stats, "input_tokens_details"), "cached_tokens"))
    trace = Trace(
        responses=responses,
        tool_calls=ledger.calls,
        rounds=rounds if rounds_known else None,
        usage=[usage] if saw_usage else [],
    )
    return {"output": responses[-1] if responses else "", "trace": trace}


__all__ = ["DEFAULT_TOOL_ERROR", "from_result"]
