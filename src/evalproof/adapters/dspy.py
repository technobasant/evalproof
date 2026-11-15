"""DSPy (experimental): ``from_prediction(prediction)`` turns a ``dspy.ReAct`` run into a trace.

Experimental: written from the SDK source, with fake objects of the confirmed shapes as the
fixtures, and not yet run against a real DSPy program.

Nothing here imports ``dspy``. It reads ``Prediction`` objects and plain dicts the same way. Two
shapes are read, because DSPy changed the one ``ReAct`` returns:

* dspy==3.5.0 (current): ``Prediction(**outputs, history=..., termination_reason=...)``
  (dspy/predict/react.py:176, :217, :329), where ``history`` is a ``dspy.History``
  (dspy/adapters/types/history.py:15) whose ``messages`` are events. An event with tool calls
  holds ``tool_calls`` (react.py:279-293), a ``ToolCalls`` (dspy/adapters/types/tool.py:378) with
  its ``tool_calls``, each a ``ToolCall`` (:379, ``id``, ``name``, ``args``) and its
  ``tool_call_results``, a ``ToolCallResults`` (:557) with ``tool_call_results``, each a
  ``ToolCallResult`` (:558, ``call_id``, ``name``, ``value``, ``is_error``). ``"submit"`` is the
  reserved final-output tool (react.py:75-77) and is not a call of yours, so it is left out.
* dspy==3.4.1 and earlier: ``Prediction(trajectory=..., **outputs)``
  (dspy/predict/react.py:121), a dict with ``thought_N``, ``tool_name_N``, ``tool_args_N`` and
  ``observation_N`` for each step N (react.py:108-115). ``"finish"`` is the reserved last step
  (react.py:62-64) and is left out. A failed tool has the observation
  ``"Execution error in <tool>: ..."`` (react.py:115).
* dspy/primitives/prediction.py:27 ``get_lm_usage()``: ``{lm name: {"prompt_tokens",
  "completion_tokens", ...}}`` (dspy/utils/usage_tracker.py:10-11), set when usage tracking is on;
  prediction.py:37-42 ``_store``, the dict of the output fields.

One prediction is one turn (a list of predictions is one turn each). The reply is the output
field when the prediction has one, else the JSON of all its output fields (``history``,
``termination_reason`` and ``trajectory`` are not output fields); ``output_field`` picks one by
name. ``rounds`` counts the history events (3.5) or the steps (3.4). Left out: cached-token counts
(provider-specific keys), cost, and the model name of a program that reports no usage.
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
    warn_experimental,
)
from evalproof.models import ModelUsage

_RESERVED = frozenset({"history", "termination_reason", "trajectory"})
_FINAL_TOOLS = frozenset({"submit", "finish"})
_ERROR_PREFIX = "Execution error in "


def _list(value: Any) -> list[Any]:
    if value is None or isinstance(value, str | Mapping):
        return []
    return list(value) if isinstance(value, Iterable) else []


def _fields(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, Mapping):
        return dict(prediction)
    store = getattr(prediction, "_store", None)
    return dict(store) if isinstance(store, Mapping) else {}


def _reply(prediction: Any, output_field: str | None) -> str:
    fields = {k: v for k, v in _fields(prediction).items() if k not in _RESERVED}
    if output_field is not None:
        return final_text(fields.get(output_field))
    if len(fields) == 1:
        return final_text(next(iter(fields.values())))
    return final_text(fields) if fields else ""


def _read_history(history: Any, ledger: Ledger, turn: int) -> int:
    """Add the calls of a ``dspy.History`` (or its dict) to the ledger; the number of events."""
    events = _list(get(history, "messages"))
    for event in events:
        calls = get(event, "tool_calls")
        results = _list(get(get(calls, "tool_call_results"), "tool_call_results"))
        for call in _list(get(calls, "tool_calls")):
            name = get(call, "name")
            if name in _FINAL_TOOLS:
                continue
            ledger.call(name, get(call, "args"), get(call, "id"), turn)
        for item in results:
            if get(item, "name") in _FINAL_TOOLS:
                continue
            failed = bool(get(item, "is_error", default=False))
            ledger.result(
                get(item, "call_id"),
                get(item, "name"),
                get(item, "value"),
                ok=not failed,
                error=(final_text(get(item, "value")) or "tool error") if failed else None,
                turn=turn,
            )
    return len(events)


def _read_trajectory(trajectory: Any, ledger: Ledger, turn: int) -> int:
    """Add the calls of a ``trajectory`` dict to the ledger; the number of steps."""
    if not isinstance(trajectory, Mapping):
        return 0
    steps = sorted(
        int(key.removeprefix("tool_name_"))
        for key in trajectory
        if isinstance(key, str)
        and key.startswith("tool_name_")
        and key.removeprefix("tool_name_").isdigit()
    )
    for step in steps:
        name = trajectory.get(f"tool_name_{step}")
        if name in _FINAL_TOOLS:
            continue
        ledger.call(name, trajectory.get(f"tool_args_{step}"), None, turn)
        observation = trajectory.get(f"observation_{step}")
        if f"observation_{step}" not in trajectory:
            continue
        failed = isinstance(observation, str) and observation.startswith(f"{_ERROR_PREFIX}{name}:")
        ledger.result(
            None,
            name,
            jsonable(observation),
            ok=not failed,
            error=observation if failed else None,
            turn=turn,
        )
    return len(steps)


def from_prediction(prediction: Any, *, output_field: str | None = None) -> AgentResult:
    """Build ``{"output", "trace"}`` from a DSPy prediction (object or dict), or a list of them.

    Tool calls come from ``history`` (DSPy 3.5) or ``trajectory`` (DSPy 3.4 and earlier). A
    prediction with neither is a reply with no tool calls.
    """
    warn_experimental(
        "dspy",
        "its prediction shapes were read from the SDK source and not yet run against a real "
        "program",
    )
    ledger = Ledger()
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    items = list(prediction) if isinstance(prediction, list | tuple) else [prediction]
    for turn, item in enumerate(i for i in items if i is not None):
        history = get(item, "history")
        if history is not None:
            rounds += _read_history(history, ledger, turn)
        else:
            rounds += _read_trajectory(get(item, "trajectory"), ledger, turn)
        text = _reply(item, output_field)
        if text:
            replies[turn] = text
        meter = getattr(item, "get_lm_usage", None)
        stats = meter() if callable(meter) else get(item, "lm_usage")
        if isinstance(stats, Mapping):
            for model, entry in stats.items():
                add_usage(
                    usage, model, get(entry, "prompt_tokens"), get(entry, "completion_tokens")
                )
    return finish(replies, ledger, rounds, usage)


__all__ = ["from_prediction"]
