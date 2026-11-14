"""Helpers shared by the adapters: duck-typed reads, JSON-safe values, call/result pairing.

Nothing here imports an agent SDK. An adapter reads SDK objects and plain dicts the same way
(attribute or key), and a missing field is a default, never an error.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Mapping
from typing import Any, TypedDict

from evalproof.models import ToolCall, Trace


class AgentResult(TypedDict, total=False):
    """What a python system returns for an agent: ``output`` is required in practice, the rest
    is optional. Tokens and cost default to the sums over ``trace.usage``."""

    output: str
    trace: Trace | dict[str, Any]
    tokens_in: int
    tokens_out: int
    cost_usd: float


def get(obj: Any, *names: str, default: Any = None) -> Any:
    """The first of ``names`` that is set (not None) on ``obj``, as a key or an attribute."""
    if obj is None:
        return default
    for name in names:
        value = obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)
        if value is not None:
            return value
    return default


def has(obj: Any, name: str) -> bool:
    """True when ``obj`` has ``name`` as a key or an attribute, even when its value is None."""
    if isinstance(obj, Mapping):
        return name in obj
    return obj is not None and hasattr(obj, name)


def count(value: Any) -> int:
    """A token count as an int; anything that is not a number is 0."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0
    return int(value)


def number(value: Any) -> float | None:
    """A cost as a float, or None when the SDK did not report one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def text_of(content: Any) -> str:
    """The text of a message content: a string, or the text blocks of a list of blocks."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list | tuple):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
                continue
            text = get(block, "text")
            if isinstance(text, str) and get(block, "type", default="text") in (
                "text",
                "output_text",
            ):
                parts.append(text)
        return "".join(parts)
    return ""


def jsonable(value: Any) -> Any:
    """``value`` as something a run file can hold: JSON types, or a string for anything else."""
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [jsonable(v) for v in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return jsonable(dump(mode="json"))
        except Exception:  # an SDK object that cannot dump itself falls through to a string
            pass
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return jsonable(dataclasses.asdict(value))
    return str(value)


def args_of(value: Any) -> dict[str, Any]:
    """Tool arguments as a dict: a dict, a JSON object in a string, or ``{"input": value}``."""
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, str):
        try:
            parsed = json.loads(value) if value.strip() else {}
        except ValueError:
            return {"input": value}
        if isinstance(parsed, dict):
            return {str(k): jsonable(v) for k, v in parsed.items()}
        return {"input": parsed}
    return {"input": jsonable(value)}


class Ledger:
    """Tool calls in the order they were made, with each result matched to its call.

    A result carrying an id is matched to the unanswered call with that id. A result with no id,
    or whose id matches no call that has one, goes to the first unanswered call with no id and
    the same name (any name, when the result has none), so ids and order are both honoured. A
    result that matches nothing is kept as a call of its own, so it is not lost.
    """

    def __init__(self) -> None:
        self.calls: list[ToolCall] = []
        self._answered: set[int] = set()

    def call(self, name: Any, args: Any, id: Any, turn: int) -> ToolCall:
        entry = ToolCall(
            name=str(name) if name is not None else "unknown",
            args=args_of(args),
            id=str(id) if id is not None else None,
            turn=turn,
        )
        self.calls.append(entry)
        return entry

    def _find(self, id: str | None, name: str | None) -> int | None:
        if id is not None:
            for i, call in enumerate(self.calls):
                if i not in self._answered and call.id == id:
                    return i
        for i, call in enumerate(self.calls):
            if i in self._answered or call.id is not None:
                continue
            if name is None or call.name == name:
                return i
        return None

    def result(
        self,
        id: Any,
        name: Any,
        result: Any,
        *,
        ok: bool = True,
        error: str | None = None,
        turn: int = 0,
    ) -> ToolCall:
        key = str(id) if id is not None else None
        label = str(name) if name is not None else None
        index = self._find(key, label)
        if index is None:
            entry = ToolCall(name=label or "unknown", id=key, turn=turn)
            self.calls.append(entry)
            index = len(self.calls) - 1
        self._answered.add(index)
        entry = self.calls[index]
        entry.result = jsonable(result)
        entry.ok = ok
        entry.error = error if not ok else None
        return entry
