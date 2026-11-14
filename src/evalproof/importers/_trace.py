"""Building a ``Trace`` from what an importer found in a file.

The adapters already turn SDK objects into tool calls (``evalproof.adapters._common``: ``args_of``,
``jsonable``, ``Ledger``); this module adds only what a recorded file needs on top:

* a JSON-encoded string argument or result is decoded when it parses (an object or an array;
  a bare number or word stays the text it was) and kept as the string otherwise;
* a source's error field makes the call ``ok=False`` and gives its ``error`` text.

Nothing here reads a file or runs anything from one.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from evalproof.adapters._common import args_of, jsonable
from evalproof.models import ModelUsage, ToolCall, Trace


def decode_json(value: Any) -> Any:
    """``value`` with a JSON object or array held in a string decoded; anything else as it is."""
    if isinstance(value, str):
        text = value.strip()
        if text[:1] in ("{", "["):
            try:
                return json.loads(text)
            except ValueError:
                return value
    return value


def error_text(error: Any) -> str | None:
    """The message of a source's error field: an object's ``message`` (or ``type``), or a string."""
    if not error:
        return None
    if isinstance(error, dict):
        text = error.get("message") or error.get("type")
        return str(text) if text else "error"
    return str(error)


def tool_call(
    name: Any,
    args: Any = None,
    result: Any = None,
    *,
    id: Any = None,
    turn: int = 0,
    error: Any = None,
    failed: bool = False,
    latency_ms: float | None = None,
) -> ToolCall:
    """One recorded tool call. ``error`` (the source's error field) or ``failed`` makes it
    ``ok=False``; ``args`` may be a dict or a JSON object held in a string."""
    message = error_text(error)
    ok = message is None and not failed
    return ToolCall(
        name=str(name) if name is not None else "unknown",
        args=args_of(decode_json(args)),
        result=jsonable(decode_json(result)),
        ok=ok,
        error=None if ok else message,
        id=str(id) if id is not None else None,
        turn=turn,
        latency_ms=latency_ms,
    )


def add_usage(
    usage: dict[str | None, ModelUsage], model: str | None, tokens_in: int, tokens_out: int
) -> None:
    """Add one model call's tokens to the running total of its model (role ``agent``)."""
    entry = usage.setdefault(model, ModelUsage(role="agent", model=model))
    entry.tokens_in += tokens_in
    entry.tokens_out += tokens_out


def build_trace(
    calls: Iterable[ToolCall],
    *,
    responses: Iterable[str] = (),
    rounds: int | None = None,
    usage: Iterable[ModelUsage] = (),
    meta: dict[str, Any] | None = None,
) -> Trace:
    return Trace(
        responses=list(responses),
        tool_calls=list(calls),
        rounds=rounds or None,
        usage=list(usage),
        meta=dict(meta or {}),
    )
