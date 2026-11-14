"""``TraceRecorder``: build a trace by hand in plain Python, and ``mcp_call`` to record an MCP
tool call from its ``CallToolResult``.

SDK fields read by ``mcp_call`` (names confirmed in the SDK source, not from memory):

* mcp==2.3.0 (mcp-types==2.3.0) mcp_types/_types.py:1472 ``CallToolResult.content``, :1474
  ``structured_content``, :1480 ``is_error``; :1153-1154 ``TextContent.type`` and ``.text``.
* mcp==1.30.0 mcp/types.py:1366 ``CallToolResult.content``, :1367 ``structuredContent``, :1369
  ``isError``; :1029-1030 ``TextContent.type`` and ``.text``.

mcp 2 spells the attributes in snake case and mcp 1 in camel case; the wire format (a plain
dict) is camel case. ``mcp_call`` reads all three spellings.
"""

from __future__ import annotations

from typing import Any

from evalproof.adapters._common import AgentResult, get, jsonable, text_of
from evalproof.models import ModelUsage, ToolCall, Trace


class TraceRecorder:
    """Record what a plain-Python agent did, then ``result()`` a dict a python system returns.

        rec = TraceRecorder()
        rec.round()
        rec.tool("get_weather", {"city": "Paris"}, result={"temp_c": 20})
        rec.usage(model="m", tokens_in=120, tokens_out=30)
        rec.reply("It is 20 degrees in Paris.")
        return rec.result()

    A tool call belongs to the user turn that is under way: the number of replies recorded so far,
    unless ``turn`` says otherwise. Pass ``ok=False`` with an ``error`` for a call that failed.
    """

    def __init__(self) -> None:
        self._calls: list[ToolCall] = []
        self._usage: list[ModelUsage] = []
        self._replies: list[str] = []
        self._rounds = 0

    def tool(
        self,
        name: str,
        args: dict[str, Any] | None = None,
        result: Any = None,
        ok: bool = True,
        error: str | None = None,
        turn: int | None = None,
    ) -> ToolCall:
        call = ToolCall(
            name=name,
            args=dict(args or {}),
            result=jsonable(result),
            ok=ok,
            error=error,
            turn=len(self._replies) if turn is None else turn,
        )
        self._calls.append(call)
        return call

    def usage(
        self,
        model: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float | None = None,
        role: str = "agent",
    ) -> ModelUsage:
        entry = ModelUsage(
            role=role, model=model, tokens_in=tokens_in, tokens_out=tokens_out, cost_usd=cost_usd
        )
        self._usage.append(entry)
        return entry

    def round(self) -> None:
        """Count one model call (a generation round)."""
        self._rounds += 1

    def reply(self, text: str) -> None:
        """Record the assistant's reply to the current user turn."""
        self._replies.append(text)

    def result(self, output: str | None = None) -> AgentResult:
        """The dict a python system returns. ``output`` defaults to the last reply."""
        if output is None:
            output = self._replies[-1] if self._replies else ""
        trace = Trace(
            responses=list(self._replies),
            tool_calls=list(self._calls),
            rounds=self._rounds or None,
            usage=list(self._usage),
        )
        return {"output": output, "trace": trace}


def mcp_call(
    rec: TraceRecorder, name: str, args: dict[str, Any] | None, call_tool_result: Any
) -> ToolCall:
    """Record an MCP tool call on ``rec`` from the ``CallToolResult`` the server returned.

    The result is the structured content when the server sent any, else the text of the content
    blocks (the raw blocks when none is text). ``isError`` makes the call fail, with the text
    as the error. Works on the SDK object or the same thing as a dict.
    """
    content = get(call_tool_result, "content")
    structured = get(call_tool_result, "structuredContent", "structured_content")
    is_error = bool(get(call_tool_result, "isError", "is_error", default=False))
    text = text_of(content)
    if structured is not None:
        result: Any = structured
    elif text or content is None:
        result = text
    else:
        result = content
    return rec.tool(
        name,
        args,
        result=result,
        ok=not is_error,
        error=(text or "tool error") if is_error else None,
    )
