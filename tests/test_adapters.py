"""The SDK adapters and ``TraceRecorder``: what each reads, how calls pair with results, and the
path from an adapter's dict through ``PythonSystem`` to a ``checks`` score.

The fakes below are shaped exactly like the fields the adapters cite from each SDK's source (see
the comment block at the top of every adapter module); no SDK is imported.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evalproof import adapters
from evalproof.adapters import AgentResult, TraceRecorder, adk, claude_agent_sdk, langchain
from evalproof.adapters import mcp_call as mcp_call_fn
from evalproof.adapters import openai_agents as oa
from evalproof.config import Pricing
from evalproof.models import Case, ToolCall, Trace
from evalproof.scorers import ChecksScorer
from evalproof.systems import PythonSystem, _coerce

NS = SimpleNamespace


def trace_of(result: AgentResult) -> Trace:
    trace = result["trace"]
    assert isinstance(trace, Trace)
    return trace


def calls_of(result: AgentResult) -> list[ToolCall]:
    return trace_of(result).tool_calls


def accepted(result: AgentResult) -> None:
    """The dict is what ``systems._coerce`` accepts, and the trace survives a JSON round trip."""
    coerced = _coerce(result)
    assert coerced.trace is not None
    assert Trace.model_validate_json(coerced.trace.model_dump_json()) == coerced.trace


# --- TraceRecorder and mcp_call ---------------------------------------------------------------


def test_recorder_builds_a_result() -> None:
    rec = TraceRecorder()
    rec.round()
    rec.tool("get_weather", {"city": "Paris"}, result={"temp_c": 20})
    rec.round()
    rec.usage(model="m", tokens_in=100, tokens_out=20, cost_usd=0.01)
    rec.usage(tokens_in=5, role="judge")
    rec.reply("It is 20 degrees in Paris.")
    result = rec.result()
    assert result["output"] == "It is 20 degrees in Paris."
    trace = trace_of(result)
    assert trace.responses == ["It is 20 degrees in Paris."]
    assert trace.rounds == 2
    assert [(u.role, u.model, u.tokens_in, u.cost_usd) for u in trace.usage] == [
        ("agent", "m", 100, 0.01),
        ("judge", None, 5, None),
    ]
    assert calls_of(result)[0].args == {"city": "Paris"}
    accepted(result)


def test_recorder_turns_follow_the_replies() -> None:
    rec = TraceRecorder()
    rec.tool("a")
    rec.reply("first")
    rec.tool("b")
    rec.tool("c", turn=0)
    rec.reply("second")
    result = rec.result()
    assert [(c.name, c.turn) for c in calls_of(result)] == [("a", 0), ("b", 1), ("c", 0)]
    assert result["output"] == "second" and trace_of(result).responses == ["first", "second"]


def test_recorder_empty_and_explicit_output() -> None:
    empty = TraceRecorder().result()
    assert empty["output"] == "" and trace_of(empty).rounds is None
    rec = TraceRecorder()
    rec.reply("said")
    assert rec.result(output="override")["output"] == "override"


def test_recorder_failed_call_and_non_json_result() -> None:
    rec = TraceRecorder()
    rec.tool("book", {"n": 1}, ok=False, error="sold out")
    rec.tool("fetch", result=NS(a=1))  # not JSON: kept as a string, so a run file can hold it
    calls = calls_of(rec.result())
    assert (calls[0].ok, calls[0].error) == (False, "sold out")
    assert isinstance(calls[1].result, str)
    accepted(rec.result())


@dataclass
class V2Text:  # mcp 2: mcp_types/_types.py TextContent
    text: str
    type: str = "text"


@dataclass
class V2Result:  # mcp 2: CallToolResult.content / structured_content / is_error
    content: list[Any]
    structured_content: Any = None
    is_error: bool = False


@dataclass
class V1Result:  # mcp 1: CallToolResult.content / structuredContent / isError
    content: list[Any]
    structuredContent: Any = None
    isError: bool = False


def test_mcp_call_reads_every_spelling() -> None:
    rec = TraceRecorder()
    mcp_call_fn(rec, "search", {"q": "x"}, V2Result([V2Text("hit one"), V2Text(" and two")]))
    mcp_call_fn(rec, "get", None, V1Result([V2Text("ignored")], structuredContent={"id": 7}))
    mcp_call_fn(rec, "wire", {}, {"content": [{"type": "text", "text": "ok"}], "isError": False})
    mcp_call_fn(rec, "boom", {}, V2Result([V2Text("quota exceeded")], is_error=True))
    mcp_call_fn(rec, "wire_err", {}, {"content": [], "isError": True})
    mcp_call_fn(rec, "img", {}, {"content": [{"type": "image", "data": "AA"}]})
    mcp_call_fn(rec, "bare", {}, {})
    calls = calls_of(rec.result())
    assert [c.result for c in calls[:3]] == ["hit one and two", {"id": 7}, "ok"]
    assert calls[1].args == {}
    assert (calls[3].ok, calls[3].error) == (False, "quota exceeded")
    assert (calls[4].ok, calls[4].error) == (False, "tool error")
    assert calls[5].result == [{"type": "image", "data": "AA"}]
    assert calls[6].ok and calls[6].result == ""


# --- Google ADK -------------------------------------------------------------------------------


class FakeEvent:
    """google-adk Event: author, content.parts, partial, usage_metadata, model_version."""

    def __init__(
        self,
        author: str,
        parts: list[Any] | None = None,
        *,
        role: str | None = None,
        partial: bool | None = None,
        usage: Any = None,
        model: str | None = None,
    ) -> None:
        self.author = author
        self.content = NS(role=role, parts=parts) if parts is not None else None
        self.partial = partial
        self.usage_metadata = usage
        self.model_version = model

    def is_final_response(self) -> bool:  # google/adk/events/event.py:305, simplified
        calls = any(getattr(p, "function_call", None) for p in (self.content.parts or []))
        resps = any(getattr(p, "function_response", None) for p in (self.content.parts or []))
        return not calls and not resps and not self.partial


def call_part(name: str, args: dict[str, Any] | None = None, id: str | None = None) -> Any:
    return NS(function_call=NS(id=id, name=name, args=args), function_response=None, text=None)


def resp_part(name: str, response: dict[str, Any], id: str | None = None) -> Any:
    return NS(function_call=None, function_response=NS(id=id, name=name, response=response))


def text_part(text: str, thought: bool = False) -> Any:
    return NS(text=text, thought=thought, function_call=None, function_response=None)


def usage(prompt: int, out: int, cached: int | None = None) -> Any:
    return NS(
        prompt_token_count=prompt, candidates_token_count=out, cached_content_token_count=cached
    )


def test_adk_pairs_calls_by_id_even_out_of_order() -> None:
    events = [
        FakeEvent(
            "agent", [call_part("search", {"q": "a"}, "c1")], role="model", usage=usage(10, 2)
        ),
        FakeEvent(
            "agent", [call_part("search", {"q": "b"}, "c2")], role="model", usage=usage(12, 2)
        ),
        FakeEvent("agent", [resp_part("search", {"hits": 2}, "c2")], role="user"),
        FakeEvent("agent", [resp_part("search", {"hits": 1}, "c1")], role="user"),
        FakeEvent(
            "agent",
            [text_part("let me think", thought=True), text_part("Found it.")],
            role="model",
            usage=usage(30, 5, cached=20),
            model="gemini-x",
        ),
    ]
    result = adk.from_events(events)
    calls = calls_of(result)
    assert [(c.id, c.args, c.result) for c in calls] == [
        ("c1", {"q": "a"}, {"hits": 1}),
        ("c2", {"q": "b"}, {"hits": 2}),
    ]
    assert result["output"] == "Found it."  # the thought part is not the reply
    trace = trace_of(result)
    assert trace.responses == ["Found it."] and trace.rounds == 3
    by_model = {u.model: u for u in trace.usage}
    assert (by_model[None].tokens_in, by_model[None].tokens_out) == (22, 4)
    assert (by_model["gemini-x"].tokens_in, by_model["gemini-x"].cached_tokens) == (30, 20)
    assert all(u.role == "agent" and u.cost_usd is None for u in trace.usage)
    accepted(result)


def test_adk_pairs_by_order_and_name_without_ids() -> None:
    events = [
        FakeEvent(
            "a", [call_part("x", {"n": 1}), call_part("y", {"n": 2}), call_part("x", {"n": 3})]
        ),
        FakeEvent("a", [resp_part("y", {"r": "y"}), resp_part("x", {"r": "x1"})]),
        FakeEvent("a", [resp_part("x", {"r": "x2"})]),
    ]
    calls = calls_of(adk.from_events(events))
    assert [(c.name, c.args["n"], c.result) for c in calls] == [
        ("x", 1, {"r": "x1"}),
        ("y", 2, {"r": "y"}),
        ("x", 3, {"r": "x2"}),
    ]


def test_adk_error_result_orphan_and_unanswered_call() -> None:
    events = [
        FakeEvent("a", [call_part("book", {"n": 1}, "c1"), call_part("never", None, "c2")]),
        FakeEvent("a", [resp_part("book", {"error": "sold out"}, "c1")]),
        FakeEvent("a", [resp_part("ghost", {"result": 1}, "zzz")]),
    ]
    calls = calls_of(adk.from_events(events))
    assert (calls[0].ok, calls[0].error, calls[0].result) == (
        False,
        "sold out",
        {"error": "sold out"},
    )
    assert calls[1].name == "never" and calls[1].result is None and calls[1].ok
    assert calls[2].name == "ghost" and calls[2].id == "zzz" and calls[2].result == {"result": 1}


def test_adk_partial_events_are_not_counted_twice() -> None:
    events = [
        FakeEvent("a", [text_part("Hel")], partial=True, usage=usage(99, 99)),
        FakeEvent("a", [text_part("Hello")], role="model", usage=usage(10, 3)),
    ]
    result = adk.from_events(events)
    trace = trace_of(result)
    assert result["output"] == "Hello" and trace.rounds == 1
    assert (trace.usage[0].tokens_in, trace.usage[0].tokens_out) == (10, 3)


def test_adk_user_events_mark_turns() -> None:
    events = [
        FakeEvent("user", [text_part("q1")], role="user"),
        FakeEvent("a", [call_part("t1", {}, "c1")], role="model"),
        FakeEvent("a", [resp_part("t1", {"ok": 1}, "c1")], role="user"),
        FakeEvent("a", [text_part("a1")], role="model"),
        FakeEvent("user", [text_part("q2")], role="user"),
        FakeEvent("a", [call_part("t2", {}, "c2")], role="model"),
        FakeEvent("a", [resp_part("t2", {"ok": 2}, "c2")], role="user"),
        FakeEvent("a", [text_part("a2")], role="model"),
    ]
    result = adk.from_events(events)
    assert [(c.name, c.turn) for c in calls_of(result)] == [("t1", 0), ("t2", 1)]
    assert trace_of(result).responses == ["a1", "a2"] and result["output"] == "a2"


def test_adk_one_list_of_events_per_run_marks_turns() -> None:
    # run_async yields no user event by default, so each run's events come as their own list.
    first = [
        FakeEvent("a", [call_part("t1", {}, "c1")], role="model", usage=usage(10, 1)),
        FakeEvent("a", [resp_part("t1", {"ok": 1}, "c1")], role="user"),
        FakeEvent("a", [text_part("a1")], role="model", usage=usage(20, 2)),
    ]
    second = [
        FakeEvent("a", [call_part("t2", {}, "c2")], role="model", usage=usage(30, 3)),
        FakeEvent("a", [resp_part("t2", {"ok": 2}, "c2")], role="user"),
        FakeEvent("a", [text_part("a2")], role="model", usage=usage(40, 4)),
    ]
    result = adk.from_events([first, second])
    assert [(c.name, c.turn) for c in calls_of(result)] == [("t1", 0), ("t2", 1)]
    trace = trace_of(result)
    assert trace.responses == ["a1", "a2"] and result["output"] == "a2" and trace.rounds == 4
    assert (trace.usage[0].tokens_in, trace.usage[0].tokens_out) == (100, 10)
    # A run that yielded the user event too (yield_user_message=True) is not a turn ahead.
    with_user = [FakeEvent("user", [text_part("q2")], role="user"), *second]
    assert trace_of(adk.from_events([first, with_user])).responses == ["a1", "a2"]
    # Joined into one list with no user event, both runs read as one turn: the reason for runs.
    assert trace_of(adk.from_events(first + second)).responses == ["a2"]


def test_adk_user_authored_response_is_not_a_turn_boundary() -> None:
    events = [
        FakeEvent("a", [call_part("slow", {}, "c1")], role="model"),
        FakeEvent("user", [resp_part("slow", {"done": True}, "c1")], role="user"),
        FakeEvent("a", [text_part("done")], role="model"),
    ]
    result = adk.from_events(events)
    assert calls_of(result)[0].result == {"done": True} and calls_of(result)[0].turn == 0


def test_adk_plain_dicts_camel_case_and_missing_attributes() -> None:
    events: list[Any] = [
        {
            "author": "a",
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"id": "c1", "name": "t", "args": {"k": 1}}}],
            },
            "usageMetadata": {
                "promptTokenCount": 7,
                "candidatesTokenCount": 3,
                "cachedContentTokenCount": 2,
            },
            "modelVersion": "m",
        },
        {
            "author": "a",
            "content": {"parts": [{"functionResponse": {"id": "c1", "response": {"v": 1}}}]},
        },
        {"author": "a", "content": {"role": "model", "parts": [{"text": "done"}]}},
        {},
        FakeEvent("a"),  # no content at all
        NS(),
    ]
    result = adk.from_events(events)
    assert result["output"] == "done"
    assert calls_of(result)[0].result == {"v": 1}
    usage_entry = trace_of(result).usage[0]
    assert (usage_entry.model, usage_entry.tokens_in, usage_entry.cached_tokens) == ("m", 7, 2)
    empty = adk.from_events([])
    assert empty["output"] == "" and trace_of(empty).rounds is None
    accepted(result)


# --- LangChain --------------------------------------------------------------------------------


def ai(content: Any = "", calls: list[dict[str, Any]] | None = None, **kw: Any) -> Any:
    return NS(type="ai", content=content, tool_calls=calls or [], **kw)


def tool_msg(id: str, content: Any, status: str = "success", name: str | None = None) -> Any:
    return NS(type="tool", tool_call_id=id, content=content, status=status, name=name)


def human(text: str) -> Any:
    return NS(type="human", content=text)


def test_langchain_pairs_by_tool_call_id() -> None:
    messages = [
        human("weather in Paris and Rome?"),
        ai(
            "",
            [
                {"name": "weather", "args": {"city": "Paris"}, "id": "t1"},
                {"name": "weather", "args": {"city": "Rome"}, "id": "t2"},
            ],
            usage_metadata={
                "input_tokens": 50,
                "output_tokens": 8,
                "total_tokens": 58,
                "input_token_details": {"cache_read": 30},
            },
            response_metadata={"model_name": "m1"},
        ),
        tool_msg("t2", "Rome: 24 C", name="weather"),
        tool_msg("t1", "Paris: 20 C", name="weather"),
        ai(
            [
                {"type": "text", "text": "Paris 20, "},
                {"type": "text", "text": "Rome 24."},
                {"type": "image"},
            ],
            usage_metadata={"input_tokens": 90, "output_tokens": 12, "total_tokens": 102},
            response_metadata={"model_name": "m1"},
        ),
    ]
    result = langchain.from_messages(messages)
    calls = calls_of(result)
    assert [(c.id, c.args["city"], c.result) for c in calls] == [
        ("t1", "Paris", "Paris: 20 C"),
        ("t2", "Rome", "Rome: 24 C"),
    ]
    assert result["output"] == "Paris 20, Rome 24."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["Paris 20, Rome 24."]
    assert len(trace.usage) == 1
    entry = trace.usage[0]
    assert (entry.model, entry.tokens_in, entry.tokens_out, entry.cached_tokens) == (
        "m1",
        140,
        20,
        30,
    )
    assert entry.cost_usd is None and "cost_usd" not in result
    accepted(result)


def test_langchain_error_status_and_order_fallback() -> None:
    messages = [
        ai("", [{"name": "a", "args": {}, "id": None}, {"name": "b", "args": {}, "id": None}]),
        tool_msg("x", "bad input", status="error", name="b"),
        tool_msg("y", [{"type": "text", "text": "fine"}], name="a"),
        ai("done"),
    ]
    calls = calls_of(langchain.from_messages(messages))
    assert (calls[1].name, calls[1].ok, calls[1].error) == ("b", False, "bad input")
    assert (calls[0].name, calls[0].ok, calls[0].result) == ("a", True, "fine")


def test_langchain_human_messages_mark_turns() -> None:
    messages = [
        human("q1"),
        ai("", [{"name": "t1", "args": {}, "id": "1"}]),
        tool_msg("1", "r1"),
        ai("a1"),
        human("q2"),
        ai("", [{"name": "t2", "args": {}, "id": "2"}]),
        tool_msg("2", "r2"),
        ai("a2"),
    ]
    result = langchain.from_messages(messages)
    assert [(c.name, c.turn) for c in calls_of(result)] == [("t1", 0), ("t2", 1)]
    assert trace_of(result).responses == ["a1", "a2"] and result["output"] == "a2"
    no_boundary = langchain.from_messages(messages[1:4])
    assert {c.turn for c in calls_of(no_boundary)} == {0}


def test_langchain_dicts_state_dict_and_missing_attributes() -> None:
    state = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "type": "ai",
                "content": "",
                "tool_calls": [{"name": "t", "args": {"a": 1}, "id": "i"}],
            },
            {"type": "tool", "tool_call_id": "i", "content": "out", "status": "success"},
            {"role": "assistant", "content": "final"},
            NS(type="system", content="ignored"),
            {},
        ]
    }
    result = langchain.from_messages(state)
    assert result["output"] == "final" and calls_of(result)[0].result == "out"
    bare = langchain.from_messages([NS(type="ai"), {"type": "tool"}])
    assert bare["output"] == "" and trace_of(bare).usage == []
    assert trace_of(bare).tool_calls[0].name == "unknown"


# --- Claude Agent SDK -------------------------------------------------------------------------
# Shaped like claude_agent_sdk/types.py: the dataclasses have no `type`, the SDK tells them apart
# by class.


@dataclass
class TextBlock:
    text: str


@dataclass
class ThinkingBlock:
    thinking: str
    signature: str = ""


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: str | list[dict[str, Any]] | None = None
    is_error: bool | None = None


@dataclass
class AssistantMessage:
    content: list[Any]
    model: str
    parent_tool_use_id: str | None = None
    message_id: str | None = None


@dataclass
class UserMessage:
    content: str | list[Any]
    parent_tool_use_id: str | None = None


@dataclass
class ResultMessage:
    subtype: str = "success"
    duration_ms: int = 1
    duration_api_ms: int = 1
    is_error: bool = False
    num_turns: int = 1
    session_id: str = "s"
    total_cost_usd: float | None = None
    result: str | None = None
    model_usage: dict[str, Any] | None = field(default=None)


def model_usage(i: int, o: int, cached: int = 0, cost: float = 0.0) -> dict[str, Any]:
    return {
        "inputTokens": i,
        "outputTokens": o,
        "cacheReadInputTokens": cached,
        "cacheCreationInputTokens": 0,
        "costUSD": cost,
    }


def test_claude_pairs_tool_use_and_results_and_reports_cost() -> None:
    messages = [
        AssistantMessage(
            [
                ThinkingBlock("hmm"),
                TextBlock("Checking."),
                ToolUseBlock("u1", "Read", {"path": "a"}),
            ],
            "claude-x",
            message_id="m1",
        ),
        AssistantMessage([ToolUseBlock("u2", "Read", {"path": "b"})], "claude-x", message_id="m1"),
        UserMessage([ToolResultBlock("u2", [{"type": "text", "text": "file b"}])]),
        UserMessage([ToolResultBlock("u1", "file a")]),
        AssistantMessage([TextBlock("Both read.")], "claude-x", message_id="m2"),
        ResultMessage(
            total_cost_usd=0.042,
            num_turns=3,
            result="Both files were read.",
            model_usage={"claude-x": model_usage(100, 20, cached=60, cost=0.04)},
        ),
    ]
    result = claude_agent_sdk.from_messages(messages)
    calls = calls_of(result)
    assert [(c.id, c.args["path"], c.result) for c in calls] == [
        ("u1", "a", "file a"),
        ("u2", "b", "file b"),
    ]
    assert result["output"] == "Both files were read."
    assert result.get("cost_usd") == 0.042
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.meta == {"num_turns": 3}
    (entry,) = trace.usage
    assert (entry.role, entry.model, entry.tokens_in, entry.tokens_out) == (
        "agent",
        "claude-x",
        100,
        20,
    )
    assert (entry.cached_tokens, entry.cost_usd) == (60, 0.04)
    accepted(result)


def test_claude_error_result_and_text_fallback() -> None:
    messages = [
        AssistantMessage([ToolUseBlock("u1", "Bash", {"cmd": "x"})], "m"),
        UserMessage([ToolResultBlock("u1", "permission denied", is_error=True)]),
        UserMessage([ToolResultBlock("u9", None, is_error=True)]),
        AssistantMessage([TextBlock("I could not run it.")], "m"),
        ResultMessage(is_error=True, result=None),
    ]
    result = claude_agent_sdk.from_messages(messages)
    calls = calls_of(result)
    assert (calls[0].ok, calls[0].error) == (False, "permission denied")
    assert (calls[1].name, calls[1].ok, calls[1].error) == ("unknown", False, "tool error")
    assert result["output"] == "I could not run it." and "cost_usd" not in result
    assert trace_of(result).meta == {"num_turns": 1, "is_error": True}


def test_claude_result_messages_mark_turns_and_totals_are_running() -> None:
    messages = [
        AssistantMessage([ToolUseBlock("u1", "A", {})], "m"),
        UserMessage([ToolResultBlock("u1", "ra")]),
        AssistantMessage([TextBlock("one")], "m"),
        ResultMessage(result="one", total_cost_usd=0.01, model_usage={"m": model_usage(10, 1)}),
        AssistantMessage([ToolUseBlock("u2", "B", {})], "m"),
        UserMessage([ToolResultBlock("u2", "rb")]),
        AssistantMessage([TextBlock("two")], "m"),
        ResultMessage(result="two", total_cost_usd=0.03, model_usage={"m": model_usage(25, 4)}),
    ]
    result = claude_agent_sdk.from_messages(messages)
    assert [(c.name, c.turn) for c in calls_of(result)] == [("A", 0), ("B", 1)]
    assert trace_of(result).responses == ["one", "two"] and result["output"] == "two"
    assert result.get("cost_usd") == 0.03  # the last running total, not 0.04
    assert trace_of(result).usage[0].tokens_in == 25
    single = claude_agent_sdk.from_messages(messages[:3])
    assert {c.turn for c in calls_of(single)} == {0} and single["output"] == "one"


def test_claude_one_list_per_query_call_adds_the_totals_of_each_session() -> None:
    first = [
        AssistantMessage([ToolUseBlock("u1", "A", {})], "m"),
        UserMessage([ToolResultBlock("u1", "ra")]),
        AssistantMessage([TextBlock("one")], "m"),
        ResultMessage(
            result="one", total_cost_usd=0.01, model_usage={"m": model_usage(10, 1, cost=0.01)}
        ),
    ]
    second = [
        AssistantMessage([ToolUseBlock("u2", "B", {})], "m"),
        UserMessage([ToolResultBlock("u2", "rb")]),
        AssistantMessage([TextBlock("two")], "m"),
        ResultMessage(
            result="two", total_cost_usd=0.02, model_usage={"m": model_usage(25, 4, cost=0.02)}
        ),
    ]
    result = claude_agent_sdk.from_messages([first, second])
    assert [(c.name, c.turn) for c in calls_of(result)] == [("A", 0), ("B", 1)]
    assert trace_of(result).responses == ["one", "two"] and result["output"] == "two"
    assert result.get("cost_usd") == pytest.approx(0.03)  # two sessions: added, not the last
    (entry,) = trace_of(result).usage
    assert (entry.tokens_in, entry.tokens_out) == (35, 5)
    assert entry.cost_usd == pytest.approx(0.03)
    # A run cut short with no ResultMessage still ends its turn, and adds no cost.
    cut = first[:3]
    short = claude_agent_sdk.from_messages([cut, second])
    assert [(c.name, c.turn) for c in calls_of(short)] == [("A", 0), ("B", 1)]
    assert trace_of(short).responses == ["one", "two"]
    assert short.get("cost_usd") == pytest.approx(0.02)


def test_claude_sub_agent_text_is_not_the_reply_plain_dicts_and_missing_attributes() -> None:
    messages: list[Any] = [
        {
            "content": [{"name": "Task", "input": {"p": 1}, "id": "t"}],
            "model": "m",
            "message_id": "a",
        },
        AssistantMessage([TextBlock("sub-agent chatter")], "m", parent_tool_use_id="t"),
        {"content": [{"tool_use_id": "t", "content": "sub done", "is_error": False}]},
        {"content": [{"text": "All done."}], "model": "m"},
        {"content": "a plain user prompt"},
        {"subtype": "init", "data": {}},
        NS(),
        {"is_error": False, "num_turns": 2, "model_usage": {"m": {"inputTokens": 3}}},
    ]
    result = claude_agent_sdk.from_messages(messages)
    assert result["output"] == "All done." and calls_of(result)[0].result == "sub done"
    entry = trace_of(result).usage[0]
    assert (entry.tokens_in, entry.tokens_out, entry.cached_tokens, entry.cost_usd) == (
        3,
        0,
        0,
        None,
    )
    bare = claude_agent_sdk.from_messages([])
    assert bare["output"] == "" and trace_of(bare).usage == []


# --- OpenAI Agents SDK ------------------------------------------------------------------------
# Shaped like agents/items.py and agents/result.py: items carry `type`, the call lives in
# `raw_item` (a function call with a JSON `arguments` string), the usage in
# `context_wrapper.usage`.


def call_item(name: str, args: str, call_id: str | None) -> Any:
    return NS(type="tool_call_item", raw_item=NS(name=name, arguments=args, call_id=call_id))


def out_item(call_id: str | None, output: Any) -> Any:
    return NS(
        type="tool_call_output_item",
        raw_item={"call_id": call_id, "output": str(output), "type": "function_call_output"},
        output=output,
    )


def msg_item(text: str) -> Any:
    return NS(
        type="message_output_item",
        raw_item=NS(content=[NS(type="output_text", text=text), NS(refusal="x")]),
    )


def run_result(
    items: list[Any], final: Any, *, requests: int = 2, i: int = 100, o: int = 20, cached: int = 40
) -> Any:
    usage_obj = NS(
        requests=requests,
        input_tokens=i,
        output_tokens=o,
        input_tokens_details=NS(cached_tokens=cached),
        output_tokens_details=NS(reasoning_tokens=0),
    )
    return NS(
        new_items=items,
        final_output=final,
        raw_responses=[NS(usage=usage_obj, response_id=None)] * requests,
        context_wrapper=NS(usage=usage_obj),
    )


def test_openai_pairs_by_call_id_and_reads_usage() -> None:
    result = oa.from_result(
        run_result(
            [
                call_item("weather", '{"city": "Paris"}', "k1"),
                call_item("weather", '{"city": "Rome"}', "k2"),
                out_item("k2", {"temp_c": 24}),
                out_item("k1", {"temp_c": 20}),
                msg_item("Paris 20, Rome 24."),
            ],
            "Paris 20, Rome 24.",
        )
    )
    calls = calls_of(result)
    assert [(c.id, c.args, c.result) for c in calls] == [
        ("k1", {"city": "Paris"}, {"temp_c": 20}),
        ("k2", {"city": "Rome"}, {"temp_c": 24}),
    ]
    assert result["output"] == "Paris 20, Rome 24."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["Paris 20, Rome 24."]
    (entry,) = trace.usage
    assert (entry.role, entry.model, entry.tokens_in, entry.tokens_out, entry.cached_tokens) == (
        "agent",
        None,
        100,
        20,
        40,
    )
    assert entry.cost_usd is None and "cost_usd" not in result
    accepted(result)


def test_openai_pairs_by_order_without_call_ids_and_error_hook() -> None:
    items = [
        call_item("a", "{}", None),
        call_item("b", "not json", None),
        out_item(None, "first"),
        out_item(None, "An error occurred while running the tool. Please try again."),
    ]
    # The SDK reports a raised tool as its fixed default text: failed out of the box.
    plain = calls_of(oa.from_result(run_result(items, "x")))
    assert [(c.name, c.result, c.ok) for c in plain] == [
        ("a", "first", True),
        ("b", oa.DEFAULT_TOOL_ERROR, False),
    ]
    assert plain[1].error == oa.DEFAULT_TOOL_ERROR
    assert plain[1].args == {"input": "not json"}
    # Only the exact text counts; a custom failure text needs error_if, which replaces the test.
    custom = [*items[:3], out_item(None, "tool failed: quota")]
    assert [c.ok for c in calls_of(oa.from_result(run_result(custom, "x")))] == [True, True]
    hooked = calls_of(
        oa.from_result(run_result(custom, "x"), error_if=lambda out: str(out).startswith("tool f"))
    )
    assert [c.ok for c in hooked] == [True, False] and hooked[1].error == "tool failed: quota"
    off = calls_of(oa.from_result(run_result(items, "x"), error_if=lambda out: False))
    assert [c.ok for c in off] == [True, True]


def test_openai_final_output_variants_and_missing_attributes() -> None:
    structured = NS(model_dump_json=lambda: '{"city": "Paris"}')
    assert oa.from_result(run_result([], structured))["output"] == '{"city": "Paris"}'
    assert oa.from_result(run_result([], {"a": 1}))["output"] == '{"a": 1}'
    assert oa.from_result(run_result([msg_item("one"), msg_item("last")], None))["output"] == "last"
    assert oa.from_result(run_result([], ""))["output"] == ""
    bare = oa.from_result(NS())
    assert bare["output"] == "" and trace_of(bare).rounds is None and trace_of(bare).usage == []
    from_dicts = oa.from_result(
        {
            "new_items": [
                {
                    "type": "tool_call_item",
                    "raw_item": {"name": "t", "arguments": '{"a": 1}', "call_id": "c"},
                },
                {"type": "tool_call_output_item", "raw_item": {"call_id": "c", "output": "raw"}},
            ],
            "final_output": "ok",
            "context_wrapper": {"usage": {"requests": 3, "input_tokens": 9, "output_tokens": 1}},
        }
    )
    assert calls_of(from_dicts)[0].result == "raw"
    assert trace_of(from_dicts).rounds == 3 and trace_of(from_dicts).usage[0].tokens_in == 9


def test_openai_list_of_results_is_one_result_per_turn() -> None:
    first = run_result(
        [call_item("t1", "{}", "a"), out_item("a", "r1")], "one", requests=1, i=10, o=1
    )
    second = run_result(
        [call_item("t2", "{}", "b"), out_item("b", "r2")], "two", requests=2, i=30, o=3
    )
    result = oa.from_result([first, second])
    assert [(c.name, c.turn) for c in calls_of(result)] == [("t1", 0), ("t2", 1)]
    trace = trace_of(result)
    assert trace.responses == ["one", "two"] and result["output"] == "two" and trace.rounds == 3
    assert (trace.usage[0].tokens_in, trace.usage[0].tokens_out) == (40, 4)
    assert oa.from_result([])["output"] == ""


def test_openai_results_sharing_one_usage_count_it_once() -> None:
    # The same RunContextWrapper passed to both runs: one cumulative Usage, already the total.
    first = run_result([], "one", requests=1, i=10, o=1)
    second = run_result([], "two", requests=2, i=40, o=4)
    second.context_wrapper = first.context_wrapper
    first.context_wrapper.usage.input_tokens, first.context_wrapper.usage.output_tokens = 40, 4
    trace = trace_of(oa.from_result([first, second]))
    assert (trace.usage[0].tokens_in, trace.usage[0].tokens_out) == (40, 4)
    assert trace.rounds == 3  # rounds come from each result's own raw_responses
    # Without raw_responses, the shared request count is not added twice either.
    for one in (first, second):
        del one.raw_responses
    first.context_wrapper.usage.requests = 3
    assert trace_of(oa.from_result([first, second])).rounds == 3


# --- through PythonSystem and the checks scorer -----------------------------------------------

AGENT_MODULE = """
from dataclasses import dataclass
from typing import Any

from evalproof.adapters import claude_agent_sdk, langchain


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict


@dataclass
class ToolResultBlock:
    tool_use_id: str
    content: Any = None
    is_error: Any = None


@dataclass
class TextBlock:
    text: str


@dataclass
class AssistantMessage:
    content: list
    model: str
    message_id: Any = None


@dataclass
class UserMessage:
    content: Any


@dataclass
class ResultMessage:
    total_cost_usd: Any
    result: str
    model_usage: dict
    num_turns: int = 2
    is_error: bool = False


def claude_agent(question):
    messages = [
        AssistantMessage([ToolUseBlock("u1", "get_weather", {"city": "Paris"})], "m", "a"),
        UserMessage([ToolResultBlock("u1", "20 C, clear")]),
        AssistantMessage([TextBlock("It is 20 C in Paris.")], "m", "b"),
        ResultMessage(0.05, "It is 20 C in Paris.", {
            "m": {"inputTokens": 120, "outputTokens": 30, "cacheReadInputTokens": 0,
                  "costUSD": 0.05}
        }),
    ]
    return claude_agent_sdk.from_messages(messages)


def langchain_agent(question):
    call = {"name": "get_weather", "args": {"city": "Paris"}, "id": "t1"}
    messages = [
        {"type": "human", "content": question},
        {"type": "ai", "content": "", "tool_calls": [call],
         "usage_metadata": {"input_tokens": 100, "output_tokens": 10, "total_tokens": 110}},
        {"type": "tool", "tool_call_id": "t1", "content": "20 C, clear", "status": "success"},
        {"type": "ai", "content": "It is 20 C in Paris.",
         "usage_metadata": {"input_tokens": 140, "output_tokens": 12, "total_tokens": 152}},
    ]
    return langchain.from_messages(messages)
"""


@pytest.fixture
def agent_module(tmp_path: Path) -> Any:
    (tmp_path / "adapter_agent.py").write_text(textwrap.dedent(AGENT_MODULE), "utf-8")
    try:
        yield tmp_path
    finally:
        sys.modules.pop("adapter_agent", None)


CHECKS = [
    {"type": "tool_called", "name": "get_weather"},
    {"type": "tool_args_contain", "name": "get_weather", "arg": "city", "value": "Paris"},
    {"type": "tool_succeeded", "name": "get_weather"},
    {"type": "output_contains", "text": "20 C"},
    {"type": "rounds", "max": 2},
]


def test_claude_adapter_through_python_system_and_checks(agent_module: Path) -> None:
    system = PythonSystem("python:adapter_agent:claude_agent", search_path=str(agent_module))
    out = system.run(Case(id="c", input="weather in Paris?"))
    assert out.error is None and out.output == "It is 20 C in Paris."
    assert (out.tokens_in, out.tokens_out, out.cost_usd) == (120, 30, 0.05)
    assert out.trace is not None and out.trace.tool_calls[0].result == "20 C, clear"
    score = ChecksScorer("behaviour", checks=CHECKS).score(Case(id="c", input="q"), out)
    assert score.passed is True, score.reason
    broken = ChecksScorer("b", checks=[{"type": "tool_called", "name": "book"}])
    assert broken.score(Case(id="c", input="q"), out).passed is False


def test_langchain_adapter_rolls_tokens_up_and_prices_them(agent_module: Path) -> None:
    pricing = Pricing(input_per_mtok=3.0, output_per_mtok=15.0)
    system = PythonSystem(
        "python:adapter_agent:langchain_agent", search_path=str(agent_module), pricing=pricing
    )
    out = system.run(Case(id="c", input="weather in Paris?"))
    assert out.error is None and out.output == "It is 20 C in Paris."
    assert (out.tokens_in, out.tokens_out) == (240, 22)
    assert out.cost_usd == pytest.approx(pricing.cost(240, 22))  # no SDK cost, so priced
    score = ChecksScorer("behaviour", checks=CHECKS).score(Case(id="c", input="q"), out)
    assert score.passed is True, score.reason


def test_public_api() -> None:
    assert set(adapters.__all__) == {
        "AgentResult",
        "TraceRecorder",
        "adk",
        "claude_agent_sdk",
        "langchain",
        "mcp_call",
        "openai_agents",
    }
