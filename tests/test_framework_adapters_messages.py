"""The experimental message-list adapters: Pydantic AI, Haystack, AutoGen and Semantic Kernel.

The fakes are shaped exactly like the fields each adapter cites from the SDK's source (see the
docstring of every adapter module); no SDK is imported. For each adapter: a tool call with its
arguments and result, several tools in order, the reply, empty and tool-free runs, objects and
plain dicts, odd input, turns, failures, and the path through the real ``checks`` scorer.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from evalproof.adapters import AgentResult, autogen, haystack, pydantic_ai, semantic_kernel
from evalproof.adapters import _common as common
from evalproof.models import Case, SystemOutput, ToolCall, Trace
from evalproof.scorers import ChecksScorer
from evalproof.systems import _coerce

NS = SimpleNamespace

pytestmark = pytest.mark.filterwarnings("ignore:evalproof")

CHECKS = [
    {"type": "tool_called", "name": "get_weather"},
    {"type": "tool_args_contain", "name": "get_weather", "arg": "city", "value": "Paris"},
    {"type": "tool_succeeded", "name": "get_weather"},
    {"type": "output_contains", "text": "20 C"},
    {"type": "rounds", "max": 2},
]


@pytest.fixture(autouse=True)
def fresh_experimental_warnings() -> None:
    common._warned.clear()


def trace_of(result: AgentResult) -> Trace:
    trace = result["trace"]
    assert isinstance(trace, Trace)
    return trace


def calls_of(result: AgentResult) -> list[ToolCall]:
    return trace_of(result).tool_calls


def system_output(result: AgentResult) -> SystemOutput:
    """What a python system would hand the scorers: the result through ``systems._coerce``."""
    coerced = _coerce(result)
    assert coerced.trace is not None
    return SystemOutput(output=coerced.output, trace=coerced.trace)


def passes_checks(result: AgentResult) -> None:
    """The result goes through ``systems._coerce`` and the real ``checks`` scorer."""
    out = system_output(result)
    score = ChecksScorer("behaviour", checks=CHECKS).score(Case(id="c", input="q"), out)
    assert score.passed is True, score.reason
    broken = ChecksScorer("b", checks=[{"type": "tool_called", "name": "book"}])
    assert broken.score(Case(id="c", input="q"), out).passed is False


# --- Pydantic AI --------------------------------------------------------------------------------


def pai_request(*parts: Any) -> Any:
    return NS(kind="request", parts=list(parts))


def pai_response(*parts: Any, model: str = "m", usage: Any = None) -> Any:
    return NS(kind="response", parts=list(parts), model_name=model, usage=usage)


def pai_prompt(text: str) -> Any:
    return NS(part_kind="user-prompt", content=text)


def pai_call(name: str, args: Any, id: str) -> Any:
    return NS(part_kind="tool-call", tool_name=name, args=args, tool_call_id=id)


def pai_return(name: str, content: Any, id: str, outcome: str | None = "success") -> Any:
    return NS(
        part_kind="tool-return", tool_name=name, content=content, tool_call_id=id, outcome=outcome
    )


def pai_text(text: str) -> Any:
    return NS(part_kind="text", content=text)


def pai_weather() -> list[Any]:
    return [
        pai_request(NS(part_kind="system-prompt", content="be brief"), pai_prompt("Weather?")),
        pai_response(
            pai_call("get_weather", {"city": "Paris"}, "t1"),
            usage=NS(input_tokens=100, output_tokens=10, cache_read_tokens=40),
        ),
        pai_request(pai_return("get_weather", "20 C, clear", "t1")),
        pai_response(
            pai_text("It is 20 C in Paris."),
            usage=NS(input_tokens=140, output_tokens=12, cache_read_tokens=60),
        ),
    ]


def pai_weather_dicts() -> list[dict[str, Any]]:
    """The same run as the dicts the SDK's own serialisation produces."""
    return [
        {"kind": "request", "parts": [{"part_kind": "user-prompt", "content": "Weather?"}]},
        {
            "kind": "response",
            "model_name": "m",
            "usage": {"input_tokens": 100, "output_tokens": 10, "cache_read_tokens": 40},
            "parts": [
                {
                    "part_kind": "tool-call",
                    "tool_name": "get_weather",
                    "args": '{"city": "Paris"}',
                    "tool_call_id": "t1",
                }
            ],
        },
        {
            "kind": "request",
            "parts": [
                {
                    "part_kind": "tool-return",
                    "tool_name": "get_weather",
                    "content": "20 C, clear",
                    "tool_call_id": "t1",
                    "outcome": "success",
                }
            ],
        },
        {
            "kind": "response",
            "model_name": "m",
            "usage": {"input_tokens": 140, "output_tokens": 12, "cache_read_tokens": 60},
            "parts": [{"part_kind": "text", "content": "It is 20 C in Paris."}],
        },
    ]


def test_pydantic_ai_call_args_result_reply_and_usage() -> None:
    result = pydantic_ai.from_messages(pai_weather())
    assert result["output"] == "It is 20 C in Paris."
    trace = trace_of(result)
    assert trace.responses == ["It is 20 C in Paris."]
    assert trace.rounds == 2
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.turn) == ("get_weather", {"city": "Paris"}, "t1", 0)
    assert (call.result, call.ok, call.error) == ("20 C, clear", True, None)
    (usage,) = trace.usage
    assert (usage.model, usage.tokens_in, usage.tokens_out, usage.cached_tokens) == (
        "m",
        240,
        22,
        100,
    )
    passes_checks(result)


def test_pydantic_ai_dicts_match_objects_and_the_state_shapes() -> None:
    from_objects = pydantic_ai.from_messages(pai_weather())
    assert pydantic_ai.from_messages(pai_weather_dicts()) == from_objects
    assert pydantic_ai.from_messages({"messages": pai_weather_dicts()}) == from_objects
    run = NS(all_messages=lambda: pai_weather(), output="It is 20 C in Paris.")
    assert pydantic_ai.from_messages(run) == from_objects


def test_pydantic_ai_several_tools_in_order_paired_by_id() -> None:
    messages = [
        pai_request(pai_prompt("go")),
        pai_response(pai_call("a", '{"x": 1}', "1"), pai_call("b", {"y": 2}, "2")),
        pai_request(pai_return("b", {"v": 2}, "2"), pai_return("a", "one", "1")),
        pai_response(pai_call("c", None, "3")),
        pai_request(pai_return("c", "three", "3")),
        pai_response(pai_text("done")),
    ]
    calls = calls_of(pydantic_ai.from_messages(messages))
    assert [c.name for c in calls] == ["a", "b", "c"]
    assert [c.args for c in calls] == [{"x": 1}, {"y": 2}, {}]
    assert [c.result for c in calls] == ["one", {"v": 2}, "three"]


def test_pydantic_ai_failed_denied_and_retried_calls() -> None:
    messages = [
        pai_request(pai_prompt("go")),
        pai_response(pai_call("a", {}, "1"), pai_call("b", {}, "2"), pai_call("c", {}, "3")),
        pai_request(
            pai_return("a", "boom", "1", outcome="failed"),
            pai_return("b", "no", "2", outcome="denied"),
            NS(
                part_kind="retry-prompt",
                tool_name="c",
                tool_call_id="3",
                content=[{"loc": ["city"], "msg": "field required"}],
            ),
        ),
        pai_response(pai_text("sorry")),
    ]
    a, b, c = calls_of(pydantic_ai.from_messages(messages))
    assert (a.ok, a.error) == (False, "boom")
    assert (b.ok, b.error) == (False, "no")
    assert c.ok is False and "field required" in (c.error or "")
    # a retry of the final output names no tool: it is not a call
    no_tool = [pai_request(NS(part_kind="retry-prompt", tool_name=None, content="bad json"))]
    assert calls_of(pydantic_ai.from_messages(no_tool)) == []


def test_pydantic_ai_turns_provider_tools_structured_output_and_text_with_a_call() -> None:
    messages = [
        pai_request(pai_prompt("one")),
        pai_response(pai_text("thinking aloud"), pai_call("a", {}, "1")),
        pai_request(pai_return("a", "x", "1")),
        pai_response(NS(part_kind="builtin-tool-call", tool_name="web", tool_call_id="w")),
        pai_response(pai_text("first reply")),
        pai_request(pai_prompt("two")),
        pai_response(pai_call("a", {}, "2")),
        pai_request(pai_return("a", "y", "2")),
        pai_response(pai_text("second reply")),
    ]
    result = pydantic_ai.from_messages(messages)
    trace = trace_of(result)
    assert trace.responses == ["first reply", "second reply"]
    assert [c.turn for c in trace.tool_calls] == [0, 1]
    assert trace.rounds == 5  # the provider-side call still was a model response
    structured = NS(all_messages=lambda: messages, output={"city": "Paris", "temp": 20})
    assert json.loads(pydantic_ai.from_messages(structured)["output"]) == {
        "city": "Paris",
        "temp": 20,
    }


def test_pydantic_ai_empty_tool_free_and_odd_input() -> None:
    for empty in (None, [], {}, {"messages": None}, "text", 3):
        result = pydantic_ai.from_messages(empty)
        assert result["output"] == "" and calls_of(result) == []
        assert trace_of(result).responses == [] and trace_of(result).rounds is None
    plain = [pai_request(pai_prompt("hi")), pai_response(pai_text("hello"))]
    assert pydantic_ai.from_messages(plain)["output"] == "hello"
    odd = [
        NS(kind="response"),  # no parts
        NS(kind="request", parts=None),
        NS(kind="response", parts=[NS(part_kind="thinking", content="hm"), NS()]),
        pai_response(NS(part_kind="tool-call", tool_name=None, args=5, tool_call_id=None)),
        pai_request(NS(part_kind="tool-return", content=None)),
    ]
    result = pydantic_ai.from_messages(odd)
    (call,) = calls_of(result)  # the id-less return answers the id-less call
    assert (call.name, call.args, call.ok) == ("unknown", {"input": 5}, True)


# --- Haystack -----------------------------------------------------------------------------------


def hs_message(
    role: str,
    *,
    texts: tuple[str, ...] = (),
    calls: tuple[Any, ...] = (),
    results: tuple[Any, ...] = (),
    meta: Any = None,
) -> Any:
    return NS(
        role=NS(value=role),
        texts=list(texts),
        text=texts[0] if texts else None,
        tool_calls=list(calls),
        tool_call_results=list(results),
        meta=meta or {},
    )


def hs_weather() -> list[Any]:
    call = NS(tool_name="get_weather", arguments={"city": "Paris"}, id="t1")
    return [
        hs_message("system", texts=("be brief",)),
        hs_message("user", texts=("Weather?",)),
        hs_message(
            "assistant",
            calls=(call,),
            meta={"model": "m", "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
        ),
        hs_message("tool", results=(NS(result="20 C, clear", origin=call, error=False),)),
        hs_message(
            "assistant",
            texts=("It is 20 C in Paris.",),
            meta={"model": "m", "usage": {"prompt_tokens": 140, "completion_tokens": 12}},
        ),
    ]


def hs_weather_dicts() -> list[dict[str, Any]]:
    origin = {"tool_name": "get_weather", "arguments": {"city": "Paris"}, "id": "t1"}
    return [
        {"role": "system", "meta": {}, "name": None, "content": [{"text": "be brief"}]},
        {"role": "user", "meta": {}, "name": None, "content": [{"text": "Weather?"}]},
        {
            "role": "assistant",
            "name": None,
            "meta": {"model": "m", "usage": {"prompt_tokens": 100, "completion_tokens": 10}},
            "content": [{"tool_call": origin}],
        },
        {
            "role": "tool",
            "meta": {},
            "name": None,
            "content": [
                {"tool_call_result": {"result": "20 C, clear", "origin": origin, "error": False}}
            ],
        },
        {
            "role": "assistant",
            "name": None,
            "meta": {"model": "m", "usage": {"prompt_tokens": 140, "completion_tokens": 12}},
            "content": [{"text": "It is 20 C in Paris."}],
        },
    ]


def test_haystack_call_args_result_reply_and_usage() -> None:
    result = haystack.from_messages(hs_weather())
    assert result["output"] == "It is 20 C in Paris."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["It is 20 C in Paris."]
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.result, call.ok) == (
        "get_weather",
        {"city": "Paris"},
        "t1",
        "20 C, clear",
        True,
    )
    (usage,) = trace.usage
    assert (usage.model, usage.tokens_in, usage.tokens_out) == ("m", 240, 22)
    passes_checks(result)


def test_haystack_dicts_match_objects_and_agent_run_dict() -> None:
    from_objects = haystack.from_messages(hs_weather())
    assert haystack.from_messages(hs_weather_dicts()) == from_objects
    assert haystack.from_messages({"messages": hs_weather(), "step_count": 2}) == from_objects
    plain_roles = [NS(**{**vars(m), "role": m.role.value}) for m in hs_weather()]
    assert haystack.from_messages(plain_roles) == from_objects


def test_haystack_other_generator_usage_names_several_tools_and_failures() -> None:
    a = NS(tool_name="a", arguments={"n": 1}, id=None)
    b = NS(tool_name="b", arguments={}, id=None)
    messages = [
        hs_message("user", texts=("go",)),
        hs_message(
            "assistant",
            calls=(a, b),
            meta={"model": "m2", "usage": {"input_tokens": 7, "output_tokens": 3}},
        ),
        hs_message(
            "tool",
            results=(
                NS(result=[NS(text="part one"), NS(text=" two")], origin=a, error=False),
                NS(result="denied by policy", origin=b, error=True),
            ),
        ),
        hs_message("assistant", texts=("done",)),
    ]
    result = haystack.from_messages(messages)
    first, second = calls_of(result)
    assert (first.name, first.result, first.ok) == ("a", "part one two", True)
    assert (second.name, second.ok, second.error) == ("b", False, "denied by policy")
    assert [(u.model, u.tokens_in, u.tokens_out) for u in trace_of(result).usage] == [("m2", 7, 3)]


def test_haystack_turns_text_with_a_call_and_odd_input() -> None:
    call = NS(tool_name="a", arguments={}, id="1")
    messages = [
        hs_message("user", texts=("one",)),
        hs_message("assistant", texts=("let me check",), calls=(call,)),
        hs_message("tool", results=(NS(result="x", origin=call, error=False),)),
        hs_message("assistant", texts=("first",)),
        hs_message("user", texts=("two",)),
        hs_message("assistant", texts=("second",)),
    ]
    trace = trace_of(haystack.from_messages(messages))
    assert trace.responses == ["first", "second"]
    assert [c.turn for c in trace.tool_calls] == [0]
    for empty in (None, [], {}, {"messages": None}, "text", 3):
        result = haystack.from_messages(empty)
        assert result["output"] == "" and calls_of(result) == []
    odd = [
        NS(),
        {"role": "assistant"},
        {"role": "assistant", "content": ["junk", {"text": 5}, {"tool_call": None}]},
        hs_message("assistant", texts=("",)),
        {"role": "tool", "content": [{"tool_call_result": {"result": None, "origin": None}}]},
    ]
    result = haystack.from_messages(odd)
    assert result["output"] == ""
    assert [c.name for c in calls_of(result)] == ["unknown"]


# --- AutoGen ------------------------------------------------------------------------------------


def ag_text(source: str, content: str, usage: Any = None) -> Any:
    return NS(type="TextMessage", source=source, content=content, models_usage=usage)


def ag_request(*calls: Any, usage: Any = None) -> Any:
    return NS(type="ToolCallRequestEvent", source="agent", content=list(calls), models_usage=usage)


def ag_execution(*results: Any) -> Any:
    return NS(type="ToolCallExecutionEvent", source="agent", content=list(results))


def ag_weather() -> list[Any]:
    call = NS(id="t1", name="get_weather", arguments='{"city": "Paris"}')
    return [
        ag_text("user", "Weather?"),
        ag_request(call, usage=NS(prompt_tokens=100, completion_tokens=10)),
        ag_execution(NS(content="20 C, clear", name="get_weather", call_id="t1", is_error=False)),
        ag_text("agent", "It is 20 C in Paris.", NS(prompt_tokens=140, completion_tokens=12)),
    ]


def ag_weather_dicts() -> list[dict[str, Any]]:
    return [
        {"type": "TextMessage", "source": "user", "content": "Weather?", "models_usage": None},
        {
            "type": "ToolCallRequestEvent",
            "source": "agent",
            "models_usage": {"prompt_tokens": 100, "completion_tokens": 10},
            "content": [{"id": "t1", "name": "get_weather", "arguments": '{"city": "Paris"}'}],
        },
        {
            "type": "ToolCallExecutionEvent",
            "source": "agent",
            "models_usage": None,
            "content": [
                {
                    "content": "20 C, clear",
                    "name": "get_weather",
                    "call_id": "t1",
                    "is_error": False,
                }
            ],
        },
        {
            "type": "TextMessage",
            "source": "agent",
            "content": "It is 20 C in Paris.",
            "models_usage": {"prompt_tokens": 140, "completion_tokens": 12},
        },
    ]


def test_autogen_call_args_result_reply_and_usage() -> None:
    result = autogen.from_messages(ag_weather())
    assert result["output"] == "It is 20 C in Paris."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["It is 20 C in Paris."]
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.result, call.ok) == (
        "get_weather",
        {"city": "Paris"},
        "t1",
        "20 C, clear",
        True,
    )
    (usage,) = trace.usage
    assert (usage.model, usage.tokens_in, usage.tokens_out) == (None, 240, 22)
    passes_checks(result)


def test_autogen_task_result_dicts_and_objects_agree() -> None:
    from_objects = autogen.from_messages(ag_weather())
    assert autogen.from_messages(NS(messages=ag_weather(), stop_reason="done")) == from_objects
    assert autogen.from_messages(ag_weather_dicts()) == from_objects
    assert autogen.from_messages({"messages": ag_weather_dicts(), "stop_reason": None}) == (
        from_objects
    )


def test_autogen_several_tools_failures_and_summary_message() -> None:
    a = NS(id="1", name="a", arguments="{}")
    b = NS(id="2", name="b", arguments={"y": 2})
    messages = [
        ag_text("user", "go"),
        ag_request(a, b),
        ag_execution(
            NS(content="nope", name="b", call_id="2", is_error=True),
            NS(content="fine", name="a", call_id="1", is_error=None),
        ),
        NS(
            type="ToolCallSummaryMessage",
            source="agent",
            content="fine\nnope",
            tool_calls=[a, b],
            results=[],
            models_usage=None,
        ),
    ]
    result = autogen.from_messages(messages)
    first, second = calls_of(result)
    assert (first.name, first.ok, first.result) == ("a", True, "fine")
    assert (second.name, second.ok, second.error) == ("b", False, "nope")
    assert result["output"] == "fine\nnope"  # the summary is the reply; its calls are not re-read
    assert len(calls_of(result)) == 2


def test_autogen_turns_structured_and_multimodal_replies() -> None:
    messages = [
        ag_text("user", "one"),
        ag_text("agent", "first"),
        ag_text("user", "two"),
        NS(type="StructuredMessage", source="agent", content={"answer": 42}, models_usage=None),
        ag_text("user", "three"),
        NS(type="MultiModalMessage", source="agent", content=["see", NS(), "this"]),
        NS(type="ThoughtEvent", source="agent", content="hmm"),
    ]
    trace = trace_of(autogen.from_messages(messages))
    assert trace.responses[0] == "first"
    assert json.loads(trace.responses[1]) == {"answer": 42}
    assert trace.responses[2] == "seethis"


def test_autogen_empty_tool_free_and_odd_input() -> None:
    for empty in (None, [], {}, {"messages": None}, "text", 3):
        result = autogen.from_messages(empty)
        assert result["output"] == "" and calls_of(result) == []
    plain = autogen.from_messages([ag_text("user", "hi"), ag_text("agent", "hello")])
    assert plain["output"] == "hello" and trace_of(plain).rounds is None
    odd = [
        NS(),
        {"type": "ToolCallRequestEvent", "content": [{"name": None, "arguments": 3}, None]},
        {"type": "ToolCallExecutionEvent", "content": None},
        {"type": "TextMessage", "source": "agent", "content": None},
        {"type": "SomethingNew", "content": "x"},
    ]
    result = autogen.from_messages(odd)
    assert [c.name for c in calls_of(result)] == ["unknown", "unknown"]
    assert result["output"] == ""


# --- Semantic Kernel ----------------------------------------------------------------------------


def sk_text(role: str, text: str) -> Any:
    return NS(role=NS(value=role), items=[NS(content_type="text", text=text)], content=text)


def sk_call(name: str, args: Any, id: str) -> Any:
    return NS(content_type="function_call", name=name, arguments=args, id=id, call_id=None)


def sk_result(name: str, value: Any, id: str) -> Any:
    return NS(content_type="function_result", name=name, result=value, id=id, call_id=None)


def sk_weather() -> list[Any]:
    return [
        sk_text("system", "be brief"),
        sk_text("user", "Weather?"),
        NS(
            role=NS(value="assistant"),
            items=[sk_call("weather-get_weather", '{"city": "Paris"}', "t1")],
        ),
        NS(role=NS(value="tool"), items=[sk_result("weather-get_weather", "20 C, clear", "t1")]),
        sk_text("assistant", "It is 20 C in Paris."),
    ]


def sk_weather_dicts() -> list[dict[str, Any]]:
    return [
        {"role": "system", "items": [{"content_type": "text", "text": "be brief"}]},
        {"role": "user", "items": [{"content_type": "text", "text": "Weather?"}]},
        {
            "role": "assistant",
            "items": [
                {
                    "content_type": "function_call",
                    "name": "weather-get_weather",
                    "arguments": {"city": "Paris"},
                    "id": "t1",
                }
            ],
        },
        {
            "role": "tool",
            "items": [
                {
                    "content_type": "function_result",
                    "name": "weather-get_weather",
                    "result": "20 C, clear",
                    "id": "t1",
                }
            ],
        },
        {"role": "assistant", "items": [{"content_type": "text", "text": "It is 20 C in Paris."}]},
    ]


def test_semantic_kernel_call_args_result_and_reply() -> None:
    result = semantic_kernel.from_history(sk_weather())
    assert result["output"] == "It is 20 C in Paris."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["It is 20 C in Paris."] and trace.usage == []
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.result, call.ok) == (
        "weather-get_weather",
        {"city": "Paris"},
        "t1",
        "20 C, clear",
        True,
    )


def test_semantic_kernel_history_dicts_and_objects_agree() -> None:
    from_objects = semantic_kernel.from_history(sk_weather())
    assert semantic_kernel.from_history(NS(messages=sk_weather())) == from_objects
    assert semantic_kernel.from_history({"messages": sk_weather_dicts()}) == from_objects
    assert semantic_kernel.from_history(sk_weather_dicts()) == from_objects
    plain_roles = [NS(**{**vars(m), "role": m.role.value}) for m in sk_weather()]
    assert semantic_kernel.from_history(plain_roles) == from_objects


def test_semantic_kernel_several_tools_in_order_and_function_name_fallback() -> None:
    messages = [
        sk_text("user", "go"),
        NS(
            role=NS(value="assistant"),
            items=[
                NS(
                    content_type="function_call",
                    name=None,
                    function_name="a",
                    arguments={"x": 1},
                    id="1",
                ),
                sk_call("p-b", None, "2"),
            ],
        ),
        NS(
            role=NS(value="tool"),
            items=[sk_result("p-b", {"v": 2}, "2"), sk_result(None, "one", "1")],
        ),
        sk_text("assistant", "done"),
    ]
    messages[2].items[1].function_name = "a"
    calls = calls_of(semantic_kernel.from_history(messages))
    assert [c.name for c in calls] == ["a", "p-b"]
    assert [c.args for c in calls] == [{"x": 1}, {}]
    assert [c.result for c in calls] == ["one", {"v": 2}]


def test_semantic_kernel_turns_content_fallback_and_odd_input() -> None:
    messages = [
        sk_text("user", "one"),
        NS(
            role=NS(value="assistant"),
            items=[sk_call("a", "{}", "1"), NS(content_type="text", text="hm")],
        ),
        NS(role=NS(value="tool"), items=[sk_result("a", "x", "1")]),
        {"role": "assistant", "items": [], "content": "first"},
        sk_text("user", "two"),
        sk_text("assistant", "second"),
    ]
    trace = trace_of(semantic_kernel.from_history(messages))
    assert trace.responses == ["first", "second"]
    assert [c.turn for c in trace.tool_calls] == [0]
    for empty in (None, [], {}, {"messages": None}, "text", 3):
        result = semantic_kernel.from_history(empty)
        assert result["output"] == "" and calls_of(result) == []
    odd = [
        NS(),
        {"role": "assistant"},
        {
            "role": "assistant",
            "items": [{"content_type": "image"}, {"content_type": "text", "text": 5}],
        },
        {"role": "developer", "items": []},
        {"role": "tool", "items": [{"content_type": "function_result", "result": None}]},
    ]
    result = semantic_kernel.from_history(odd)
    assert result["output"] == ""
    assert [c.name for c in calls_of(result)] == ["unknown"]


def test_semantic_kernel_through_the_real_checks_scorer() -> None:
    result = semantic_kernel.from_history(sk_weather())
    # the tool is named with its plugin, as the SDK names it
    out = system_output(result)
    checks = [
        {"type": "tool_called", "name": "weather-get_weather"},
        {
            "type": "tool_args_contain",
            "name": "weather-get_weather",
            "arg": "city",
            "value": "Paris",
        },
        {"type": "tool_succeeded", "name": "weather-get_weather"},
        {"type": "output_contains", "text": "20 C"},
        {"type": "rounds", "max": 2},
    ]
    score = ChecksScorer("behaviour", checks=checks).score(Case(id="c", input="q"), out)
    assert score.passed is True, score.reason
