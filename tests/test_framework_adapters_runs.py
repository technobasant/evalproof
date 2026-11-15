"""The experimental run-result adapters: LlamaIndex, CrewAI and DSPy, and the experimental notice
all seven new adapters share.

The fakes are shaped exactly like the fields each adapter cites from the SDK's source (see the
docstring of every adapter module); no SDK is imported.
"""

from __future__ import annotations

import json
import warnings
from types import SimpleNamespace
from typing import Any

import pytest

from evalproof.adapters import (
    AgentResult,
    autogen,
    crewai,
    dspy,
    haystack,
    llama_index,
    pydantic_ai,
    semantic_kernel,
)
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


def passes_checks(result: AgentResult, checks: list[dict[str, Any]] | None = None) -> None:
    """The result goes through ``systems._coerce`` and the real ``checks`` scorer."""
    coerced = _coerce(result)
    assert coerced.trace is not None
    out = SystemOutput(output=coerced.output, trace=coerced.trace)
    score = ChecksScorer("behaviour", checks=checks or CHECKS).score(Case(id="c", input="q"), out)
    assert score.passed is True, score.reason
    broken = ChecksScorer("b", checks=[{"type": "tool_called", "name": "book"}])
    assert broken.score(Case(id="c", input="q"), out).passed is False


# --- LlamaIndex ---------------------------------------------------------------------------------


def li_output(text: str, *selections: Any, structured: Any = None) -> Any:
    return NS(
        response=NS(content=text, blocks=[NS(text=text)]),
        tool_calls=list(selections),
        structured_response=structured,
        current_agent_name="agent",
    )


def li_selection(name: str, kwargs: Any, id: str) -> Any:
    return NS(tool_id=id, tool_name=name, tool_kwargs=kwargs)


def li_tool_call(name: str, kwargs: Any, id: str) -> Any:
    return NS(tool_name=name, tool_kwargs=kwargs, tool_id=id)


def li_tool_result(name: str, kwargs: Any, id: str, text: str, is_error: bool = False) -> Any:
    output = NS(content=text, tool_name=name, raw_input=kwargs, raw_output=text, is_error=is_error)
    return NS(
        tool_name=name, tool_kwargs=kwargs, tool_id=id, tool_output=output, return_direct=False
    )


def li_weather() -> list[Any]:
    kwargs = {"city": "Paris"}
    return [
        NS(input=[NS(content="Weather?")], current_agent_name="agent"),
        NS(delta="", response="", current_agent_name="agent", tool_calls=[]),
        li_tool_call("get_weather", kwargs, "t1"),
        li_tool_call("get_weather", kwargs, "t1"),  # emitted twice
        li_output("", li_selection("get_weather", kwargs, "t1")),
        li_tool_result("get_weather", kwargs, "t1", "20 C, clear"),
        NS(input=[NS(content="Weather?")], current_agent_name="agent"),
        NS(delta="It", response="It", current_agent_name="agent", tool_calls=[]),
        li_output("It is 20 C in Paris."),
    ]


def li_weather_dicts() -> list[dict[str, Any]]:
    kwargs = {"city": "Paris"}
    return [
        {"input": [{"blocks": [{"text": "Weather?"}]}], "current_agent_name": "agent"},
        {"delta": "", "response": "", "current_agent_name": "agent", "tool_calls": []},
        {"tool_name": "get_weather", "tool_kwargs": kwargs, "tool_id": "t1"},
        {
            "response": {"blocks": [{"block_type": "text", "text": ""}]},
            "tool_calls": [{"tool_id": "t1", "tool_name": "get_weather", "tool_kwargs": kwargs}],
            "current_agent_name": "agent",
        },
        {
            "tool_name": "get_weather",
            "tool_kwargs": kwargs,
            "tool_id": "t1",
            "tool_output": {
                "blocks": [{"block_type": "text", "text": "20 C, clear"}],
                "tool_name": "get_weather",
                "is_error": False,
            },
            "return_direct": False,
        },
        {
            "response": {"blocks": [{"block_type": "text", "text": "It is 20 C in Paris."}]},
            "tool_calls": [],
            "current_agent_name": "agent",
        },
    ]


def test_llama_index_call_args_result_reply_deduplicated_by_id() -> None:
    result = llama_index.from_events(li_weather())
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
    passes_checks(result)


def test_llama_index_dicts_match_objects() -> None:
    from_objects = llama_index.from_events(li_weather())
    from_dicts = llama_index.from_events(li_weather_dicts())
    assert from_dicts["output"] == from_objects["output"]
    assert calls_of(from_dicts) == calls_of(from_objects)
    assert trace_of(from_dicts).rounds == 2


def test_llama_index_several_tools_failures_and_a_result_with_no_call_event() -> None:
    events = [
        li_output("", li_selection("a", {"x": 1}, "1"), li_selection("b", {}, "2")),
        li_tool_result("b", {}, "2", "", is_error=True),
        li_tool_result("a", {"x": 1}, "1", "one"),
        li_tool_result("c", {"z": 3}, "3", "three"),  # no ToolCall and no selection saw it
        li_output("done"),
    ]
    a, b, c = calls_of(llama_index.from_events(events))
    assert (a.name, a.result, a.ok) == ("a", "one", True)
    assert (b.name, b.ok, b.error) == ("b", False, "tool error")
    assert (c.name, c.args, c.result) == ("c", {"z": 3}, "three")


def test_llama_index_turns_structured_output_and_agent_chat_response() -> None:
    turn_one = [li_output("", li_selection("a", {}, "1")), li_tool_result("a", {}, "1", "x")]
    turn_one.append(li_output("first"))
    turn_two = [li_output("", structured={"answer": 42})]
    trace = trace_of(llama_index.from_events([turn_one, turn_two]))
    assert trace.responses[0] == "first" and json.loads(trace.responses[1]) == {"answer": 42}
    assert [c.turn for c in trace.tool_calls] == [0]
    chat = NS(
        response="It is 20 C in Paris.",
        sources=[
            NS(
                tool_name="get_weather",
                raw_input={"city": "Paris"},
                content="20 C, clear",
                raw_output="20 C, clear",
                is_error=False,
            ),
            NS(tool_name="broken", raw_input={}, content="", raw_output={"e": 1}, is_error=True),
        ],
    )
    result = llama_index.from_events(chat)
    assert result["output"] == "It is 20 C in Paris."
    ok, failed = calls_of(result)
    assert (ok.name, ok.args, ok.result, ok.ok) == (
        "get_weather",
        {"city": "Paris"},
        "20 C, clear",
        True,
    )
    assert (failed.ok, failed.error) == (False, "tool error")
    assert trace_of(result).rounds is None
    two = trace_of(llama_index.from_events([chat, NS(response="second", sources=[])]))
    assert two.responses == ["It is 20 C in Paris.", "second"]
    assert [c.turn for c in two.tool_calls] == [0, 0]


def test_llama_index_empty_tool_free_and_odd_input() -> None:
    for empty in (None, [], [[]], {}):
        result = llama_index.from_events(empty)
        assert result["output"] == "" and calls_of(result) == []
    plain = llama_index.from_events([li_output("hello")])
    assert plain["output"] == "hello" and trace_of(plain).rounds == 1
    odd = [NS(), {"tool_kwargs": None, "tool_name": None}, NS(response=NS(), tool_calls=None)]
    result = llama_index.from_events(odd)
    assert [c.name for c in calls_of(result)] == ["unknown"]
    assert result["output"] == ""


# --- CrewAI -------------------------------------------------------------------------------------


def crew_call(name: str, args: Any, id: str) -> Any:
    return {"id": id, "type": "function", "function": {"name": name, "arguments": args}}


def crew_weather_task(**extra: Any) -> Any:
    return NS(
        description="weather",
        agent="Analyst",
        raw="It is 20 C in Paris.",
        messages=[
            {"role": "system", "content": "you are an analyst"},
            {"role": "user", "content": "Weather?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [crew_call("get_weather", '{"city": "Paris"}', "t1")],
            },
            {"role": "tool", "tool_call_id": "t1", "name": "get_weather", "content": "20 C, clear"},
            {"role": "assistant", "content": "It is 20 C in Paris."},
        ],
        tool_failures=[],
        **extra,
    )


def crew_weather() -> Any:
    return NS(
        raw="It is 20 C in Paris.",
        tasks_output=[crew_weather_task()],
        token_usage=NS(
            total_tokens=262, prompt_tokens=240, completion_tokens=22, cached_prompt_tokens=100
        ),
    )


def test_crewai_call_args_result_reply_and_usage() -> None:
    result = crewai.from_output(crew_weather())
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
    assert (usage.model, usage.tokens_in, usage.tokens_out, usage.cached_tokens) == (
        None,
        240,
        22,
        100,
    )
    passes_checks(result)


def test_crewai_dicts_task_output_and_turns() -> None:
    task = crew_weather_task()
    as_dict = {
        "raw": "It is 20 C in Paris.",
        "tasks_output": [vars(task)],
        "token_usage": {"prompt_tokens": 240, "completion_tokens": 22, "cached_prompt_tokens": 100},
    }
    assert crewai.from_output(as_dict) == crewai.from_output(crew_weather())
    only_task = crewai.from_output(task)
    assert only_task["output"] == "It is 20 C in Paris." and trace_of(only_task).usage == []
    two = trace_of(crewai.from_output([crew_weather(), crew_weather()]))
    assert two.responses == ["It is 20 C in Paris."] * 2
    assert [c.turn for c in two.tool_calls] == [0, 1]


def test_crewai_several_tasks_tools_in_order_and_failures() -> None:
    first = NS(
        raw="",
        messages=[
            {
                "role": "assistant",
                "tool_calls": [
                    crew_call("a", {"x": 1}, "1"),
                    crew_call("b", "{}", "2"),
                ],
            },
            {"role": "tool", "tool_call_id": "2", "name": "b", "content": "Error: boom"},
            {"role": "tool", "tool_call_id": "1", "name": "a", "content": "one"},
        ],
        tool_failures=[
            NS(tool_name="b", failure=NS(message="boom"), tool_args={}),
            NS(tool_name="never", failure=NS(message="x"), tool_args=None),
        ],
    )
    second = NS(
        raw="final",
        messages=[
            {"role": "assistant", "tool_calls": [crew_call("c", None, "3")]},
            {"role": "tool", "tool_call_id": "3", "name": "c", "content": [{"text": "three"}]},
            {"role": "assistant", "content": "wrapped up"},
        ],
        tool_failures=[],
    )
    result = crewai.from_output(NS(raw="", tasks_output=[first, second], token_usage=None))
    a, b, c = calls_of(result)
    assert [x.name for x in (a, b, c)] == ["a", "b", "c"]
    assert (a.ok, a.result) == (True, "one")
    assert (b.ok, b.error) == (False, "boom")
    assert (c.args, c.result) == ({}, "three")
    assert result["output"] == "final"  # the last task's raw, when the crew output has none
    no_raw = NS(raw="", tasks_output=[NS(raw="", messages=second.messages, tool_failures=[])])
    assert crewai.from_output(no_raw)["output"] == "wrapped up"


def test_crewai_empty_tool_free_and_odd_input() -> None:
    for empty in (None, [], {}):
        result = crewai.from_output(empty)
        assert result["output"] == "" and calls_of(result) == []
    plain = crewai.from_output(NS(raw="hello", tasks_output=[]))
    assert plain["output"] == "hello" and trace_of(plain).rounds is None
    odd = NS(
        raw=None,
        tasks_output=[
            NS(
                raw=None,
                messages=[
                    NS(),
                    {"role": "assistant", "tool_calls": [{"function": None}, None]},
                    {"role": "tool", "content": None},
                ],
                tool_failures=[NS(tool_name=None)],
            )
        ],
    )
    result = crewai.from_output(odd)
    assert [c.name for c in calls_of(result)] == ["unknown", "unknown"]
    assert result["output"] == ""


# --- DSPy ---------------------------------------------------------------------------------------


class FakePrediction:
    """A ``dspy.Prediction``: output fields in ``_store``, read as attributes."""

    def __init__(self, usage: Any = None, **fields: Any) -> None:
        self._store = fields
        self._usage = usage

    def __getattr__(self, name: str) -> Any:
        try:
            return self._store[name]
        except KeyError:
            raise AttributeError(name) from None

    def get_lm_usage(self) -> Any:
        return self._usage


def dspy_history_weather(history_type: Any = NS) -> Any:
    first = {
        "question": "Weather?",
        "next_thought": "I need the weather.",
        "tool_calls": NS(
            tool_calls=[NS(id="call_0_0", name="get_weather", args={"city": "Paris"})],
            tool_call_results=NS(
                tool_call_results=[
                    NS(call_id="call_0_0", name="get_weather", value="20 C, clear", is_error=False)
                ]
            ),
        ),
    }
    last = {
        "question": "Weather?",
        "tool_calls": NS(
            tool_calls=[NS(id="call_1_0", name="submit", args={"answer": "It is 20 C in Paris."})],
            tool_call_results=NS(
                tool_call_results=[
                    NS(call_id="call_1_0", name="submit", value="ok", is_error=False)
                ]
            ),
        ),
    }
    return history_type(messages=[first, last])


def dspy_trajectory_weather() -> dict[str, Any]:
    return {
        "thought_0": "I need the weather.",
        "tool_name_0": "get_weather",
        "tool_args_0": {"city": "Paris"},
        "observation_0": "20 C, clear",
        "thought_1": "Done.",
        "tool_name_1": "finish",
        "tool_args_1": {},
        "observation_1": "Completed.",
    }


USAGE = {"openai/m": {"prompt_tokens": 240, "completion_tokens": 22, "total_tokens": 262}}


def test_dspy_history_shape_call_args_result_reply_and_usage() -> None:
    prediction = FakePrediction(
        usage=USAGE,
        answer="It is 20 C in Paris.",
        history=dspy_history_weather(),
        termination_reason="submit",
    )
    result = dspy.from_prediction(prediction)
    assert result["output"] == "It is 20 C in Paris."
    trace = trace_of(result)
    assert trace.rounds == 2 and trace.responses == ["It is 20 C in Paris."]
    (call,) = trace.tool_calls  # submit is the reserved final-output tool, not a call
    assert (call.name, call.args, call.id, call.result, call.ok) == (
        "get_weather",
        {"city": "Paris"},
        "call_0_0",
        "20 C, clear",
        True,
    )
    (usage,) = trace.usage
    assert (usage.model, usage.tokens_in, usage.tokens_out) == ("openai/m", 240, 22)
    passes_checks(result)


def test_dspy_trajectory_shape_and_dicts() -> None:
    prediction = FakePrediction(
        answer="It is 20 C in Paris.", trajectory=dspy_trajectory_weather(), usage=USAGE
    )
    result = dspy.from_prediction(prediction)
    assert result["output"] == "It is 20 C in Paris."
    (call,) = calls_of(result)  # finish is the reserved last step
    assert (call.name, call.args, call.result, call.ok) == (
        "get_weather",
        {"city": "Paris"},
        "20 C, clear",
        True,
    )
    assert trace_of(result).rounds == 2
    as_dict = {"answer": "It is 20 C in Paris.", "trajectory": dspy_trajectory_weather()}
    from_dict = dspy.from_prediction(as_dict)
    assert from_dict["output"] == result["output"] and calls_of(from_dict) == calls_of(result)
    history_dict = {
        "answer": "x",
        "history": {
            "messages": [
                {
                    "tool_calls": {
                        "tool_calls": [{"id": "1", "name": "a", "args": {"n": 1}}],
                        "tool_call_results": {
                            "tool_call_results": [{"call_id": "1", "name": "a", "value": 5}]
                        },
                    }
                }
            ]
        },
    }
    (call,) = calls_of(dspy.from_prediction(history_dict))
    assert (call.name, call.args, call.result) == ("a", {"n": 1}, 5)


def test_dspy_several_tools_failures_sparse_trajectory_and_output_fields() -> None:
    trajectory = {
        "tool_name_2": "b",
        "tool_args_2": {},
        "observation_2": "Execution error in b: ValueError: boom",
        "tool_name_3": "c",
        "tool_args_3": None,
        "observation_3": {"v": 3},
        "tool_name_4": "d",
        "tool_args_4": {},
    }
    result = dspy.from_prediction({"answer": "x", "trajectory": trajectory})
    b, c, d = calls_of(result)
    assert [x.name for x in (b, c, d)] == ["b", "c", "d"]
    assert (b.ok, b.error) == (False, "Execution error in b: ValueError: boom")
    assert (c.args, c.result, c.ok) == ({}, {"v": 3}, True)
    assert d.result is None  # no observation key: no result yet
    fields = FakePrediction(answer="a", confidence=0.5, history=None, trajectory={})
    assert json.loads(dspy.from_prediction(fields)["output"]) == {"answer": "a", "confidence": 0.5}
    assert dspy.from_prediction(fields, output_field="answer")["output"] == "a"
    assert dspy.from_prediction(fields, output_field="missing")["output"] == ""
    failed = dspy_history_weather()
    failed.messages[0]["tool_calls"].tool_call_results.tool_call_results[0].is_error = True
    (call,) = calls_of(dspy.from_prediction(FakePrediction(answer="x", history=failed)))
    assert (call.ok, call.error) == (False, "20 C, clear")


def test_dspy_turns_and_empty_and_odd_input() -> None:
    first = FakePrediction(answer="one", trajectory=dspy_trajectory_weather())
    second = {"answer": "two"}
    trace = trace_of(dspy.from_prediction([first, None, second]))
    assert trace.responses == ["one", "two"]
    assert [c.turn for c in trace.tool_calls] == [0]
    for empty in (None, {}, [], FakePrediction(), {"history": None, "trajectory": 3}):
        result = dspy.from_prediction(empty)
        assert result["output"] == "" and calls_of(result) == []
    plain = dspy.from_prediction(FakePrediction(answer="hello", usage={"m": {}}))
    assert plain["output"] == "hello" and trace_of(plain).rounds is None
    odd = {"answer": 5, "history": {"messages": [NS(), {"tool_calls": {"tool_calls": [{}]}}]}}
    result = dspy.from_prediction(odd)
    assert result["output"] == "5"
    assert [c.name for c in calls_of(result)] == ["unknown"]


# --- the experimental notice, for all seven new adapters ----------------------------------------

ADAPTERS = [
    ("pydantic_ai", pydantic_ai.from_messages, []),
    ("haystack", haystack.from_messages, []),
    ("autogen", autogen.from_messages, []),
    ("semantic_kernel", semantic_kernel.from_history, []),
    ("llama_index", llama_index.from_events, []),
    ("crewai", crewai.from_output, None),
    ("dspy", dspy.from_prediction, {}),
]


@pytest.mark.parametrize(("name", "adapter", "empty"), ADAPTERS)
def test_experimental_warning_once_per_process_naming_the_caller(
    name: str, adapter: Any, empty: Any
) -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter(empty)
        adapter(empty)
    notices = [w for w in caught if issubclass(w.category, UserWarning)]
    assert len(notices) == 1
    assert f"the {name} adapter is experimental" in str(notices[0].message)
    assert notices[0].filename == __file__  # stacklevel points at the caller of the adapter


@pytest.mark.parametrize(("name", "adapter", "empty"), ADAPTERS)
def test_experimental_warning_can_be_an_error(name: str, adapter: Any, empty: Any) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match=name):
            adapter(empty)
    # the process-wide notice is spent once raised: a later call runs without it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter(empty)


def test_each_adapter_warns_for_itself() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _, adapter, empty in ADAPTERS:
            adapter(empty)
    assert len([w for w in caught if "is experimental" in str(w.message)]) == len(ADAPTERS)
