"""The OpenTelemetry adapter: every row of the tool-span table, both span encodings, the file
shapes, content capture off, picking one trace, and the path to a ``checks`` score.

The fixtures in ``fixtures/otel`` are listed in ``fixtures/README.md``: ``trace.json`` is the real
OTLP example, the rest are written from the specifications, not captured from a framework.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import pytest

from evalproof.adapters import AgentResult, otel
from evalproof.models import Case, Trace
from evalproof.scorers import ChecksScorer
from evalproof.systems import PythonSystem

FIXTURES = Path(__file__).parent / "fixtures" / "otel"
TRACE = "5b8efff798038103d269b633813fc60c"
OTHER_TRACE = "0af7651916cd43dd8448eb211c80319c"
T0 = 1_700_000_000_000_000_000  # ns, 2023-11-14T22:13:20Z

pytestmark = pytest.mark.filterwarnings("ignore:evalproof")


@pytest.fixture(autouse=True)
def fresh_experimental_warning() -> None:
    otel._warned.clear()


def trace_of(result: AgentResult) -> Trace:
    trace = result["trace"]
    assert isinstance(trace, Trace)
    return trace


def run(source: Any, **kwargs: Any) -> tuple[AgentResult, list[str]]:
    """The result, and the warnings other than the experimental notice."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = otel.from_spans(source, **kwargs)
    return result, [str(w.message) for w in caught if "is experimental" not in str(w.message)]


def otlp_value(value: Any) -> dict[str, Any]:
    if isinstance(value, bool):
        return {"boolValue": value}
    if isinstance(value, int):
        return {"intValue": str(value)}
    if isinstance(value, float):
        return {"doubleValue": value}
    return {"stringValue": str(value)}


def kv(attributes: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"key": k, "value": otlp_value(v)} for k, v in attributes.items()]


def span(
    name: str,
    attributes: dict[str, Any],
    *,
    start: int | None = 0,
    end: int | None = None,
    trace: str | None = TRACE,
    code: int | str | None = None,
    message: str | None = None,
) -> dict[str, Any]:
    """An OTLP span (times in ms after T0)."""
    out: dict[str, Any] = {"name": name, "attributes": kv(attributes)}
    if trace is not None:
        out["traceId"] = trace.upper()
    if start is not None:
        out["startTimeUnixNano"] = str(T0 + start * 1_000_000)
    if end is not None:
        out["endTimeUnixNano"] = str(T0 + end * 1_000_000)
    if code is not None:
        out["status"] = {"code": code, **({"message": message} if message else {})}
    return out


def wrap(*spans: dict[str, Any]) -> dict[str, Any]:
    return {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}


def tool(name: str, args: str | None = None, result: str | None = None, **extra: Any) -> dict:
    attributes: dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
    if args is not None:
        attributes["gen_ai.tool.call.arguments"] = args
    if result is not None:
        attributes["gen_ai.tool.call.result"] = result
    attributes.update(extra)
    return span(f"execute_tool {name}", attributes)


def fixture(name: str) -> Path:
    return FIXTURES / name


# --- the real OTLP example ---------------------------------------------------------------------


def test_real_otlp_example_is_read_but_has_no_agent_semantics() -> None:
    result, extra = run(fixture("trace.json"))
    trace = trace_of(result)
    assert extra == []
    assert trace.meta == {"conventions": [], "trace_id": TRACE}  # ids are read case-insensitively
    assert (result["output"], trace.tool_calls, trace.rounds, trace.usage) == ("", [], None, [])


# --- GenAI semantic conventions: OTLP file, SDK console file, collector file ---------------------


def test_genai_content_on_otlp_file() -> None:
    result, extra = run(fixture("genai-content-on.json"))
    trace = trace_of(result)
    assert extra == []  # nothing is hidden
    assert result["output"] == "It is 20 C in Paris."
    assert trace.responses == ["It is 20 C in Paris."]
    assert [c.name for c in trace.tool_calls] == ["get_weather", "get_forecast"]  # by start time
    weather, forecast = trace.tool_calls
    assert (weather.args, weather.result, weather.ok, weather.id) == (
        {"city": "Paris"},
        {"temp_c": 20},
        True,
        "call_1",
    )
    assert weather.latency_ms == 200.0 and weather.turn == 0
    assert (forecast.args, forecast.result, forecast.ok) == (
        {"city": "Paris", "days": 3},
        None,
        False,
    )
    assert forecast.error == "TimeoutError: forecast service timed out"
    assert trace.rounds == 2  # the invoke_agent span is neither a tool nor a model call
    assert [(u.model, u.tokens_in, u.tokens_out) for u in trace.usage] == [
        ("gpt-4o-mini-2024-07-18", 320, 55)
    ]
    assert trace.meta == {"conventions": ["genai-semconv"], "trace_id": TRACE}


def test_sdk_console_output_and_collector_file_read_the_same_as_the_otlp_file() -> None:
    expected = run(fixture("genai-content-on.json"))[0]
    for name in ("sdk-console-concatenated.json", "collector.jsonl"):
        got = run(fixture(name))[0]
        assert got["output"] == expected["output"], name
        assert trace_of(got).model_dump() == trace_of(expected).model_dump(), name


def test_sdk_form_decodes_prefix_iso_times_status_and_flat_attributes() -> None:
    spans = [
        {
            "name": "execute_tool lookup",
            "context": {"trace_id": "0x" + TRACE, "span_id": "0x00f067aa0ba902b7"},
            "kind": "SpanKind.INTERNAL",
            "start_time": "2023-11-14T22:13:20.000000Z",
            "end_time": "2023-11-14T22:13:20.250000Z",
            "status": {"status_code": "ERROR", "description": "boom"},
            "attributes": {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "lookup",
                "gen_ai.tool.call.arguments": '{"id": 7}',
            },
        }
    ]
    result, _ = run(spans)
    (call,) = trace_of(result).tool_calls
    assert (call.args, call.ok, call.error, call.latency_ms) == ({"id": 7}, False, "boom", 250.0)
    assert trace_of(result).meta["trace_id"] == TRACE


# --- the other conventions ---------------------------------------------------------------------


def test_openinference_file() -> None:
    result, extra = run(fixture("openinference.json"))
    trace = trace_of(result)
    assert extra == []
    assert result["output"] == "It is 20 C in Paris."
    weather, forecast = trace.tool_calls
    assert (weather.name, weather.args, weather.result, weather.id) == (
        "get_weather",
        {"city": "Paris"},
        {"temp_c": 20},
        "call_1",
    )
    assert (forecast.ok, forecast.error, forecast.result) == (
        False,
        "forecast service timed out",
        None,
    )
    assert trace.rounds == 2
    assert [(u.model, u.tokens_in, u.tokens_out) for u in trace.usage] == [("gpt-4o-mini", 320, 55)]
    assert trace.meta["conventions"] == ["openinference"]


def test_adk_schema_1_file() -> None:
    result, extra = run(fixture("adk-schema-1.json"))
    trace = trace_of(result)
    assert extra == []
    weather, forecast = trace.tool_calls
    assert (weather.args, weather.result, weather.id) == (
        {"city": "Paris"},
        {"temp_c": 20},
        "adk-1",
    )
    assert forecast.id is None  # "<not specified>" is no id
    assert forecast.result == {"days": [18, 19, 21]}
    assert trace.rounds == 2  # call_llm spans
    assert result["output"] == ""  # schema 1 puts the reply somewhere this adapter does not read
    assert trace.meta["conventions"] == ["adk-schema-1"]


def test_adk_merged_tools_span_of_parallel_calls_is_not_a_call() -> None:
    """ADK writes one more execute_tool span when tools ran in parallel (google/adk
    telemetry/tracing.py trace_merged_tool_calls); each call also has its own span."""
    merged = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "(merged tools)",
        "gen_ai.tool.description": "(merged tools)",
        "gen_ai.tool.call.id": "evt-9",
        "gcp.vertex.agent.tool_call_args": "N/A",
        "gcp.vertex.agent.event_id": "evt-9",
        "gcp.vertex.agent.tool_response": "{}",
    }
    result, _ = run(
        wrap(
            tool("a", '{"x": 1}', '{"ok": true}'),
            tool("b", '{"x": 2}', '{"ok": true}'),
            span("execute_tool (merged)", merged, start=5),
        )
    )
    assert [c.name for c in trace_of(result).tool_calls] == ["a", "b"]


def test_adk_schema_2_is_the_genai_convention_even_with_the_adk_event_id() -> None:
    spans = wrap(
        span(
            "generate_content gemini-2.5-flash",
            {
                "gen_ai.operation.name": "generate_content",
                "gen_ai.request.model": "gemini-2.5-flash",
                "gen_ai.usage.input_tokens": 50,
                "gen_ai.usage.output_tokens": 5,
            },
            start=0,
        ),
        span(
            "execute_tool get_weather",
            {
                "gen_ai.operation.name": "execute_tool",
                "gen_ai.tool.name": "get_weather",
                "gen_ai.tool.call.id": "<not specified>",
                "gen_ai.tool.call.arguments": '{"city": "Paris"}',
                "gen_ai.tool.call.result": '{"temp_c": 20}',
                "gcp.vertex.agent.event_id": "evt-1",
            },
            start=10,
        ),
    )
    result, extra = run(spans)
    trace = trace_of(result)
    assert extra == []
    (call,) = trace.tool_calls
    assert (call.id, call.args, call.result) == (None, {"city": "Paris"}, {"temp_c": 20})
    assert trace.meta["conventions"] == ["genai-semconv"]
    assert trace.rounds == 1 and trace.usage[0].model == "gemini-2.5-flash"


def test_vercel_legacy_file() -> None:
    result, extra = run(fixture("vercel-legacy.json"))
    trace = trace_of(result)
    assert extra == []
    assert result["output"] == "It is 20 C in Paris."
    weather, forecast = trace.tool_calls
    assert (weather.name, weather.args, weather.result, weather.id) == (
        "get_weather",
        {"city": "Paris"},
        {"temp_c": 20},
        "call_1",
    )
    assert forecast.ok is False and forecast.error == "forecast service timed out"
    assert trace.rounds == 2
    assert [(u.tokens_in, u.tokens_out) for u in trace.usage] == [(320, 55)]
    assert trace.meta["conventions"] == ["vercel-legacy"]


def test_vercel_legacy_tool_span_is_found_by_operation_id_alone() -> None:
    spans = wrap(
        span(
            "weather lookup",
            {
                "ai.operationId": "ai.toolCall",
                "ai.toolCall.name": "get_weather",
                "ai.toolCall.args": '{"city": "Rome"}',
                "ai.toolCall.result": "sunny",
            },
        )
    )
    (call,) = trace_of(run(spans)[0]).tool_calls
    assert (call.name, call.args, call.result) == ("get_weather", {"city": "Rome"}, "sunny")


def test_vercel_opentelemetry_integration_is_the_genai_convention() -> None:
    spans = wrap(
        span(
            "chat gpt-4o",
            {"gen_ai.operation.name": "chat", "gen_ai.response.model": "gpt-4o"},
            start=0,
        ),
        tool("get_weather", '{"city": "Oslo"}', '{"temp_c": 3}', **{"gen_ai.tool.call.id": "c1"}),
    )
    result, _ = run(spans)
    assert trace_of(result).tool_calls[0].args == {"city": "Oslo"}


def test_a_trace_can_mix_conventions() -> None:
    adk_1 = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "b",
        "gcp.vertex.agent.tool_call_args": '{"x": 1}',
        "gcp.vertex.agent.tool_response": '{"y": 2}',
    }
    spans = wrap(
        span("get_weather", {"openinference.span.kind": "TOOL", "tool.name": "get_weather"}),
        span("execute_tool b", adk_1),
    )
    assert trace_of(run(spans)[0]).meta["conventions"] == ["adk-schema-1", "openinference"]


# --- tool arguments, results, ids and errors ---------------------------------------------------


def test_arguments_that_are_not_a_json_object_are_kept_under_input() -> None:
    spans = wrap(
        span(
            "get_weather",
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "get_weather",
                "input.value": "weather in Paris",
                "output.value": "20 C",
            },
        )
    )
    (call,) = trace_of(run(spans)[0]).tool_calls
    assert call.args == {"input": "weather in Paris"} and call.result == "20 C"


def test_a_result_that_is_json_text_is_decoded_and_other_text_is_not() -> None:
    spans = wrap(
        tool("a", "{}", "[1, 2]", **{"gen_ai.tool.call.id": "1"}),
        tool("b", "{}", "20", **{"gen_ai.tool.call.id": "2"}),
        tool("c", "{}", "{not json", **{"gen_ai.tool.call.id": "3"}),
    )
    results = [c.result for c in trace_of(run(spans)[0]).tool_calls]
    assert results == [[1, 2], "20", "{not json"]


@pytest.mark.parametrize(
    ("error_type", "code", "message", "expected"),
    [
        (None, None, None, None),
        ("ValueError", None, None, "ValueError"),
        (None, 2, "boom", "boom"),
        ("ValueError", 2, "boom", "ValueError: boom"),
        ("boom", 2, "boom", "boom"),
        (None, 2, None, "error"),
        (None, "STATUS_CODE_ERROR", "named", "named"),
        (None, 1, "ok status", None),
        (None, 0, None, None),
    ],
)
def test_error_comes_from_error_type_or_status_error(
    error_type: str | None, code: int | str | None, message: str | None, expected: str | None
) -> None:
    extra = {"error.type": error_type} if error_type else {}
    spans = wrap(tool("t", "{}", "{}", **extra) | {"status": {"code": code, "message": message}})
    (call,) = trace_of(run(spans)[0]).tool_calls
    assert call.error == expected and call.ok is (expected is None)


def test_latency_needs_both_times_in_order() -> None:
    spans = wrap(
        tool("a", "{}", "{}") | {"endTimeUnixNano": str(T0 + 5_000_000)},
        span(
            "execute_tool b",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "b"},
            start=100,
            end=50,
        ),
        span(
            "execute_tool c",
            {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "c"},
            start=None,
        ),
    )
    assert [c.latency_ms for c in trace_of(run(spans)[0]).tool_calls] == [5.0, None, None]


def test_a_tool_span_without_a_name_is_kept_as_unknown() -> None:
    spans = wrap(span("execute_tool", {"gen_ai.operation.name": "execute_tool"}))
    assert trace_of(run(spans)[0]).tool_calls[0].name == "unknown"


# --- rounds, usage, output ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "inference",
    [
        ("chat gpt", {"gen_ai.operation.name": "chat"}),
        ("generate_content g", {"gen_ai.operation.name": "generate_content"}),
        ("text_completion g", {"gen_ai.operation.name": "text_completion"}),
        ("ChatCompletion", {"openinference.span.kind": "LLM"}),
        ("ai.generateText.doGenerate", {}),
        ("ai.streamText.doStream", {}),
        ("llm call", {"ai.operationId": "ai.generateText.doGenerate"}),
        ("call_llm", {}),
    ],
    ids=lambda p: p[0],
)
def test_every_inference_span_counts_as_a_round(inference: tuple[str, dict[str, Any]]) -> None:
    name, attributes = inference
    spans = wrap(span(name, attributes, start=0), span(name, attributes, start=1))
    assert trace_of(run(spans)[0]).rounds == 2


def test_other_spans_are_not_rounds() -> None:
    spans = wrap(
        span("invoke_agent a", {"gen_ai.operation.name": "invoke_agent"}),
        span("embeddings", {"gen_ai.operation.name": "embeddings"}),
        span("http GET", {}),
    )
    assert trace_of(run(spans)[0]).rounds is None


def test_usage_reads_current_and_deprecated_names_and_each_convention() -> None:
    spans = wrap(
        span(
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "req",
                "gen_ai.usage.input_tokens": 10,
                "gen_ai.usage.output_tokens": 1,
                "gen_ai.usage.prompt_tokens": 999,  # the deprecated name loses to the current one
            },
            start=0,
        ),
        span(
            "chat",
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "req",
                "gen_ai.response.model": "resp",
                "gen_ai.usage.prompt_tokens": 20,
                "gen_ai.usage.completion_tokens": 2,
            },
            start=1,
        ),
        span(
            "x",
            {
                "openinference.span.kind": "LLM",
                "llm.model_name": "oi",
                "llm.token_count.prompt": 30,
                "llm.token_count.completion": 3,
            },
            start=2,
        ),
        span(
            "ai.generateText.doGenerate",
            {"ai.usage.promptTokens": 40, "ai.usage.completionTokens": 4},
            start=3,
        ),
        span("chat", {"gen_ai.operation.name": "chat", "gen_ai.usage.output_tokens": 5}, start=4),
        span("chat", {"gen_ai.operation.name": "chat"}, start=5),  # no usage: no entry
    )
    usage = {u.model: (u.tokens_in, u.tokens_out) for u in trace_of(run(spans)[0]).usage}
    assert usage == {"req": (10, 1), "resp": (20, 2), "oi": (30, 3), None: (40, 9)}


@pytest.mark.parametrize("raw", ["120", 120, 120.0])
def test_token_counts_decode_from_decimal_strings_numbers_and_doubles(raw: Any) -> None:
    attributes = {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": raw}
    usage = trace_of(run(wrap(span("chat", attributes)))[0]).usage
    assert usage[0].tokens_in == 120


def test_output_is_the_text_of_the_last_inference_span() -> None:
    messages = (
        '[{"role": "assistant", "parts": [{"type": "tool_call", "name": "t"}]},'
        ' {"role": "assistant", "parts": [{"type": "text", "content": "Hello "},'
        ' {"type": "text", "content": "there"}]}]'
    )
    spans = wrap(
        span(
            "chat", {"gen_ai.operation.name": "chat", "gen_ai.output.messages": messages}, start=0
        ),
        span("chat", {"gen_ai.operation.name": "chat"}, start=1),  # the last has no text
    )
    assert run(spans)[0]["output"] == ""  # it is the last span's text, not the last text there is
    spans = wrap(
        span("chat", {"gen_ai.operation.name": "chat", "gen_ai.output.messages": messages})
    )
    assert run(spans)[0]["output"] == "Hello there"


def test_output_from_flattened_openinference_messages_takes_the_first_content() -> None:
    attributes = {
        "openinference.span.kind": "LLM",
        "llm.output_messages.1.message.content": "second",
        "llm.output_messages.0.message.content": "first",
        "llm.output_messages.0.message.role": "assistant",
    }
    assert run(wrap(span("x", attributes)))[0]["output"] == "first"


def test_output_from_vercel_response_text() -> None:
    attributes = {"ai.operationId": "ai.generateText.doGenerate", "ai.response.text": "hi"}
    assert run(wrap(span("ai.generateText.doGenerate", attributes)))[0]["output"] == "hi"


def test_output_messages_that_are_not_a_list_give_no_output() -> None:
    for value in ("not json", '{"parts": []}', "[1, 2]", '[{"parts": "x"}]'):
        attributes = {"gen_ai.operation.name": "chat", "gen_ai.output.messages": value}
        assert run(wrap(span("chat", attributes)))[0]["output"] == ""


# --- content capture off -----------------------------------------------------------------------


def test_genai_content_off_names_the_setting_and_marks_the_trace() -> None:
    result, extra = run(fixture("genai-content-off.json"))
    trace = trace_of(result)
    (message,) = extra
    assert "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY" in message
    assert "SPAN_AND_EVENT" in message and "recordInputs" in message and "2 span(s)" in message
    assert trace.meta["content_captured"] is False
    assert [(c.name, c.args, c.result) for c in trace.tool_calls] == [
        ("get_weather", {}, None),
        ("get_forecast", {}, None),
    ]
    assert trace.tool_calls[1].ok is False  # the error is still there
    assert (result["output"], trace.rounds) == ("", 2)


def test_content_on_sets_no_content_flag_and_warns_of_nothing() -> None:
    result, extra = run(fixture("genai-content-on.json"))
    assert extra == [] and "content_captured" not in trace_of(result).meta


def test_adk_schema_1_with_capture_off_writes_empty_objects() -> None:
    off = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "get_weather",
        "gcp.vertex.agent.tool_call_args": "{}",
        "gcp.vertex.agent.tool_response": "{}",
    }
    result, extra = run(wrap(span("execute_tool get_weather", off)))
    (message,) = extra
    assert "ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS" in message
    assert trace_of(result).meta["content_captured"] is False
    assert trace_of(result).tool_calls[0].args == {}


def test_adk_schema_1_tool_with_a_real_response_is_not_content_off() -> None:
    on = {
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "ping",
        "gcp.vertex.agent.tool_call_args": "{}",  # a tool that takes nothing
        "gcp.vertex.agent.tool_response": '{"ok": true}',
    }
    result, extra = run(wrap(span("execute_tool ping", on)))
    assert extra == [] and "content_captured" not in trace_of(result).meta


def test_vercel_legacy_with_record_inputs_and_outputs_off() -> None:
    off = {"ai.operationId": "ai.toolCall", "ai.toolCall.name": "get_weather"}
    result, extra = run(wrap(span("ai.toolCall", off)))
    (message,) = extra
    assert "recordInputs" in message and "recordOutputs" in message
    assert trace_of(result).meta["content_captured"] is False
    assert trace_of(result).meta["conventions"] == ["vercel-legacy"]


def test_openinference_redacted_tool_and_reply() -> None:
    spans = wrap(
        span(
            "get_weather",
            {
                "openinference.span.kind": "TOOL",
                "tool.name": "get_weather",
                "input.value": otel.REDACTED,
                "output.value": '{"temp_c": 20}',
            },
            start=0,
        ),
        span(
            "ChatCompletion",
            {
                "openinference.span.kind": "LLM",
                "llm.output_messages.0.message.content": otel.REDACTED,
            },
            start=1,
        ),
    )
    result, extra = run(spans)
    trace = trace_of(result)
    (message,) = extra
    assert (
        "OPENINFERENCE_HIDE_INPUTS" in message and "OPENINFERENCE_HIDE_OUTPUT_MESSAGES" in message
    )
    assert "2 span(s)" in message
    assert trace.meta["content_captured"] is False
    assert result["output"] == ""  # the placeholder is not the reply
    assert trace.tool_calls[0].args == {} and trace.tool_calls[0].result == {"temp_c": 20}


def test_one_warning_per_convention() -> None:
    spans = wrap(
        tool("a"),
        span("ai.toolCall", {"ai.operationId": "ai.toolCall", "ai.toolCall.name": "b"}),
    )
    _, extra = run(spans)
    assert len(extra) == 2
    assert sum("OTEL_INSTRUMENTATION_GENAI" in m for m in extra) == 1
    assert sum("telemetry settings" in m for m in extra) == 1


def test_the_content_warning_points_at_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        otel.from_spans(fixture("genai-content-off.json"))
    assert {Path(w.filename).name for w in caught} == {Path(__file__).name}


# --- the experimental notice -------------------------------------------------------------------


def test_the_experimental_warning_is_shown_once_per_process_and_names_the_caller() -> None:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        otel.from_spans(fixture("genai-content-on.json"))
        otel.from_spans(fixture("openinference.json"))
    notices = [w for w in caught if "is experimental" in str(w.message)]
    assert len(notices) == 1
    assert issubclass(notices[0].category, UserWarning)
    assert "evalproof: the otel adapter is experimental" in str(notices[0].message)
    assert Path(notices[0].filename).name == Path(__file__).name


def test_the_experimental_warning_can_be_turned_into_an_error() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(UserWarning, match="experimental"):
            otel.from_spans(fixture("genai-content-on.json"))


# --- what the source can be --------------------------------------------------------------------


def test_sources_dict_list_single_span_path_string_and_array_file(tmp_path: Path) -> None:
    one = tool("t", "{}", "{}")
    for source in (wrap(one), [one], one, [wrap(one)]):
        assert [c.name for c in trace_of(run(source)[0]).tool_calls] == ["t"]
    path = fixture("genai-content-on.json")
    assert run(str(path))[0]["output"] == run(path)[0]["output"] == "It is 20 C in Paris."
    array = tmp_path / "spans.json"
    array.write_text(
        '[\n  {"name": "x", "attributes": {}}, {"name": "y", "attributes": {}}\n]\n',
        encoding="utf-8",
    )
    assert trace_of(run(array)[0]).tool_calls == []


def test_a_file_with_a_byte_order_mark_blank_lines_and_ndjson(tmp_path: Path) -> None:
    line = '{"resourceSpans": []}'
    spans = wrap(tool("t", "{}", "{}"))
    text = "\ufeff" + "\n\n".join([line, json.dumps(spans), json.dumps(spans)]) + "\n\n"
    path = tmp_path / "batches.jsonl"
    path.write_text(text, encoding="utf-8")
    assert len(trace_of(run(path)[0]).tool_calls) == 2


def test_spans_are_ordered_by_start_time_then_input_order() -> None:
    late = tool("late", "{}", "{}") | {"startTimeUnixNano": str(T0 + 9_000_000)}
    early = tool("early", "{}", "{}") | {"startTimeUnixNano": str(T0 + 1_000_000)}
    untimed_a = span(
        "execute_tool a",
        {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "a"},
        start=None,
    )
    untimed_b = span(
        "execute_tool b",
        {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "b"},
        start=None,
    )
    names = [c.name for c in trace_of(run(wrap(untimed_a, late, untimed_b, early))[0]).tool_calls]
    assert names == ["early", "late", "a", "b"]


# --- decoding AnyValue and times ---------------------------------------------------------------


def test_any_value_decodes_every_otlp_kind() -> None:
    value = {
        "kvlistValue": {
            "values": [
                {"key": "s", "value": {"stringValue": "x"}},
                {"key": "i", "value": {"intValue": "9007199254740993"}},  # beyond a float
                {"key": "d", "value": {"doubleValue": 1.5}},
                {"key": "b", "value": {"boolValue": True}},
                {
                    "key": "a",
                    "value": {"arrayValue": {"values": [{"intValue": "1"}, {"stringValue": "z"}]}},
                },
                {"key": "y", "value": {"bytesValue": "AAEC"}},
                {"key": "n", "value": {}},
                {"key": "bad", "value": {"intValue": "nope"}},
                {"key": "badd", "value": {"doubleValue": "nope"}},
                {"key": "empty", "value": {"arrayValue": {}}},
                {"value": {"stringValue": "no key"}},
            ]
        }
    }
    assert otel._any_value(value) == {
        "s": "x",
        "i": 9007199254740993,
        "d": 1.5,
        "b": True,
        "a": [1, "z"],
        "y": "AAEC",
        "n": None,
        "bad": "nope",
        "badd": "nope",
        "empty": [],
    }
    assert otel._any_value("already plain") == "already plain"
    assert otel._any_value({"kvlistValue": {"values": "x"}}) == {}


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (str(T0), T0),
        (T0, T0),
        (float(T0), T0),
        ("2023-11-14T22:13:20Z", T0),
        ("2023-11-14T22:13:20", T0),
        ("2023-11-14 22:13:20.5+00:00", T0 + 500_000_000),
        ("2023-11-14T22:13:20.123456Z", T0 + 123_456_000),
        ("2023-11-15T03:43:20+05:30", T0),
        ("2023-11-14T14:13:20-0800", T0),
        ("2023-13-45T22:13:20Z", None),
        ("yesterday", None),
        (True, None),
        (None, None),
        ({"a": 1}, None),
    ],
)
def test_times_are_nanoseconds_from_otlp_numbers_or_iso_text(
    raw: Any, expected: int | None
) -> None:
    assert otel._time_ns(raw) == expected


# --- choosing a trace --------------------------------------------------------------------------


def test_two_traces_need_a_trace_id() -> None:
    with pytest.raises(ValueError, match="belong to 2 traces") as info:
        otel.from_spans(fixture("two-traces.json"))
    assert TRACE in str(info.value) and OTHER_TRACE in str(info.value)
    assert "pass trace_id=" in str(info.value)


@pytest.mark.parametrize(
    "wanted", [OTHER_TRACE, OTHER_TRACE.upper(), "0x" + OTHER_TRACE, "0X" + OTHER_TRACE.upper()]
)
def test_trace_id_picks_one_trace_in_any_notation(wanted: str) -> None:
    result, _ = run(fixture("two-traces.json"), trace_id=wanted)
    trace = trace_of(result)
    assert result["output"] == "Order 7 has shipped."
    assert [(c.name, c.args, c.result) for c in trace.tool_calls] == [
        ("lookup_order", {"order_id": 7}, "shipped")
    ]
    assert trace.meta["trace_id"] == OTHER_TRACE
    assert [(u.tokens_in, u.tokens_out) for u in trace.usage] == [(100, 18)]


def test_the_other_trace_is_unaffected() -> None:
    result, _ = run(fixture("two-traces.json"), trace_id=TRACE)
    assert result["output"] == "It is 20 C in Paris."
    assert len(trace_of(result).tool_calls) == 2


def test_an_unknown_trace_id_lists_the_ones_there_are() -> None:
    with pytest.raises(ValueError, match="is not in the spans") as info:
        otel.from_spans(fixture("two-traces.json"), trace_id="ff" * 16)
    assert TRACE in str(info.value)


def test_many_traces_are_listed_with_an_ellipsis() -> None:
    spans = [span("x", {}, trace=f"{i:032x}") for i in range(12)]
    with pytest.raises(ValueError, match="12 traces") as info:
        otel.from_spans(spans)
    assert str(info.value).endswith(", ...); pass trace_id= to pick one")


def test_spans_without_a_trace_id_are_one_trace() -> None:
    result, _ = run([span("execute_tool t", {"gen_ai.operation.name": "execute_tool"}, trace=None)])
    assert "trace_id" not in trace_of(result).meta
    with pytest.raises(ValueError, match=r"\(no trace id\)"):
        otel.from_spans([span("x", {}, trace=None)], trace_id=TRACE)


# --- bad input ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ({"resourceSpans": []}, "no spans found"),
        ([], "no spans found"),
        ({}, "not an OTLP export"),
        ({"name": "no attributes"}, "not an OTLP export"),
        ([1, 2], "not an OTLP export"),
        ({"resourceSpans": [{"scopeSpans": [{"spans": ["x"]}]}]}, "not a JSON object"),
    ],
)
def test_input_that_holds_no_spans_is_refused(source: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        otel.from_spans(source)


def test_an_empty_file_and_broken_json_name_the_file(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="no spans found"):
        otel.from_spans(empty)
    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"resourceSpans": []}\n{"resourceSpans": [\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r"broken\.jsonl is not JSON at line 3"):
        otel.from_spans(broken)


def test_a_binary_file_says_which_encodings_are_not_read(tmp_path: Path) -> None:
    binary = tmp_path / "traces.pb"
    binary.write_bytes(b"\x0a\xff\xfe\xfd\x80")
    with pytest.raises(ValueError, match="proto and zstd encodings are not read"):
        otel.from_spans(binary)


def test_a_missing_file_is_the_usual_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        otel.from_spans(tmp_path / "nope.json")


def test_booleans_are_neither_status_codes_nor_token_counts() -> None:
    flat = {
        "name": "execute_tool t",
        "attributes": {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "t"},
        "status": {"code": True},
    }
    chat = {
        "name": "chat",
        "attributes": {"gen_ai.operation.name": "chat", "gen_ai.usage.input_tokens": True},
    }
    trace = trace_of(run([flat, chat])[0])
    assert trace.tool_calls[0].ok is True and trace.usage == []


# --- the public surface ------------------------------------------------------------------------


def test_from_spans_is_the_only_public_name() -> None:
    assert otel.__all__ == ["from_spans"]


# --- from a spans file to a checks score -------------------------------------------------------

AGENT = """
from evalproof.adapters import otel


def agent(question):
    return otel.from_spans({path!r})
"""


def system_for(tmp_path: Path, spans_file: str) -> Any:
    (tmp_path / "otel_agent.py").write_text(AGENT.format(path=str(fixture(spans_file))), "utf-8")
    return PythonSystem("python:otel_agent:agent", search_path=str(tmp_path))


def test_spans_through_python_system_and_checks(tmp_path: Path) -> None:
    try:
        out = system_for(tmp_path, "genai-content-on.json").run(Case(id="c", input="weather?"))
    finally:
        sys.modules.pop("otel_agent", None)
    assert out.error is None and out.output == "It is 20 C in Paris."
    assert (out.tokens_in, out.tokens_out) == (320, 55)  # rolled up from the spans' usage
    checks = [
        {"type": "tool_called", "name": "get_weather"},
        {"type": "tool_order", "calls": ["get_weather", "get_forecast"], "match": "exact"},
        {"type": "tool_args_contain", "name": "get_weather", "arg": "city", "value": "Paris"},
        {"type": "tool_succeeded", "name": "get_weather"},
        {"type": "output_contains", "text": "20 C"},
        {"type": "rounds", "max": 2},
    ]
    case = Case(id="c", input="q")
    score = ChecksScorer("behaviour", checks=checks).score(case, out)
    assert score.passed is True, score.reason
    for bad in (
        {"type": "tool_order", "calls": ["get_forecast", "get_weather"]},
        {"type": "tool_called", "name": "book"},
        {"type": "tool_succeeded", "name": "get_forecast"},
    ):
        assert ChecksScorer("b", checks=[bad]).score(case, out).passed is False


def test_content_off_spans_keep_the_order_checks_and_fail_the_argument_checks(
    tmp_path: Path,
) -> None:
    try:
        out = system_for(tmp_path, "genai-content-off.json").run(Case(id="c", input="weather?"))
    finally:
        sys.modules.pop("otel_agent", None)
    assert out.error is None and out.trace is not None
    assert out.trace.meta["content_captured"] is False
    case = Case(id="c", input="q")
    order = {"type": "tool_order", "calls": ["get_weather", "get_forecast"]}
    assert ChecksScorer("b", checks=[order]).score(case, out).passed is True
    called = {"type": "tool_called", "name": "get_weather"}
    assert ChecksScorer("b", checks=[called]).score(case, out).passed is True
    args = {"type": "tool_args_contain", "name": "get_weather", "arg": "city", "value": "Paris"}
    assert ChecksScorer("b", checks=[args]).score(case, out).passed is False
