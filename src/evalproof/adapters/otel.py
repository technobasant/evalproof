"""OpenTelemetry spans (experimental): ``from_spans(source)`` turns the spans of one trace into an
evalproof trace.

Experimental because the GenAI semantic conventions are still in Development status and no real
exported span of any agent framework was available to build the fixtures from; they are written
from the specifications (see tests/fixtures/README.md).

Nothing here imports an OpenTelemetry package. It reads JSON and never runs anything from it.

File encodings read (versions and sources as read on 2026-11-14):

* OTLP/JSON: opentelemetry-proto v1.11.1 docs/specification.md:447-492 (keys are lowerCamelCase,
  ``traceId`` and ``spanId`` are case-insensitive hex strings, enums are integers, 64-bit integers
  including ``startTimeUnixNano`` are decimal strings or numbers); common.proto:28-38 ``AnyValue``
  (``stringValue``, ``boolValue``, ``intValue``, ``doubleValue``, ``arrayValue``, ``kvlistValue``,
  ``bytesValue``); trace.proto (commit 94f2f2d5) ``Span.status`` :307, ``Status.message`` :316,
  ``Status.code`` :331, ``STATUS_CODE_ERROR = 2`` :327.
* Collector ``file`` exporter, ``format: json`` with no compression: one OTLP/JSON object per line
  (opentelemetry-collector-contrib v0.165.1 exporter/fileexporter/README.md:117-132). Its proto
  and zstd encodings are binary and are not read.
* Python SDK ``span.to_json()``: opentelemetry-python v1.45.2 opentelemetry-sdk
  ``trace/__init__.py:524-600`` (``context.trace_id`` with a ``0x`` prefix, ``parent_id``, ISO
  ``start_time`` and ``end_time``, ``status.status_code`` and ``status.description``, flat
  ``attributes``). ``ConsoleSpanExporter`` (``trace/export/__init__.py:304``) writes one
  pretty-printed object per span, joined, so a console capture holds several objects in a row.

Attributes read:

* GenAI semantic conventions (github.com/open-telemetry/semantic-conventions-genai, no release
  tag, Status Development): docs/gen-ai/execute-tool.md:72-83 ``gen_ai.operation.name`` =
  ``execute_tool``, ``gen_ai.tool.name``, ``gen_ai.tool.call.id``, ``gen_ai.tool.call.arguments``,
  ``gen_ai.tool.call.result``, ``error.type``; docs/gen-ai/client-inference.md:75-126
  ``gen_ai.operation.name`` (``chat``, ``generate_content``, ``text_completion``),
  ``gen_ai.request.model``, ``gen_ai.response.model``, ``gen_ai.usage.input_tokens``,
  ``gen_ai.usage.output_tokens``, ``gen_ai.output.messages`` (a list of messages, each with
  ``parts``; a text part is ``{"type": "text", "content": ...}``). On spans a structured value
  is a JSON string (semantic-conventions v1.41.0 docs/gen-ai/gen-ai-spans.md:742-744).
  ``gen_ai.usage.prompt_tokens`` and ``completion_tokens`` are the deprecated names
  (semantic-conventions v1.37.0 docs/registry/attributes/gen-ai.md:202-212).
* Content capture, off by default: ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` is
  ``NO_CONTENT`` (default), ``SPAN_ONLY``, ``EVENT_ONLY`` or ``SPAN_AND_EVENT`` (opentelemetry-
  python-contrib util/opentelemetry-util-genai README.rst:34-57). With capture off, the spec's own
  example has an ``execute_tool`` span without arguments and result, and an inference span
  without messages (docs/gen-ai/non-normative/examples-llm-calls.md:345-406).
* Google ADK google-adk v2.14.1 (telemetry modules read on ``main``): schema 2 (default) uses the
  GenAI attributes above and a ``generate_content {model}`` span, and a missing tool id is the
  literal ``"<not specified>"`` (telemetry/tracing.py:314-402, :1256); schema 1
  (``ADK_TELEMETRY_SCHEMA_VERSION_OPT_IN`` = ``1``, supported until at least 2027-03-31,
  telemetry/_schema_version.py:15-47) names its LLM span ``call_llm`` (tracing.py:744) and writes
  the tool's arguments and response to ``gcp.vertex.agent.tool_call_args`` and
  ``gcp.vertex.agent.tool_response`` (tracing.py:350-364, :389-396), the literal ``"{}"`` when
  ``ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS`` is falsy (telemetry/context.py, default on).
* OpenInference python semantic-conventions 0.1.45 (spec/semantic_conventions.md:28-165,
  spec/configuration.md:9-31, spec/tool_calling.md:35-49): ``openinference.span.kind`` (``TOOL``,
  ``LLM``), ``tool.name``, ``tool.id``, ``input.value``, ``output.value``,
  ``llm.output_messages.N.message.content`` (lists are flattened, zero-based),
  ``llm.model_name``, ``llm.token_count.prompt``, ``llm.token_count.completion``. Hidden content
  is the string ``"__REDACTED__"``; ``OPENINFERENCE_HIDE_INPUTS``, ``OPENINFERENCE_HIDE_OUTPUTS``,
  ``OPENINFERENCE_HIDE_INPUT_MESSAGES`` and ``OPENINFERENCE_HIDE_OUTPUT_MESSAGES`` hide it.
* Vercel AI SDK ``ai`` 7.0.173 (docs 03-ai-sdk-core/60-telemetry.mdx): the ``OpenTelemetry``
  integration emits the GenAI attributes above (:628-691, :810-820; arguments only with
  ``recordInputs``, result only with ``recordOutputs`` and only when the call succeeds); the
  ``LegacyOpenTelemetry`` integration emits ``ai.toolCall`` spans with ``ai.toolCall.name``,
  ``ai.toolCall.id``, ``ai.toolCall.args``, ``ai.toolCall.result`` (:1113-1122), ``ai.*.doGenerate``
  and ``ai.*.doStream`` spans with ``ai.usage.promptTokens`` and ``ai.usage.completionTokens``
  (:1063-1074), and ``ai.response.text`` (:933-1123).

What the sources do not say, so what this module does not rely on: that an OpenInference ``TOOL``
span sets ``input.value`` and ``output.value`` (the mapping reads them when they are there), that
hidden OpenInference content is always written as ``"__REDACTED__"`` rather than left out (only
the placeholder is detected), the usage attributes of ADK's ``call_llm`` and ``generate_content``
spans (``gen_ai.usage.*`` is read when it is there), and where ADK schema 1 puts the reply text
(its output is not read). A trace has one turn: no source says where a user turn begins.
"""

from __future__ import annotations

import calendar
import json
import os
import re
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from evalproof.adapters._common import AgentResult, Ledger, get, jsonable
from evalproof.models import ModelUsage, ToolCall, Trace

REDACTED = "__REDACTED__"
NOT_SPECIFIED = "<not specified>"  # what ADK writes for a tool call id it does not have
EXECUTE_TOOL = "execute_tool"
INFERENCE_OPERATIONS = frozenset({"chat", "generate_content", "text_completion"})
_VERCEL_INFERENCE = re.compile(r"^ai\.[A-Za-z]+\.do(?:Generate|Stream)$")
_OI_CONTENT = re.compile(r"^llm\.output_messages\.(\d+)\.message\.content$")

GENAI = "genai-semconv"  # also Google ADK schema 2 and the Vercel OpenTelemetry integration
ADK_1 = "adk-schema-1"
OPENINFERENCE = "openinference"
VERCEL_LEGACY = "vercel-legacy"

_INPUT_TOKENS = (
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.prompt_tokens",
    "llm.token_count.prompt",
    "ai.usage.promptTokens",
)
_OUTPUT_TOKENS = (
    "gen_ai.usage.output_tokens",
    "gen_ai.usage.completion_tokens",
    "llm.token_count.completion",
    "ai.usage.completionTokens",
)
_MODELS = ("gen_ai.response.model", "gen_ai.request.model", "llm.model_name")

# How to turn content capture on again, per convention; the warning names it.
_CONTENT_SETTING = {
    GENAI: (
        "carry no arguments and no result. Set "
        "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=SPAN_ONLY (or SPAN_AND_EVENT) for an "
        "OpenTelemetry GenAI instrumentation or Google ADK schema 2, or recordInputs and "
        "recordOutputs for the Vercel AI SDK"
    ),
    ADK_1: (
        'hold "{}" for the arguments and the response. Set ADK_CAPTURE_MESSAGE_CONTENT_IN_SPANS '
        "to true (ADK schema 1 captures by default, so it was turned off)"
    ),
    VERCEL_LEGACY: (
        "carry no arguments and no result. Set recordInputs and recordOutputs in the call's "
        "telemetry settings"
    ),
    OPENINFERENCE: (
        f'hold "{REDACTED}". Unset OPENINFERENCE_HIDE_INPUTS, OPENINFERENCE_HIDE_OUTPUTS, '
        "OPENINFERENCE_HIDE_INPUT_MESSAGES and OPENINFERENCE_HIDE_OUTPUT_MESSAGES, or the "
        "hide_* options of the OpenInference TraceConfig"
    ),
}

_warned: set[str] = set()
_warn_lock = threading.Lock()


def _warn_experimental() -> None:
    """A ``UserWarning``, once per process, the first time ``from_spans`` runs.

    ``warnings.warn`` and not a line on stderr like the experimental providers print: this runs
    inside the user's own python target, where a warning can be filtered, turned into an error
    (``-W error``) and seen by pytest, and ``stacklevel`` points it at the caller.
    """
    with _warn_lock:
        if "otel" in _warned:
            return
        _warned.add("otel")
    warnings.warn(
        "evalproof: the otel adapter is experimental: the OpenTelemetry GenAI conventions it "
        "reads are in Development status and may change",
        UserWarning,
        stacklevel=3,
    )


# --- reading spans: the three file encodings and the two span encodings -------------------------


@dataclass
class _Span:
    name: str
    attrs: dict[str, Any]
    trace_id: str | None
    start_ns: int | None
    end_ns: int | None
    failed: bool
    message: str | None
    index: int  # position in the input, to keep the order of spans that start together


def _any_value(value: Any) -> Any:
    """An OTLP ``AnyValue``; ``intValue`` is a decimal string in JSON (an int64)."""
    if not isinstance(value, Mapping):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        raw = value["intValue"]
        try:
            return int(raw)
        except (TypeError, ValueError):
            return raw
    if "doubleValue" in value:
        raw = value["doubleValue"]
        try:
            return float(raw)
        except (TypeError, ValueError):
            return raw
    if "boolValue" in value:
        return value["boolValue"]
    if "arrayValue" in value:
        items = get(value["arrayValue"], "values", default=[])
        return [_any_value(v) for v in items] if isinstance(items, list) else []
    if "kvlistValue" in value:
        return _key_values(get(value["kvlistValue"], "values", default=[]))
    if "bytesValue" in value:
        return value["bytesValue"]  # base64 text, kept as it is
    return None


def _key_values(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, Mapping) and "key" in item:
                out[str(item["key"])] = _any_value(item.get("value"))
    return out


def _attributes(raw: Any) -> dict[str, Any]:
    """Attributes as a dict: OTLP ``[{key, value: {...}}]`` or the SDK's flat dict."""
    if isinstance(raw, Mapping):
        return {str(k): v for k, v in raw.items()}
    return _key_values(raw)


_ISO = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})(?:\.(\d+))?(Z|[+-]\d{2}:?\d{2})?$")


def _time_ns(value: Any) -> int | None:
    """Nanoseconds since the epoch from an OTLP time (decimal string or number) or from the
    SDK's ISO 8601 string (UTC when it names no offset)."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return int(value)
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.isdigit():
        return int(text)
    match = _ISO.match(text)
    if match is None:
        return None
    try:
        moment = datetime.strptime(f"{match[1]}T{match[2]}", "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    offset = 0
    zone = match[4]
    if zone and zone != "Z":
        digits = zone[1:].replace(":", "")
        offset = (int(digits[:2]) * 60 + int(digits[2:])) * 60
        offset = -offset if zone[0] == "-" else offset
    seconds = calendar.timegm(moment.timetuple()) - offset
    return seconds * 1_000_000_000 + int((match[3] or "").ljust(9, "0")[:9])


def _hex_id(value: Any) -> str | None:
    """A trace id as lower-case hex: OTLP gives it as hex, the SDK with a ``0x`` prefix."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    return text[2:] if text.startswith("0x") else text


def _failed(status: Any) -> bool:
    """Whether a span status is ERROR: OTLP's integer 2 (or its name), the SDK's ``"ERROR"``."""
    code = get(status, "code", "status_code")
    if isinstance(code, bool):
        return False
    if isinstance(code, int):
        return code == 2
    return isinstance(code, str) and code.strip().upper() in {"2", "ERROR", "STATUS_CODE_ERROR"}


def _span_from(raw: Mapping[str, Any], index: int) -> _Span:
    status = get(raw, "status")
    message = get(status, "message", "description")
    return _Span(
        name=str(get(raw, "name", default="")),
        attrs=_attributes(get(raw, "attributes")),
        trace_id=_hex_id(get(raw, "traceId", "trace_id") or get(get(raw, "context"), "trace_id")),
        start_ns=_time_ns(get(raw, "startTimeUnixNano", "start_time")),
        end_ns=_time_ns(get(raw, "endTimeUnixNano", "end_time")),
        failed=_failed(status),
        message=str(message) if message else None,
        index=index,
    )


def _looks_like_span(value: Mapping[str, Any]) -> bool:
    return "name" in value and any(
        key in value for key in ("attributes", "context", "traceId", "spanId")
    )


def _collect(value: Any, out: list[_Span]) -> None:
    """Add the spans in ``value``: an OTLP export, one span, or a list of either."""
    if isinstance(value, Mapping):
        if "resourceSpans" in value:
            for resource in value["resourceSpans"] or []:
                for scope in get(resource, "scopeSpans", default=[]):
                    for raw in get(scope, "spans", default=[]):
                        if not isinstance(raw, Mapping):
                            raise ValueError("an OTLP span is not a JSON object")
                        out.append(_span_from(raw, len(out)))
            return
        if _looks_like_span(value):
            out.append(_span_from(value, len(out)))
            return
    elif isinstance(value, list):
        for item in value:
            _collect(item, out)
        return
    raise ValueError(
        "not an OTLP export (resourceSpans) and not a span (an object with a name and "
        "attributes or a context)"
    )


def _read(path: Path) -> list[Any]:
    """Every JSON value in a file, one after the other. That reads one OTLP object, an OTLP
    object per line (the collector's file exporter), a JSON array of spans, and the several
    pretty-printed objects ``ConsoleSpanExporter`` writes, which are not one JSON document."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"{path} is not UTF-8 text. The collector's proto and zstd encodings are not read; "
            "export with format json and no compression"
        ) from exc
    decoder = json.JSONDecoder()
    values: list[Any] = []
    pos = 0
    while True:
        while pos < len(text) and text[pos].isspace():
            pos += 1
        if pos >= len(text):
            return values
        try:
            value, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} is not JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}"
            ) from exc
        values.append(value)


def _select(spans: list[_Span], trace_id: str | None) -> tuple[list[_Span], str | None]:
    found = list(dict.fromkeys(s.trace_id for s in spans))
    shown = ", ".join(i if i is not None else "(no trace id)" for i in found[:10])
    shown += ", ..." if len(found) > 10 else ""
    if trace_id is not None:
        wanted = _hex_id(trace_id)
        picked = [s for s in spans if s.trace_id == wanted]
        if not picked:
            raise ValueError(f"trace_id {trace_id!r} is not in the spans; they hold: {shown}")
        return picked, wanted
    if len(found) > 1:
        raise ValueError(
            f"the spans belong to {len(found)} traces ({shown}); pass trace_id= to pick one"
        )
    return spans, found[0]


# --- what a span is ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    """A non-empty string, or None."""
    return value if isinstance(value, str) and value else None


def _number(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _first(attrs: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if attrs.get(name) is not None:
            return attrs[name]
    return None


def _decoded(value: Any) -> Any:
    """A JSON object or array held in a string, as the value it holds; anything else as it is."""
    if isinstance(value, str) and value.lstrip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except ValueError:
            return value
    return value


@dataclass
class _Tool:
    convention: str
    name: Any
    args: Any
    result: Any
    id: Any
    content_off: bool


def _tool_of(span: _Span) -> _Tool | None:
    """The tool call a span records, or None when the span is not a tool span."""
    a = span.attrs
    if span.name == "ai.toolCall" or a.get("ai.operationId") == "ai.toolCall":
        args, result = _text(a.get("ai.toolCall.args")), _text(a.get("ai.toolCall.result"))
        return _Tool(
            VERCEL_LEGACY,
            a.get("ai.toolCall.name"),
            args,
            result,
            a.get("ai.toolCall.id"),
            args is None and result is None,
        )
    if a.get("openinference.span.kind") == "TOOL":
        args, result = a.get("input.value"), a.get("output.value")
        hidden = REDACTED in (args, result)
        return _Tool(
            OPENINFERENCE,
            a.get("tool.name"),
            None if args == REDACTED else args,
            None if result == REDACTED else result,
            a.get("tool.id"),
            hidden,
        )
    if a.get("gen_ai.operation.name") == EXECUTE_TOOL:
        # ADK schema 1 sets both gcp.vertex.agent attributes on every tool span; schema 2 sets
        # neither (it shares gcp.vertex.agent.event_id, so that one says nothing).
        legacy = "gcp.vertex.agent.tool_call_args" in a or "gcp.vertex.agent.tool_response" in a
        if legacy and "gen_ai.tool.call.arguments" not in a and "gen_ai.tool.call.result" not in a:
            args = a.get("gcp.vertex.agent.tool_call_args")
            result = a.get("gcp.vertex.agent.tool_response")
            return _Tool(
                ADK_1,
                a.get("gen_ai.tool.name"),
                args,
                result,
                a.get("gen_ai.tool.call.id"),
                args in (None, "", "{}") and result in (None, "", "{}"),
            )
        args, result = a.get("gen_ai.tool.call.arguments"), a.get("gen_ai.tool.call.result")
        return _Tool(
            GENAI,
            a.get("gen_ai.tool.name"),
            args,
            result,
            a.get("gen_ai.tool.call.id"),
            args in (None, "") and result in (None, ""),
        )
    return None


def _inference_of(span: _Span) -> str | None:
    """The convention of an inference (model call) span, or None."""
    a = span.attrs
    if a.get("openinference.span.kind") == "LLM":
        return OPENINFERENCE
    if _VERCEL_INFERENCE.match(span.name) or _VERCEL_INFERENCE.match(
        str(a.get("ai.operationId", ""))
    ):
        return VERCEL_LEGACY
    if span.name == "call_llm":
        return ADK_1
    if a.get("gen_ai.operation.name") in INFERENCE_OPERATIONS:
        return GENAI
    return None


def _tool_call(ledger: Ledger, span: _Span, tool: _Tool) -> ToolCall:
    ident = tool.id if tool.id not in (None, "", NOT_SPECIFIED) else None
    entry = ledger.call(tool.name, tool.args, ident, 0)
    if tool.result is not None:
        entry.result = jsonable(_decoded(tool.result))
    error_type = span.attrs.get("error.type")
    if error_type is not None or span.failed:
        entry.ok = False
        parts = [str(p) for p in (error_type, span.message) if p]
        entry.error = ": ".join(dict.fromkeys(parts)) or "error"
    if span.start_ns is not None and span.end_ns is not None and span.end_ns >= span.start_ns:
        entry.latency_ms = (span.end_ns - span.start_ns) / 1_000_000
    return entry


def _message_text(messages: Any) -> str | None:
    """The text parts of the first of ``gen_ai.output.messages`` that has any."""
    messages = _decoded(messages)
    if not isinstance(messages, list):
        return None
    for message in messages:
        parts = get(message, "parts")
        if not isinstance(parts, list):
            continue
        text = "".join(
            c for p in parts if get(p, "type") == "text" and isinstance(c := get(p, "content"), str)
        )
        if text:
            return text
    return None


def _output_of(span: _Span) -> tuple[str | None, bool]:
    """The reply an inference span recorded, and whether OpenInference hid it."""
    a = span.attrs
    text = _message_text(a.get("gen_ai.output.messages"))
    if text:
        return text, False
    indexed = sorted(
        (int(m[1]), v)
        for k, v in a.items()
        if (m := _OI_CONTENT.match(k)) and isinstance(v, str) and v
    )
    if indexed:
        value = indexed[0][1]
        return (None, True) if value == REDACTED else (value, False)
    return _text(a.get("ai.response.text")), False


# --- the adapter ------------------------------------------------------------------------------


def from_spans(
    source: str | os.PathLike[str] | Mapping[str, Any] | list[Any],
    *,
    trace_id: str | None = None,
) -> AgentResult:
    """Build ``{"output", "trace"}`` from the spans of one trace (experimental).

    ``source`` is a path to a ``.json`` or ``.jsonl`` file, an OTLP/JSON object
    (``resourceSpans``), one span or a list of spans, as OTLP spans or as the Python SDK's
    ``span.to_json()`` objects. A file may hold one OTLP object, one per line (the collector's
    file exporter), or several objects in a row (``ConsoleSpanExporter``). Spans are ordered by
    start time. When the spans belong to more than one trace, ``trace_id`` picks one (hex, with
    or without ``0x``) and ``ValueError`` is raised without it.

    Tool calls come from tool spans of GenAI semantic conventions (also Google ADK schema 2 and
    the Vercel ``OpenTelemetry`` integration), Google ADK schema 1, OpenInference and the Vercel
    legacy ``ai.toolCall`` spans; an error type or an ERROR status makes a call fail. ``rounds``
    counts inference spans, usage is summed per model, and the reply is the text of the last
    inference span. A tool span exported with content capture off has no arguments or result: the
    call is kept, a ``UserWarning`` names the setting to turn on, and ``trace.meta
    ["content_captured"]`` is ``False``. Every call is turn 0.
    """
    _warn_experimental()
    values = _read(Path(source)) if isinstance(source, str | os.PathLike) else [source]
    spans: list[_Span] = []
    for value in values:
        _collect(value, spans)
    if not spans:
        raise ValueError("no spans found")
    picked, chosen = _select(spans, trace_id)
    picked.sort(key=lambda s: (s.start_ns is None, s.start_ns or 0, s.index))

    ledger = Ledger()
    conventions: set[str] = set()
    hidden: dict[str, int] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    last: _Span | None = None
    last_convention = GENAI
    for span in picked:
        tool = _tool_of(span)
        if tool is not None:
            conventions.add(tool.convention)
            _tool_call(ledger, span, tool)
            if tool.content_off:
                hidden[tool.convention] = hidden.get(tool.convention, 0) + 1
            continue
        convention = _inference_of(span)
        if convention is None:
            continue
        conventions.add(convention)
        rounds += 1
        last, last_convention = span, convention
        tokens_in = _number(_first(span.attrs, _INPUT_TOKENS))
        tokens_out = _number(_first(span.attrs, _OUTPUT_TOKENS))
        if tokens_in is not None or tokens_out is not None:
            model = _text(_first(span.attrs, _MODELS))
            entry = usage.setdefault(model, ModelUsage(role="agent", model=model))
            entry.tokens_in += tokens_in or 0
            entry.tokens_out += tokens_out or 0

    output = ""
    if last is not None:
        text, redacted = _output_of(last)
        output = text or ""
        if redacted:
            hidden[last_convention] = hidden.get(last_convention, 0) + 1

    meta: dict[str, Any] = {"conventions": sorted(conventions)}
    if chosen is not None:
        meta["trace_id"] = chosen
    if hidden:
        meta["content_captured"] = False
        for convention, count in hidden.items():
            warnings.warn(
                f"evalproof otel adapter: {count} span(s) {_CONTENT_SETTING[convention]}. "
                "Export the spans again; checks on tool arguments, tool results or the reply "
                "fail until then.",
                UserWarning,
                stacklevel=2,
            )
    trace = Trace(
        responses=[output] if output else [],
        tool_calls=ledger.calls,
        rounds=rounds or None,
        usage=list(usage.values()),
        meta=meta,
    )
    return {"output": output, "trace": trace}


__all__ = ["from_spans"]
