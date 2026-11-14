# Fixtures for the importers

These files are data, never code. Tests read them and never execute them.

- `promptfoo/sample-export.json`: verbatim from promptfoo, `test/__fixtures__/sample-export.json` (MIT licence). Two providers, two tests, every result a failure.
- `inspect/log_formats.json`: verbatim from Inspect AI, `tests/log/test_eval_log/log_formats.json` (MIT licence). One model, one sample. The `.eval` zip form is built from it inside the tests.
- `deepeval/test_run.json`: synthetic, written to the schema in DeepEval's `test_run/test_run.py` and `api.py`; it has a metric error, a positional `test_case_N` name and a conversational case.
- `lm-eval/<model>/samples_<task>_<time>.jsonl`: synthetic, written to the record shape built in lm-evaluation-harness `evaluator.py` and `loggers/evaluation_tracker.py`; `gpt2` has one filter, `llama3` has two.
- `table/long.csv`, `table/long.jsonl`, `table/wide.csv`: synthetic, for the generic importers (two systems, an error row, an undecided cell).
- `otel/trace.json`: verbatim from open-telemetry/opentelemetry-proto, `examples/trace.json` at commit `94f2f2d5f39c5139141dc7e564daa84aa2826531` (Apache-2.0 licence). One OTLP/JSON span; it shows the encoding (upper-case hex ids, string nanosecond times, kind and status as integers) and carries no agent attributes.
- `otel/genai-content-on.json`, `otel/genai-content-off.json`: synthetic, built from the GenAI semantic conventions (semantic-conventions-genai `docs/gen-ai/execute-tool.md` and `client-inference.md`), as OTLP/JSON. The first has arguments, results and `gen_ai.output.messages`, with its spans in reverse order; the second has none of them, as with `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=NO_CONTENT`.
- `otel/openinference.json`: synthetic, built from the OpenInference semantic conventions 0.1.45 (`spec/semantic_conventions.md`, flattened `llm.output_messages.N.message.content`). Whether a real `TOOL` span sets `input.value` and `output.value` is not verified.
- `otel/adk-schema-1.json`: synthetic, built from Google ADK 2.14.1 `telemetry/tracing.py` (`call_llm`, `gcp.vertex.agent.tool_call_args`, `gcp.vertex.agent.tool_response`). The attributes of the `call_llm` span itself are not verified, so it carries only `gcp.vertex.agent.event_id`.
- `otel/vercel-legacy.json`: synthetic, built from the Vercel AI SDK 7 `LegacyOpenTelemetry` documentation (`ai.toolCall.*`, `ai.usage.promptTokens`, `ai.response.text`).
- `otel/collector.jsonl`: synthetic, the same trace as `genai-content-on.json` in two batches, one OTLP/JSON object per line, as the collector's `file` exporter writes with `format: json`.
- `otel/sdk-console-concatenated.json`: synthetic, the same trace as five pretty-printed objects in a row, in the Python SDK's `span.to_json()` form (`0x` ids, ISO times, flat attributes), as `ConsoleSpanExporter` prints them.
- `otel/two-traces.json`: synthetic, the GenAI trace and a second three-span trace (two traceIds) in one OTLP/JSON object.
