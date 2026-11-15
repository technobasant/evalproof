"""Shared importer rules added in 0.8: tags, trace kept on collapse, source-order trials, and
the Inspect ToolEvent trace (a real log plus the cases the real log does not contain)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.importers import ImportFormatError, inspect_ai
from evalproof.importers._common import (
    Meta,
    Options,
    Sample,
    build_run,
    in_source_order,
)
from evalproof.models import CaseResult, Score, SystemOutput, ToolCall, Trace

FIXTURES = Path(__file__).parent / "fixtures"
INSPECT_LOG = FIXTURES / "inspect" / "log_message_deduplication.eval"


def _result(
    case_id: str, *, value: float = 1.0, trace: Trace | None = None, output: str = "out"
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        output=SystemOutput(output=output, trace=trace),
        scores={"m": Score(value=value, passed=value >= 0.5)},
    )


def _trace(*names: str) -> Trace:
    return Trace(tool_calls=[ToolCall(name=n) for n in names], rounds=len(names))


def test_inspect_real_log_has_tool_trace() -> None:
    run = import_run("inspect", INSPECT_LOG)
    assert len(run.results) == 1
    trace = run.results[0].output.trace
    assert trace is not None
    assert [c.name for c in trace.tool_calls] == ["bash"] * 5
    first = trace.tool_calls[0]
    assert first.args == {"cmd": "ls"}
    assert first.id == "hardcoded_0"
    assert first.ok is True and first.error is None
    assert first.turn == 0
    assert first.latency_ms is not None and first.latency_ms > 0
    assert trace.rounds == 6
    assert trace.responses == ["done"]
    assert [(u.model, u.tokens_in, u.tokens_out) for u in trace.usage] == [
        ("hardcoded/hardcoded", 0, 0)
    ]


def test_inspect_trace_resolves_attachments_and_marks_errors() -> None:
    told: list[str] = []
    raw: dict[str, Any] = {
        "attachments": {"abc": "the long argument"},
        "events": [
            {
                "event": "tool",
                "id": "c1",
                "function": "search",
                "arguments": {"q": "attachment://abc"},
                "result": "attachment://missing",
                "error": {"type": "unknown", "message": "boom"},
                "working_time": 0.5,
            },
            {"event": "tool", "id": "c2", "function": "fetch", "arguments": {}, "failed": True},
            {"event": "info", "data": "ignored"},
            {
                "event": "model",
                "model": "m1",
                "output": {
                    "usage": {"input_tokens": 3, "output_tokens": 4},
                    "choices": [{"message": {"content": "hi"}}],
                },
            },
        ],
    }
    trace = inspect_ai._trace(raw, lambda key, message: told.append(key))
    assert trace is not None
    search, fetch = trace.tool_calls
    assert search.args == {"q": "the long argument"}
    assert search.result == "attachment://missing"  # unresolved: kept as it is
    assert search.ok is False and search.error == "boom"
    assert search.latency_ms == pytest.approx(500)
    assert fetch.ok is False
    assert trace.rounds == 1
    assert (trace.usage[0].tokens_in, trace.usage[0].tokens_out) == (3, 4)
    assert told == ["attachment"]


def test_inspect_scorer_events_are_not_the_agents() -> None:
    """A model-graded scorer calls a model inside the scorers span; the real log nests a scorer
    span in a scorers span, as here. Those events are the grader's: no round, usage or reply."""

    def model(span: str, text: str, tokens: int) -> dict[str, Any]:
        usage = {"input_tokens": tokens, "output_tokens": tokens}
        return {
            "event": "model",
            "span_id": span,
            "model": "agent" if span == "gen" else "grader",
            "output": {"usage": usage, "choices": [{"message": {"content": text}}]},
        }

    raw: dict[str, Any] = {
        "events": [
            {"event": "span_begin", "id": "solvers", "type": "solvers", "name": "solvers"},
            {"event": "span_begin", "id": "gen", "parent_id": "solvers", "type": "solver"},
            model("gen", "the answer", 5),
            {"event": "tool", "span_id": "gen", "id": "c1", "function": "search", "arguments": {}},
            {"event": "span_begin", "id": "scorers", "type": "scorers", "name": "scorers"},
            {"event": "span_begin", "id": "mg", "parent_id": "scorers", "type": "scorer"},
            {"event": "span_begin", "id": "inner", "parent_id": "mg", "type": "tool"},
            model("mg", "GRADE: C", 50),
            {"event": "tool", "span_id": "inner", "id": "g1", "function": "lookup"},
            {"event": "score", "span_id": "mg", "score": {"value": "C"}},
        ]
    }
    trace = inspect_ai._trace(raw, lambda key, message: None)
    assert trace is not None
    assert [c.name for c in trace.tool_calls] == ["search"]
    assert trace.rounds == 1
    assert trace.responses == ["the answer"]
    assert [(u.model, u.tokens_in) for u in trace.usage] == [("agent", 5)]


def test_inspect_condensed_log_has_no_trace_and_warns() -> None:
    told: list[str] = []
    for raw in ({"events_data": {"messages": []}}, {"input_refs": [[0, 1]]}, {"events": []}):
        assert inspect_ai._trace(raw, lambda key, message: told.append(key)) is None
    assert told == ["condensed", "condensed"]  # a log with no events and no refs says nothing


def test_in_source_order_sorts_by_index_then_position() -> None:
    a, b, c, d = (_result(x) for x in "abcd")
    assert in_source_order([(2, a), (1, b), (3, c)]) == [b, a, c]
    assert in_source_order([(None, a), (None, b)]) == [a, b]
    # a bool is not an index: it takes its file position (0), so it ties with b and stays first
    assert in_source_order([(True, a), (0, b)]) == [a, b]
    assert in_source_order([(1, a), (None, b), (0, c), (1, d)]) == [c, a, b, d]


def test_build_run_sets_tags_without_changing_the_case_hash(tmp_path: Path) -> None:
    def build(tags: list[str]) -> CaseResult:
        sample = Sample("case-1", "question", [_result("case-1")], tags=tags)
        return build_run("jsonl", tmp_path / "f.jsonl", [sample], Meta(), Options()).results[0]

    tagged = build(["Safety", "probe:X", "safety"])
    plain = build([])
    assert tagged.tags == ["safety", "probe:x"]
    assert plain.tags == []
    assert tagged.case_hash == plain.case_hash == tagged.content_hash


@pytest.mark.parametrize("epochs", ["mean", "first", "all", "any"])
def test_build_run_keeps_the_trace_when_repeats_collapse(tmp_path: Path, epochs: str) -> None:
    repeats = [
        _result("c", trace=_trace("first_call")),
        _result("c", trace=_trace("second_call")),
    ]
    sample = Sample("c", "q", repeats)
    run = build_run("jsonl", tmp_path / "f.jsonl", [sample], Meta(), Options(epochs=epochs))
    trace = run.results[0].output.trace
    assert trace is not None
    assert [c.name for c in trace.tool_calls] == ["first_call"]
    assert run.info.environment.get("trace_from") == "first repeat"


def test_build_run_keeps_every_trace_with_trials_and_says_nothing(tmp_path: Path) -> None:
    repeats = [_result("c", trace=_trace("one")), _result("c", trace=_trace("two"))]
    run = build_run(
        "jsonl", tmp_path / "f.jsonl", [Sample("c", "q", repeats)], Meta(), Options(epochs="trials")
    )
    assert [r.output.trace.tool_calls[0].name for r in run.results if r.output.trace] == [
        "one",
        "two",
    ]
    assert "trace_from" not in run.info.environment


def test_threshold_is_for_garak_only(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match="garak"):
        import_run("jsonl", tmp_path / "missing.jsonl", threshold=0.5)
