"""The ADK eval_history importer. No real result file exists (see tests/fixtures/README.md), so the
fixtures are synthetic, written to the field names in the ADK source; the one real file is an
eval set, used to check that a dataset is refused."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.cli import main
from evalproof.importers import EXPERIMENTAL_FORMATS, FORMATS, ImportFormatError, _common
from evalproof.importers._common import imported_case_hash
from evalproof.models import Run
from evalproof.runner import load_run

ADK = Path(__file__).parent / "fixtures" / "adk"
PLAIN = ADK / "synthetic.evalset_result.json"
DOUBLE = ADK / "synthetic-legacy-double-encoded.evalset_result.json"
EVAL_SET = ADK / "basic_criteria.home_automation.evalset.json"
TURN_OFF = "turn_off_bedroom_device"
LIVING = "get_living_room_temperature"
SCALE_NOTE = "response_evaluation_score, tool_call_count_v1"


def _quiet(_: str) -> None:
    pass


def _data() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(PLAIN.read_text(encoding="utf-8"))
    return loaded


def _write(tmp_path: Path, data: Any, name: str = "result.evalset_result.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _trials(path: Path = PLAIN) -> dict[tuple[str, int], Any]:
    run = import_run("adk", path, epochs="trials", warn=_quiet)
    return {(r.case_id, r.trial): r for r in run.results}


def _case(data: dict[str, Any], eval_id: str, rep: int) -> dict[str, Any]:
    (found,) = [
        c
        for c in data["eval_case_results"]
        if c["eval_id"] == eval_id and c["repetition_index"] == rep
    ]
    return found


def _without_path(run: Run) -> tuple[Any, Any]:
    info = run.info.model_dump()
    info.pop("dataset_path")
    return info, [r.model_dump() for r in run.results]


def test_adk_is_an_experimental_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "adk" in FORMATS
    assert "adk" in EXPERIMENTAL_FORMATS
    monkeypatch.setattr(_common, "_warned", set())  # the note is once per process
    run = import_run("adk", PLAIN, warn=_quiet)
    import_run("adk", PLAIN, warn=_quiet)
    assert capsys.readouterr().err == "evalproof: the adk importer is experimental\n"
    assert run.info.environment["import_experimental"] == "true"
    assert run.info.environment["imported_from"] == "adk"
    assert run.info.environment["source_schema"] == "adk-eval-set-result"


def test_maps_fields_and_notes_the_scales() -> None:
    notes: list[str] = []
    run = import_run("adk", PLAIN, warn=notes.append)
    assert len(notes) == 1
    assert SCALE_NOTE in notes[0]
    assert "kept as given" in notes[0]
    info = run.info
    assert info.suite == "basic_criteria"
    assert info.system == "basic_criteria"  # no --system: the eval set id
    assert info.started_at == "2026-09-07T21:53:11+00:00"  # creation_timestamp, a time.time()
    assert [r.case_id for r in run.results] == [TURN_OFF, LIVING]
    assert all(r.tags == [] for r in run.results)
    turn_off, living = run.results
    # the repeats collapse by mean: two runs of each case, scores as ADK gave them
    assert turn_off.scores["tool_trajectory_avg_score"].value == pytest.approx(0.5)
    assert turn_off.scores["response_match_score"].value == pytest.approx(0.475)
    assert living.scores["response_match_score"].value == pytest.approx(0.6)
    assert living.scores["response_evaluation_score"].value == pytest.approx(3.0)
    assert living.scores["final_response_match_v2"].value is None
    assert turn_off.case_hash == imported_case_hash(TURN_OFF, "Turn off device_2.")


def test_system_label_comes_from_the_option() -> None:
    assert import_run("adk", PLAIN, system="my-agent", warn=_quiet).info.system == "my-agent"


def test_both_encodings_give_the_same_run() -> None:
    assert isinstance(
        json.loads(DOUBLE.read_text(encoding="utf-8")), str
    )  # the pre-1.22.0 file is a string
    assert isinstance(json.loads(PLAIN.read_text(encoding="utf-8")), dict)
    plain = import_run("adk", PLAIN, warn=_quiet)
    double = import_run("adk", DOUBLE, warn=_quiet)
    assert double.info.dataset_path != plain.info.dataset_path
    assert _without_path(double) == _without_path(plain)


_CAMEL = [
    "eval_set_result_id",
    "eval_set_result_name",
    "eval_set_id",
    "eval_case_results",
    "creation_timestamp",
    "eval_set_file",
    "eval_id",
    "final_eval_status",
    "eval_metric_results",
    "overall_eval_metric_results",
    "eval_metric_result_per_invocation",
    "session_id",
    "session_details",
    "user_id",
    "repetition_index",
    "metric_name",
    "custom_function_path",
    "eval_status",
    "actual_invocation",
    "expected_invocation",
    "invocation_id",
    "user_content",
    "final_response",
    "intermediate_data",
    "invocation_events",
    "tool_uses",
    "tool_responses",
    "intermediate_responses",
    "function_call",
    "function_response",
    "usage_metadata",
    "model_version",
    "prompt_token_count",
    "candidates_token_count",
    "app_name",
    "last_update_time",
]


def _camelled(value: Any) -> Any:
    """The ADK field names as camelCase (the dev server's API); tool arguments stay as they are."""
    if isinstance(value, list):
        return [_camelled(v) for v in value]
    if not isinstance(value, dict):
        return value
    out = {}
    for key, item in value.items():
        head, *rest = key.split("_")
        name = head + "".join(w.capitalize() for w in rest) if key in _CAMEL else key
        out[name] = item if key in ("args", "response", "state") else _camelled(item)
    return out


def test_camel_case_keys_are_read_too(tmp_path: Path) -> None:
    camel = _write(tmp_path, _camelled(_data()))
    assert '"evalCaseResults"' in camel.read_text(encoding="utf-8")
    assert '"device_id"' in camel.read_text(encoding="utf-8")
    assert _without_path(import_run("adk", camel, warn=_quiet)) == _without_path(
        import_run("adk", PLAIN, warn=_quiet)
    )
    assert _without_path(import_run("adk", camel, epochs="trials", warn=_quiet)) == _without_path(
        import_run("adk", PLAIN, epochs="trials", warn=_quiet)
    )


def test_trials_follow_repetition_index_not_file_order() -> None:
    data = _data()
    assert [c["repetition_index"] for c in data["eval_case_results"][:2]] == [1, 0]  # file order
    trials = _trials()
    assert set(trials) == {(TURN_OFF, 0), (TURN_OFF, 1), (LIVING, 0), (LIVING, 1)}
    first, second = trials[(TURN_OFF, 0)], trials[(TURN_OFF, 1)]
    assert first.output.output == "device_2 is now turned off."
    assert first.scores["tool_trajectory_avg_score"].passed is True
    assert second.output.output == "device_2 is on."
    assert second.scores["tool_trajectory_avg_score"].passed is False
    run = import_run("adk", PLAIN, epochs="trials", warn=_quiet)
    assert run.info.trials == 2


def test_repetitions_with_no_index_keep_file_order(tmp_path: Path) -> None:
    data = _data()
    for entry in data["eval_case_results"]:
        del entry["repetition_index"]  # a file written before the field existed
    trials = _trials(_write(tmp_path, data))
    assert trials[(TURN_OFF, 0)].output.output == "device_2 is on."  # the file's first
    assert trials[(TURN_OFF, 1)].output.output == "device_2 is now turned off."


def test_eval_status_decides_passed(tmp_path: Path) -> None:
    trials = _trials()
    steps = trials[(TURN_OFF, 0)].scores
    assert (steps["response_match_score"].value, steps["response_match_score"].passed) == (
        0.75,
        True,
    )
    failed = trials[(LIVING, 1)].scores["response_match_score"]
    assert (failed.value, failed.passed) == (0.3, False)
    # INFORMATIONAL (4): the number is kept and there is no verdict
    info = steps["tool_call_count_v1"]
    assert (info.value, info.passed) == (1.0, None)
    assert info.reason is not None and info.reason.startswith("informational")
    # NOT_EVALUATED (3): neither a value nor a verdict
    skipped = trials[(LIVING, 0)].scores["final_response_match_v2"]
    assert (skipped.value, skipped.passed, skipped.reason) == (None, None, "not evaluated")
    # a score given with status 3 is not a value; a null score with status 1 decides nothing
    data = _data()
    metrics = _case(data, LIVING, 0)["overall_eval_metric_results"]
    next(m for m in metrics if m["metric_name"] == "final_response_match_v2")["score"] = 0.7
    next(m for m in metrics if m["metric_name"] == "response_match_score")["score"] = None
    scores = _trials(_write(tmp_path, data))[(LIVING, 0)].scores
    assert (scores["final_response_match_v2"].value, scores["final_response_match_v2"].passed) == (
        None,
        None,
    )
    assert (scores["response_match_score"].value, scores["response_match_score"].passed) == (
        None,
        None,
    )


@pytest.mark.parametrize("status", [0, 5, "PASSED", None, True, 1.5])
def test_unknown_eval_status_is_refused(tmp_path: Path, status: Any) -> None:
    data = _data()
    _case(data, TURN_OFF, 0)["overall_eval_metric_results"][0]["eval_status"] = status
    with pytest.raises(ImportFormatError, match=r"eval_status is .*not 1 \(passed\)"):
        import_run("adk", _write(tmp_path, data), warn=_quiet)


def test_a_nan_score_is_refused_naming_the_case_and_metric(tmp_path: Path) -> None:
    data = _data()
    _case(data, TURN_OFF, 0)["overall_eval_metric_results"][0]["score"] = float("nan")
    with pytest.raises(ImportFormatError, match=f"{TURN_OFF}.*tool_trajectory_avg_score.*NaN"):
        import_run("adk", _write(tmp_path, data), warn=_quiet)


def test_out_of_range_scores_are_named_even_for_a_custom_metric(tmp_path: Path) -> None:
    data = _data()
    for entry in data["eval_case_results"]:
        entry["overall_eval_metric_results"].append(
            {"metric_name": "my_metric", "score": 12.0, "eval_status": 1}
        )
    notes: list[str] = []
    import_run("adk", _write(tmp_path, data), warn=notes.append)
    assert "my_metric, response_evaluation_score, tool_call_count_v1" in notes[0]


def test_trace_from_the_session_events() -> None:
    trace = _trials()[(TURN_OFF, 0)].output.trace
    assert trace is not None
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.turn, call.ok) == (
        "set_device_info",
        {"device_id": "device_2", "status": "OFF"},
        "call_a1",
        0,
        True,
    )
    assert call.result == {"device_id": "device_2", "status": "OFF"}
    # rounds and tokens exist only in events, not in the invocation's intermediate data
    assert trace.rounds == 2
    (usage,) = trace.usage
    assert (usage.model, usage.tokens_in, usage.tokens_out) == ("gemini-2.5-flash", 470, 27)
    assert trace.responses == ["device_2 is now turned off."]


def test_trace_from_invocation_events_when_there_is_no_session() -> None:
    assert _case(_data(), TURN_OFF, 1)["session_details"] is None
    trace = _trials()[(TURN_OFF, 1)].output.trace
    assert trace is not None
    (call,) = trace.tool_calls
    assert (call.name, call.args, call.id, call.turn) == (
        "get_device_info",
        {"device_id": "device_2"},
        "call_b1",
        0,
    )
    assert call.result == {"device_id": "device_2", "status": "ON"}  # paired by id
    assert trace.rounds is None
    assert trace.responses == ["device_2 is on."]


def test_trace_from_tool_uses_pairs_responses_by_id_and_flags_errors() -> None:
    trials = _trials()
    ok = trials[(LIVING, 0)].output.trace
    assert ok is not None
    assert [(c.name, c.id, c.result, c.ok) for c in ok.tool_calls] == [
        ("get_temperature", "call_c1", {"temperature": 22}, True)
    ]
    failed = trials[(LIVING, 1)].output.trace
    assert failed is not None
    (call,) = failed.tool_calls
    assert (call.ok, call.error, call.result) == (
        False,
        "sensor offline",
        {"error": "sensor offline"},
    )


def test_turn_is_the_invocation_index_and_output_the_last_reply(tmp_path: Path) -> None:
    data = _data()
    entry = _case(data, LIVING, 0)
    first = entry["eval_metric_result_per_invocation"][0]
    second = copy.deepcopy(first)
    second["actual_invocation"]["user_content"]["parts"][0]["text"] = "And the kitchen?"
    second["actual_invocation"]["final_response"] = None  # the agent said nothing this turn
    second["actual_invocation"]["intermediate_data"] = {
        "tool_uses": [{"id": "k1", "name": "get_temperature", "args": {"location": "Kitchen"}}],
        "tool_responses": [
            {"id": "k1", "name": "get_temperature", "response": {"temperature": 19}}
        ],
    }
    entry["eval_metric_result_per_invocation"].append(second)
    result = _trials(_write(tmp_path, data))[(LIVING, 0)]
    assert result.output.output == "The living room is 22 degrees."  # the last reply there is
    assert result.output.trace is not None
    assert [(c.turn, c.args) for c in result.output.trace.tool_calls] == [
        (0, {"location": "Living Room"}),
        (1, {"location": "Kitchen"}),
    ]
    assert result.output.trace.responses == ["The living room is 22 degrees.", ""]
    # the input is the first user turn of the first run
    assert import_run("adk", _write(tmp_path, data), warn=_quiet).results[1].case_hash == (
        imported_case_hash(LIVING, "What is the temperature in the living room?")
    )


def test_no_tool_call_means_no_trace(tmp_path: Path) -> None:
    data = _data()
    session = _case(data, TURN_OFF, 0)["session_details"]
    session["events"] = session["events"][:1]  # only the user's message
    for entry in data["eval_case_results"]:
        for per in entry["eval_metric_result_per_invocation"]:
            per["actual_invocation"]["intermediate_data"] = None
    run = import_run("adk", _write(tmp_path, data), epochs="trials", warn=_quiet)
    assert all(r.output.trace is None for r in run.results)


def test_a_failed_tool_response_in_events_is_a_failed_call(tmp_path: Path) -> None:
    data = _data()
    events = _case(data, TURN_OFF, 0)["session_details"]["events"]
    events[2]["content"]["parts"][0]["function_response"]["response"] = {"error": "offline"}
    trace = _trials(_write(tmp_path, data))[(TURN_OFF, 0)].output.trace
    assert trace is not None
    assert (trace.tool_calls[0].ok, trace.tool_calls[0].error) == (False, "offline")


def test_text_ignores_thought_parts_and_a_case_without_invocations_still_imports(
    tmp_path: Path,
) -> None:
    data = _data()
    parts = _case(data, TURN_OFF, 0)["eval_metric_result_per_invocation"][0]["actual_invocation"][
        "final_response"
    ]["parts"]
    parts.insert(0, {"text": "thinking...", "thought": True})
    for entry in data["eval_case_results"]:
        if entry["eval_id"] == LIVING:
            entry["eval_metric_result_per_invocation"] = []
    trials = _trials(_write(tmp_path, data))
    assert trials[(TURN_OFF, 0)].output.output == "device_2 is now turned off."
    bare = trials[(LIVING, 0)]
    assert bare.output.output == "" and bare.output.trace is None
    assert bare.case_hash == imported_case_hash(LIVING, "")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (json.loads(EVAL_SET.read_text(encoding="utf-8")), "this is an ADK eval set"),
        ({"hello": "world"}, "no eval_case_results list"),
        ([1, 2], "expected a JSON object"),
        ({"eval_case_results": []}, "no eval_case_results"),
        ({"eval_case_results": [{"eval_id": ""}]}, r"eval_case_results\[0\] has no eval_id"),
        ({"eval_case_results": [{"eval_id": "a"}]}, "nothing to score"),
        (
            {"eval_case_results": [{"eval_id": "a", "overall_eval_metric_results": [{}]}]},
            "has no metric_name",
        ),
    ],
)
def test_other_files_are_refused(tmp_path: Path, content: Any, message: str) -> None:
    with pytest.raises(ImportFormatError, match=message):
        import_run("adk", _write(tmp_path, content), warn=_quiet)


def test_the_real_eval_set_is_refused_as_not_a_result() -> None:
    with pytest.raises(ImportFormatError, match="not a result"):
        import_run("adk", EVAL_SET, warn=_quiet)


def test_a_string_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "x.evalset_result.json"
    path.write_text(json.dumps("not json at all"), encoding="utf-8")
    with pytest.raises(ImportFormatError, match="not valid JSON"):
        import_run("adk", path, warn=_quiet)


def test_trials_need_the_same_number_of_runs_for_every_case(tmp_path: Path) -> None:
    data = _data()
    data["eval_case_results"].pop()  # the last run of the second case
    with pytest.raises(ImportFormatError, match="same number of repeats"):
        import_run("adk", _write(tmp_path, data), epochs="trials", warn=_quiet)


def test_table_options_are_refused() -> None:
    with pytest.raises(ImportFormatError, match="apply to the jsonl and csv formats"):
        import_run("adk", PLAIN, wide=True)
    with pytest.raises(ImportFormatError, match="applies to the garak format"):
        import_run("adk", PLAIN, threshold=0.5)


def test_import_on_the_command_line(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    code = main(
        ["import", "adk", str(DOUBLE), "--out", str(out), "--system", "v2", "--epochs", "trials"]
    )
    assert code == 0
    run = load_run(out)
    assert (run.info.system, run.info.trials, len(run.results)) == ("v2", 2, 4)
    assert run.info.environment["import_experimental"] == "true"
