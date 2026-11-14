"""ADK eval sets as a dataset: ``evalproof import-dataset adk``.

The real files are verbatim from google/adk-python v2.14.1 (see ``tests/fixtures/README.md``);
``synthetic-mixed.evalset.json`` is invented, for the fields the real ones do not use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.cli import main
from evalproof.config import GenerationParams
from evalproof.dataset import load_dataset, write_dataset
from evalproof.importers import DATASET_FORMATS, ImportFormatError, import_dataset
from evalproof.models import Case, SystemOutput, ToolCall, Trace
from evalproof.scorers import ChecksScorer, RubricSetJudge
from tests.conftest import ScriptedProvider

FIXTURES = Path(__file__).parent / "fixtures" / "adk"
BASIC = FIXTURES / "basic_criteria.home_automation.evalset.json"
RUBRIC = FIXTURES / "rubric_criteria.home_automation.evalset.json"
MULTI = FIXTURES / "test_file_vs_evalset.multi_session.evalset.json"
SINGLE = FIXTURES / "test_file_vs_evalset.single_turn.test.json"
LEGACY = FIXTURES / "home_automation_agent.simple_test2.test.json"
USERSIM = FIXTURES / "user_simulation.home_automation.evalset.json"
SYNTHETIC = FIXTURES / "synthetic-mixed.evalset.json"
RESULT = FIXTURES / "synthetic.evalset_result.json"


def read(path: Path) -> tuple[list[Case], list[str]]:
    warnings: list[str] = []
    return import_dataset("adk", path, warn=warnings.append), warnings


def by_id(cases: list[Case]) -> dict[str, Case]:
    return {case.id: case for case in cases}


def write(tmp_path: Path, data: Any, name: str = "set.evalset.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def invocation(text: str = "hi", *, uses: Any = None, final: str | None = None) -> dict[str, Any]:
    item: dict[str, Any] = {"user_content": {"parts": [{"text": text}], "role": "user"}}
    if final is not None:
        item["final_response"] = {"parts": [{"text": final}], "role": "model"}
    if uses is not None:
        item["intermediate_data"] = {"tool_uses": uses}
    return item


def eval_set(*cases: dict[str, Any]) -> dict[str, Any]:
    return {"eval_set_id": "s", "eval_cases": list(cases)}


def fails(path: Path, *needles: str) -> str:
    with pytest.raises(ImportFormatError) as raised:
        read(path)
    message = str(raised.value)
    for needle in needles:
        assert needle in message
    return message


# --- the real files ---------------------------------------------------------------------------


def test_the_basic_eval_set_is_two_single_turn_cases() -> None:
    cases, warnings = read(BASIC)
    assert warnings == []
    assert [c.id for c in cases] == ["turn_off_bedroom_device", "get_living_room_temperature"]
    first = cases[0]
    assert first.input == "Turn off device_2."
    assert first.reference == "device_2 is now turned off."
    assert first.field("turns") is None
    assert first.field("state") is None
    assert first.field("rubrics") is None
    assert first.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [
                {"name": "set_device_info", "args": {"device_id": "device_2", "status": "OFF"}}
            ],
        }
    ]


def test_the_rubric_criteria_eval_set_carries_no_rubrics() -> None:
    # the real file keeps its rubrics in eval_config.json, which is another file
    (case,) = read(RUBRIC)[0]
    assert case.id == "list_bedroom_devices"
    assert case.field("rubrics") is None
    assert case.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [{"name": "list_devices", "args": {"location": "Bedroom"}}],
        }
    ]


def test_a_multi_turn_case_is_an_input_and_turns_with_one_check_per_invocation() -> None:
    cases, _ = read(MULTI)
    case = by_id(cases)["list_then_turn_off"]
    assert case.input == "Which devices are on?"
    assert case.field("turns") == ["Turn that one off."]
    # the reference is the last invocation's final_response
    assert case.reference == "I have turned off the Living Room device (device_1)."
    assert case.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [{"name": "list_devices", "args": {"status": "ON"}}],
        },
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 1,
            "calls": [
                {"name": "set_device_info", "args": {"device_id": "device_1", "status": "OFF"}}
            ],
        },
    ]
    single = by_id(cases)["set_bedroom_temperature"]
    assert single.field("turns") is None


def test_the_single_turn_test_file_is_an_eval_set_too() -> None:
    # single_turn.test.json is named like the legacy files but holds an EvalSet
    (case,) = read(SINGLE)[0]
    assert case.id == "kitchen_temperature"
    assert case.input == "What's the temperature in the Kitchen?"
    assert case.reference == "The temperature in the Kitchen is 24 degrees Celsius."


def test_the_legacy_flat_list_is_read() -> None:
    (case,) = read(LEGACY)[0]
    assert case.id == "home_automation_agent.simple_test2.test-0"  # no eval_id: file stem, index
    assert case.input == "Turn off device_3 in the Bedroom."
    assert case.reference == "I have set the device_3 status to off."
    assert case.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [
                {
                    "name": "set_device_info",
                    "args": {"location": "Bedroom", "device_id": "device_3", "status": "OFF"},
                }
            ],
        }
    ]


def test_a_legacy_list_without_expected_tools_or_reference(tmp_path: Path) -> None:
    path = write(
        tmp_path, [{"query": "hello"}, {"query": "bye", "expected_tool_use": []}], "x.json"
    )
    cases, _ = read(path)
    assert [c.id for c in cases] == ["x-0", "x-1"]
    assert cases[0].reference is None
    assert cases[0].field("expect") is None


# --- the synthetic file: every field the real ones leave out -----------------------------------


def test_state_rubrics_and_event_form_calls_map_to_case_fields() -> None:
    cases, _ = read(SYNTHETIC)
    refund = by_id(cases)["refund_then_status"]
    assert refund.field("state") == {"customer_tier": "gold"}
    assert refund.field("turns") == ["Is it done?"]
    assert refund.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [
                {"name": "lookup_order", "args": {"order_id": 1042}},
                {"name": "issue_refund", "args": {"order_id": 1042, "reason": "customer request"}},
            ],
        }
    ]  # the second invocation expects no call, so it has no check
    # the case's rubrics, then the invocation's; type only when the source has one
    assert refund.field("rubrics") == [
        {"id": "polite", "text": "The reply is polite."},
        {
            "id": "no_card_number",
            "text": "The agent does not repeat a card number.",
            "type": "TOOL_USE_QUALITY",
        },
        {
            "id": "confirms_order",
            "text": "The reply names the order that was refunded.",
            "type": "FINAL_RESPONSE_QUALITY",
        },
    ]
    # camelCase keys, the invocation_events form and an empty state
    weather = by_id(cases)["weather_in_chicago"]
    assert weather.input == "Weather in chicago?"
    assert weather.reference == "It is clear in Chicago, 24 degrees Celsius."
    assert weather.field("state") is None
    assert weather.field("expect") == [
        {
            "type": "tool_order",
            "match": "exact",
            "turn": 0,
            "calls": [{"name": "get_weather", "args": {"city": "chicago"}}],
        }
    ]


def test_a_call_with_no_arguments_has_no_args_key(tmp_path: Path) -> None:
    path = write(
        tmp_path, eval_set({"eval_id": "a", "conversation": [invocation(uses=[{"name": "ping"}])]})
    )
    (case,) = read(path)[0]
    assert case.field("expect") == [
        {"type": "tool_order", "match": "exact", "turn": 0, "calls": [{"name": "ping"}]}
    ]


# --- the produced checks and rubrics do their job ----------------------------------------------


def trace(*calls: ToolCall) -> SystemOutput:
    return SystemOutput(output="ok", trace=Trace(responses=["ok"], tool_calls=list(calls)))


def test_the_tool_order_check_passes_on_a_matching_trace() -> None:
    cases, _ = read(MULTI)
    case = by_id(cases)["list_then_turn_off"]
    scorer = ChecksScorer("tools")
    matching = trace(
        ToolCall(name="list_devices", args={"status": "ON"}, turn=0),
        ToolCall(name="set_device_info", args={"device_id": "device_1", "status": "OFF"}, turn=1),
    )
    score = scorer.score(case, matching)
    assert score.passed is True, score.reason
    assert score.value == 1.0


def test_the_tool_order_check_fails_on_a_wrong_call_a_wrong_turn_or_no_trace() -> None:
    cases, _ = read(MULTI)
    case = by_id(cases)["list_then_turn_off"]
    scorer = ChecksScorer("tools")
    wrong_args = trace(
        ToolCall(name="list_devices", args={"status": "ON"}, turn=0),
        ToolCall(name="set_device_info", args={"device_id": "device_2", "status": "OFF"}, turn=1),
    )
    assert scorer.score(case, wrong_args).passed is False
    wrong_turn = trace(
        ToolCall(name="list_devices", args={"status": "ON"}, turn=1),
        ToolCall(name="set_device_info", args={"device_id": "device_1", "status": "OFF"}, turn=1),
    )
    assert scorer.score(case, wrong_turn).passed is False
    extra = trace(
        ToolCall(name="list_devices", args={"status": "ON"}, turn=0),
        ToolCall(name="get_temperature", turn=0),
        ToolCall(name="set_device_info", args={"device_id": "device_1", "status": "OFF"}, turn=1),
    )
    assert scorer.score(case, extra).passed is False  # exact: no call in between
    no_trace = scorer.score(case, SystemOutput(output="ok"))
    assert no_trace.passed is False
    assert "no trace" in (no_trace.reason or "")


def test_the_legacy_list_check_passes_on_a_matching_trace() -> None:
    (case,) = read(LEGACY)[0]
    call = ToolCall(
        name="set_device_info",
        args={"location": "Bedroom", "device_id": "device_3", "status": "OFF"},
    )
    assert ChecksScorer("tools").score(case, trace(call)).passed is True


def test_case_rubrics_reach_a_rubric_set_judge() -> None:
    cases, _ = read(SYNTHETIC)
    refund = by_id(cases)["refund_then_status"]
    judge = RubricSetJudge(
        "rs",
        ScriptedProvider([]),
        GenerationParams(),
        rubrics=[{"id": "polite", "text": "Be very polite.", "weight": 2}, "Be brief."],
    )
    rubrics = judge.rubrics_for(refund)
    # the base set first; a case rubric with a base id replaces it; the rest follow
    assert [(r.id, r.text) for r in rubrics] == [
        ("polite", "The reply is polite."),
        ("r2", "Be brief."),
        ("no_card_number", "The agent does not repeat a card number."),
        ("confirms_order", "The reply names the order that was refunded."),
    ]
    assert {r.id: r.type for r in rubrics} == {
        "polite": None,
        "r2": None,
        "no_card_number": "TOOL_USE_QUALITY",
        "confirms_order": "FINAL_RESPONSE_QUALITY",
    }
    only_quality = RubricSetJudge(
        "rs",
        ScriptedProvider([]),
        GenerationParams(),
        rubric_types=["FINAL_RESPONSE_QUALITY"],
    )
    assert [r.id for r in only_quality.rubrics_for(refund)] == ["polite", "confirms_order"]
    assert (
        RubricSetJudge("rs", ScriptedProvider([]), GenerationParams()).rubrics_for(
            by_id(cases)["weather_in_chicago"]
        )
        == []
    )


# --- skipped, empty and unsupported files ------------------------------------------------------


def test_scenario_cases_are_skipped_under_one_warning() -> None:
    cases, warnings = read(SYNTHETIC)
    assert [c.id for c in cases] == ["refund_then_status", "weather_in_chicago"]
    assert len(warnings) == 1
    assert "skipped 1 case(s) that use conversation_scenario" in warnings[0]
    assert "simulated_novice" in warnings[0]
    assert str(SYNTHETIC) in warnings[0]


def test_one_warning_names_every_skipped_case(tmp_path: Path) -> None:
    scenario = {"starting_prompt": "hi", "conversation_plan": "p"}
    path = write(
        tmp_path,
        eval_set(
            {"eval_id": "keep", "conversation": [invocation()]},
            {"eval_id": "sim_a", "conversation_scenario": scenario},
            {"evalId": "sim_b", "conversationScenario": scenario},
        ),
    )
    cases, warnings = read(path)
    assert [c.id for c in cases] == ["keep"]
    assert len(warnings) == 1
    assert "skipped 2 case(s)" in warnings[0]
    assert "sim_a, sim_b" in warnings[0]


def test_a_file_of_only_scenario_cases_is_refused() -> None:
    warnings: list[str] = []
    with pytest.raises(ImportFormatError, match="every eval case uses conversation_scenario"):
        import_dataset("adk", USERSIM, warn=warnings.append)
    assert len(warnings) == 1  # the skipped cases are still named
    assert "comfortable_bedroom_novice, audit_devices_expert" in warnings[0]


@pytest.mark.parametrize(
    ("data", "needle"),
    [([], "the list has no cases"), ({"eval_set_id": "s", "eval_cases": []}, "has no eval_cases")],
    ids=["list", "eval_cases"],
)
def test_a_file_with_no_cases_is_refused(tmp_path: Path, data: Any, needle: str) -> None:
    fails(write(tmp_path, data), needle)


@pytest.mark.parametrize(
    "data",
    [
        {"name": "t", "data": [{"query": "q"}], "initial_session": {}},
        {"foo": 1},
        {"eval_cases": "nope"},
        "text",
        3,
        None,
    ],
    ids=[
        "name-data-initial_session",
        "object",
        "eval_cases-not-a-list",
        "string",
        "number",
        "null",
    ],
)
def test_an_unknown_shape_is_refused(tmp_path: Path, data: Any) -> None:
    fails(write(tmp_path, data), "not an ADK eval set or test file")


def test_a_result_is_refused_with_a_pointer_to_import_adk() -> None:
    fails(RESULT, "an ADK eval result", "evalproof import adk")


def test_a_camel_case_eval_set_gets_a_hint(tmp_path: Path) -> None:
    fails(write(tmp_path, {"evalSetId": "s", "evalCases": []}), "snake_case")


def test_a_list_item_in_the_old_name_data_shape_is_named(tmp_path: Path) -> None:
    fails(
        write(tmp_path, [{"name": "t", "data": [], "initial_session": {}}]),
        "no query",
        "not supported",
    )


def test_a_file_that_is_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ImportFormatError):
        read(path)


@pytest.mark.parametrize(
    ("data", "needle"),
    [
        ([3], "item 0 is not an object"),
        ([{"query": ""}], "query must be a non-empty string"),
        ([{"query": "q", "expected_tool_use": "x"}], "expected_tool_use must be a list"),
        ([{"query": "q", "expected_tool_use": [{"tool_input": {}}]}], "has no tool_name"),
        (
            [{"query": "q", "expected_tool_use": [{"tool_name": "t", "tool_input": 3}]}],
            "not an object",
        ),
    ],
    ids=["item", "query", "uses", "tool_name", "tool_input"],
)
def test_bad_legacy_items_are_refused(tmp_path: Path, data: Any, needle: str) -> None:
    fails(write(tmp_path, data), needle)


def test_bad_eval_cases_are_refused(tmp_path: Path) -> None:
    scenario = {"starting_prompt": "hi"}
    cases: list[tuple[Any, str]] = [
        (3, "is not an object"),
        ({"conversation": [invocation()]}, "has no eval_id"),
        ({"eval_id": " ", "conversation": [invocation()]}, "has no eval_id"),
        ({"eval_id": "a"}, "no conversation"),
        ({"eval_id": "a", "conversation": []}, "no conversation"),
        ({"eval_id": "a", "conversation": [3]}, "invocation 0 is not an object"),
        (
            {"eval_id": "a", "conversation": [{"user_content": {"parts": []}}]},
            "no user_content text",
        ),
        (
            {"eval_id": "a", "conversation": [invocation()], "conversation_scenario": scenario},
            "both conversation and conversation_scenario",
        ),
        (
            {"eval_id": "a", "conversation": [invocation()], "session_input": {"state": [1]}},
            "state is not an object",
        ),
        (
            {"eval_id": "a", "conversation": [invocation(uses=[{"args": {}}])]},
            "a tool use has no name",
        ),
        (
            {"eval_id": "a", "conversation": [invocation(uses=[{"name": "t", "args": 3}])]},
            "arguments of 't' are not an object",
        ),
    ]
    for case, needle in cases:
        fails(write(tmp_path, eval_set(case)), needle)
    twice = eval_set(
        {"eval_id": "a", "conversation": [invocation()]},
        {"eval_id": "a", "conversation": [invocation()]},
    )
    fails(write(tmp_path, twice), "two eval cases have the eval_id 'a'")


def test_bad_rubrics_are_refused(tmp_path: Path) -> None:
    def rubric(rid: Any = "r", text: Any = "t") -> dict[str, Any]:
        return {"rubric_id": rid, "rubric_content": {"text_property": text}}

    def with_rubrics(
        case_rubrics: list[Any], inv_rubrics: list[Any] | None = None
    ) -> dict[str, Any]:
        item = invocation()
        if inv_rubrics is not None:
            item["rubrics"] = inv_rubrics
        return eval_set({"eval_id": "a", "conversation": [item], "rubrics": case_rubrics})

    fails(write(tmp_path, with_rubrics([rubric(rid=None)])), "has no rubric_id")
    fails(write(tmp_path, with_rubrics([rubric(text=" ")])), "has no rubric_content.text_property")
    fails(
        write(tmp_path, with_rubrics([rubric(text="one")], [rubric(text="two")])),
        "rubric 'r' has two different texts",
    )
    # the same rubric on the case and an invocation is kept once
    (case,) = read(write(tmp_path, with_rubrics([rubric()], [rubric()]), "same.evalset.json"))[0]
    assert case.field("rubrics") == [{"id": "r", "text": "t"}]


def test_a_case_the_dataset_loader_would_refuse_is_an_import_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def refuse(case: Case) -> None:
        raise ValueError(f"case {case.id!r}: invented problem")

    monkeypatch.setattr("evalproof.importers.adk_dataset.validate_case_fields", refuse)
    path = write(tmp_path, eval_set({"eval_id": "a", "conversation": [invocation()]}))
    fails(path, str(path), "case 'a': invented problem")


def test_without_a_warn_function_the_skips_are_python_warnings() -> None:
    with pytest.warns(UserWarning, match="simulated_novice"):
        cases = import_dataset("adk", SYNTHETIC)
    assert len(cases) == 2


# --- the output loads and validates ------------------------------------------------------------


@pytest.mark.parametrize("source", [BASIC, MULTI, SINGLE, LEGACY, RUBRIC, SYNTHETIC])
def test_written_cases_load_back_unchanged(tmp_path: Path, source: Path) -> None:
    cases, _ = read(source)
    out = write_dataset(cases, tmp_path / "nested" / "cases.jsonl")
    assert load_dataset(out) == cases
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(cases)
    assert all(json.loads(line)["id"] for line in lines)


# --- the command -------------------------------------------------------------------------------


def test_the_command_writes_a_dataset_that_loads(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    code = main(["import-dataset", "adk", str(MULTI), "--out", str(out)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.out == ""
    assert captured.err == f"wrote 2 case(s) to {out}\n"
    cases = load_dataset(out)
    assert [c.id for c in cases] == ["list_then_turn_off", "set_bedroom_temperature"]
    assert cases == read(MULTI)[0]


def test_the_command_prints_the_skip_warning_on_stderr(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    assert main(["import-dataset", "adk", str(SYNTHETIC), "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert err.startswith("evalproof: warning: ")
    assert "simulated_novice" in err
    assert err.endswith(f"wrote 2 case(s) to {out}\n")


def test_the_command_replaces_an_existing_out_file(tmp_path: Path) -> None:
    out = tmp_path / "cases.jsonl"
    out.write_text('{"id": "old", "input": "old"}\n', encoding="utf-8")
    assert main(["import-dataset", "adk", str(SINGLE), "--out", str(out)]) == 0
    assert [c.id for c in load_dataset(out)] == ["kitchen_temperature"]  # `import --out` does too


def test_the_command_fails_with_exit_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    assert main(["import-dataset", "adk", str(USERSIM), "--out", str(out)]) == 2
    err = capsys.readouterr().err
    assert "evalproof: error: " in err
    assert "no case to import" in err
    assert not out.exists()


def test_the_command_requires_out_and_a_known_format(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as no_out:
        main(["import-dataset", "adk", str(BASIC)])
    assert no_out.value.code == 2
    with pytest.raises(SystemExit) as bad_format:
        main(["import-dataset", "garak", str(BASIC), "--out", str(tmp_path / "x.jsonl")])
    assert bad_format.value.code == 2
    assert DATASET_FORMATS == ("adk",)


def test_an_unknown_format_is_refused_by_the_function(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match="unknown dataset format 'csv'"):
        import_dataset("csv", BASIC)
