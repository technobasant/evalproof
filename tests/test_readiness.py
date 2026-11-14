"""``evalproof readiness``: which dimensions of quality a suite covers, from the suite and the
dataset alone (and a run file when given)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.cli import main
from evalproof.config import ConfigError, load_config
from evalproof.dataset import load_dataset
from evalproof.models import Trace
from evalproof.readiness import (
    COVERED,
    DIMENSION_NAMES,
    MISSING,
    OUT_OF_SCOPE,
    PARTIAL,
    _cell,
    assess,
    parse_required,
    readiness_json,
    readiness_markdown,
)
from evalproof.runner import load_run, save_run
from tests.conftest import make_run

CHECK_TOOL = {"type": "tool_called", "name": "lookup"}
CHECK_TEXT = {"type": "output_contains", "text": "ok"}


def _write(
    tmp_path: Path,
    scorers: str = "[{name: m, type: exact_match}]",
    *,
    cases: list[dict[str, Any]] | None = None,
    extra: str = "",
) -> Path:
    rows = cases if cases is not None else [{"id": f"c{i}", "input": f"q{i}"} for i in range(4)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in rows), "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: demo\ndataset: cases.jsonl\nsystem: {target: 'ollama:m'}\n"
        f"scorers: {scorers}\n{extra}",
        "utf-8",
    )
    return suite


def _assess(suite: Path, **kw: Any):
    cfg = load_config(suite)
    return assess(cfg, load_dataset(cfg.resolve(cfg.dataset)), **kw)


# --- the table ---------------------------------------------------------------------------------


def test_every_dimension_is_reported_in_order(tmp_path: Path) -> None:
    report = _assess(_write(tmp_path))
    assert tuple(d.name for d in report.dimensions) == DIMENSION_NAMES
    assert {d.status for d in report.dimensions} <= {COVERED, PARTIAL, MISSING, OUT_OF_SCOPE}


def test_correctness_is_covered_by_a_deterministic_scorer_other_than_checks(tmp_path: Path) -> None:
    for scorer in ("{name: m, type: contains, params: {any_of: [x]}}", "{name: m, type: mrr}"):
        report = _assess(_write(tmp_path, f"[{scorer}]"))
        assert report.get("correctness").status == COVERED
    only_checks = _assess(_write(tmp_path, "[{name: m, type: checks, params: {checks: []}}]"))
    assert only_checks.get("correctness").status == MISSING


def test_a_python_scorer_is_partial_only_when_it_declares_itself_deterministic(
    tmp_path: Path,
) -> None:
    declared = (
        "[{name: m, type: python, params: {fn: 'scorer_fns:long_enough', deterministic: true}}]"
    )
    assert _assess(_write(tmp_path, declared)).get("correctness").status == PARTIAL
    undeclared = "[{name: m, type: python, params: {fn: 'scorer_fns:long_enough'}}]"
    assert _assess(_write(tmp_path, undeclared)).get("correctness").status == MISSING


def test_the_evidence_names_the_scorer(tmp_path: Path) -> None:
    report = _assess(_write(tmp_path))
    assert "exact_match `m`" in report.get("correctness").evidence


def test_load_and_production_monitoring_are_out_of_scope_and_name_a_tool(tmp_path: Path) -> None:
    report = _assess(_write(tmp_path))
    load, monitoring = report.get("load"), report.get("production-monitoring")
    assert load.status == monitoring.status == OUT_OF_SCOPE
    assert "k6" in load.evidence and "observability" in monitoring.evidence


def test_reliability_is_missing_with_one_trial_per_case(tmp_path: Path) -> None:
    gate = "gate: {metric: m, max_drop: 0.02}\n"
    report = _assess(_write(tmp_path, extra=gate))
    assert report.get("reliability").status == MISSING
    assert report.get("reliability").evidence == "one trial per case: set `trials: 3` or more"


# --- quality -----------------------------------------------------------------------------------

JUDGE = "judge: {target: 'ollama:j'}\n"
JUDGED = "[{name: m, type: exact_match}, {name: j, type: llm_judge}]"


def test_quality_is_missing_without_an_llm_judge(tmp_path: Path) -> None:
    assert _assess(_write(tmp_path)).get("quality").status == MISSING


def test_quality_is_partial_for_a_judge_with_no_calibration(tmp_path: Path) -> None:
    quality = _assess(_write(tmp_path, JUDGED, extra=JUDGE)).get("quality")
    assert quality.status == PARTIAL
    assert "no calibration" in quality.evidence and "`j`" in quality.evidence


def test_quality_is_covered_for_a_calibrated_judge(tmp_path: Path) -> None:
    cal = "calibration: {judge_scorer: j, labels: labels.jsonl}\n"
    quality = _assess(_write(tmp_path, JUDGED, extra=JUDGE + cal)).get("quality")
    assert quality.status == COVERED
    assert "calibrated against labels labels.jsonl" in quality.evidence


def test_quality_names_the_gold_scorer_when_calibrating_against_one(tmp_path: Path) -> None:
    cal = "calibration: {judge_scorer: j, gold_scorer: m, gold_sample: 2}\n"
    quality = _assess(_write(tmp_path, JUDGED, extra=JUDGE + cal)).get("quality")
    assert quality.status == COVERED and "gold scorer `m`" in quality.evidence


def test_quality_is_partial_when_the_calibration_names_a_scorer_that_is_not_the_judge(
    tmp_path: Path,
) -> None:
    cal = "calibration: {judge_scorer: m, labels: labels.jsonl}\n"
    quality = _assess(_write(tmp_path, JUDGED, extra=JUDGE + cal)).get("quality")
    assert quality.status == PARTIAL and "not an llm_judge scorer" in quality.evidence


# --- regression gate, budgets ------------------------------------------------------------------


def test_regression_gate_is_missing_without_a_gate_block(tmp_path: Path) -> None:
    assert _assess(_write(tmp_path)).get("regression-gate").status == MISSING


def test_regression_gate_is_covered_and_lists_its_settings(tmp_path: Path) -> None:
    gate = "gate: {metric: m, min_cases: 3, min_pass_rate: 0.9}\n"
    row = _assess(_write(tmp_path, extra=gate)).get("regression-gate")
    assert row.status == COVERED
    assert "`m`" in row.evidence and "min_cases 3" in row.evidence
    assert "no baseline" in row.evidence  # min_pass_rate is an absolute check


def test_regression_gate_is_partial_when_min_cases_exceeds_the_dataset(tmp_path: Path) -> None:
    row = _assess(_write(tmp_path, extra="gate: {metric: m, min_cases: 20}\n")).get(
        "regression-gate"
    )
    assert row.status == PARTIAL and "4 cases" in row.evidence


def test_budgets_are_covered_by_cost_or_latency_or_both(tmp_path: Path) -> None:
    assert _assess(_write(tmp_path, extra="gate: {metric: m}\n")).get("cost-latency").status == (
        MISSING
    )
    cost = _assess(_write(tmp_path, extra="gate: {metric: m, max_cost_usd: 0.5}\n"))
    assert cost.get("cost-latency").status == COVERED
    assert "max_cost_usd 0.5" in cost.get("cost-latency").evidence
    both = _assess(
        _write(tmp_path, extra="gate: {metric: m, max_cost_usd: 1, max_p95_latency_ms: 800}\n")
    )
    assert "max_p95_latency_ms 800" in both.get("cost-latency").evidence


# --- safety, robustness, multi-turn ------------------------------------------------------------


def test_safety_and_robustness_count_tagged_cases(tmp_path: Path) -> None:
    cases = [
        {"id": "a", "input": "q", "tags": ["safety"]},
        {"id": "b", "input": "q", "tags": ["Safety", "robustness"]},
        {"id": "c", "input": "q", "tags": ["core"]},
        {"id": "d", "input": "q"},
    ]
    report = _assess(_write(tmp_path, cases=cases))
    assert report.get("safety").status == COVERED
    assert report.get("safety").evidence == "2 of 4 cases tagged `safety`"
    assert report.get("robustness").evidence == "1 of 4 cases tagged `robustness`"


def test_safety_and_robustness_are_missing_when_no_case_is_tagged(tmp_path: Path) -> None:
    report = _assess(_write(tmp_path))
    assert report.get("safety").status == report.get("robustness").status == MISSING
    assert "no case is tagged `safety`" in report.get("safety").evidence


def test_multi_turn_is_partial_when_a_case_has_turns(tmp_path: Path) -> None:
    cases = [{"id": "a", "input": "hi", "turns": ["and then?"]}, {"id": "b", "input": "hi"}]
    row = _assess(_write(tmp_path, cases=cases)).get("multi-turn")
    assert row.status == PARTIAL and "1 of 2 cases have turns" in row.evidence
    assert _assess(_write(tmp_path)).get("multi-turn").status == MISSING


# --- behaviour (tool use) ----------------------------------------------------------------------

CHECKS = "[{name: b, type: checks}]"


def _behaviour(tmp_path: Path, scorers: str, expects: list[Any]) -> Any:
    cases = [
        {"id": f"c{i}", "input": "q", **({"expect": e} if e is not None else {})}
        for i, e in enumerate(expects)
    ]
    return _assess(_write(tmp_path, scorers, cases=cases)).get("behaviour")


def test_behaviour_is_missing_without_a_checks_scorer(tmp_path: Path) -> None:
    row = _behaviour(tmp_path, "[{name: m, type: exact_match}]", [[CHECK_TOOL]])
    assert row.status == MISSING and row.evidence == "no checks scorer"


def test_behaviour_is_covered_when_half_the_cases_check_the_tool_calls(tmp_path: Path) -> None:
    row = _behaviour(tmp_path, CHECKS, [[CHECK_TOOL], [CHECK_TOOL], None, None])
    assert row.status == COVERED and "2 of 4 cases" in row.evidence


def test_behaviour_is_partial_below_half_and_missing_at_zero(tmp_path: Path) -> None:
    partial = _behaviour(tmp_path, CHECKS, [[CHECK_TOOL], None, None, None])
    assert partial.status == PARTIAL and "1 of 4 cases" in partial.evidence
    assert _behaviour(tmp_path, CHECKS, [None, None]).status == MISSING


def test_an_output_only_check_is_not_a_check_on_behaviour(tmp_path: Path) -> None:
    assert _behaviour(tmp_path, CHECKS, [[CHECK_TEXT], [CHECK_TEXT]]).status == MISSING


def test_a_trace_check_inside_any_of_or_not_counts(tmp_path: Path) -> None:
    nested = [
        [{"type": "any_of", "checks": [CHECK_TEXT, CHECK_TOOL]}],
        [{"type": "not", "check": {"type": "tool_not_called", "name": "delete"}}],
    ]
    assert _behaviour(tmp_path, CHECKS, nested).status == COVERED


def test_a_suite_wide_check_covers_every_case(tmp_path: Path) -> None:
    scorers = "[{name: b, type: checks, params: {checks: [{type: rounds, max: 3}]}}]"
    row = _behaviour(tmp_path, scorers, [None, None, None])
    assert row.status == COVERED and "3 of 3 cases" in row.evidence


def test_the_checks_scorer_field_param_selects_where_a_case_keeps_its_checks(
    tmp_path: Path,
) -> None:
    scorers = "[{name: b, type: checks, params: {field: guard}}]"
    cases = [{"id": f"c{i}", "input": "q", "guard": [CHECK_TOOL]} for i in range(2)]
    row = _assess(_write(tmp_path, scorers, cases=cases)).get("behaviour")
    assert row.status == COVERED
    on_expect = [{"id": "c0", "input": "q", "expect": [CHECK_TOOL]}]
    assert _assess(_write(tmp_path, scorers, cases=on_expect)).get("behaviour").status == MISSING


# --- --run -------------------------------------------------------------------------------------


def _run_with_traces(path: Path, *, traced: int, errored: int, total: int = 4) -> Path:
    run = make_run(
        {f"c{i}": {"m": True} for i in range(total)}, errors={f"c{i}" for i in range(errored)}
    )
    for result in run.results[:traced]:
        result.output.trace = Trace(responses=["x"])
    return save_run(run, path)


def test_a_run_adds_trace_and_error_counts(tmp_path: Path) -> None:
    suite = _write(tmp_path, CHECKS, cases=[{"id": "a", "input": "q", "expect": [CHECK_TOOL]}])
    path = _run_with_traces(tmp_path / "run.json", traced=3, errored=1)
    report = _assess(suite, run=load_run(path), run_path=str(path))
    assert report.run is not None
    assert (report.run.cases, report.run.with_trace, report.run.errors) == (4, 3, 1)
    assert "3 of 4 cases carry a trace" in report.get("behaviour").evidence
    assert "3 with a trace, 1 errored" in readiness_markdown(report)


def test_a_run_with_no_traces_warns_in_the_behaviour_evidence(tmp_path: Path) -> None:
    suite = _write(tmp_path, CHECKS, cases=[{"id": "a", "input": "q", "expect": [CHECK_TOOL]}])
    path = _run_with_traces(tmp_path / "run.json", traced=0, errored=0)
    report = _assess(suite, run=load_run(path))
    assert "no case carries a trace" in report.get("behaviour").evidence


def test_a_run_does_not_invent_behaviour_evidence_when_there_is_no_checks_scorer(
    tmp_path: Path,
) -> None:
    path = _run_with_traces(tmp_path / "run.json", traced=4, errored=0)
    report = _assess(_write(tmp_path), run=load_run(path))
    assert report.get("behaviour").evidence == "no checks scorer"


# --- --require ---------------------------------------------------------------------------------


def test_parse_required_accepts_commas_repeats_and_underscores() -> None:
    assert parse_required(["correctness,safety", "cost_latency", "safety"]) == [
        "correctness",
        "safety",
        "cost-latency",
    ]
    assert parse_required(None) == [] and parse_required([""]) == []


def test_parse_required_lists_the_valid_names_for_an_unknown_one() -> None:
    with pytest.raises(ConfigError) as exc:
        parse_required(["correctness,bogus"])
    message = str(exc.value)
    assert "unknown dimension 'bogus'" in message
    for name in DIMENSION_NAMES:
        assert name in message


def test_unmet_lists_only_the_required_dimensions_that_are_not_covered(tmp_path: Path) -> None:
    report = _assess(_write(tmp_path))
    unmet = report.unmet(["correctness", "safety", "load"])
    assert [d.name for d in unmet] == ["safety", "load"]  # out of scope is not covered either


# --- output ------------------------------------------------------------------------------------


def test_markdown_is_a_table_with_one_row_per_dimension(tmp_path: Path) -> None:
    text = readiness_markdown(_assess(_write(tmp_path)))
    assert text.startswith("## Readiness: demo\n")
    assert "| Dimension | Status | Evidence |" in text
    assert sum(line.startswith("| ") for line in text.splitlines()) == len(DIMENSION_NAMES) + 1
    assert "Not ready" not in text


def test_a_pipe_or_newline_in_evidence_cannot_break_the_table() -> None:
    assert _cell("a|b\nc") == "a\\|b c"


def test_json_has_the_dimensions_the_required_list_and_the_verdict(tmp_path: Path) -> None:
    data = json.loads(readiness_json(_assess(_write(tmp_path)), ["correctness", "safety"]))
    assert data["suite"] == "demo" and data["cases"] == 4 and data["run"] is None
    assert [d["name"] for d in data["dimensions"]] == list(DIMENSION_NAMES)
    assert data["required"] == ["correctness", "safety"]
    assert data["unmet"] == ["safety"] and data["ok"] is False


# --- the command -------------------------------------------------------------------------------


def test_cli_prints_the_table_and_exits_0(tmp_path: Path, capsys) -> None:
    code = main(["readiness", str(_write(tmp_path))])
    out = capsys.readouterr().out
    assert code == 0
    assert "| correctness | covered |" in out and "| load | out of scope |" in out


def test_cli_exits_1_when_a_required_dimension_is_not_covered(tmp_path: Path, capsys) -> None:
    suite = str(_write(tmp_path))
    assert main(["readiness", suite, "--require", "correctness"]) == 0
    capsys.readouterr()
    code = main(["readiness", suite, "--require", "correctness,safety"])
    out = capsys.readouterr().out
    assert code == 1
    assert "Not ready" in out and "safety (missing)" in out


def test_cli_require_may_repeat(tmp_path: Path, capsys) -> None:
    code = main(["readiness", str(_write(tmp_path)), "--require", "safety", "--require", "load"])
    assert code == 1
    assert "safety (missing), load (out of scope)" in capsys.readouterr().out


def test_cli_unknown_require_is_a_usage_error_that_lists_the_valid_names(
    tmp_path: Path, capsys
) -> None:
    code = main(["readiness", str(_write(tmp_path)), "--require", "correctness,nope"])
    captured = capsys.readouterr()
    assert code == 2 and captured.out == ""
    assert "unknown dimension 'nope'" in captured.err and "production-monitoring" in captured.err


def test_cli_json_is_parseable_and_carries_the_exit_verdict(tmp_path: Path, capsys) -> None:
    code = main(["readiness", str(_write(tmp_path)), "--json", "--require", "safety"])
    data = json.loads(capsys.readouterr().out)
    assert code == 1 and data["ok"] is False and data["unmet"] == ["safety"]


def test_cli_with_a_run_file_reports_its_path_and_counts(tmp_path: Path, capsys) -> None:
    suite = _write(tmp_path, CHECKS, cases=[{"id": "a", "input": "q", "expect": [CHECK_TOOL]}])
    path = _run_with_traces(tmp_path / "run.json", traced=2, errored=1)
    assert main(["readiness", str(suite), "--run", str(path), "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["run"] == {"path": str(path), "cases": 4, "with_trace": 2, "errors": 1}


def test_cli_a_missing_suite_or_run_is_a_usage_error(tmp_path: Path, capsys) -> None:
    assert main(["readiness", str(tmp_path / "nope.yaml")]) == 2
    suite = str(_write(tmp_path))
    assert main(["readiness", suite, "--run", str(tmp_path / "nope.json")]) == 2
    capsys.readouterr()


def test_cli_makes_no_model_call(tmp_path: Path, monkeypatch) -> None:
    from evalproof import providers

    def boom(*a: Any, **k: Any) -> None:
        raise AssertionError("readiness must not build a provider")

    monkeypatch.setattr(providers, "make_provider", boom)
    assert main(["readiness", str(_write(tmp_path, JUDGED, extra=JUDGE))]) == 0


def test_shipped_examples_report_without_error(capsys) -> None:
    root = Path(__file__).resolve().parent.parent / "examples"
    suites = sorted(root.glob("*/suite.yaml"))
    assert suites
    for suite in suites:
        assert main(["readiness", str(suite)]) == 0
        assert "| correctness |" in capsys.readouterr().out
