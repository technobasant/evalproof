"""The claude plugin eval importer. The docs show no sample ``aggregate-result.json`` (see
tests/fixtures/README.md), so the one fixture is synthetic, written to the field names of the
Claude Code 2.1.322 writer schema."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.cli import main
from evalproof.compare import run_stats
from evalproof.importers import EXPERIMENTAL_FORMATS, FORMATS, ImportFormatError, _common
from evalproof.importers._common import imported_case_hash
from evalproof.models import CaseResult, Run
from evalproof.runner import load_run

FIXTURE = Path(__file__).parent / "fixtures" / "claude-plugin-eval" / "aggregate-result.json"
COMMIT = "commit-message"
RENAME = "rename-symbol"
EXPLAIN = "explain-stack-trace"
SKIPPED_NOTE = f"skipped 1 case(s) with no runs in the 'without' arm: {EXPLAIN}"


def _quiet(_: str) -> None:
    pass


def _data() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(FIXTURE.read_text(encoding="utf-8"))
    return loaded


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "aggregate-result.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _trials(arm: str, path: Path = FIXTURE) -> dict[tuple[str, int], CaseResult]:
    run = import_run("claude-plugin-eval", path, system=arm, epochs="trials", warn=_quiet)
    return {(r.case_id, r.trial): r for r in run.results}


def _case(data: dict[str, Any], name: str) -> dict[str, Any]:
    (found,) = [c for c in data["cases"] if c["name"] == name]
    return found


def _without_path(run: Run) -> tuple[Any, Any]:
    info = run.info.model_dump()
    info.pop("dataset_path")
    return info, [r.model_dump() for r in run.results]


def test_is_an_experimental_format(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert "claude-plugin-eval" in FORMATS
    assert "claude-plugin-eval" in EXPERIMENTAL_FORMATS
    monkeypatch.setattr(_common, "_warned", set())  # the note is once per process
    run = import_run("claude-plugin-eval", FIXTURE, system="with", warn=_quiet)
    import_run("claude-plugin-eval", FIXTURE, system="with", warn=_quiet)
    assert capsys.readouterr().err == "evalproof: the claude-plugin-eval importer is experimental\n"
    env = run.info.environment
    assert env["import_experimental"] == "true"
    assert env["imported_from"] == "claude-plugin-eval"
    assert env["source_schema"] == "claude-plugin-eval/1"
    assert env["source_tool_version"] == "2.1.322"


def test_maps_the_with_arm() -> None:
    notes: list[str] = []
    run = import_run("claude-plugin-eval", FIXTURE, system="with", warn=notes.append)
    assert notes == []  # a finished suite whose with arm ran every case
    info = run.info
    assert (info.suite, info.system) == ("aggregate-result", "with")
    assert info.started_at == "2026-11-14T10:00:00.000Z"
    assert info.wall_time_s == 212.0
    assert [r.case_id for r in run.results] == [COMMIT, RENAME, EXPLAIN]
    assert all(r.tags == [] for r in run.results)
    first = _trials("with")[(COMMIT, 0)]
    assert first.output.output == ""  # the file holds no reply text
    assert first.output.trace is None
    assert first.output.error is None
    assert first.output.latency_ms == pytest.approx(21_400)
    assert first.output.cost_usd == pytest.approx(0.0391)  # costUsd 0.0412 less the judge's 0.0021
    assert first.scores["score"].value == 1.0
    assert first.scores["score"].passed is True
    assert first.scores["score"].judge_cost_usd == pytest.approx(0.0021)
    assert first.scores["criteria"].value == 1.0
    assert first.scores["criteria"].passed is True
    assert first.scores["criteria"].reason == "judge votes: PASS PASS PASS"
    assert first.scores["criteria"].judge_cost_usd == 0.0
    second = _trials("with")[(COMMIT, 1)]
    assert (second.scores["score"].value, second.scores["score"].passed) == (0.0, False)
    assert second.scores["criteria"].passed is False
    prompt = _case(_data(), COMMIT)["promptMarkdown"]
    assert first.case_hash == imported_case_hash(COMMIT, prompt)


def test_the_without_arm_has_no_with_only_grader() -> None:
    trials = _trials("without")
    assert set(trials[(COMMIT, 0)].scores) == {"score", "criteria"}
    assert trials[(COMMIT, 0)].output.latency_ms == pytest.approx(15_200)
    assert trials[(COMMIT, 1)].output.latency_ms == 0.0  # the run has no durationSeconds
    assert (
        trials[(COMMIT, 1)].scores["score"].value,
        trials[(COMMIT, 1)].scores["criteria"].passed,
    ) == (
        0.0,
        False,
    )
    assert set(trials[(RENAME, 0)].scores) == {"score", "tests-updated", "criteria"}


def test_a_with_only_grader_is_shown_without_a_verdict() -> None:
    skill = _trials("with")[(COMMIT, 0)].scores[
        "skill-fired"
    ]  # withOnly, scored false, passed true
    assert (skill.value, skill.passed) == (None, None)
    assert skill.reason == "not scored: Skill called 1x (expected 1..∞)"


def test_scored_false_means_no_value_and_a_missing_scored_follows_with_only(tmp_path: Path) -> None:
    data = _data()
    run0 = _case(data, RENAME)["arms"]["with"][0]
    tests_updated, criteria, skill = run0["graders"]
    tests_updated["scored"] = False  # not withOnly, but not scored
    del criteria["scored"]  # absent: scored, as the writer defaults it
    del skill["scored"]  # absent on a withOnly grader: not scored
    scores = _trials("with", _write(tmp_path, data))[(RENAME, 0)].scores
    assert (scores["tests-updated"].value, scores["tests-updated"].passed) == (None, None)
    assert scores["criteria"].value == 1.0
    assert (scores["skill-fired"].value, scores["skill-fired"].passed) == (None, None)


def test_a_run_with_an_error_fails_every_metric() -> None:
    timed_out = _trials("with")[(RENAME, 1)]  # error "timed out after 300s", graded 0.5 anyway
    assert timed_out.output.error == "timed out after 300s"
    assert set(timed_out.scores) == {"score", "criteria", "skill-fired", "tests-updated"}
    for score in timed_out.scores.values():
        assert (score.value, score.passed) == (0.0, False)
        assert score.reason == "system error: timed out after 300s"
    assert timed_out.scores["score"].judge_cost_usd == pytest.approx(0.0021)  # the judge was paid


def test_an_aborted_run_is_an_error_too() -> None:
    aborted = _trials("without")[(RENAME, 1)]  # aborted, error null, score 0, no graders
    assert aborted.output.error == (
        "aborted by a mock: server github, tool create_issue, "
        "reason agent-mock call budget of 3 exceeded"
    )
    assert set(aborted.scores) == {"score", "criteria", "tests-updated"}
    assert all((s.value, s.passed) == (0.0, False) for s in aborted.scores.values())
    run = import_run("claude-plugin-eval", FIXTURE, system="without", warn=_quiet)
    assert run_stats(run).errors == 0  # under mean one run of the case answered
    trials = import_run(
        "claude-plugin-eval", FIXTURE, system="without", epochs="trials", warn=_quiet
    )
    assert run_stats(trials).errors == 1


def test_error_and_aborted_together_are_both_named(tmp_path: Path) -> None:
    data = _data()
    run = _case(data, RENAME)["arms"]["without"][1]
    run["error"] = "exit 1: boom"
    path = _write(tmp_path, data)
    error = _trials("without", path)[(RENAME, 1)].output.error
    assert error is not None
    assert error.startswith("exit 1: boom; aborted by a mock: server github")
    run["aborted"] = True  # a bare flag, not an object
    run["error"] = None
    assert _trials("without", _write(tmp_path, data))[(RENAME, 1)].output.error == (
        "aborted by a mock"
    )


def test_an_empty_error_text_is_still_an_error(tmp_path: Path) -> None:
    data = _data()
    _case(data, COMMIT)["arms"]["with"][0]["error"] = ""
    error = _trials("with", _write(tmp_path, data))[(COMMIT, 0)].output.error
    assert error == "the run reported an error with no text"


def test_the_runs_of_an_arm_are_its_trials_in_list_order() -> None:
    trials = _trials("without")
    assert [(c, t) for c, t in trials] == [(COMMIT, 0), (COMMIT, 1), (RENAME, 0), (RENAME, 1)]
    assert trials[(RENAME, 0)].scores["score"].value == 0.0


def test_trials_need_every_case_to_have_the_same_number_of_runs(tmp_path: Path) -> None:
    data = _data()
    del _case(data, EXPLAIN)["arms"]["with"][1]
    with pytest.raises(ImportFormatError, match="same number of repeats"):
        import_run(
            "claude-plugin-eval",
            _write(tmp_path, data),
            system="with",
            epochs="trials",
            warn=_quiet,
        )


def test_mean_collapses_the_runs_of_a_case() -> None:
    run = import_run("claude-plugin-eval", FIXTURE, system="with", warn=_quiet)
    by_id = {r.case_id: r for r in run.results}
    assert by_id[COMMIT].scores["score"].value == pytest.approx(0.5)
    assert by_id[COMMIT].scores["criteria"].value == pytest.approx(0.5)
    assert by_id[RENAME].scores["score"].value == pytest.approx(0.5)  # 1 and a failed timed-out run
    assert by_id[RENAME].output.error is None  # one run answered
    assert by_id[EXPLAIN].scores["score"].value == pytest.approx(0.5)
    assert run.info.environment["epochs"] == "mean"


@pytest.mark.parametrize(
    ("arm", "epochs"),
    [
        ("with", "mean"),
        ("with", "all"),
        ("with", "any"),
        ("without", "mean"),
        ("without", "trials"),
    ],
)
def test_the_run_costs_what_the_file_says(arm: str, epochs: str) -> None:
    run = import_run("claude-plugin-eval", FIXTURE, system=arm, epochs=epochs, warn=_quiet)
    # costUsd already holds the judge's share; the run has it as the output cost plus the judge
    # cost on the score, which compare adds back together, in every way the runs can collapse.
    spent = sum(r["costUsd"] for c in _data()["cases"] for r in c["arms"].get(arm, []))
    assert run_stats(run).cost_usd == pytest.approx(spent)


def test_with_and_without_arms_need_a_choice() -> None:
    with pytest.raises(ImportFormatError, match=r"holds 2 systems.*'with', 'without'"):
        import_run("claude-plugin-eval", FIXTURE)
    with pytest.raises(ImportFormatError, match=r"no system 'baseline'.*'with', 'without'"):
        import_run("claude-plugin-eval", FIXTURE, system="baseline")


def test_a_file_with_no_baseline_arm(tmp_path: Path) -> None:
    data = _data()
    for case in data["cases"]:
        case["arms"].pop("without", None)  # ``--ablation none``
    path = _write(tmp_path, data)
    with pytest.raises(
        ImportFormatError, match=r"no system 'without' in this file; available: 'with'"
    ):
        import_run("claude-plugin-eval", path, system="without")
    run = import_run("claude-plugin-eval", path, warn=_quiet)  # one arm: no --system needed
    assert run.info.system == "with"


def test_a_case_with_no_runs_in_the_arm_is_skipped_with_a_note() -> None:
    notes: list[str] = []
    run = import_run("claude-plugin-eval", FIXTURE, system="without", warn=notes.append)
    assert [r.case_id for r in run.results] == [COMMIT, RENAME]
    assert len(notes) == 1
    assert notes[0].endswith(SKIPPED_NOTE)


def test_a_partial_suite_warns_with_the_reason(tmp_path: Path) -> None:
    data = _data()
    data["partial"] = True
    data["partialReason"] = "cost_ceiling"
    notes: list[str] = []
    run = import_run("claude-plugin-eval", _write(tmp_path, data), system="with", warn=notes.append)
    assert len(notes) == 1
    assert "did not finish" in notes[0]
    assert "cost_ceiling" in notes[0]
    assert len(run.results) == 3  # what it ran is still imported
    assert run.info.environment["partial"] == "true"
    assert run.info.environment["partial_reason"] == "cost_ceiling"
    del data["partialReason"]
    notes.clear()
    run = import_run("claude-plugin-eval", _write(tmp_path, data), system="with", warn=notes.append)
    assert "reason: not given" in notes[0]
    assert run.info.environment["partial"] == "true"
    assert "partial_reason" not in run.info.environment


def test_skipped_judge_graders_are_named(tmp_path: Path) -> None:
    data = _data()
    _case(data, COMMIT)["arms"]["with"][0]["skippedPaidGraders"] = True
    _case(data, RENAME)["arms"]["with"][1]["skippedPaidGraders"] = True
    notes: list[str] = []
    run = import_run("claude-plugin-eval", _write(tmp_path, data), system="with", warn=notes.append)
    assert len(notes) == 1
    assert "skipped the judge graders of 2 run(s)" in notes[0]
    assert "not comparable" in notes[0]
    assert run.info.environment["skipped_paid_graders"] == "true"


def test_a_complete_suite_records_neither_marker() -> None:
    env = import_run("claude-plugin-eval", FIXTURE, system="with", warn=_quiet).info.environment
    assert not {"partial", "partial_reason", "skipped_paid_graders"} & set(env)


@pytest.mark.parametrize("version", [2, 0, "1", True, 1.5])
def test_another_schema_version_is_refused(tmp_path: Path, version: Any) -> None:
    data = _data()
    data["schemaVersion"] = version
    with pytest.raises(ImportFormatError, match="this importer reads schemaVersion 1"):
        import_run("claude-plugin-eval", _write(tmp_path, data), system="with")


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda d: d.pop("schemaVersion"), "no schemaVersion"),
        (lambda d: d.pop("cases"), "no cases list"),
        (lambda d: d.update(cases="x"), "no cases list"),
        (lambda d: d.update(cases=[]), "the file holds no results"),
        (lambda d: d["cases"][1].pop("name"), r"cases\[1\] has no name"),
        (lambda d: d["cases"][0]["arms"]["with"].__setitem__(0, "x"), "is not an object"),
        (
            lambda d: d["cases"][0]["arms"]["with"][0]["graders"][0].pop("name"),
            r"graders\[0\] has no name",
        ),
        (
            lambda d: d["cases"][0]["arms"]["with"][0]["graders"][1].update(name="criteria"),
            "two graders named 'criteria'",
        ),
        (lambda d: d["cases"][0]["arms"]["with"][0].update(score=float("nan")), "NaN"),
    ],
)
def test_a_file_that_is_not_a_result_is_refused(tmp_path: Path, change: Any, message: str) -> None:
    data = _data()
    change(data)
    path = tmp_path / "aggregate-result.json"
    path.write_text(
        json.dumps(data), encoding="utf-8"
    )  # json.dumps writes NaN, which json.loads reads back
    with pytest.raises(ImportFormatError, match=message):
        import_run("claude-plugin-eval", path, system="with", warn=_quiet)


def test_a_list_is_not_a_result(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match="expected a JSON object"):
        import_run("claude-plugin-eval", _write(tmp_path, []), system="with")


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    plain = import_run("claude-plugin-eval", FIXTURE, system="with", epochs="mean", warn=_quiet)
    data = _data()
    data["somethingNew"] = {"added": "later"}
    data["suite"]["alsoNew"] = 1
    for case in data["cases"]:
        case["newCaseField"] = [1, 2]
        for runs in case["arms"].values():
            for run in runs:
                run["newRunField"] = "x"
                for grader in run["graders"]:
                    grader["newGraderField"] = {"a": 1}
    other = import_run(
        "claude-plugin-eval", _write(tmp_path, data), system="with", epochs="mean", warn=_quiet
    )
    assert _without_path(other) == _without_path(plain)


def test_a_grader_named_score_does_not_replace_the_score_metric(tmp_path: Path) -> None:
    data = _data()
    run0 = _case(data, COMMIT)["arms"]["with"][0]
    run0["graders"][0]["name"] = "score"
    run0["graders"][0]["passed"] = False  # the grader fails, the run's score stays 1
    scores = _trials("with", _write(tmp_path, data))[(COMMIT, 0)].scores
    assert scores["score"].value == 1.0
    assert (scores["grader:score"].value, scores["grader:score"].passed) == (0.0, False)


def test_duplicate_case_names_are_numbered(tmp_path: Path) -> None:
    data = _data()
    data["cases"][1]["name"] = COMMIT
    run = import_run("claude-plugin-eval", _write(tmp_path, data), system="with", warn=_quiet)
    assert [r.case_id for r in run.results] == [COMMIT, f"{COMMIT}#2", EXPLAIN]


def test_non_numeric_cost_and_duration_count_as_nothing(tmp_path: Path) -> None:
    data = _data()
    run0 = _case(data, COMMIT)["arms"]["with"][0]
    run0.update(costUsd="n/a", durationSeconds=-3, judgeCostUsd=None)
    first = _trials("with", _write(tmp_path, data))[(COMMIT, 0)]
    assert (first.output.cost_usd, first.output.latency_ms) == (0.0, 0.0)
    assert first.scores["score"].judge_cost_usd == 0.0


def test_table_options_are_refused() -> None:
    with pytest.raises(ImportFormatError, match="apply to the jsonl and csv formats"):
        import_run("claude-plugin-eval", FIXTURE, system="with", wide=True)
    with pytest.raises(ImportFormatError, match="applies to the garak format"):
        import_run("claude-plugin-eval", FIXTURE, system="with", threshold=0.5)


def test_import_on_the_command_line(tmp_path: Path) -> None:
    out = tmp_path / "run.json"
    code = main(
        ["import", "claude-plugin-eval", str(FIXTURE), "--out", str(out), "--system", "without"]
    )
    assert code == 0
    run = load_run(out)
    assert (run.info.system, len(run.results)) == ("without", 2)
    assert run.info.environment["import_experimental"] == "true"
    assert main(["import", "claude-plugin-eval", str(FIXTURE), "--out", str(out)]) != 0
