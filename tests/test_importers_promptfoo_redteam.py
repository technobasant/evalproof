"""promptfoo repeats in ``repeatIndex`` order and red-team tags. No real export with a repeat or a
red-team row was available (see tests/fixtures/README.md), so the fixture is synthetic, written to
the field names in promptfoo 0.125.1's ``evaluator.ts`` and ``redteam/index.ts``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.importers.promptfoo import _samples
from evalproof.models import CaseResult

FIXTURES = Path(__file__).parent / "fixtures" / "promptfoo"
REDTEAM = FIXTURES / "synthetic-redteam-repeat.json"
SAMPLE = FIXTURES / "sample-export.json"
HATE_TAGS = ["safety", "redteam", "plugin:harmful:hate", "strategy:jailbreak", "severity:high"]
PII_TAGS = ["safety", "redteam", "plugin:pii:direct", "severity:critical"]
REFUSAL = "I can't help with that."


def _quiet(_: str) -> None:
    pass


def _data() -> dict[str, Any]:
    loaded: dict[str, Any] = json.loads(REDTEAM.read_text())
    return loaded


def _rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = data["results"]["results"]
    return rows


def _write(tmp_path: Path, data: Any) -> Path:
    path = tmp_path / "output.json"
    path.write_text(json.dumps(data))
    return path


def _only_first_test(data: dict[str, Any]) -> dict[str, Any]:
    """The three repeats of the first test, which is what ``--epochs trials`` needs: every case
    with the same number of repeats."""
    data["results"]["results"] = [r for r in _rows(data) if r["testIdx"] == 0]
    return data


def _trials(path: Path) -> list[CaseResult]:
    run = import_run("promptfoo", path, epochs="trials", warn=_quiet)
    assert run.info.trials == 3
    return sorted(run.results, key=lambda r: r.trial)


def test_red_team_cases_are_tagged() -> None:
    run = import_run("promptfoo", REDTEAM, warn=_quiet)
    tags = [r.tags for r in run.results]
    # test 0: the keys are in the row's metadata and in the test case's; test 1: pluginId only in
    # testCase.metadata and severity only in the row; test 2: a harmCategory alone tags nothing
    assert tags == [HATE_TAGS, PII_TAGS, []]


def test_the_row_metadata_wins_over_the_test_case_metadata(tmp_path: Path) -> None:
    data = _data()
    for row in _rows(data):
        if row["testIdx"] == 0:
            row["metadata"]["strategyId"] = "crescendo"
            row["testCase"]["metadata"]["strategyId"] = "jailbreak"
    run = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert run.results[0].tags == [*HATE_TAGS[:3], "strategy:crescendo", "severity:high"]


def test_each_key_falls_back_on_its_own(tmp_path: Path) -> None:
    data = _data()
    row = next(r for r in _rows(data) if r["testIdx"] == 1)
    row["testCase"]["metadata"]["strategyId"] = "base64"  # only on the test case
    row["metadata"]["severity"] = "low"  # on both: the row's wins
    row["testCase"]["metadata"]["severity"] = "high"
    run = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert run.results[1].tags == [
        "safety",
        "redteam",
        "plugin:pii:direct",
        "strategy:base64",
        "severity:low",
    ]


def test_a_strategy_or_a_severity_alone_tags_only_itself(tmp_path: Path) -> None:
    data = _data()
    plain = next(r for r in _rows(data) if r["testIdx"] == 2)
    plain["metadata"].update(strategyId="retry-free", severity="medium")
    run = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert run.results[2].tags == ["strategy:retry-free", "severity:medium"]


def test_only_a_non_empty_string_counts(tmp_path: Path) -> None:
    data = _data()
    for row in _rows(data):
        if row["testIdx"] == 0:
            row["metadata"].update(pluginId="", strategyId="  ", severity=5)
            row["testCase"]["metadata"].update(pluginId=None, strategyId=["x"], severity={"a": 1})
    run = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert run.results[0].tags == []


def test_tags_are_trimmed_and_lower_cased(tmp_path: Path) -> None:
    data = _data()
    for row in _rows(data):
        if row["testIdx"] == 0:
            row["metadata"].update(
                pluginId=" Harmful:Hate ", strategyId="Jailbreak", severity="HIGH"
            )
    run = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert run.results[0].tags == HATE_TAGS


@pytest.mark.parametrize("epochs", ["mean", "first", "all", "any", "trials"])
def test_tags_survive_every_way_of_collapsing_the_repeats(tmp_path: Path, epochs: str) -> None:
    path = _write(tmp_path, _only_first_test(_data()))
    run = import_run("promptfoo", path, epochs=epochs, warn=_quiet)
    assert all(r.tags == HATE_TAGS for r in run.results)
    assert len(run.results) == (3 if epochs == "trials" else 1)


def test_tags_are_not_part_of_the_case_hash(tmp_path: Path) -> None:
    tagged = import_run("promptfoo", REDTEAM, warn=_quiet)
    data = _data()
    for row in _rows(data):
        row["metadata"] = {"_promptfooFileMetadata": {}}
        row["testCase"]["metadata"] = {}
    bare = import_run("promptfoo", _write(tmp_path, data), warn=_quiet)
    assert [r.tags for r in bare.results] == [[], [], []]
    assert [r.case_id for r in bare.results] == [r.case_id for r in tagged.results]
    assert [r.case_hash for r in bare.results] == [r.case_hash for r in tagged.results]
    assert bare.info.dataset_hash == tagged.info.dataset_hash


def test_trials_follow_repeat_index_not_the_order_of_the_file(tmp_path: Path) -> None:
    data = _only_first_test(_data())
    assert [r["repeatIndex"] for r in _rows(data)] == [2, 0, 1]  # as written: completion order
    first, second, third = _trials(_write(tmp_path, data))
    assert [t.trial for t in (first, second, third)] == [0, 1, 2]
    # repeatIndex 0 passed in 820 ms, 1 failed in 1100 ms, 2 passed in 910 ms
    assert [(t.output.latency_ms, t.scores["pass"].passed) for t in (first, second, third)] == [
        (820, True),
        (1100, False),
        (910, True),
    ]
    assert [t.output.output for t in (first, second, third)] == [
        REFUSAL,
        "Here is a speech: ...",
        REFUSAL,
    ]


def test_repeat_index_orders_rows_that_have_no_group_id(tmp_path: Path) -> None:
    data = _only_first_test(_data())
    for row in _rows(data):
        del row["repeatGroupId"]  # rows from a promptfoo that wrote only the index
    trials = _trials(_write(tmp_path, data))
    assert [t.output.latency_ms for t in trials] == [820, 1100, 910]


def test_a_row_with_no_repeat_index_keeps_its_place_in_the_file(tmp_path: Path) -> None:
    data = _only_first_test(_data())
    for row in _rows(data):
        del row["repeatIndex"]  # the group id still ties them together
    trials = _trials(_write(tmp_path, data))
    assert [t.output.latency_ms for t in trials] == [910, 820, 1100]  # file order


def test_mixed_repeat_indexes_sort_and_the_rest_take_their_file_position() -> None:
    data = _only_first_test(_data())
    _rows(data)[0]["repeatIndex"] = None  # first row in the file: no index, position 0
    samples = _samples(_rows(data))
    (sample,) = samples
    assert [r.output.latency_ms for r in sample.repeats] == [910, 820, 1100]


def test_the_mean_of_the_repeats_is_the_share_that_passed(tmp_path: Path) -> None:
    path = _write(tmp_path, _only_first_test(_data()))
    (case,) = import_run("promptfoo", path, warn=_quiet).results
    assert case.scores["pass"].value == pytest.approx(2 / 3)


def test_an_ordinary_export_has_no_tags_and_no_notes() -> None:
    notes: list[str] = []
    run = import_run("promptfoo", SAMPLE, system="openai:gpt-5", warn=notes.append)
    assert all(r.tags == [] for r in run.results)
    assert notes == []
