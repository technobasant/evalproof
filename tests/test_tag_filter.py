"""``--tags`` and ``--exclude-tags``: choose cases by tag, record the filter, check and compare it.

The suite's system is a python function, so no model is called.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs, dataset_change_note
from evalproof.config import load_config
from evalproof.dataset import DatasetError, make_tag_filter, parse_tag_list
from evalproof.models import Case, Run, TagFilter
from evalproof.runner import load_run, run_suite, save_run
from evalproof.verify import verify_markdown, verify_run
from tests.conftest import ScriptedProvider

CASES: list[dict[str, Any]] = [
    {"id": "c0", "input": {"q": "a"}, "accept": ["A"], "tags": ["smoke"]},
    {"id": "c1", "input": {"q": "b"}, "accept": ["B"], "tags": ["smoke", "slow"]},
    {"id": "c2", "input": {"q": "c"}, "accept": ["C"], "tags": ["slow"]},
    {"id": "c3", "input": {"q": "d"}, "accept": ["D"], "tags": []},
    {"id": "c4", "input": {"q": "e"}, "accept": ["E"], "tags": ["smoke"]},
    {"id": "c5", "input": {"q": "f"}, "accept": ["F"], "tags": ["edge"]},
]


def _suite(tmp_path: Path, cases: list[dict[str, Any]] | None = None) -> Path:
    rows = CASES if cases is None else cases
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in rows) + "\n", "utf-8")
    (tmp_path / "answer.py").write_text("def ask(x):\n    return x['q'].upper()\n", "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\n"
        "system: {target: 'python:answer:ask', label: v1}\n"
        "scorers:\n  - {name: m, type: contains, params: {any_of_field: accept}}\n"
        "gate: {metric: m, min_cases: 2}\n",
        "utf-8",
    )
    return suite


def _run(suite: Path, **kw: Any) -> Run:
    return run_suite(load_config(suite), use_cache=False, **kw)


def _ids(run: Run) -> list[str]:
    return [r.case_id for r in run.results]


# --- the option values and the filter ------------------------------------------------------------


def test_tag_options_repeat_and_split_on_commas_into_a_sorted_set() -> None:
    assert parse_tag_list(["b, a", "c", "a"], "--tags") == ["a", "b", "c"]
    assert parse_tag_list(None, "--tags") == []
    flt = make_tag_filter(["smoke,slow"], ["edge", " flaky "])
    assert flt == TagFilter(include=["slow", "smoke"], exclude=["edge", "flaky"])
    assert make_tag_filter(None, None) is None


@pytest.mark.parametrize("value", ["", " , ,"])
def test_an_option_that_names_no_tag_is_an_error(value: str) -> None:
    with pytest.raises(ValueError, match="--exclude-tags needs at least one tag"):
        make_tag_filter(None, [value])


def test_a_case_needs_one_included_tag_and_none_of_the_excluded() -> None:
    def case(*tags: str) -> Case:
        return Case(id="x", input="q", tags=list(tags))

    flt = TagFilter(include=["smoke", "edge"], exclude=["slow"])
    assert flt.selects(case("smoke")) and flt.selects(case("edge", "other"))
    assert not flt.selects(case("smoke", "slow"))  # an exclusion wins
    assert not flt.selects(case()) and not flt.selects(case("other"))
    only_exclude = TagFilter(exclude=["slow"])
    assert only_exclude.selects(case()) and not only_exclude.selects(case("slow"))
    assert not TagFilter(include=["Smoke"]).selects(case("smoke"))  # tags match with case
    assert flt.describe() == "--tags smoke,edge --exclude-tags slow"


# --- run -----------------------------------------------------------------------------------------


def test_a_run_without_a_filter_runs_every_case_and_records_none(tmp_path: Path) -> None:
    run = _run(_suite(tmp_path))
    assert _ids(run) == ["c0", "c1", "c2", "c3", "c4", "c5"]
    assert run.info.tag_filter is None
    assert '"tag_filter": null' in run.model_dump_json(indent=2)


def test_the_filter_keeps_the_cases_in_dataset_order_and_is_recorded(tmp_path: Path) -> None:
    run = _run(_suite(tmp_path), tag_filter=make_tag_filter(["slow", "edge"], ["smoke"]))
    assert _ids(run) == ["c2", "c5"]  # c1 is slow, but it is also smoke
    assert run.info.tag_filter == TagFilter(include=["edge", "slow"], exclude=["smoke"])
    path = save_run(run, tmp_path / "run.json")
    assert load_run(path).info.tag_filter == run.info.tag_filter  # survives the file


def test_the_filter_is_applied_before_limit(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    run = _run(suite, tag_filter=make_tag_filter(["slow"], None), limit=1)
    assert _ids(run) == ["c1"]  # the first slow case, not c0 with slow dropped afterwards
    run = _run(suite, tag_filter=make_tag_filter(["smoke"], None), limit=2)
    assert _ids(run) == ["c0", "c1"]


def test_a_filter_that_selects_nothing_names_the_tags_there_are(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="--tags nope selects no case: the dataset's tags are"):
        _run(_suite(tmp_path), tag_filter=make_tag_filter(["nope"], None))
    untagged = [{"id": "u", "input": {"q": "a"}, "accept": ["A"]}]
    with pytest.raises(DatasetError, match="no case has a tag"):
        _run(_suite(tmp_path, untagged), tag_filter=make_tag_filter(["smoke"], None))


def test_run_takes_repeated_and_comma_separated_tags(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    out = tmp_path / "run.json"
    argv = ["run", str(suite), "--out", str(out), "--quiet", "--no-cache"]
    code = main([*argv, "--tags", "edge", "--tags", "slow,smoke", "--exclude-tags", "slow"])
    assert code == 0
    run = load_run(out)
    assert _ids(run) == ["c0", "c4", "c5"]
    assert run.info.tag_filter == TagFilter(include=["edge", "slow", "smoke"], exclude=["slow"])


def test_run_exits_2_and_writes_nothing_when_no_case_is_selected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path)
    out = tmp_path / "run.json"
    assert main(["run", str(suite), "--out", str(out), "--tags", "nope", "--no-cache"]) == 2
    err = capsys.readouterr().err
    assert "evalproof: error: --tags nope selects no case" in err and not out.exists()
    assert main(["run", str(suite), "--out", str(out), "--tags", ""]) == 2
    assert "--tags needs at least one tag" in capsys.readouterr().err and not out.exists()


def test_an_old_run_file_without_the_field_loads_with_no_filter(tmp_path: Path) -> None:
    path = save_run(_run(_suite(tmp_path)), tmp_path / "run.json")
    data = json.loads(path.read_text("utf-8"))
    del data["info"]["tag_filter"]
    path.write_text(json.dumps(data), "utf-8")
    assert load_run(path).info.tag_filter is None


# --- pairwise ------------------------------------------------------------------------------------


def test_pairwise_judges_only_the_selected_cases_and_applies_limit_after_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path)
    a = save_run(_run(suite), tmp_path / "a.json")
    b = save_run(_run(suite), tmp_path / "b.json")
    suite.write_text(  # the judge is added after the runs: only pairwise reads it
        suite.read_text("utf-8") + "judge: {target: 'anthropic:claude-haiku-5-5'}\n", "utf-8"
    )
    judge = ScriptedProvider(['{"winner": "A"}'])
    monkeypatch.setattr("evalproof.cli.make_provider", lambda *args, **kw: judge)
    base = ["pairwise", str(a), str(b), "--suite", str(suite), "--no-cache", "--no-swap"]
    assert main([*base, "--tags", "slow", "--exclude-tags", "smoke"]) == 0
    assert len(judge.calls) == 1  # c2 only
    assert "1 cases" in capsys.readouterr().out
    judge.calls.clear()
    assert main([*base, "--tags", "smoke", "--limit", "2"]) == 0
    assert len(judge.calls) == 2
    judge.calls.clear()
    assert main([*base, "--tags", "nope"]) == 2
    assert "selects no case" in capsys.readouterr().err and judge.calls == []


# --- verify --------------------------------------------------------------------------------------


def _filtered(tmp_path: Path, **kw: Any) -> tuple[Path, Path]:
    suite = _suite(tmp_path)
    run = _run(suite, tag_filter=make_tag_filter(["smoke"], ["slow"]), **kw)
    return suite, save_run(run, tmp_path / "run.json")


def test_a_filtered_run_verifies_against_the_same_filter(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _filtered(tmp_path)
    assert main(["verify", str(run), str(suite)]) == 0
    out = capsys.readouterr().out
    assert "the 2 cases its tag filter (--tags smoke --exclude-tags slow) selects" in out
    result = verify_run(load_run(run), load_config(suite))
    assert result.ok and result.cases == 2
    assert "all 6 cases" in verify_markdown(
        verify_run(_run(suite), load_config(suite)), "r.json", "s.yaml"
    )


def test_a_filtered_run_still_fails_when_a_selected_case_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _filtered(tmp_path, limit=1)
    assert main(["verify", str(run), str(suite)]) == 4
    out = capsys.readouterr().out
    assert "1 case(s) in the dataset are missing from the run" in out and "`c4`" in out


def test_a_selected_case_that_arrives_later_is_missing_from_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, run = _filtered(tmp_path)
    more = [*CASES, {"id": "c6", "input": {"q": "g"}, "accept": ["G"], "tags": ["smoke"]}]
    suite = _suite(tmp_path, more)
    assert main(["verify", str(run), str(suite)]) == 4
    assert "`c6`" in capsys.readouterr().out
    # A new case the filter does not select does not matter.
    quiet = [*CASES, {"id": "c7", "input": {"q": "h"}, "accept": ["H"], "tags": ["other"]}]
    assert main(["verify", str(run), str(_suite(tmp_path, quiet))]) == 0


def test_losing_or_gaining_a_filtered_tag_is_a_change(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, run = _filtered(tmp_path)
    lost = [{**c, "tags": []} if c["id"] == "c4" else c for c in CASES]
    assert main(["verify", str(run), str(_suite(tmp_path, lost))]) == 4
    out = capsys.readouterr().out
    assert "changed since the run was made (content or tags)" in out and "`c4`" in out
    assert "not in the dataset any more" not in out  # it is still in the dataset
    gained = [{**c, "tags": ["slow", "smoke"]} if c["id"] == "c0" else c for c in CASES]
    assert main(["verify", str(run), str(_suite(tmp_path, gained))]) == 4
    assert "`c0`" in capsys.readouterr().out


def test_a_filtered_run_holding_a_case_outside_its_filter_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path)
    full = _run(suite)
    full.info.tag_filter = TagFilter(include=["smoke"], exclude=["slow"])  # claims to be filtered
    path = save_run(full, tmp_path / "run.json")
    assert main(["verify", str(path), str(suite)]) == 4
    out = capsys.readouterr().out
    assert "4 case(s) in the run are outside the run's tag filter" in out
    assert "(--tags smoke --exclude-tags slow)" in out


def test_a_filter_that_selects_nothing_now_fails_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, run = _filtered(tmp_path)
    untagged = [{k: v for k, v in c.items() if k != "tags"} for c in CASES]
    assert main(["verify", str(run), str(_suite(tmp_path, untagged))]) == 4
    assert "tag filter (--tags smoke --exclude-tags slow) selects no case any more" in (
        capsys.readouterr().out
    )


# --- compare -------------------------------------------------------------------------------------


def test_compare_warns_when_the_runs_used_different_filters(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    full = _run(suite)
    smoke = _run(suite, tag_filter=make_tag_filter(["smoke"], None))
    cmp = compare_runs(full, smoke)
    assert cmp.tag_filter_change == (
        "the runs used different tag filters (baseline: no filter; candidate: --tags smoke)"
    )
    note = dataset_change_note(cmp)
    assert note is not None and note.startswith(cmp.tag_filter_change)
    assert "gated on the 3 cases both runs share" in note  # the usual note follows
    assert cmp.paired_ids == ["c0", "c1", "c4"]


def test_compare_says_nothing_when_the_filters_match_or_are_both_absent(tmp_path: Path) -> None:
    suite = _suite(tmp_path)
    for flt in (None, make_tag_filter(["slow", "smoke"], None)):
        cmp = compare_runs(_run(suite, tag_filter=flt), _run(suite, tag_filter=flt))
        assert cmp.tag_filter_change is None and dataset_change_note(cmp) is None
    # The same filter written in another order is the same filter.
    one = _run(suite, tag_filter=make_tag_filter(["smoke,slow"], None))
    two = _run(suite, tag_filter=make_tag_filter(["slow", "smoke"], None))
    assert compare_runs(one, two).tag_filter_change is None


def test_the_gate_refuses_runs_made_with_different_tag_filters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path)
    full = save_run(_run(suite), tmp_path / "full.json")
    smoke = save_run(_run(suite, tag_filter=make_tag_filter(["smoke"], None)), tmp_path / "s.json")
    code = main(["compare", str(full), str(smoke), "--suite", str(suite), "--gate"])
    out = capsys.readouterr().out
    # A filter does not show in the pull request's diff, so a narrowed run cannot pass the gate
    # on the subset it kept: the gate cannot decide (exit 4).
    assert code == 4
    assert "the runs used different tag filters" in out
    assert "same --tags and --exclude-tags" in out
    # Without --gate the report still compares the shared cases and warns.
    assert main(["compare", str(full), str(smoke), "--suite", str(suite)]) == 0
    assert "different tag filters" in capsys.readouterr().out
    # Two runs made with the same filter gate as usual.
    smoke2 = save_run(
        _run(suite, tag_filter=make_tag_filter(["smoke"], None)), tmp_path / "s2.json"
    )
    assert main(["compare", str(smoke), str(smoke2), "--suite", str(suite), "--gate"]) == 0
    code = main(["compare", str(full), str(full), "--suite", str(suite), "--gate"])
    assert "different tag filters" not in capsys.readouterr().out
