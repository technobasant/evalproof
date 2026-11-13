"""Absolute gate checks (min_pass_rate, must_pass, max_new_failures) and the exit-code fixes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import evalproof.cli as cli
from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig, load_config
from evalproof.gate import (
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    absolute_checks,
    decide,
)
from evalproof.runner import save_run
from tests.conftest import make_run


def _ids(n: int) -> list[str]:
    return [f"c{i:02d}" for i in range(n)]


def _all(ids: list[str], passed: bool = True, **kw: bool) -> dict[str, dict[str, bool | None]]:
    return {i: {"m": passed, **kw} for i in ids}


def _decide(base, cand, **gate):
    cmp = compare_runs(base, cand)
    return decide(cmp, GateConfig(metric="m", min_cases=10, **gate), cand)


# --- min_pass_rate ------------------------------------------------------------------------


def test_min_pass_rate_fails_below_and_passes_at_the_floor() -> None:
    ids = _ids(30)
    base = make_run(_all(ids))
    cand = make_run({i: {"m": n < 21} for n, i in enumerate(ids)})  # 21/30 = 0.700
    d = _decide(base, cand, min_pass_rate=0.8, alpha=0.0001)
    assert d.exit_code == EXIT_REGRESSION
    assert any("pass rate 0.700 below the minimum 0.800" in r for r in d.reasons)
    assert _decide(base, cand, min_pass_rate=0.7, alpha=0.0001).exit_code == EXIT_PASS


def test_a_float_applies_to_every_gated_metric_and_a_dict_to_its_keys() -> None:
    ids = _ids(30)
    base = make_run({i: {"m": True, "n": True} for i in ids})
    cand = make_run({i: {"m": True, "n": n < 15} for n, i in enumerate(ids)})
    cmp = compare_runs(base, cand)
    both = GateConfig(metrics=["m", "n"], min_cases=10, min_pass_rate=0.9, alpha=1e-9)
    failures, _ = absolute_checks(cand, both)
    assert len(failures) == 1 and failures[0].startswith("n: pass rate 0.500")
    only_m = GateConfig(metrics=["m", "n"], min_cases=10, min_pass_rate={"m": 0.9}, alpha=1e-9)
    assert absolute_checks(cand, only_m)[0] == []
    assert decide(cmp, both, cand).exit_code == EXIT_REGRESSION


def test_errored_cases_count_as_failures() -> None:
    ids = _ids(10)
    # The scorers passed every case, but two of them had a system error: 8/10, not 10/10.
    cand = make_run(_all(ids), errors={"c00", "c01"})
    failures, _ = absolute_checks(cand, GateConfig(metric="m", min_pass_rate=0.9))
    assert failures and "pass rate 0.800 below the minimum 0.900 (8/10 cases)" in failures[0]


def test_undecided_cases_are_left_out_of_the_rate() -> None:
    cand = make_run({**_all(_ids(8)), "u1": {"m": None}, "u2": {"m": None}})
    assert absolute_checks(cand, GateConfig(metric="m", min_pass_rate=1.0)) == ([], [])


def test_a_metric_with_no_decided_scores_fails() -> None:
    cand = make_run({"a": {"m": None}, "b": {"m": None}}, errors={"b"})  # undecided, errored
    failures, _ = absolute_checks(cand, GateConfig(metric="m", min_pass_rate=0.5))
    assert "pass rate 0.000 below the minimum 0.500 (0/1 cases)" in failures[0]
    undecided = make_run({"a": {"m": None}, "b": {"m": None}})
    failures, _ = absolute_checks(undecided, GateConfig(metric="m", min_pass_rate=0.5))
    assert "no case has a decided pass/fail score" in failures[0]


def test_min_pass_rate_on_a_continuous_metric_cannot_gate() -> None:
    # A setup problem, not a regression: exit 4 with the fix, never exit 1.
    ids = _ids(12)
    base = make_run({i: {"m": 0.9} for i in ids})
    gate = GateConfig(metric="m", min_cases=10, min_pass_rate=0.5)
    assert absolute_checks(base, gate) == ([], [])
    d = decide(compare_runs(base, base), gate, base)
    assert d.exit_code == EXIT_CANNOT_GATE
    assert "min_pass_rate needs a pass/fail metric" in d.reasons[0]


def test_max_new_failures_on_a_continuous_metric_warns() -> None:
    ids = _ids(12)
    base = make_run({i: {"m": 0.9} for i in ids})
    cand = make_run({i: {"m": 0.1 if n < 3 else 0.9} for n, i in enumerate(ids)})
    gate = GateConfig(metric="m", min_cases=10, max_new_failures=0, max_drop=1.0)
    d = decide(compare_runs(base, cand), gate, cand)
    assert any("max_new_failures counts pass/fail cases" in w for w in d.warnings)


def test_min_pass_rate_is_validated() -> None:
    with pytest.raises(ValidationError, match="between 0 and 1"):
        GateConfig(metric="m", min_pass_rate=80)
    with pytest.raises(ValidationError, match="between 0 and 1"):
        GateConfig(metric="m", min_pass_rate={"m": -0.1})
    with pytest.raises(ValidationError, match=">= 0"):
        GateConfig(metric="m", max_new_failures=-1)


def test_the_suite_checks_min_pass_rate_names(tmp_path: Path) -> None:
    (tmp_path / "cases.jsonl").write_text('{"id": "a", "input": "q"}\n', "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:x:y'}\n"
        "scorers: [{name: m, type: contains}]\n"
        "gate: {metric: m, min_pass_rate: {typo: 0.5}}\n",
        "utf-8",
    )
    with pytest.raises(Exception, match=r"gate\.min_pass_rate 'typo' is not a scorer name"):
        load_config(suite)


# --- must_pass ----------------------------------------------------------------------------


def test_a_failing_must_pass_case_fails_the_gate() -> None:
    ids = _ids(30)
    tags = {"c03": ["safety"], "c04": ["safety"], "c05": ["other"]}
    base = make_run(_all(ids), tags=tags)
    outcomes = _all(ids)
    outcomes["c04"] = {"m": False}
    outcomes["c05"] = {"m": False}  # not must-pass: one regression, within tolerance
    cand = make_run(outcomes, tags=tags)
    d = _decide(base, cand, must_pass=["safety"])
    assert d.exit_code == EXIT_REGRESSION
    assert any("1 must-pass case(s)" in r and "`c04`" in r and "c05" not in r for r in d.reasons)
    assert _decide(base, cand, must_pass=["other"]).exit_code == EXIT_REGRESSION
    assert _decide(base, cand).exit_code == EXIT_PASS


def test_an_errored_or_undecided_must_pass_case_fails_too() -> None:
    tags = {"a": ["safety"], "b": ["safety"], "c": ["safety"]}
    gate = GateConfig(metric="m", must_pass=["safety"])
    errored = make_run({"a": {"m": True}, "b": {"m": True}}, tags=tags, errors={"b"})
    assert "`b`" in absolute_checks(errored, gate)[0][0]
    undecided = make_run({"a": {"m": True}, "c": {"m": None}}, tags=tags)
    assert "`c`" in absolute_checks(undecided, gate)[0][0]
    assert absolute_checks(make_run({"a": {"m": True}}, tags=tags), gate) == ([], [])


def test_must_pass_needs_every_gated_metric_and_lists_at_most_ten() -> None:
    ids = _ids(14)
    run = make_run({i: {"m": True, "n": False} for i in ids}, tags={i: ["core"] for i in ids})
    gate = GateConfig(metrics=["m", "n"], must_pass=["core"])
    (reason,), _ = absolute_checks(run, gate)
    assert reason.startswith("14 must-pass case(s)") and reason.endswith("and 4 more")
    assert reason.count("`c") == 10


def test_must_pass_on_a_continuous_metric_fails_only_errors_and_undecided() -> None:
    # A continuous score has no pass/fail, so a scored case is not a failure; an errored or
    # undecided one still is, and the gate says must_pass checked nothing more.
    tags = {i: ["core"] for i in "abcd"}
    gate = GateConfig(metric="m", must_pass=["core"])
    scored = make_run({"a": {"m": 0.2}, "b": {"m": 0.9}}, tags=tags)
    failures, warnings = absolute_checks(scored, gate)
    assert failures == []
    assert any("must_pass needs a pass/fail gated metric" in w for w in warnings)
    mixed = make_run({"a": {"m": 0.2}, "c": {"m": None}, "d": {"m": 0.7}}, tags=tags, errors={"d"})
    failures, warnings = absolute_checks(mixed, gate)
    assert len(failures) == 1 and "`c`" in failures[0] and "`d`" in failures[0]
    assert "`a`" not in failures[0]
    assert any("must_pass needs a pass/fail gated metric" in w for w in warnings)


def test_a_must_pass_tag_no_case_carries_is_a_warning() -> None:
    run = make_run({"a": {"m": True}}, tags={"a": ["real"]})
    failures, warnings = absolute_checks(run, GateConfig(metric="m", must_pass=["typo"]))
    assert failures == [] and warnings == ["no case carries the must_pass tag 'typo'"]


def test_an_absolute_check_without_the_candidate_cannot_be_decided() -> None:
    run = make_run(_all(_ids(12)))
    d = decide(compare_runs(run, run), GateConfig(metric="m", min_cases=10, must_pass=["x"]))
    assert d.exit_code == EXIT_CANNOT_GATE and "need the candidate run" in d.reasons[0]


# --- max_new_failures ---------------------------------------------------------------------


def test_max_new_failures_zero_catches_five_deterministic_regressions_in_thirty() -> None:
    ids = _ids(30)
    base = make_run(_all(ids))
    cand = make_run({i: {"m": n >= 5} for n, i in enumerate(ids)})  # five regress, none improve
    # Five against none is p = 0.0625 on the exact test, so the statistics alone let it pass.
    assert _decide(base, cand).exit_code == EXIT_PASS
    d = _decide(base, cand, max_new_failures=0, max_drop=0.5)
    assert d.exit_code == EXIT_REGRESSION
    assert "5 case(s) newly failing, more than the 0 allowed" in d.reasons[0]
    assert "`c00`" in d.reasons[0]
    assert _decide(base, cand, max_new_failures=5, max_drop=0.5).exit_code == EXIT_PASS


def test_max_new_failures_ignores_continuous_metrics() -> None:
    ids = _ids(12)
    base = make_run({i: {"m": 0.9} for i in ids})
    cand = make_run({i: {"m": 0.8} for i in ids})
    assert _decide(base, cand, max_new_failures=0, max_drop=0.5).exit_code == EXIT_PASS


# --- through the CLI ----------------------------------------------------------------------


def test_compare_flags_set_the_absolute_checks(tmp_path: Path, capsys) -> None:
    ids = _ids(30)
    tags = {"c00": ["safety"]}
    base = save_run(make_run(_all(ids), tags=tags), tmp_path / "b.json")
    cand = save_run(
        make_run({i: {"m": n >= 5} for n, i in enumerate(ids)}, tags=tags), tmp_path / "c.json"
    )
    argv = ["compare", str(base), str(cand), "--metric", "m", "--min-cases", "10", "--gate"]
    assert main(argv) == EXIT_PASS
    assert main([*argv, "--max-new-failures", "0"]) == EXIT_REGRESSION
    assert main([*argv, "--min-pass-rate", "0.9"]) == EXIT_REGRESSION
    assert main([*argv, "--must-pass", "safety"]) == EXIT_REGRESSION
    out = capsys.readouterr().out
    assert "newly failing" in out and "pass rate 0.833 below the minimum 0.900" in out
    assert "must-pass case(s)" in out


def test_an_unknown_metric_is_a_usage_error_naming_the_scorers(tmp_path: Path, capsys) -> None:
    run = make_run(_all(_ids(12), n=True))
    path = save_run(run, tmp_path / "r.json")
    code = main(["compare", str(path), str(path), "--metric", "typo", "--gate"])
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "'typo'" in err and "available: m, n" in err


def test_a_metric_present_but_with_no_paired_scores_stays_exit_4(tmp_path: Path) -> None:
    only_a = save_run(make_run({i: {"a": True} for i in _ids(12)}), tmp_path / "a.json")
    only_b = save_run(make_run({i: {"b": True} for i in _ids(12)}), tmp_path / "b.json")
    assert (
        main(["compare", str(only_a), str(only_b), "--metric", "a", "--gate"]) == EXIT_CANNOT_GATE
    )


# --- run --gate, run exit codes -----------------------------------------------------------

ANSWER = """\
import os
def ask(x):
    marker = os.environ.get("EP_MARKER")
    if marker:
        open(marker, "a").write("x")
    if os.environ.get("EP_FAIL") == "all":
        raise RuntimeError("the model is down")
    if os.environ.get("EP_FAIL") == "some" and x["q"] in ("q0", "q1"):
        raise RuntimeError("one case failed")
    return x["q"].upper()
"""


def _suite(tmp_path: Path, gate: str = "{metric: m}") -> Path:
    cases = [
        {"id": f"c{i}", "input": {"q": f"q{i}"}, "accept": [f"Q{i}"] if i % 2 == 0 else ["no"],
         "tags": ["core"] if i < 2 else []}
        for i in range(10)
    ]  # fmt: skip
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    (tmp_path / "answer.py").write_text(ANSWER, "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:answer:ask'}\n"
        "scorers:\n  - {name: m, type: contains, params: {any_of_field: accept}}\n"
        f"gate: {gate}\n",
        "utf-8",
    )
    return suite


def test_run_gate_applies_min_pass_rate_with_no_baseline(tmp_path: Path, capsys) -> None:
    suite = _suite(tmp_path, "{metric: m, min_pass_rate: 0.6}")  # the cases pass 5 of 10
    out = tmp_path / "r.json"
    code = main(["run", str(suite), "--out", str(out), "--quiet", "--gate"])
    assert code == EXIT_REGRESSION and out.exists()
    text = capsys.readouterr().out
    assert "pass rate 0.500 below the minimum 0.600" in text and "❌ fail" in text
    ok = _suite(tmp_path, "{metric: m, min_pass_rate: 0.5}")
    assert main(["run", str(ok), "--out", str(out), "--quiet", "--gate"]) == EXIT_PASS
    assert "✅ pass" in capsys.readouterr().out


def test_run_gate_applies_must_pass(tmp_path: Path, capsys) -> None:
    suite = _suite(tmp_path, "{metric: m, must_pass: [core]}")  # c1 is core and fails
    code = main(["run", str(suite), "--out", str(tmp_path / "r.json"), "--quiet", "--gate"])
    assert code == EXIT_REGRESSION
    assert "1 must-pass case(s)" in capsys.readouterr().out


def test_run_gate_with_min_pass_rate_on_a_continuous_metric_exits_4(tmp_path: Path, capsys) -> None:
    (tmp_path / "cases.jsonl").write_text(
        "".join(
            json.dumps({"id": f"c{i}", "input": "q", "relevant": ["a"]}) + "\n" for i in range(3)
        ),
        "utf-8",
    )
    (tmp_path / "rag.py").write_text(
        "def ask(x):\n    return {'output': 'x', 'retrieved': ['a']}\n", "utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:rag:ask'}\n"
        "scorers: [{name: r, type: mrr}]\ngate: {metric: r, min_pass_rate: 0.5}\n",
        "utf-8",
    )
    code = main(["run", str(suite), "--out", str(tmp_path / "r.json"), "--quiet", "--gate"])
    out = capsys.readouterr().out
    assert code == EXIT_CANNOT_GATE
    assert "cannot gate" in out and "min_pass_rate needs a pass/fail metric" in out


def test_without_gate_the_run_does_not_apply_the_checks(tmp_path: Path) -> None:
    suite = _suite(tmp_path, "{metric: m, min_pass_rate: 0.99}")
    assert main(["run", str(suite), "--out", str(tmp_path / "r.json"), "--quiet"]) == EXIT_PASS


@pytest.mark.parametrize("gate", [None, "{metric: m}", "{metric: m, max_new_failures: 0}"])
def test_run_gate_without_absolute_checks_is_a_usage_error(
    tmp_path: Path, capsys, monkeypatch, gate: str | None
) -> None:
    suite = _suite(tmp_path, gate or "{metric: m}")
    if gate is None:
        text = suite.read_text("utf-8")
        suite.write_text(text[: text.index("gate:")], "utf-8")
    marker = tmp_path / "marker"
    monkeypatch.setenv("EP_MARKER", str(marker))
    code = main(["run", str(suite), "--out", str(tmp_path / "r.json"), "--quiet", "--gate"])
    assert code == EXIT_USAGE and not marker.exists()  # refused before any case ran
    assert "absolute checks" in capsys.readouterr().err


def test_a_run_where_every_case_errored_writes_the_file_and_exits_2(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    monkeypatch.setenv("EP_FAIL", "all")
    out = tmp_path / "r.json"
    code = main(["run", str(_suite(tmp_path)), "--out", str(out), "--quiet"])
    assert code == EXIT_USAGE and out.exists()
    err = capsys.readouterr().err
    assert "every case errored; the first error: RuntimeError: the model is down" in err


def test_a_run_with_some_errors_still_exits_0(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EP_FAIL", "some")
    assert main(["run", str(_suite(tmp_path)), "--out", str(tmp_path / "r.json"), "--quiet"]) == 0


def test_the_out_directory_is_checked_before_any_case_runs(
    tmp_path: Path, capsys, monkeypatch
) -> None:
    suite = _suite(tmp_path)
    marker = tmp_path / "marker"
    monkeypatch.setenv("EP_MARKER", str(marker))
    blocker = tmp_path / "a-file"
    blocker.write_text("not a directory", "utf-8")
    code = main(["run", str(suite), "--out", str(blocker / "r.json"), "--quiet"])
    assert code == EXIT_USAGE and not marker.exists()
    assert "cannot write the run file" in capsys.readouterr().err
    # An --out that is an existing directory is refused up front too.
    code = main(["run", str(suite), "--out", str(tmp_path), "--quiet"])
    assert code == EXIT_USAGE and not marker.exists()
    assert "it is a directory" in capsys.readouterr().err
    # A missing parent is created, and the check leaves nothing behind.
    deep = tmp_path / "new" / "dir" / "r.json"
    assert main(["run", str(suite), "--out", str(deep), "--quiet"]) == EXIT_PASS
    assert [p.name for p in deep.parent.iterdir()] == ["r.json"]


# --- main() and calibrate -----------------------------------------------------------------


def test_an_unexpected_exception_is_a_one_line_error_with_a_debug_hint(monkeypatch, capsys) -> None:
    def boom(_args):
        raise KeyError("surprise")

    monkeypatch.setattr(cli, "cmd_report", boom)
    monkeypatch.delenv("EVALPROOF_DEBUG", raising=False)
    assert main(["report", "x.json"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "evalproof: error: KeyError: 'surprise'" in err
    assert "set EVALPROOF_DEBUG=1 for the traceback" in err
    assert "Traceback" not in err


def test_evalproof_debug_re_raises(monkeypatch) -> None:
    def boom(_args):
        raise KeyError("surprise")

    monkeypatch.setattr(cli, "cmd_report", boom)
    monkeypatch.setenv("EVALPROOF_DEBUG", "1")
    with pytest.raises(KeyError):
        main(["report", "x.json"])


def test_ctrl_c_exits_130(monkeypatch, capsys) -> None:
    def interrupted(_args):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "cmd_report", interrupted)
    assert main(["report", "x.json"]) == 130
    assert "interrupted" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "labels.jsonl:2: not valid JSON"),
        ('{"id": "a"}', 'labels.jsonl:2: each line needs an "id" and a "label"'),
        ('{"label": true}', 'labels.jsonl:2: each line needs an "id" and a "label"'),
        ("[1, 2]", 'labels.jsonl:2: each line needs an "id" and a "label"'),
    ],
)
def test_calibrate_names_the_bad_labels_line(
    tmp_path: Path, capsys, monkeypatch, line: str, message: str
) -> None:
    monkeypatch.chdir(tmp_path)
    run = save_run(make_run(_all(_ids(4))), tmp_path / "r.json")
    Path("labels.jsonl").write_text('{"id": "c00", "label": true}\n' + line + "\n", "utf-8")
    code = main(["calibrate", str(run), "--judge-scorer", "m", "--labels", "labels.jsonl"])
    assert code == EXIT_USAGE
    assert message in capsys.readouterr().err
