"""JUnit XML reports: ``run --junit`` and ``compare --gate --junit``. Every test parses the XML back
and checks the structure and the counts a CI system reads."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    GateCheck,
    decide,
    refusal,
)
from evalproof.junit import (
    OUTPUT_CHARS,
    clean,
    gate_testsuite,
    run_testsuite,
    to_xml,
    write_junit,
)
from evalproof.junit import testsuites as wrap  # not `testsuites`: pytest would collect it
from evalproof.models import Run
from evalproof.report import run_markdown, to_json
from evalproof.runner import save_run
from tests.conftest import make_run

IDS = [f"c{i:02d}" for i in range(30)]
GATE = ["--metric", "m", "--min-cases", "10"]


# --- helpers ------------------------------------------------------------------------------------


def _parse(path: Path | str) -> ET.Element:
    return ET.parse(path).getroot()


def _roundtrip(root: ET.Element) -> ET.Element:
    """The document as a CI system would receive it."""
    return ET.fromstring(to_xml(root).encode("utf-8"))


def _cases(root: ET.Element) -> dict[str, ET.Element]:
    return {tc.get("name", ""): tc for tc in root.iter("testcase")}


def _outcome(case: ET.Element) -> str:
    for tag in ("failure", "error", "skipped"):
        if case.find(tag) is not None:
            return tag
    return "pass"


def _counts(suite: ET.Element) -> tuple[int, int, int, int]:
    return tuple(int(suite.get(k, "-1")) for k in ("tests", "failures", "errors", "skipped"))  # type: ignore[return-value]


def _props(suite: ET.Element) -> dict[str, str]:
    return {p.get("name", ""): p.get("value", "") for p in suite.iter("property")}


def _single(failing: int, *, system: str, latency: float = 10.0) -> Run:
    """One trial per case; the first ``failing`` cases fail ``m``."""
    return make_run(
        {cid: {"m": i >= failing} for i, cid in enumerate(IDS)}, system=system, latency=latency
    )


def _repeated(rows: dict[str, list[bool]], *, system: str = "sys") -> Run:
    results = []
    for cid, values in rows.items():
        for trial, value in enumerate(values):
            one = make_run({cid: {"m": value}}).results[0]
            results.append(one.model_copy(update={"trial": trial}))
    info = make_run({"x": {}}, system=system).info.model_copy(
        update={"trials": len(next(iter(rows.values())))}
    )
    return Run(info=info, results=results)


def _save(tmp_path: Path, name: str, run: Run) -> str:
    return str(save_run(run, tmp_path / f"{name}.json"))


# --- a run -------------------------------------------------------------------------------------


def test_an_all_pass_run_is_one_suite_with_a_testcase_per_case() -> None:
    run = make_run({"a": {"m": True}, "b": {"m": True}, "c": {"m": 0.7}}, latency=250.0)
    root = _roundtrip(wrap([run_testsuite(run)]))
    assert root.tag == "testsuites"
    assert (root.get("tests"), root.get("failures"), root.get("errors"), root.get("skipped")) == (
        "3",
        "0",
        "0",
        "0",
    )
    (suite,) = root.findall("testsuite")
    assert suite.get("name") == "t.sys"
    assert _counts(suite) == (3, 0, 0, 0)
    assert suite.get("time") == "0.750" and root.get("time") == "0.750"
    cases = _cases(suite)
    assert list(cases) == ["a", "b", "c"]
    for case in cases.values():
        assert case.get("classname") == "t.sys"
        assert case.get("time") == "0.250"
        assert _outcome(case) == "pass"
        assert case.findtext("system-out") == "x"


def test_the_suite_carries_the_run_facts_as_properties() -> None:
    run = make_run({"a": {"m": True}}, dataset_hash="abc123")
    run.info.environment["judge_target"] = "claude-code:haiku"
    props = _props(_roundtrip(wrap([run_testsuite(run)])).find("testsuite"))  # type: ignore[arg-type]
    assert props["evalproof.version"] == "0.1.0"
    assert props["dataset.hash"] == "abc123"
    assert props["system"] == "sys" and props["trials"] == "1" and props["seed"] == "0"
    assert props["environment.judge_target"] == "claude-code:haiku"
    assert "variant" not in props and "suite.hash" not in props  # empty values are left out


def test_a_failed_case_is_a_failure_that_lists_the_failing_scorers_and_scores() -> None:
    run = make_run(
        {"ok": {"m": True, "n": True}, "bad": {"m": False, "n": 0.4, "k": False}},
    )
    run.results[1].scores["m"].reason = "the answer was wrong"
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None and _counts(suite) == (2, 1, 0, 0)
    cases = _cases(suite)
    assert _outcome(cases["ok"]) == "pass"
    failure = cases["bad"].find("failure")
    assert failure is not None
    assert failure.get("message") == "failed m (0), k (0)"
    assert "m: the answer was wrong" in (failure.text or "")
    assert "k: score 0" in (failure.text or "")  # no reason recorded: the score stands in


def test_the_failures_are_exactly_the_cases_the_run_report_lists_as_failed() -> None:
    run = make_run(
        {
            "pass": {"m": True},
            "fail": {"m": False},
            "undecided": {"m": None},
            "errored": {"m": False},
            "continuous": {"m": 0.2},
        },
        errors={"errored"},
    )
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None
    flagged = {name for name, case in _cases(suite).items() if _outcome(case) != "pass"}
    listed = {
        line.split("`")[1] for line in run_markdown(run, 10).splitlines() if line.startswith("- `")
    }
    assert flagged == listed == {"fail", "undecided", "errored"}


def test_an_errored_case_is_an_error_with_the_error_text() -> None:
    run = make_run({"a": {"m": True}, "e": {"m": False}}, errors={"e"})
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None and _counts(suite) == (2, 0, 1, 0)
    error = _cases(suite)["e"].find("error")
    assert error is not None
    assert error.get("message") == "boom" and error.text == "boom"
    assert error.get("type") == "system error"
    assert _cases(suite)["e"].find("failure") is None
    assert _cases(suite)["e"].find("system-out") is None  # an errored case has no output


def test_an_undecided_case_is_skipped_with_a_message() -> None:
    run = make_run({"a": {"m": True}, "u": {"m": None, "n": True}})
    run.results[1].scores["m"].reason = "the judge reply was not JSON"
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None and _counts(suite) == (2, 0, 0, 1)
    skipped = _cases(suite)["u"].find("skipped")
    assert skipped is not None
    assert skipped.get("message") == "undecided on m: no scorer reached a verdict"
    assert "the judge reply was not JSON" in (skipped.text or "")


def test_a_case_that_failed_one_scorer_and_is_undecided_on_another_is_a_failure() -> None:
    run = make_run({"a": {"m": False, "n": None}})
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None and _counts(suite) == (1, 1, 0, 0)


def test_trials_make_a_testcase_per_trial() -> None:
    run = _repeated({"a": [True, False, True], "b": [True, True, True]})
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None and _counts(suite) == (6, 1, 0, 0)
    cases = _cases(suite)
    assert list(cases) == [
        "a [trial 0]",
        "a [trial 1]",
        "a [trial 2]",
        "b [trial 0]",
        "b [trial 1]",
        "b [trial 2]",
    ]
    assert _outcome(cases["a [trial 1]"]) == "failure"
    assert _outcome(cases["a [trial 0]"]) == "pass"
    assert _props(suite)["trials"] == "3"


def test_one_trial_keeps_the_plain_case_id() -> None:
    suite = _roundtrip(wrap([run_testsuite(make_run({"a": {"m": True}}))])).find("testsuite")
    assert suite is not None and list(_cases(suite)) == ["a"]


def test_a_run_of_a_variant_is_named_after_the_variant_and_runs_sum_in_the_totals() -> None:
    base = make_run({"a": {"m": True}, "b": {"m": False}})
    base.info.variant = "base"
    loud = make_run({"a": {"m": True}, "b": {"m": True}, "c": {"m": False}})
    loud.info.variant = "loud"
    plain = make_run({"a": {"m": True}}, system="other")
    root = _roundtrip(wrap(run_testsuite(r) for r in (base, loud, plain)))
    names = [s.get("name") for s in root.findall("testsuite")]
    assert names == ["t.base", "t.loud", "t.other"]
    assert {c.get("classname") for c in root.findall("testsuite")[1].iter("testcase")} == {"t.loud"}
    assert (root.get("tests"), root.get("failures")) == ("6", "2")
    assert _props(root.findall("testsuite")[0])["variant"] == "base"


def test_characters_xml_cannot_hold_are_stripped_from_every_value() -> None:
    nasty = "a\x00b\x07c\x1bd\ud800e￾f\x0bg" + "\U0001f600" + "\r\nh\ti"
    run = make_run({"id\x01": {"m": False}, "err\x02": {"m": False}}, errors={"err\x02"})
    run.results[0].output.output = nasty
    run.results[0].scores["m"].reason = nasty
    run.results[1].output.error = nasty
    run.info.environment["k\x03"] = nasty
    run.info.system = "sys\x04"
    root = _roundtrip(wrap([run_testsuite(run)]))  # parses: nothing invalid is left
    expected = "abcdefg\U0001f600\nh\ti"
    suite = root.find("testsuite")
    assert suite is not None and suite.get("name") == "t.sys"
    cases = _cases(suite)
    assert list(cases) == ["id", "err"]
    assert cases["id"].findtext("system-out") == expected
    failure = cases["id"].find("failure")
    assert failure is not None and failure.text == f"m: {expected}"
    error = cases["err"].find("error")
    assert error is not None and error.text == expected
    assert _props(suite)["environment.k"] == expected
    assert clean("ok\x00") == "ok" and clean("tab\tnl\n") == "tab\tnl\n"


def test_a_long_output_is_truncated_to_about_two_thousand_characters() -> None:
    run = make_run({"a": {"m": True}})
    run.results[0].output.output = "z" * (OUTPUT_CHARS + 500)
    suite = _roundtrip(wrap([run_testsuite(run)])).find("testsuite")
    assert suite is not None
    out = _cases(suite)["a"].findtext("system-out") or ""
    assert out.startswith("z" * OUTPUT_CHARS) and "500 more characters" in out
    assert len(out) < OUTPUT_CHARS + 60


def test_the_file_is_utf_8_with_a_declaration_and_lf_line_endings(tmp_path: Path) -> None:
    run = make_run({"a": {"m": False}})
    run.results[0].output.output = "café ✓\r\nline two"
    path = tmp_path / "out" / "r.xml"
    path.parent.mkdir()
    write_junit(path, wrap([run_testsuite(run)]))
    raw = path.read_bytes()
    assert raw.startswith(b'<?xml version="1.0" encoding="UTF-8"?>\n<testsuites ')
    assert b"\r" not in raw and raw.endswith(b"</testsuites>\n")
    assert "café ✓\nline two".encode() in raw
    assert _parse(path).tag == "testsuites"


# --- evalproof run --junit ---------------------------------------------------------------------

APP = """
def ask(x, ctx):
    if x["q"] == "boom":
        raise RuntimeError("the system fell over")
    return {"output": x["q"]}
"""


def _suite(tmp_path: Path, questions: list[str], extra: str = "") -> Path:
    rows = [{"id": f"c{i}", "input": {"q": q}, "accept": ["q0"]} for i, q in enumerate(questions)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (tmp_path / f"junitapp_{tmp_path.name}.py").write_text(APP, "utf-8")
    suite = tmp_path / "suite.yaml"
    target = f"python:junitapp_{tmp_path.name}:ask"
    suite.write_text(
        f"name: s\ndataset: cases.jsonl\nsystem: {{target: '{target}'}}\n"
        "scorers: [{name: m, type: contains, params: {any_of_field: accept}}]\n"
        "gate: {metric: m, min_cases: 2, min_pass_rate: 0.5}\n" + extra,
        "utf-8",
    )
    return suite


def test_run_junit_writes_a_testcase_per_case_and_changes_no_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path, ["q0", "q1", "q0"])
    monkeypatch.chdir(tmp_path)
    plain = main(["run", str(suite), "--quiet", "--out", str(tmp_path / "a.json")])
    junit = tmp_path / "reports" / "junit.xml"
    code = main(
        ["run", str(suite), "--quiet", "--out", str(tmp_path / "b.json"), "--junit", str(junit)]
    )
    assert code == plain == EXIT_PASS
    root = _parse(junit)
    (case_suite,) = root.findall("testsuite")
    assert case_suite.get("name") == "s.python:junitapp_" + tmp_path.name + ":ask"
    assert _counts(case_suite) == (3, 1, 0, 0)
    cases = _cases(case_suite)
    assert _outcome(cases["c1"]) == "failure" and _outcome(cases["c0"]) == "pass"
    assert cases["c0"].findtext("system-out") == "q0"


def test_run_junit_with_a_gate_failure_still_exits_one_and_writes_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path, ["q1", "q2", "q0"])
    monkeypatch.chdir(tmp_path)
    junit = tmp_path / "j.xml"
    args = ["run", str(suite), "--quiet", "--gate"]
    assert main([*args, "--out", str(tmp_path / "a.json")]) == EXIT_REGRESSION
    assert (
        main([*args, "--out", str(tmp_path / "b.json"), "--junit", str(junit)]) == EXIT_REGRESSION
    )
    assert _counts(_parse(junit).find("testsuite"))[:2] == (3, 2)  # type: ignore[arg-type]


def test_run_junit_with_trials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _suite(tmp_path, ["q0", "q1"])
    monkeypatch.chdir(tmp_path)
    junit = tmp_path / "j.xml"
    out = str(tmp_path / "r.json")
    assert (
        main(["run", str(suite), "--quiet", "--trials", "2", "--out", out, "--junit", str(junit)])
        == 0
    )
    case_suite = _parse(junit).find("testsuite")
    assert case_suite is not None
    assert list(_cases(case_suite)) == [
        "c0 [trial 0]",
        "c0 [trial 1]",
        "c1 [trial 0]",
        "c1 [trial 1]",
    ]


def test_run_junit_with_variants_is_one_file_with_a_suite_per_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path, ["q0", "q0"], extra="variants:\n  - {name: base}\n  - {name: loud}\n")
    monkeypatch.chdir(tmp_path)
    junit = tmp_path / "j.xml"
    assert main(["run", str(suite), "--all-variants", "--quiet", "--junit", str(junit)]) == 0
    root = _parse(junit)
    assert [s.get("name") for s in root.findall("testsuite")] == ["s.base", "s.loud"]
    assert root.get("tests") == "4"


def test_run_junit_is_written_when_every_case_errors_and_the_exit_stays_two(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _suite(tmp_path, ["boom", "boom"])
    monkeypatch.chdir(tmp_path)
    junit = tmp_path / "j.xml"
    code = main(
        ["run", str(suite), "--quiet", "--out", str(tmp_path / "r.json"), "--junit", str(junit)]
    )
    assert code == EXIT_USAGE
    case_suite = _parse(junit).find("testsuite")
    assert case_suite is not None and _counts(case_suite) == (2, 0, 2, 0)
    error = _cases(case_suite)["c0"].find("error")
    assert error is not None and "the system fell over" in (error.text or "")


def test_run_junit_to_a_directory_is_a_usage_error_before_any_case_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path, ["q0"])
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "r.json"
    assert main(["run", str(suite), "--quiet", "--out", str(out), "--junit", str(tmp_path)]) == 2
    assert "cannot write the JUnit file" in capsys.readouterr().err
    assert not out.exists()


# --- evalproof compare --gate --junit ----------------------------------------------------------


def _compare(
    tmp_path: Path,
    baseline: Run,
    *candidates: Run,
    extra: Iterable[str] = (),
    gate_args: Iterable[str] = GATE,
) -> tuple[int, ET.Element, dict]:
    """Run ``compare --gate --junit`` and ``--json`` and return the exit code, the parsed JUnit
    and the JSON payload."""
    base = _save(tmp_path, "base", baseline)
    paths = [_save(tmp_path, f"cand{i}", c) for i, c in enumerate(candidates)]
    junit, js = tmp_path / "gate.xml", tmp_path / "gate.json"
    argv = ["compare", base, *paths, "--gate", *gate_args, *extra]
    code = main([*argv, "--junit", str(junit), "--json", str(js)])
    return code, _parse(junit), json.loads(js.read_text("utf-8"))


def test_a_passing_gate_is_one_suite_with_a_passing_testcase_per_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, root, payload = _compare(tmp_path, _single(0, system="base"), _single(0, system="cand"))
    assert code == EXIT_PASS
    (suite,) = root.findall("testsuite")
    assert suite.get("name") == "evalproof gate"
    assert _counts(suite) == (1, 0, 0, 0)
    (case,) = suite.findall("testcase")
    assert case.get("name") == "max_drop: m" and case.get("classname") == "evalproof gate"
    assert _outcome(case) == "pass"
    assert "no drop beyond 0.020" in (case.findtext("system-out") or "")
    props = _props(suite)
    assert (
        props["exit_code"] == "0" and props["baseline"] == "base" and props["candidate"] == "cand"
    )
    assert props["gate.metrics"] == "m" and props["gate.min_cases"] == "10"
    assert [c["outcome"] for c in payload["gate"]["checks"]] == ["pass"]
    capsys.readouterr()


def test_a_failed_drop_test_is_a_failure_with_the_terminal_reason(tmp_path: Path) -> None:
    code, root, payload = _compare(tmp_path, _single(0, system="base"), _single(10, system="cand"))
    assert code == EXIT_REGRESSION
    suite = root.find("testsuite")
    assert suite is not None and _counts(suite) == (1, 1, 0, 0)
    failure = suite.find("testcase/failure")
    assert failure is not None
    (reason,) = payload["gate"]["reasons"]
    assert failure.get("message") == reason and failure.text == reason
    assert failure.get("type") == "gate check failed"
    assert _props(suite)["exit_code"] == "1"


def test_several_gated_metrics_get_a_check_each_and_the_holm_text(tmp_path: Path) -> None:
    ids = IDS
    baseline = make_run({c: {"m1": True, "m2": True} for c in ids}, system="base")
    candidate = make_run({c: {"m1": i >= 12, "m2": True} for i, c in enumerate(ids)}, system="cand")
    code, root, payload = _compare(
        tmp_path,
        baseline,
        candidate,
        gate_args=["--metric", "m1", "--metric", "m2", "--min-cases", "10"],
    )
    suite = root.find("testsuite")
    assert suite is not None
    cases = _cases(suite)
    assert set(cases) == {"max_drop: m1", "max_drop: m2"}
    assert (
        _outcome(cases["max_drop: m1"]) == "failure" and _outcome(cases["max_drop: m2"]) == "pass"
    )
    failure = cases["max_drop: m1"].find("failure")
    assert failure is not None and "Holm-adjusted p=" in (failure.get("message") or "")
    assert code == EXIT_REGRESSION
    assert payload["gate"]["reasons"] == [failure.get("message")]


def test_every_configured_check_is_a_testcase_and_failures_match_the_reasons(
    tmp_path: Path,
) -> None:
    baseline = _single(0, system="base")
    candidate = make_run(
        {cid: {"m": i != 0} for i, cid in enumerate(IDS)},
        system="cand",
        tags={"c00": ["critical"]},
        latency=500.0,
    )
    for result in candidate.results:
        result.output.cost_usd = 0.01
    code, root, payload = _compare(
        tmp_path,
        baseline,
        candidate,
        extra=[
            "--min-pass-rate", "0.99",
            "--must-pass", "critical",
            "--max-new-failures", "0",
            "--max-p95-ms", "100",
            "--max-cost", "1.0",
        ],
    )  # fmt: skip
    suite = root.find("testsuite")
    assert suite is not None
    outcomes = {name: _outcome(case) for name, case in _cases(suite).items()}
    assert outcomes == {
        "max_drop: m": "pass",  # one case dropped: below the threshold of significance
        "max_new_failures: m": "failure",
        "min_pass_rate: m": "failure",
        "must_pass": "failure",
        "max_p95_latency_ms": "failure",
        "max_cost_usd": "pass",
    }
    reasons = payload["gate"]["reasons"]
    failures = [c.find("failure") for c in suite.iter("testcase") if _outcome(c) == "failure"]
    assert [f.get("message") for f in failures if f is not None] == reasons
    assert code == EXIT_REGRESSION and _counts(suite) == (6, 4, 0, 0)


def test_a_budget_breach_is_a_failure_and_exits_three(tmp_path: Path) -> None:
    candidate = _single(0, system="cand")
    for result in candidate.results:
        result.output.cost_usd = 1.0
    code, root, _ = _compare(
        tmp_path,
        _single(0, system="base"),
        candidate,
        extra=["--max-cost", "5", "--max-p95-ms", "5"],
    )
    suite = root.find("testsuite")
    assert suite is not None
    outcomes = {name: _outcome(case) for name, case in _cases(suite).items()}
    assert outcomes == {
        "max_drop: m": "pass",
        "max_p95_latency_ms": "failure",
        "max_cost_usd": "failure",
    }
    failure = _cases(suite)["max_cost_usd"].find("failure")
    assert failure is not None and failure.get("message") == "cost $30.0000 exceeds $5.0000"
    assert code == EXIT_BUDGET


def test_a_matrix_prefixes_each_check_with_its_candidate_label(tmp_path: Path) -> None:
    code, root, payload = _compare(
        tmp_path,
        _single(0, system="base"),
        _single(0, system="steady"),
        _single(12, system="worse"),
    )
    (suite,) = root.findall("testsuite")
    assert suite.get("name") == "evalproof gate"
    outcomes = {name: _outcome(case) for name, case in _cases(suite).items()}
    assert outcomes == {"steady / max_drop: m": "pass", "worse / max_drop: m": "failure"}
    failure = _cases(suite)["worse / max_drop: m"].find("failure")
    assert failure is not None
    assert payload["candidates"][1]["gate"]["reasons"] == [failure.get("message")]
    props = _props(suite)
    assert props["candidate.1"] == "steady" and props["candidate.2"] == "worse"
    assert props["exit_code"] == "1" and code == EXIT_REGRESSION
    assert _counts(suite) == (2, 1, 0, 0)


def test_a_gate_that_cannot_decide_reports_an_error_per_reason(tmp_path: Path) -> None:
    code, root, payload = _compare(
        tmp_path,
        _single(0, system="base"),
        _single(0, system="cand"),
        extra=["--min-cases", "100"],
    )
    assert code == EXIT_CANNOT_GATE
    suite = root.find("testsuite")
    assert suite is not None and _counts(suite) == (1, 0, 1, 0)
    (case,) = suite.findall("testcase")
    assert case.get("name") == "min_cases: m"
    error = case.find("error")
    assert error is not None and error.get("type") == "cannot gate"
    assert error.get("message") == payload["gate"]["reasons"][0]
    assert _props(suite)["exit_code"] == "4"


def test_runs_that_cannot_be_compared_at_all_are_one_error_testcase(tmp_path: Path) -> None:
    code, root, payload = _compare(
        tmp_path, _single(0, system="base"), _repeated({"a": [True, True]}, system="cand")
    )
    assert code == EXIT_CANNOT_GATE
    suite = root.find("testsuite")
    assert suite is not None and _counts(suite) == (1, 0, 1, 0)
    case = suite.find("testcase")
    assert case is not None and case.get("name") == "comparison"
    error = case.find("error")
    assert error is not None and error.get("message") == payload["gate"]["reasons"][0]


def test_a_matrix_candidate_that_cannot_be_compared_is_an_error_and_the_rest_still_report(
    tmp_path: Path,
) -> None:
    code, root, _ = _compare(
        tmp_path,
        _single(0, system="base"),
        _single(0, system="fine"),
        _repeated({"a": [True, True]}, system="odd"),
    )
    assert code == EXIT_CANNOT_GATE
    suite = root.find("testsuite")
    assert suite is not None
    outcomes = {name: _outcome(case) for name, case in _cases(suite).items()}
    assert outcomes == {"fine / max_drop: m": "pass", "odd / comparison": "error"}


def test_the_gate_warnings_go_to_the_suite_output(tmp_path: Path) -> None:
    baseline = _single(0, system="base")
    candidate = make_run({cid: {"m": i != 0} for i, cid in enumerate(IDS)}, system="cand")
    _, root, payload = _compare(tmp_path, baseline, candidate)
    suite = root.find("testsuite")
    assert suite is not None
    out = suite.findtext("system-out") or ""
    assert payload["gate"]["warnings"] and out == "\n".join(
        f"warning: {w}" for w in payload["gate"]["warnings"]
    )


# --- --junit without --gate, and exit codes ----------------------------------------------------


@pytest.mark.parametrize("candidates", [1, 2])
def test_junit_without_gate_is_a_usage_error_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], candidates: int
) -> None:
    base = _save(tmp_path, "base", _single(0, system="base"))
    cands = [_save(tmp_path, f"c{i}", _single(0, system=f"c{i}")) for i in range(candidates)]
    junit = tmp_path / "gate.xml"
    code = main(["compare", base, *cands, "--junit", str(junit)])
    assert code == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--junit needs --gate" in err and "no checks to report" in err
    assert not junit.exists()


@pytest.mark.parametrize(
    ("candidate_failing", "extra", "expected"),
    [
        (0, [], EXIT_PASS),
        (12, [], EXIT_REGRESSION),
        (0, ["--max-p95-ms", "1"], EXIT_BUDGET),
        (0, ["--min-cases", "99"], EXIT_CANNOT_GATE),
    ],
)
def test_the_exit_code_is_the_same_with_and_without_junit(
    tmp_path: Path, candidate_failing: int, extra: list[str], expected: int
) -> None:
    base = _save(tmp_path, "base", _single(0, system="base"))
    cand = _save(tmp_path, "cand", _single(candidate_failing, system="cand"))
    argv = ["compare", base, cand, "--gate", *GATE, *extra]
    without = main(argv)
    junit = tmp_path / "gate.xml"
    with_junit = main([*argv, "--junit", str(junit)])
    assert without == with_junit == expected
    assert _parse(junit).tag == "testsuites"


def test_the_gate_json_gains_the_checks_and_nothing_else_changed(tmp_path: Path) -> None:
    _, _, payload = _compare(tmp_path, _single(0, system="base"), _single(10, system="cand"))
    assert set(payload["gate"]) == {
        "status",
        "exit_code",
        "reasons",
        "warnings",
        "adjusted_p",
        "power",
        "checks",
    }
    assert payload["gate"]["checks"] == [
        {"name": "max_drop: m", "outcome": "fail", "message": payload["gate"]["reasons"][0]}
    ]


# --- the decision's checks ---------------------------------------------------------------------


def _decision(baseline: Run, candidate: Run, **gate: object):  # type: ignore[no-untyped-def]
    cfg = GateConfig(**{"metric": "m", "min_cases": 10, **gate})  # type: ignore[arg-type]
    return decide(compare_runs(baseline, candidate), cfg, candidate)


@pytest.mark.parametrize(
    "gate",
    [
        {},
        {"min_pass_rate": 1.0},
        {"min_pass_rate": {"m": 0.5}},
        {"must_pass": ["critical"]},
        {"must_pass": ["nothing-carries-this"]},
        {"max_new_failures": 0},
        {"max_new_failures": 5},
        {"max_p95_latency_ms": 1.0},
        {"max_cost_usd": 0.0001},
        {
            "min_pass_rate": 0.99,
            "must_pass": ["critical"],
            "max_new_failures": 0,
            "max_p95_latency_ms": 1.0,
            "max_cost_usd": 0.0001,
        },
        {"min_cases": 50},
        {"allow_judge_change": True},
    ],
)
@pytest.mark.parametrize("failing", [0, 1, 12])
def test_the_failed_checks_are_exactly_the_reasons(gate: dict[str, object], failing: int) -> None:
    """The reasons and the checks come from one place, so they cannot drift apart."""
    baseline = _single(0, system="base")
    candidate = make_run(
        {cid: {"m": i >= failing} for i, cid in enumerate(IDS)},
        system="cand",
        tags={"c00": ["critical"]},
    )
    for result in candidate.results:
        result.output.cost_usd = 0.01
    decision = _decision(baseline, candidate, **gate)
    assert decision.checks, "a decision lists the checks it evaluated"
    bad = [c.message for c in decision.checks if c.outcome in ("fail", "error")]
    assert bad == decision.reasons
    assert bool(bad) == bool(decision.exit_code)
    assert len({c.name for c in decision.checks}) == len(decision.checks)  # names are unique


def test_a_refusal_is_an_error_check() -> None:
    decision = refusal("nope")
    assert decision.exit_code == EXIT_CANNOT_GATE and decision.reasons == ["nope"]
    assert decision.checks == [GateCheck("comparison", "error", "nope")]
    assert json.loads(to_json(decision))["checks"][0]["outcome"] == "error"


def test_gate_testsuite_is_built_from_the_decisions_checks_alone() -> None:
    baseline, candidate = _single(0, system="base"), _single(0, system="cand")
    decision = _decision(baseline, candidate)
    decision.checks.append(GateCheck("a_check_added_later", "fail", "it failed"))
    suite = _roundtrip(
        wrap([gate_testsuite(baseline, [("cand", candidate, decision)], GateConfig(metric="m"), 1)])
    ).find("testsuite")
    assert suite is not None
    assert {n: _outcome(c) for n, c in _cases(suite).items()} == {
        "max_drop: m": "pass",
        "a_check_added_later": "failure",
    }
