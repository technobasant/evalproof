"""``evalproof compare`` with several candidates, ``--aggregate``, and the exit-code order. One
candidate must behave exactly as it always did."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

import evalproof.cli as cli
from evalproof.cli import _worst, main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    decide,
)
from evalproof.models import Run
from evalproof.report import comparison_markdown, to_json
from evalproof.runner import save_run
from tests.conftest import make_run

IDS = [f"c{i:02d}" for i in range(30)]


def _single(failing: int, *, system: str, latency: float = 10.0) -> Run:
    """One trial per case; the first ``failing`` cases fail ``m``."""
    return make_run(
        {cid: {"m": i >= failing} for i, cid in enumerate(IDS)}, system=system, latency=latency
    )


def _repeated(
    rows: dict[str, list[bool]], *, system: str, errors: Iterable[tuple[str, int]] = ()
) -> Run:
    failed = set(errors)
    k = len(next(iter(rows.values())))
    results = []
    for cid, values in rows.items():
        for trial, value in enumerate(values):
            errs = {cid} if (cid, trial) in failed else None
            one = make_run({cid: {"m": value}}, errors=errs).results[0]
            results.append(one.model_copy(update={"trial": trial}))
    info = make_run({"x": {}}, system=system).info.model_copy(update={"trials": k})
    return Run(info=info, results=results)


def _save(tmp_path: Path, name: str, run: Run) -> str:
    return str(save_run(run, tmp_path / f"{name}.json"))


GATE = ["--metric", "m", "--min-cases", "10"]


# --- one candidate is exactly as before --------------------------------------------------------


def test_one_candidate_writes_the_same_report_and_json_as_before(tmp_path: Path, capsys) -> None:
    base_run, cand_run = _single(0, system="base"), _single(7, system="cand")
    base, cand = _save(tmp_path, "b", base_run), _save(tmp_path, "c", cand_run)
    md, js = tmp_path / "o.md", tmp_path / "o.json"
    code = main(["compare", base, cand, *GATE, "--gate", "--markdown", str(md), "--json", str(js)])

    gate = GateConfig(metric="m", min_cases=10)
    cmp = compare_runs(base_run, cand_run)
    decision = decide(cmp, gate, cand_run)
    expected_md = comparison_markdown(cmp, decision, gate.gated, runs=(base_run, cand_run), show=5)
    assert code == decision.exit_code == EXIT_REGRESSION
    assert capsys.readouterr().out == expected_md
    assert md.read_text(encoding="utf-8") == expected_md + "\n"
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert list(payload) == ["comparison", "gate"]  # no "candidates" wrapper for one candidate
    assert payload["comparison"] == json.loads(to_json(cmp))
    assert payload["gate"] == json.loads(to_json(decision))
    assert "Holm" not in expected_md and "against" not in expected_md


def test_one_candidate_without_gate_exits_zero_and_writes_no_gate(tmp_path: Path) -> None:
    base, cand = (
        _save(tmp_path, "b", _single(0, system="b")),
        _save(tmp_path, "c", _single(7, system="c")),
    )
    js = tmp_path / "o.json"
    assert main(["compare", base, cand, "--json", str(js), "--seed", "3"]) == EXIT_PASS
    assert list(json.loads(js.read_text(encoding="utf-8"))) == ["comparison"]


# --- the matrix --------------------------------------------------------------------------------


def _matrix(tmp_path: Path) -> tuple[str, list[str]]:
    base = _save(tmp_path, "b", _single(0, system="base"))
    cands = [
        _save(tmp_path, "c1", _single(6, system="c1")),  # raw p 0.031: significant alone
        _save(tmp_path, "c2", _single(0, system="c2")),  # identical to the baseline
        _save(tmp_path, "c3", _single(3, system="c3")),  # raw p 0.25
    ]
    return base, cands


def test_holm_runs_across_every_candidate_so_a_lone_significant_drop_stops_failing(
    tmp_path: Path, capsys
) -> None:
    base, (c1, c2, c3) = _matrix(tmp_path)
    # Alone, c1's drop of 20 points at p = 0.031 fails the gate.
    assert main(["compare", base, c1, *GATE, "--gate", "--show", "0"]) == EXIT_REGRESSION
    capsys.readouterr()
    # Among three candidates the family is three tests, and 0.031 * 3 = 0.094 no longer is.
    js = tmp_path / "o.json"
    code = main(["compare", base, c1, c2, c3, *GATE, "--gate", "--show", "0", "--json", str(js)])
    out = capsys.readouterr().out
    assert code == EXIT_PASS
    assert "p-values are Holm-adjusted across all 3 candidates and 1 gated metric(s)" in out
    payload = json.loads(js.read_text(encoding="utf-8"))
    adjusted = [c["gate"]["adjusted_p"]["m"] for c in payload["candidates"]]
    assert adjusted == pytest.approx([0.03125 * 3, 1.0, 0.25 * 2])


def test_holm_family_is_candidates_times_gated_metrics(tmp_path: Path) -> None:
    ids = IDS
    base = make_run({cid: {"m": True, "n": True} for cid in ids}, system="base")

    def cand(system: str, fail_m: int, fail_n: int) -> Run:
        rows = {cid: {"m": i >= fail_m, "n": i >= fail_n} for i, cid in enumerate(ids)}
        return make_run(rows, system=system)

    paths = [
        _save(tmp_path, "b", base),
        _save(tmp_path, "c1", cand("c1", 6, 0)),
        _save(tmp_path, "c2", cand("c2", 0, 0)),
    ]
    js = tmp_path / "o.json"
    main(["compare", *paths, "--metric", "m", "--metric", "n", "--min-cases", "10", "--gate",
          "--show", "0", "--json", str(js)])  # fmt: skip
    payload = json.loads(js.read_text(encoding="utf-8"))
    first, second = (c["gate"]["adjusted_p"] for c in payload["candidates"])
    # Four tests: c1/m (p 0.03125), the other three are 1.0. Holm multiplies the smallest by 4.
    assert first["m"] == pytest.approx(0.03125 * 4)
    assert first["n"] == second["m"] == second["n"] == 1.0


def test_matrix_json_has_a_baseline_and_a_decision_per_candidate(tmp_path: Path) -> None:
    base, cands = _matrix(tmp_path)
    js = tmp_path / "o.json"
    main(["compare", base, *cands, *GATE, "--gate", "--show", "0", "--json", str(js)])
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert list(payload) == ["baseline", "candidates"] and payload["baseline"] == "base"
    assert [c["label"] for c in payload["candidates"]] == ["c1", "c2", "c3"]
    for entry, path in zip(payload["candidates"], cands, strict=True):
        assert set(entry) >= {"label", "comparison", "gate"}
        assert entry["path"] == path
        assert entry["comparison"]["metrics"]["m"]["n_paired"] == 30
        assert set(entry["gate"]["power"]) == {"m"}
    # Power divides alpha among the candidates too.
    assert payload["candidates"][0]["gate"]["power"]["m"]["alpha_used"] == pytest.approx(0.05 / 3)


def test_matrix_markdown_has_a_row_per_candidate_and_its_own_details(
    tmp_path: Path, capsys
) -> None:
    base, cands = _matrix(tmp_path)
    main(["compare", base, *cands, *GATE, "--gate", "--show", "0"])
    out = capsys.readouterr().out
    assert "| Candidate | m: Δ (95% CI) · adj p · verdict | Gate |" in out
    for label in ("c1", "c2", "c3"):
        assert f"| `{label}` |" in out
        assert f"<details><summary><code>{label}</code>" in out
    assert "⚠️ drop, not significant" in out and "✅ no regression" in out
    assert out.count("Baseline `base` vs candidate") == 3


def test_a_matrix_without_gate_shows_raw_p_values_and_exits_zero(tmp_path: Path, capsys) -> None:
    base, cands = _matrix(tmp_path)
    js = tmp_path / "o.json"
    assert main(["compare", base, *cands, "--json", str(js)]) == EXIT_PASS
    out = capsys.readouterr().out
    assert "Holm" not in out and "m: Δ (95% CI) · p |" in out
    payload = json.loads(js.read_text(encoding="utf-8"))
    assert all(c["gate"] is None for c in payload["candidates"])


def test_candidates_with_the_same_label_are_told_apart_by_file(tmp_path: Path, capsys) -> None:
    base = _save(tmp_path, "b", _single(0, system="base"))
    one = _save(tmp_path, "one", _single(2, system="same"))
    two = _save(tmp_path, "two", _single(4, system="same"))
    main(["compare", base, one, two, *GATE, "--gate", "--show", "0"])
    out = capsys.readouterr().out
    assert "`same [one.json]`" in out and "`same [two.json]`" in out


# --- exit codes --------------------------------------------------------------------------------


def test_the_worst_exit_is_usage_then_cannot_gate_then_regression_then_budget_then_pass() -> None:
    assert (EXIT_USAGE, EXIT_CANNOT_GATE, EXIT_REGRESSION, EXIT_BUDGET, EXIT_PASS) == (
        2,
        4,
        1,
        3,
        0,
    )
    assert _worst([EXIT_PASS, EXIT_PASS]) == EXIT_PASS
    assert _worst([EXIT_PASS, EXIT_BUDGET]) == EXIT_BUDGET
    assert _worst([EXIT_BUDGET, EXIT_REGRESSION]) == EXIT_REGRESSION
    assert _worst([EXIT_REGRESSION, EXIT_CANNOT_GATE, EXIT_PASS]) == EXIT_CANNOT_GATE
    assert _worst([EXIT_CANNOT_GATE, EXIT_USAGE, EXIT_REGRESSION]) == EXIT_USAGE
    assert _worst([]) == EXIT_PASS


def test_a_matrix_exits_with_the_worst_candidate(tmp_path: Path, capsys) -> None:
    base = _save(tmp_path, "b", _single(0, system="base"))
    slow = _save(tmp_path, "slow", _single(0, system="slow", latency=900.0))
    worse = _save(tmp_path, "worse", _single(15, system="worse"))
    same = _save(tmp_path, "same", _single(0, system="same"))
    argv = ["--gate", "--show", "0", *GATE, "--max-p95-ms", "100"]
    assert main(["compare", base, same, *argv]) == EXIT_PASS
    assert main(["compare", base, slow, *argv]) == EXIT_BUDGET
    assert main(["compare", base, same, slow, *argv]) == EXIT_BUDGET
    assert main(["compare", base, slow, worse, *argv]) == EXIT_REGRESSION
    assert main(["compare", base, worse, slow, same, *argv]) == EXIT_REGRESSION


def test_a_candidate_that_cannot_be_gated_outranks_a_regression(tmp_path: Path, capsys) -> None:
    base = _save(tmp_path, "b", _single(0, system="base"))
    worse = _save(tmp_path, "worse", _single(15, system="worse"))
    few = {cid: {"m": True} for cid in IDS[:5]}  # too few cases to gate
    small = _save(tmp_path, "small", make_run(few, system="small"))
    assert main(["compare", base, worse, small, *GATE, "--gate", "--show", "0"]) == EXIT_CANNOT_GATE
    assert "only 5 paired cases" in capsys.readouterr().out


# --- repeated trials through the CLI -----------------------------------------------------------


def _k_rows(failing_trials: int, cases: int = 30) -> dict[str, list[bool]]:
    """Every case passes trials 0 and 1 and fails trial 2, for the first ``failing_trials``."""
    return {cid: [True, True, i >= failing_trials] for i, cid in enumerate(IDS[:cases])}


def test_aggregate_flag_changes_how_trials_become_a_case_value(tmp_path: Path, capsys) -> None:
    base = _save(tmp_path, "b", _repeated({cid: [True] * 3 for cid in IDS}, system="base"))
    cand = _save(tmp_path, "c", _repeated(_k_rows(15), system="cand"))
    argv = ["compare", base, cand, *GATE, "--gate", "--show", "0", "--max-drop", "0.3"]
    assert main(argv) == EXIT_PASS  # mean: 15 cases lost a third of their trials, a 17-point drop
    out = capsys.readouterr().out
    assert "(trials: 3, aggregate: mean)" in out and "sign-flip permutation" in out
    assert main([*argv, "--aggregate", "all"]) == EXIT_REGRESSION  # all: half the cases failed
    out = capsys.readouterr().out
    assert "(trials: 3, aggregate: all)" in out and "exact McNemar" in out
    assert main([*argv, "--aggregate", "any"]) == EXIT_PASS  # any: every case still passed once


def test_gate_aggregate_in_the_suite_applies_and_the_flag_overrides_it(
    tmp_path: Path, capsys
) -> None:
    (tmp_path / "cases.jsonl").write_text('{"id": "a", "input": "q"}\n', encoding="utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:nothing:ask'}\n"
        "scorers: [{name: m, type: contains, params: {any_of: ['x']}}]\n"
        "gate: {metric: m, min_cases: 10, max_drop: 0.3, aggregate: all}\n",
        encoding="utf-8",
    )
    base = _save(tmp_path, "b", _repeated({cid: [True] * 3 for cid in IDS}, system="base"))
    cand = _save(tmp_path, "c", _repeated(_k_rows(15), system="cand"))
    argv = ["compare", base, cand, "--suite", str(suite), "--gate", "--show", "0"]
    assert main(argv) == EXIT_REGRESSION
    assert "aggregate: all" in capsys.readouterr().out
    assert main([*argv, "--aggregate", "mean"]) == EXIT_PASS
    assert "aggregate: mean" in capsys.readouterr().out


def test_a_bad_aggregate_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["compare", "a.json", "b.json", "--aggregate", "median"])
    assert exc.value.code == 2


def test_runs_with_different_trials_cannot_be_gated_and_are_a_usage_error_otherwise(
    tmp_path: Path, capsys
) -> None:
    three = _save(tmp_path, "three", _repeated({cid: [True] * 3 for cid in IDS}, system="a"))
    five = _save(tmp_path, "five", _repeated({cid: [True] * 5 for cid in IDS}, system="b"))
    js = tmp_path / "o.json"
    code = main(["compare", three, five, *GATE, "--gate", "--json", str(js)])
    assert code == EXIT_CANNOT_GATE
    text = capsys.readouterr().out
    assert "cannot gate" in text and "baseline has k=3, candidate has k=5" in text
    gate = json.loads(js.read_text(encoding="utf-8"))["gate"]
    assert gate["exit_code"] == EXIT_CANNOT_GATE and "k=3" in gate["reasons"][0]
    assert main(["compare", three, five]) == EXIT_USAGE
    assert "same --trials" in capsys.readouterr().err


def test_a_matrix_candidate_with_other_trials_is_refused_and_the_rest_still_compare(
    tmp_path: Path, capsys
) -> None:
    three = _save(tmp_path, "three", _repeated({cid: [True] * 3 for cid in IDS}, system="base"))
    same = _save(tmp_path, "same", _repeated({cid: [True] * 3 for cid in IDS}, system="same"))
    five = _save(tmp_path, "five", _repeated({cid: [True] * 5 for cid in IDS}, system="five"))
    code = main(["compare", three, same, five, *GATE, "--gate", "--show", "0"])
    out = capsys.readouterr().out
    assert code == EXIT_CANNOT_GATE
    assert "five: baseline has k=3, candidate has k=5" in out
    assert "| `same` |" in out and "✅ pass" in out
    assert main(["compare", three, same, five]) == EXIT_USAGE  # without --gate: a usage error
    assert "five.json" in capsys.readouterr().err


# --- nothing reaches a model -------------------------------------------------------------------


def test_comparing_builds_no_provider_and_runs_no_suite(tmp_path: Path, monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("compare must not call a model or run a suite")

    monkeypatch.setattr(cli, "make_provider", boom)
    monkeypatch.setattr(cli, "run_suite", boom)
    base = _save(tmp_path, "b", _repeated({cid: [True] * 3 for cid in IDS}, system="base"))
    one = _save(tmp_path, "c1", _repeated(_k_rows(10), system="c1"))
    two = _save(tmp_path, "c2", _repeated(_k_rows(20), system="c2"))
    assert main(["compare", base, one, two, *GATE, "--gate", "--show", "0"]) in (
        EXIT_PASS,
        EXIT_REGRESSION,
    )
