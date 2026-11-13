"""Comparing runs across dataset changes: per-case hashes decide which cases are paired."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig, SuiteConfig, load_config
from evalproof.dataset import case_hash, load_dataset
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, decide
from evalproof.importers import import_run
from evalproof.models import Case
from evalproof.runner import load_run, run_suite, save_run
from tests.conftest import make_run

GATE = GateConfig(metric="m", min_cases=10)


def _runs(n: int = 30):
    ids = [f"c{i:02d}" for i in range(n)]
    outcomes = {cid: {"m": True} for cid in ids}
    return ids, outcomes


def test_a_case_hash_is_stable_and_sensitive_to_content() -> None:
    a = Case(id="x", input="q", reference="r", tags=["t"], accept=["a"])
    same = Case(id="x", input="q", reference="r", tags=["t"], accept=["a"])
    other = Case(id="x", input="q", reference="r2", tags=["t"], accept=["a"])
    assert case_hash(a) == case_hash(same) and len(case_hash(a)) == 16
    assert case_hash(a) != case_hash(other)


def test_adding_a_case_gates_on_the_shared_cases_with_a_warning() -> None:
    _ids, outcomes = _runs()
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    candidate = make_run(outcomes | {"new": {"m": False}}, dataset_hash="new", case_hashes=True)
    cmp = compare_runs(baseline, candidate)
    assert cmp.comparable and not cmp.same_dataset
    assert cmp.only_candidate == ["new"] and cmp.edited == [] and len(cmp.paired_ids) == 30
    assert cmp.metrics["m"].n_paired == 30
    decision = decide(cmp, GATE)
    assert decision.exit_code == EXIT_PASS and decision.status == "warn"
    assert decision.warnings == [
        "1 case(s) added, 0 removed, 0 edited: excluded from the comparison; "
        "gated on the 30 cases both runs share"
    ]


def test_a_removed_case_is_counted_too() -> None:
    _, outcomes = _runs()
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    candidate = make_run(
        {k: v for k, v in outcomes.items() if k != "c00"}, dataset_hash="new", case_hashes=True
    )
    decision = decide(compare_runs(baseline, candidate), GATE)
    assert decision.exit_code == EXIT_PASS
    assert decision.warnings[0].startswith("0 case(s) added, 1 removed, 0 edited")


def test_an_edited_case_is_excluded_from_every_metric() -> None:
    ids, outcomes = _runs()
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    # c00 was edited and now fails; it must not count as a regression.
    candidate = make_run(
        outcomes | {"c00": {"m": False}},
        dataset_hash="new",
        case_hashes={cid: f"h-{cid}" for cid in ids} | {"c00": "h-c00-v2"},
    )
    cmp = compare_runs(baseline, candidate)
    assert cmp.edited == ["c00"] and "c00" not in cmp.paired_ids
    mc = cmp.metrics["m"]
    assert mc.n_paired == 29 and mc.regressed == [] and mc.candidate == 1.0
    decision = decide(cmp, GATE)
    assert decision.exit_code == EXIT_PASS
    assert "1 edited" in decision.warnings[0] and "29 cases" in decision.warnings[0]


def test_an_old_baseline_and_a_changed_dataset_cannot_be_gated() -> None:
    _, outcomes = _runs()
    old = make_run(outcomes, dataset_hash="old")  # no case hashes: written before 0.5
    new = make_run(outcomes | {"new": {"m": True}}, dataset_hash="new", case_hashes=True)
    for pair in ((old, new), (new, old), (old, make_run(outcomes, dataset_hash="new"))):
        cmp = compare_runs(*pair)
        assert not cmp.comparable
        decision = decide(cmp, GATE)
        assert decision.exit_code == EXIT_CANNOT_GATE
        assert "predates per-case hashes" in decision.reasons[0]
        assert "re-run the baseline with evalproof 0.5 or newer" in decision.reasons[0]


def test_two_old_runs_with_the_same_dataset_hash_behave_as_before() -> None:
    _, outcomes = _runs()
    old_a = make_run(outcomes, dataset_hash="same")
    old_b = make_run(outcomes | {"c01": {"m": False}}, dataset_hash="same")
    cmp = compare_runs(old_a, old_b)
    assert cmp.comparable and cmp.same_dataset and cmp.edited == []
    assert cmp.metrics["m"].n_paired == 30 and cmp.metrics["m"].regressed == ["c01"]
    assert decide(cmp, GATE).exit_code == EXIT_PASS
    # An old baseline against a new candidate on the same dataset also still works.
    mixed = compare_runs(old_a, make_run(outcomes, dataset_hash="same", case_hashes=True))
    assert mixed.comparable and len(mixed.paired_ids) == 30


def test_min_cases_counts_the_paired_cases() -> None:
    _, outcomes = _runs(12)
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    extra = {f"n{i}": {"m": True} for i in range(10)}
    candidate = make_run(outcomes | extra, dataset_hash="new", case_hashes=True)
    ok = decide(compare_runs(baseline, candidate), GateConfig(metric="m", min_cases=12))
    assert ok.exit_code == EXIT_PASS
    short = decide(compare_runs(baseline, candidate), GateConfig(metric="m", min_cases=13))
    assert short.exit_code == EXIT_CANNOT_GATE and "only 12 paired cases" in short.reasons[0]


# -- the runner and the importers write the hashes ------------------------------------------


def _suite(tmp_path: Path, cases: list[dict[str, object]]) -> SuiteConfig:
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    (tmp_path / "answer.py").write_text("def ask(x):\n    return x['q'].upper()\n", "utf-8")
    (tmp_path / "suite.yaml").write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:answer:ask'}\n"
        "scorers:\n  - {name: m, type: contains, params: {any_of_field: accept}}\n"
        "gate: {metric: m, min_cases: 10}\n",
        "utf-8",
    )
    return load_config(tmp_path / "suite.yaml")


def _cases(n: int = 30) -> list[dict[str, object]]:
    return [{"id": f"c{i:02d}", "input": {"q": f"q{i}"}, "accept": [f"Q{i}"]} for i in range(n)]


def test_the_runner_stamps_every_result(tmp_path: Path) -> None:
    cfg = _suite(tmp_path, _cases(5))
    run = run_suite(cfg, use_cache=False)
    by_id = {c.id: c for c in load_dataset(tmp_path / "cases.jsonl")}
    assert all(r.case_hash == case_hash(by_id[r.case_id]) for r in run.results)
    assert load_run(save_run(run, tmp_path / "r.json")).results[0].case_hash is not None


def test_a_limit_run_pairs_with_a_full_baseline_on_the_shared_cases(tmp_path: Path) -> None:
    cfg = _suite(tmp_path, _cases(30))
    full = run_suite(cfg, use_cache=False)
    partial = run_suite(cfg, limit=12, use_cache=False)
    assert full.info.dataset_hash != partial.info.dataset_hash
    cmp = compare_runs(full, partial)
    assert cmp.comparable and len(cmp.paired_ids) == 12 and cmp.only_baseline
    decision = decide(cmp, GateConfig(metric="m", min_cases=10))
    assert decision.exit_code == EXIT_PASS
    assert "0 case(s) added, 18 removed, 0 edited" in decision.warnings[0]
    # Through the CLI as well, since that is how CI runs it.
    save_run(full, tmp_path / "full.json")
    save_run(partial, tmp_path / "partial.json")
    argv = ["compare", str(tmp_path / "full.json"), str(tmp_path / "partial.json")]
    assert main([*argv, "--suite", str(tmp_path / "suite.yaml"), "--gate"]) == EXIT_PASS


def test_editing_a_case_file_shows_up_as_edited_end_to_end(tmp_path: Path) -> None:
    cases = _cases(30)
    cfg = _suite(tmp_path, cases)
    before = run_suite(cfg, use_cache=False)
    cases[3]["reference"] = "now with a reference"
    cases.append({"id": "extra", "input": {"q": "z"}, "accept": ["Z"]})
    cfg = _suite(tmp_path, cases)
    after = run_suite(cfg, use_cache=False)
    cmp = compare_runs(before, after)
    assert cmp.edited == ["c03"] and cmp.only_candidate == ["extra"] and cmp.comparable


def test_imported_runs_carry_case_hashes(tmp_path: Path) -> None:
    rows = [
        {"id": f"c{i}", "metric": "m", "passed": True, "input": f"question {i}"} for i in range(3)
    ]
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    run = import_run("jsonl", path)
    hashes = [r.case_hash for r in run.results]
    assert all(h is not None and len(h) == 16 for h in hashes) and len(set(hashes)) == 3
    again = import_run("jsonl", path)
    assert [r.case_hash for r in again.results] == hashes


@pytest.mark.parametrize("fmt", ["jsonl"])
def test_an_imported_case_whose_input_changed_is_edited(tmp_path: Path, fmt: str) -> None:
    def write(name: str, first_input: str) -> Path:
        rows = [{"id": f"c{i}", "metric": "m", "passed": True, "input": f"q{i}"} for i in range(12)]
        rows[0]["input"] = first_input
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
        return path

    a = import_run(fmt, write("a.jsonl", "q0"))
    b = import_run(fmt, write("b.jsonl", "q0 changed"))
    cmp = compare_runs(a, b)
    assert cmp.edited == ["c0"] and cmp.comparable and len(cmp.paired_ids) == 11
