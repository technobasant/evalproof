"""``evalproof rehash``: per-case hashes for a pre-0.5 run file, from the dataset it scored."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig, load_config
from evalproof.dataset import case_hash, content_hash, load_dataset
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, decide
from evalproof.rehash import RehashError, rehash_run
from evalproof.runner import load_run, run_suite, save_run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "deterministic"


@pytest.fixture
def example(tmp_path: Path) -> Path:
    dst = tmp_path / "deterministic"
    shutil.copytree(
        EXAMPLE, dst, ignore=shutil.ignore_patterns("runs", ".evalproof", "__pycache__")
    )
    return dst


def _old_run(example: Path) -> Path:
    """A run as evalproof 0.4 wrote it: no case hashes and no suite hash."""
    run = run_suite(load_config(example / "suite.yaml"), use_cache=False)
    for r in run.results:
        r.case_hash = r.content_hash = None
    run.info.suite_hash = None
    return save_run(run, example / "old.json")


def _add_case(example: Path) -> None:
    row = {"id": "t99", "input": "My card was declined.", "reference": "billing", "tags": []}
    with (example / "cases.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _insert_case(example: Path) -> None:
    """A case added before the others, so the run's cases are no longer the first ones."""
    cases = example / "cases.jsonl"
    rows = [json.loads(line) for line in cases.read_text("utf-8").splitlines()]
    rows.insert(0, {"id": "t00", "input": "Where is my refund?", "reference": "billing"})
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")


def test_rehash_restores_the_hashes_the_runner_writes(
    example: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = example / "suite.yaml"
    fresh = run_suite(load_config(suite), use_cache=False)
    old = _old_run(example)
    assert all(r.case_hash is None for r in load_run(old).results)

    assert main(["rehash", str(old), str(suite)]) == EXIT_PASS
    assert f"rehashed {len(fresh.results)} cases in {old}" in capsys.readouterr().err
    rehashed = load_run(old)
    assert [(r.case_id, r.case_hash, r.content_hash) for r in rehashed.results] == [
        (r.case_id, r.case_hash, r.content_hash) for r in fresh.results
    ]
    assert all(r.case_hash and r.content_hash for r in rehashed.results)


def test_rehash_changes_nothing_but_the_hashes(example: Path) -> None:
    suite = load_config(example / "suite.yaml")
    old = load_run(_old_run(example))
    out = rehash_run(old, suite)
    assert out is not old and all(r.case_hash is None for r in old.results)  # a copy
    stripped = out.model_copy(deep=True)
    for r in stripped.results:
        r.case_hash = r.content_hash = None
    assert stripped == old  # scores, outputs and info, suite hash included, are as they were
    cases = {c.id: c for c in load_dataset(example / "cases.jsonl")}
    for r in out.results:
        assert r.case_hash == case_hash(cases[r.case_id])
        assert r.content_hash == content_hash(cases[r.case_id])


def test_rehash_is_idempotent(example: Path) -> None:
    suite = str(example / "suite.yaml")
    old = _old_run(example)
    assert main(["rehash", str(old), suite]) == EXIT_PASS
    once = old.read_text("utf-8")
    assert main(["rehash", str(old), suite]) == EXIT_PASS
    assert old.read_text("utf-8") == once
    # A file the runner wrote, with every hash already in place, is fine too.
    fresh = save_run(run_suite(load_config(suite), use_cache=False), example / "fresh.json")
    before = fresh.read_text("utf-8")
    assert main(["rehash", str(fresh), suite]) == EXIT_PASS
    assert fresh.read_text("utf-8") == before


def test_a_changed_dataset_is_refused_and_the_file_is_untouched(
    example: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _old_run(example)
    before = old.read_text("utf-8")
    _insert_case(example)
    assert main(["rehash", str(old), str(example / "suite.yaml")]) == EXIT_CANNOT_GATE
    err = capsys.readouterr().err
    assert "dataset has changed since the run was made" in err
    assert "base branch" in err and "re-run the suite" in err
    assert old.read_text("utf-8") == before


def test_an_edited_case_is_refused_too(example: Path) -> None:
    old = _old_run(example)
    cases = example / "cases.jsonl"
    rows = [json.loads(line) for line in cases.read_text("utf-8").splitlines()]
    rows[0]["tags"].append("new-tag")  # the dataset hash sees tags, so even this is refused
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    assert main(["rehash", str(old), str(example / "suite.yaml")]) == EXIT_CANNOT_GATE


def test_cases_appended_after_the_run_do_not_stop_a_rehash(example: Path) -> None:
    """The run's cases are still the first ones in the file, unchanged, and the dataset hash
    proves it, so they can be hashed; the appended case is simply not in the run."""
    suite = example / "suite.yaml"
    old = _old_run(example)
    n = len(load_run(old).results)
    _add_case(example)
    assert main(["rehash", str(old), str(suite)]) == EXIT_PASS
    cases = {c.id: c for c in load_dataset(example / "cases.jsonl")}
    rehashed = load_run(old)
    assert len(rehashed.results) == n and "t99" not in rehashed.by_id()
    assert all(r.case_hash == case_hash(cases[r.case_id]) for r in rehashed.results)


def test_a_limit_run_rehashes_its_cases(example: Path) -> None:
    cfg = load_config(example / "suite.yaml")
    run = run_suite(cfg, limit=5, use_cache=False)
    want = [(r.case_id, r.case_hash, r.content_hash) for r in run.results]
    for r in run.results:
        r.case_hash = r.content_hash = None
    out = rehash_run(run, cfg)
    assert [(r.case_id, r.case_hash, r.content_hash) for r in out.results] == want


def test_a_result_for_a_case_the_dataset_lacks_is_refused(example: Path) -> None:
    cfg = load_config(example / "suite.yaml")
    run = load_run(_old_run(example))
    run.results[0].case_id = "not-in-the-dataset"
    with pytest.raises(RehashError, match="not-in-the-dataset"):
        rehash_run(run, cfg)


def test_out_writes_elsewhere_and_leaves_the_source_alone(
    example: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    old = _old_run(example)
    before = old.read_text("utf-8")
    dest = tmp_path / "nested" / "rehashed.json"
    assert main(["rehash", str(old), str(example / "suite.yaml"), "--out", str(dest)]) == 0
    assert f"in {dest}" in capsys.readouterr().err
    assert old.read_text("utf-8") == before
    assert all(r.case_hash for r in load_run(dest).results)


def test_a_refused_rehash_does_not_create_the_out_file(example: Path, tmp_path: Path) -> None:
    old = _old_run(example)
    _insert_case(example)
    dest = tmp_path / "nested" / "rehashed.json"
    assert main(["rehash", str(old), str(example / "suite.yaml"), "--out", str(dest)]) == 4
    assert not dest.exists() and not dest.parent.exists()


def test_a_missing_file_is_a_usage_error(example: Path, tmp_path: Path) -> None:
    suite = str(example / "suite.yaml")
    assert main(["rehash", str(tmp_path / "nope.json"), suite]) == 2
    assert main(["rehash", str(_old_run(example)), str(tmp_path / "nope.yaml")]) == 2


def test_a_rehashed_baseline_gates_against_a_candidate_with_an_added_case(example: Path) -> None:
    suite = example / "suite.yaml"
    old = _old_run(example)
    gate = GateConfig(metric="queue", min_cases=30)

    # Before: the dataset changes and the baseline has no case hashes, so it cannot be gated.
    _add_case(example)
    candidate = run_suite(load_config(suite), use_cache=False)
    refused = decide(compare_runs(load_run(old), candidate), gate)
    assert refused.exit_code == EXIT_CANNOT_GATE and "evalproof rehash" in refused.reasons[0]

    # The added case is appended, so the run's cases are provably the first ones: rehash, gate.
    assert main(["rehash", str(old), str(suite)]) == EXIT_PASS

    cmp = compare_runs(load_run(old), candidate)
    assert cmp.comparable and cmp.only_candidate == ["t99"] and cmp.edited == []
    assert decide(cmp, gate).exit_code == EXIT_PASS


def test_verify_still_refuses_a_rehashed_file_without_a_suite_hash(
    example: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = str(example / "suite.yaml")
    old = _old_run(example)
    assert main(["rehash", str(old), suite]) == EXIT_PASS
    assert load_run(old).info.suite_hash is None
    capsys.readouterr()
    assert main(["verify", str(old), suite]) == 4
    assert "no suite hash" in capsys.readouterr().out
