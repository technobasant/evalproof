"""A tag change is not an edit: ``compare`` pairs on a content hash that leaves tags out, falls
back to the case hash for 0.5.0 run files, and ``verify`` still treats a tag change as a change."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.dataset import case_hash, content_hash
from evalproof.gate import EXIT_PASS
from evalproof.models import Case
from evalproof.runner import load_run
from tests.conftest import make_run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "deterministic"


def _outcomes(n: int = 30) -> dict[str, dict[str, bool | float | None]]:
    return {f"c{i:02d}": {"m": True} for i in range(n)}


def _stamps(outcomes: dict[str, dict[str, bool | float | None]]) -> dict[str, str]:
    return {cid: f"h-{cid}" for cid in outcomes}


def test_the_content_hash_ignores_tags_but_not_content() -> None:
    base = Case(id="x", input="q", reference="r", tags=["t"], accept=["a"])
    retagged = Case(id="x", input="q", reference="r", tags=["t", "new"], accept=["a"])
    untagged = Case(id="x", input="q", reference="r", accept=["a"])
    assert content_hash(base) == content_hash(retagged) == content_hash(untagged)
    assert len(content_hash(base)) == 16
    assert case_hash(base) != case_hash(retagged)  # the case hash still sees the tags
    for changed in (
        Case(id="x", input="q2", reference="r", tags=["t"], accept=["a"]),
        Case(id="x", input="q", reference="r2", tags=["t"], accept=["a"]),
        Case(id="x", input="q", reference="r", tags=["t"], accept=["b"]),
        Case(id="y", input="q", reference="r", tags=["t"], accept=["a"]),
    ):
        assert content_hash(changed) != content_hash(base)


def test_a_tag_change_pairs_the_case() -> None:
    outcomes = _outcomes()
    baseline = make_run(
        outcomes, dataset_hash="old", case_hashes=True, content_hashes=True, tags={"c00": ["a"]}
    )
    # c00 gained a tag: its case hash moves, its content hash does not.
    candidate = make_run(
        outcomes,
        dataset_hash="new",
        case_hashes=_stamps(outcomes) | {"c00": "h-c00-tagged"},
        content_hashes=True,
        tags={"c00": ["a", "b"]},
    )
    cmp = compare_runs(baseline, candidate)
    assert cmp.comparable and cmp.edited == []
    assert "c00" in cmp.paired_ids and len(cmp.paired_ids) == 30
    assert cmp.metrics["m"].n_paired == 30


def test_a_changed_input_is_still_an_edit() -> None:
    outcomes = _outcomes()
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True, content_hashes=True)
    candidate = make_run(
        outcomes,
        dataset_hash="new",
        case_hashes=_stamps(outcomes) | {"c00": "h-c00-v2"},
        content_hashes={cid: f"c-{cid}" for cid in outcomes} | {"c00": "c-c00-v2"},
    )
    cmp = compare_runs(baseline, candidate)
    assert cmp.edited == ["c00"] and "c00" not in cmp.paired_ids
    assert cmp.metrics["m"].n_paired == 29


def test_a_0_5_0_baseline_still_pairs_on_the_case_hash() -> None:
    outcomes = _outcomes()
    # The baseline has case hashes only, as evalproof 0.5.0 wrote them; the candidate has both.
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    unchanged = make_run(outcomes, dataset_hash="new", case_hashes=True, content_hashes=True)
    cmp = compare_runs(baseline, unchanged)
    assert cmp.comparable and cmp.edited == [] and len(cmp.paired_ids) == 30

    # No content hash on one side means the case hash decides, tags included.
    moved = make_run(
        outcomes,
        dataset_hash="new",
        case_hashes=_stamps(outcomes) | {"c00": "h-c00-tagged"},
        content_hashes=True,
    )
    cmp = compare_runs(baseline, moved)
    assert cmp.comparable and cmp.edited == ["c00"]


def test_two_0_5_0_runs_compare_as_before() -> None:
    outcomes = _outcomes()
    baseline = make_run(outcomes, dataset_hash="old", case_hashes=True)
    candidate = make_run(
        outcomes, dataset_hash="new", case_hashes=_stamps(outcomes) | {"c01": "h-c01-v2"}
    )
    assert compare_runs(baseline, candidate).edited == ["c01"]


@pytest.fixture
def example(tmp_path: Path) -> Path:
    dst = tmp_path / "deterministic"
    shutil.copytree(
        EXAMPLE, dst, ignore=shutil.ignore_patterns("runs", ".evalproof", "__pycache__")
    )
    return dst


def test_adding_a_tag_end_to_end(example: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = str(example / "suite.yaml")
    before = example / "before.json"
    after = example / "after.json"
    assert main(["run", suite, "--out", str(before), "--quiet"]) == 0

    cases = example / "cases.jsonl"
    rows = [json.loads(line) for line in cases.read_text("utf-8").splitlines()]
    assert "reviewed" not in rows[0]["tags"]
    rows[0]["tags"].append("reviewed")
    cases.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    assert main(["run", suite, "--out", str(after), "--quiet"]) == 0

    old, new = load_run(before), load_run(after)
    assert old.info.dataset_hash != new.info.dataset_hash  # the dataset hash sees the tag
    cmp = compare_runs(old, new)
    assert cmp.comparable and cmp.edited == []
    assert len(cmp.paired_ids) == len(rows) and cmp.only_baseline == cmp.only_candidate == []
    capsys.readouterr()
    assert main(["compare", str(before), str(after), "--suite", suite, "--gate"]) == EXIT_PASS
    assert "edited" not in capsys.readouterr().out

    # verify checks the tags too: must_pass reads them from the run file, so a run that carries
    # the old tags no longer matches the dataset.
    assert main(["verify", str(before), suite]) == 4
    out = capsys.readouterr().out
    assert "(content or tags)" in out and f"`{rows[0]['id']}`" in out
    assert main(["verify", str(after), suite]) == 0
