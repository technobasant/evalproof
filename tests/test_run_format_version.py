"""``Run.format_version``: written into every run file, defaulted for a legacy file, and a file
from a newer evalproof is refused instead of half-read."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.config import load_config
from evalproof.gate import EXIT_PASS, EXIT_USAGE
from evalproof.models import RUN_FORMAT_VERSION, Run
from evalproof.rehash import rehash_run
from evalproof.runner import RunFileError, load_run, run_suite, save_run
from tests.conftest import make_run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "deterministic"


def _write(path: Path, **changes: object) -> Path:
    """A run file with ``changes`` applied to its top level; a value of ... drops the key."""
    data = json.loads(make_run({"a": {"m": True}}).model_dump_json())
    for key, value in changes.items():
        if value is ...:
            data.pop(key, None)
        else:
            data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_the_current_format_version_is_1() -> None:
    assert RUN_FORMAT_VERSION == 1
    assert Run(info=make_run({}).info, results=[]).format_version == RUN_FORMAT_VERSION


def test_a_written_run_file_carries_the_format_version(tmp_path: Path) -> None:
    path = save_run(make_run({"a": {"m": True}}), tmp_path / "run.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format_version"] == 1
    assert next(iter(data)) == "format_version"  # first, so a reader sees it before the rest
    assert load_run(path).format_version == 1


def test_a_run_from_run_suite_carries_it(tmp_path: Path) -> None:
    example = tmp_path / "deterministic"
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("runs", ".evalproof"))
    run = run_suite(load_config(example / "suite.yaml"), use_cache=False)
    assert run.format_version == 1
    saved = save_run(run, tmp_path / "run.json")
    assert json.loads(saved.read_text(encoding="utf-8"))["format_version"] == 1


def test_a_legacy_file_without_it_loads_as_format_1(tmp_path: Path) -> None:
    path = _write(tmp_path / "old.json", format_version=...)
    assert "format_version" not in json.loads(path.read_text(encoding="utf-8"))
    run = load_run(path)
    assert run.format_version == 1
    assert [r.case_id for r in run.results] == ["a"]


def test_rehash_writes_the_current_format_version(tmp_path: Path) -> None:
    example = tmp_path / "deterministic"
    shutil.copytree(EXAMPLE, example, ignore=shutil.ignore_patterns("runs", ".evalproof"))
    cfg = load_config(example / "suite.yaml")
    run = run_suite(cfg, use_cache=False)
    old = save_run(run, example / "old.json")
    data = json.loads(old.read_text(encoding="utf-8"))
    del data["format_version"]
    old.write_text(json.dumps(data), encoding="utf-8")  # a file from before 1.0
    assert main(["rehash", str(old), str(example / "suite.yaml")]) == EXIT_PASS
    assert json.loads(old.read_text(encoding="utf-8"))["format_version"] == 1
    assert rehash_run(load_run(old), cfg).format_version == RUN_FORMAT_VERSION


def test_a_newer_file_is_refused_naming_the_file_and_what_to_do(tmp_path: Path) -> None:
    path = _write(tmp_path / "future.json", format_version=2)
    with pytest.raises(RunFileError) as err:
        load_run(path)
    message = str(err.value)
    assert str(path) in message
    assert "format version 2" in message
    assert "newer evalproof" in message
    assert "reads up to format version 1" in message
    assert "upgrade evalproof" in message


def test_a_newer_file_is_refused_before_the_rest_of_it_is_validated(tmp_path: Path) -> None:
    """A future layout may well not match today's models, and "upgrade" is the useful answer."""
    path = tmp_path / "future.json"
    path.write_text(json.dumps({"format_version": 7, "runs": []}), encoding="utf-8")
    with pytest.raises(RunFileError, match="upgrade evalproof"):
        load_run(path)


@pytest.mark.parametrize("bad", ["2", 1.5, 2.0, True, None, [1], 0, -1])
def test_a_format_version_that_is_not_a_whole_number_from_1_is_refused(
    tmp_path: Path, bad: object
) -> None:
    path = _write(tmp_path / "bad.json", format_version=bad)
    with pytest.raises(RunFileError, match="format_version"):
        load_run(path)


@pytest.mark.parametrize("args", [["report"], ["verify", "SUITE"], ["rehash", "SUITE"]])
def test_a_newer_file_is_a_usage_error_from_the_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    path = _write(tmp_path / "future.json", format_version=2)
    suite = tmp_path / "suite.yaml"
    suite.write_text("name: s\n", encoding="utf-8")  # never reached: the run file is read first
    argv = [args[0], str(path), *[str(suite) for a in args[1:]]]
    assert main(argv) == EXIT_USAGE
    err = capsys.readouterr().err
    assert err.startswith("evalproof: error: ")
    assert str(path) in err
    assert "upgrade evalproof" in err


def test_compare_refuses_a_newer_file_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    ok = save_run(make_run({"a": {"m": True}}), tmp_path / "ok.json")
    future = _write(tmp_path / "future.json", format_version=2)
    assert main(["compare", str(ok), str(future)]) == EXIT_USAGE
    assert "upgrade evalproof" in capsys.readouterr().err
    assert main(["compare", str(future), str(ok), "--gate"]) == EXIT_USAGE


def test_a_file_that_is_not_json_still_gets_the_old_error(tmp_path: Path) -> None:
    path = tmp_path / "junk.json"
    path.write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_run(path)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_run(path)
