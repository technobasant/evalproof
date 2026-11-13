"""``evalproof verify``: a run file must match the suite and dataset it is committed with."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.config import load_config, suite_fingerprint
from evalproof.runner import load_run, run_suite, save_run
from evalproof.verify import verify_run

PROMPT = "Answer in capitals."


def _write(tmp_path: Path, *, cases: list[dict[str, object]] | None = None, **opts: str) -> Path:
    """A python-target suite. ``opts`` override the system prompt, scorer params and label."""
    cases = cases if cases is not None else _cases()
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    (tmp_path / "answer.py").write_text("def ask(x):\n    return x['q'].upper()\n", "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\n"
        f"system: {{target: 'python:answer:ask', system_prompt: '{opts.get('prompt', PROMPT)}', "
        f"label: '{opts.get('label', 'v1')}'}}\n"
        f"scorers:\n  - {{name: m, type: contains, params: {{any_of_field: accept, "
        f"case_sensitive: {opts.get('case_sensitive', 'false')}}}}}\n"
        "gate: {metric: m, min_cases: 10}\n",
        "utf-8",
    )
    return suite


def _cases(n: int = 12) -> list[dict[str, object]]:
    return [{"id": f"c{i:02d}", "input": {"q": f"q{i}"}, "accept": [f"Q{i}"]} for i in range(n)]


def _made(tmp_path: Path, **kw: object):
    suite = _write(tmp_path)
    run = run_suite(load_config(suite), use_cache=False, **kw)  # type: ignore[arg-type]
    return suite, save_run(run, tmp_path / "run.json")


def _verify(tmp_path: Path, run: Path, suite: Path, *extra: str) -> int:
    return main(["verify", str(run), str(suite), *extra])


def test_a_fresh_run_verifies(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite, run = _made(tmp_path)
    assert _verify(tmp_path, run, suite) == 0
    out = capsys.readouterr().out
    assert "evalproof verify" in out and "12 cases" in out


def test_the_verdict_is_appended_to_the_markdown_file(tmp_path: Path) -> None:
    suite, run = _made(tmp_path)
    summary = tmp_path / "summary.md"
    summary.write_text("earlier\n", "utf-8")
    assert _verify(tmp_path, run, suite, "--markdown", str(summary)) == 0
    text = summary.read_text("utf-8")
    assert text.startswith("earlier\n") and "matches" in text


def test_a_label_change_alone_still_verifies(tmp_path: Path) -> None:
    _, run = _made(tmp_path)
    relabelled = _write(tmp_path, label="v2")
    assert _verify(tmp_path, run, relabelled) == 0


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"prompt": "Answer in lower case."}, "system, judge, scorers or seed differ"),
        ({"case_sensitive": "true"}, "system, judge, scorers or seed differ"),
    ],
)
def test_a_changed_suite_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, str], message: str
) -> None:
    _, run = _made(tmp_path)
    changed = _write(tmp_path, **change)
    assert _verify(tmp_path, run, changed) == 4
    assert message in capsys.readouterr().out


def test_an_added_case_fails_and_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, run = _made(tmp_path)
    extra = [*_cases(), {"id": "new", "input": {"q": "z"}, "accept": ["Z"]}]
    suite = _write(tmp_path, cases=extra)
    assert _verify(tmp_path, run, suite) == 4
    out = capsys.readouterr().out
    assert "missing from the run" in out and "`new`" in out


def test_an_edited_case_fails_and_is_named(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, run = _made(tmp_path)
    cases = _cases()
    cases[4]["reference"] = "now with a reference"
    suite = _write(tmp_path, cases=cases)
    assert _verify(tmp_path, run, suite) == 4
    out = capsys.readouterr().out
    assert "changed since the run was made" in out and "`c04`" in out


def test_a_removed_case_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    _, run = _made(tmp_path)
    suite = _write(tmp_path, cases=_cases()[:-1])
    assert _verify(tmp_path, run, suite) == 4
    assert "not in the dataset any more" in capsys.readouterr().out


def test_a_limit_run_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _write(tmp_path)
    run = save_run(run_suite(load_config(suite), limit=5, use_cache=False), tmp_path / "run.json")
    assert _verify(tmp_path, run, suite) == 4
    out = capsys.readouterr().out
    assert "7 case(s) in the dataset are missing from the run" in out and "--limit" in out


def test_a_run_without_a_suite_hash_fails_with_the_fix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _made(tmp_path)
    data = json.loads(run.read_text("utf-8"))
    del data["info"]["suite_hash"]
    run.write_text(json.dumps(data), "utf-8")
    assert _verify(tmp_path, run, suite) == 4
    assert "re-run with evalproof 0.5 or newer" in capsys.readouterr().out


def test_a_run_without_case_hashes_fails(tmp_path: Path) -> None:
    suite, path = _made(tmp_path)
    run = load_run(path)
    for r in run.results:
        r.case_hash = None
    result = verify_run(run, load_config(suite))
    assert not result.ok and any("changed since" in p for p in result.problems)


def test_long_lists_of_ids_are_cut(tmp_path: Path) -> None:
    suite = _write(tmp_path, cases=_cases(30))
    run = run_suite(load_config(suite), limit=3, use_cache=False)
    result = verify_run(run, load_config(suite))
    (missing,) = [p for p in result.problems if "missing from the run" in p]
    assert "27 case(s)" in missing and "and 17 more" in missing and missing.count("`c") == 10


def test_a_usage_error_is_exit_2(tmp_path: Path) -> None:
    suite, _ = _made(tmp_path)
    assert _verify(tmp_path, tmp_path / "nope.json", suite) == 2
    assert _verify(tmp_path, tmp_path / "run.json", tmp_path / "nope.yaml") == 2


def test_the_fingerprint_ignores_what_does_not_change_scores(tmp_path: Path) -> None:
    base = suite_fingerprint(load_config(_write(tmp_path)))
    assert suite_fingerprint(load_config(_write(tmp_path, label="other"))) == base
    assert suite_fingerprint(load_config(_write(tmp_path, prompt="different"))) != base
