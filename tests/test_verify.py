"""``evalproof verify``: a run file must match the suite and dataset it is committed with."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalproof.cli import main
from evalproof.config import load_config, suite_fingerprint
from evalproof.models import Score
from evalproof.runner import load_run, run_suite, save_run
from evalproof.verify import _same_score, verify_run
from tests.conftest import mock_client

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


# --- re-scoring: the deterministic scores are checked, the model-judged ones are not ----------


def _edit(run: Path, change: Callable[[dict[str, Any]], None]) -> None:
    data = json.loads(run.read_text("utf-8"))
    change(data)
    run.write_text(json.dumps(data), "utf-8")


def _judge_handler(req: httpx.Request) -> httpx.Response:
    if req.url.path == "/api/chat":
        reply = '{"reason": "fine", "pass": true, "score": 1}'
        return httpx.Response(200, json={"message": {"content": reply}})
    return httpx.Response(200, json={"version": "0.0.test", "models": []})


def _judged(tmp_path: Path, *, with_contains: bool) -> tuple[Path, Path]:
    """A suite with a model judge (an offline mock) and, optionally, the deterministic scorer."""
    suite = _write(tmp_path)
    scorers = "  - {name: j, type: llm_judge}\n"
    if with_contains:
        scorers += "  - {name: m, type: contains, params: {any_of_field: accept}}\n"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\n"
        "system: {target: 'python:answer:ask', system_prompt: 'Answer in capitals.'}\n"
        f"judge: {{target: 'ollama:judge-m'}}\nscorers:\n{scorers}"
        f"gate: {{metric: {'m' if with_contains else 'j'}, min_cases: 10}}\n",
        "utf-8",
    )
    run = run_suite(load_config(suite), client=mock_client(_judge_handler), use_cache=False)
    return suite, save_run(run, tmp_path / "run.json")


def test_an_untouched_run_is_rescored_and_says_so(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _made(tmp_path)
    assert verify_run(load_run(run), load_config(suite)).rescored == ["m"]
    assert _verify(tmp_path, run, suite) == 0
    assert "re-scored the stored outputs with m." in capsys.readouterr().out


def test_a_hand_edited_score_fails_and_names_the_case(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _made(tmp_path)

    def flip(data: dict[str, Any]) -> None:
        score = data["results"][3]["scores"]["m"]
        assert score["passed"] is True and score["value"] == 1.0
        score["passed"], score["value"] = False, 0.0

    _edit(run, flip)
    assert _verify(tmp_path, run, suite) == 4
    out = capsys.readouterr().out
    assert "1 score(s) differ from re-scoring the stored outputs" in out
    assert "`c03` on `m`" in out and "`c02`" not in out


def test_a_hand_edited_value_alone_fails(tmp_path: Path) -> None:
    suite, run = _made(tmp_path)
    _edit(run, lambda d: d["results"][0]["scores"]["m"].update(value=0.5))
    result = verify_run(load_run(run), load_config(suite))
    assert not result.ok and any("`c00` on `m`" in p for p in result.problems)


def test_a_deleted_score_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite, run = _made(tmp_path)
    _edit(run, lambda d: d["results"][7]["scores"].pop("m"))
    assert _verify(tmp_path, run, suite) == 4
    assert "`c07` on `m`" in capsys.readouterr().out


def test_a_changed_case_is_reported_once_not_as_a_score_difference(tmp_path: Path) -> None:
    _, run = _made(tmp_path)
    cases = _cases()
    cases[4]["accept"] = ["something the output never says"]  # the stored score is now stale
    suite = _write(tmp_path, cases=cases)
    result = verify_run(load_run(run), load_config(suite))
    assert any("changed since the run was made" in p for p in result.problems)
    assert not any("score(s) differ" in p for p in result.problems)


def test_long_lists_of_differing_scores_are_cut(tmp_path: Path) -> None:
    suite, run = _made(tmp_path)

    def flip_all(data: dict[str, Any]) -> None:
        for r in data["results"]:
            r["scores"]["m"].update(passed=False, value=0.0)

    _edit(run, flip_all)
    (problem,) = verify_run(load_run(run), load_config(suite)).problems
    assert "12 score(s) differ" in problem and "and 2 more" in problem
    assert problem.count("on `m`") == 10


def test_a_scorer_that_cannot_be_built_is_a_problem_not_a_crash(tmp_path: Path) -> None:
    suite, run = _made(tmp_path)
    suite.write_text(suite.read_text("utf-8").replace("case_sensitive: false", "bogus: 1"), "utf-8")
    result = verify_run(load_run(run), load_config(suite))
    assert not result.ok and result.rescored == []
    assert any("scorer `m` could not be built" in p for p in result.problems)


def test_a_suite_with_only_a_model_judge_has_nothing_to_rescore(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _judged(tmp_path, with_contains=False)
    assert _verify(tmp_path, run, suite) == 0
    out = capsys.readouterr().out
    assert "No scorer could be re-scored without a model call" in out
    assert "so the scores themselves are not checked" in out and "re-scored the stored" not in out


def test_model_judged_scores_are_not_checked(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite, run = _judged(tmp_path, with_contains=True)
    _edit(run, lambda d: d["results"][0]["scores"]["j"].update(passed=False, value=0.0))
    assert _verify(tmp_path, run, suite) == 0  # a model call would be needed to tell
    out = capsys.readouterr().out
    assert "re-scored the stored outputs with m." in out
    assert "Not checked (they need a model call): j." in out
    _edit(run, lambda d: d["results"][0]["scores"]["m"].update(passed=False, value=0.0))
    assert _verify(tmp_path, run, suite) == 4  # the deterministic scorer next to it still is


def test_scores_match_to_float_noise_and_ignore_the_reason() -> None:
    fresh = Score(value=0.3, passed=True, reason="a")
    assert _same_score(Score(value=0.3 + 1e-12, passed=True, reason="b"), fresh)
    assert not _same_score(Score(value=0.3001, passed=True), fresh)
    assert not _same_score(Score(value=0.3, passed=False), fresh)
    assert not _same_score(None, fresh)
    undecided = Score(value=None)
    assert _same_score(Score(value=None, reason="x"), undecided)
    assert not _same_score(Score(value=0.0, passed=None), undecided)
