"""The ``python`` scorer: a function of your own, and ``verify`` re-scoring only the ones that
promise to be deterministic."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig, TargetConfig, load_config
from evalproof.models import Case, Score, SystemOutput
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import (
    DETERMINISTIC_TYPES,
    PythonScorer,
    is_deterministic,
    make_scorer,
)
from evalproof.verify import verify_run

CASE = Case(id="c1", input="q")


def _scorer(fn: str, **params: Any) -> PythonScorer:
    return PythonScorer("p", f"tests.scorer_fns:{fn}", **params)


def _score(fn: str, output: str = "all good here, ok", **params: Any) -> Score:
    return _scorer(fn, **params).score(CASE, SystemOutput(output=output))


# --- return types -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fn", "output", "value", "passed"),
    [
        ("long_enough", "one two three", 1.0, True),
        ("long_enough", "one two", 0.0, False),
        ("short", "one", 1.0, True),
    ],
)
def test_a_bool_is_a_binary_score(fn: str, output: str, value: float, passed: bool) -> None:
    score = _score(fn, output)
    assert (score.value, score.passed, score.reason) == (value, passed, None)


def test_a_float_is_the_value_and_passes_at_the_threshold() -> None:
    low = _score("length_share", "abcd")  # 0.4
    assert (low.value, low.passed) == (0.4, False)
    high = _score("length_share", "abcdef")  # 0.6
    assert (high.value, high.passed) == (0.6, True)
    exact = _score("length_share", "abcde")  # 0.5 is at the default threshold of 0.5
    assert (exact.value, exact.passed) == (0.5, True)
    strict = _score("length_share", "abcdef", threshold=0.9)
    assert (strict.value, strict.passed) == (0.6, False)


def test_an_int_of_zero_or_one_is_a_number_score() -> None:
    score = _score("exact_one")
    assert (score.value, score.passed) == (1.0, True)


def test_a_bool_and_reason_pair() -> None:
    bad = _score("with_reason", "nothing")
    assert (bad.value, bad.passed, bad.reason) == (0.0, False, "the reply never says ok")
    good = _score("with_reason", "ok then")
    assert (good.value, good.passed) == (1.0, True)
    assert _score("with_no_reason", "ok").reason is None


def test_a_score_passes_through_untouched() -> None:
    score = _score("returns_score")
    assert (score.value, score.passed, score.reason) == (0.25, False, "a hand-built score")
    undecided = _score("undecided_score")
    assert undecided.value is None and undecided.passed is None


@pytest.mark.parametrize(
    ("fn", "reason"),
    [
        ("raises", "scorer error: ValueError: boom"),
        ("returns_none", "scorer error: returned NoneType; expected a Score, a bool"),
        ("returns_text", "scorer error: returned str; expected"),
        ("too_big", "scorer error: returned 1.5; a number score must be between 0 and 1"),
        ("not_a_number", "scorer error: returned nan"),
        ("three_items", "scorer error: returned tuple; expected"),
        ("bad_pair", "scorer error: returned tuple; expected"),
    ],
)
def test_an_exception_or_another_return_type_is_undecided(fn: str, reason: str) -> None:
    score = _score(fn)
    assert score.value is None and score.passed is None
    assert score.reason is not None and score.reason.startswith(reason)


def test_the_function_receives_the_case_and_the_output() -> None:
    scorer = _scorer("case_aware")
    assert scorer.score(CASE, SystemOutput(output="the c1 answer")).passed is True
    assert scorer.score(CASE, SystemOutput(output="other")).passed is False


# --- building it ------------------------------------------------------------------------------


def test_a_function_that_cannot_be_found_fails_when_the_scorer_is_built() -> None:
    for spec, message in [
        ("tests.scorer_fns:nope", "tests.scorer_fns.nope is not callable"),
        ("tests.scorer_fns:NOT_CALLABLE", "tests.scorer_fns.NOT_CALLABLE is not callable"),
        ("tests.no_such_module:fn", "cannot import 'tests.no_such_module'"),
        ("no colon", "fn must be 'module:function'"),
    ]:
        with pytest.raises(ConfigError, match=message):
            PythonScorer("p", spec)


def test_the_suite_directory_is_searched_for_the_module(tmp_path: Path) -> None:
    (tmp_path / "my_scorer_mod.py").write_text(
        "def ok(case, out):\n    return True\n\ndef nope(case, out):\n    return False\n", "utf-8"
    )
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="python:a:b"),
        scorers=[ScorerConfig(name="p", type="python", params={"fn": "my_scorer_mod:ok"})],
        base_dir=str(tmp_path),
    )
    scorer = make_scorer(suite.scorers[0], suite)
    assert isinstance(scorer, PythonScorer)
    assert scorer.score(CASE, SystemOutput(output="x")).passed is True


def test_make_scorer_names_the_scorer_in_a_bad_build() -> None:
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="python:a:b"),
        scorers=[ScorerConfig(name="mine", type="python", params={"fn": "tests.nope:fn"})],
    )
    with pytest.raises(ConfigError, match=r"scorer 'mine': cannot import 'tests\.nope'"):
        make_scorer(suite.scorers[0], suite)


# --- params at suite load ---------------------------------------------------------------------


def _suite_file(tmp_path: Path, params: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\ndataset: d.jsonl\nsystem: {target: 'python:a:b'}\n"
        f"scorers:\n  - {{name: mine, type: python, params: {params}}}\n",
        "utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ("{}", "params.fn must be 'module:function', got None"),
        ("{fn: nocolon}", "params.fn must be 'module:function', got 'nocolon'"),
        ("{fn: 'a:b', deterministic: yes please}", "params.deterministic must be true or false"),
        ("{fn: 'a:b', threshold: 2}", "params.threshold must be a number between 0 and 1"),
        ("{fn: 'a:b', threshold: true}", "params.threshold must be a number between 0 and 1"),
        ("{fn: 'a:b', treshold: 1}", "params has unknown key 'treshold'"),
    ],
)
def test_python_params_are_validated_when_the_suite_loads(
    tmp_path: Path, params: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=r"suite\.yaml: scorers\.0: " + message):
        load_config(_suite_file(tmp_path, params))


def test_good_python_params_load(tmp_path: Path) -> None:
    cfg = load_config(_suite_file(tmp_path, "{fn: 'a:b', deterministic: true, threshold: 0.7}"))
    assert cfg.scorers[0].params == {"fn": "a:b", "deterministic": True, "threshold": 0.7}


# --- verify -----------------------------------------------------------------------------------


def test_only_an_instance_that_says_so_is_deterministic() -> None:
    def cfg(**params: Any) -> ScorerConfig:
        return ScorerConfig(name="p", type="python", params={"fn": "a:b", **params})

    assert "python" not in DETERMINISTIC_TYPES and "checks" in DETERMINISTIC_TYPES
    assert not is_deterministic(cfg())
    assert not is_deterministic(cfg(deterministic=False))
    assert is_deterministic(cfg(deterministic=True))
    assert is_deterministic(ScorerConfig(name="c", type="checks"))


def _verify_suite(tmp_path: Path, *, deterministic: bool) -> Path:
    (tmp_path / "cases.jsonl").write_text(
        "\n".join(json.dumps({"id": f"c{i}", "input": {"q": f"w{i}"}}) for i in range(4)) + "\n",
        "utf-8",
    )
    (tmp_path / "py_scorer_answer.py").write_text(
        "def ask(x):\n    return x['q'] + ' ok'\n", "utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:py_scorer_answer:ask'}\n"
        "scorers:\n"
        f"  - {{name: mine, type: python, params: {{fn: 'tests.scorer_fns:with_reason', "
        f"deterministic: {str(deterministic).lower()}}}}}\n"
        "  - {name: other, type: contains, params: {all_of: [ok]}}\n",
        "utf-8",
    )
    return suite


def _tamper(run_path: Path, scorer: str) -> None:
    data = json.loads(run_path.read_text("utf-8"))
    data["results"][0]["scores"][scorer] = {"value": 0.0, "passed": False}
    run_path.write_text(json.dumps(data), "utf-8")


def test_verify_re_scores_a_deterministic_python_scorer(tmp_path: Path) -> None:
    suite = _verify_suite(tmp_path, deterministic=True)
    cfg = load_config(suite)
    run_path = save_run(run_suite(cfg, use_cache=False), tmp_path / "run.json")

    ok = verify_run(load_run(run_path), cfg)
    assert ok.ok and ok.rescored == ["mine", "other"] and ok.not_rescored == []

    _tamper(run_path, "mine")
    bad = verify_run(load_run(run_path), cfg)
    assert not bad.ok
    assert any("`c0` on `mine`" in p for p in bad.problems)


def test_verify_skips_a_python_scorer_that_does_not_promise_it(tmp_path: Path) -> None:
    suite = _verify_suite(tmp_path, deterministic=False)
    cfg = load_config(suite)
    run_path = save_run(run_suite(cfg, use_cache=False), tmp_path / "run.json")

    _tamper(run_path, "mine")  # a hand-edited score nobody can check is not an error
    result = verify_run(load_run(run_path), cfg)
    assert result.ok
    assert result.rescored == ["other"] and result.not_rescored == ["mine"]

    _tamper(run_path, "other")  # the deterministic one is still checked
    assert not verify_run(load_run(run_path), cfg).ok
