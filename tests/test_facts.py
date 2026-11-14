"""The ``facts`` scorer: numbers (within a tolerance) and phrases a response must contain."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig, TargetConfig, load_config
from evalproof.models import Case, SystemOutput
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import DETERMINISTIC_TYPES, Facts, is_deterministic, make_scorer
from evalproof.verify import verify_run


def out(text: str) -> SystemOutput:
    return SystemOutput(output=text)


def case(**fields: Any) -> Case:
    return Case(id="c", input="q", **fields)  # type: ignore[arg-type]


def test_every_fact_must_match_some_number_in_the_output() -> None:
    scorer = Facts("f")
    c = case(facts=[30, 1200])
    assert scorer.score(c, out("30 days, 1,200 per year")).passed is True
    partial = scorer.score(c, out("30 days"))
    assert partial.passed is False and partial.value == 0.5
    assert partial.reason == "missing: 1200"
    assert scorer.score(c, out("no numbers here")).value == 0.0


def test_a_tolerance_is_relative_absolute_or_per_fact() -> None:
    c = case(facts=[100])
    assert Facts("f").score(c, out("about 101")).passed is False
    assert Facts("f", rel_tol=0.02).score(c, out("about 101")).passed is True
    assert Facts("f", abs_tol=1).score(c, out("about 101")).passed is True
    assert Facts("f", abs_tol=1).score(c, out("about 103")).passed is False
    own = case(facts=[{"value": 100, "rel_tol": 0.1}, {"value": 5}])
    assert Facts("f").score(own, out("95 and 5")).passed is True
    assert Facts("f").score(own, out("95 and 6")).passed is False


def test_an_optional_fact_lifts_the_value_but_not_the_verdict() -> None:
    c = case(facts=[{"value": 30}, {"value": 7, "required": False, "label": "trial days"}])
    miss_optional = Facts("f").score(c, out("30 days"))
    assert miss_optional.passed is True and miss_optional.value == 0.5
    assert miss_optional.reason == "missing: trial days (optional)"
    miss_required = Facts("f").score(c, out("7 days"))
    assert miss_required.passed is False and miss_required.reason == "missing: 30"
    assert Facts("f").score(c, out("30 and 7")).value == 1.0


def test_mentions_must_appear_and_count_in_the_value() -> None:
    c = case(facts=[30], must_mention=["Refund Policy", "pro plan"])
    full = Facts("f").score(c, out("30 days, per our refund   policy, on the Pro plan."))
    assert full.passed is True and full.value == 1.0
    some = Facts("f").score(c, out("30 days under the refund policy"))
    assert some.passed is False and some.value == pytest.approx(2 / 3)
    assert some.reason == "missing: 'pro plan'"
    exact = Facts("f", normalize=False).score(case(must_mention=["Pro"]), out("pro"))
    assert exact.passed is False
    assert Facts("f").score(case(must_mention="refund"), out("A refund.")).passed is True


def test_numbers_in_a_string_or_with_commas_are_read() -> None:
    c = case(facts=["1,234", {"value": "5.5"}, 2])
    assert Facts("f").score(c, out("1,234 and 5.5 and 2")).passed is True
    assert Facts("f").score(case(facts=7), out("it is 7")).passed is True  # one fact, no list


def test_a_case_with_no_facts_or_mentions_is_undecided() -> None:
    score = Facts("f").score(case(), out("30"))
    assert score.value is None and score.reason == "case has no facts or must_mention"
    custom = Facts("f", field="numbers", mention_field="phrases")
    assert custom.score(case(numbers=[3], phrases=["x"]), out("3 x")).passed is True
    assert custom.score(case(facts=[3]), out("3")).value is None


@pytest.mark.parametrize(
    ("facts", "message"),
    [
        (["thirty"], "fact value 'thirty' is not a number"),
        ([True], "fact value True is not a number"),
        ([{"label": "x"}], "fact value None is not a number"),
        ([{"value": float("inf")}], "is not a number"),
        ([{"value": 3, "tol": 1}], "fact has unknown key 'tol'"),
        ([{"value": 3, "rel_tol": -1}], "rel_tol must be a number >= 0"),
        ([{"value": 3, "abs_tol": "x"}], "abs_tol must be a number >= 0"),
        ([{"value": 3, "required": "yes"}], "'required' must be true or false"),
    ],
)
def test_a_malformed_fact_raises_naming_the_case(facts: list[Any], message: str) -> None:
    with pytest.raises(ValueError, match=f"case 'c': .*{message}"):
        Facts("f").score(case(facts=facts), out("3"))


def test_negative_tolerances_are_refused_at_construction() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        Facts("f", rel_tol=-0.1)
    with pytest.raises(ValueError, match=">= 0"):
        Facts("f", abs_tol=-1)


def test_facts_is_a_deterministic_scorer_that_make_scorer_builds() -> None:
    assert "facts" in DETERMINISTIC_TYPES
    cfg = ScorerConfig(name="f", type="facts", params={"rel_tol": 0.01})
    assert is_deterministic(cfg)
    suite = SuiteConfig(
        name="s", dataset="d.jsonl", system=TargetConfig(target="ollama:m"), scorers=[cfg]
    )
    built = make_scorer(cfg, suite)
    assert isinstance(built, Facts) and built.rel_tol == 0.01
    with pytest.raises(ConfigError):
        make_scorer(ScorerConfig(name="f", type="facts", params={"rel_tolerance": 1}), suite)


def verify_suite(tmp_path: Path) -> Path:
    rows = [
        {"id": f"c{i}", "input": {"q": f"{i}"}, "facts": [i + 10], "must_mention": ["days"]}
        for i in range(4)
    ]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (tmp_path / "facts_answer.py").write_text(
        "def ask(x):\n    return f\"{int(x['q']) + 10} days\"\n", "utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:facts_answer:ask'}\n"
        "scorers:\n  - {name: f, type: facts}\n",
        "utf-8",
    )
    return suite


def test_verify_re_scores_facts_and_catches_an_edited_score(tmp_path: Path) -> None:
    cfg = load_config(verify_suite(tmp_path))
    run = run_suite(cfg, use_cache=False)
    assert all(r.scores["f"].passed for r in run.results)
    path = save_run(run, tmp_path / "run.json")

    ok = verify_run(load_run(path), cfg)
    assert ok.ok and ok.rescored == ["f"] and ok.not_rescored == []

    data = json.loads(path.read_text("utf-8"))
    data["results"][0]["scores"]["f"] = {"value": 0.0, "passed": False}
    path.write_text(json.dumps(data), "utf-8")
    bad = verify_run(load_run(path), cfg)
    assert not bad.ok and any("`c0` on `f`" in p for p in bad.problems)
