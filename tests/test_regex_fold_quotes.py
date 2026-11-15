"""``fold_quotes`` on the regex scorer: match an output's curly quotes as plain ones.

The option is off by default and changes nothing for a suite that does not set it: the scores and
the suite fingerprint a committed run file carries stay exactly as they were.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.config import ConfigError, ScorerConfig, SuiteConfig, load_config, suite_fingerprint
from evalproof.models import Case, SystemOutput
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import Regex, make_scorer
from evalproof.scorers.deterministic import QUOTE_FOLD
from evalproof.verify import verify_run

# U+2019 U+2018 U+201B U+2032 U+FF07, and U+201C U+201D U+201F U+2033 U+FF02.
SINGLE = "\u2019 \u2018 \u201b \u2032 \uff07"
DOUBLE = "\u201c \u201d \u201f \u2033 \uff02"


def out(text: str) -> SystemOutput:
    return SystemOutput(output=text)


def case(**fields: list[str]) -> Case:
    return Case(id="c", input="q", **fields)  # type: ignore[arg-type]


def test_a_curly_apostrophe_fails_without_the_option_and_passes_with_it() -> None:
    c = case(accept=["it's open"])
    curly = out("Yes, it\u2019s open today.")
    assert Regex("r", any_of_field="accept").score(c, curly).passed is False
    assert Regex("r", any_of_field="accept", fold_quotes=False).score(c, curly).passed is False
    assert Regex("r", any_of_field="accept", fold_quotes=True).score(c, curly).passed is True


def test_every_listed_single_and_double_quote_is_folded() -> None:
    singles = Regex("r", pattern="^(' ){4}'$", fold_quotes=True)
    doubles = Regex("r", pattern='^(" ){4}"$', fold_quotes=True)
    assert singles.score(case(), out(SINGLE)).passed is True
    assert doubles.score(case(), out(DOUBLE)).passed is True
    assert set(QUOTE_FOLD) == {ord(ch) for ch in SINGLE + DOUBLE if ch != " "}


def test_a_curly_quote_around_a_quoted_phrase_matches_a_plain_pattern() -> None:
    s = Regex("r", pattern='says "hello"', fold_quotes=True)
    assert s.score(case(), out("It says \u201chello\u201d.")).passed is True
    assert (
        Regex("r", pattern='says "hello"').score(case(), out("It says \u201chello\u201d.")).passed
        is False
    )


def test_a_reject_pattern_sees_the_folded_output_too() -> None:
    c = case(accept=["open"], reject=["don't"])
    text = out("We are open, but don\u2019t come early.")
    assert Regex("r", any_of_field="accept", none_of_field="reject").score(c, text).passed is True
    folded = Regex("r", any_of_field="accept", none_of_field="reject", fold_quotes=True)
    result = folded.score(c, text)
    assert result.passed is False and "don't" in (result.reason or "")


def test_the_patterns_are_used_as_written_only_the_output_is_folded() -> None:
    # A curly quote in the pattern is not folded, so it no longer meets the (folded) output.
    s = Regex("r", pattern="it\u2019s", fold_quotes=True)
    assert s.score(case(), out("it\u2019s")).passed is False
    assert s.score(case(), out("it's")).passed is False
    assert Regex("r", pattern="it\u2019s").score(case(), out("it\u2019s")).passed is True
    # And the case's own accept list is not folded either.
    c = case(accept=["it\u2019s"])
    assert (
        Regex("r", any_of_field="accept", fold_quotes=True).score(c, out("it\u2019s")).passed
        is False
    )


def test_other_characters_are_left_alone() -> None:
    s = Regex("r", pattern="caf\u00e9 \u2014 ok", fold_quotes=True)
    assert s.score(case(), out("caf\u00e9 \u2014 ok")).passed is True


def test_a_value_that_is_not_a_bool_is_rejected() -> None:
    for bad in ("true", 1, None, "false"):
        with pytest.raises(TypeError, match="fold_quotes must be true or false"):
            Regex("r", pattern="x", fold_quotes=bad)  # type: ignore[arg-type]


# --- config --------------------------------------------------------------------------------------

BASE = {
    "name": "s",
    "dataset": "d.jsonl",
    "system": {"target": "python:app:answer"},
    "scorers": [
        {
            "name": "r",
            "type": "regex",
            "params": {"any_of_field": "accept", "none_of_field": "reject", "flags": "i"},
        }
    ],
}
# The fingerprint evalproof computed for BASE before the option existed (release/1.0 at its start).
# A suite that does not set fold_quotes must hash exactly like this, so a committed run file keeps
# verifying: this value may only change on purpose, with a note in the changelog.
BASE_FINGERPRINT = "2640179cdc02ca25"


def with_params(**extra: object) -> SuiteConfig:
    params = {**BASE["scorers"][0]["params"], **extra}  # type: ignore[index]
    scorer = {**BASE["scorers"][0], "params": params}  # type: ignore[dict-item]
    return SuiteConfig.model_validate({**BASE, "scorers": [scorer]})


def test_a_suite_without_the_key_hashes_exactly_as_before() -> None:
    assert suite_fingerprint(SuiteConfig.model_validate(BASE)) == BASE_FINGERPRINT


def test_setting_the_option_changes_the_hash_because_it_changes_the_scores() -> None:
    assert suite_fingerprint(with_params(fold_quotes=True)) != BASE_FINGERPRINT


def test_the_option_flows_from_config_to_the_scorer_and_round_trips() -> None:
    cfg = with_params(fold_quotes=True)
    scorer = make_scorer(cfg.scorers[0], cfg)
    assert isinstance(scorer, Regex) and scorer.fold_quotes is True
    assert make_scorer(SuiteConfig.model_validate(BASE).scorers[0], cfg).fold_quotes is False  # type: ignore[attr-defined]
    again = SuiteConfig.model_validate(json.loads(cfg.model_dump_json()))
    assert again.scorers[0].params["fold_quotes"] is True
    assert suite_fingerprint(again) == suite_fingerprint(cfg)
    back = ScorerConfig.model_validate(cfg.scorers[0].model_dump(mode="json"))
    assert back == cfg.scorers[0]


def test_an_unknown_regex_key_is_still_an_error_naming_the_scorer() -> None:
    cfg = with_params(fold_quote=True)  # a typo
    with pytest.raises(ConfigError, match=r"scorer 'r'.*fold_quote"):
        make_scorer(cfg.scorers[0], cfg)


def test_a_non_bool_value_stops_the_suite_at_load(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"fold_quotes must be true or false, got 'yes'"):
        with_params(fold_quotes="yes")
    suite = tmp_path / "suite.yaml"
    params = {**BASE["scorers"][0]["params"], "fold_quotes": 1}  # type: ignore[index]
    scorer = {**BASE["scorers"][0], "params": params}  # type: ignore[dict-item]
    suite.write_text(json.dumps({**BASE, "scorers": [scorer]}), encoding="utf-8")
    with pytest.raises(ConfigError, match=r"fold_quotes must be true or false, got 1"):
        load_config(suite)


# --- verify --------------------------------------------------------------------------------------


def write_suite(tmp_path: Path, *, fold: bool) -> Path:
    cases = [{"id": f"c{i}", "input": {"q": f"q{i}"}, "accept": [f"it's q{i}"]} for i in range(12)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    (tmp_path / "fold_quotes_answer.py").write_text(
        "def ask(x):\n    return 'It\\u2019s ' + x['q'] + '.'\n", "utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:fold_quotes_answer:ask'}\n"
        "scorers:\n  - name: r\n    type: regex\n"
        f"    params: {{any_of_field: accept, fold_quotes: {str(fold).lower()}}}\n"
        "gate: {metric: r, min_cases: 10}\n",
        "utf-8",
    )
    return suite


def test_verify_rescores_with_the_option(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = write_suite(tmp_path, fold=True)
    run = run_suite(load_config(suite), use_cache=False)
    # The answers carry a curly apostrophe and the answer key a plain one: only a fold passes.
    assert all(r.scores["r"].passed is True for r in run.results)
    path = save_run(run, tmp_path / "run.json")
    assert verify_run(load_run(path), load_config(suite)).rescored == ["r"]
    assert main(["verify", str(path), str(suite)]) == 0
    assert "re-scored the stored outputs with r." in capsys.readouterr().out


def test_the_same_suite_without_the_option_scores_differently_and_does_not_verify(
    tmp_path: Path,
) -> None:
    folded = write_suite(tmp_path, fold=True)
    path = save_run(run_suite(load_config(folded), use_cache=False), tmp_path / "run.json")
    plain = write_suite(tmp_path, fold=False)
    assert all(
        r.scores["r"].passed is False
        for r in run_suite(load_config(plain), use_cache=False).results
    )
    assert main(["verify", str(path), str(plain)]) != 0
