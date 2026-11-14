"""Rubric sets: an ``llm_judge`` that grades one response against several rubrics, and the
multi-metric scorer interface (``score_all``) that lets one scorer write a metric per rubric."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from evalproof.compare import compare_runs
from evalproof.config import (
    ConfigError,
    GenerationParams,
    Pricing,
    ScorerConfig,
    load_config,
)
from evalproof.models import Case, Score, SystemOutput
from evalproof.providers import Message, ProviderError
from evalproof.runner import run_suite, score_case
from evalproof.scorers import (
    RubricSetJudge,
    make_scorer,
    metric_names,
    score_metrics,
)
from evalproof.scorers.judge import (
    normalize_rubrics,
    parse_rubric_verdicts,
    per_rubric_metric_names,
    validate_judge_params,
)
from tests.conftest import ScriptedProvider, mock_client

OUT = SystemOutput(output="the answer")
CASE = Case(id="c", input="q", reference="ref")


def verdicts(**flags: bool | str) -> str:
    """A batched reply: ``verdicts(a=True, b="no")`` passes a and fails b (a string is the
    failing reason, ``True`` passes)."""
    rows = []
    for rid, flag in flags.items():
        passed = flag is True
        rows.append({"id": rid, "pass": passed, "reason": "" if passed else str(flag)})
    return json.dumps({"rubrics": rows})


def ids_in(messages: list[Message]) -> list[str]:
    """The rubric ids a batched prompt lists."""
    return re.findall(r"(?m)^\[([^\]]+)\] ", messages[-1].content)


def judge(provider: Any, **kw: Any) -> RubricSetJudge:
    return RubricSetJudge("rs", provider, GenerationParams(), **kw)


# --- normalising and parsing ---------------------------------------------------------------------


def test_strings_and_objects_become_rubrics_with_ids() -> None:
    rubrics = normalize_rubrics(["be brief", {"id": "tone", "text": "polite", "weight": 2}])
    assert [(r.id, r.text, r.weight, r.type) for r in rubrics] == [
        ("r1", "be brief", 1.0, None),
        ("tone", "polite", 2.0, None),
    ]


@pytest.mark.parametrize(
    ("items", "message"),
    [
        ("text", "must be a list"),
        ([""], "is empty"),
        ([3], "must be a string or an object"),
        ([{"text": "a", "wieght": 2}], "unknown key 'wieght'"),
        ([{"id": "a"}], "text must be a non-empty string"),
        ([{"text": "a", "weight": 0}], "weight must be a number greater than 0"),
        ([{"text": "a", "weight": True}], "weight must be a number greater than 0"),
        ([{"text": "a", "type": 3}], "type must be a string"),
        ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "repeated ids"),
    ],
)
def test_a_bad_rubric_list_is_refused_naming_where(items: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        normalize_rubrics(items, "params.rubrics")


def test_parse_rubric_verdicts_needs_every_id_and_ignores_invented_ones() -> None:
    ids = ["a", "b"]
    got = parse_rubric_verdicts(verdicts(a=True, b="late", zzz=True), ids)
    assert got == {"a": (True, ""), "b": (False, "late")}
    echoed = json.dumps({"rubrics": [{"id": "[a]", "pass": "yes"}, {"id": "b", "pass": "no"}]})
    assert parse_rubric_verdicts(echoed, ids) == {"a": (True, ""), "b": (False, "")}
    for bad in (
        verdicts(a=True),  # b is missing
        json.dumps({"rubrics": [{"id": "a", "pass": True}] * 2}),  # a twice
        json.dumps({"rubrics": [{"id": "a"}]}),  # no verdict
        json.dumps({"rubrics": "x"}),
        "[]",
    ):
        with pytest.raises(ValueError):
            parse_rubric_verdicts(bad, ids)


def test_the_constructor_refuses_what_the_loader_would() -> None:
    for kw in ({"mode": "x"}, {"num_samples": 0}, {"min_score": 2}):
        with pytest.raises(ValueError):
            judge(ScriptedProvider([]), rubrics=["a"], **kw)
    with pytest.raises(ValueError, match="id must be a non-empty string"):
        normalize_rubrics([{"id": 3, "text": "a"}])
    with pytest.raises(ValueError, match="'pass' must be a boolean"):
        parse_rubric_verdicts(json.dumps({"rubrics": [{"id": "a", "pass": 3}]}), ["a"])


# --- batched mode --------------------------------------------------------------------------------


def test_batched_scores_the_fraction_passed_in_one_call() -> None:
    provider = ScriptedProvider([verdicts(r1=True, b="too long", r3=True)])
    score = judge(provider, rubrics=["x", {"id": "b", "text": "y"}, "z"]).score(CASE, OUT)
    assert score.value == pytest.approx(2 / 3) and score.passed is False
    assert score.reason == "2/3 rubrics; failed: b (too long)"
    assert len(provider.calls) == 1 and score.judge_tokens == 15
    prompt = provider.calls[0][-1].content
    assert "[r1] x" in prompt and "[b] y" in prompt and "Reference answer:\nref" in prompt


def test_a_reply_that_answers_other_ids_is_not_a_verdict() -> None:
    provider = ScriptedProvider([verdicts(a=True, c=True)])
    score = judge(provider, rubrics=[{"id": "a", "text": "a"}, {"id": "b", "text": "b"}]).score(
        CASE, OUT
    )
    assert score.value is None and len(provider.calls) == 2  # asked again, then gave up


def test_min_score_sets_where_a_case_passes() -> None:
    reply = verdicts(a=True, b=True, c="no", d="no")
    rubrics = [{"id": i, "text": i} for i in "abcd"]
    strict = judge(ScriptedProvider([reply]), rubrics=rubrics).score(CASE, OUT)
    lenient = judge(ScriptedProvider([reply]), rubrics=rubrics, min_score=0.5).score(CASE, OUT)
    assert strict.value == lenient.value == 0.5
    assert strict.passed is False and lenient.passed is True


def test_weights_count_in_the_fraction() -> None:
    rubrics = [{"id": "a", "text": "a", "weight": 3}, {"id": "b", "text": "b"}]
    score = judge(ScriptedProvider([verdicts(a="no", b=True)]), rubrics=rubrics).score(CASE, OUT)
    assert score.value == pytest.approx(0.25)


def test_an_unparseable_batched_reply_is_retried_then_undecided() -> None:
    provider = ScriptedProvider(["I think so", verdicts(a=True)])
    score = judge(provider, rubrics=[{"id": "a", "text": "a"}]).score(CASE, OUT)
    assert score.passed is True and score.judge_tokens == 30
    nudge = provider.calls[1][-1]
    assert nudge.role == "user" and "rubric id" in nudge.content

    giving_up = ScriptedProvider(["no", "still no"])
    score = judge(giving_up, rubrics=[{"id": "a", "text": "a"}]).score(CASE, OUT)
    assert score.value is None and score.passed is None
    assert "unparseable" in (score.reason or "") and score.judge_tokens == 30
    single = ScriptedProvider(["no"])
    once = judge(single, rubrics=[{"id": "a", "text": "a"}], max_retries=0).score(CASE, OUT)
    assert once.value is None and len(single.calls) == 1


def test_a_provider_error_is_undecided_not_a_fail() -> None:
    def broken(_messages: list[Message]) -> str:
        raise ProviderError("down")

    score = judge(ScriptedProvider(broken), rubrics=[{"id": "a", "text": "a"}]).score(CASE, OUT)
    assert score.value is None and "judge error: down" in (score.reason or "")


def test_a_system_error_fails_without_asking_the_judge() -> None:
    provider = ScriptedProvider([])
    scorer = judge(provider, rubrics=[{"id": "a", "text": "a"}], per_rubric_metrics=True)
    scores = scorer.score_all(CASE, SystemOutput(output="", error="timeout"))
    assert provider.calls == []
    assert scores["rs"].passed is False and scores["rs/a"].passed is False


def test_no_rubric_for_a_case_is_undecided() -> None:
    provider = ScriptedProvider([])
    scorer = judge(provider)  # no base rubrics: the cases bring their own
    score = scorer.score(CASE, OUT)
    assert score.value is None and score.reason == "no rubrics" and provider.calls == []


# --- which rubrics apply -------------------------------------------------------------------------


def case_with(rubrics: Any) -> Case:
    return Case(id="c", input="q", rubrics=rubrics)  # type: ignore[call-arg]


def test_case_rubrics_join_the_base_and_replace_by_id() -> None:
    scorer = judge(
        ScriptedProvider([]), rubrics=[{"id": "a", "text": "base a"}, {"id": "b", "text": "b"}]
    )
    case = case_with([{"id": "a", "text": "case a"}, {"id": "c", "text": "c"}])
    got = scorer.rubrics_for(case)
    assert [(r.id, r.text) for r in got] == [("a", "case a"), ("b", "b"), ("c", "c")]
    off = judge(ScriptedProvider([]), rubrics=[{"id": "a", "text": "base a"}], case_rubrics=False)
    assert [r.text for r in off.rubrics_for(case)] == ["base a"]


def test_a_malformed_case_rubrics_field_names_the_case() -> None:
    scorer = judge(ScriptedProvider([]), rubrics=[{"id": "a", "text": "a"}])
    with pytest.raises(ValueError, match=r"case 'c': rubrics\[0\]\.text"):
        scorer.score(case_with([{"text": ""}]), OUT)


def test_rubric_types_keep_listed_and_untyped_rubrics() -> None:
    scorer = judge(
        ScriptedProvider([]),
        rubrics=[
            {"id": "a", "text": "a", "type": "SAFETY"},
            {"id": "b", "text": "b", "type": "TONE"},
            {"id": "c", "text": "c"},
        ],
        rubric_types=["SAFETY"],
    )
    assert [r.id for r in scorer.rubrics_for(CASE)] == ["a", "c"]


def test_the_judge_sees_the_rubrics_that_apply() -> None:
    provider = ScriptedProvider(lambda m: verdicts(**{i: True for i in ids_in(m)}))
    scorer = judge(
        provider, rubrics=[{"id": "a", "text": "a", "type": "X"}, {"id": "b", "text": "b"}]
    )
    case = case_with([{"id": "own", "text": "its own"}])
    assert scorer.score(case, OUT).value == 1.0
    assert ids_in(provider.calls[0]) == ["a", "b", "own"]


# --- samples -------------------------------------------------------------------------------------


def test_samples_vote_by_majority_and_a_tie_fails() -> None:
    rubrics = [{"id": "a", "text": "a"}]
    three = ScriptedProvider([verdicts(a=True), verdicts(a="no"), verdicts(a=True)])
    assert judge(three, rubrics=rubrics, num_samples=3).score(CASE, OUT).value == 1.0
    two = ScriptedProvider([verdicts(a=True), verdicts(a="no")])
    tie = judge(two, rubrics=rubrics, num_samples=2).score(CASE, OUT)
    assert tie.value == 0.0 and tie.judge_tokens == 30
    assert tie.reason == "0/1 rubrics; failed: a (no)"


def test_each_sample_gets_its_own_seed() -> None:
    seeds: list[int | None] = []

    class Recording(ScriptedProvider):
        def chat(self, messages, params):  # type: ignore[no-untyped-def]
            seeds.append(params.seed)
            return super().chat(messages, params)

    scorer = RubricSetJudge(
        "rs",
        Recording([verdicts(a=True)]),
        GenerationParams(seed=10, temperature=0.7),
        rubrics=[{"id": "a", "text": "a"}],
        num_samples=3,
    )
    scorer.score(CASE, OUT)
    assert seeds == [10, 11, 12]


def test_an_undecided_sample_makes_the_whole_score_undecided() -> None:
    provider = ScriptedProvider([verdicts(a=True), "garbage", "garbage"])
    score = judge(provider, rubrics=[{"id": "a", "text": "a"}], num_samples=2).score(CASE, OUT)
    assert score.value is None and score.judge_tokens == 45  # the spent tokens are still counted


# --- per_rubric mode -----------------------------------------------------------------------------


def single_reply(messages: list[Message]) -> str:
    """The single-rubric judge's reply: pass when the rubric text asks for 'good'."""
    ok = "good" in messages[-1].content.split("Response to grade:")[0]
    return json.dumps({"reason": "ok" if ok else "bad", "pass": ok, "score": 1 if ok else 0})


def test_per_rubric_mode_makes_one_call_per_rubric() -> None:
    provider = ScriptedProvider(single_reply)
    scorer = judge(
        provider,
        mode="per_rubric",
        rubrics=[{"id": "a", "text": "be good"}, {"id": "b", "text": "be odd"}],
    )
    score = scorer.score(CASE, OUT)
    assert len(provider.calls) == 2 and score.judge_tokens == 30
    assert score.value == 0.5 and score.reason == "1/2 rubrics; failed: b (bad)"
    assert "be good" in provider.calls[0][-1].content and "be odd" in provider.calls[1][-1].content


def test_per_rubric_mode_is_undecided_when_one_rubric_is_undecided() -> None:
    provider = ScriptedProvider(["good", "nope", "nope"])  # first parses? no: all unparseable
    scorer = judge(provider, mode="per_rubric", rubrics=[{"id": "a", "text": "a"}])
    score = scorer.score(CASE, OUT)
    assert score.value is None and "rubric 'a'" in (score.reason or "")


# --- per-rubric metrics --------------------------------------------------------------------------


def test_per_rubric_metrics_add_one_metric_per_base_rubric() -> None:
    provider = ScriptedProvider([verdicts(a=True, b="rude")])
    scorer = judge(
        provider,
        rubrics=[{"id": "a", "text": "a"}, {"id": "b", "text": "b", "type": "TONE"}],
        per_rubric_metrics=True,
        pricing=Pricing(input_per_mtok=1_000_000, output_per_mtok=1_000_000),
    )
    scores = score_metrics(scorer, CASE, OUT)
    assert list(scores) == ["rs", "rs/a", "rs/b"]
    assert scores["rs/a"].value == 1.0 and scores["rs/a"].passed is True
    assert scores["rs/b"].value == 0.0 and scores["rs/b"].reason == "rude"
    # Only the main score carries the judge's cost, so a run's cost counts each call once.
    assert scores["rs"].judge_cost_usd == pytest.approx(15.0)
    assert scores["rs/a"].judge_cost_usd == scores["rs/b"].judge_cost_usd == 0.0
    assert scores["rs/a"].judge_tokens == scores["rs/b"].judge_tokens == 0


def test_a_per_rubric_metric_is_undecided_when_the_rubric_did_not_apply() -> None:
    scorer = judge(
        ScriptedProvider([verdicts(a=True)]),
        rubrics=[{"id": "a", "text": "a"}, {"id": "b", "text": "b", "type": "TONE"}],
        rubric_types=["SAFETY"],
        per_rubric_metrics=True,
    )
    scores = scorer.score_all(CASE, OUT)
    assert scores["rs/a"].passed is True
    assert scores["rs/b"].value is None and "not applied" in (scores["rs/b"].reason or "")


def test_a_per_rubric_metric_is_undecided_when_the_judge_was() -> None:
    scorer = judge(
        ScriptedProvider(["garbage"]), rubrics=[{"id": "a", "text": "a"}], per_rubric_metrics=True
    )
    scores = scorer.score_all(CASE, OUT)
    assert scores["rs"].value is None and scores["rs/a"].value is None
    assert scores["rs/a"].reason == scores["rs"].reason


def test_without_per_rubric_metrics_only_the_scorer_writes_a_metric() -> None:
    scorer = judge(ScriptedProvider([verdicts(a=True)]), rubrics=[{"id": "a", "text": "a"}])
    assert list(scorer.score_all(CASE, OUT)) == ["rs"]


class Stray:
    """A scorer whose ``score_all`` breaks the naming rule."""

    name = "s"

    def __init__(self, names: list[str]) -> None:
        self.names = names

    def score(self, case: Case, out: SystemOutput) -> Score:
        return Score(value=1.0, passed=True)

    def score_all(self, case: Case, out: SystemOutput) -> dict[str, Score]:
        return {n: Score(value=1.0, passed=True) for n in self.names}


def test_score_metrics_enforces_the_naming_rule() -> None:
    assert list(score_metrics(Stray(["s", "s/x"]), CASE, OUT)) == ["s", "s/x"]
    with pytest.raises(ValueError, match="own name"):
        score_metrics(Stray(["s/x"]), CASE, OUT)
    with pytest.raises(ValueError, match="only 's/<id>'"):
        score_metrics(Stray(["s", "other"]), CASE, OUT)
    with pytest.raises(ValueError, match="only 's/<id>'"):
        score_metrics(Stray(["s", "s2"]), CASE, OUT)  # a prefix is not the namespace


def test_a_scorer_without_score_all_writes_its_one_metric() -> None:
    class Plain:
        name = "p"

        def score(self, case: Case, out: SystemOutput) -> Score:
            return Score(value=1.0, passed=True)

    assert list(score_metrics(Plain(), CASE, OUT)) == ["p"]


# --- config: load-time validation ----------------------------------------------------------------


def suite_text(scorers: str, *, gate: str = "", judge_params: str = "") -> str:
    return (
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'ollama:m'}\n"
        f"judge: {{target: 'ollama:j'{judge_params}}}\n"
        f"scorers:\n{scorers}\n{gate}"
    )


def load(tmp_path: Path, text: str):
    (tmp_path / "cases.jsonl").write_text('{"id": "a", "input": "q"}\n', "utf-8")
    (tmp_path / "suite.yaml").write_text(text, "utf-8")
    return load_config(tmp_path / "suite.yaml")


@pytest.mark.parametrize(
    ("params", "message"),
    [
        ("{rubirc: x}", "unknown key 'rubirc' \\(did you mean 'rubric'"),
        ("{rubrics: [a], mod: batched}", "unknown key 'mod' \\(did you mean 'mode'"),
        ("{rubrics: [a], mode: sequential}", "mode must be one of batched, per_rubric"),
        ("{rubrics: [a], num_samples: 0}", "num_samples must be a whole number of at least 1"),
        ("{rubrics: [a], num_samples: 2.5}", "num_samples must be a whole number"),
        ("{rubrics: [a], min_score: 1.5}", "min_score must be a number between 0 and 1"),
        ("{rubrics: [a], case_rubrics: maybe}", "case_rubrics must be true or false"),
        ("{rubrics: [a], per_rubric_metrics: 1}", "per_rubric_metrics must be true or false"),
        ("{rubrics: [a], rubric_types: []}", "rubric_types must be a non-empty list"),
        ("{rubrics: [{text: a, wieght: 2}]}", "params.rubrics\\[0\\] has unknown key 'wieght'"),
        ("{mode: per_rubric}", "need rubric-set mode"),
        ("{num_samples: 3}", "need rubric-set mode"),
        ("{rubrics: [a], rubric: be nice}", "cannot be combined"),
        ("{case_rubrics: true, rubric_file: r.md}", "cannot be combined"),
        ("{case_rubrics: true, per_rubric_metrics: true}", "needs base 'rubrics'"),
    ],
)
def test_a_bad_judge_param_fails_when_the_suite_loads(
    tmp_path: Path, params: str, message: str
) -> None:
    text = suite_text(f"  - {{name: j, type: llm_judge, params: {params}}}")
    with pytest.raises(ConfigError, match=message):
        load(tmp_path, text)


def test_a_judge_without_params_and_a_single_rubric_still_load(tmp_path: Path) -> None:
    text = suite_text(
        "  - {name: j, type: llm_judge}\n  - {name: k, type: llm_judge, params: {rubric: nice}}"
    )
    assert [s.name for s in load(tmp_path, text).scorers] == ["j", "k"]


def test_the_scorer_config_validates_judge_params_directly() -> None:
    with pytest.raises(ValidationError, match="unknown key"):
        ScorerConfig(name="j", type="llm_judge", params={"x": 1})
    ScorerConfig(name="j", type="llm_judge", params={"rubrics": ["a"], "mode": "per_rubric"})


def test_validate_judge_params_hints_at_the_valid_keys_when_nothing_is_close() -> None:
    with pytest.raises(ValueError, match="valid keys: rubric, rubric_file, max_retries"):
        validate_judge_params({"zzzzzz": 1})


def test_gate_and_calibration_may_name_a_per_rubric_metric(tmp_path: Path) -> None:
    scorers = (
        "  - {name: rs, type: llm_judge, params: {per_rubric_metrics: true, rubrics: "
        "[{id: tone, text: polite}, {id: facts, text: right}]}}"
    )
    cfg = load(tmp_path, suite_text(scorers, gate="gate: {metric: rs/tone, min_cases: 1}"))
    assert cfg.gate is not None and cfg.gate.gated == ["rs/tone"]
    both = "gate: {metric: rs, metrics: [rs/facts], min_cases: 1}"
    assert load(tmp_path, suite_text(scorers, gate=both)).gate is not None
    assert [metric_names(s) for s in cfg.scorers] == [["rs", "rs/tone", "rs/facts"]]
    assert per_rubric_metric_names("rs", cfg.scorers[0].params) == ["rs/tone", "rs/facts"]


def test_gating_on_an_unknown_per_rubric_metric_fails(tmp_path: Path) -> None:
    scorers = (
        "  - {name: rs, type: llm_judge, params: {per_rubric_metrics: true, "
        "rubrics: [{id: tone, text: polite}]}}"
    )
    with pytest.raises(ConfigError, match="rs/nope"):
        load(tmp_path, suite_text(scorers, gate="gate: {metric: rs/nope, min_cases: 1}"))
    plain = "  - {name: rs, type: llm_judge, params: {rubrics: [{id: tone, text: polite}]}}"
    with pytest.raises(ConfigError, match="rs/tone"):  # no per_rubric_metrics, no such metric
        load(tmp_path, suite_text(plain, gate="gate: {metric: rs/tone, min_cases: 1}"))


def test_a_per_rubric_metric_may_not_shadow_another_scorer(tmp_path: Path) -> None:
    scorers = (
        "  - {name: rs, type: llm_judge, params: {per_rubric_metrics: true, "
        "rubrics: [{id: tone, text: polite}]}}\n  - {name: rs/tone, type: mrr}"
    )
    with pytest.raises(ConfigError, match="rs/tone"):
        load(tmp_path, suite_text(scorers))


def test_samples_at_temperature_zero_warn_at_load(tmp_path: Path) -> None:
    scorers = "  - {name: j, type: llm_judge, params: {rubrics: [a], num_samples: 3}}"
    with pytest.warns(UserWarning, match="temperature is 0"):
        load(tmp_path, suite_text(scorers))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load(tmp_path, suite_text(scorers, judge_params=", params: {temperature: 0.7}"))
        load(
            tmp_path,
            suite_text("  - {name: j, type: llm_judge, params: {rubrics: [a]}}"),
        )


# --- make_scorer ---------------------------------------------------------------------------------


def test_make_scorer_builds_a_rubric_set_judge(tmp_path: Path) -> None:
    scorers = (
        "  - {name: rs, type: llm_judge, params: {rubrics: [a], max_retries: 2, "
        "num_samples: 2, min_score: 0.5, mode: per_rubric}}"
    )
    cfg = load(tmp_path, suite_text(scorers, judge_params=", params: {temperature: 0.5}"))
    built = make_scorer(cfg.scorers[0], cfg, ScriptedProvider([]))
    assert isinstance(built, RubricSetJudge)
    assert (built.mode, built.num_samples, built.min_score, built.max_retries) == (
        "per_rubric",
        2,
        0.5,
        2,
    )
    with pytest.raises(ConfigError, match="needs a judge provider"):
        make_scorer(cfg.scorers[0], cfg, None)


def test_make_scorer_wraps_a_constructor_error_as_a_config_error(tmp_path: Path) -> None:
    cfg = load(tmp_path, suite_text("  - {name: rs, type: llm_judge, params: {rubrics: [a]}}"))
    broken = ScorerConfig.model_construct(
        name="rs", type="llm_judge", params={"rubrics": ["a"], "min_score": 7}
    )
    with pytest.raises(ConfigError, match="scorer 'rs'"):
        make_scorer(broken, cfg, ScriptedProvider([]))


# --- through a run -------------------------------------------------------------------------------


def fake_ollama(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/api/version":
        return httpx.Response(200, json={"version": "0.0.test"})
    if request.url.path == "/api/tags":
        return httpx.Response(200, json={"models": [{"name": "j", "digest": "abc123def4567"}]})
    prompt = json.loads(request.content)["messages"][-1]["content"]
    answer = prompt.split("Response to grade:\n", 1)[1].split("\n\n", 1)[0]
    rows = [
        {"id": rid, "pass": not (rid == "tone" and answer == "rude"), "reason": "not polite"}
        for rid in re.findall(r"(?m)^\[([^\]]+)\] ", prompt)
    ]
    return httpx.Response(
        200,
        json={
            "message": {"content": json.dumps({"rubrics": rows})},
            "prompt_eval_count": 40,
            "eval_count": 10,
        },
    )


def e2e_suite(tmp_path: Path, answer_a: str, answer_b: str) -> Path:
    rows = [
        {"id": "c1", "input": {"q": "1"}, "answers": {"v": answer_a}},
        {"id": "c2", "input": {"q": "2"}, "answers": {"v": answer_b}},
    ]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (tmp_path / "rubric_set_answer.py").write_text(
        "def ask(x):\n    return {'1': 'rude', '2': 'kind'}[x['q']]\n", "utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\n"
        "system: {target: 'python:rubric_set_answer:ask'}\n"
        "judge: {target: 'ollama:j'}\n"
        "scorers:\n  - {name: rs, type: llm_judge, params: {per_rubric_metrics: true, rubrics: "
        "[{id: tone, text: polite}, {id: facts, text: right}]}}\n"
        "gate: {metric: rs/tone, min_cases: 1}\n",
        "utf-8",
    )
    return suite


def test_per_rubric_metrics_reach_the_run_compare_and_the_cost(tmp_path: Path) -> None:
    cfg = load_config(e2e_suite(tmp_path, "a", "b"))
    run = run_suite(cfg, client=mock_client(fake_ollama), use_cache=False)
    first, second = run.results
    assert set(first.scores) == {"rs", "rs/tone", "rs/facts"}
    assert first.scores["rs/tone"].value == 0.0 and first.scores["rs/facts"].value == 1.0
    assert second.scores["rs"].value == 1.0
    assert first.scores["rs"].judge_tokens == 50 and first.scores["rs/tone"].judge_tokens == 0
    comparison = compare_runs(run, run, metrics=["rs/tone"])
    assert list(comparison.metrics) == ["rs/tone"]


def test_score_case_merges_every_metric_a_scorer_writes() -> None:
    class System:
        def run(self, case: Case, trial: int = 0) -> SystemOutput:
            return OUT

    result = score_case(CASE, System(), [Stray(["s", "s/x"])])  # type: ignore[arg-type]
    assert set(result.scores) == {"s", "s/x"}
