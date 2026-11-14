from __future__ import annotations

import pytest
from pydantic import ValidationError

from evalproof.config import ConfigError, GenerationParams, ScorerConfig, SuiteConfig, TargetConfig
from evalproof.models import Case, SystemOutput
from evalproof.providers import Message
from evalproof.scorers import (
    MRR,
    NDCG,
    Contains,
    ExactMatch,
    JsonSchema,
    LLMJudge,
    NumericTolerance,
    RecallAtK,
    Regex,
    make_scorer,
    pairwise,
)
from evalproof.scorers.deterministic import extract_json, normalize
from evalproof.scorers.judge import parse_verdict
from tests.conftest import ScriptedProvider


def out(text: str, retrieved: list[str] | None = None) -> SystemOutput:
    return SystemOutput(output=text, retrieved=retrieved or [])


def test_normalize() -> None:
    assert normalize("  Paris.  ") == "paris"
    assert normalize("Two\n  words!") == "two words"


def test_exact_match() -> None:
    s = ExactMatch("em")
    case = Case(id="c", input="q", reference="Paris")
    assert s.score(case, out("paris.")).passed is True
    assert s.score(case, out("Lyon")).passed is False
    assert s.score(Case(id="d", input="q"), out("x")).value is None
    assert ExactMatch("em", normalize=False).score(case, out("paris")).passed is False


def test_contains() -> None:
    case = Case(id="c", input="q", must=["30 days", "refund"])  # type: ignore[call-arg]
    s = Contains("c", all_of_field="must", any_of=["pro", "team"])
    assert s.score(case, out("Pro plan: refund within 30 days")).passed is True
    res = s.score(case, out("refund within 14 days on Pro"))
    assert res.passed is False and "30 days" in (res.reason or "")
    assert Contains("c", any_of=["x"]).score(case, out("y")).reason == "none of any_of found"
    assert Contains("c").score(case, out("y")).value is None
    assert (
        Contains("c", any_of=["Pro"], case_sensitive=True).score(case, out("pro")).passed is False
    )


def test_regex_accept_and_reject() -> None:
    case = Case(id="c", input="q", accept=[r"\b30 days\b"], reject=[r"\b14 days\b"])  # type: ignore[call-arg]
    s = Regex("ak", any_of_field="accept", none_of_field="reject")
    assert s.score(case, out("You have 30 days.")).passed is True
    assert s.score(case, out("30 days, or 14 days on Free")).passed is False
    assert s.score(case, out("a month")).reason == "no accept pattern matched"
    assert Regex("r").score(case, out("x")).value is None
    assert Regex("r", pattern="^a.b$", flags="ms").score(case, out("a\nb")).passed is True


def test_extract_json_variants() -> None:
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Sure!\n```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('<think>hmm {"no": 1</think>Verdict: {"a": 3} done') == {"a": 3}
    assert extract_json('noise {bad} then {"a": 4}') == {"a": 4}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_json_schema() -> None:
    schema = {
        "type": "object",
        "required": ["plan", "price", "tags"],
        "properties": {
            "plan": {"type": "string", "enum": ["free", "pro"]},
            "price": {"type": "number"},
            "seats": {"type": "integer"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "ok": {"type": "boolean"},
            "none": {"type": "null"},
        },
    }
    s = JsonSchema("js", schema=schema)
    case = Case(id="c", input="q")
    good = '{"plan": "pro", "price": 9.5, "seats": 3, "tags": ["a"], "ok": true, "none": null}'
    assert s.score(case, out(good)).passed is True
    bad = s.score(case, out('{"plan": "gold", "price": "9", "seats": 2.5, "tags": [1], "ok": 1}'))
    assert bad.passed is False and "not in enum" in (bad.reason or "")
    assert s.score(case, out('{"plan": "pro"}')).passed is False
    assert s.score(case, out("not json")).reason == "output is not JSON"
    assert s.score(case, out("[1]")).passed is False
    assert JsonSchema("js").score(case, out("{}")).value is None
    by_field = JsonSchema("js", schema_field="schema")
    assert by_field.score(Case(id="d", input="q", schema={"type": "array"}), out("[]")).passed  # type: ignore[call-arg]


def test_numeric_tolerance() -> None:
    case = Case(id="c", input="q", reference="1,200")
    assert NumericTolerance("n").score(case, out("It costs 1,200 per year")).passed is True
    assert NumericTolerance("n", rel_tol=0.01).score(case, out("about 1205")).passed is True
    assert NumericTolerance("n").score(case, out("12 then 1200")).passed is False
    assert NumericTolerance("n", pick="last").score(case, out("12 then 1200")).passed is True
    assert NumericTolerance("n", pick="any").score(case, out("12 then 1200")).passed is True
    assert NumericTolerance("n").score(case, out("none")).reason == "no number in output"
    assert NumericTolerance("n").score(Case(id="d", input="q"), out("1")).value is None
    with pytest.raises(ValueError):
        NumericTolerance("n", pick="median")


def test_retrieval_metrics_known_values() -> None:
    case = Case(id="c", input="q", relevant=["d1", "d3"])  # type: ignore[call-arg]
    ranked = out("", ["d2", "d1", "d4", "d3"])
    assert RecallAtK("r", k=2).score(case, ranked).value == pytest.approx(0.5)
    assert RecallAtK("r", k=4, pass_threshold=1.0).score(case, ranked).passed is True
    assert MRR("m").score(case, ranked).value == pytest.approx(0.5)
    assert MRR("m").score(case, out("", ["x"])).value == 0.0
    # DCG = 1/log2(3) + 1/log2(5) = .630930 + .430677; IDCG = 1 + .630930
    expected = (1 / 1.5849625 + 1 / 2.3219281) / (1 + 1 / 1.5849625)
    assert NDCG("n", k=4).score(case, ranked).value == pytest.approx(expected, rel=1e-6)
    graded = Case(id="g", input="q", relevant={"a": 2, "b": 1})  # type: ignore[call-arg]
    assert NDCG("n", k=2).score(graded, out("", ["a", "b"])).value == pytest.approx(1.0)
    assert NDCG("n").score(Case(id="e", input="q"), ranked).value is None
    with pytest.raises(ValueError):
        RecallAtK("r", k=0)


def test_parse_verdict() -> None:
    assert parse_verdict('{"reason": "ok", "pass": true, "score": 0.9}') == (True, 0.9, "ok")
    assert parse_verdict('{"pass": "yes"}') == (True, 1.0, "")
    assert parse_verdict('{"pass": false, "score": 7}')[1] == 1.0  # clipped
    for bad in ('{"score": 1}', '{"pass": 3}', '{"pass": true, "score": "high"}', "[1]"):
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_judge_retries_unparseable_reply_once() -> None:
    provider = ScriptedProvider(["I think it passes", '{"reason": "r", "pass": true, "score": 1}'])
    judge = LLMJudge("j", provider, GenerationParams())
    case = Case(id="c", input={"question": "q?"}, reference="ref")
    score = judge.score(case, out("answer"))
    assert score.passed is True and score.judge_tokens == 30
    assert len(provider.calls) == 2 and provider.calls[1][-1].role == "user"
    assert "Reference answer:\nref" in provider.calls[0][1].content


def test_judge_gives_up_as_undecided() -> None:
    judge = LLMJudge("j", ScriptedProvider(["nope", "still nope"]), GenerationParams())
    score = judge.score(Case(id="c", input="q"), out("a"))
    assert score.value is None and score.passed is None and "unparseable" in (score.reason or "")


def test_judge_scores_system_errors_as_fail() -> None:
    judge = LLMJudge("j", ScriptedProvider([]), GenerationParams())
    score = judge.score(Case(id="c", input="q"), SystemOutput(output="", error="timeout"))
    assert score.passed is False


def test_pairwise_consistent_and_position_biased() -> None:
    case = Case(id="c", input="q", reference="r")
    always_a = ScriptedProvider(['{"winner": "A"}'])  # prefers position one, whatever it holds
    biased = pairwise(always_a, GenerationParams(), case, "x", "y")
    assert biased.winner == "tie" and not biased.consistent

    def judge_y(messages: list[Message]) -> str:
        text = messages[-1].content
        return '{"winner": "A"}' if "Response A:\ny" in text else '{"winner": "B"}'

    fair = pairwise(ScriptedProvider(judge_y), GenerationParams(), case, "x", "y")
    assert fair.winner == "b" and fair.consistent
    once = pairwise(
        ScriptedProvider(['{"winner": "tie"}']), GenerationParams(), case, "x", "y", swap=False
    )
    assert once.winner == "tie"
    with pytest.raises(ValueError):
        pairwise(ScriptedProvider(['{"winner": "C"}']), GenerationParams(), case, "x", "y")


def test_make_scorer(tmp_path) -> None:
    (tmp_path / "rubric.md").write_text("custom rubric", "utf-8")
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="ollama:m"),
        judge=TargetConfig(target="ollama:j"),
        scorers=[ScorerConfig(name="j", type="llm_judge", params={"rubric_file": "rubric.md"})],
        base_dir=str(tmp_path),
    )
    judge = make_scorer(suite.scorers[0], suite, ScriptedProvider([]))
    assert isinstance(judge, LLMJudge) and judge.rubric == "custom rubric"
    with pytest.raises(ConfigError):
        make_scorer(suite.scorers[0], suite, None)
    with pytest.raises(ValidationError, match="unknown key"):  # a typo now fails at load
        ScorerConfig(name="j", type="llm_judge", params={"x": 1})
    with pytest.raises(ConfigError):  # and make_scorer still refuses one that got past the loader
        make_scorer(
            ScorerConfig.model_construct(name="j", type="llm_judge", params={"x": 1}),
            suite,
            ScriptedProvider([]),
        )
    with pytest.raises(ConfigError):
        make_scorer(ScorerConfig(name="e", type="exact_match", params={"bogus": 1}), suite)
    assert isinstance(make_scorer(ScorerConfig(name="e", type="exact_match"), suite), ExactMatch)
