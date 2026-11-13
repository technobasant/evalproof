"""LLM-as-judge scorers.

A judge is a classifier, so it is treated like one: a strict JSON verdict, a parse
retry instead of a guess, ``undecided`` instead of a silent pass, and — in
``evalproof.calibration`` — measured error rates instead of trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evalproof.config import GenerationParams, Pricing
from evalproof.models import Case, Score, SystemOutput
from evalproof.providers import Message, Provider, ProviderError, response_cost
from evalproof.scorers.deterministic import extract_json

DEFAULT_RUBRIC = (
    "Pass if the response answers the question correctly and is consistent with the "
    "reference answer. Fail if it is wrong, contradicts the reference, invents facts the "
    "reference does not support, or does not answer. Wording and length do not matter."
)

JUDGE_SYSTEM = (
    "You are a strict, consistent evaluator. You grade one response at a time against a "
    "rubric and a reference answer. Reply with a single JSON object and nothing else."
)

JUDGE_TEMPLATE = """Rubric:
{rubric}

Question:
{question}

Reference answer:
{reference}

Response to grade:
{response}

Reply with exactly this JSON shape:
{{"reason": "<one sentence>", "pass": true or false, "score": <number from 0 to 1>}}"""

RETRY_NUDGE = (
    "That was not a valid JSON object of the requested shape. Reply with only the JSON "
    'object: {"reason": "...", "pass": true or false, "score": 0 to 1}'
)


def _question(case: Case) -> str:
    if isinstance(case.input, dict):
        return str(case.input.get("question") or case.input)
    return case.input


def parse_verdict(text: str) -> tuple[bool, float, str]:
    data = extract_json(text)
    if not isinstance(data, dict) or "pass" not in data:
        raise ValueError("verdict needs a 'pass' field")
    passed = data["pass"]
    if isinstance(passed, str):
        passed = passed.strip().lower() in {"true", "yes", "pass"}
    if not isinstance(passed, bool):
        raise ValueError("'pass' must be a boolean")
    raw_score = data.get("score", 1.0 if passed else 0.0)
    try:
        score = min(1.0, max(0.0, float(raw_score)))
    except (TypeError, ValueError) as exc:
        raise ValueError("'score' must be a number") from exc
    return passed, score, str(data.get("reason", ""))[:500]


class LLMJudge:
    def __init__(
        self,
        name: str,
        provider: Provider,
        params: GenerationParams,
        rubric: str = DEFAULT_RUBRIC,
        max_retries: int = 1,
        pricing: Pricing | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.pricing = pricing or Pricing()
        self.params = params
        self.rubric = rubric
        self.max_retries = max_retries

    def messages(self, case: Case, out: SystemOutput) -> list[Message]:
        prompt = JUDGE_TEMPLATE.format(
            rubric=self.rubric.strip(),
            question=_question(case),
            reference=case.reference or "(none provided)",
            response=out.output or "(empty)",
        )
        return [Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=prompt)]

    def score(self, case: Case, out: SystemOutput) -> Score:
        if out.error:
            return Score(value=0.0, passed=False, reason=f"system error: {out.error[:200]}")
        messages = self.messages(case, out)
        tokens = 0
        cost = 0.0
        last_text = ""
        for _attempt in range(self.max_retries + 1):
            try:
                resp = self.provider.chat(messages, self.params)
            except ProviderError as exc:
                return Score(
                    value=None,
                    reason=f"judge error: {exc}",
                    judge_tokens=tokens,
                    judge_cost_usd=cost,
                )
            tokens += resp.tokens_in + resp.tokens_out
            cost += response_cost(resp, self.pricing)
            last_text = resp.text
            try:
                passed, value, reason = parse_verdict(resp.text)
            except ValueError:
                messages = [
                    *messages,
                    Message(role="assistant", content=resp.text),
                    Message(role="user", content=RETRY_NUDGE),
                ]
                continue
            return Score(
                value=value,
                passed=passed,
                reason=reason,
                judge_tokens=tokens,
                judge_cost_usd=cost,
            )
        return Score(
            value=None,
            reason=f"unparseable judge reply: {last_text[:200]!r}",
            judge_tokens=tokens,
            judge_cost_usd=cost,
        )


PAIRWISE_TEMPLATE = """Question:
{question}

Reference answer:
{reference}

Response A:
{a}

Response B:
{b}

Which response answers the question better, judged against the reference? Reply with
exactly this JSON shape: {{"reason": "<one sentence>", "winner": "A" or "B" or "tie"}}"""

Winner = Literal["a", "b", "tie"]


@dataclass(frozen=True)
class PairwiseResult:
    winner: Winner
    consistent: bool
    first: str  # verdict with (a, b) order, in a/b terms
    second: str  # verdict with (b, a) order, mapped back to a/b terms


def _pairwise_once(
    provider: Provider, params: GenerationParams, case: Case, first: str, second: str
) -> str:
    prompt = PAIRWISE_TEMPLATE.format(
        question=_question(case), reference=case.reference or "(none provided)", a=first, b=second
    )
    resp = provider.chat(
        [Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=prompt)],
        params,
    )
    data = extract_json(resp.text)
    winner = str(data.get("winner", "")).strip().lower() if isinstance(data, dict) else ""
    if winner not in {"a", "b", "tie"}:
        raise ValueError(f"bad pairwise verdict: {resp.text[:120]!r}")
    return winner


def pairwise(
    provider: Provider,
    params: GenerationParams,
    case: Case,
    a: str,
    b: str,
    swap: bool = True,
) -> PairwiseResult:
    """Compare two outputs. With ``swap`` the judge sees both orders; a verdict that flips
    with position is reported as a tie and flagged inconsistent (position bias)."""
    first = _pairwise_once(provider, params, case, a, b)
    if not swap:
        return PairwiseResult(winner=first, consistent=True, first=first, second=first)  # type: ignore[arg-type]
    raw = _pairwise_once(provider, params, case, b, a)
    second = {"a": "b", "b": "a", "tie": "tie"}[raw]
    consistent = first == second
    winner: Winner = first if consistent else "tie"  # type: ignore[assignment]
    return PairwiseResult(winner=winner, consistent=consistent, first=first, second=second)
