"""LLM-as-judge scorers.

A judge is a classifier, so it is treated like one: a strict JSON verdict, a parse
retry instead of a guess, ``undecided`` instead of a silent pass, and — in
``evalproof.calibration`` — measured error rates instead of trust.
"""

from __future__ import annotations

import difflib
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from evalproof.config import GenerationParams, Pricing, trial_params
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


# --- rubric sets ---------------------------------------------------------------------------------

JUDGE_PARAMS = ("rubric", "rubric_file", "max_retries")
RUBRIC_SET_PARAMS = (
    "rubrics",
    "case_rubrics",
    "rubric_types",
    "mode",
    "num_samples",
    "min_score",
    "per_rubric_metrics",
)
RUBRIC_MODES = ("batched", "per_rubric")
RUBRIC_ITEM_KEYS = ("id", "text", "weight", "type")
MAX_REASON = 500

RUBRIC_SET_TEMPLATE = """Rubrics:
{rubrics}

Question:
{question}

Reference answer:
{reference}

Response to grade:
{response}

Judge every rubric on its own, as a pass or a fail. Reply with exactly this JSON shape, with one
entry for every rubric id listed above:
{{"rubrics": [{{"id": "<rubric id>", "pass": true or false, "reason": "<one sentence>"}}]}}"""

RUBRIC_SET_RETRY_NUDGE = (
    "That was not a valid JSON object of the requested shape, or it left out a rubric id. Reply "
    'with only the JSON object: {"rubrics": [{"id": "...", "pass": true or false, '
    '"reason": "..."}]} with one entry for every rubric id.'
)


@dataclass(frozen=True)
class Rubric:
    """One thing a response must do. ``type`` is free text (an ADK import carries
    ``FINAL_RESPONSE_QUALITY`` and the like); ``weight`` counts in the fraction that passed."""

    id: str
    text: str
    weight: float = 1.0
    type: str | None = None


def normalize_rubrics(items: Any, path: str = "rubrics", prefix: str = "r") -> list[Rubric]:
    """Read a list of rubrics: strings (given the ids ``r1``, ``r2``, ... by position, or
    ``<prefix>1``, ...) or ``{id, text, weight?, type?}`` objects. Raises ``ValueError`` naming
    ``path``."""
    if not isinstance(items, list):
        raise ValueError(f"{path} must be a list")
    rubrics: list[Rubric] = []
    for position, item in enumerate(items, start=1):
        where = f"{path}[{position - 1}]"
        if isinstance(item, str):
            if not item.strip():
                raise ValueError(f"{where} is empty")
            rubrics.append(Rubric(id=f"{prefix}{position}", text=item))
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{where} must be a string or an object with 'text'")
        unknown = [k for k in item if k not in RUBRIC_ITEM_KEYS]
        if unknown:
            raise ValueError(
                f"{where} has unknown key {str(unknown[0])!r} "
                f"(valid keys: {', '.join(RUBRIC_ITEM_KEYS)})"
            )
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{where}.text must be a non-empty string")
        rid = item.get("id", f"{prefix}{position}")
        if not isinstance(rid, str) or not rid.strip():
            raise ValueError(f"{where}.id must be a non-empty string")
        weight = item.get("weight", 1.0)
        if (
            isinstance(weight, bool)
            or not isinstance(weight, int | float)
            or not math.isfinite(weight)
            or weight <= 0
        ):
            raise ValueError(f"{where}.weight must be a number greater than 0")
        rtype = item.get("type")
        if rtype is not None and not isinstance(rtype, str):
            raise ValueError(f"{where}.type must be a string")
        rubrics.append(Rubric(id=rid.strip(), text=text, weight=float(weight), type=rtype))
    ids = [r.id for r in rubrics]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{path} has repeated ids {dupes}")
    return rubrics


def rubric_set_mode(params: dict[str, Any]) -> bool:
    """True when an ``llm_judge`` scorer's params switch rubric-set mode on: base ``rubrics``, or
    ``case_rubrics: true`` to read them from the cases."""
    return bool(params.get("rubrics")) or params.get("case_rubrics") is True


def validate_judge_params(params: dict[str, Any], path: str = "params") -> None:
    """Check an ``llm_judge`` scorer's params without building it, so a typo fails when the suite
    loads. Raises ``ValueError``."""
    known = [*JUDGE_PARAMS, *RUBRIC_SET_PARAMS]
    for key in params:
        if key not in known:
            close = difflib.get_close_matches(str(key), known, n=1)
            hint = f"did you mean {close[0]!r}?" if close else f"valid keys: {', '.join(known)}"
            raise ValueError(f"{path} has unknown key {str(key)!r} ({hint})")
    present = [key for key in RUBRIC_SET_PARAMS if key in params]
    if "rubrics" in params:
        normalize_rubrics(params["rubrics"], f"{path}.rubrics")
    if "case_rubrics" in params and not isinstance(params["case_rubrics"], bool):
        raise ValueError(f"{path}.case_rubrics must be true or false")
    if "per_rubric_metrics" in params and not isinstance(params["per_rubric_metrics"], bool):
        raise ValueError(f"{path}.per_rubric_metrics must be true or false")
    types = params.get("rubric_types")
    if "rubric_types" in params and (
        not isinstance(types, list) or not types or not all(isinstance(t, str) for t in types)
    ):
        raise ValueError(f"{path}.rubric_types must be a non-empty list of strings")
    if "mode" in params and params["mode"] not in RUBRIC_MODES:
        raise ValueError(f"{path}.mode must be one of {', '.join(RUBRIC_MODES)}")
    n = params.get("num_samples", 1)
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"{path}.num_samples must be a whole number of at least 1")
    min_score = params.get("min_score", 1.0)
    if (
        isinstance(min_score, bool)
        or not isinstance(min_score, int | float)
        or not 0.0 <= min_score <= 1.0
    ):
        raise ValueError(f"{path}.min_score must be a number between 0 and 1")
    if present and not rubric_set_mode(params):
        raise ValueError(
            f"{path}: {', '.join(present)} need rubric-set mode: give 'rubrics', or "
            "'case_rubrics: true' to read them from the cases"
        )
    if rubric_set_mode(params) and ("rubric" in params or "rubric_file" in params):
        raise ValueError(
            f"{path}: 'rubric' and 'rubric_file' cannot be combined with 'rubrics' or "
            "'case_rubrics: true'; put the text in a rubric of the set"
        )
    if params.get("per_rubric_metrics") and not params.get("rubrics"):
        raise ValueError(f"{path}.per_rubric_metrics needs base 'rubrics' to name the metrics")


def per_rubric_metric_names(name: str, params: dict[str, Any]) -> list[str]:
    """The extra metrics ``<name>/<id>`` a rubric-set judge writes: one per base rubric, when
    ``per_rubric_metrics`` is on. Empty otherwise."""
    if not (rubric_set_mode(params) and params.get("per_rubric_metrics") is True):
        return []
    return [f"{name}/{r.id}" for r in normalize_rubrics(params.get("rubrics") or [])]


def _pass_flag(value: Any) -> bool:
    if isinstance(value, str):
        value = value.strip().lower() in {"true", "yes", "pass"}
    if not isinstance(value, bool):
        raise ValueError("'pass' must be a boolean")
    return value


def parse_rubric_verdicts(text: str, ids: list[str]) -> dict[str, tuple[bool, str]]:
    """Read a batched reply: ``{"rubrics": [{"id", "pass", "reason"}]}``. Every id in ``ids``
    must have a verdict; an id the judge invented is ignored. Raises ``ValueError``."""
    data = extract_json(text)
    entries = data.get("rubrics") if isinstance(data, dict) else None
    if not isinstance(entries, list):
        raise ValueError("verdict needs a 'rubrics' list")
    found: dict[str, tuple[bool, str]] = {}
    for entry in entries:
        if not isinstance(entry, dict) or "id" not in entry or "pass" not in entry:
            raise ValueError("every rubric verdict needs 'id' and 'pass'")
        rid = str(entry["id"]).strip()
        if rid not in ids and rid.startswith("[") and rid.endswith("]") and rid[1:-1] in ids:
            rid = rid[1:-1]  # the judge echoed the "[id]" of the prompt
        if rid in found:
            raise ValueError(f"rubric {rid!r} has two verdicts")
        found[rid] = (_pass_flag(entry["pass"]), str(entry.get("reason", ""))[:MAX_REASON])
    missing = [i for i in ids if i not in found]
    if missing:
        raise ValueError(f"no verdict for rubric(s) {missing}")
    return {i: found[i] for i in ids}


def _case_rubrics(case: Case) -> list[Rubric]:
    """A case's own rubrics. One without an id is numbered ``c1``, ``c2``, ... so it is added
    beside the base rubrics; only an explicit id equal to a base id replaces that rubric."""
    raw = case.field("rubrics")
    if raw is None:
        return []
    try:
        return normalize_rubrics(raw, "rubrics", prefix="c")
    except ValueError as exc:
        raise ValueError(f"case {case.id!r}: {exc}") from exc


@dataclass
class _Votes:
    """The verdicts one rubric got over the judge's samples."""

    passes: int = 0
    total: int = 0
    reasons: list[str] | None = None  # reasons of the failing samples

    def add(self, passed: bool, reason: str) -> None:
        self.total += 1
        self.passes += passed
        if not passed and reason:
            self.reasons = [*(self.reasons or []), reason]

    @property
    def passed(self) -> bool:
        """The majority of pass verdicts wins; a tie is a fail (as in ADK's rubric judge)."""
        return self.passes * 2 > self.total


class RubricSetJudge:
    """An ``llm_judge`` that grades a response against several rubrics at once.

    The score is the weighted fraction of rubrics that passed, and the case passes when that
    fraction is at least ``min_score``. ``mode`` ``batched`` asks the judge about every rubric in
    one call, ``per_rubric`` makes one call per rubric (with the single-rubric prompt). With
    ``num_samples`` above 1 the judge is asked that many times, each sample with its own seed
    offset, and a rubric passes by majority (a tie fails). Any call the judge cannot answer makes
    the whole score undecided: a case is never scored on a part of its rubrics.

    ``score_all`` also returns ``<name>/<id>`` for each base rubric when ``per_rubric_metrics`` is
    set, so the gate and the comparison can see which rubric moved."""

    def __init__(
        self,
        name: str,
        provider: Provider,
        params: GenerationParams,
        rubrics: list[Any] | None = None,
        case_rubrics: bool = True,
        rubric_types: list[str] | None = None,
        mode: str = "batched",
        num_samples: int = 1,
        min_score: float = 1.0,
        per_rubric_metrics: bool = False,
        max_retries: int = 1,
        pricing: Pricing | None = None,
    ) -> None:
        self.name = name
        self.provider = provider
        self.params = params
        self.pricing = pricing or Pricing()
        self.rubrics = normalize_rubrics(rubrics or [])
        self.case_rubrics = case_rubrics
        self.rubric_types = list(rubric_types) if rubric_types is not None else None
        if mode not in RUBRIC_MODES:
            raise ValueError(f"mode must be one of {', '.join(RUBRIC_MODES)}")
        self.mode = mode
        if num_samples < 1:
            raise ValueError("num_samples must be at least 1")
        self.num_samples = num_samples
        if not 0.0 <= min_score <= 1.0:
            raise ValueError("min_score must be between 0 and 1")
        self.min_score = min_score
        self.per_rubric_metrics = per_rubric_metrics
        self.max_retries = max_retries

    # -- which rubrics apply to a case

    def rubrics_for(self, case: Case) -> list[Rubric]:
        """The base rubrics, with the case's own merged in (a case rubric whose id equals a base
        id replaces it in place), then only the listed ``rubric_types``; untyped ones stay."""
        merged = {r.id: r for r in self.rubrics}
        if self.case_rubrics:
            merged.update({r.id: r for r in _case_rubrics(case)})
        rubrics = list(merged.values())
        if self.rubric_types is not None:
            rubrics = [r for r in rubrics if r.type is None or r.type in self.rubric_types]
        return rubrics

    # -- asking the judge

    def _batched_messages(
        self, case: Case, out: SystemOutput, rubrics: list[Rubric]
    ) -> list[Message]:
        prompt = RUBRIC_SET_TEMPLATE.format(
            rubrics="\n".join(f"[{r.id}] {r.text.strip()}" for r in rubrics),
            question=_question(case),
            reference=case.reference or "(none provided)",
            response=out.output or "(empty)",
        )
        return [Message(role="system", content=JUDGE_SYSTEM), Message(role="user", content=prompt)]

    def _ask_batched(
        self, case: Case, out: SystemOutput, rubrics: list[Rubric], params: GenerationParams
    ) -> tuple[dict[str, tuple[bool, str]] | str, int, float]:
        """One sample in batched mode: the verdicts, or the reason there are none, with the
        tokens and cost spent."""
        ids = [r.id for r in rubrics]
        messages = self._batched_messages(case, out, rubrics)
        tokens, cost, last_text = 0, 0.0, ""
        for _attempt in range(self.max_retries + 1):
            try:
                resp = self.provider.chat(messages, params)
            except ProviderError as exc:
                return f"judge error: {exc}", tokens, cost
            tokens += resp.tokens_in + resp.tokens_out
            cost += response_cost(resp, self.pricing)
            last_text = resp.text
            try:
                return parse_rubric_verdicts(resp.text, ids), tokens, cost
            except ValueError:
                messages = [
                    *messages,
                    Message(role="assistant", content=resp.text),
                    Message(role="user", content=RUBRIC_SET_RETRY_NUDGE),
                ]
        return f"unparseable judge reply: {last_text[:200]!r}", tokens, cost

    def _ask_per_rubric(
        self, case: Case, out: SystemOutput, rubric: Rubric, params: GenerationParams
    ) -> tuple[tuple[bool, str] | str, int, float]:
        single = LLMJudge(
            f"{self.name}/{rubric.id}",
            self.provider,
            params,
            rubric=rubric.text,
            max_retries=self.max_retries,
            pricing=self.pricing,
        )
        score = single.score(case, out)
        if score.passed is None:
            return score.reason or "undecided", score.judge_tokens, score.judge_cost_usd
        return (score.passed, score.reason or ""), score.judge_tokens, score.judge_cost_usd

    def _judge(
        self, case: Case, out: SystemOutput, rubrics: list[Rubric]
    ) -> tuple[Score, dict[str, _Votes]]:
        """Score a case against ``rubrics`` (not empty). The votes are empty when undecided."""
        votes = {r.id: _Votes() for r in rubrics}
        tokens, cost = 0, 0.0
        for sample in range(self.num_samples):
            params = trial_params(self.params, sample)
            if self.mode == "batched":
                verdicts, spent_tokens, spent = self._ask_batched(case, out, rubrics, params)
                tokens, cost = tokens + spent_tokens, cost + spent
                if isinstance(verdicts, str):
                    return Score(
                        value=None, reason=verdicts, judge_tokens=tokens, judge_cost_usd=cost
                    ), {}
                for rid, (passed, reason) in verdicts.items():
                    votes[rid].add(passed, reason)
            else:
                for rubric in rubrics:
                    verdict, spent_tokens, spent = self._ask_per_rubric(case, out, rubric, params)
                    tokens, cost = tokens + spent_tokens, cost + spent
                    if isinstance(verdict, str):
                        return (
                            Score(
                                value=None,
                                reason=f"rubric {rubric.id!r}: {verdict}"[:MAX_REASON],
                                judge_tokens=tokens,
                                judge_cost_usd=cost,
                            ),
                            {},
                        )
                    votes[rubric.id].add(*verdict)
        total = sum(r.weight for r in rubrics)
        value = sum(r.weight for r in rubrics if votes[r.id].passed) / total
        failed = [r for r in rubrics if not votes[r.id].passed]
        reason = f"{len(rubrics) - len(failed)}/{len(rubrics)} rubrics"
        if failed:
            parts = []
            for r in failed:
                why = (votes[r.id].reasons or [""])[0]
                parts.append(f"{r.id} ({why})" if why else r.id)
            reason += f"; failed: {', '.join(parts)}"
        score = Score(
            value=value,
            passed=value >= self.min_score - 1e-9,
            reason=reason[:MAX_REASON],
            judge_tokens=tokens,
            judge_cost_usd=cost,
        )
        return score, votes

    # -- the scorer interface

    def score_all(self, case: Case, out: SystemOutput) -> dict[str, Score]:
        """This scorer's own score under ``name``, and with ``per_rubric_metrics`` one metric
        ``<name>/<id>`` per base rubric (1.0 or 0.0; undecided when the rubric did not apply to
        the case or the judge could not decide). The extra metrics carry no judge cost, so the
        run's cost counts each call once."""
        if out.error:
            main = Score(value=0.0, passed=False, reason=f"system error: {out.error[:200]}")
            return self._extras(
                main, lambda _rubric: Score(value=0.0, passed=False, reason=main.reason)
            )
        rubrics = self.rubrics_for(case)
        if not rubrics:
            main = Score(value=None, reason="no rubrics")
            return self._extras(main, lambda _rubric: Score(value=None, reason="no rubrics"))
        main, votes = self._judge(case, out, rubrics)
        applied = {r.id for r in rubrics}

        def metric(rubric: Rubric) -> Score:
            if rubric.id not in applied:
                return Score(value=None, reason="rubric not applied to this case")
            vote = votes.get(rubric.id)
            if vote is None:  # the judge could not decide the case
                return Score(value=None, reason=main.reason)
            reason = None if vote.passed else next(iter(vote.reasons or []), None)
            return Score(value=1.0 if vote.passed else 0.0, passed=vote.passed, reason=reason)

        return self._extras(main, metric)

    def score(self, case: Case, out: SystemOutput) -> Score:
        return self.score_all(case, out)[self.name]

    def _extras(self, main: Score, metric: Callable[[Rubric], Score]) -> dict[str, Score]:
        scores = {self.name: main}
        if self.per_rubric_metrics:
            scores.update({f"{self.name}/{r.id}": metric(r) for r in self.rubrics})
        return scores


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
