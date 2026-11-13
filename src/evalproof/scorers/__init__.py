"""Scorer registry: build scorers from suite config."""

from __future__ import annotations

from typing import Any, Protocol

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig
from evalproof.models import Case, Score, SystemOutput
from evalproof.providers import Provider
from evalproof.scorers.deterministic import (
    Contains,
    ExactMatch,
    JsonSchema,
    NumericTolerance,
    Regex,
)
from evalproof.scorers.judge import DEFAULT_RUBRIC, LLMJudge, PairwiseResult, pairwise
from evalproof.scorers.retrieval import MRR, NDCG, RecallAtK

__all__ = [
    "MRR",
    "NDCG",
    "Contains",
    "ExactMatch",
    "JsonSchema",
    "LLMJudge",
    "NumericTolerance",
    "PairwiseResult",
    "RecallAtK",
    "Regex",
    "Scorer",
    "make_scorer",
    "pairwise",
]


class Scorer(Protocol):
    name: str

    def score(self, case: Case, out: SystemOutput) -> Score: ...


_SIMPLE: dict[str, Any] = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "regex": Regex,
    "json_schema": JsonSchema,
    "numeric_tolerance": NumericTolerance,
    "recall_at_k": RecallAtK,
    "mrr": MRR,
    "ndcg": NDCG,
}


def make_scorer(
    cfg: ScorerConfig, suite: SuiteConfig, judge_provider: Provider | None = None
) -> Scorer:
    params = dict(cfg.params)
    if cfg.type == "llm_judge":
        if judge_provider is None or suite.judge is None:
            raise ConfigError(f"scorer {cfg.name!r} needs a judge provider")
        rubric = params.pop("rubric", None)
        rubric_file = params.pop("rubric_file", None)
        if rubric_file:
            rubric = suite.resolve(rubric_file).read_text(encoding="utf-8")
        max_retries = int(params.pop("max_retries", 1))
        if params:
            raise ConfigError(f"scorer {cfg.name!r}: unknown params {sorted(params)}")
        return LLMJudge(
            cfg.name,
            judge_provider,
            suite.judge.params,
            rubric=rubric or DEFAULT_RUBRIC,
            max_retries=max_retries,
            pricing=suite.judge.pricing,
        )
    try:
        scorer: Scorer = _SIMPLE[cfg.type](cfg.name, **params)
    except TypeError as exc:
        raise ConfigError(f"scorer {cfg.name!r}: {exc}") from exc
    return scorer
