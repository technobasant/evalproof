"""Scorer registry: build scorers from suite config."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig
from evalproof.models import Case, Score, SystemOutput
from evalproof.providers import Provider
from evalproof.scorers.checks import ChecksScorer, validate_checks_params
from evalproof.scorers.deterministic import (
    Contains,
    ExactMatch,
    JsonSchema,
    NumericTolerance,
    Regex,
)
from evalproof.scorers.judge import DEFAULT_RUBRIC, LLMJudge, PairwiseResult, pairwise
from evalproof.scorers.python_scorer import PythonScorer, validate_python_params
from evalproof.scorers.retrieval import MRR, NDCG, RecallAtK

__all__ = [
    "DETERMINISTIC_TYPES",
    "MRR",
    "NDCG",
    "ChecksScorer",
    "Contains",
    "ExactMatch",
    "JsonSchema",
    "LLMJudge",
    "NumericTolerance",
    "PairwiseResult",
    "PythonScorer",
    "RecallAtK",
    "Regex",
    "Scorer",
    "is_deterministic",
    "make_scorer",
    "pairwise",
    "validate_scorer_params",
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

# The scorer types that need no model call: the same output and case always score the same, so
# ``evalproof verify`` can re-score a stored output and compare.
DETERMINISTIC_TYPES = frozenset([*_SIMPLE, "checks"])


def is_deterministic(cfg: ScorerConfig) -> bool:
    """True when the scorer runs without a model, so its scores can be reproduced offline.

    A ``python`` scorer is not deterministic by type: only an instance that sets
    ``deterministic: true`` is, because evalproof cannot know what the function does."""
    if cfg.type == "python":
        return cfg.params.get("deterministic") is True
    return cfg.type in DETERMINISTIC_TYPES


def validate_scorer_params(scorer_type: str, params: dict[str, Any]) -> None:
    """Check the params of a scorer type that has a fixed vocabulary (``checks`` and ``python``)
    without building the scorer or importing anything. Raises ``ValueError`` naming the path."""
    if scorer_type == "checks":
        validate_checks_params(params)
    elif scorer_type == "python":
        validate_python_params(params)


def make_scorer(
    cfg: ScorerConfig, suite: SuiteConfig, judge_provider: Provider | None = None
) -> Scorer:
    params = dict(cfg.params)
    # Python scorers and checks import code from the suite's directory, as a python system does.
    search_path = str(Path(suite.base_dir).resolve())
    if cfg.type in ("checks", "python"):
        try:
            validate_scorer_params(cfg.type, params)
            if cfg.type == "checks":
                return ChecksScorer(cfg.name, search_path=search_path, **params)
            return PythonScorer(cfg.name, search_path=search_path, **params)
        except ConfigError:
            raise
        except ValueError as exc:
            raise ConfigError(f"scorer {cfg.name!r}: {exc}") from exc
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
