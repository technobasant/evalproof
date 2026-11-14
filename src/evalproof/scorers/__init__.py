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
    Facts,
    JsonSchema,
    NumericTolerance,
    Regex,
)
from evalproof.scorers.judge import (
    DEFAULT_RUBRIC,
    LLMJudge,
    PairwiseResult,
    RubricSetJudge,
    pairwise,
    per_rubric_metric_names,
    rubric_set_mode,
    validate_judge_params,
)
from evalproof.scorers.python_scorer import PythonScorer, validate_python_params
from evalproof.scorers.retrieval import MRR, NDCG, RecallAtK

__all__ = [
    "DETERMINISTIC_TYPES",
    "MRR",
    "NDCG",
    "ChecksScorer",
    "Contains",
    "ExactMatch",
    "Facts",
    "JsonSchema",
    "LLMJudge",
    "NumericTolerance",
    "PairwiseResult",
    "PythonScorer",
    "RecallAtK",
    "Regex",
    "RubricSetJudge",
    "Scorer",
    "is_deterministic",
    "make_scorer",
    "metric_names",
    "pairwise",
    "score_metrics",
    "validate_scorer_params",
]


class Scorer(Protocol):
    name: str

    def score(self, case: Case, out: SystemOutput) -> Score: ...


def score_metrics(scorer: Scorer, case: Case, out: SystemOutput) -> dict[str, Score]:
    """Every metric one scorer writes for a case.

    A scorer may define ``score_all(case, out) -> dict[str, Score]`` to write more than one
    (a rubric-set judge with ``per_rubric_metrics`` does); the dict replaces ``score``'s single
    result, so the scorer is run once. It must hold the scorer's own name, and every other name
    must be ``<scorer>/<id>`` so it cannot collide with another scorer's. Raises ``ValueError``
    when it does not."""
    score_all = getattr(scorer, "score_all", None)
    if score_all is None:
        return {scorer.name: scorer.score(case, out)}
    scores: dict[str, Score] = dict(score_all(case, out))
    if scorer.name not in scores:
        raise ValueError(f"scorer {scorer.name!r}: score_all must include the scorer's own name")
    stray = [
        name for name in scores if name != scorer.name and not name.startswith(f"{scorer.name}/")
    ]
    if stray:
        raise ValueError(
            f"scorer {scorer.name!r}: score_all may add only '{scorer.name}/<id>' metrics, "
            f"got {stray[0]!r}"
        )
    return scores


def metric_names(cfg: ScorerConfig) -> list[str]:
    """The metric names a scorer config can write: its own name, and ``<name>/<id>`` for each base
    rubric of a rubric-set judge with ``per_rubric_metrics``."""
    names = [cfg.name]
    if cfg.type == "llm_judge":
        names += per_rubric_metric_names(cfg.name, cfg.params)
    return names


_SIMPLE: dict[str, Any] = {
    "exact_match": ExactMatch,
    "contains": Contains,
    "regex": Regex,
    "json_schema": JsonSchema,
    "numeric_tolerance": NumericTolerance,
    "facts": Facts,
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
    """Check the params of a scorer type that has a fixed vocabulary (``checks``, ``python`` and
    ``llm_judge``) without building the scorer or importing anything. Raises ``ValueError`` naming
    the path."""
    if scorer_type == "checks":
        validate_checks_params(params)
    elif scorer_type == "python":
        validate_python_params(params)
    elif scorer_type == "llm_judge":
        validate_judge_params(params)


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
        if rubric_set_mode(params):
            try:
                validate_judge_params(params)
                set_params = {k: v for k, v in params.items() if k != "max_retries"}
                return RubricSetJudge(
                    cfg.name,
                    judge_provider,
                    suite.judge.params,
                    max_retries=int(params.get("max_retries", 1)),
                    pricing=suite.judge.pricing,
                    **set_params,
                )
            except ValueError as exc:
                raise ConfigError(f"scorer {cfg.name!r}: {exc}") from exc
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
