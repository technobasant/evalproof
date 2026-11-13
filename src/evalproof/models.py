"""Data shared by every stage: a case, what the system said, how it scored, the run."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


class Case(BaseModel):
    """One golden-set item. Extra fields (answer keys, relevant ids) stay on the case."""

    model_config = {"extra": "allow"}

    id: str
    input: str | dict[str, Any]
    reference: str | None = None
    tags: list[str] = Field(default_factory=list)

    def field(self, name: str) -> Any:
        """Read a named field, including extras such as ``accept`` or ``relevant``."""
        if name in type(self).model_fields:
            return getattr(self, name)
        extra = self.model_extra or {}
        return extra.get(name)

    def render_vars(self) -> dict[str, Any]:
        """Variables available to prompt templates."""
        out: dict[str, Any] = {"id": self.id, "reference": self.reference or ""}
        if isinstance(self.input, dict):
            out.update(self.input)
            out["input"] = self.input
        else:
            out["input"] = self.input
        out.update(self.model_extra or {})
        return out


class SystemOutput(BaseModel):
    """What the system under test returned for one case."""

    output: str
    retrieved: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None


class Score(BaseModel):
    """One scorer's verdict on one case.

    ``passed`` is the binary outcome used by pass rates and McNemar;
    ``value`` is the continuous score used by bootstrap comparisons.
    Both are None when the scorer could not decide (e.g. an unparseable judge reply);
    undecided cases are counted and reported, never silently scored as passes.
    """

    value: float | None
    passed: bool | None = None
    reason: str | None = None
    judge_tokens: int = 0
    judge_cost_usd: float = 0.0

    @field_validator("value")
    @classmethod
    def _finite(cls, v: float | None) -> float | None:
        if v is not None and (math.isinf(v) or math.isnan(v)):
            raise ValueError("score value must be finite (use None for undecided)")
        return v

    @property
    def decided(self) -> bool:
        return self.value is not None


class CaseResult(BaseModel):
    case_id: str
    tags: list[str] = Field(default_factory=list)
    output: SystemOutput
    scores: dict[str, Score] = Field(default_factory=dict)
    # A short hash of the case as it was scored, so ``compare`` can tell a case that was edited
    # from one that was not. None in run files written before evalproof 0.5.
    case_hash: str | None = None


class RunInfo(BaseModel):
    suite: str
    system: str
    dataset_path: str
    dataset_hash: str
    started_at: str
    finished_at: str
    wall_time_s: float
    evalproof_version: str
    seed: int
    environment: dict[str, str] = Field(default_factory=dict)
    # Fingerprint of the system, judge, scorers and seed the run used (``suite_fingerprint``),
    # so ``evalproof verify`` can tell whether a committed run matches the committed suite.
    # None in run files written before evalproof 0.5 and in imported runs.
    suite_hash: str | None = None


class Run(BaseModel):
    """A complete, self-describing run artifact. Compare two of these to gate a change."""

    info: RunInfo
    results: list[CaseResult]

    def by_id(self) -> dict[str, CaseResult]:
        return {r.case_id: r for r in self.results}

    def scorer_names(self) -> list[str]:
        names: list[str] = []
        for r in self.results:
            for name in r.scores:
                if name not in names:
                    names.append(name)
        return names
