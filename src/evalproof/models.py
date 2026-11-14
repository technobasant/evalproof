"""Data shared by every stage: a case, what the system said, how it scored, the run."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator

_CASE_INPUT: Any = object()  # "render the case's own input", as opposed to a later turn


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

    @property
    def turns(self) -> list[str | dict[str, Any]]:
        """The user turns that follow ``input`` (turn 0); empty for a single-turn case."""
        value = (self.model_extra or {}).get("turns")
        return list(value) if isinstance(value, list) else []

    @property
    def state(self) -> dict[str, Any]:
        """The initial session state the case hands to the system; empty when it sets none."""
        value = (self.model_extra or {}).get("state")
        return dict(value) if isinstance(value, dict) else {}

    @property
    def expect(self) -> list[dict[str, Any]]:
        """The case's inline checks (for the ``checks`` scorer); empty when it has none."""
        value = (self.model_extra or {}).get("expect")
        return [c for c in value if isinstance(c, dict)] if isinstance(value, list) else []

    def render_vars(self, turn: Any = _CASE_INPUT) -> dict[str, Any]:
        """Variables available to prompt templates.

        ``turn`` renders a later user turn instead of ``input``: a string is ``{input}``, a dict
        supplies its keys as fields, exactly as ``input`` does. The case's other fields (its id,
        reference and extras) are the same for every turn.
        """
        source = self.input if turn is _CASE_INPUT else turn
        out: dict[str, Any] = {"id": self.id, "reference": self.reference or ""}
        if isinstance(source, dict):
            out.update(source)
        out["input"] = source
        out.update(self.model_extra or {})
        return out


class ToolCall(BaseModel):
    """One tool call an agent made while answering."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    result: Any = None
    ok: bool = True
    error: str | None = None
    id: str | None = None
    turn: int = 0  # the 0-based user turn the call belongs to
    latency_ms: float | None = None


class ModelUsage(BaseModel):
    """Tokens and cost of the model calls of one role (the agent, a judge, a worker...)."""

    role: str = "agent"  # free text: agent, judge, worker, ...
    model: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    cached_tokens: int = 0
    cost_usd: float | None = None


class Trace(BaseModel):
    """What an agent did, not only what it said: its replies, tool calls and model usage."""

    responses: list[str] = Field(default_factory=list)  # the reply per user turn; last is output
    tool_calls: list[ToolCall] = Field(default_factory=list)
    rounds: int | None = None  # model calls or generation rounds, when known
    usage: list[ModelUsage] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)  # anything else (agent name, routing...)


class SystemOutput(BaseModel):
    """What the system under test returned for one case.

    ``trace`` is set by systems that report what they did (tool calls, replies per turn, model
    usage). Run files written before it existed have none.
    """

    output: str
    retrieved: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    cached: bool = False
    error: str | None = None
    trace: Trace | None = None


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
    # The same without the case's tags, which label a case and don't change what the system is
    # asked, so ``compare`` does not count a tag change as an edit. None in run files written by
    # evalproof 0.5.0, which pair on ``case_hash``.
    content_hash: str | None = None


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
    # The suite variant the run was made with (``evalproof run --variant``); None for a run of the
    # suite as written, and in run files written before evalproof 0.6.
    variant: str | None = None


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
