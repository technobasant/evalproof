"""``evalproof readiness``: which parts of an AI app's quality a suite covers, and which it leaves
untested.

Static: it reads the suite, the dataset and, when given one, a run file. It makes no model call
and runs no scorer, so it is free and the same every time. Each dimension is ``covered``,
``partial``, ``missing`` or ``out of scope``, with the evidence for it. The report says what the
suite measures; it does not say the measurement is good enough.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig
from evalproof.models import Case, Run
from evalproof.scorers import DETERMINISTIC_TYPES
from evalproof.scorers.checks import TRACE_TYPES

COVERED = "covered"
PARTIAL = "partial"
MISSING = "missing"
OUT_OF_SCOPE = "out of scope"
STATUSES = (COVERED, PARTIAL, MISSING, OUT_OF_SCOPE)

EXIT_OK = 0
EXIT_NOT_READY = 1  # a --require'd dimension is not covered

# (name used by --require, label shown in the table), in table order.
DIMENSIONS: tuple[tuple[str, str], ...] = (
    ("correctness", "correctness"),
    ("behaviour", "behaviour (tool use)"),
    ("quality", "quality"),
    ("regression-gate", "regression gate"),
    ("reliability", "reliability"),
    ("cost-latency", "cost & latency"),
    ("safety", "safety"),
    ("robustness", "robustness"),
    ("multi-turn", "multi-turn"),
    ("load", "load"),
    ("production-monitoring", "production monitoring"),
)
DIMENSION_NAMES = tuple(name for name, _ in DIMENSIONS)

# The deterministic scorers that grade an answer against a key or a ranking. ``checks`` grades
# behaviour and has its own dimension.
CORRECTNESS_TYPES = tuple(sorted(DETERMINISTIC_TYPES - {"checks"}))

# The checks that read the trace (what the agent did): the ones a plain-text reply cannot satisfy.
BEHAVIOUR_CHECKS = TRACE_TYPES

TAGGED = {"safety": "safety", "robustness": "robustness"}


@dataclass(frozen=True)
class Dimension:
    name: str  # what --require takes
    label: str
    status: str
    evidence: str


@dataclass(frozen=True)
class RunEvidence:
    """What a run file adds: whether the cases carry traces, and how many errored."""

    path: str | None
    cases: int
    with_trace: int
    errors: int


@dataclass(frozen=True)
class Readiness:
    suite: str
    cases: int
    dimensions: tuple[Dimension, ...]
    run: RunEvidence | None = None

    def get(self, name: str) -> Dimension:
        return next(d for d in self.dimensions if d.name == name)

    def unmet(self, required: Sequence[str]) -> list[Dimension]:
        """The required dimensions that are not ``covered``, in table order."""
        wanted = set(required)
        return [d for d in self.dimensions if d.name in wanted and d.status != COVERED]


# --- reading the suite -------------------------------------------------------------------------


def _scorers(cfg: SuiteConfig, types: Iterable[str]) -> list[ScorerConfig]:
    wanted = set(types)
    return [s for s in cfg.scorers if s.type in wanted]


def _named(scorers: Iterable[ScorerConfig]) -> str:
    return ", ".join(f"{s.type} `{s.name}`" for s in scorers)


def _has_behaviour_check(check: Any) -> bool:
    """True when a check, or one inside an ``any_of`` or a ``not``, reads the trace."""
    if not isinstance(check, dict):
        return False
    kind = check.get("type")
    if kind in BEHAVIOUR_CHECKS:
        return True
    if kind == "any_of":
        children = check.get("checks")
        return isinstance(children, list) and any(_has_behaviour_check(c) for c in children)
    if kind == "not":
        return _has_behaviour_check(check.get("check"))
    return False


def _case_has_behaviour_check(scorer: ScorerConfig, case: Case) -> bool:
    """A case has one when the scorer applies one to every case (``params.checks``) or the case
    lists one in the field the scorer reads (``expect`` unless ``params.field`` says otherwise)."""
    suite_wide = scorer.params.get("checks")
    if isinstance(suite_wide, list) and any(_has_behaviour_check(c) for c in suite_wide):
        return True
    own = case.field(str(scorer.params.get("field", "expect")))
    return isinstance(own, list) and any(_has_behaviour_check(c) for c in own)


def _count(n: int, total: int) -> str:
    return f"{n} of {total}"


def _correctness(cfg: SuiteConfig) -> tuple[str, str]:
    found = _scorers(cfg, CORRECTNESS_TYPES)
    if found:
        return COVERED, _named(found)
    custom = [s for s in _scorers(cfg, ["python"]) if s.params.get("deterministic") is True]
    if custom:
        return PARTIAL, (
            f"{_named(custom)}: a deterministic function of your own, and evalproof cannot tell "
            "what it checks"
        )
    return MISSING, f"no scorer grades answers against a key ({', '.join(CORRECTNESS_TYPES)})"


def _behaviour(cfg: SuiteConfig, cases: list[Case]) -> tuple[str, str]:
    checkers = _scorers(cfg, ["checks"])
    if not checkers:
        return MISSING, "no checks scorer"
    covered = [c for c in cases if any(_case_has_behaviour_check(s, c) for s in checkers)]
    share = f"{_count(len(covered), len(cases))} cases have a check on the tool calls or rounds"
    names = ", ".join(f"`{s.name}`" for s in checkers)
    evidence = f"{share} (checks {names})"
    if cases and len(covered) * 2 >= len(cases):
        return COVERED, evidence
    return (PARTIAL if covered else MISSING), evidence


def _quality(cfg: SuiteConfig) -> tuple[str, str]:
    judges = _scorers(cfg, ["llm_judge"])
    if not judges:
        return MISSING, "no llm_judge scorer"
    calibration = cfg.calibration
    calibrated = [s for s in judges if calibration and s.name == calibration.judge_scorer]
    if calibrated:
        assert calibration is not None
        source = (
            f"labels {calibration.labels}"
            if calibration.labels
            else (f"gold scorer `{calibration.gold_scorer}`")
        )
        return COVERED, f"llm_judge `{calibrated[0].name}`, calibrated against {source}"
    if calibration:
        return PARTIAL, (
            f"{_named(judges)}: the calibration block names `{calibration.judge_scorer}`, "
            "which is not an llm_judge scorer"
        )
    return PARTIAL, (
        f"{_named(judges)}: no calibration block, so the judge's error rate is unmeasured "
        "(`evalproof calibrate` measures it against gold labels)"
    )


def _gate(cfg: SuiteConfig, cases: int) -> tuple[str, str]:
    gate = cfg.gate
    if gate is None:
        return MISSING, "no gate block"
    parts = [
        f"gate on {', '.join(f'`{m}`' for m in gate.gated)}",
        f"max_drop {gate.max_drop:g}",
        f"alpha {gate.alpha:g}",
        f"min_cases {gate.min_cases}",
    ]
    if gate.has_absolute_checks:
        parts.append("with checks that need no baseline")
    evidence = ", ".join(parts)
    if gate.min_cases > cases:
        return PARTIAL, (
            f"{evidence}; the dataset has {cases} cases, fewer than min_cases, so a comparison "
            "cannot be gated"
        )
    return COVERED, evidence


def _budgets(cfg: SuiteConfig) -> tuple[str, str]:
    gate = cfg.gate
    set_ = []
    if gate and gate.max_cost_usd is not None:
        set_.append(f"max_cost_usd {gate.max_cost_usd:g}")
    if gate and gate.max_p95_latency_ms is not None:
        set_.append(f"max_p95_latency_ms {gate.max_p95_latency_ms:g}")
    if set_:
        return COVERED, ", ".join(set_)
    return MISSING, "no max_cost_usd or max_p95_latency_ms in the gate"


def _tagged(cases: list[Case], tag: str) -> tuple[str, str]:
    n = sum(1 for c in cases if tag in {t.lower() for t in c.tags})
    if n:
        return COVERED, f"{_count(n, len(cases))} cases tagged `{tag}`"
    return MISSING, f"no case is tagged `{tag}`"


def _multi_turn(cases: list[Case]) -> tuple[str, str]:
    n = sum(1 for c in cases if c.turns)
    if n:
        return PARTIAL, (
            f"{_count(n, len(cases))} cases have turns; informational, since evalproof cannot "
            "tell how many multi-turn cases are enough"
        )
    return MISSING, "no case has turns"


def _run_evidence(run: Run, path: str | None) -> RunEvidence:
    return RunEvidence(
        path=path,
        cases=len(run.results),
        with_trace=sum(1 for r in run.results if r.output.trace is not None),
        errors=sum(1 for r in run.results if r.output.error),
    )


def assess(
    cfg: SuiteConfig, cases: list[Case], run: Run | None = None, run_path: str | None = None
) -> Readiness:
    """The readiness of a suite and its dataset (and, with ``run``, what that run file shows)."""
    evidence = _run_evidence(run, run_path) if run is not None else None
    behaviour = _behaviour(cfg, cases)
    if evidence is not None and behaviour[0] != MISSING:
        if evidence.with_trace:
            seen = f"run: {_count(evidence.with_trace, evidence.cases)} cases carry a trace"
        else:
            seen = "run: no case carries a trace, so a check on the tool calls fails there"
        behaviour = (behaviour[0], f"{behaviour[1]}; {seen}")
    rows: dict[str, tuple[str, str]] = {
        "correctness": _correctness(cfg),
        "behaviour": behaviour,
        "quality": _quality(cfg),
        "regression-gate": _gate(cfg, len(cases)),
        "reliability": (MISSING, "repeated trials are not in this release"),
        "cost-latency": _budgets(cfg),
        "safety": _tagged(cases, "safety"),
        "robustness": _tagged(cases, "robustness"),
        "multi-turn": _multi_turn(cases),
        "load": (OUT_OF_SCOPE, "use a load-testing tool such as k6 or Locust"),
        "production-monitoring": (
            OUT_OF_SCOPE,
            "use an observability platform; evalproof tests a fixed set of cases before release",
        ),
    }
    dimensions = tuple(Dimension(name, label, *rows[name]) for name, label in DIMENSIONS)
    return Readiness(suite=cfg.name, cases=len(cases), dimensions=dimensions, run=evidence)


# --- --require ---------------------------------------------------------------------------------


def parse_required(values: Sequence[str] | None) -> list[str]:
    """The dimension names of ``--require`` (comma-separated, and the flag may repeat).

    Raises ``ConfigError`` listing the valid names when one is not a dimension."""
    names: list[str] = []
    for value in values or []:
        for raw in value.split(","):
            name = raw.strip().lower().replace("_", "-").replace(" ", "-")
            if not name:
                continue
            if name not in DIMENSION_NAMES:
                raise ConfigError(
                    f"--require: unknown dimension {raw.strip()!r}; "
                    f"valid names: {', '.join(DIMENSION_NAMES)}"
                )
            if name not in names:
                names.append(name)
    return names


# --- output ------------------------------------------------------------------------------------


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _verdict(rep: Readiness, required: Sequence[str]) -> str:
    unmet = rep.unmet(required)
    if not unmet:
        return f"Required: {', '.join(required)}; all covered." if required else ""
    listed = ", ".join(f"{d.name} ({d.status})" for d in unmet)
    return f"**Not ready.** Required but not covered: {listed}."


def readiness_markdown(rep: Readiness, required: Sequence[str] = ()) -> str:
    lines = [
        f"## Readiness: {rep.suite}",
        "",
        f"{rep.cases} cases. A static reading of the suite and the dataset; no model was called.",
        "",
        "| Dimension | Status | Evidence |",
        "|---|---|---|",
    ]
    lines += [f"| {d.label} | {d.status} | {_cell(d.evidence)} |" for d in rep.dimensions]
    if rep.run is not None:
        where = f"`{rep.run.path}`: " if rep.run.path else ""
        lines += [
            "",
            f"Run file {where}{rep.run.cases} cases, {rep.run.with_trace} with a trace, "
            f"{rep.run.errors} errored.",
        ]
    verdict = _verdict(rep, required)
    if verdict:
        lines += ["", verdict]
    return "\n".join(lines) + "\n"


def readiness_json(rep: Readiness, required: Sequence[str] = ()) -> str:
    unmet = [d.name for d in rep.unmet(required)]
    payload: dict[str, Any] = {
        "suite": rep.suite,
        "cases": rep.cases,
        "dimensions": [
            {"name": d.name, "label": d.label, "status": d.status, "evidence": d.evidence}
            for d in rep.dimensions
        ],
        "required": list(required),
        "unmet": unmet,
        "ok": not unmet,
        "run": (
            None
            if rep.run is None
            else {
                "path": rep.run.path,
                "cases": rep.run.cases,
                "with_trace": rep.run.with_trace,
                "errors": rep.run.errors,
            }
        ),
    }
    return json.dumps(payload, indent=2) + "\n"


def exit_code(rep: Readiness, required: Sequence[str] = ()) -> int:
    return EXIT_NOT_READY if rep.unmet(required) else EXIT_OK
