"""What a scan finds and reports: findings (what the code is), suites, and gaps (what is not
tested), with the JSON shape ``evalproof scan --json`` writes and ``evalproof draft`` reads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FORMAT_VERSION = 1

# Finding kinds, in the order the report lists them.
FINDING_KINDS = ("suite", "model_call", "agent", "tool", "prompt", "multi_turn", "test_cases")

# Gap kinds, in the order the report lists them.
GAP_KINDS = (
    "no_suite",
    "uncovered_code",
    "tool_unchecked",
    "no_multi_turn_cases",
    "test_cases_not_imported",
    "few_cases",
)

# The gaps a model may draft inputs for. The others are about a missing suite or about test
# cases that already exist, and need a person or an importer, not a model.
MODEL_GAP_KINDS = ("tool_unchecked", "no_multi_turn_cases", "uncovered_code", "few_cases")

# A suite's gate refuses to pass fewer cases than this unless the suite says otherwise; it is
# ``GateConfig.min_cases``' default.
DEFAULT_MIN_CASES = 20


@dataclass
class Finding:
    """One thing the scan found: a model call, an agent, a tool, a prompt, evidence of multi-turn
    use, a file of test cases or a suite. ``extra`` holds what only some kinds carry (a tool's
    ``name`` and ``signature``, a prompt's ``excerpt``, a test-case file's ``command``)."""

    kind: str
    path: str  # relative to the scanned directory, with "/" on every platform
    line: int
    framework: str | None
    detail: str
    language: str | None = "python"  # None for a file that is not code
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "framework": self.framework,
            "detail": self.detail,
            "language": self.language,
        }
        out.update(self.extra)
        return out


@dataclass
class Gap:
    """Something the code has that no suite tests, with the exact next command."""

    kind: str
    message: str
    suggest: str
    path: str | None = None
    line: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "kind": self.kind,
            "path": self.path,
            "line": self.line,
            "message": self.message,
            "suggest": self.suggest,
        }
        out.update(self.extra)
        return out


@dataclass
class SuiteInfo:
    """An evalproof suite the scan found (or was given), and what its target reaches."""

    path: str
    name: str
    dataset: str
    cases: int | None  # None when the dataset could not be read
    target: str
    judge: str | None
    min_cases: int
    reachability: str  # "known" for a python target found in the repo, else "unknown"
    reaches: list[str] = field(default_factory=list)
    reason: str | None = None  # why reachability is unknown

    def to_json(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "dataset": self.dataset,
            "cases": self.cases,
            "target": self.target,
            "judge": self.judge,
            "min_cases": self.min_cases,
            "reachability": self.reachability,
            "reaches": self.reaches,
            "reason": self.reason,
        }


@dataclass
class ScanReport:
    root: str
    files: int
    findings: list[Finding]
    suites: list[SuiteInfo]
    gaps: list[Gap]
    not_computed: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    unparsable: int = 0
    too_large: int = 0
    invalid_suites: int = 0

    def of_kind(self, kind: str) -> list[Finding]:
        return [f for f in self.findings if f.kind == kind]

    def to_json(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "root": self.root,
            "findings": [f.to_json() for f in self.findings],
            "suites": [s.to_json() for s in self.suites],
            "gaps": [g.to_json() for g in self.gaps],
            "not_computed": self.not_computed,
            "notes": self.notes,
            "skipped": {
                "unparsable": self.unparsable,
                "too_large": self.too_large,
                "invalid_suites": self.invalid_suites,
            },
        }
