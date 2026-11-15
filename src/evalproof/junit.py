"""JUnit XML reports, so GitLab, Jenkins, Azure DevOps and other CI systems show results natively.

Two reports, both built as ``xml.etree.ElementTree`` elements by pure functions:

* ``run_testsuite``: one ``<testsuite>`` for a run, one ``<testcase>`` per case result (per trial
  when the run repeats its cases). It counts a case as failed exactly as the run report does.
* ``gate_testsuite``: one ``<testsuite name="evalproof gate">`` for a gate decision, one
  ``<testcase>`` per check the gate evaluated (``GateDecision.checks``).

``testsuites`` wraps suites in the ``<testsuites>`` root, and ``write_junit`` writes it. Every
text and attribute value is stripped of the characters XML 1.0 cannot hold.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from evalproof._version import __version__
from evalproof.compare import run_label
from evalproof.config import GateConfig
from evalproof.gate import GateDecision
from evalproof.models import CaseResult, Run, Score
from evalproof.report import case_problem

GATE_SUITE = "evalproof gate"  # the suite name and the classname of every gate testcase
OUTPUT_CHARS = 2000  # a case's output in <system-out>
MESSAGE_CHARS = 300  # the one-line message of a failed, errored or skipped case

# Everything XML 1.0 allows: tab, newline, carriage return and the Unicode ranges below. Control
# characters, lone surrogates and U+FFFE/U+FFFF make a document that parsers refuse.
_INVALID_XML = re.compile(r"[^\x09\x0a\x0d\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")


def clean(text: object) -> str:
    """``text`` as a string with the characters XML 1.0 cannot hold removed. Line endings become
    ``\\n``, which is what a parser would make of them."""
    value = str(text).replace("\r\n", "\n").replace("\r", "\n")
    return _INVALID_XML.sub("", value)


def _one_line(text: str, limit: int | None = None) -> str:
    text = " ".join(clean(text).split())
    return text if limit is None or len(text) <= limit else text[: limit - 1] + "…"


def _sub(parent: ET.Element, tag: str, text: str | None = None, **attrs: object) -> ET.Element:
    element = ET.SubElement(parent, tag, {k: clean(v) for k, v in attrs.items()})
    if text is not None:
        element.text = clean(text)
    return element


def _properties(suite: ET.Element, values: Mapping[str, object]) -> None:
    """A ``<properties>`` block of the non-empty values."""
    shown = {k: v for k, v in values.items() if v is not None and str(v) != ""}
    if shown:
        block = ET.SubElement(suite, "properties")
        for name, value in shown.items():
            _sub(block, "property", name=name, value=value)


def _seconds(value: float) -> str:
    return f"{value:.3f}"


def _suite(
    name: str,
    cases: list[ET.Element],
    properties: Mapping[str, object],
    *,
    timestamp: str | None = None,
    output: str | None = None,
) -> ET.Element:
    """A ``<testsuite>`` whose counts are read off its testcases, so they cannot disagree."""

    def count(tag: str) -> int:
        return sum(1 for case in cases if case.find(tag) is not None)

    attrs = {
        "name": name,
        "tests": len(cases),
        "failures": count("failure"),
        "errors": count("error"),
        "skipped": count("skipped"),
        "time": _seconds(sum(float(case.get("time", "0")) for case in cases)),
    }
    if timestamp:
        attrs["timestamp"] = timestamp
    suite = ET.Element("testsuite", {k: clean(v) for k, v in attrs.items()})
    _properties(suite, properties)
    suite.extend(cases)
    if output:
        _sub(suite, "system-out", output)
    return suite


def _score_text(score: Score) -> str:
    return "undecided" if score.value is None else f"{score.value:.3g}"


def _result_case(run: Run, result: CaseResult, classname: str) -> ET.Element:
    """One case result as a testcase: failed, errored and undecided exactly as the run report's
    failed-cases list counts them (``report.case_problem``)."""
    repeated = run.trials > 1
    name = f"{result.case_id} [trial {result.trial}]" if repeated else result.case_id
    case = ET.Element(
        "testcase",
        {
            "classname": clean(classname),
            "name": clean(name),
            "time": _seconds(result.output.latency_ms / 1000),
        },
    )
    problem = case_problem(result)
    if problem is not None:
        kind, names = problem
        if kind == "errored":
            text = result.output.error or ""
            _sub(case, "error", text, message=_one_line(text, MESSAGE_CHARS), type="system error")
        elif kind == "failed":
            shown = ", ".join(f"{n} ({_score_text(result.scores[n])})" for n in names)
            body = "\n".join(
                f"{n}: {result.scores[n].reason or 'score ' + _score_text(result.scores[n])}"
                for n in names
            )
            _sub(case, "failure", body, message=f"failed {shown}", type="scorer failed")
        else:
            body = "\n".join(f"{n}: {result.scores[n].reason or 'no verdict'}" for n in names)
            message = f"undecided on {', '.join(names)}: no scorer reached a verdict"
            _sub(case, "skipped", body, message=message)
    if result.output.output:
        shown_output = result.output.output
        if len(shown_output) > OUTPUT_CHARS:
            more = len(shown_output) - OUTPUT_CHARS
            shown_output = f"{shown_output[:OUTPUT_CHARS]}\n… ({more} more characters)"
        _sub(case, "system-out", shown_output)
    return case


def run_testsuite(run: Run) -> ET.Element:
    """One ``<testsuite>`` for a run, named ``<suite>.<variant or system>``. The testcases'
    ``classname`` is the same string. A trial of a repeated case is ``<case id> [trial N]``,
    numbered from 0 as in the run report."""
    info = run.info
    name = f"{info.suite}.{info.variant or info.system}"
    cases = [_result_case(run, r, name) for r in run.results]
    properties: dict[str, object] = {
        "evalproof.version": info.evalproof_version,
        "dataset.hash": info.dataset_hash,
        "suite.hash": info.suite_hash,
        "system": info.system,
        "variant": info.variant,
        "trials": info.trials,
        "seed": info.seed,
        "started_at": info.started_at,
        "finished_at": info.finished_at,
        "wall_time_s": _seconds(info.wall_time_s),
        "cost_usd": f"{sum(r.output.cost_usd for r in run.results):.4f}",
    }
    properties.update({f"environment.{k}": v for k, v in info.environment.items()})
    return _suite(name, cases, properties, timestamp=info.started_at)


def gate_testsuite(
    baseline: Run,
    candidates: Sequence[tuple[str, Run, GateDecision]],
    gate: GateConfig,
    exit_code: int,
) -> ET.Element:
    """One ``<testsuite name="evalproof gate">`` for a gate decision: a testcase per check the
    decision evaluated. A failed check is a ``<failure>`` with the reason the terminal report
    shows; a check the gate could not decide (exit 4) is an ``<error>``. ``candidates`` holds
    ``(label, run, decision)``; with several, each testcase name starts with its candidate's
    label. ``exit_code`` is the command's exit code."""
    several = len(candidates) > 1
    cases: list[ET.Element] = []
    notes: list[str] = []
    for label, _run, decision in candidates:
        for check in decision.checks:
            name = f"{label} / {check.name}" if several else check.name
            case = ET.Element(
                "testcase", {"classname": GATE_SUITE, "name": clean(name), "time": _seconds(0)}
            )
            message = _one_line(check.message)
            if check.outcome == "fail":
                _sub(case, "failure", check.message, message=message, type="gate check failed")
            elif check.outcome == "error":
                _sub(case, "error", check.message, message=message, type="cannot gate")
            else:
                _sub(case, "system-out", check.message)
            cases.append(case)
        notes.extend(
            f"warning: [{label}] {w}" if several else f"warning: {w}" for w in decision.warnings
        )
    properties: dict[str, object] = {
        "evalproof.version": __version__,
        "exit_code": exit_code,
        "baseline": run_label(baseline),
        "baseline.dataset_hash": baseline.info.dataset_hash,
        "gate.metrics": ", ".join(gate.gated),
        "gate.max_drop": gate.max_drop,
        "gate.alpha": gate.alpha,
        "gate.min_cases": gate.min_cases,
        "gate.aggregate": gate.aggregate,
        "gate.max_p95_latency_ms": gate.max_p95_latency_ms,
        "gate.max_cost_usd": gate.max_cost_usd,
        "gate.min_pass_rate": gate.min_pass_rate,
        "gate.must_pass": ", ".join(gate.must_pass),
        "gate.max_new_failures": gate.max_new_failures,
        "gate.allow_judge_change": "true" if gate.allow_judge_change else None,
    }
    for i, (label, run, _decision) in enumerate(candidates, start=1):
        key = f"candidate.{i}" if several else "candidate"
        properties[key] = label
        properties[f"{key}.dataset_hash"] = run.info.dataset_hash
    return _suite(GATE_SUITE, cases, properties, output="\n".join(notes) or None)


def testsuites(suites: Iterable[ET.Element]) -> ET.Element:
    """The ``<testsuites>`` root, with the totals of its suites."""
    found = list(suites)

    def total(attr: str) -> int:
        return sum(int(s.get(attr, "0")) for s in found)

    attrs = {
        "name": "evalproof",
        "tests": total("tests"),
        "failures": total("failures"),
        "errors": total("errors"),
        "skipped": total("skipped"),
        "time": _seconds(sum(float(s.get("time", "0")) for s in found)),
    }
    root = ET.Element("testsuites", {k: clean(v) for k, v in attrs.items()})
    root.extend(found)
    return root


def to_xml(root: ET.Element) -> str:
    """The document: an XML declaration, then ``root`` indented by two spaces."""
    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n"


def write_junit(path: str | Path, root: ET.Element) -> None:
    """Write ``root`` to ``path``: UTF-8 with LF line endings, replacing a file already there."""
    with Path(path).open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(to_xml(root))
