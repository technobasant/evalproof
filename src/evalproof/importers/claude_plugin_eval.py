"""Claude Code plugin evals: ``aggregate-result.json``, the result ``claude plugin eval`` writes.

Experimental: the docs show no sample file, so the shape below is the writer's schema (the zod
schema bundled in Claude Code 2.1.322, read from the installed binary) and the docs' description
of it, not a captured file. ``claude plugin eval`` arrived in Claude Code 2.1.269
(https://code.claude.com/docs/en/changelog). The document is versioned: "``aggregate-result.json``,
and ``--json`` output, is a versioned document with ``schemaVersion: 1`` ... new fields are added
without renaming existing ones, so write your script to ignore fields it doesn't recognize"
(https://code.claude.com/docs/en/plugin-evals, "Read the results", "JSON result"). Field names
are camelCase.

Top level, as written by 2.1.322:

* ``schemaVersion`` (the literal ``1``): any other value is refused. ``claudeVersion`` (the Claude
  Code that ran the suite) is the run's source tool version; ``startedAt`` (ISO time) its start;
  ``durationSeconds`` (wall-clock) its wall time.
* ``partial`` and ``partialReason`` (``cost_ceiling``, ``interrupted`` or ``auth_failed``): when
  ``partial`` is true the suite did not finish, and the import warns with the reason, since the
  cases it never reached are missing.
* ``cases[]``: ``name``, ``promptMarkdown`` and ``arms``. ``arms.with`` is an array of runs, one per
  ``runsPerCase``; ``arms.without`` is the same for the no-plugin baseline and is present only when
  the suite ran with ``--ablation with-without``. An arm is a plain array, not keyed by plugin.
  Every other key (``suite``, ``aggregates``, ``dir``, ``mocks``, ``advisories`` and the rest) is
  ignored.

What it makes of them:

* one system per arm, ``with`` and ``without``. ``--system`` picks one, and a file holding both
  needs it; the error lists the arms the file has, and ``--system without`` on a file with no
  baseline arm is an error. The run's system is the arm. A case with no runs in the chosen arm is
  skipped with a warning;
* one case per ``cases[].name`` (a repeated name takes ``#2``, ``#3``), whose input is
  ``promptMarkdown``; the runs of the arm, in list order, are its repeats, so ``--epochs trials``
  numbers the trials in that order;
* per run, ``costUsd`` is the cost, ``durationSeconds`` (when present, it is optional) the latency,
  and a non-null ``error`` (``timed out after 300s``, ``exit <code>: ...``) or an ``aborted``
  object (``{server, tool, reason}``: a mock stopped the run, and ``error`` stays null) the run's
  error, which scores as a failure on every metric. The docs say some errored runs are still graded
  on what they produced; the import does not use that grade. ``costUsd`` is the agent's cost plus
  the judge's (the docs: "including judge calls"), so the run's cost is ``costUsd`` less
  ``judgeCostUsd``, and the judge's share is kept on the score metric where ``compare`` adds it
  back: the total is the file's ``costUsd``;
* one metric called ``score``, valued with the run's ``score`` (the weighted fraction of its graders
  that passed, 0 to 1) and passed as its ``passed`` (``score >= 1``), carrying ``judgeCostUsd``;
* one metric per ``graders[].name`` of the run, valued 1.0 or 0.0 from the grader's ``passed``, with
  its ``explanation`` as the reason. A grader with ``scored: false`` (a ``withOnly`` indicator, such
  as "the Skill tool was called", is shown but not part of the score) has no value and no verdict;
  a grader without ``scored`` is scored unless it is ``withOnly``, as the writer defaults it. In the
  ``without`` arm a ``withOnly`` grader is left out. A grader named ``score`` is imported as
  ``grader:score`` so that it does not replace the run's own metric;
* when the cost ceiling skipped a run's judge graders (``skippedPaidGraders``) its score is not
  comparable with the others, and the import names how many runs that is in a warning;
* no reply text (the file holds none), no tokens and no trace: ``tracePath`` names a ``trace.jsonl``
  in the run's sandbox that this importer does not read, and ``turns`` is not imported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalproof.importers._common import (
    ImportFormatError,
    Meta,
    Options,
    Sample,
    as_dict,
    binary,
    build_run,
    finite,
    number_duplicates,
    pick_system,
    read_json,
    reject_table_options,
    score_value,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "claude-plugin-eval"
SCHEMA = "claude-plugin-eval/1"
ARMS = ("with", "without")


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    data = read_json(path)
    if not isinstance(data, dict):
        raise ImportFormatError(
            f"{path}: not a claude plugin eval result: expected a JSON object, the "
            "aggregate-result.json that `claude plugin eval` wrote"
        )
    version = data.get("schemaVersion")
    if version is None:
        raise ImportFormatError(
            f"{path}: not a claude plugin eval result (no schemaVersion); expected the "
            "aggregate-result.json that `claude plugin eval` wrote"
        )
    if isinstance(version, bool) or version != 1:
        raise ImportFormatError(
            f"{path}: schemaVersion is {version!r}; this importer reads schemaVersion 1"
        )
    cases = data.get("cases")
    if not isinstance(cases, list):
        raise ImportFormatError(f"{path}: not a claude plugin eval result (no cases list)")
    if data.get("partial") is True:
        reason = data.get("partialReason")
        opts.warn(
            f"{path}: the suite did not finish (partial, reason: "
            f"{reason if isinstance(reason, str) and reason else 'not given'}); the cases it "
            "never reached are missing from this run"
        )

    arm = pick_system([a for a in ARMS if any(_runs(c, a) for c in cases)], opts.system)
    names = []
    for position, raw in enumerate(cases):
        name = as_dict(raw).get("name")
        if not isinstance(name, str) or not name:
            raise ImportFormatError(f"{path}: cases[{position}] has no name")
        names.append(name)

    samples, skipped = [], []
    for case_id, raw in zip(number_duplicates(names), cases, strict=True):
        runs = _runs(raw, arm)
        if not runs:
            skipped.append(case_id)
            continue
        repeats = [_run(case_id, run, arm, index, path) for index, run in enumerate(runs)]
        samples.append(Sample(case_id, text_of(as_dict(raw).get("promptMarkdown")), repeats))
    if skipped:
        opts.warn(
            f"{path}: skipped {len(skipped)} case(s) with no runs in the {arm!r} arm: "
            + ", ".join(skipped)
        )
    unjudged = sum(
        1
        for case in cases
        for run in _runs(case, arm)
        if isinstance(run, dict) and run.get("skippedPaidGraders") is True
    )
    if unjudged:
        opts.warn(
            f"{path}: the cost ceiling skipped the judge graders of {unjudged} run(s) in the "
            f"{arm!r} arm; their scores are not comparable with the others"
        )

    claude_version = data.get("claudeVersion")
    started_at = data.get("startedAt")
    meta = Meta(
        system=arm,
        started_at=started_at if isinstance(started_at, str) else "",
        wall_time_s=finite(data.get("durationSeconds")) or 0.0,
        schema=SCHEMA,
        tool_version=claude_version if isinstance(claude_version, str) else "",
    )
    return build_run(FORMAT, path, samples, meta, opts)


def _runs(case: Any, arm: str) -> list[Any]:
    """The runs of ``arm`` in one case, or none when the case did not run that arm."""
    runs = as_dict(as_dict(case).get("arms")).get(arm)
    return runs if isinstance(runs, list) else []


def _run(case_id: str, raw: Any, arm: str, index: int, path: Path) -> CaseResult:
    """One run of one case: its ``score``, its graders and what the run cost."""
    if not isinstance(raw, dict):
        raise ImportFormatError(f"{path}: case {case_id!r}, arms.{arm}[{index}] is not an object")
    passed = raw.get("passed")
    scores = {
        "score": Score(
            value=score_value(raw.get("score"), case_id, "score"),
            passed=passed if isinstance(passed, bool) else None,
            judge_cost_usd=_dollars(raw.get("judgeCostUsd")),
        )
    }
    graders = raw.get("graders")
    for position, grader in enumerate(graders if isinstance(graders, list) else []):
        item = as_dict(grader)
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ImportFormatError(
                f"{path}: case {case_id!r}, arms.{arm}[{index}].graders[{position}] has no name"
            )
        with_only = item.get("withOnly") is True
        if with_only and arm == "without":
            continue
        metric = f"grader:{name}" if name == "score" else name
        if metric in scores:
            raise ImportFormatError(
                f"{path}: case {case_id!r}, arms.{arm}[{index}] has two graders named {name!r}"
            )
        scores[metric] = _grader(item, with_only)
    duration = finite(raw.get("durationSeconds"))
    output = SystemOutput(
        output="",
        latency_ms=duration * 1000 if duration is not None and duration >= 0 else 0.0,
        # costUsd holds the judge's share too, which the score metric carries (compare adds both).
        cost_usd=max(_dollars(raw.get("costUsd")) - _dollars(raw.get("judgeCostUsd")), 0.0),
        error=_error(raw),
    )
    return CaseResult(case_id=case_id, output=output, scores=scores)


def _grader(item: dict[str, Any], with_only: bool) -> Score:
    explanation = item.get("explanation")
    reason = explanation if isinstance(explanation, str) and explanation else None
    scored = item.get("scored")
    if not (scored if isinstance(scored, bool) else not with_only):
        return Score(value=None, reason=f"not scored: {reason}" if reason else "not scored")
    passed = item.get("passed")
    if not isinstance(passed, bool):
        return Score(value=None, reason=reason)
    return binary(passed, reason)


def _error(raw: dict[str, Any]) -> str | None:
    """Why the run ended abnormally: its ``error``, and the mock that aborted it, if any."""
    parts = []
    if raw.get("error") is not None:
        parts.append(text_of(raw["error"]) or "the run reported an error with no text")
    if raw.get("aborted"):
        detail = as_dict(raw["aborted"])
        named = ", ".join(f"{k} {detail[k]}" for k in ("server", "tool", "reason") if detail.get(k))
        parts.append(f"aborted by a mock: {named}" if named else "aborted by a mock")
    return "; ".join(parts) or None


def _dollars(x: Any) -> float:
    value = finite(x)
    return value if value is not None and value > 0 else 0.0
