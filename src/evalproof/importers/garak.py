"""garak: ``garak.<run>.report.jsonl``, the report a ``garak`` scan writes.

Field references are NVIDIA/garak v0.17.1 unless a version is given. A report is JSON lines, each
with an ``entry_type`` (``docs/source/reporting.rst:11-19``); this importer reads:

* ``start_run setup`` (``garak/command.py:79-108``): flat dotted keys, of which
  ``plugins.target_name`` names the system, ``plugins.target_type``, ``plugins.probe_spec`` and
  ``run.generations`` are recorded, and ``run.eval_threshold`` (default 0.5,
  ``garak/resources/garak.core.yaml:15``) is the threshold the scan was evaluated at. The real
  0.14 fixture has no ``run.eval_threshold``. The 0.9 report has ``config`` (``model_name``)
  instead of a setup row.
* ``init`` (``command.py:109-120``): ``garak_version``, ``start_time``. ``completion``
  (``command.py:131-139``): ``end_time``.
* ``attempt`` (``garak/attempt.py:268-301``): ``uuid``, ``status``, ``probe_classname``, ``prompt``,
  ``outputs``, ``detector_results``. An attempt is written twice with one uuid, status 1 with
  ``detector_results == {}`` and then status 2 with the scores (``probes/base.py:353-382``,
  ``harnesses/base.py:277-281``); garak's own readers keep status 2
  (``analyze/aggregate_reports.py:38-41``, ``analyze/qual_review.py:76``) and so does this one.
  An attempt that has no status-2 row is skipped, and one warning counts them.
* ``prompt``: from 0.13.0 a Conversation, ``{"turns": [{"role", "content": {"text", ...}}]}``
  (``attempt.py:21-51,90-99,119-130``), where the input is the text of the last ``user`` turn;
  before 0.13.0 (0.9 and 0.12, ``attempt.py:103-121`` at v0.12.0) a plain string.
* ``outputs`` (``attempt.py:318-333``): one Message dict (``text``) per generation, ``null`` when
  the generator returned nothing; before 0.13.0 a list of strings.
* ``detector_results`` (``harnesses/base.py:155-157``): ``{detector: [score or null per output]}``,
  scores in 0.0-1.0 where 1.0 is a hit (``detectors/base.py:74-78``); the 0.9 report has integers.
  The ``null`` of an output a detector could not score (``detectors/base.py:237-240``) stays an
  undecided score.

What it makes of them:

* one case per attempt; id ``<probe_classname>:<first 12 hex of sha256(input)>``, numbered ``#2``,
  ``#3`` when a probe sent the same prompt twice. Tags: ``safety``, ``garak``, ``probe:<name>``;
* the repeats of a case are its generations, in output order (``attempt.py:373-377`` expands the
  prompt to one conversation per output), so ``--epochs mean`` is the share of generations that
  passed and ``--epochs trials`` keeps each generation as a trial;
* one metric per detector, named as garak names it. An output passes when its score is below the
  threshold (``evaluators/base.py:533-543``: ``ThresholdEvaluator.test`` is ``score < threshold``),
  and the metric is 1.0 for a pass and 0.0 for a hit, with the raw score in ``reason``;
* the threshold is ``--threshold``, else the setup's ``run.eval_threshold``, else 0.5; the one
  used and where it came from are in ``environment["threshold"]`` and
  ``environment["threshold_source"]`` (``cli``, ``eval_threshold`` or ``default``).

The ``eval`` rows are not read: they are totals garak computed from the same scores
(``evaluators/base.py:213-245``: ``passed``, ``fails``, ``nones``, ``total_evaluated``, or ``total``
before 0.14.0), and they cannot give a per-case result. They are what the tests reconcile the
imported scores against. The trace is not imported: a report holds prompts and replies, not tool
calls.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
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
    parse_jsonl,
    read_text,
    reject_table_options,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "garak"
DEFAULT_THRESHOLD = 0.5
_ENTRY_TYPES = {"start_run setup", "config", "init", "attempt", "eval", "completion", "digest"}


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    rows = parse_jsonl(read_text(path), path)
    if not any(r.get("entry_type") in _ENTRY_TYPES for r in rows):
        raise ImportFormatError(
            f"{path}: not a garak report (no line has a known entry_type); "
            "expected the garak.<run>.report.jsonl a scan writes"
        )
    attempts = [r for r in rows if r.get("entry_type") == "attempt"]
    if not attempts:
        raise ImportFormatError(f"{path}: the report has no attempt entries")
    first: dict[Any, dict[str, Any]] = {}
    for row in rows:
        first.setdefault(row.get("entry_type"), row)
    setup = {**as_dict(first.get("config")), **as_dict(first.get("start_run setup"))}
    init, completion = as_dict(first.get("init")), as_dict(first.get("completion"))
    threshold, source = _threshold(opts, setup)

    done = [a for a in attempts if a.get("status") == 2]
    finished = {a["uuid"] for a in done if a.get("uuid") is not None}
    unscored = {a["uuid"] for a in attempts if a.get("uuid") not in (None, *finished)}
    unscored_count = len(unscored) + sum(
        1 for a in attempts if a.get("status") != 2 and a.get("uuid") is None
    )
    if not done:
        raise ImportFormatError(
            f"{path}: no attempt has status 2, which garak writes after the detectors ran; "
            "the scan did not finish or was not evaluated"
        )
    if unscored_count:
        opts.warn(
            f"{unscored_count} attempt(s) have no status-2 entry (not scored by the detectors) "
            "and were skipped"
        )

    prompts = [_input_text(a, path) for a in done]
    ids = number_duplicates(
        f"{a.get('probe_classname')}:{hashlib.sha256(text.encode()).hexdigest()[:12]}"
        for a, text in zip(done, prompts, strict=True)
    )
    samples: list[Sample] = []
    empty = 0
    for cid, attempt, text in zip(ids, done, prompts, strict=True):
        repeats = _repeats(cid, attempt, threshold, path)
        if not repeats:
            empty += 1
            continue
        probe = str(attempt.get("probe_classname"))
        samples.append(Sample(cid, text, repeats, tags=["safety", "garak", f"probe:{probe}"]))
    if empty:
        opts.warn(f"{empty} attempt(s) have no outputs and were skipped")
    if not any(r.scores for s in samples for r in s.repeats):
        raise ImportFormatError(
            f"{path}: no attempt has detector results, so there is nothing to score"
        )
    meta = Meta(
        suite=_text(setup.get("plugins.probe_spec")) or "garak",
        system=opts.system
        or _text(setup.get("plugins.target_name"))
        or _text(setup.get("model_name"))
        or "garak",
        started_at=_text(init.get("start_time")),
        finished_at=_text(completion.get("end_time")),
        wall_time_s=_seconds(init.get("start_time"), completion.get("end_time")),
        schema="garak-report",
        tool_version=_text(init.get("garak_version")) or _text(setup.get("_config.version")),
        environment={
            "threshold": f"{threshold:g}",
            "threshold_source": source,
            **{
                key: _text(setup.get(field))
                for key, field in (
                    ("target_type", "plugins.target_type"),
                    ("generations", "run.generations"),
                )
                if _text(setup.get(field))
            },
        },
    )
    return build_run(FORMAT, path, samples, meta, opts)


def _text(value: Any) -> str:
    return "" if value is None or isinstance(value, bool | dict | list) else str(value)


def _seconds(start: Any, end: Any) -> float:
    try:
        return max(
            0.0,
            (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds(),
        )
    except ValueError:
        return 0.0


def _threshold(opts: Options, setup: dict[str, Any]) -> tuple[float, str]:
    """The score at which an output counts as a hit, and where that number came from."""
    if opts.threshold is not None:
        value = finite(opts.threshold)
        if value is None:
            raise ImportFormatError(f"--threshold must be a finite number, got {opts.threshold!r}")
        return value, "cli"
    recorded = finite(setup.get("run.eval_threshold"))
    if recorded is not None:
        return recorded, "eval_threshold"
    return DEFAULT_THRESHOLD, "default"


def _message_text(content: Any) -> str:
    """The text of a Message dict (``text``), or of a string content (``Turn.from_dict``)."""
    return text_of(content.get("text")) if isinstance(content, dict) else text_of(content)


def _input_text(attempt: dict[str, Any], path: Path) -> str:
    prompt = attempt.get("prompt")
    if isinstance(prompt, str):
        return prompt
    turns = as_dict(prompt).get("turns")
    if isinstance(turns, list):
        for turn in reversed(turns):
            if isinstance(turn, dict) and turn.get("role") == "user":
                return _message_text(turn.get("content"))
    raise ImportFormatError(
        f"{path}: attempt {attempt.get('uuid')!r} (seq {attempt.get('seq')!r}) has no prompt "
        "text: expected a string or a conversation with a user turn"
    )


def _repeats(cid: str, attempt: dict[str, Any], threshold: float, path: Path) -> list[CaseResult]:
    outputs = attempt.get("outputs")
    if outputs is None:
        outputs = []
    if not isinstance(outputs, list):
        raise ImportFormatError(f"{path}: attempt {attempt.get('uuid')!r}: outputs is not a list")
    detectors = attempt.get("detector_results")
    if detectors is None:
        detectors = {}
    if not isinstance(detectors, dict):
        raise ImportFormatError(
            f"{path}: attempt {attempt.get('uuid')!r}: detector_results is not an object"
        )
    for name, values in detectors.items():
        if not isinstance(values, list) or len(values) != len(outputs):
            raise ImportFormatError(
                f"{path}: attempt {attempt.get('uuid')!r}: detector {name!r} must list one score "
                f"per output ({len(outputs)})"
            )
    results = []
    for i, output in enumerate(outputs):
        scores = {
            name: _score(values[i], threshold, name, path) for name, values in detectors.items()
        }
        results.append(
            CaseResult(
                case_id=cid,
                output=SystemOutput(output=_message_text(output)),
                scores=scores,
            )
        )
    return results


def _score(value: Any, threshold: float, detector: str, path: Path) -> Score:
    if value is None:
        return Score(value=None)
    number = finite(value)
    if number is None:
        raise ImportFormatError(
            f"{path}: detector {detector!r} gave a score that is not a number: {value!r}"
        )
    return binary(number < threshold, reason=f"detector score {number:g}")
