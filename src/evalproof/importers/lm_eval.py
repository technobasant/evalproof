"""lm-evaluation-harness: ``samples_<task>_<time>.jsonl`` written by ``--log_samples``.

The file is one task for one model, and the harness puts it in a directory named after the
model, so the system is that directory (or ``--system``). One line is one document under one
filter; a case is a document, with the filters merged in as separate metrics.

* id: ``<task>:<doc_id>``, the harness's own stable document index;
* dataset hash input: ``doc_hash``, the harness's hash of the document;
* metrics are the numeric top-level fields named in each line's ``metrics`` list, called
  ``<metric>`` or, when the file has several filters, ``<metric>,<filter>`` as in the
  harness's results file;
* ``acc``, ``acc_norm`` and ``exact_match`` with 0/1 values are pass/fail per document, so
  they compare with McNemar; every other number is continuous.

The harness has no per-document error, and it has no repeats, so a document that appears
twice under one filter is an error rather than something to average.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from evalproof.importers._common import (
    ImportFormatError,
    Meta,
    Options,
    Sample,
    binary,
    build_run,
    continuous,
    finite,
    parse_jsonl,
    read_text,
    reject_table_options,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "lm-eval"
_NAME = re.compile(r"samples_(?P<task>.+)_(?P<date>\d{4}-\d{2}-\d{2}T[\d.\-]+)\.jsonl")
_BINARY_METRICS = {"acc", "acc_norm", "exact_match"}


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    task, finished = _from_filename(path)
    rows = parse_jsonl(read_text(path), path)
    if not rows:
        raise ImportFormatError(f"{path}: no samples")
    for i, row in enumerate(rows, start=1):
        if "doc_id" not in row or not isinstance(row.get("metrics"), list):
            raise ImportFormatError(
                f"{path}:{i}: not an lm-eval sample (needs 'doc_id' and a 'metrics' list); "
                "was the harness run with --log_samples?"
            )
    several = len({str(r.get("filter", "none")) for r in rows}) > 1
    skipped: set[str] = set()
    cases: dict[str, Sample] = {}
    seen: set[tuple[str, str]] = set()
    for row in rows:
        cid, flt = f"{task}:{row['doc_id']}", str(row.get("filter", "none"))
        if (cid, flt) in seen:
            raise ImportFormatError(
                f"{path}: document {row['doc_id']} appears twice under filter {flt!r}"
            )
        seen.add((cid, flt))
        scores = _scores(cid, row, flt if several else None, skipped, opts)
        if cid not in cases:
            output = SystemOutput(output=_response(row.get("filtered_resps")))
            cases[cid] = Sample(cid, row.get("doc_hash"), [CaseResult(case_id=cid, output=output)])
        cases[cid].repeats[0].scores.update(scores)
    meta = Meta(
        suite=task,
        system=opts.system or path.resolve().parent.name or path.stem,
        finished_at=finished,
    )
    return build_run(FORMAT, path, list(cases.values()), meta, opts)


def _from_filename(path: Path) -> tuple[str, str]:
    """The task name and finish time the harness wrote into the file name."""
    match = _NAME.fullmatch(path.name)
    if match is None:
        return path.stem, ""
    day, _, clock = match["date"].partition("T")
    return match["task"], f"{day}T{clock.replace('-', ':')}"


def _scores(
    cid: str, row: dict[str, Any], flt: str | None, skipped: set[str], opts: Options
) -> dict[str, Score]:
    scores: dict[str, Score] = {}
    for metric in row["metrics"]:
        value = row.get(metric)
        name = f"{metric},{flt}" if flt is not None else str(metric)
        number = finite(value)
        if isinstance(value, bool) or (number in (0.0, 1.0) and metric in _BINARY_METRICS):
            scores[name] = binary(bool(value))
        elif number is not None or isinstance(value, float):
            scores[name] = continuous(value, case_id=cid, metric=name)
        elif metric not in skipped:
            skipped.add(metric)
            opts.warn(
                f"metric {metric!r} is not a number (lm-eval wrote {type(value).__name__}); skipped"
            )
    return scores


def _response(filtered: Any) -> str:
    if isinstance(filtered, list) and len(filtered) == 1:
        return text_of(filtered[0])
    return text_of(filtered)
