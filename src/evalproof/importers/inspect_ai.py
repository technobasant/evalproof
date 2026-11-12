"""Inspect AI: ``.json`` logs and the default ``.eval`` zip.

One log is one model on one task, so the system is ``eval.model`` and there is nothing to
choose. A sample is a case (id ``str(sample.id)``, which the task's dataset keeps stable);
each epoch is a repeat of it. Inspect scorers return their verdict in ``value``:

* ``"C"`` is a pass and ``"I"`` or ``"N"`` (no answer) a fail, all binary;
* ``"P"`` (partial) is 0.5 and undecided as pass/fail, so the metric compares as continuous;
* a bool is binary and a number is continuous;
* a dict becomes one score per key, named ``scorer.key``;
* a list, or a string that is none of ``C I N P``, has no meaning here and is skipped with
  a warning rather than guessed at.

A log whose status is not ``success`` (cancelled, errored) is still imported, but its
status is recorded in ``environment["source_status"]`` and a warning says so: an
incomplete log can hide a regression behind samples that never ran.
"""

from __future__ import annotations

import re
import zipfile
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
    continuous,
    count_of,
    finite,
    parse_json,
    read_json,
    reject_table_options,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "inspect"
_MEMBER = re.compile(r"samples/(.*)_epoch_(\d+)\.json", re.DOTALL)
_VERDICTS = {"C": True, "I": False, "N": False}
_ZSTD_HINT = (
    "this .eval log is zstd-compressed and needs Python 3.14+ to read; "
    "or run `inspect log convert --to json <file>` and import the .json"
)


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    log = _read_eval(path) if path.suffix.lower() == ".eval" else _read_json(path)
    spec = log["eval"]
    stats = as_dict(log.get("stats"))
    meta = Meta(
        suite=str(spec.get("task") or ""),
        system=opts.system or str(spec.get("model") or "unknown model"),
        started_at=str(stats.get("started_at") or ""),
        finished_at=str(stats.get("completed_at") or ""),
        wall_time_s=_seconds_between(stats.get("started_at"), stats.get("completed_at")),
        schema=str(log.get("version") or ""),
        tool_version=str((spec.get("packages") or {}).get("inspect_ai") or ""),
    )
    status = str(log.get("status") or "")
    if status and status != "success":
        meta.environment["source_status"] = status
        opts.warn(f"the log's status is {status!r}, not 'success': the run may be incomplete")
    skipped: set[str] = set()
    by_id: dict[str, Sample] = {}
    for raw in log["samples"]:
        cid = str(raw["id"])
        result = _result(cid, raw, lambda name: _skip(name, skipped, opts))
        sample = by_id.setdefault(cid, Sample(cid, raw.get("input"), []))
        sample.repeats.append(result)
    return build_run(FORMAT, path, list(by_id.values()), meta, opts)


def _read_json(path: Path) -> dict[str, Any]:
    return _check(read_json(path), path)


def _read_eval(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as zf:
            if "header.json" not in zf.namelist():
                raise ImportFormatError(f"{path}: not an Inspect .eval log (no header.json)")
            log = parse_json(zf.read("header.json").decode("utf-8"), f"{path}!header.json")
            members = sorted((m for m in zf.namelist() if _MEMBER.fullmatch(m)), key=_member_order)
            log = dict(_check(log, path, need_samples=False))
            log["samples"] = [
                parse_json(zf.read(m).decode("utf-8"), f"{path}!{m}") for m in members
            ]
    except zipfile.BadZipFile as exc:
        raise ImportFormatError(f"{path}: not a zip file; is it an Inspect .eval log?") from exc
    except NotImplementedError as exc:  # zipfile reads zstd (method 93) only on Python 3.14+
        raise ImportFormatError(_ZSTD_HINT) from exc
    return _check(log, path)


def _member_order(name: str) -> tuple[int, int | str, int]:
    match = _MEMBER.fullmatch(name)
    assert match is not None
    sid, epoch = match.group(1), int(match.group(2))
    return (0, int(sid), epoch) if sid.isdigit() else (1, sid, epoch)


def _check(log: Any, path: Path, *, need_samples: bool = True) -> dict[str, Any]:
    if not isinstance(log, dict) or not isinstance(log.get("eval"), dict):
        raise ImportFormatError(f"{path}: not an Inspect log (no 'eval' section)")
    if need_samples and not isinstance(log.get("samples"), list):
        raise ImportFormatError(
            f"{path}: this log has no samples; run the eval with --log-samples "
            "(the default) or import a log that kept them"
        )
    return log


def _seconds_between(start: Any, end: Any) -> float:
    try:
        return max(
            0.0,
            (datetime.fromisoformat(str(end)) - datetime.fromisoformat(str(start))).total_seconds(),
        )
    except ValueError:
        return 0.0


def _skip(name: str, skipped: set[str], opts: Options) -> None:
    if name not in skipped:
        skipped.add(name)
        opts.warn(
            f"score {name!r} has a value evalproof cannot read (a list or unknown text); skipped"
        )


def _result(cid: str, raw: dict[str, Any], skip: Any) -> CaseResult:
    error = raw.get("error")
    output = as_dict(raw.get("output"))
    usage = as_dict(output.get("usage"))
    seconds = finite(raw.get("working_time"))
    if seconds is None:
        seconds = finite(raw.get("total_time"))
    scores: dict[str, Score] = {}
    for name, entry in (raw.get("scores") or {}).items():
        entry = entry if isinstance(entry, dict) else {"value": entry}
        reason = entry.get("explanation")
        value = entry.get("value")
        parts = (
            {f"{name}.{k}": v for k, v in value.items()}
            if isinstance(value, dict)
            else {name: value}
        )
        for part, v in parts.items():
            score = _score(v, str(reason) if reason else None, cid, part)
            if score is None:
                skip(part)
            else:
                scores[part] = score
    return CaseResult(
        case_id=cid,
        output=SystemOutput(
            output=_completion(output),
            latency_ms=seconds * 1000 if seconds is not None else 0.0,
            tokens_in=count_of(usage.get("input_tokens")),
            tokens_out=count_of(usage.get("output_tokens")),
            error=str(error.get("message") if isinstance(error, dict) else error)
            if error
            else None,
        ),
        scores=scores,
    )


def _score(value: Any, reason: str | None, case_id: str, metric: str) -> Score | None:
    if isinstance(value, bool):
        return binary(value, reason)
    if isinstance(value, str):
        if value in _VERDICTS:
            return binary(_VERDICTS[value], reason)
        return Score(value=0.5, reason=reason) if value == "P" else None
    if isinstance(value, int | float):
        return continuous(value, reason, case_id=case_id, metric=metric)
    return None


def _completion(output: dict[str, Any]) -> str:
    choices = output.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, list):  # content blocks: keep the text ones
        return "".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
    return text_of(content)
