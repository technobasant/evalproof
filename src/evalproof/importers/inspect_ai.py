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

Epochs and trace. Each (id, epoch) is its own sample document; ``epoch`` is 1-based
(inspect_ai 0.3.289 ``log/_log.py:455``, ``samples/{id}_epoch_{n}.json`` in a ``.eval``), so the
repeats of a case are sorted by it and trial ``i`` is epoch ``i + 1``. The sample's ``events``
give the trace (inspect_ai 0.3.289 ``event/_tool.py`` and ``event/_model.py``):

* ``ToolEvent`` (``event == "tool"``, ``_tool.py:27``): ``id`` :33, ``function`` :36, ``arguments``
  :39, ``result`` :45, ``error`` :51 (a ``ToolCallError``, ``tool/_tool_call.py:93-94``: ``type``,
  ``message``), ``working_time`` :66 in seconds, ``failed`` :81. A call is ``ok`` when it has
  neither ``error`` nor ``failed``. The keys are absent, not null, when unset.
* ``ModelEvent`` (``event == "model"``, ``_model.py:88``): ``model`` :91 and ``output`` :120
  (``usage.input_tokens``, ``usage.output_tokens``, ``choices[0].message.content``). Each one adds a
  round and its tokens; the reply of the last one is the trace's response.
* Event strings longer than 100 characters are stored as ``attachment://<hash>`` and kept in the
  sample's ``attachments`` (``log/_condense.py:78,473-474,521-525``); they are resolved here. A
  reference that is not in ``attachments`` stays as it is, with one warning.
* Events inside a ``span_begin`` of type ``scorers`` or ``scorer``, or a span nested in one (by
  ``span_id`` and ``parent_id``), are the scorers' (a model-graded scorer calls a model there)
  and are skipped.
* A sample with ``events_data`` (``_log.py:589``) or ``input_refs`` (``_model.py:108``) and no
  ``events`` is the newer condensed log; there is no trace for it and one warning says so.

Every call has ``turn`` 0: a log does not say which user turn a call answered.
"""

from __future__ import annotations

import re
import zipfile
from collections.abc import Callable
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
    in_source_order,
    parse_json,
    read_json,
    reject_table_options,
    text_of,
)
from evalproof.importers._trace import add_usage, build_trace, tool_call
from evalproof.models import CaseResult, ModelUsage, Run, Score, SystemOutput, ToolCall, Trace

_ATTACHMENT = "attachment://"

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
    told: set[str] = set()

    def tell(key: str, message: str) -> None:
        if key not in told:
            told.add(key)
            opts.warn(message)

    inputs: dict[str, Any] = {}
    epochs: dict[str, list[tuple[Any, CaseResult]]] = {}
    for raw in log["samples"]:
        cid = str(raw["id"])
        result = _result(cid, raw, lambda name: _skip(name, skipped, opts), tell)
        inputs.setdefault(cid, _stable_input(raw.get("input")))
        epochs.setdefault(cid, []).append((raw.get("epoch"), result))
    samples = [Sample(cid, inputs[cid], in_source_order(items)) for cid, items in epochs.items()]
    return build_run(FORMAT, path, samples, meta, opts)


def _stable_input(value: Any) -> Any:
    """The sample input without per-run noise. Inspect gives every chat message a fresh
    random ``id``, so hashing it would make two runs of one dataset look different."""
    if isinstance(value, list):
        return [
            {k: v for k, v in m.items() if k != "id"} if isinstance(m, dict) else m for m in value
        ]
    return value


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


def _resolve(value: Any, attachments: dict[str, Any], tell: Callable[[str, str], None]) -> Any:
    """``value`` with every ``attachment://<hash>`` string replaced by the text it stands for."""
    if isinstance(value, str):
        if value.startswith(_ATTACHMENT):
            found = attachments.get(value[len(_ATTACHMENT) :])
            if isinstance(found, str):
                return found
            tell(
                "attachment",
                f"an attachment reference ({value[:40]}...) is not in the sample's attachments; "
                "it is kept as it is",
            )
        return value
    if isinstance(value, list):
        return [_resolve(v, attachments, tell) for v in value]
    if isinstance(value, dict):
        return {k: _resolve(v, attachments, tell) for k, v in value.items()}
    return value


def _scorer_spans(events: list[Any]) -> set[str]:
    """The ids of the ``scorers`` and ``scorer`` spans and of every span inside them. The model
    and tool events of a model-graded scorer sit there; they are the grader's, not the agent's."""
    parent: dict[str, Any] = {}
    roots: set[str] = set()
    for event in events:
        if isinstance(event, dict) and event.get("event") == "span_begin":
            sid = event.get("id")
            if isinstance(sid, str):
                parent[sid] = event.get("parent_id")
                if event.get("type") in ("scorers", "scorer"):
                    roots.add(sid)
    inside: set[str] = set()
    for sid in parent:
        seen: set[str] = set()
        cur: Any = sid
        while isinstance(cur, str) and cur not in seen:
            if cur in roots:
                inside.add(sid)
                break
            seen.add(cur)
            cur = parent.get(cur)
    return inside


def _trace(raw: dict[str, Any], tell: Callable[[str, str], None]) -> Trace | None:
    """The tool calls and model calls of a sample's ``events``; None when it has none to read."""
    events = raw.get("events")
    if not isinstance(events, list) or not events:
        if raw.get("events_data") is not None or "input_refs" in raw:
            tell(
                "condensed",
                "condensed event log: trace not read (a sample has events_data or input_refs "
                "and no events); the scores are imported without a trace",
            )
        return None
    attachments = as_dict(raw.get("attachments"))
    scoring = _scorer_spans(events)
    calls: list[ToolCall] = []
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    reply = ""
    for event in events:
        if not isinstance(event, dict) or event.get("span_id") in scoring:
            continue
        kind = event.get("event")
        if kind == "tool":
            seconds = finite(event.get("working_time"))
            calls.append(
                tool_call(
                    event.get("function"),
                    _resolve(event.get("arguments"), attachments, tell),
                    _resolve(event.get("result"), attachments, tell),
                    id=event.get("id"),
                    error=_resolve(event.get("error"), attachments, tell),
                    failed=event.get("failed") is True,
                    latency_ms=seconds * 1000 if seconds is not None else None,
                )
            )
        elif kind == "model":
            rounds += 1
            output = as_dict(_resolve(event.get("output"), attachments, tell))
            tokens = as_dict(output.get("usage"))
            model = event.get("model")
            add_usage(
                usage,
                str(model) if model else None,
                count_of(tokens.get("input_tokens")),
                count_of(tokens.get("output_tokens")),
            )
            reply = _completion(output) or reply
    if not calls and not rounds:
        return None
    return build_trace(
        calls, responses=[reply] if reply else [], rounds=rounds, usage=usage.values()
    )


def _result(
    cid: str, raw: dict[str, Any], skip: Any, tell: Callable[[str, str], None]
) -> CaseResult:
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
            trace=_trace(raw, tell),
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
