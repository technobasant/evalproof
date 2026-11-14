"""Google ADK: ``<app>_<eval set>_<time>.evalset_result.json``, the result ``adk eval`` saves.

Experimental: no real result file was found in the repository, tests or docs (only the first 200
characters of one in issue #3869), so the shape below is the model and writer code, not a sample.
Field references are google/adk-python v2.14.1, ``src/google/adk/evaluation/``.

* The file (``<agents_dir>/<app>/.adk/eval_history/``, ``local_eval_set_results_manager.py:63-66``)
  is one ``EvalSetResult`` (``eval_result.py:85-96``): ``eval_set_id``, ``creation_timestamp`` (the
  ``time.time()`` of the run, ``_eval_set_results_manager_utils.py:31-47``) and
  ``eval_case_results``. Keys are snake_case on disk (``model_dump_json``, no ``by_alias``,
  ``local_eval_set_results_manager.py:68-69``) and camelCase in the dev server's API
  (``eval_result.py:34-37,88-91``); both are read.
* Two encodings (``_eval_set_results_manager_utils.py:50-69``): a JSON object, written since
  v1.22.0 (fix ``fc4e3d6f60``), and a JSON string holding that object's text, written before
  (issue #3869). A directory can hold both; the string is decoded a second time.
* ``EvalCaseResult`` (``eval_result.py:31-82``) is one run of one eval case: ``eval_id``,
  ``repetition_index`` (0-based among the runs of the case, ``None`` when not recorded,
  ``local_eval_service.py:238-270``), ``overall_eval_metric_results``,
  ``eval_metric_result_per_invocation``, ``session_details``.
* ``EvalMetricResult`` (``eval_metrics.py:518-533``): ``metric_name``, ``score`` (``None`` when not
  evaluated) and ``eval_status``, the int of ``EvalStatus`` (``eval_metrics.py:47-55``):
  1 PASSED, 2 FAILED, 3 NOT_EVALUATED, 4 INFORMATIONAL (a reported value that neither passes nor
  fails: the efficiency metrics). Any other value is refused.
* ``EvalMetricResultPerInvocation`` (``eval_metrics.py:536-555``) holds ``actual_invocation``, an
  ``Invocation`` (``eval_case.py:97-154``): ``user_content`` and ``final_response`` are genai
  ``Content`` (``parts[].text``, a ``thought`` part is not text of the reply), and
  ``intermediate_data`` is ``IntermediateData`` (``tool_uses`` and ``tool_responses``,
  ``eval_case.py:33-51``) or ``InvocationEvents`` (``invocation_events[].content.parts[]``
  ``function_call`` and ``function_response``, ``eval_case.py:54-91``; the form inference writes).
* ``session_details`` (``Session``, ``sessions/session.py:34-90``) holds ``events``.

What it makes of them:

* one case per ``eval_id``, whose repeats are the ``eval_case_results`` of that id in
  ``repetition_index`` order (file order for those with none), so ``--epochs trials`` numbers the
  trials as ADK numbered the runs;
* input: the text of ``user_content`` of the first invocation of the first repeat that has one;
  output: the text of the last ``final_response`` of the repeat's invocations;
* one metric per ``overall_eval_metric_results[].metric_name``, valued with ``score`` as given (no
  rescaling). 1 is passed and 2 is failed; 3 is undecided and has no value; 4 has the value and no
  verdict. A ``null`` score is undecided. The 1-5 ``response_evaluation_score``, the efficiency
  metrics and any score outside 0-1 are named in a warning, since ``mean`` and ``compare`` treat
  scores on the same scale;
* trace: the ``session_details.events`` through ``evalproof.adapters.adk.from_events`` when there
  are any; otherwise the ``intermediate_data`` of each invocation, ``tool_uses`` paired with
  ``tool_responses`` by id and ``invocation_events`` read as events, each call at the turn of
  its invocation index. A response dict with an ``error`` key makes the call ``ok=False``, as in
  ``from_events``. A repeat with no tool call and no model response has no trace;
* ``suite`` and, unless ``--system`` gives a label, the system are the ``eval_set_id``; there are
  no tags. Latency, tokens and the judge's ``rubric_scores`` are not imported.

An ADK eval set (a dataset, with ``eval_cases``) is not a result and is refused with a pointer.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evalproof.adapters._common import Ledger
from evalproof.adapters.adk import from_events
from evalproof.importers._common import (
    ImportFormatError,
    Meta,
    Options,
    Sample,
    as_dict,
    build_run,
    in_source_order,
    parse_json,
    read_text,
    reject_table_options,
    score_value,
)
from evalproof.importers._trace import build_trace
from evalproof.models import CaseResult, Run, Score, SystemOutput, Trace

FORMAT = "adk"

# EvalStatus (eval_metrics.py:47-55): the int on disk -> the verdict, and why there is none.
_PASSED = {1: True, 2: False, 3: None, 4: None}
_NO_VERDICT = {3: "not evaluated", 4: "informational: reported, not passed or failed"}

# Metrics ADK reports outside 0-1 (metric_info_providers.py: response_evaluation_score is 1-5, the
# efficiency metrics have no interval).
_NOT_UNIT = frozenset(
    {
        "response_evaluation_score",
        "tool_call_count_v1",
        "inference_call_count_v1",
        "token_usage_v1",
        "invocation_duration_v1",
    }
)


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    result = _load(path)
    raw_cases = _field(result, "eval_case_results")
    if not isinstance(raw_cases, list):
        if isinstance(_field(result, "eval_cases"), list):
            raise ImportFormatError(
                f"{path}: this is an ADK eval set (a dataset of cases, with eval_cases), not a "
                "result; import the *.evalset_result.json that `adk eval` saved under "
                ".adk/eval_history"
            )
        raise ImportFormatError(
            f"{path}: not an ADK eval result (no eval_case_results list); expected the "
            "*.evalset_result.json that `adk eval` saved under .adk/eval_history"
        )
    if not raw_cases:
        raise ImportFormatError(f"{path}: the result has no eval_case_results")

    groups: dict[str, list[tuple[Any, CaseResult]]] = {}
    inputs: dict[int, str] = {}
    for position, raw in enumerate(raw_cases):
        eval_id = _field(raw, "eval_id")
        if not isinstance(eval_id, str) or not eval_id:
            raise ImportFormatError(f"{path}: eval_case_results[{position}] has no eval_id")
        invocations = [
            as_dict(_field(entry, "actual_invocation"))
            for entry in _list(_field(raw, "eval_metric_result_per_invocation"))
        ]
        repeat = CaseResult(
            case_id=eval_id,
            output=SystemOutput(output=_final_text(invocations), trace=_trace(raw, invocations)),
            scores=_scores(eval_id, raw, path),
        )
        inputs[id(repeat)] = _input_text(invocations)
        groups.setdefault(eval_id, []).append((_field(raw, "repetition_index"), repeat))

    samples = []
    for eval_id, items in groups.items():
        repeats = in_source_order(items)
        text = next((inputs[id(r)] for r in repeats if inputs[id(r)]), "")
        samples.append(Sample(eval_id, text, repeats))
    if not any(r.scores for s in samples for r in s.repeats):
        raise ImportFormatError(
            f"{path}: no case result has overall_eval_metric_results, so there is nothing to score"
        )
    _note_scales(samples, opts)

    eval_set_id = _text(_field(result, "eval_set_id"))
    meta = Meta(
        suite=eval_set_id,
        system=opts.system or eval_set_id or FORMAT,
        started_at=_iso(_field(result, "creation_timestamp")),
        schema="adk-eval-set-result",
    )
    return build_run(FORMAT, path, samples, meta, opts)


def _load(path: Path) -> dict[str, Any]:
    """The ``EvalSetResult`` object, whichever of the two encodings the file uses."""
    data = parse_json(read_text(path), path)
    if isinstance(data, str):  # written before v1.22.0: the object's JSON text, as a string
        data = parse_json(data, path)
    if not isinstance(data, dict):
        raise ImportFormatError(
            f"{path}: not an ADK eval result: expected a JSON object (or a string holding one)"
        )
    return data


def _camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(word.capitalize() for word in rest)


def _field(obj: Any, name: str) -> Any:
    """``name`` of a JSON object as snake_case or camelCase (ADK reads both), else ``None``."""
    data = as_dict(obj)
    value = data.get(name)
    return value if value is not None else data.get(_camel(name))


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _text(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _iso(timestamp: Any) -> str:
    """A ``time.time()`` float as an ISO 8601 UTC string, or "" when it is not a usable time."""
    if (
        isinstance(timestamp, bool)
        or not isinstance(timestamp, int | float)
        or not math.isfinite(timestamp)
        or timestamp <= 0
    ):
        return ""
    try:
        return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return ""


def _content_text(content: Any) -> str:
    """The text of a genai ``Content``: its text parts, joined, without the model's thoughts."""
    return "".join(
        text
        for part in _list(_field(content, "parts"))
        if isinstance(text := _field(part, "text"), str) and not _field(part, "thought")
    )


def _input_text(invocations: list[dict[str, Any]]) -> str:
    return _content_text(_field(invocations[0], "user_content")) if invocations else ""


def _final_text(invocations: list[dict[str, Any]]) -> str:
    replies = [t for inv in invocations if (t := _content_text(_field(inv, "final_response")))]
    return replies[-1] if replies else ""


def _scores(eval_id: str, raw: Any, path: Path) -> dict[str, Score]:
    scores: dict[str, Score] = {}
    for metric in _list(_field(raw, "overall_eval_metric_results")):
        name = _field(metric, "metric_name")
        if not isinstance(name, str) or not name:
            raise ImportFormatError(
                f"{path}: case {eval_id!r}: an overall_eval_metric_results entry has no metric_name"
            )
        status = _field(metric, "eval_status")
        if isinstance(status, bool) or not isinstance(status, int) or status not in _PASSED:
            raise ImportFormatError(
                f"{path}: case {eval_id!r}, metric {name!r}: eval_status is {status!r}, not 1 "
                "(passed), 2 (failed), 3 (not evaluated) or 4 (informational)"
            )
        value = None if status == 3 else score_value(_field(metric, "score"), eval_id, name)
        # A null score decides nothing, whatever the status says: the pair must agree.
        passed = _PASSED[status] if value is not None else None
        scores[name] = Score(value=value, passed=passed, reason=_NO_VERDICT.get(status))
    return scores


def _note_scales(samples: list[Sample], opts: Options) -> None:
    """Say which metrics are not scored 0-1: they are kept as given and not rescaled."""
    odd = {
        name
        for s in samples
        for r in s.repeats
        for name, score in r.scores.items()
        if name in _NOT_UNIT or (score.value is not None and not 0.0 <= score.value <= 1.0)
    }
    if odd:
        opts.warn(
            f"metric(s) {', '.join(sorted(odd))} are not scored 0-1; the scores are kept as given "
            "and not rescaled, so compare them only with the same metric in another run"
        )


def _trace(raw: Any, invocations: list[dict[str, Any]]) -> Trace | None:
    events = _list(_field(_field(raw, "session_details"), "events"))
    if events:
        trace = from_events(events).get("trace")
        return trace if isinstance(trace, Trace) and (trace.tool_calls or trace.rounds) else None
    return _trace_from_invocations(invocations)


def _trace_from_invocations(invocations: list[dict[str, Any]]) -> Trace | None:
    ledger = Ledger()
    for turn, invocation in enumerate(invocations):
        data = _field(invocation, "intermediate_data")
        for use in _list(_field(data, "tool_uses")):
            _add_call(ledger, use, turn)
        for response in _list(_field(data, "tool_responses")):
            _add_result(ledger, response, turn)
        for event in _list(_field(data, "invocation_events")):
            for part in _list(_field(_field(event, "content"), "parts")):
                if (call := _field(part, "function_call")) is not None:
                    _add_call(ledger, call, turn)
                if (response := _field(part, "function_response")) is not None:
                    _add_result(ledger, response, turn)
    if not ledger.calls:
        return None
    replies = [_content_text(_field(inv, "final_response")) for inv in invocations]
    return build_trace(ledger.calls, responses=replies if any(replies) else ())


def _add_call(ledger: Ledger, call: Any, turn: int) -> None:
    ledger.call(_field(call, "name"), _field(call, "args"), _field(call, "id"), turn)


def _add_result(ledger: Ledger, response: Any, turn: int) -> None:
    """A tool response pairs with its call by id; ``{"error": ...}`` is a failed call, the way
    ``adapters.adk.from_events`` reads it (``tools/function_tool.py:385,528``)."""
    body = _field(response, "response")
    error = _field(body, "error")
    ledger.result(
        _field(response, "id"),
        _field(response, "name"),
        body,
        ok=not error,
        error=str(error) if error else None,
        turn=turn,
    )
