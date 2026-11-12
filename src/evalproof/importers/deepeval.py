"""DeepEval: the test-run JSON (``.deepeval/.latest_test_run.json`` or a bare test run).

DeepEval's metrics are LLM judges that return a score in [0, 1] and a pass/fail against a
threshold. Both are kept, because they answer different questions: ``<name>`` is the
threshold verdict (binary, so McNemar applies) and ``<name>:score`` the raw score
(continuous, so it can show a shift the threshold hides).

A metric that raised (``error`` set) is *undecided*, not failed: the judge broke, the
system under test did not. Conversational test cases have no single input and output to
pair on, so they are skipped with a warning.

Case ids are the test case's ``name`` when it is unique and not DeepEval's positional
default (``test_case_0``); otherwise ``sha256(input)[:16]``, numbered ``#2``, ``#3`` for
repeated inputs.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
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
    count_of,
    finite,
    number_duplicates,
    read_json,
    reject_table_options,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "deepeval"
_POSITIONAL = re.compile(r"test_case_\d+")


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    doc = read_json(path)
    data = doc.get("testRunData", doc) if isinstance(doc, dict) else None
    if not isinstance(data, dict) or not isinstance(data.get("testCases"), list):
        raise ImportFormatError(f"{path}: not a DeepEval test run (no 'testCases' list)")
    skipped = data.get("conversationalTestCases")
    if skipped:
        opts.warn(
            f"skipped {len(skipped)} conversational test case(s): evalproof pairs single turns"
        )
    cases = [c for c in data["testCases"] if isinstance(c, dict)]
    meta = Meta(
        suite=Path(str(data.get("testFile") or path.stem)).stem,
        system=opts.system or _system(data),
        wall_time_s=finite(data.get("runDuration")) or 0.0,
    )
    return build_run(FORMAT, path, _samples(cases), meta, opts)


def _system(data: dict[str, Any]) -> str:
    params = data.get("hyperparameters")
    model = params.get("model") if isinstance(params, dict) else None
    return str(data.get("identifier") or model or "deepeval")


def _ids(cases: list[dict[str, Any]]) -> list[str]:
    names = Counter(str(c.get("name") or "") for c in cases)
    hashes = [hashlib.sha256(str(c.get("input", "")).encode()).hexdigest()[:16] for c in cases]
    ids = []
    for case, h in zip(cases, hashes, strict=True):
        name = str(case.get("name") or "")
        usable = name and names[name] == 1 and not _POSITIONAL.fullmatch(name)
        ids.append(name if usable else h)
    # Only hash-derived ids can collide; numbering leaves the unique names alone.
    numbered = number_duplicates(ids)
    return numbered


def _samples(cases: list[dict[str, Any]]) -> list[Sample]:
    return [
        Sample(cid, str(case.get("input", "")), [_result(cid, case)])
        for cid, case in zip(_ids(cases), cases, strict=True)
    ]


def _result(cid: str, case: dict[str, Any]) -> CaseResult:
    seconds = finite(case.get("completionTime"))
    context = case.get("retrievalContext")
    output = SystemOutput(
        output=text_of(case.get("actualOutput")),
        retrieved=[text_of(c) for c in context] if isinstance(context, list) else [],
        latency_ms=seconds * 1000 if seconds is not None else 0.0,
        tokens_in=count_of(case.get("inputTokenCount")),
        tokens_out=count_of(case.get("outputTokenCount")),
        cost_usd=finite(case.get("tokenCost")) or 0.0,
    )
    scores: dict[str, Score] = {}
    metrics = [m for m in case.get("metricsData") or [] if isinstance(m, dict)]
    for name, metric in zip(
        number_duplicates(str(m.get("name") or "metric") for m in metrics), metrics, strict=True
    ):
        scores.update(_metric_scores(cid, name, metric))
    return CaseResult(
        case_id=cid, tags=[str(t) for t in case.get("tags") or []], output=output, scores=scores
    )


def _metric_scores(cid: str, name: str, metric: dict[str, Any]) -> dict[str, Score]:
    reason = metric.get("reason")
    reason = str(reason) if reason else None
    error = metric.get("error")
    if error:
        undecided = Score(value=None, reason=f"metric error: {str(error)[:200]}")
        return {name: undecided, f"{name}:score": undecided}
    success = metric.get("success")
    verdict = (
        binary(success, reason) if isinstance(success, bool) else Score(value=None, reason=reason)
    )
    verdict.judge_cost_usd = finite(metric.get("evaluationCost")) or 0.0
    raw = continuous(metric.get("score"), reason, case_id=cid, metric=f"{name}:score")
    return {name: verdict, f"{name}:score": raw}
