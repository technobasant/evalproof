"""promptfoo: the JSON written by ``promptfoo eval -o out.json`` and the JSONL variant.

promptfoo evaluates every test against every provider (and prompt) in one file, so a file
may hold several systems; ``--system`` picks one. Case ids come from what the test asked,
never from its position, so the same tests keep their ids when a file is re-ordered or a
test is inserted:

    id = sha256(canonical_json({"vars": ..., "description": ...}))[:16]

Two tests with identical vars and description get ``#2``, ``#3`` in testIdx order. Rows that
promptfoo marks as repeats of one test (``repeatGroupId``, or an increasing ``repeatIndex``)
are one case with several repeats, collapsed by ``--epochs``.

Scores: ``pass`` (the test's verdict), one continuous score per ``namedScores`` key, and a
binary ``assert:<metric or type>`` per assertion. A system error (``failureReason == 2``) is
an error row; a failed assertion (``failureReason == 1``) is an ordinary failure.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from evalproof.dataset import canonical_json
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
    parse_jsonl,
    pick_system,
    read_text,
    reject_table_options,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

FORMAT = "promptfoo"
SYSTEM_ERROR = 2  # promptfoo's ResultFailureReason.ERROR


def read(path: Path, opts: Options) -> Run:
    reject_table_options(opts, FORMAT)
    rows, meta = _load(path)
    labels = _labels(rows)
    system = pick_system(list(dict.fromkeys(labels)), opts.system)
    mine = [r for r, label in zip(rows, labels, strict=True) if label == system]
    meta.system = system
    return build_run(FORMAT, path, _samples(mine), meta, opts)


def _load(path: Path) -> tuple[list[dict[str, Any]], Meta]:
    text = read_text(path)
    meta = Meta()
    if path.suffix.lower() == ".jsonl":
        return _check_rows(parse_jsonl(text, path), path), meta
    doc = parse_json(text, path)
    outer: dict[str, Any] = doc if isinstance(doc, dict) else {}
    inner = outer.get("results")
    if isinstance(inner, dict):  # {evalId, results: {version, results: [...]}, config, metadata}
        rows, summary = inner.get("results"), inner
    elif isinstance(inner, list):  # the bare summary: {version, results: [...], ...}
        rows, summary = inner, outer
    else:
        rows, summary = doc, {}
    if not isinstance(rows, list):
        raise ImportFormatError(f"{path}: not a promptfoo output file (no results list)")
    info = as_dict(outer.get("metadata"))
    config = as_dict(outer.get("config"))
    meta.suite = str(config.get("description") or "")
    meta.started_at = str(summary.get("timestamp") or info.get("evaluationCreatedAt") or "")
    meta.schema = str(summary.get("version") or "")
    meta.tool_version = str(info.get("promptfooVersion") or "")
    return _check_rows(rows, path), meta


def _check_rows(rows: list[Any], path: Path) -> list[dict[str, Any]]:
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or not isinstance(row.get("success"), bool):
            raise ImportFormatError(
                f"{path}: result {i} has no boolean 'success'; "
                "is this a promptfoo output file (promptfoo eval -o out.json)?"
            )
    return rows


def _labels(rows: list[dict[str, Any]]) -> list[str]:
    """The system each row belongs to: its provider, plus the prompt if the file has several."""
    prompts = {_prompt_label(r) for r in rows}
    labels = []
    for row in rows:
        provider = row.get("provider")
        if isinstance(provider, dict):
            label = str(provider.get("label") or provider.get("id") or "unknown provider")
        else:
            label = str(provider or "unknown provider")
        labels.append(f"{label} · {_prompt_label(row)}" if len(prompts) > 1 else label)
    return labels


def _prompt_label(row: dict[str, Any]) -> str:
    """The prompt template. ``raw`` is the prompt rendered for one test, so it differs per row."""
    prompt = row.get("prompt")
    if not isinstance(prompt, dict):
        return str(prompt or "")[:60]
    return str(prompt.get("label") or prompt.get("raw") or "")[:60]


def _test_key(row: dict[str, Any]) -> tuple[str, Any]:
    case = as_dict(row.get("testCase"))
    vars_ = row.get("vars") if row.get("vars") is not None else case.get("vars")
    key = {"vars": vars_ or {}, "description": case.get("description")}
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()[:16], key


def _occurrences(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows into tests: repeats of one test together, identical tests apart.

    ``repeatGroupId`` names a test's repeats outright. Without it, a row joins the open
    group of the same test only if its ``repeatIndex`` goes up; a missing or non-increasing
    index starts a new test (which is how two identical tests stay two cases).
    """
    groups: list[list[dict[str, Any]]] = []
    by_gid: dict[str, list[dict[str, Any]]] = {}
    open_group: dict[str, list[dict[str, Any]]] = {}
    last_index: dict[str, int] = {}
    for row in sorted(rows, key=lambda r: (r.get("testIdx", 0), r.get("repeatIndex") or 0)):
        gid, idx, h = row.get("repeatGroupId"), row.get("repeatIndex"), _test_key(row)[0]
        if isinstance(gid, str):
            if gid not in by_gid:
                by_gid[gid] = []
                groups.append(by_gid[gid])
            by_gid[gid].append(row)
        elif not isinstance(idx, int):
            open_group.pop(h, None)
            groups.append([row])
        elif h in open_group and idx > last_index[h]:
            open_group[h].append(row)
        else:
            open_group[h] = [row]
            groups.append(open_group[h])
        if isinstance(idx, int):
            last_index[h] = idx
    return groups


def _samples(rows: list[dict[str, Any]]) -> list[Sample]:
    seen: dict[str, int] = {}
    samples = []
    for group in _occurrences(rows):
        h, key = _test_key(group[0])
        seen[h] = seen.get(h, 0) + 1
        cid = h if seen[h] == 1 else f"{h}#{seen[h]}"
        samples.append(Sample(cid, key, [_result(cid, r) for r in group]))
    return samples


def _result(case_id: str, row: dict[str, Any]) -> CaseResult:
    response = as_dict(row.get("response"))
    tokens = as_dict(response.get("tokenUsage"))
    error = None
    if row.get("failureReason") == SYSTEM_ERROR:
        error = str(row.get("error") or response.get("error") or "promptfoo reported an error")
    output = SystemOutput(
        output=text_of(response.get("output")),
        latency_ms=finite(row.get("latencyMs")) or 0.0,
        tokens_in=count_of(tokens.get("prompt")),
        tokens_out=count_of(tokens.get("completion")),
        cost_usd=finite(row.get("cost")) or 0.0,
        cached=response.get("cached") is True,
        error=error,
    )
    return CaseResult(case_id=case_id, output=output, scores=_scores(case_id, row))


def _scores(case_id: str, row: dict[str, Any]) -> dict[str, Score]:
    grading = as_dict(row.get("gradingResult"))
    reason = grading.get("reason") or row.get("error")
    scores: dict[str, Score] = {}
    named = row.get("namedScores") or grading.get("namedScores") or {}
    for name, value in named.items() if isinstance(named, dict) else []:
        scores[str(name)] = continuous(value, case_id=case_id, metric=str(name))
    for name, passed in _assertions(grading.get("componentResults")).items():
        scores[f"assert:{name}"] = binary(passed)
    scores["pass"] = binary(row["success"], str(reason) if reason else None)
    return scores


def _assertions(components: Any) -> dict[str, bool]:
    """Pass/fail per assertion name. Assertions sharing a name must all pass."""
    verdicts: dict[str, bool] = {}
    for comp in _flatten(components):
        assertion = comp.get("assertion")
        if not isinstance(assertion, dict) or not isinstance(comp.get("pass"), bool):
            continue
        name = assertion.get("metric") or assertion.get("type")
        if name:
            verdicts[str(name)] = verdicts.get(str(name), True) and comp["pass"]
    return verdicts


def _flatten(components: Any) -> Iterable[dict[str, Any]]:
    """Leaf assertion results, looking inside assert-sets."""
    for comp in components if isinstance(components, list) else []:
        if not isinstance(comp, dict):
            continue
        nested = comp.get("componentResults")
        if nested:
            yield from _flatten(nested)
        else:
            yield comp
