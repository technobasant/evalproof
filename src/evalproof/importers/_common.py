"""What every importer shares: the error type, options, and one rule for repeated runs.

An importer reads one tool's file and returns ``Sample`` objects: a stable case id, the
input as the file gave it, and one ``CaseResult`` per repeat (epoch). ``build_run`` then
applies the rules that must not differ between formats, so that a run imported from
promptfoo and one from Inspect behave the same in ``compare`` and the gate:

* system errors score as failures on every metric, as they do in native runs;
* repeats collapse to one result per case (``mean``, ``first``, ``all`` or ``any``);
* the dataset hash is computed the same way.
"""

from __future__ import annotations

import hashlib
import json
import math
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalproof._version import __version__
from evalproof.dataset import canonical_json
from evalproof.models import CaseResult, Run, RunInfo, Score, SystemOutput

EPOCHS = ("mean", "first", "all", "any")


class ImportFormatError(ValueError):
    """The file is not in the format asked for, or cannot be imported as asked."""


def _default_warn(message: str) -> None:
    warnings.warn(message, stacklevel=3)


@dataclass(frozen=True)
class Options:
    """What the caller asked for. ``fields`` and ``wide`` apply to ``jsonl`` and ``csv`` only."""

    system: str | None = None
    epochs: str = "mean"
    fields: Mapping[str, str] = field(default_factory=dict)
    wide: bool = False
    warn: Callable[[str], None] = _default_warn


@dataclass
class Sample:
    """One case as found in a file: its id, its input, and one result per repeat."""

    case_id: str
    input: Any
    repeats: list[CaseResult]


@dataclass
class Meta:
    """Run-level facts a format can read from the file; blanks are filled with defaults."""

    suite: str = ""
    system: str = ""
    started_at: str = ""
    finished_at: str = ""
    wall_time_s: float = 0.0
    schema: str = ""
    tool_version: str = ""
    environment: dict[str, str] = field(default_factory=dict)


def imported_dataset_hash(items: Iterable[tuple[str, Any]]) -> str:
    """Hash of the cases an import found: sha256 over ``(case_id, input)`` sorted by id.

    The input is hashed as the file gave it (a string, or a dict of variables), in
    canonical JSON. Formats keep different things in their input, so a hash only matches
    between runs imported by the same importer; that is also what stops a promptfoo run
    being gated against an Inspect run by accident.
    """
    ordered = sorted(items, key=lambda item: item[0])
    return hashlib.sha256(canonical_json(ordered).encode()).hexdigest()[:16]


def imported_case_hash(case_id: str, case_input: Any) -> str:
    """Hash of one imported case, built from the same ``(case_id, input)`` pair as the dataset
    hash, so an unchanged case keeps its hash and an edited one does not."""
    return hashlib.sha256(canonical_json([case_id, case_input]).encode()).hexdigest()[:16]


def pick_system(found: Sequence[str], wanted: str | None) -> str:
    """Choose the system to import, or say which ones the file holds."""
    if not found:
        raise ImportFormatError("the file holds no results")
    listing = ", ".join(repr(s) for s in found)
    if wanted is not None:
        if wanted in found:
            return wanted
        raise ImportFormatError(f"no system {wanted!r} in this file; available: {listing}")
    if len(found) > 1:
        raise ImportFormatError(
            f"this file holds {len(found)} systems; choose one with --system. Available: {listing}"
        )
    return found[0]


def reject_table_options(opts: Options, fmt: str) -> None:
    if opts.fields or opts.wide:
        raise ImportFormatError(
            f"--id-field, --metric-field, --score-field, --passed-field, --system-field and "
            f"--wide apply to the jsonl and csv formats, not {fmt}"
        )


def finite(x: Any) -> float | None:
    """A real, finite number as a float; anything else (bool, NaN, text) is None."""
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    try:
        value = float(x)
    except OverflowError:
        return None
    return value if math.isfinite(value) else None


def binary(passed: bool, reason: str | None = None) -> Score:
    return Score(value=1.0 if passed else 0.0, passed=passed, reason=reason)


def score_value(x: Any, case_id: str, metric: str) -> float | None:
    """The number behind a score: absent or not a number is undecided, NaN or infinity an error.

    Every format reads its numeric scores through here (via ``continuous`` or directly).
    NaN must not get through to ``compare``: every comparison with it is false, so the
    paired bootstrap reports p = 0, a "significant" difference conjured from a broken cell.
    Failing at import names the case and metric to fix instead.
    """
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    try:
        value = float(x)
    except OverflowError:  # an integer too large for a float
        value = math.inf
    if not math.isfinite(value):
        raise ImportFormatError(
            f"case {case_id!r}, metric {metric!r}: the score is {value!r}; NaN and infinite "
            "scores cannot be compared. Fix the value or drop the row"
        )
    return value


def continuous(value: Any, reason: str | None = None, *, case_id: str, metric: str) -> Score:
    """A continuous score for one case and metric; see ``score_value`` for what it rejects."""
    return Score(value=score_value(value, case_id, metric), reason=reason)


def as_dict(x: Any) -> dict[str, Any]:
    """``x`` if it is a JSON object, else an empty one: a file is not trusted to have a shape."""
    return x if isinstance(x, dict) else {}


def text_of(x: Any) -> str:
    """A model output as text: strings as they are, anything structured as canonical JSON."""
    if x is None:
        return ""
    return x if isinstance(x, str) else canonical_json(x)


def count_of(x: Any) -> int:
    """A token count from a file; missing or malformed counts are 0."""
    value = finite(x)
    return int(value) if value is not None and value > 0 else 0


def number_duplicates(ids: Iterable[str]) -> list[str]:
    """Second and later occurrences of an id get ``#2``, ``#3``: ids stay stable per position."""
    seen: dict[str, int] = {}
    out = []
    for cid in ids:
        seen[cid] = seen.get(cid, 0) + 1
        out.append(cid if seen[cid] == 1 else f"{cid}#{seen[cid]}")
    return out


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ImportFormatError(f"{path}: not UTF-8 text ({exc.reason})") from exc


def parse_json(text: str, path: Path | str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ImportFormatError(f"{path}: not valid JSON: {exc.msg} (line {exc.lineno})") from exc


def read_json(path: Path) -> Any:
    return parse_json(read_text(path), path)


def parse_jsonl(text: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        row = parse_json(line, f"{path}:{lineno}")
        if not isinstance(row, dict):
            raise ImportFormatError(f"{path}:{lineno}: each line must be a JSON object")
        rows.append(row)
    return rows


def _merge_scores(scores: list[Score], epochs: str, repeated: bool) -> Score:
    """Collapse one metric's scores over a case's repeats into a single score."""
    if len(scores) == 1 and (epochs != "mean" or not repeated):
        return scores[0]
    decided = [s for s in scores if s.value is not None]
    values = [s.value for s in decided if s.value is not None]
    if not values:
        return Score(value=None)
    if epochs == "mean":
        # A mean of pass/fail repeats is a fraction, so the case is no longer binary.
        return Score(value=sum(values) / len(values))
    if all(s.passed is not None for s in decided):
        flags = [bool(s.passed) for s in decided]
        undecided = len(decided) < len(scores)
        if epochs == "all":
            verdict = None if all(flags) and undecided else all(flags)
        else:
            verdict = None if not any(flags) and undecided else any(flags)
        if verdict is None:
            return Score(value=None)
        return binary(verdict)
    # Continuous scores have no pass/fail: ``all`` is the worst repeat, ``any`` the best.
    return Score(value=min(values) if epochs == "all" else max(values))


def _merge_output(repeats: list[CaseResult], epochs: str) -> SystemOutput:
    outputs = [r.output for r in repeats]
    ok = [o for o in outputs if not o.error]
    errors = [o.error for o in outputs if o.error]
    # The case is an error when no repeat answered, or under ``all``, where one failed repeat
    # fails the case. Under ``any`` and ``mean`` a case with a good repeat is not an error;
    # its failed repeats already count as failures in the scores.
    if ok and epochs != "all":
        errors = []
    return SystemOutput(
        output=(ok or outputs)[0].output,
        retrieved=(ok or outputs)[0].retrieved,
        latency_ms=sum(o.latency_ms for o in ok) / len(ok) if ok else 0.0,
        tokens_in=sum(o.tokens_in for o in outputs),
        tokens_out=sum(o.tokens_out for o in outputs),
        cost_usd=sum(o.cost_usd for o in outputs),
        cached=all(o.cached for o in outputs),
        error=f"{len(errors)} of {len(outputs)} repeats errored: {errors[0]}" if errors else None,
    )


def _collapse(sample: Sample, epochs: str, repeated: bool) -> CaseResult:
    repeats = sample.repeats
    if epochs == "first" or (len(repeats) == 1 and not (epochs == "mean" and repeated)):
        return repeats[0].model_copy(update={"case_id": sample.case_id})
    names = list(dict.fromkeys(n for r in repeats for n in r.scores))
    scores = {
        name: _merge_scores([r.scores[name] for r in repeats if name in r.scores], epochs, repeated)
        for name in names
    }
    return CaseResult(case_id=sample.case_id, output=_merge_output(repeats, epochs), scores=scores)


def build_run(fmt: str, path: Path, samples: list[Sample], meta: Meta, opts: Options) -> Run:
    """Apply the shared rules and assemble the ``Run`` an importer found."""
    if opts.epochs not in EPOCHS:
        raise ImportFormatError(f"epochs must be one of {', '.join(EPOCHS)}, got {opts.epochs!r}")
    if not samples:
        raise ImportFormatError(f"{path}: no cases found")
    ids = [s.case_id for s in samples]
    if len(set(ids)) != len(ids):
        raise ImportFormatError(f"{path}: duplicate case ids after import")  # pragma: no cover

    # A system error is a failure on every metric the file reports, not a missing score.
    names = list(dict.fromkeys(n for s in samples for r in s.repeats for n in r.scores))
    for sample in samples:
        for r in sample.repeats:
            if r.output.error:
                reason = f"system error: {r.output.error[:200]}"
                r.scores = {n: Score(value=0.0, passed=False, reason=reason) for n in names}

    repeated = any(len(s.repeats) > 1 for s in samples)
    results = [_collapse(s, opts.epochs, repeated) for s in samples]
    for sample, result in zip(samples, results, strict=True):
        # An imported case carries no tags, so its content hash is its case hash.
        result.case_hash = result.content_hash = imported_case_hash(sample.case_id, sample.input)

    environment = {"imported_from": fmt}
    if meta.schema:
        environment["source_schema"] = meta.schema
    if meta.tool_version:
        environment["source_tool_version"] = meta.tool_version
    if repeated:
        environment["epochs"] = opts.epochs
    environment.update(meta.environment)

    info = RunInfo(
        suite=meta.suite or path.stem,
        system=meta.system,
        dataset_path=str(path),
        dataset_hash=imported_dataset_hash((s.case_id, s.input) for s in samples),
        started_at=meta.started_at,
        finished_at=meta.finished_at,
        wall_time_s=meta.wall_time_s,
        evalproof_version=__version__,
        seed=0,
        environment=environment,
    )
    return Run(info=info, results=results)
