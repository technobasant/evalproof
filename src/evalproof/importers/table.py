"""Generic ``jsonl`` and ``csv``: the escape hatch for any tool that can write a table.

Long format, one row per (case, metric)::

    id,metric,passed,score,system,repeat,error,input,output,latency_ms

Only ``id``, ``metric`` and one of ``passed`` / ``score`` are needed. A row with ``passed``
and no ``score`` is a binary result; ``score`` alone is continuous; both together keep both.
``passed`` reads true/false/1/0/yes/no, and an empty cell means undecided. ``repeat`` numbers
the epochs of one case (collapsed by ``--epochs``); two rows for the same (case, metric)
without it are an error, because averaging them silently would hide a duplicated export.
``error`` marks a system error for that case. Column names can be remapped with
``--id-field``, ``--metric-field``, ``--score-field``, ``--passed-field`` and
``--system-field``; the optional columns above keep their names.

Wide format (``--wide``): one row per case, every column that is not ``id``, ``system``,
``repeat``, ``error``, ``input``, ``output`` or ``latency_ms`` is a metric. A column holding
only true/false/yes/no/1/0 is binary, any other numeric column is continuous.

With a ``system`` column the file may hold several systems and ``--system`` picks one;
without one, ``--system`` is just the label (default: the file name).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evalproof.importers._common import (
    ImportFormatError,
    Meta,
    Options,
    Sample,
    binary,
    build_run,
    finite,
    parse_jsonl,
    pick_system,
    read_text,
    score_value,
    text_of,
)
from evalproof.models import CaseResult, Run, Score, SystemOutput

_TRUE = {"true", "1", "yes"}
_FALSE = {"false", "0", "no"}
_OPTIONAL = ("repeat", "error", "input", "output", "latency_ms")
_FIELD_NAMES = ("id", "metric", "score", "passed", "system")


def read_jsonl(path: Path, opts: Options) -> Run:
    return _read("jsonl", path, parse_jsonl(read_text(path), path), opts)


def read_csv(path: Path, opts: Options) -> Run:
    text = read_text(path)
    rows = [
        {k: v for k, v in row.items() if k is not None}
        for row in csv.DictReader(io.StringIO(text, newline=""))
    ]
    return _read("csv", path, rows, opts)


class _Table:
    """The rows plus the column names this read uses, so errors can name the missing one."""

    def __init__(self, fmt: str, path: Path, rows: list[dict[str, Any]], opts: Options) -> None:
        unknown = set(opts.fields) - set(_FIELD_NAMES)
        if unknown:
            raise ImportFormatError(f"unknown field option(s): {', '.join(sorted(unknown))}")
        self.fmt, self.path, self.rows, self.opts = fmt, path, rows, opts
        self.col = {name: opts.fields.get(name, name) for name in _FIELD_NAMES}
        self.header = {k for row in rows for k in row}

    def need(self, name: str, flag: str) -> str:
        col = self.col[name]
        if col not in self.header:
            raise ImportFormatError(
                f"{self.path}: no {col!r} column (found: {', '.join(sorted(self.header))}); "
                f"name the column with {flag}"
            )
        return col

    def has(self, name: str) -> bool:
        return self.col[name] in self.header


def _read(fmt: str, path: Path, rows: list[dict[str, Any]], opts: Options) -> Run:
    if not rows:
        raise ImportFormatError(f"{path}: no rows")
    table = _Table(fmt, path, rows, opts)
    if opts.wide:
        extra = [k for k in ("metric", "score", "passed") if k in opts.fields]
        if extra:
            raise ImportFormatError(
                "--metric-field, --score-field and --passed-field do not apply to --wide"
            )
        records = _wide_records(table)
    else:
        records = _long_records(table)
    systems = list(dict.fromkeys(r["system"] for r in records)) if table.has("system") else []
    if systems:
        system = pick_system(systems, opts.system)
        records = [r for r in records if r["system"] == system]
    else:
        system = opts.system or path.stem
    return build_run(fmt, path, _samples(records, path), Meta(system=system), opts)


def _long_records(table: _Table) -> list[dict[str, Any]]:
    id_col = table.need("id", "--id-field")
    metric_col = table.need("metric", "--metric-field (or --wide for one column per metric)")
    if not (table.has("score") or table.has("passed")):
        raise ImportFormatError(
            f"{table.path}: needs a {table.col['passed']!r} or {table.col['score']!r} column "
            "(name them with --passed-field / --score-field)"
        )
    out = []
    for n, row in enumerate(table.rows, start=1):
        passed = _flag(row.get(table.col["passed"]), table.path, n, table.col["passed"])
        record = _common(table, row, n)
        record["id"] = _need_text(row.get(id_col), table.path, n, id_col)
        metric = _need_text(row.get(metric_col), table.path, n, metric_col)
        value = _score_cell(row, table.col["score"], table, n, record["id"], metric)
        if value is None and passed is not None:
            value = 1.0 if passed else 0.0
        record["scores"] = {metric: Score(value=value, passed=passed)}
        out.append(record)
    return out


def _wide_records(table: _Table) -> list[dict[str, Any]]:
    id_col = table.need("id", "--id-field")
    reserved = {id_col, table.col["system"], *_OPTIONAL}
    metrics = [c for c in dict.fromkeys(k for r in table.rows for k in r) if c not in reserved]
    if not metrics:
        raise ImportFormatError(f"{table.path}: --wide needs at least one metric column")
    binary_cols = {
        m for m in metrics if all(_is_flag(r.get(m)) for r in table.rows if _filled(r.get(m)))
    }
    out = []
    for n, row in enumerate(table.rows, start=1):
        record = _common(table, row, n)
        record["id"] = _need_text(row.get(id_col), table.path, n, id_col)
        scores = {}
        for m in metrics:
            cell = row.get(m)
            if not _filled(cell):
                scores[m] = Score(value=None)
            elif m in binary_cols:
                scores[m] = binary(bool(_flag(cell, table.path, n, m)))
            else:
                scores[m] = Score(value=_score_cell(row, m, table, n, record["id"], m))
        record["scores"] = scores
        out.append(record)
    return out


def _common(table: _Table, row: Mapping[str, Any], n: int) -> dict[str, Any]:
    """The columns both layouts share: system, repeat, error, input, output, latency."""
    sys_col = table.col["system"]
    latency = finite(_number(row.get("latency_ms"), table.path, n, "latency_ms"))
    return {
        "system": str(row.get(sys_col) if row.get(sys_col) is not None else ""),
        "repeat": row.get("repeat") if _filled(row.get("repeat")) else None,
        "error": str(row["error"]) if _filled(row.get("error")) else None,
        "input": row.get("input") if _filled(row.get("input")) else None,
        "output": text_of(row.get("output")) if _filled(row.get("output")) else "",
        "latency_ms": latency or 0.0,
        "row": n,
    }


def _samples(records: list[dict[str, Any]], path: Path) -> list[Sample]:
    """Group rows into cases and repeats; each (case, repeat) gets one result."""
    cases: dict[str, dict[Any, CaseResult]] = {}
    inputs: dict[str, Any] = {}
    for rec in records:
        reps = cases.setdefault(rec["id"], {})
        inputs.setdefault(rec["id"], rec["input"])
        result = reps.setdefault(
            rec["repeat"], CaseResult(case_id=rec["id"], output=SystemOutput(output=""))
        )
        for name, score in rec["scores"].items():
            if name in result.scores:
                raise ImportFormatError(
                    f"{path}: row {rec['row']}: {rec['id']!r} has {name!r} twice; "
                    "add a repeat column if these are repeats of one case"
                )
            result.scores[name] = score
        out = result.output
        out.output = out.output or rec["output"]
        out.latency_ms = out.latency_ms or rec["latency_ms"]
        out.error = out.error or rec["error"]
    return [Sample(cid, inputs[cid], list(reps.values())) for cid, reps in cases.items()]


def _filled(x: Any) -> bool:
    return x is not None and not (isinstance(x, str) and not x.strip())


def _need_text(x: Any, path: Path, n: int, col: str) -> str:
    if not _filled(x):
        raise ImportFormatError(f"{path}: row {n}: empty {col!r}")
    return str(x).strip()


def _is_flag(x: Any) -> bool:
    if isinstance(x, bool):
        return True
    return isinstance(x, str | int | float) and str(x).strip().lower() in _TRUE | _FALSE


def _flag(x: Any, path: Path, n: int, col: str) -> bool | None:
    if not _filled(x):
        return None
    if isinstance(x, bool):
        return x
    word = str(x).strip().lower()
    if word in _TRUE | _FALSE:
        return word in _TRUE
    raise ImportFormatError(f"{path}: row {n}: {col!r} is {x!r}; use true/false/1/0/yes/no")


def _number(x: Any, path: Path, n: int, col: str) -> float | None:
    """A cell as a number, or None when empty. NaN and infinity are returned, not judged here."""
    if not _filled(x):
        return None
    if isinstance(x, bool):
        return 1.0 if x else 0.0
    try:
        return float(x)
    except OverflowError:  # a JSON integer too large for a float
        return float("inf")
    except (TypeError, ValueError):
        raise ImportFormatError(f"{path}: row {n}: {col!r} is {x!r}, not a number") from None


def _score_cell(
    row: Mapping[str, Any], col: str, table: _Table, n: int, case_id: str, metric: str
) -> float | None:
    """A score cell: empty is undecided, NaN and infinity are rejected, naming case and metric."""
    return score_value(_number(row.get(col), table.path, n, col), case_id, metric)
