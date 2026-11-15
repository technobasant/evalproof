"""Read DeepEval goldens as an evalproof dataset (``evalproof import-dataset deepeval``).

A golden file is JSON, JSONL or CSV. A single-turn golden has ``input`` and ``expected_output``
(``name`` is its id); a multi-turn one has ``turns`` (``role`` and ``content``) and
``expected_outcome``; ``context`` and ``additional_metadata`` ride along as the case's ``context``
and ``metadata``. In a CSV, list cells are ``|``-separated, as DeepEval's own loader reads them.

DeepEval stores no assertions in a golden. The expected text is free prose, so it becomes the
case's ``reference`` and the ``llm_judge`` scorer compares answers with it; an exact match on prose
would fail correct answers. A golden that is only a ``scenario`` (a user simulation) has no turns
to replay and is skipped with a count. ``actual_output`` and ``retrieval_context`` are what a past
run produced, not part of the test, and are not carried over.

Langfuse dataset items have the same keys (``input``, ``expected_output`` or ``expectedOutput``,
``metadata``, ``id``) and map the same way, so ``read_goldens`` also serves ``langfuse``.
No real sample of a saved golden file was available: this reader was written from DeepEval's
documented fields and is experimental.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    add_reference,
    csv_cell,
    first_present,
    jsonable,
    read_records,
    reject_fields,
    source_id,
)

FORMAT = "deepeval"
_IGNORED = (
    "actual_output",
    "retrieval_context",
    "tools_called",
    "expected_tools",
    "comments",
    "custom_column_key_values",
    "sourceTraceId",
    "sourceObservationId",
    "datasetId",
    "datasetName",
    "status",
    "createdAt",
    "updatedAt",
)


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    return read_goldens(path, FORMAT, id_keys=("name", "id"))


def read_goldens(path: Path, fmt: str, *, id_keys: tuple[str, ...]) -> ImportedDataset:
    builder = Builder(fmt, path)
    is_csv = path.suffix.lower() == ".csv"
    for position, record in enumerate(read_records(path), start=1):
        where = f"record {position}"
        turns = csv_cell(record.get("turns")) if is_csv else record.get("turns")
        resolved: tuple[str | dict[str, Any], list[Any]] | None
        if isinstance(turns, list) and turns:
            resolved = builder.chat(turns, where)
        else:
            resolved = builder.input_of(record.get("input"), where)
        if resolved is None:
            builder.tally("scenario_only" if record.get("scenario") else "no_input")
            continue
        case_input, later = resolved
        draft = Draft(
            input=case_input,
            turns=later,
            source_id=source_id(first_present(record, id_keys)),
            where=where,
        )
        expected = first_present(record, ("expected_output", "expectedOutput", "expected_outcome"))
        draft.key = expected
        add_reference(builder, draft, expected)
        context = record.get("context")
        if isinstance(context, str) and is_csv:
            context = [c for c in context.split("|") if c]
        if context:
            draft.extras["context"] = jsonable(context)
        meta = record.get("additional_metadata", record.get("metadata"))
        meta = csv_cell(meta) if is_csv else meta
        if meta:
            draft.extras["metadata"] = jsonable(meta)
        builder.ignore_present(record, _IGNORED)
        builder.add(draft)
    return builder.finish()
