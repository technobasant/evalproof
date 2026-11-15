"""Read Braintrust dataset records as a dataset (``evalproof import-dataset braintrust``).

A record has ``input`` (any JSON: a string, an object, or chat messages), ``expected`` (any JSON),
``metadata`` and ``tags``, in a JSON list or a JSONL file, with ``id`` when the export has one.
``expected`` becomes the ``reference`` (a string as it is, anything else as its JSON) with the
``llm_judge`` scorer, since an expected value is free-form. ``tags`` become tags and ``metadata``
stays as ``metadata``. Experimental: Braintrust's export file shape was not confirmed against a
real file, so this reader takes the documented record fields.
"""

from __future__ import annotations

from pathlib import Path

from evalproof.importers._common import ImportFormatError
from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    add_reference,
    as_tag_list,
    jsonable,
    read_records,
    reject_fields,
    source_id,
)

FORMAT = "braintrust"


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    if path.suffix.lower() == ".csv":
        raise ImportFormatError(f"{path}: the braintrust format reads .json or .jsonl records")
    builder = Builder(FORMAT, path)
    for position, record in enumerate(read_records(path), start=1):
        where = f"record {position}"
        resolved = builder.input_of(record.get("input"), where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        draft = Draft(
            input=case_input,
            turns=turns,
            tags=as_tag_list(record.get("tags")),
            source_id=source_id(record.get("id")),
            key=record.get("expected"),
            where=where,
        )
        add_reference(builder, draft, record.get("expected"))
        if record.get("metadata"):
            draft.extras["metadata"] = jsonable(record["metadata"])
        builder.add(draft)
    return builder.finish()
