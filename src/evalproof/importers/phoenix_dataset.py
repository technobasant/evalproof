"""Read Arize Phoenix dataset examples as a dataset (``evalproof import-dataset phoenix``).

An example has ``input`` and ``output`` objects (``output`` may be null), ``metadata``, and an
``id`` when the export carries one; ``splits`` (a list or a string) become tags. ``input`` is the
case input (an object, or a chat when it is ``{"messages": [...]}``). ``output`` becomes the
``reference`` (the one value it holds, or its JSON when it has several keys) with the ``llm_judge``
scorer. Experimental: Phoenix's JSON export shape was not confirmed against a real file, so this
reader takes the documented example fields from a JSON list or JSONL.
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
    one_output,
    read_records,
    reject_fields,
    source_id,
)

FORMAT = "phoenix"


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    if path.suffix.lower() == ".csv":
        raise ImportFormatError(f"{path}: the phoenix format reads .json or .jsonl examples")
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
            tags=as_tag_list(record.get("splits", record.get("split"))),
            source_id=source_id(record.get("id")),
            where=where,
        )
        expected = one_output(builder, record.get("output"))
        draft.key = expected
        add_reference(builder, draft, expected)
        if record.get("metadata"):
            draft.extras["metadata"] = jsonable(record["metadata"])
        builder.add(draft)
    return builder.finish()
