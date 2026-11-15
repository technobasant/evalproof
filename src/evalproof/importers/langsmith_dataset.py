"""Read LangSmith dataset examples as an evalproof dataset (``evalproof import-dataset langsmith``).

An example is ``{"inputs": {...}, "outputs": {...}, "metadata": {...}, "id": ..., "split": ...}``,
one per line of a JSONL file or in a JSON list (LangSmith has no built-in file export confirmed
here; teams write these files from ``list_examples``). ``inputs`` becomes the case input (an
object, or a chat when it is ``{"messages": [...]}``); ``outputs`` becomes the ``reference``
(the one value it holds, or its JSON when it has several keys) with the ``llm_judge`` scorer,
because the outputs are free text. The example ``id`` is the case id; ``split`` (or
``metadata.dataset_split``) becomes tags; ``metadata`` is kept as ``metadata``. Experimental:
written from the documented example fields, with no real export to check against.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

FORMAT = "langsmith"


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    if path.suffix.lower() == ".csv":
        raise ImportFormatError(f"{path}: the langsmith format reads .json or .jsonl examples")
    builder = Builder(FORMAT, path)
    for position, record in enumerate(read_records(path), start=1):
        where = f"record {position}"
        resolved = builder.input_of(record.get("inputs"), where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        raw_meta = record.get("metadata")
        metadata: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
        tags = as_tag_list(record.get("split")) or as_tag_list(metadata.get("dataset_split"))
        draft = Draft(
            input=case_input,
            turns=turns,
            tags=tags,
            source_id=source_id(record.get("id")),
            where=where,
        )
        expected = one_output(builder, record.get("outputs"))
        draft.key = expected
        add_reference(builder, draft, expected)
        if metadata:
            draft.extras["metadata"] = jsonable(metadata)
        builder.add(draft)
    return builder.finish()
