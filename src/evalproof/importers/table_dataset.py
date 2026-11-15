"""Read any JSONL or CSV file as an evalproof dataset, with the columns named on the command line
(``evalproof import-dataset jsonl FILE --input-field question --reference-field answer``).

``--input-field`` names the column that holds the input; several (repeat the flag, or comma
separate) make the input an object of those columns. With no ``--input-field`` an ``input`` column
is used when the file has one. ``--reference-field`` becomes the ``reference`` (the ``llm_judge``
scorer compares answers with it: change the scorer to ``exact_match`` in the suite if the
references are short labels), ``--id-field`` the case id, ``--tags-field`` the tags (a list, or a
comma-separated string). Other columns are not carried over.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evalproof.importers._common import ImportFormatError, parse_jsonl, read_text
from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    add_reference,
    as_tag_list,
    csv_rows,
    source_id,
)


def read_jsonl(path: Path, opts: DatasetOptions) -> ImportedDataset:
    return _read("jsonl", parse_jsonl(read_text(path), path), path, opts)


def read_csv(path: Path, opts: DatasetOptions) -> ImportedDataset:
    return _read("csv", csv_rows(read_text(path), path), path, opts)


def _read(
    fmt: str, records: list[dict[str, Any]], path: Path, opts: DatasetOptions
) -> ImportedDataset:
    if not records:
        raise ImportFormatError(f"{path}: no records found")
    columns = set().union(*(r.keys() for r in records[:50]))
    wanted = list(opts.input_fields) or (["input"] if "input" in columns else [])
    if not wanted:
        raise ImportFormatError(
            f"{path}: say which column holds the input with --input-field "
            f"(columns found: {', '.join(sorted(map(str, columns)))})"
        )
    named = [*wanted, *(f for f in (opts.id_field, opts.reference_field, opts.tags_field) if f)]
    missing = [name for name in named if name not in columns]
    if missing:
        raise ImportFormatError(
            f"{path}: no column {', '.join(repr(m) for m in missing)} "
            f"(columns found: {', '.join(sorted(map(str, columns)))})"
        )
    builder = Builder(fmt, path)
    for position, record in enumerate(records, start=1):
        where = f"record {position}"
        values = {name: record.get(name) for name in wanted}
        if all(v in (None, "", [], {}) for v in values.values()):
            builder.tally("no_input")
            continue
        raw: Any = values[wanted[0]] if len(wanted) == 1 else values
        resolved = builder.input_of(raw, where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        reference = record.get(opts.reference_field) if opts.reference_field else None
        draft = Draft(
            input=case_input,
            turns=turns,
            source_id=source_id(record.get(opts.id_field)) if opts.id_field else None,
            tags=as_tag_list(record.get(opts.tags_field)) if opts.tags_field else [],
            key=reference,
            where=where,
        )
        add_reference(builder, draft, reference)
        builder.add(draft)
    return builder.finish()
