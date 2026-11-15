"""Read Inspect AI samples as an evalproof dataset (``evalproof import-dataset inspect``).

A sample has ``input`` (a string or chat messages), ``target`` (a string or a list), ``choices``,
``id``, ``metadata`` and more, in a JSON list, JSONL or CSV (a CSV ``choices`` cell is a JSON
list, or comma-separated). The id is the sample's ``id`` when it has one (Inspect numbers samples
from 1 when it has none, which a file does not record, so the content hash is used instead).

* With ``choices`` the ``target`` is a letter. The choices are written into the input as ``A) ...``
  lines with an instruction to answer with the letter, the letter is the ``reference``, and the
  scorer is ``exact_match``: that is what Inspect's multiple-choice scoring compares.
* A list of targets becomes the ``accept`` list with the ``contains`` scorer.
* A free-text ``target`` becomes the ``reference`` with the ``llm_judge`` scorer, because the
  sample does not store which Inspect scorer it was run with.

``metadata.tags`` (when it is a list) become tags and the rest of ``metadata`` stays as
``metadata``. ``sandbox``, ``files``, ``setup`` and ``description`` are not carried over.
Experimental: written from Inspect's documented ``Sample`` fields, with no real file to check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    add_reference,
    as_tag_list,
    csv_cell,
    jsonable,
    read_records,
    reject_fields,
    source_id,
    text_of_value,
)

FORMAT = "inspect"
_IGNORED = ("sandbox", "files", "setup", "description")
_INSTRUCTION = "Answer with the letter of the correct choice only."


def _choices(value: Any) -> list[str]:
    if isinstance(value, str):
        value = csv_cell(value) if value.strip()[:1] == "[" else value.split(",")
    return [str(c).strip() for c in value] if isinstance(value, list) else []


def _with_choices(case_input: str | dict[str, Any], choices: list[str]) -> str | dict[str, Any]:
    lines = [f"{chr(65 + i)}) {c}" for i, c in enumerate(choices)]
    if isinstance(case_input, str):
        return f"{case_input}\n\n" + "\n".join(lines) + f"\n\n{_INSTRUCTION}"
    return {**case_input, "choices": choices}


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    builder = Builder(FORMAT, path)
    is_csv = path.suffix.lower() == ".csv"
    for position, record in enumerate(read_records(path), start=1):
        where = f"record {position}"
        raw_input = record.get("input")
        if is_csv:
            raw_input = csv_cell(raw_input)
        resolved = builder.input_of(raw_input, where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        choices = _choices(record.get("choices"))
        if choices:
            case_input = _with_choices(case_input, choices)
        target = csv_cell(record.get("target")) if is_csv else record.get("target")
        draft = Draft(
            input=case_input,
            turns=turns,
            source_id=source_id(record.get("id")),
            key=[target, choices],
            where=where,
        )
        raw_targets = target if isinstance(target, list) else [target]
        targets = [text_of_value(t) for t in raw_targets if t not in (None, "")]
        if len(targets) == 1:
            add_reference(builder, draft, targets[0], exact=bool(choices))
        elif targets:
            draft.reference = targets[0]
            draft.add_list("accept", targets, case_sensitive=False)
        meta = csv_cell(record.get("metadata")) if is_csv else record.get("metadata")
        if isinstance(meta, dict):
            draft.tags = as_tag_list(meta.get("tags"))
            rest = {k: v for k, v in meta.items() if k != "tags"}
            if rest:
                draft.extras["metadata"] = jsonable(rest)
        elif meta:
            draft.extras["metadata"] = json.loads(json.dumps(meta, default=str))
        builder.ignore_present(record, _IGNORED)
        builder.add(draft)
    return builder.finish()
