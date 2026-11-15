"""Read Ragas samples as an evalproof dataset (``evalproof import-dataset ragas``).

Ragas names, v2 first and the legacy v1 spelling after it:
``user_input`` (``question``), ``reference`` (``ground_truth``, or a ``ground_truths`` list whose
first entry is used). A multi-turn sample's ``user_input`` is a list of messages
(``type`` human, ai or tool, with ``content``), read by the chat rule: system messages dropped,
user-only messages become ``input`` and ``turns``.

The reference is free text, so it becomes the case ``reference`` and the ``llm_judge`` scorer.
``rubrics`` (score-level descriptions, not pass or fail criteria), ``reference_tool_calls`` and
``reference_topics`` have no scorer here and stay on the case under ``unmapped``. What a past run
produced (``response``, ``answer``, ``retrieved_contexts``, ``contexts``) is not part of the test
and is not carried over. Written from Ragas's documented sample fields, no real file was available:
experimental.
"""

from __future__ import annotations

from pathlib import Path

from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    add_reference,
    csv_cell,
    first_present,
    read_records,
    reject_fields,
    source_id,
)

FORMAT = "ragas"
_IGNORED = (
    "response",
    "answer",
    "retrieved_contexts",
    "contexts",
    "reference_contexts",
    "multi_responses",
)
_UNMAPPED = ("rubrics", "reference_tool_calls", "reference_topics")


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    builder = Builder(FORMAT, path)
    is_csv = path.suffix.lower() == ".csv"
    for position, raw in enumerate(read_records(path), start=1):
        where = f"record {position}"
        record = {k: csv_cell(v) for k, v in raw.items()} if is_csv else raw
        resolved = builder.input_of(first_present(record, ("user_input", "question")), where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        draft = Draft(
            input=case_input, turns=turns, source_id=source_id(record.get("id")), where=where
        )
        reference = first_present(record, ("reference", "ground_truth"))
        truths = record.get("ground_truths")
        if reference is None and isinstance(truths, list) and truths:
            reference = truths[0]
            if len(truths) > 1:
                builder.tally("first_truth")
        draft.key = reference
        add_reference(builder, draft, reference)
        for name in _UNMAPPED:
            if record.get(name) not in (None, "", [], {}):
                builder.unmap(draft, name, {name: record[name]})
        builder.ignore_present(record, _IGNORED)
        builder.add(draft)
    return builder.finish()
