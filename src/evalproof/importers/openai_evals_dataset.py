"""Read OpenAI Evals samples as an evalproof dataset (``evalproof import-dataset openai-evals``).

A ``samples.jsonl`` line is ``{"input": <chat messages or a string>, "ideal": <string or list>}``;
a platform dataset line wraps the same in ``{"item": {...}}``. ``input`` follows the chat rule
(system messages dropped; user-only messages become ``input`` and ``turns``).

``samples.jsonl`` does not say how the eval compares (Match, Includes, FuzzyMatch): that lives in
the eval's registry YAML, which is not the file given here. One ``ideal`` is read as Match: it
becomes the ``reference`` with the ``exact_match`` scorer. A list of ideals is read as Includes:
the ``accept`` list with the ``contains`` scorer. A note says so; edit the suite's scorers if the
eval was another kind (FuzzyMatch has no scorer here). A platform item with no ``input`` keeps
all of its keys as the input and has no reference. Experimental: no verbatim sample line was
available.
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
    read_records,
    reject_fields,
    source_id,
    text_of_value,
)

FORMAT = "openai-evals"


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    if path.suffix.lower() == ".csv":
        raise ImportFormatError(f"{path}: the openai-evals format reads .jsonl or .json samples")
    builder = Builder(FORMAT, path)
    assumed = False
    for position, record in enumerate(read_records(path), start=1):
        where = f"record {position}"
        item = record["item"] if isinstance(record.get("item"), dict) else record
        raw_input = item.get("input", item.get("prompt"))
        ideal = item.get("ideal")
        if raw_input is None:
            rest = {k: v for k, v in item.items() if k != "ideal"}
            if not rest:
                builder.tally("no_input")
                continue
            builder.note("an item with no 'input' keeps all of its keys (but 'ideal') as the input")
            raw_input = rest
        resolved = builder.input_of(raw_input, where)
        if resolved is None:
            builder.tally("no_input")
            continue
        case_input, turns = resolved
        draft = Draft(input=case_input, turns=turns, key=ideal, where=where)
        source = source_id(item.get("id", record.get("id")))
        draft.source_id = source
        raw_ideals = ideal if isinstance(ideal, list) else [ideal]
        ideals = [text_of_value(i) for i in raw_ideals if i not in (None, "")]
        if len(ideals) == 1:
            add_reference(builder, draft, ideals[0], exact=True)
            assumed = True
        elif ideals:
            draft.reference = ideals[0]
            draft.add_list("accept", ideals, case_sensitive=False)
            assumed = True
        if "input" in item:
            builder.ignore_present(item, [k for k in item if k not in ("input", "ideal", "id")])
        builder.add(draft)
    if assumed:
        builder.note(
            "samples.jsonl does not record how the eval compares answers: one ideal is read as "
            "Match (exact_match), a list of ideals as Includes (contains); edit the scorers in "
            "the suite if the eval was another kind"
        )
    return builder.finish()
