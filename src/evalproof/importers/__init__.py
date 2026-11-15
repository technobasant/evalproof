"""Import results from other evaluation tools into an evalproof ``Run``.

Teams already have eval results in promptfoo, Inspect, DeepEval, lm-eval, garak, Google ADK or
Claude Code's plugin eval. evalproof's value is the statistics and the gate, not running the
evals, so these importers turn those tools' own output files into the run artifact that
``compare`` and the gate read, without re-running anything.

Every importer follows the same rules (see ``_common``): case ids are stable across runs of
the same dataset, system errors score as failures, repeats collapse by ``epochs``, and
a NaN or infinite score is an error naming the case and metric (it would otherwise make the
paired bootstrap report p = 0).

The dataset hash of an imported run is a hash over the case ids and the inputs *as that
tool recorded them*. Two runs imported by the same importer from the same dataset have the
same hash and compare; a promptfoo run and an Inspect run never do, because each tool keeps
different things in its input. Compare runs imported by the same importer.

``adk`` and ``claude-plugin-eval`` are experimental (``EXPERIMENTAL_FORMATS``): no real sample of
their files was available, so they were written from the tools' schemas. ``adk_dataset`` is not an
importer of runs: it turns an ADK eval set into a dataset (``evalproof import-dataset adk``).

The ``*_dataset`` modules turn other tools' TEST CASES into evalproof cases and the scorers those
cases need (``import_cases``; ``import_dataset`` returns just the cases). Their shared rules are in
``_dataset``. ``EXPERIMENTAL_DATASET_FORMATS`` lists the readers written from documentation alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from evalproof.importers import (
    adk,
    adk_dataset,
    braintrust_dataset,
    claude_plugin_eval,
    deepeval,
    deepeval_dataset,
    garak,
    inspect_ai,
    inspect_dataset,
    langfuse_dataset,
    langsmith_dataset,
    lm_eval,
    openai_evals_dataset,
    phoenix_dataset,
    promptfoo,
    promptfoo_dataset,
    ragas_dataset,
    table,
    table_dataset,
)
from evalproof.importers._common import (
    EPOCHS,
    EXPERIMENTAL_DATASET_FORMATS,
    EXPERIMENTAL_FORMATS,
    ImportFormatError,
    Options,
    _default_warn,
    imported_dataset_hash,
    warn_experimental,
)
from evalproof.importers._dataset import DatasetOptions, ImportedDataset
from evalproof.importers._detect import detect_format
from evalproof.models import Case, Run

_READERS: dict[str, Callable[[Path, Options], Run]] = {
    "promptfoo": promptfoo.read,
    "inspect": inspect_ai.read,
    "deepeval": deepeval.read,
    "lm-eval": lm_eval.read,
    "garak": garak.read,
    "adk": adk.read,
    "claude-plugin-eval": claude_plugin_eval.read,
    "jsonl": table.read_jsonl,
    "csv": table.read_csv,
}
FORMATS = tuple(_READERS)

_DATASET_READERS: dict[str, Callable[[Path, DatasetOptions], ImportedDataset]] = {
    "promptfoo": promptfoo_dataset.read,
    "deepeval": deepeval_dataset.read,
    "ragas": ragas_dataset.read,
    "langsmith": langsmith_dataset.read,
    "openai-evals": openai_evals_dataset.read,
    "inspect": inspect_dataset.read,
    "braintrust": braintrust_dataset.read,
    "langfuse": langfuse_dataset.read,
    "phoenix": phoenix_dataset.read,
    "adk": adk_dataset.read_dataset,
    "jsonl": table_dataset.read_jsonl,
    "csv": table_dataset.read_csv,
}
# ``auto`` is not a reader: it picks one of the others from the file (``detect_format``).
DATASET_FORMATS = ("auto", *_DATASET_READERS)

__all__ = [
    "DATASET_FORMATS",
    "EPOCHS",
    "EXPERIMENTAL_DATASET_FORMATS",
    "EXPERIMENTAL_FORMATS",
    "FORMATS",
    "DatasetOptions",
    "ImportFormatError",
    "ImportedDataset",
    "detect_format",
    "import_cases",
    "import_dataset",
    "import_run",
    "imported_dataset_hash",
]


def import_run(
    fmt: str,
    path: str | Path,
    *,
    system: str | None = None,
    epochs: str = "mean",
    warn: Callable[[str], None] | None = None,
    fields: Mapping[str, str] | None = None,
    wide: bool = False,
    threshold: float | None = None,
) -> Run:
    """Read ``path`` as ``fmt`` and return it as a ``Run``.

    ``system`` picks one system out of a file that holds several (promptfoo, claude-plugin-eval's
    ``with`` and ``without`` arms, and jsonl or csv with a ``system`` column) and is otherwise the
    label to give the run. ``epochs`` says how repeats of one case collapse: ``mean`` (the
    default), ``first``, ``all`` (a pass only if every repeat passed) or ``any``; ``trials`` keeps
    each repeat as its own trial of the case (every case needs the same number of repeats).
    ``warn`` receives notes about anything skipped or incomplete; by default they are raised as
    Python warnings. ``fields`` and ``wide`` apply to ``jsonl`` and ``csv`` only; ``threshold``
    (the detector score at which a garak output counts as a hit) to ``garak`` only.

    Raises ``ImportFormatError`` (a ``ValueError``) when the file is not in the format
    asked for or cannot be imported as asked.
    """
    reader = _READERS.get(fmt)
    if reader is None:
        raise ImportFormatError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    if threshold is not None and fmt != "garak":
        raise ImportFormatError(f"--threshold applies to the garak format, not {fmt}")
    options = Options(
        system=system, epochs=epochs, fields=dict(fields or {}), wide=wide, threshold=threshold
    )
    if warn is not None:
        options = replace(options, warn=warn)
    return reader(Path(path), options)


def import_cases(
    fmt: str,
    path: str | Path,
    *,
    warn: Callable[[str], None] | None = None,
    id_field: str | None = None,
    input_fields: Sequence[str] = (),
    reference_field: str | None = None,
    tags_field: str | None = None,
) -> ImportedDataset:
    """Read the test cases in ``path`` as ``fmt`` and return them with the scorers they need.

    ``fmt`` is one of ``DATASET_FORMATS``; ``auto`` picks the format from the file and says so in
    ``ImportedDataset.detected``. ``id_field``, ``input_fields``, ``reference_field`` and
    ``tags_field`` name the columns of a generic ``jsonl`` or ``csv`` file and apply to those two
    formats only. ``warn`` receives the notes (what was not imported, and why); by default they
    are raised as Python warnings. Raises ``ImportFormatError`` (a ``ValueError``) when the file
    is not in the format asked for, or ``auto`` cannot tell.
    """
    target = Path(path)
    detected = fmt == "auto"
    if detected:
        fmt = detect_format(target, DATASET_FORMATS)
    reader = _DATASET_READERS.get(fmt)
    if reader is None:
        raise ImportFormatError(
            f"unknown dataset format {fmt!r}; choose one of {', '.join(DATASET_FORMATS)}"
        )
    say = warn or _default_warn
    if fmt in EXPERIMENTAL_DATASET_FORMATS:
        warn_experimental(fmt)
    result = reader(
        target,
        DatasetOptions(
            warn=say,
            id_field=id_field,
            input_fields=tuple(input_fields),
            reference_field=reference_field,
            tags_field=tags_field,
        ),
    )
    if detected and fmt == "deepeval":
        result.notes.insert(
            0,
            "Langfuse dataset items have the same keys and map the same way; "
            "name the format langfuse to use the item id as the case id",
        )
    result.format = fmt
    result.detected = detected
    return result


def import_dataset(
    fmt: str, path: str | Path, *, warn: Callable[[str], None] | None = None
) -> list[Case]:
    """Read the test cases in ``path`` as ``fmt`` and return them as evalproof cases, ready for
    ``write_dataset``. ``import_cases`` also returns the scorers the cases need.

    ``fmt`` is ``adk`` (an ``*.evalset.json`` or the flat ``*.test.json`` list) or any other of
    ``DATASET_FORMATS``; ``auto`` picks the format from the file. ``warn`` receives notes about
    cases it skipped or fields it did not carry over; by default they are raised as Python
    warnings. Raises ``ImportFormatError`` (a ``ValueError``) when the file is not in the format
    asked for.
    """
    return import_cases(fmt, path, warn=warn).cases
