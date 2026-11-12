"""Import results from other evaluation tools into an evalproof ``Run``.

Teams already have eval results in promptfoo, Inspect, DeepEval or lm-eval. evalproof's
value is the statistics and the gate, not running the evals, so these importers turn those
tools' own output files into the run artifact that ``compare`` and the gate read, without
re-running anything.

Every importer follows the same rules (see ``_common``): case ids are stable across runs of
the same dataset, system errors score as failures, repeats collapse by ``epochs``, and
a NaN or infinite score is an error naming the case and metric (it would otherwise make the
paired bootstrap report p = 0).

The dataset hash of an imported run is a hash over the case ids and the inputs *as that
tool recorded them*. Two runs imported by the same importer from the same dataset have the
same hash and compare; a promptfoo run and an Inspect run never do, because each tool keeps
different things in its input. Compare runs imported by the same importer.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from evalproof.importers import deepeval, inspect_ai, lm_eval, promptfoo, table
from evalproof.importers._common import (
    EPOCHS,
    ImportFormatError,
    Options,
    imported_dataset_hash,
)
from evalproof.models import Run

_READERS: dict[str, Callable[[Path, Options], Run]] = {
    "promptfoo": promptfoo.read,
    "inspect": inspect_ai.read,
    "deepeval": deepeval.read,
    "lm-eval": lm_eval.read,
    "jsonl": table.read_jsonl,
    "csv": table.read_csv,
}
FORMATS = tuple(_READERS)

__all__ = ["EPOCHS", "FORMATS", "ImportFormatError", "import_run", "imported_dataset_hash"]


def import_run(
    fmt: str,
    path: str | Path,
    *,
    system: str | None = None,
    epochs: str = "mean",
    warn: Callable[[str], None] | None = None,
    fields: Mapping[str, str] | None = None,
    wide: bool = False,
) -> Run:
    """Read ``path`` as ``fmt`` and return it as a ``Run``.

    ``system`` picks one system out of a file that holds several (promptfoo, and jsonl or
    csv with a ``system`` column) and is otherwise the label to give the run. ``epochs``
    says how repeats of one case collapse: ``mean`` (the default), ``first``, ``all`` (a
    pass only if every repeat passed) or ``any``. ``warn`` receives notes about anything
    skipped or incomplete; by default they are raised as Python warnings. ``fields`` and
    ``wide`` apply to ``jsonl`` and ``csv`` only.

    Raises ``ImportFormatError`` (a ``ValueError``) when the file is not in the format
    asked for or cannot be imported as asked.
    """
    reader = _READERS.get(fmt)
    if reader is None:
        raise ImportFormatError(f"unknown format {fmt!r}; choose one of {', '.join(FORMATS)}")
    options = Options(system=system, epochs=epochs, fields=dict(fields or {}), wide=wide)
    if warn is not None:
        options = replace(options, warn=warn)
    return reader(Path(path), options)
