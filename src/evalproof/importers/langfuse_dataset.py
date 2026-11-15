"""Read Langfuse dataset items as an evalproof dataset (``evalproof import-dataset langfuse``).

An item has ``id``, ``input``, ``expected_output`` (or ``expectedOutput``) and ``metadata``. It maps
exactly as a DeepEval golden does (``deepeval_dataset``), except that the case id is the item's
``id``. The export endpoint and file shape were not confirmed against a real file, so this reader
is experimental: it takes a JSON list, a JSONL file or a CSV with those keys.
"""

from __future__ import annotations

from pathlib import Path

from evalproof.importers._dataset import DatasetOptions, ImportedDataset, reject_fields
from evalproof.importers.deepeval_dataset import read_goldens

FORMAT = "langfuse"


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    reject_fields(opts, FORMAT)
    return read_goldens(path, FORMAT, id_keys=("id", "name"))
