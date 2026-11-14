"""``evalproof rehash``: give a pre-0.5 run file the per-case hashes it lacks, without a model call.

A run written before evalproof 0.5 has no ``case_hash`` per case, so it cannot be gated against
a changed dataset: ``compare`` cannot tell the cases that stayed the same from the ones that
changed. The file does not store case inputs, but it does store ``dataset_hash``. When the
suite's dataset hashes to that value, the dataset is the one the run scored, and the hashes of
its cases are the run's. The same holds for the dataset's first cases when they hash to it: a
``--limit`` run, or a dataset with cases appended since. Otherwise nothing can be proved and
nothing is written.

Rehash proves the dataset matched, not the suite: ``suite_hash`` is left as it is, so ``verify``
still refuses a file that has none.
"""

from __future__ import annotations

from evalproof.config import SuiteConfig
from evalproof.dataset import case_hash, content_hash, dataset_hash, load_dataset
from evalproof.models import Run


class RehashError(ValueError):
    """The run cannot be matched to the suite's dataset, so its hashes cannot be recomputed."""


def rehash_run(run: Run, cfg: SuiteConfig) -> Run:
    """A copy of ``run`` with ``case_hash`` and ``content_hash`` set from the suite's dataset.

    Raises ``RehashError`` unless the dataset is the one the run scored: its hash must equal
    ``run.info.dataset_hash`` and it must hold every case the run has a result for. Scores,
    outputs and ``suite_hash`` are not touched. Running it on a file that already has hashes
    gives the same file.
    """
    cases = load_dataset(cfg.resolve(cfg.dataset))
    # A ``--limit N`` run scored the dataset's first N cases and hashed only those.
    # (A run with several trials per case holds one result per case and trial.)
    head = cases[: len({r.case_id for r in run.results})]
    if dataset_hash(head) == run.info.dataset_hash and {c.id for c in head} == {
        r.case_id for r in run.results
    }:
        cases = head
    elif dataset_hash(cases) != run.info.dataset_hash:
        raise RehashError(
            "the dataset has changed since the run was made, so its cases can't be matched to "
            "the run's results: check out the dataset as it was (for example on the base "
            "branch, before your change) and rehash there, or re-run the suite"
        )
    by_id = {c.id: c for c in cases}
    unknown = [r.case_id for r in run.results if r.case_id not in by_id]
    if unknown:
        raise RehashError(
            f"{len(unknown)} result(s) in the run are for cases the dataset does not have "
            f"(for example `{unknown[0]}`), so its cases can't be matched to the run's "
            "results: check out the dataset as it was (for example on the base branch, before "
            "your change) and rehash there, or re-run the suite"
        )
    out = run.model_copy(deep=True)
    for r in out.results:
        case = by_id[r.case_id]
        r.case_hash = case_hash(case)
        r.content_hash = content_hash(case)
    return out
