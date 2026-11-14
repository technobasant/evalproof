"""Read a run's repeated trials: one outcome per trial, and the counts per case.

A case that runs ``k`` times has ``k`` results. What the run report, and later the gate and
compare, need from them is the same few facts: did each trial pass, how many of a case's decided
trials passed, and is the case flaky. They are kept here so every reader counts the same way.
"""

from __future__ import annotations

import numpy as np

from evalproof.models import CaseResult, Run
from evalproof.stats import bootstrap_ci


def trial_outcome(result: CaseResult, scorer: str) -> bool | None:
    """``True`` when the trial passed ``scorer``, ``False`` when it failed, ``None`` when the
    scorer did not decide it (no score, or a continuous or undecided score). An errored trial
    is a failure whatever the scorer made of its empty output, as in ``gate.pass_counts``."""
    if result.output.error:
        return False
    score = result.scores.get(scorer)
    if score is None or score.passed is None:
        return None
    return bool(score.passed)


def is_binary(run: Run, scorer: str) -> bool:
    """Whether ``scorer`` gives a pass or fail somewhere in the run (a continuous one does not)."""
    return any(
        (s := r.scores.get(scorer)) is not None and s.passed is not None for r in run.results
    )


def case_counts(run: Run, scorer: str) -> dict[str, tuple[int, int]]:
    """``case id -> (passes, decided trials)`` for one scorer, in case order. A case with no
    decided trial is left out, and a continuous scorer has no counts."""
    counts: dict[str, tuple[int, int]] = {}
    if not is_binary(run, scorer):
        return counts
    for case_id, results in run.by_case().items():
        outcomes = [o for r in results if (o := trial_outcome(r, scorer)) is not None]
        if outcomes:
            counts[case_id] = (sum(outcomes), len(outcomes))
    return counts


def is_flaky(passes: int, decided: int) -> bool:
    """A pass and a fail among the decided trials."""
    return 0 < passes < decided


def pooled_rate_ci(counts: list[tuple[int, int]]) -> tuple[float, float]:
    """95% interval for the pooled pass rate (passes over decided trials), resampling cases.
    Trials of one case move together, so resampling trials would make the interval too narrow."""
    passes = np.array([p for p, _ in counts], dtype=np.float64)
    decided = np.array([d for _, d in counts], dtype=np.float64)

    def pooled(picked: np.ndarray) -> float:
        i = picked.astype(np.intp)
        return float(passes[i].sum() / decided[i].sum())

    return bootstrap_ci(np.arange(len(counts), dtype=np.float64), pooled)


def all_trials_identical(run: Run) -> bool:
    """Every case gave the same output (and the same error, if any) in all of its trials, and
    had more than one trial. With one trial per case there is nothing to compare."""
    groups = list(run.by_case().values())
    return bool(groups) and all(
        len(g) > 1 and len({(r.output.output, r.output.error) for r in g}) == 1 for g in groups
    )
