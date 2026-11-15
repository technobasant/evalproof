"""How big a drop a suite can detect: informational, never part of a gate decision.

A gate that passes says "no regression was proven", which is weak when the suite is too small to
prove one. This puts a number on that: the smallest true drop the paired test would catch 80% of
the time (the minimum detectable effect), and how many cases it would take to catch ``max_drop``.

Two bounds are used. The normal approximation gives the 80%-power figure, adequate for a rule of
thumb and said to be one. A binary metric compared with the exact McNemar test (one trial per
case, or repeated trials combined with ``all`` or ``any``) has a floor the approximation does not
see: the test needs ``k`` cases worse and none better (``k`` from alpha alone), so no drop under
``k / n`` is ever significant. The reported figure is the larger of the two bounds. Other
repeated-trial comparisons (a permutation test) and continuous metrics use the normal
approximation alone.

The floor is taken at the gate's own alpha, not the share of it the approximation uses: with
several gated metrics the gate compares Holm-adjusted p-values with alpha, and Holm can let a
metric through at a raw p up to alpha itself, so only ``k(alpha)`` is a floor that always holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from evalproof.compare import MCNEMAR_TEST, MetricComparison
from evalproof.stats import min_significant_discordant, z_for

POWER = 0.8


@dataclass
class Power:
    mde: float | None  # the drop detected with 80% power; None when it cannot be estimated
    cases_needed: int | None  # cases to detect ``max_drop`` with 80% power
    n: int  # paired cases
    alpha_used: float  # alpha after dividing among the gated metrics (and candidates)
    note: str


def _amount(mc: MetricComparison, x: float) -> str:
    return f"{100 * x:.1f} points" if mc.kind in ("binary", "rate") else f"{x:.3f}"


def _cases_for(k: int, max_drop: float) -> int | None:
    """Cases at which ``k`` one-way discordant cases are a drop of ``max_drop``."""
    return math.ceil(k / max_drop - 1e-9) if max_drop > 0 else None


def estimate_power(
    mc: MetricComparison,
    max_drop: float,
    alpha_used: float,
    floor_alpha: float | None = None,
) -> Power:
    """Power of the paired comparison of ``mc`` at ``alpha_used``, for a drop of ``max_drop``.

    ``mde = (z(1 - a/2) + z(0.8)) * s / sqrt(n)`` with ``s`` the sample standard deviation of the
    paired per-case differences. When that is zero or unknown (runs that agree on every case, or
    a single case) it is ``sqrt(2 v)`` with ``v`` the variance of the baseline's per-case values,
    which is the spread two independent systems like the baseline would show.

    For a binary metric tested with the exact McNemar test the exact floor ``k / n`` applies too
    (``k`` is ``min_significant_discordant(floor_alpha)``; ``floor_alpha`` is the gate's alpha
    before any division, and defaults to ``alpha_used``): the mde is the larger of the two, the
    cases needed at least ``k / max_drop``, and a suite with fewer than ``k`` cases is told that
    no drop can be significant. A uniform baseline still has the floor, so it is reported then.
    """
    n = mc.n_paired
    level = alpha_used if floor_alpha is None else floor_alpha
    floor_k: int | None = None
    if mc.kind == "binary" and mc.test == MCNEMAR_TEST and n >= 1 and 0 < level < 1:
        floor_k = min_significant_discordant(level)
        if floor_k > n:
            needed = _cases_for(floor_k, max_drop)
            note = (
                f"With {n} cases no drop can reach significance: the exact test needs at least "
                f"{floor_k} cases worse (and none better) at α={level:.3g}, more than the "
                "suite has"
            )
            if needed is not None:
                note += f"; at least {needed} cases are needed to detect max_drop"
                note += f" ({_amount(mc, max_drop)})"
            return Power(None, needed, n, alpha_used, note + ".")
    if n < 2:
        return Power(None, None, n, alpha_used, "one paired case: power cannot be estimated")
    spread = mc.diff_sd if mc.diff_sd is not None and mc.diff_sd > 0 else None
    if spread is None:
        variance = mc.baseline_var
        if variance is None or variance <= 0:
            what = (
                "all pass or all fail"
                if mc.kind in ("binary", "rate")
                else "every case has the same score"
            )
            if floor_k is not None:
                floor = floor_k / n
                needed = _cases_for(floor_k, max_drop)
                note = (
                    f"With {n} cases the smallest drop the exact test can call significant is "
                    f"{_amount(mc, floor)} ({floor_k} cases worse, none better); the baseline is "
                    f"uniform ({what}), so power at that size is not estimated"
                )
                if needed is not None and floor > max_drop:
                    note += (
                        f"; at least {needed} cases are needed to detect max_drop "
                        f"({_amount(mc, max_drop)})"
                    )
                return Power(floor, needed, n, alpha_used, note + ".")
            return Power(
                None,
                None,
                n,
                alpha_used,
                f"baseline is uniform ({what}); power cannot be estimated",
            )
        spread = math.sqrt(2 * variance)
    z = z_for(alpha_used) + NormalDist().inv_cdf(POWER)
    mde = z * spread / math.sqrt(n)
    needed = math.ceil((z * spread / max_drop) ** 2) if max_drop > 0 else None
    floor_binds = False
    if floor_k is not None:
        floor_needed = _cases_for(floor_k, max_drop)
        if needed is not None and floor_needed is not None:
            needed = max(needed, floor_needed)
        floor_binds = floor_k / n > mde
        mde = max(mde, floor_k / n)
    if floor_binds and floor_k is not None:
        note = (
            f"With {n} cases the exact test cannot call a drop smaller than "
            f"{_amount(mc, mde)} significant ({floor_k} cases worse, none better)"
        )
    else:
        note = (
            f"With {n} cases this gate reliably detects drops of about {_amount(mc, mde)} "
            "(80% power)"
        )
    if needed is not None and mde > max_drop:
        note += f"; about {needed} cases are needed to detect max_drop ({_amount(mc, max_drop)})"
    return Power(mde, needed, n, alpha_used, note + ".")
