"""How big a drop a suite can detect: informational, never part of a gate decision.

A gate that passes says "no regression was proven", which is weak when the suite is too small to
prove one. This puts a number on that: the smallest true drop the paired test would catch 80% of
the time (the minimum detectable effect), and how many cases it would take to catch ``max_drop``.
The normal approximation is used, which is adequate for a rule of thumb and is said to be one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

from evalproof.compare import MetricComparison
from evalproof.stats import z_for

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


def estimate_power(mc: MetricComparison, max_drop: float, alpha_used: float) -> Power:
    """Power of the paired comparison of ``mc`` at ``alpha_used``, for a drop of ``max_drop``.

    ``mde = (z(1 - a/2) + z(0.8)) * s / sqrt(n)`` with ``s`` the sample standard deviation of the
    paired per-case differences. When that is zero or unknown (runs that agree on every case, or
    a single case) it is ``sqrt(2 v)`` with ``v`` the variance of the baseline's per-case values,
    which is the spread two independent systems like the baseline would show.
    """
    n = mc.n_paired
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
    note = (
        f"With {n} cases this gate reliably detects drops of about {_amount(mc, mde)} (80% power)"
    )
    if needed is not None and mde > max_drop:
        note += f"; about {needed} cases are needed to detect max_drop ({_amount(mc, max_drop)})"
    return Power(mde, needed, n, alpha_used, note + ".")
