"""Small, readable statistics. numpy and the standard library only — no SciPy.

Every function here has a known-value test in ``tests/test_stats.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def z_for(alpha: float) -> float:
    """Two-sided critical value: ``z_for(0.05)`` ≈ 1.959964."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    return NormalDist().inv_cdf(1 - alpha / 2)


def wilson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Behaves at 0/n and n/n."""
    if n <= 0:
        return (0.0, 1.0)
    if not 0 <= k <= n:
        raise ValueError("need 0 <= k <= n")
    z = z_for(alpha)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At k=0 the lower bound is exactly 0 (centre == half), and at k=n the upper
    # bound is exactly 1. Computing them by subtraction leaves platform-dependent
    # rounding (2.8e-17 on x86_64 Linux, 0.0 on arm64 macOS), so pin them.
    lo = 0.0 if k == 0 else max(0.0, centre - half)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return (lo, hi)


def bootstrap_ci(
    values: Sequence[float] | FloatArray,
    stat: Callable[[FloatArray], float] = lambda x: float(np.mean(x)),
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval, seeded so a report is reproducible."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("no values")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.array([stat(arr[row]) for row in idx])
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    return (float(lo), float(hi))


@dataclass(frozen=True)
class PairedDelta:
    """``delta = mean(candidate) - mean(baseline)`` over paired cases."""

    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    n: int


def paired_bootstrap(
    baseline: Sequence[float] | FloatArray,
    candidate: Sequence[float] | FloatArray,
    *,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PairedDelta:
    """Resample *case indices*, so each replicate keeps baseline and candidate paired.

    The two-sided p-value is the percentile-bootstrap approximation
    ``2 * min(P(d* <= 0), P(d* >= 0))`` — adequate for gating, and stated as such.
    """
    a = np.asarray(baseline, dtype=np.float64)
    b = np.asarray(candidate, dtype=np.float64)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("need two non-empty arrays of equal length")
    diff = b - a
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diff.size, size=(n_boot, diff.size))
    reps = diff[idx].mean(axis=1)
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    if np.all(diff == 0):
        p = 1.0
    else:
        p = float(min(1.0, 2 * min(np.mean(reps <= 0), np.mean(reps >= 0))))
    return PairedDelta(float(diff.mean()), float(lo), float(hi), p, int(diff.size))


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    """``P(X <= k)`` for ``X ~ Binomial(n, p)``, summed exactly and rounded once.

    In floating point, ``comb(n, i)`` alone passes the largest float from about
    n = 1,030, so a comparison with that many discordant pairs raised OverflowError
    instead of returning a p-value. Integers and fractions do not overflow.
    """
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    q = Fraction(p)
    if q == Fraction(1, 2):
        # The McNemar case: every outcome has probability 2**-n, so sum the binomial
        # coefficients as integers and divide once (int / int is correctly rounded).
        term = total = 1
        for i in range(1, k + 1):
            term = term * (n - i + 1) // i
            total += term
        return total / (1 << n)
    return float(sum(math.comb(n, i) * q**i * (1 - q) ** (n - i) for i in range(k + 1)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs.

    ``b`` = baseline passed, candidate failed; ``c`` = baseline failed, candidate passed.
    Only discordant pairs carry information about a difference between the two systems.
    """
    if b < 0 or c < 0:
        raise ValueError("counts must be non-negative")
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(b, c), n))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
