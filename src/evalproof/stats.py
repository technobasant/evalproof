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


PERMUTATION_DRAWS = 20_000  # sign draws of the Monte Carlo test
EXACT_ENUMERATION_MAX = 16  # most nonzero differences that are enumerated (2**16 sign patterns)
# |S*| >= |S| - tolerance counts as extreme, so ties are not lost to rounding
_TIE_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SignFlip:
    """The result of ``sign_flip_test``: ``method`` is ``"exact"`` (``draws`` is 0) or
    ``"monte carlo"`` (``draws`` sign draws)."""

    p_value: float
    method: str
    draws: int = 0


def _integer_dp_p(steps: list[int], observed: int) -> float:
    """Exact ``P(|S*| >= |observed|)`` for ``S* = sum(±step)`` with equal-odds signs, by dynamic
    programming over the integer sums.

    ``S* = 2W - T`` where ``T`` is the sum of the steps and ``W`` the sum of those given a plus
    sign, so it tracks the distribution of ``W`` one step at a time. The probabilities are floats
    (halving is exact until 2**-53, and the error after that is far below anything a p-value is
    compared with), which keeps a suite of thousands of cases fast.
    """
    total = sum(steps)
    dist = np.zeros(total + 1, dtype=np.float64)
    dist[0] = 1.0
    for step in steps:
        shifted = np.zeros_like(dist)
        shifted[step:] = dist[: total + 1 - step]
        dist = 0.5 * (dist + shifted)
    sums = 2 * np.arange(total + 1) - total
    return float(min(1.0, dist[np.abs(sums) >= abs(observed)].sum()))


def sign_flip_test(
    diffs: Sequence[float] | FloatArray,
    *,
    grid: int | None = None,
    n_draws: int = PERMUTATION_DRAWS,
    seed: int = 0,
) -> SignFlip:
    """Two-sided paired sign-flip permutation test on per-case differences.

    Under the null hypothesis that the two systems are alike, each difference is as likely to be
    positive as negative. The statistic is the sum ``S`` of the differences and the p-value is
    ``P(|S*| >= |S| - 1e-12)`` over random sign assignments ``S*``:

    * every difference zero: ``p = 1``;
    * ``grid`` given and every ``d * grid`` a whole number (pass/fail rates over ``grid`` trials):
      exact, by dynamic programming over integer sums, for any number of cases;
    * otherwise exact by enumerating all sign patterns when at most 16 differences are nonzero;
    * otherwise Monte Carlo: ``n_draws`` seeded sign draws, ``p = (hits + 1) / (draws + 1)``,
      which is never 0 and never anti-conservative.
    """
    d = np.asarray(diffs, dtype=np.float64)
    if d.size == 0:
        raise ValueError("no differences")
    nonzero = d[np.abs(d) > _TIE_TOLERANCE]
    if nonzero.size == 0:
        return SignFlip(1.0, "exact")
    if grid is not None and grid > 0:
        scaled = nonzero * grid
        whole = np.rint(scaled)
        if np.allclose(scaled, whole, rtol=0.0, atol=1e-9):
            steps = [int(abs(x)) for x in whole]
            return SignFlip(_integer_dp_p(steps, int(whole.sum())), "exact")
    observed = abs(float(nonzero.sum()))
    m = int(nonzero.size)
    if m <= EXACT_ENUMERATION_MAX:
        patterns = np.arange(1 << m)[:, None] >> np.arange(m)[None, :] & 1
        sums = (1.0 - 2.0 * patterns) @ nonzero
        hits = int(np.sum(np.abs(sums) >= observed - _TIE_TOLERANCE))
        return SignFlip(hits / (1 << m), "exact")
    rng = np.random.default_rng(seed)
    hits = 0
    drawn = 0
    chunk = max(1, 2_000_000 // m)  # bounds the sign matrix, however many cases there are
    while drawn < n_draws:
        size = min(chunk, n_draws - drawn)
        signs = rng.integers(0, 2, size=(size, m), dtype=np.int8)
        sums = (1.0 - 2.0 * signs) @ nonzero
        hits += int(np.sum(np.abs(sums) >= observed - _TIE_TOLERANCE))
        drawn += size
    return SignFlip((hits + 1) / (n_draws + 1), "monte carlo", n_draws)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm–Bonferroni adjusted p-values, in the order given.

    Step-down: sort ascending, multiply the i-th smallest (0-based) by ``m - i``, carry the
    running maximum so the adjusted values never decrease with the raw ones, cap at 1.
    Rejecting every hypothesis whose adjusted p is below alpha keeps the family-wise error
    rate at alpha under any dependence between the tests, and is never less powerful than
    plain Bonferroni (``min(1, m * p)``).
    """
    if any(math.isnan(p) or not 0.0 <= p <= 1.0 for p in pvalues):
        raise ValueError("p-values must be in [0, 1]")
    m = len(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: pvalues[j])):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))
