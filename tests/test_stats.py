"""Known-value tests: every expected number below is worked by hand in the comment."""

from __future__ import annotations

import math
from statistics import NormalDist

import pytest

from evalproof.stats import (
    binom_cdf,
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap,
    percentile,
    wilson,
    z_for,
)


def test_z_for_95() -> None:
    assert z_for(0.05) == pytest.approx(1.959964, abs=1e-6)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1])
def test_z_for_rejects_bad_alpha(alpha: float) -> None:
    with pytest.raises(ValueError):
        z_for(alpha)


def test_wilson_zero_of_ten() -> None:
    # z²=3.8415; denom=1.38415; centre=(3.8415/20)/1.38415=0.13876; half=0.13876
    lo, hi = wilson(0, 10)
    assert lo == 0.0
    assert hi == pytest.approx(0.27753, abs=1e-4)


def test_wilson_half() -> None:
    # p=.5, n=10: centre .5, half = 1.96*sqrt(.025+.0096037)/1.38415 = .26336
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-4)
    assert hi == pytest.approx(0.76341, abs=1e-4)


def test_wilson_degenerate_and_invalid() -> None:
    assert wilson(0, 0) == (0.0, 1.0)
    with pytest.raises(ValueError):
        wilson(11, 10)


def test_binom_cdf() -> None:
    # P(X<=2), X~Bin(10,.5) = (1+10+45)/1024
    assert binom_cdf(2, 10) == pytest.approx(56 / 1024)


@pytest.mark.parametrize(
    ("b", "c", "expected"),
    [
        (0, 5, 2 / 32),  # 2 * .5^5
        (2, 8, 2 * 56 / 1024),  # 2 * P(X<=2 | n=10)
        (5, 5, 1.0),  # symmetric → capped at 1
        (0, 0, 1.0),  # no discordant pairs, no evidence
    ],
)
def test_mcnemar_exact(b: int, c: int, expected: float) -> None:
    assert mcnemar_exact(b, c) == pytest.approx(expected)


def test_binom_cdf_other_p_and_edges() -> None:
    # P(X<=1), X~Bin(3,.25) = .75^3 + 3·.25·.75^2
    assert binom_cdf(1, 3, 0.25) == pytest.approx(0.84375)
    assert binom_cdf(-1, 10) == 0.0
    assert binom_cdf(10, 10) == 1.0


def test_mcnemar_exact_many_discordant_pairs() -> None:
    # Past ~1,030 discordant pairs comb(n, i) no longer fits in a float; this raised
    # OverflowError. The exact p-value must stay finite, symmetric and close to the
    # continuity-corrected normal approximation at this size.
    assert mcnemar_exact(600, 600) == 1.0
    p = mcnemar_exact(540, 560)
    assert p == mcnemar_exact(560, 540)
    z = (abs(540 - 560) - 1) / math.sqrt(1100)
    assert p == pytest.approx(2 * (1 - NormalDist().cdf(z)), rel=0.01)
    assert mcnemar_exact(500, 700) < 1e-8


def test_mcnemar_rejects_negative() -> None:
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 2)


def test_paired_bootstrap_identical_runs() -> None:
    res = paired_bootstrap([1, 0, 1, 1], [1, 0, 1, 1])
    assert res.delta == 0.0
    assert res.p_value == 1.0
    assert (res.ci_low, res.ci_high) == (0.0, 0.0)


def test_paired_bootstrap_total_regression() -> None:
    res = paired_bootstrap([1.0] * 30, [0.0] * 30)
    assert res.delta == -1.0
    assert res.ci_low == res.ci_high == -1.0
    assert res.p_value == 0.0
    assert res.n == 30


def test_paired_bootstrap_is_seeded() -> None:
    a = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]
    b = [1, 1, 0, 1, 0, 0, 0, 1, 1, 0]
    assert paired_bootstrap(a, b, seed=7) == paired_bootstrap(a, b, seed=7)


def test_paired_bootstrap_validates() -> None:
    with pytest.raises(ValueError):
        paired_bootstrap([1, 0], [1])


def test_bootstrap_ci_brackets_mean() -> None:
    values = [float(i) for i in range(100)]
    lo, hi = bootstrap_ci(values, seed=1)
    assert lo < 49.5 < hi
    with pytest.raises(ValueError):
        bootstrap_ci([])


def test_percentile() -> None:
    assert percentile([], 95) == 0.0
    assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0
    assert math.isclose(percentile([0.0, 100.0], 95), 95.0)


def test_wilson_all_of_ten_is_exactly_one() -> None:
    # Mirror of 0/10: the upper bound is exactly 1, not 1 - rounding error.
    lo, hi = wilson(10, 10)
    assert hi == 1.0
    assert lo == pytest.approx(1 - 0.27753, abs=1e-4)
