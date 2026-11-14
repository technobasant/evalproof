"""The paired sign-flip permutation test, and the power estimate that sits beside the gate."""

from __future__ import annotations

import itertools
import time
from math import comb

import numpy as np
import pytest

from evalproof.compare import MetricComparison
from evalproof.power import estimate_power
from evalproof.stats import (
    EXACT_ENUMERATION_MAX,
    PERMUTATION_DRAWS,
    sign_flip_test,
    z_for,
)


def _brute_force(diffs: list[float]) -> float:
    """``P(|S*| >= |S|)`` over every sign assignment, summed the slow and obvious way."""
    observed = abs(sum(diffs))
    hits = sum(
        abs(sum(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12
        for signs in itertools.product((1, -1), repeat=len(diffs))
    )
    return hits / 2 ** len(diffs)


# --- the exact paths ---------------------------------------------------------------------------


def test_all_zero_differences_give_p_one_exactly() -> None:
    result = sign_flip_test([0.0, 0.0, 0.0])
    assert (result.p_value, result.method, result.draws) == (1.0, "exact", 0)


def test_no_differences_is_an_error() -> None:
    with pytest.raises(ValueError, match="no differences"):
        sign_flip_test([])


@pytest.mark.parametrize(
    "diffs",
    [
        [1.0],
        [0.5, -0.25],
        [1 / 3, 1 / 3, -1 / 3, 2 / 3],
        [0.1, 0.2, 0.3, -0.15, 0.4, 0.0, 0.0],
        [-1.0] * 5 + [1.0] * 2,
        [0.7, -0.2, 0.31, 0.9, -0.05, 0.12, 0.44, 0.6],
    ],
)
def test_enumeration_matches_brute_force(diffs: list[float]) -> None:
    result = sign_flip_test(diffs)
    assert result.method == "exact"
    assert result.p_value == pytest.approx(_brute_force(diffs), abs=1e-12)


def test_the_enumeration_limit_is_sixteen_nonzero_differences() -> None:
    assert EXACT_ENUMERATION_MAX == 16
    rng = np.random.default_rng(7)
    sixteen = list(rng.normal(size=16))
    assert sign_flip_test(sixteen).method == "exact"
    assert sign_flip_test([*sixteen, 0.0, 0.0]).method == "exact"  # zeros are not counted
    assert sign_flip_test([*sixteen, 0.3]).method == "monte carlo"


@pytest.mark.parametrize("grid", [2, 3, 5])
def test_the_integer_dp_matches_brute_force_on_a_grid(grid: int) -> None:
    rng = np.random.default_rng(grid)
    diffs = [float(x) / grid for x in rng.integers(-grid, grid + 1, size=12)]
    result = sign_flip_test(diffs, grid=grid)
    assert result.method == "exact"
    assert result.draws == 0
    assert result.p_value == pytest.approx(_brute_force(diffs), abs=1e-9)


def test_the_dp_gives_the_same_answer_as_enumeration() -> None:
    diffs = [1 / 3, -2 / 3, 1.0, 1 / 3, -1 / 3, 2 / 3, 1 / 3, 1.0, -1 / 3, 1 / 3]
    assert sign_flip_test(diffs, grid=3).p_value == pytest.approx(
        sign_flip_test(diffs).p_value, abs=1e-12
    )


def test_the_dp_is_exact_for_a_large_suite() -> None:
    """Hundreds of cases are far past enumeration; the DP stays exact, and fast."""
    n = 600
    diffs = [1 / 5] * 40 + [-1 / 5] * 10 + [0.0] * (n - 50)
    start = time.perf_counter()
    result = sign_flip_test(diffs, grid=5)
    assert time.perf_counter() - start < 2.0
    assert result.method == "exact"
    # 40 plus steps and 10 minus: S = 30 steps of 1; the sign-flip distribution of 50 unit steps
    # is binomial, so the two-sided p-value is P(|2W - 50| >= 30).
    expected = sum(comb(50, w) for w in range(51) if abs(2 * w - 50) >= 30) / 2**50
    assert result.p_value == pytest.approx(expected, rel=1e-9)


def test_a_grid_that_does_not_fit_falls_back_to_the_other_methods() -> None:
    # 0.1 is not a multiple of 1/3, so the DP is not used; three nonzero differences enumerate.
    result = sign_flip_test([0.1, 0.2, -0.05], grid=3)
    assert result.method == "exact"
    assert result.p_value == pytest.approx(_brute_force([0.1, 0.2, -0.05]), abs=1e-12)


def test_ties_are_counted_as_extreme() -> None:
    # Of the four sign patterns of [1, 1], two give |S*| = 2 (the observed pattern and its
    # mirror), and the observed one must count, so p is 2/4 and not 1/4.
    assert sign_flip_test([1.0, 1.0]).p_value == pytest.approx(0.5)
    # Floating point must not break the tie: 0.1 + 0.2 vs 0.3 differ in the last bit.
    assert sign_flip_test([0.1, 0.2, -0.3]).p_value == pytest.approx(
        _brute_force([0.1, 0.2, -0.3]), abs=1e-12
    )


# --- the Monte Carlo path ----------------------------------------------------------------------


def _wide(n: int = 40, seed: int = 3) -> list[float]:
    rng = np.random.default_rng(seed)
    return list(rng.normal(0.2, 1.0, size=n))


def test_monte_carlo_is_seeded_and_reproducible() -> None:
    diffs = _wide()
    a, b = sign_flip_test(diffs, seed=11), sign_flip_test(diffs, seed=11)
    assert a == b
    assert a.method == "monte carlo" and a.draws == PERMUTATION_DRAWS == 20_000
    assert sign_flip_test(diffs, seed=12).p_value != a.p_value


def test_monte_carlo_p_is_hits_plus_one_over_draws_plus_one() -> None:
    diffs = _wide()
    result = sign_flip_test(diffs, n_draws=1000, seed=0)
    assert result.draws == 1000
    hits = round(result.p_value * 1001) - 1
    assert 0 <= hits <= 1000
    assert result.p_value == pytest.approx((hits + 1) / 1001)


def test_monte_carlo_is_never_zero_even_for_an_overwhelming_effect() -> None:
    result = sign_flip_test([1.0] * 40, n_draws=500)
    assert result.method == "monte carlo"
    assert result.p_value == pytest.approx(1 / 501)  # only the observed pattern, counted once


def test_monte_carlo_agrees_with_the_exact_answer_where_both_are_known() -> None:
    n = 20
    diffs = [1.0] * 15 + [-1.0] * 5  # exact: P(|2W - 20| >= 10) for a Binomial(20, 1/2)
    exact = sum(comb(n, w) for w in range(n + 1) if abs(2 * w - n) >= 10) / 2**n
    result = sign_flip_test(diffs)
    assert result.method == "monte carlo"
    assert abs(result.p_value - exact) < 0.01


def test_monte_carlo_is_chunked_so_a_huge_suite_does_not_allocate_it_all() -> None:
    rng = np.random.default_rng(1)
    diffs = rng.normal(0.0, 1.0, size=5000)
    result = sign_flip_test(diffs, n_draws=2000)
    assert result.draws == 2000
    assert 0.0 < result.p_value <= 1.0


# --- power -------------------------------------------------------------------------------------


def _mc(
    kind: str = "rate",
    *,
    n: int = 100,
    diff_sd: float | None = 0.3,
    baseline_var: float | None = 0.2,
) -> MetricComparison:
    return MetricComparison(
        name="m",
        kind=kind,
        n_paired=n,
        baseline=0.8,
        candidate=0.7,
        delta=-0.1,
        ci_low=-0.2,
        ci_high=0.0,
        p_value=0.1,
        test="t",
        diff_sd=diff_sd,
        baseline_var=baseline_var,
    )


def test_mde_is_the_normal_approximation_at_eighty_percent_power() -> None:
    power = estimate_power(_mc(n=100, diff_sd=0.3), max_drop=0.02, alpha_used=0.05)
    z = z_for(0.05) + 0.8416212335729143
    assert power.mde == pytest.approx(z * 0.3 / 10)
    assert power.cases_needed == int(np.ceil((z * 0.3 / 0.02) ** 2))
    assert (power.n, power.alpha_used) == (100, 0.05)
    assert "100 cases" in power.note and "80% power" in power.note
    assert "cases are needed to detect max_drop" in power.note  # the mde is above max_drop


def test_the_note_stays_short_when_the_suite_is_big_enough() -> None:
    power = estimate_power(_mc(n=10_000, diff_sd=0.05), max_drop=0.05, alpha_used=0.05)
    assert power.mde is not None and power.mde < 0.05
    assert "needed" not in power.note


def test_alpha_is_shared_so_a_smaller_alpha_needs_a_bigger_effect() -> None:
    wide = estimate_power(_mc(), 0.02, 0.05).mde
    narrow = estimate_power(_mc(), 0.02, 0.05 / 6).mde
    assert wide is not None and narrow is not None and narrow > wide


def test_a_zero_diff_sd_falls_back_to_the_baseline_spread() -> None:
    power = estimate_power(_mc(n=50, diff_sd=0.0, baseline_var=0.08), 0.02, 0.05)
    z = z_for(0.05) + 0.8416212335729143
    assert power.mde == pytest.approx(z * np.sqrt(2 * 0.08) / np.sqrt(50))


def test_a_uniform_baseline_cannot_be_estimated() -> None:
    power = estimate_power(_mc(diff_sd=0.0, baseline_var=0.0), 0.02, 0.05)
    assert power.mde is None and power.cases_needed is None
    assert "baseline is uniform (all pass or all fail)" in power.note
    cont = estimate_power(_mc("continuous", diff_sd=None, baseline_var=0.0), 0.02, 0.05)
    assert "every case has the same score" in cont.note


def test_one_paired_case_cannot_be_estimated() -> None:
    power = estimate_power(_mc(n=1, diff_sd=None, baseline_var=None), 0.02, 0.05)
    assert power.mde is None and "one paired case" in power.note


def test_units_follow_the_kind() -> None:
    assert "points" in estimate_power(_mc("binary"), 0.02, 0.05).note
    assert "points" in estimate_power(_mc("rate"), 0.02, 0.05).note
    assert "points" not in estimate_power(_mc("continuous"), 0.02, 0.05).note
