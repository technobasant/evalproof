"""The paired sign-flip permutation test, and the power estimate that sits beside the gate."""

from __future__ import annotations

import itertools
import time
from math import comb

import numpy as np
import pytest

from evalproof.compare import MCNEMAR_TEST, MetricComparison, compare_runs
from evalproof.config import GateConfig
from evalproof.gate import decide
from evalproof.power import estimate_power
from evalproof.stats import (
    EXACT_ENUMERATION_MAX,
    PERMUTATION_DRAWS,
    mcnemar_exact,
    min_significant_discordant,
    sign_flip_test,
    z_for,
)
from tests.conftest import make_run


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
    test: str = "t",
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
        test=test,
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


# --- the exact floor under the minimum detectable drop -----------------------------------------


def _exact(n: int, *, diff_sd: float | None = 0.3, baseline_var: float | None = 0.2):
    """A binary comparison tested with the exact McNemar test, as the gate runs it."""
    return _mc("binary", n=n, diff_sd=diff_sd, baseline_var=baseline_var, test=MCNEMAR_TEST)


@pytest.mark.parametrize(
    ("alpha", "k"),
    [
        (0.5, 3),  # 2 * 0.5**2 = 0.5 is not below 0.5: the comparison is strict, as the gate's is
        (0.9, 2),
        (0.1, 5),
        (0.0626, 5),  # 2 * 0.5**5 = 0.0625
        (0.0625, 6),  # ... and not below itself
        (0.05, 6),
        (0.05 / 2, 7),
        (0.05 / 6, 8),
        (0.01, 8),
        (0.05 / 12, 9),
        (0.001, 11),
    ],
)
def test_min_significant_discordant_is_the_first_k_the_gate_would_call_significant(
    alpha: float, k: int
) -> None:
    assert min_significant_discordant(alpha) == k
    assert mcnemar_exact(k, 0) < alpha
    assert mcnemar_exact(k - 1, 0) >= alpha or k == 1


def test_min_significant_discordant_rejects_an_alpha_outside_zero_one() -> None:
    for bad in (0.0, 1.0, -0.1, 2.0):
        with pytest.raises(ValueError, match="alpha"):
            min_significant_discordant(bad)


def test_the_floor_is_the_smallest_drop_the_gate_can_actually_call_significant() -> None:
    """Six cases worse (and none better) is the first the gate fails at alpha 0.05; five is not."""
    n = 42
    gate = GateConfig(metric="m", alpha=0.05, max_drop=0.02)
    base = make_run({f"c{i:03d}": {"m": True} for i in range(n)}, system="base")
    for worse, significant in ((5, False), (6, True)):
        cand = make_run({f"c{i:03d}": {"m": i >= worse} for i in range(n)}, system="cand")
        cmp = compare_runs(base, cand)
        assert cmp.metrics["m"].test == MCNEMAR_TEST
        assert (cmp.metrics["m"].p_value < gate.alpha) is significant
        decision = decide(cmp, gate)
        assert (decision.exit_code != 0) is significant
        assert decision.power["m"].mde >= 6 / n  # never below the floor, whatever the data say


def test_n42_binary_exact_floor_is_six_cases_and_binds_over_the_normal_figure() -> None:
    power = estimate_power(_exact(42), max_drop=0.02, alpha_used=0.05)
    normal = (z_for(0.05) + 0.8416212335729143) * 0.3 / np.sqrt(42)
    assert normal < 6 / 42  # the approximation alone would have said about 13 points
    assert power.mde == pytest.approx(6 / 42)
    assert power.cases_needed == int(np.ceil((z_for(0.05) + 0.8416212335729143) ** 2 * 0.09 / 4e-4))
    assert power.note == (
        "With 42 cases the exact test cannot call a drop smaller than 14.3 points significant "
        "(6 cases worse, none better); about 1766 cases are needed to detect max_drop "
        "(2.0 points)."
    )


def test_a_uniform_baseline_reports_the_floor_instead_of_giving_up() -> None:
    power = estimate_power(_exact(42, diff_sd=0.0, baseline_var=0.0), 0.02, 0.05)
    assert power.mde == pytest.approx(6 / 42)
    assert power.cases_needed == 300  # ceil(6 / 0.02)
    assert power.note == (
        "With 42 cases the smallest drop the exact test can call significant is 14.3 points "
        "(6 cases worse, none better); the baseline is uniform (all pass or all fail), so power "
        "at that size is not estimated; at least 300 cases are needed to detect max_drop "
        "(2.0 points)."
    )


def test_a_uniform_baseline_does_not_ask_for_more_cases_when_the_floor_is_under_max_drop() -> None:
    power = estimate_power(_exact(1000, diff_sd=0.0, baseline_var=0.0), 0.02, 0.05)
    assert power.mde == pytest.approx(6 / 1000)
    assert power.cases_needed == 300 and "needed" not in power.note
    assert "power at that size is not estimated" in power.note


def test_no_drop_can_be_significant_when_the_suite_has_fewer_cases_than_the_floor() -> None:
    for n, kw in ((4, {}), (4, {"diff_sd": 0.0, "baseline_var": 0.0})):
        power = estimate_power(_exact(n, **kw), 0.02, 0.05)
        assert power.mde is None and power.cases_needed == 300
        assert power.note == (
            f"With {n} cases no drop can reach significance: the exact test needs at least 6 "
            "cases worse (and none better) at α=0.05, more than the suite has; at least 300 "
            "cases are needed to detect max_drop (2.0 points)."
        )
    # Exactly the floor is enough: six cases can all be worse.
    six = estimate_power(_exact(6), 0.02, 0.05)
    assert six.mde == pytest.approx(1.0) and "no drop can reach" not in six.note


def test_a_shared_alpha_raises_the_floor() -> None:
    tight = {"diff_sd": 0.05}  # a normal figure far under the floor, so the floor shows
    assert estimate_power(_exact(100, **tight), 0.02, 0.05).mde == pytest.approx(0.06)  # k is 6
    assert estimate_power(_exact(100, **tight), 0.02, 0.05 / 6).mde == pytest.approx(0.08)  # k is 8


def test_floor_alpha_sets_the_floor_apart_from_the_alpha_of_the_normal_figure() -> None:
    tight = {"diff_sd": 0.05}
    split = estimate_power(_exact(100, **tight), 0.02, 0.05 / 2, floor_alpha=0.05)
    assert split.mde == pytest.approx(0.06)  # k is 6 at 0.05, not 7 at 0.025
    assert "(6 cases worse, none better)" in split.note
    few = estimate_power(_exact(4), 0.02, 0.05 / 2, floor_alpha=0.05)
    assert "at α=0.05" in few.note


def test_with_two_gated_metrics_the_floor_uses_the_gates_alpha() -> None:
    """Holm can pass a metric at a raw p up to alpha, so the floor is k at alpha, not alpha / 2."""
    n = 50
    base = make_run({f"c{i:03d}": {"a": True, "b": True} for i in range(n)}, system="base")
    cand = make_run({f"c{i:03d}": {"a": True, "b": True} for i in range(n)}, system="cand")
    decision = decide(compare_runs(base, cand), GateConfig(metrics=["a", "b"], max_drop=0.02))
    for name in ("a", "b"):
        assert decision.power[name].mde == pytest.approx(6 / n)  # 12 points; 7 / 50 would be 14


def test_the_normal_figure_still_wins_when_it_is_larger() -> None:
    power = estimate_power(_exact(100, diff_sd=0.9), max_drop=0.02, alpha_used=0.05)
    normal = (z_for(0.05) + 0.8416212335729143) * 0.9 / 10
    assert normal > 6 / 100
    assert power.mde == pytest.approx(normal)
    assert "this gate reliably detects drops of about" in power.note
    assert "80% power" in power.note and "exact test cannot" not in power.note
    z = z_for(0.05) + 0.8416212335729143
    assert power.cases_needed == int(np.ceil((z * 0.9 / 0.02) ** 2))


def test_the_floor_does_not_apply_to_rate_continuous_or_a_non_exact_binary_test() -> None:
    for kind, test in (("rate", "t"), ("rate", "sign-flip permutation"), ("continuous", "t")):
        mc = _mc(kind, n=42, test=test)
        power = estimate_power(mc, 0.02, 0.05)
        z = z_for(0.05) + 0.8416212335729143
        assert power.mde == pytest.approx(z * 0.3 / np.sqrt(42))
        assert "exact test" not in power.note
    # A binary metric that did not use the exact test (nothing to say about a floor) is unchanged.
    plain = estimate_power(_mc("binary", n=42, test="permutation"), 0.02, 0.05)
    assert "exact test" not in plain.note
    uniform = estimate_power(_mc("rate", diff_sd=0.0, baseline_var=0.0), 0.02, 0.05)
    assert uniform.mde is None and "cannot be estimated" in uniform.note


def test_the_power_wording_in_the_docs_names_the_floor() -> None:
    from pathlib import Path

    docs = Path(__file__).resolve().parent.parent / "docs"
    guide = (docs / "guides" / "repeated-trials.md").read_text(encoding="utf-8")
    glossary = (docs / "glossary.md").read_text(encoding="utf-8")
    for text in (guide, glossary):
        assert "exact" in text and "McNemar" in text
    assert "6 cases" in guide or "six cases" in guide
