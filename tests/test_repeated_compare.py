"""Comparing and gating runs that repeat each case: aggregation, the rate test, the gate rules at
more than one trial, power, and the invariants that keep one trial per case exactly as it was."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable

import pytest
from pydantic import ValidationError

from evalproof.compare import (
    ComparisonRefused,
    compare_metric,
    compare_runs,
)
from evalproof.config import GateConfig
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    decide,
)
from evalproof.models import Run
from evalproof.report import comparison_markdown, to_json
from evalproof.stats import sign_flip_test
from tests.conftest import make_run


def _k(
    outcomes: dict[str, list[bool | float | None]],
    *,
    scorer: str = "m",
    tags: dict[str, list[str]] | None = None,
    errors: Iterable[tuple[str, int]] = (),
    latencies: dict[int, float] | None = None,
    system: str = "sys",
) -> Run:
    """A run with ``len(trials)`` trials per case, from ``{case: [outcome of trial 0, 1, ...]}``.
    ``latencies`` gives a latency by trial index."""
    failed = set(errors)
    k = len(next(iter(outcomes.values())))
    results = []
    for cid, values in outcomes.items():
        for trial, value in enumerate(values):
            errs = {cid} if (cid, trial) in failed else None
            one = make_run({cid: {scorer: value}}, tags=tags, errors=errs).results[0]
            if latencies and trial in latencies:
                one.output = one.output.model_copy(update={"latency_ms": latencies[trial]})
            results.append(one.model_copy(update={"trial": trial}))
    info = make_run({"x": {}}, system=system).info.model_copy(update={"trials": k})
    return Run(info=info, results=results)


def _body(text: str) -> str:
    """The report from its ``Baseline ... vs candidate ...`` line on, without power lines."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Baseline `"))
    kept = [line for line in lines[start:] if not line.startswith("Power on")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(kept))


def _ids(n: int) -> list[str]:
    return [f"c{i:02d}" for i in range(n)]


def _all_pass(n: int, k: int = 3) -> dict[str, list[bool | float | None]]:
    return {cid: [True] * k for cid in _ids(n)}


# --- aggregation -------------------------------------------------------------------------------


def test_mean_compares_pass_rates_with_the_sign_flip_test() -> None:
    base = _k({"a": [True] * 3, "b": [True] * 3, "c": [True, True, False], "d": [True] * 3})
    cand = _k(
        {"a": [True] * 3, "b": [True, False, False], "c": [True, True, False], "d": [False] * 3}
    )
    mc = compare_runs(base, cand).metrics["m"]
    assert mc.kind == "rate" and mc.trials == 3 and mc.aggregate == "mean"
    assert mc.baseline == pytest.approx((1 + 1 + 2 / 3 + 1) / 4)
    assert mc.candidate == pytest.approx((1 + 1 / 3 + 2 / 3 + 0) / 4)
    assert mc.delta == pytest.approx(mc.candidate - mc.baseline)
    assert mc.test == "sign-flip permutation (exact)"
    # The differences are 0, -2/3, 0 and -1; the test is on those, not on the 12 trials.
    assert mc.p_value == pytest.approx(sign_flip_test([0, -2 / 3, 0, -1]).p_value)
    assert mc.n_paired == 4 and mc.ci_low <= mc.delta <= mc.ci_high
    assert mc.baseline_ci is not None and mc.candidate_ci is not None


def test_the_rate_p_value_is_exact_on_a_large_suite_when_every_case_has_k_trials() -> None:
    """Forty cases are far past enumeration; with a whole number of trials each, the dynamic
    program is exact, so the label does not say Monte Carlo and the p-value is repeatable."""
    base = _k(_all_pass(40))
    cand_rows: dict[str, list[bool | float | None]] = {
        cid: [True, True, False] if i < 25 else [True] * 3 for i, cid in enumerate(_ids(40))
    }
    cand = _k(cand_rows)
    mc = compare_runs(base, cand).metrics["m"]
    assert mc.test == "sign-flip permutation (exact)"
    assert mc.p_value == compare_runs(base, cand).metrics["m"].p_value
    assert 0 < mc.p_value < 0.001  # 25 cases each down a third of a point, none up


def test_an_undecided_trial_moves_a_case_off_the_grid_and_the_test_is_monte_carlo() -> None:
    ids = _ids(24)
    base = _k({cid: [True, True, True] for cid in ids})
    rows: dict[str, list[bool | float | None]] = {
        cid: [True, False, True] if i % 2 else [False, True, True] for i, cid in enumerate(ids)
    }
    rows["c00"] = [None, False, True]  # one trial the scorer did not decide: rate 1/2, off 1/3s
    mc = compare_runs(base, _k(rows)).metrics["m"]
    assert mc.test == "sign-flip permutation (Monte Carlo, 20000)"
    assert 0 < mc.p_value < 0.05


def test_all_counts_a_case_only_when_every_trial_passed() -> None:
    base = _k({"a": [True] * 3, "b": [True] * 3, "c": [True] * 3, "d": [False] * 3})
    cand = _k(
        {"a": [True] * 3, "b": [True, True, False], "c": [True, False, False], "d": [True] * 3}
    )
    mc = compare_runs(base, cand, aggregate="all").metrics["m"]
    assert mc.kind == "binary" and mc.test == "exact McNemar" and mc.aggregate == "all"
    assert (mc.baseline, mc.candidate) == (0.75, 0.5)
    assert mc.regressed == ["b", "c"] and mc.improved == ["d"]
    assert mc.flaky_candidate == ["b", "c"]


def test_any_counts_a_case_when_at_least_one_trial_passed() -> None:
    base = _k({"a": [True] * 3, "b": [True] * 3, "c": [False] * 3, "d": [False, False, True]})
    cand = _k({"a": [True, False, False], "b": [False] * 3, "c": [False] * 3, "d": [False] * 3})
    mc = compare_runs(base, cand, aggregate="any").metrics["m"]
    assert mc.kind == "binary" and mc.aggregate == "any"
    assert (mc.baseline, mc.candidate) == (0.75, 0.25)
    assert mc.regressed == ["b", "d"] and mc.improved == []


def test_an_errored_trial_is_a_failure_and_an_undecided_one_is_dropped() -> None:
    base = _k({"a": [True] * 3, "b": [True] * 3})
    cand = _k({"a": [True, True, True], "b": [None, True, True]}, errors=[("a", 2)])
    mc = compare_runs(base, cand).metrics["m"]
    assert mc.candidate == pytest.approx((2 / 3 + 1.0) / 2)  # a errored once; b has two decided


def test_a_continuous_metric_is_tested_on_its_per_case_mean() -> None:
    base = _k({"a": [0.5, 0.5], "b": [0.9, 0.7], "c": [0.2, 0.4]}, scorer="s")
    cand = _k({"a": [0.5, 0.7], "b": [0.5, 0.5], "c": [0.2, 0.2]}, scorer="s")
    mc = compare_runs(base, cand).metrics["s"]
    assert mc.kind == "continuous" and mc.aggregate == "mean"
    assert mc.baseline == pytest.approx((0.5 + 0.8 + 0.3) / 3)
    assert mc.candidate == pytest.approx((0.6 + 0.5 + 0.2) / 3)
    assert mc.regressed == ["b", "c"] and mc.improved == ["a"]
    assert mc.test.startswith("sign-flip permutation")


def test_all_and_any_do_not_apply_to_a_continuous_metric() -> None:
    base = _k({"a": [0.5, 0.5], "b": [0.9, 0.7]}, scorer="s")
    cand = _k({"a": [0.5, 0.7], "b": [0.5, 0.5]}, scorer="s")
    for aggregate in ("all", "any"):
        cmp = compare_runs(base, cand, aggregate=aggregate)
        assert cmp.aggregate_ignored == ["s"]
        assert cmp.metrics["s"].aggregate == "mean"  # still compared, on the mean
    assert compare_runs(base, cand).aggregate_ignored == []


def test_the_flaky_cases_are_those_with_a_pass_and_a_fail_among_decided_trials() -> None:
    base = _k({"a": [True, False, True], "b": [True] * 3, "c": [False] * 3})
    cand = _k({"a": [True] * 3, "b": [None, True, False], "c": [None, False, False]})
    mc = compare_runs(base, cand).metrics["m"]
    assert mc.flaky_baseline == ["a"]
    assert mc.flaky_candidate == ["b"]


def test_majority_flips_are_cases_whose_rate_crosses_one_half() -> None:
    base = _k(
        {"a": [True] * 3, "b": [True] * 3, "c": [True, True, False], "d": [True, False, False]}
    )
    cand = _k(
        {"a": [True, True, False], "b": [True, False, False], "c": [False] * 3, "d": [True] * 3}
    )
    mc = compare_runs(base, cand).metrics["m"]
    assert mc.regressed == ["b", "c"]  # a is 3/3 -> 2/3, still a majority
    assert mc.improved == ["d"]


def test_a_different_number_of_trials_is_refused() -> None:
    three, five = _k(_all_pass(5, 3)), _k(_all_pass(5, 5))
    with pytest.raises(ComparisonRefused, match=r"baseline has k=3, candidate has k=5"):
        compare_runs(three, five)
    with pytest.raises(ComparisonRefused, match="same --trials"):
        compare_metric("m", three, five, _ids(5))
    one = make_run({cid: {"m": True} for cid in _ids(5)})
    with pytest.raises(ComparisonRefused):
        compare_runs(one, three)  # one trial against several is a mismatch too


def test_an_unknown_aggregate_is_an_error_not_a_silent_mean() -> None:
    run = _k(_all_pass(3))
    with pytest.raises(ValueError, match="aggregate"):
        compare_metric("m", run, run, _ids(3), aggregate="median")


# --- the gate at k > 1 -------------------------------------------------------------------------


def _gate(**kw: object) -> GateConfig:
    base: dict[str, object] = {"metric": "m", "min_cases": 10}
    return GateConfig(**{**base, **kw})  # type: ignore[arg-type]


def _big_drop() -> tuple[Run, Run]:
    base = _k(_all_pass(30))
    rows: dict[str, list[bool | float | None]] = {
        cid: [True, False, False] if i < 12 else [True] * 3 for i, cid in enumerate(_ids(30))
    }
    return base, _k(rows)


def test_max_drop_and_alpha_apply_to_the_aggregated_values() -> None:
    base, cand = _big_drop()
    cmp = compare_runs(base, cand)
    decision = decide(cmp, _gate(max_drop=0.1), cand)
    assert decision.exit_code == EXIT_REGRESSION
    assert "sign-flip permutation (exact)" in decision.reasons[0]
    mc = cmp.metrics["m"]
    assert mc.delta == pytest.approx(-12 * (2 / 3) / 30)
    # A drop smaller than max_drop passes, however significant.
    assert decide(cmp, _gate(max_drop=0.5), cand).exit_code == EXIT_PASS
    # A drop that is not significant warns and passes.
    mild = compare_runs(base, _k({**_all_pass(30), "c00": [True, False, False]}))
    assert decide(mild, _gate(max_drop=0.0001), cand).exit_code == EXIT_PASS


def test_aggregate_all_makes_the_same_cases_a_bigger_drop() -> None:
    base = _k(_all_pass(30))
    cand = _k(
        {cid: [True, True, False] if i < 12 else [True] * 3 for i, cid in enumerate(_ids(30))}
    )
    mean = compare_runs(base, cand, aggregate="mean").metrics["m"]
    every = compare_runs(base, cand, aggregate="all").metrics["m"]
    assert mean.delta == pytest.approx(-12 * (1 / 3) / 30)
    assert every.delta == pytest.approx(-12 / 30)
    gate = _gate(max_drop=0.3)
    assert decide(compare_runs(base, cand, aggregate="mean"), gate).exit_code == EXIT_PASS
    assert decide(compare_runs(base, cand, aggregate="all"), gate).exit_code == EXIT_REGRESSION


def test_min_cases_counts_cases_and_not_trials() -> None:
    base, cand = _k(_all_pass(8, 5)), _k(_all_pass(8, 5))
    assert len(base.results) == 40  # forty results, eight cases
    decision = decide(compare_runs(base, cand), _gate(min_cases=20), cand)
    assert decision.exit_code == EXIT_CANNOT_GATE
    assert "only 8 paired cases" in decision.reasons[0]
    assert decide(compare_runs(base, cand), _gate(min_cases=8), cand).exit_code == EXIT_PASS


def test_max_new_failures_with_a_rate_counts_majority_flips() -> None:
    ids = _ids(20)
    base = _k(_all_pass(20))
    # c00 goes 3/3 -> 2/3 (still a majority pass); c01 goes 3/3 -> 1/3 (a flip).
    rows: dict[str, list[bool | float | None]] = {cid: [True] * 3 for cid in ids}
    rows["c00"] = [True, True, False]
    cand = _k(rows)
    assert decide(compare_runs(base, cand), _gate(max_new_failures=0), cand).exit_code == EXIT_PASS
    rows["c01"] = [True, False, False]
    cand = _k(rows)
    decision = decide(compare_runs(base, cand), _gate(max_new_failures=0), cand)
    assert decision.exit_code == EXIT_REGRESSION
    assert "newly failing by majority of their trials" in decision.reasons[-1]
    assert "`c01`" in decision.reasons[-1] and "`c00`" not in decision.reasons[-1]
    assert decide(compare_runs(base, cand), _gate(max_new_failures=1), cand).exit_code == EXIT_PASS


def test_max_new_failures_under_all_counts_cases_that_stopped_passing_every_trial() -> None:
    base = _k(_all_pass(20))
    rows: dict[str, list[bool | float | None]] = {cid: [True] * 3 for cid in _ids(20)}
    rows["c00"] = [True, True, False]
    cand = _k(rows)
    gate = _gate(max_new_failures=0, aggregate="all")
    decision = decide(compare_runs(base, cand, aggregate="all"), gate, cand)
    assert decision.exit_code == EXIT_REGRESSION
    assert "by majority" not in decision.reasons[-1]


def test_must_pass_requires_every_trial_of_a_tagged_case() -> None:
    tags = {"s1": ["safety"], "s2": ["safety"]}
    base_rows: dict[str, list[bool | float | None]] = {cid: [True] * 3 for cid in ["s1", "s2", "x"]}
    base = _k(base_rows, tags=tags)
    cand = _k({**base_rows, "s2": [True, True, False]}, tags=tags)  # passes by majority only
    decision = decide(compare_runs(base, cand), _gate(min_cases=3, must_pass=["safety"]), cand)
    assert decision.exit_code == EXIT_REGRESSION
    message = next(r for r in decision.reasons if "must-pass" in r)
    assert "1 must-pass case(s)" in message and "`s2`" in message
    assert "in at least one trial" in message
    clean = _k(base_rows, tags=tags)
    ok = decide(compare_runs(base, clean), _gate(min_cases=3, must_pass=["safety"]), clean)
    assert ok.exit_code == EXIT_PASS


def test_must_pass_counts_an_errored_trial_and_names_a_case_once() -> None:
    tags = {"s1": ["safety"]}
    base = _k({"s1": [True] * 3, "x": [True] * 3}, tags=tags)
    cand = _k({"s1": [False, False, True], "x": [True] * 3}, tags=tags, errors=[("s1", 2)])
    decision = decide(compare_runs(base, cand), _gate(min_cases=2, must_pass=["safety"]), cand)
    message = next(r for r in decision.reasons if "must-pass" in r)
    assert "1 must-pass case(s)" in message  # three bad trials, one case


def test_min_pass_rate_is_pooled_over_all_trials() -> None:
    base = _k(_all_pass(20))
    rows: dict[str, list[bool | float | None]] = {cid: [True] * 3 for cid in _ids(20)}
    for cid in _ids(20)[:6]:
        rows[cid] = [True, False, False]  # six cases at 1/3: 6 * 2 failed trials of 60
    cand = _k(rows)  # pooled pass rate 48 / 60 = 0.8
    cmp = compare_runs(base, cand)
    assert decide(cmp, _gate(max_drop=0.9, min_pass_rate=0.75), cand).exit_code == EXIT_PASS
    low = decide(cmp, _gate(max_drop=0.9, min_pass_rate=0.85), cand)
    assert low.exit_code == EXIT_REGRESSION
    assert "48/60" in " ".join(low.reasons) and "trials" in " ".join(low.reasons)


def test_budgets_are_over_all_trials_not_just_the_first() -> None:
    base = _k(_all_pass(20), latencies={0: 10.0, 1: 10.0, 2: 10.0})
    cand = _k(_all_pass(20), latencies={0: 10.0, 1: 900.0, 2: 900.0})  # trial 0 alone is fast
    decision = decide(compare_runs(base, cand), _gate(max_p95_latency_ms=100.0), cand)
    assert decision.exit_code == EXIT_BUDGET
    assert "p95 latency" in decision.reasons[0]


def test_a_gate_on_a_continuous_metric_with_all_or_any_cannot_decide() -> None:
    base = _k({cid: [0.5, 0.5] for cid in _ids(12)}, scorer="s")
    cand = _k({cid: [0.5, 0.4] for cid in _ids(12)}, scorer="s")
    gate = GateConfig(metric="s", min_cases=5, aggregate="any")
    decision = decide(compare_runs(base, cand, aggregate="any"), gate, cand)
    assert decision.exit_code == EXIT_CANNOT_GATE
    assert "aggregate any needs a pass/fail metric" in decision.reasons[0]
    assert "'s'" in decision.reasons[0]
    # The mean is allowed, and the gate then decides.
    assert decide(
        compare_runs(base, cand), GateConfig(metric="s", min_cases=5), cand
    ).exit_code in (
        EXIT_PASS,
        EXIT_REGRESSION,
    )


def test_gate_config_aggregate_defaults_to_mean_and_stays_strict() -> None:
    assert GateConfig(metric="m").aggregate == "mean"
    assert GateConfig(metric="m", aggregate="all").aggregate == "all"
    with pytest.raises(ValidationError):
        GateConfig(metric="m", aggregate="median")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        GateConfig(metric="m", aggregates="all")  # type: ignore[call-arg]


# --- power: informational only -----------------------------------------------------------------


def test_power_is_reported_for_each_gated_metric_and_never_changes_the_decision() -> None:
    base, cand = _big_drop()
    cmp = compare_runs(base, cand)
    decision = decide(cmp, _gate(max_drop=0.9), cand)  # a huge max_drop: passes
    assert decision.exit_code == EXIT_PASS
    # The only warning is the significant drop inside the tolerance; a suite too small to detect
    # a drop of max_drop adds none of its own.
    assert len(decision.warnings) == 1 and "within the 0.900 tolerance" in decision.warnings[0]
    power = decision.power["m"]
    assert power.n == 30 and power.alpha_used == 0.05
    assert power.mde is not None and power.mde > 0
    assert "30 cases" in power.note

    # The same comparison under a different power situation gives the same status and codes.
    tiny = compare_runs(_k(_all_pass(10)), _k(_all_pass(10)))
    quiet = decide(tiny, _gate(), _k(_all_pass(10)))
    assert (quiet.status, quiet.exit_code, quiet.warnings) == ("pass", EXIT_PASS, [])
    assert quiet.power["m"].mde is None  # a uniform baseline: nothing to estimate


def test_power_alpha_is_shared_among_gated_metrics_and_candidates() -> None:
    base, cand = _big_drop()
    cmp = compare_runs(base, cand)
    assert decide(cmp, _gate()).power["m"].alpha_used == pytest.approx(0.05)
    assert decide(cmp, _gate(), adjusted={"m": 0.5}, candidates=3).power["m"].alpha_used == (
        pytest.approx(0.05 / 3)
    )
    base2 = make_run({cid: {"m": True, "n": True} for cid in _ids(30)})
    cand2 = make_run({cid: {"m": i >= 10, "n": i >= 5} for i, cid in enumerate(_ids(30))})
    two = decide(compare_runs(base2, cand2), GateConfig(metrics=["m", "n"], min_cases=10))
    assert two.power["m"].alpha_used == pytest.approx(0.025)
    both = decide(
        compare_runs(base2, cand2),
        GateConfig(metrics=["m", "n"], min_cases=10),
        adjusted={"m": 0.5, "n": 0.5},
        candidates=2,
    )
    assert both.power["n"].alpha_used == pytest.approx(0.05 / 4)


def test_power_does_not_change_the_k1_decision_only_adds_a_field() -> None:
    base = make_run({cid: {"m": True} for cid in _ids(30)})
    cand = make_run({cid: {"m": i >= 4} for i, cid in enumerate(_ids(30))})
    decision = decide(compare_runs(base, cand), _gate())
    assert decision.exit_code == EXIT_PASS or decision.exit_code == EXIT_REGRESSION
    payload = json.loads(to_json(decision))
    assert set(payload["power"]) == {"m"}
    assert set(payload["power"]["m"]) == {"mde", "cases_needed", "n", "alpha_used", "note"}


# --- markdown and JSON -------------------------------------------------------------------------


def test_the_report_names_trials_aggregate_flaky_cases_and_power() -> None:
    base, cand = _big_drop()
    cmp = compare_runs(base, cand)
    gate = _gate(max_drop=0.1)
    text = comparison_markdown(cmp, decide(cmp, gate, cand), gate.gated, runs=(base, cand), show=1)
    assert "on 30 paired cases (trials: 3, aggregate: mean)." in text
    assert "Power on **m**: With 30 cases this gate reliably detects drops of about" in text
    assert "Flaky in candidate (12): `c00`" in text
    assert "Flaky in baseline" not in text
    assert "| baseline | 30 × 3 |" in text and "| candidate | 30 × 3 |" in text
    assert "Regressed on **m** by majority (12)" in text
    assert "(trial 1)" in text  # the example names the failing trial


def test_the_report_warns_when_all_or_any_met_a_continuous_metric() -> None:
    binary_b = _k({cid: [True, True] for cid in _ids(6)})
    binary_c = _k({cid: [True, False] for cid in _ids(6)})
    cmp = compare_runs(binary_b, binary_c, aggregate="all")
    assert cmp.aggregate_ignored == []  # a pass/fail metric takes any aggregate
    cont_b = _k({cid: [0.5, 0.5] for cid in _ids(6)}, scorer="s")
    cont_c = _k({cid: [0.5, 0.4] for cid in _ids(6)}, scorer="s")
    text = comparison_markdown(compare_runs(cont_b, cont_c, aggregate="all"))
    assert "aggregate all needs a pass/fail metric: 's' is continuous and was compared" in text


def test_json_carries_trials_aggregate_flaky_and_power() -> None:
    base, cand = _big_drop()
    cmp = compare_runs(base, cand)
    gate = _gate(max_drop=0.1)
    payload = json.loads(to_json(cmp))
    assert payload["trials"] == 3 and payload["aggregate"] == "mean"
    metric = payload["metrics"]["m"]
    assert metric["kind"] == "rate" and metric["trials"] == 3
    assert metric["flaky_candidate"][0] == "c00" and metric["flaky_baseline"] == []
    assert json.loads(to_json(decide(cmp, gate, cand)))["power"]["m"]["n"] == 30


# --- one trial per case is exactly as it was ---------------------------------------------------


def test_one_trial_ignores_the_aggregate_and_reads_no_new_fields() -> None:
    ids = _ids(30)
    base = make_run({cid: {"m": True, "s": 0.5 + 0.01 * i} for i, cid in enumerate(ids)})
    cand = make_run({cid: {"m": i >= 5, "s": 0.4 + 0.01 * i} for i, cid in enumerate(ids)})
    reference = compare_runs(base, cand)
    for aggregate in ("all", "any"):
        other = compare_runs(base, cand, aggregate=aggregate)
        assert other == reference  # the same comparison, field for field
    mc = reference.metrics["m"]
    assert (mc.kind, mc.test, mc.trials, mc.aggregate) == ("binary", "exact McNemar", 1, None)
    assert (mc.flaky_baseline, mc.flaky_candidate) == ([], [])
    cont = reference.metrics["s"]
    assert cont.kind == "continuous" and cont.test == "paired bootstrap"
    assert reference.trials == 1 and reference.aggregate is None
    assert reference.aggregate_ignored == []


def test_one_trial_report_has_no_trials_wording_and_only_the_power_line_is_new() -> None:
    ids = _ids(30)
    base = make_run({cid: {"m": True} for cid in ids})
    cand = make_run({cid: {"m": i >= 5} for i, cid in enumerate(ids)})
    cmp = compare_runs(base, cand)
    gate = _gate()
    plain = comparison_markdown(cmp, None, ["m"], runs=(base, cand), show=1)
    gated = comparison_markdown(cmp, decide(cmp, gate, cand), gate.gated, runs=(base, cand), show=1)
    for text in (plain, gated):
        assert "trials" not in text and "aggregate" not in text and "by majority" not in text
        assert "Flaky" not in text and "×" not in text.split("| Run |")[1]
        assert "exact McNemar" in text
    assert "Power on" not in plain  # without a gate there is no decision, and so no power
    assert "Power on **m**" in gated
    # Everything from the header on is the same: the gated report is the plain one plus the power.
    assert _body(gated) == _body(plain)
