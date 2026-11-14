from __future__ import annotations

import json

import pytest

from evalproof.calibration import calibrate
from evalproof.compare import compare_runs
from evalproof.config import GateConfig
from evalproof.gate import EXIT_BUDGET, EXIT_CANNOT_GATE, EXIT_PASS, EXIT_REGRESSION, decide
from evalproof.report import calibration_markdown, comparison_markdown, run_markdown, to_json
from tests.conftest import make_run


def _pair(n: int, regress: int, improve: int = 0, **kw):
    """Baseline passes every case; the candidate fails ``regress`` of them, and on
    ``improve`` extra cases the baseline fails while the candidate passes."""
    base, cand = {}, {}
    for i in range(n):
        base[f"c{i:03d}"] = {"m": True}
        cand[f"c{i:03d}"] = {"m": i >= regress}
    for j in range(improve):
        base[f"i{j:03d}"] = {"m": False}
        cand[f"i{j:03d}"] = {"m": True}
    return make_run(base, system="base"), make_run(cand, system="cand", **kw)


def test_binary_comparison_counts_discordant_pairs() -> None:
    b, c = _pair(50, regress=10, improve=2)
    cmp = compare_runs(b, c)
    mc = cmp.metrics["m"]
    assert mc.kind == "binary" and mc.n_paired == 52
    assert len(mc.regressed) == 10 and len(mc.improved) == 2
    assert mc.delta == pytest.approx((42 - 50) / 52)
    assert mc.test == "exact McNemar" and mc.p_value < 0.05
    assert cmp.same_dataset and cmp.baseline_label == "base"


def test_variant_runs_of_one_target_get_distinct_labels() -> None:
    b, c = _pair(10, regress=0)
    b.info.system = c.info.system = "python:agent:run"
    b.info.variant, c.info.variant = "prompt-v1", "prompt-v2"
    cmp = compare_runs(b, c)
    assert cmp.baseline_label == "python:agent:run (prompt-v1)"
    assert cmp.candidate_label == "python:agent:run (prompt-v2)"


def test_continuous_comparison_and_undecided() -> None:
    base = make_run({f"c{i}": {"s": 0.8} for i in range(30)} | {"u": {"s": None}})
    cand = make_run({f"c{i}": {"s": 0.6} for i in range(30)} | {"u": {"s": 0.5}})
    mc = compare_runs(base, cand).metrics["s"]
    assert mc.kind == "continuous" and mc.test == "paired bootstrap"
    assert mc.delta == pytest.approx(-0.2) and mc.undecided == 1 and mc.p_value == 0.0


def test_compare_skips_missing_scores_and_reports_unpaired() -> None:
    base = make_run({"a": {"m": True}, "b": {"m": True}})
    cand = make_run({"a": {"other": True}, "c": {"m": True}})
    cmp = compare_runs(base, cand, metrics=["m"])
    assert cmp.metrics == {}
    assert cmp.only_baseline == ["b"] and cmp.only_candidate == ["c"]


def test_gate_fails_on_significant_large_drop() -> None:
    b, c = _pair(60, regress=12)
    d = decide(compare_runs(b, c), GateConfig(metric="m", max_drop=0.02, min_cases=30))
    assert d.exit_code == EXIT_REGRESSION and d.status == "fail" and "McNemar" in d.reasons[0]


def test_gate_warns_on_large_but_insignificant_drop() -> None:
    b, c = _pair(60, regress=2)  # -3.3 pts, p = .5
    d = decide(compare_runs(b, c), GateConfig(metric="m", max_drop=0.02, min_cases=30))
    assert d.exit_code == EXIT_PASS and d.status == "warn"
    assert "not distinguishable from noise" in d.warnings[0]


def test_gate_warns_on_significant_small_drop() -> None:
    b, c = _pair(400, regress=12)  # -3 pts, p ≈ .0005, but within a 5-point tolerance
    d = decide(compare_runs(b, c), GateConfig(metric="m", max_drop=0.05, min_cases=30))
    assert d.exit_code == EXIT_PASS and "within" in d.warnings[0]


def test_gate_passes_clean_change() -> None:
    b, c = _pair(40, regress=0)
    d = decide(compare_runs(b, c), GateConfig(metric="m", min_cases=30))
    assert (d.status, d.exit_code, d.reasons, d.warnings) == ("pass", EXIT_PASS, [], [])


@pytest.mark.parametrize(
    ("gate", "kwargs", "reason"),
    [
        (GateConfig(metric="m", min_cases=100), {}, "paired cases"),
        (GateConfig(metric="zzz"), {}, "no paired scores"),
        (GateConfig(metric="m"), {"dataset_hash": "other"}, "re-run the baseline"),
    ],
)
def test_gate_cannot_gate(gate: GateConfig, kwargs: dict[str, str], reason: str) -> None:
    b, c = _pair(40, regress=0, **kwargs)
    d = decide(compare_runs(b, c), gate)
    assert d.exit_code == EXIT_CANNOT_GATE and reason in d.reasons[0]


def test_gate_budgets() -> None:
    b, c = _pair(40, regress=0, latency=900.0)
    gate = GateConfig(metric="m", min_cases=30, max_p95_latency_ms=500, max_cost_usd=-1)
    d = decide(compare_runs(b, c), gate)
    assert d.exit_code == EXIT_BUDGET and len(d.reasons) == 2


def test_gate_regression_outranks_budget_and_lists_extra_warnings() -> None:
    b, c = _pair(60, regress=15, latency=900.0)
    c.results[0].output.error = "timeout"
    c.results.pop()
    gate = GateConfig(metric="m", min_cases=30, max_p95_latency_ms=10)
    d = decide(compare_runs(b, c), gate)
    assert d.exit_code == EXIT_REGRESSION
    assert any("errors rose" in w for w in d.warnings)
    assert any("1 removed" in w and "excluded from the comparison" in w for w in d.warnings)


def test_comparison_markdown_and_json() -> None:
    b, c = _pair(60, regress=25)
    cmp = compare_runs(b, c)
    d = decide(cmp, GateConfig(metric="m", min_cases=30))
    md = comparison_markdown(cmp, d, "m")
    assert "❌ fail" in md and "**m**" in md and "and 15 more" in md
    assert "-41.7 pts" in md
    payload = json.loads(to_json(cmp))
    assert payload["metrics"]["m"]["n_paired"] == 60


def test_comparison_markdown_continuous_and_improved() -> None:
    base = make_run({f"c{i}": {"s": 0.5} for i in range(5)})
    cand = make_run({f"c{i}": {"s": 0.7} for i in range(5)})
    md = comparison_markdown(compare_runs(base, cand), None, "s")
    assert "+0.200" in md and "Improved on **s**: 5" in md


def test_run_markdown() -> None:
    run = make_run({"a": {"m": True, "s": 0.4}, "b": {"m": False, "s": None}})
    md = run_markdown(run)
    assert "| m | 2/2 | 50.0%" in md and "| s | 1/2 | — | 0.400 |" in md


def test_calibration_markdown() -> None:
    outcomes = {f"p{i}": {"judge": i < 9, "gold": True} for i in range(10)}
    outcomes |= {f"n{i}": {"judge": i < 3, "gold": False} for i in range(10)}
    rep = calibrate(make_run(outcomes), "judge", gold_scorer="gold", gold_sample=12, n_boot=200)
    md = calibration_markdown(rep)
    assert "Cohen's κ" in md and "PPI" in md and "truth (all gold labels)" in md
    assert "Judge disagreed with gold on" in md
