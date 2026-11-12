"""Holm-corrected multi-metric gating: the p-value adjustment, the decision, config, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import ConfigError, GateConfig, load_config
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, EXIT_REGRESSION, decide
from evalproof.report import comparison_markdown
from evalproof.runner import save_run
from evalproof.stats import holm
from tests.conftest import make_run


def _two(n: int, regress1: int, regress2: int = 0, **kw):
    """Baseline passes both metrics everywhere; the candidate fails the first
    ``regress1`` cases on m1 and the first ``regress2`` on m2."""
    base = {f"c{i:03d}": {"m1": True, "m2": True} for i in range(n)}
    cand = {f"c{i:03d}": {"m1": i >= regress1, "m2": i >= regress2} for i in range(n)}
    return make_run(base, system="base"), make_run(cand, system="cand", **kw)


# --- holm() ---------------------------------------------------------------------------


def test_holm_known_values() -> None:
    # Sorted p: .01 x3, .03 x2, .04 x1 -> .03, .06, .06; the last is raised to the running max.
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert holm([0.01, 0.02, 0.03, 0.04]) == pytest.approx([0.04, 0.06, 0.06, 0.06])


def test_holm_is_monotone_bounded_and_no_worse_than_bonferroni() -> None:
    ps = [0.2, 0.001, 0.5, 0.03, 0.04, 0.9, 0.0]
    adjusted = holm(ps)
    by_p = [a for _, a in sorted(zip(ps, adjusted, strict=True))]
    assert by_p == sorted(by_p)  # the adjustment never reorders the hypotheses
    for p, a in zip(ps, adjusted, strict=True):
        assert p <= a <= min(1.0, len(ps) * p) or a == 1.0
        assert a <= 1.0


def test_holm_edge_cases() -> None:
    assert holm([]) == []
    assert holm([0.04]) == [0.04]  # one test: nothing to correct
    assert holm([0.9, 0.8]) == [1.0, 1.0]  # capped at 1
    with pytest.raises(ValueError, match="p-values"):
        holm([0.1, 1.5])
    with pytest.raises(ValueError, match="p-values"):
        holm([float("nan")])


# --- GateConfig ------------------------------------------------------------------------


def test_gate_config_forms_and_dedupe() -> None:
    assert GateConfig(metric="a").gated == ["a"]
    assert GateConfig(metrics=["a", "b"]).gated == ["a", "b"]
    assert GateConfig(metric="a", metrics=["b", "a", "b"]).gated == ["a", "b"]
    with pytest.raises(ValidationError, match="needs 'metric' or 'metrics'"):
        GateConfig()


def test_suite_checks_every_gated_name(tmp_path: Path) -> None:
    head = "name: x\ndataset: d.jsonl\nsystem: {target: 'ollama:m'}\n"
    scorers = "scorers:\n  - {name: a, type: regex}\n  - {name: b, type: regex}\n"
    (tmp_path / "ok.yaml").write_text(head + scorers + "gate: {metrics: [a, b]}\n", "utf-8")
    assert load_config(tmp_path / "ok.yaml").gate.gated == ["a", "b"]  # type: ignore[union-attr]
    (tmp_path / "bad.yaml").write_text(head + scorers + "gate: {metrics: [a, zzz]}\n", "utf-8")
    with pytest.raises(ConfigError, match="'zzz' is not a scorer"):
        load_config(tmp_path / "bad.yaml")


# --- decide() --------------------------------------------------------------------------


def test_two_metric_gate_fails_on_a_clear_regression() -> None:
    b, c = _two(60, regress1=15)
    gate = GateConfig(metrics=["m1", "m2"], max_drop=0.02, min_cases=30)
    d = decide(compare_runs(b, c), gate)
    assert d.exit_code == EXIT_REGRESSION and d.status == "fail"
    assert "m1" in d.reasons[0] and "Holm-adjusted p=" in d.reasons[0]
    assert set(d.adjusted_p) == {"m1", "m2"}
    assert d.adjusted_p["m2"] == 1.0  # nothing changed on m2


def test_holm_turns_a_marginal_single_metric_fail_into_a_warning() -> None:
    # Six cases regress and none improve: raw exact-McNemar p = 2 * 0.5**6 = 0.03125.
    b, c = _two(60, regress1=6)
    cmp = compare_runs(b, c)
    alone = decide(cmp, GateConfig(metric="m1", max_drop=0.02, min_cases=30))
    assert alone.exit_code == EXIT_REGRESSION
    both = decide(cmp, GateConfig(metrics=["m1", "m2"], max_drop=0.02, min_cases=30))
    assert both.exit_code == EXIT_PASS and both.status == "warn"
    assert both.adjusted_p["m1"] == pytest.approx(0.0625)
    assert "Holm-adjusted p=0.062" in both.warnings[0] and "raw p=0.031" in both.warnings[0]


def test_single_metric_gate_is_the_same_however_it_is_spelled() -> None:
    b, c = _two(60, regress1=12)
    cmp = compare_runs(b, c)
    spellings = [
        GateConfig(metric="m1", max_drop=0.02, min_cases=30),
        GateConfig(metrics=["m1"], max_drop=0.02, min_cases=30),
        GateConfig(metric="m1", metrics=["m1"], max_drop=0.02, min_cases=30),
    ]
    decisions = [decide(cmp, g) for g in spellings]
    assert decisions[0] == decisions[1] == decisions[2]
    assert decisions[0].adjusted_p == {} and "Holm" not in decisions[0].reasons[0]
    assert "p=" in decisions[0].reasons[0] and "McNemar" in decisions[0].reasons[0]


def test_each_metric_must_be_gateable() -> None:
    b, c = _two(60, regress1=0)
    d = decide(compare_runs(b, c), GateConfig(metrics=["m1", "zzz"], min_cases=30))
    assert d.exit_code == EXIT_CANNOT_GATE and "zzz" in d.reasons[0]
    # min_cases applies per metric: m2 is only scored on 10 cases.
    base = {f"c{i}": {"m1": True} | ({"m2": True} if i < 10 else {}) for i in range(60)}
    cand = {f"c{i}": {"m1": True} | ({"m2": True} if i < 10 else {}) for i in range(60)}
    d = decide(compare_runs(make_run(base), make_run(cand)), GateConfig(metrics=["m1", "m2"]))
    assert d.exit_code == EXIT_CANNOT_GATE
    assert any("m2" in r and "paired cases" in r for r in d.reasons)


def test_undecided_warning_uses_the_metric_name_for_many() -> None:
    base = {f"c{i}": {"m1": True, "m2": True} for i in range(40)}
    cand = {f"c{i}": {"m1": True, "m2": True if i else None} for i in range(40)}
    d = decide(
        compare_runs(make_run(base), make_run(cand)),
        GateConfig(metrics=["m1", "m2"], min_cases=30),
    )
    assert d.exit_code == EXIT_PASS and any("undecided" in w and "m2" in w for w in d.warnings)


# --- report and CLI ---------------------------------------------------------------------


def test_report_adds_the_adjusted_column_only_for_several_metrics() -> None:
    b, c = _two(60, regress1=15)
    cmp = compare_runs(b, c)
    many = GateConfig(metrics=["m1", "m2"], min_cases=30)
    md = comparison_markdown(cmp, decide(cmp, many), many.gated)
    assert "Holm-adjusted p" in md and "**m1**" in md and "**m2**" in md
    one = GateConfig(metric="m1", min_cases=30)
    md1 = comparison_markdown(cmp, decide(cmp, one), "m1")
    assert "Holm-adjusted p" not in md1 and "**m1**" in md1 and "**m2**" not in md1


def test_cli_repeated_metric_flag(tmp_path: Path, capsys) -> None:
    b, c = _two(60, regress1=15)
    paths = {"b": tmp_path / "b.json", "c": tmp_path / "c.json"}
    save_run(b, paths["b"])
    save_run(c, paths["c"])
    out = tmp_path / "gate.json"
    argv = ["compare", str(paths["b"]), str(paths["c"]), "--metric", "m1", "--metric", "m2"]
    argv += ["--min-cases", "30", "--gate", "--json", str(out)]
    assert main(argv) == EXIT_REGRESSION
    assert "Holm-adjusted p" in capsys.readouterr().out
    gate = json.loads(out.read_text("utf-8"))["gate"]
    assert set(gate["adjusted_p"]) == {"m1", "m2"}
    # A single --metric is the old, uncorrected gate.
    argv = ["compare", str(paths["b"]), str(paths["c"]), "--metric", "m2", "--min-cases", "30"]
    assert main([*argv, "--gate"]) == EXIT_PASS
