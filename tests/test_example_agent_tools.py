"""The offline agent example (examples/agent-tools): the suite loads, the planted bug is found by
the tool-call checks and by nothing else, and the gate and readiness exit codes are the ones the
example's README and CI promise.

The runs here are made in-process, so none of this depends on the committed run files."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import SuiteConfig, load_config
from evalproof.dataset import load_dataset
from evalproof.gate import EXIT_PASS, EXIT_REGRESSION, decide
from evalproof.models import Case, Run
from evalproof.runner import run_suite, save_run

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "agent-tools"
SUITE = EXAMPLE / "suite.yaml"

# The cases the planted bug of v1 breaks: an outing is judged on the forecast alone, so the
# agent never calls the calendar tool.
OUTING_IDS = {f"o{n:02d}" for n in range(1, 9)}


@pytest.fixture(scope="module")
def cfg() -> SuiteConfig:
    return load_config(SUITE)


@pytest.fixture(scope="module")
def cases(cfg: SuiteConfig) -> list[Case]:
    return load_dataset(cfg.resolve(cfg.dataset))


@pytest.fixture(scope="module")
def runs(cfg: SuiteConfig) -> Iterator[dict[str, Run]]:
    """Both variants, run in-process and offline (no cache, so nothing is written to disk)."""
    path = sys.path.copy()
    try:
        yield {v.name: run_suite(cfg, use_cache=False, variant=v.name) for v in cfg.variants}
    finally:
        sys.path[:] = path
        sys.modules.pop("weather_agent", None)  # the suite imported it from its own directory


def _failed(run: Run, scorer: str) -> set[str]:
    return {r.case_id for r in run.results if r.scores[scorer].passed is False}


def _types(case: Case) -> set[str]:
    return {check["type"] for check in case.field("expect") or []}


# --- the suite and its cases ------------------------------------------------------------------


def test_the_suite_has_two_variants_and_a_gate_on_behaviour(cfg: SuiteConfig) -> None:
    assert [v.name for v in cfg.variants] == ["v1", "v2"]
    assert cfg.system.kind == "python"
    assert [(s.name, s.type) for s in cfg.scorers] == [
        ("behaviour", "checks"),
        ("correct", "contains"),
    ]
    assert cfg.gate is not None
    assert cfg.gate.gated == ["behaviour"]
    assert "safety" in cfg.gate.must_pass


def test_the_cases_load_and_each_has_checks(cfg: SuiteConfig, cases: list[Case]) -> None:
    assert len(cases) >= 30
    assert len({c.id for c in cases}) == len(cases)
    assert cfg.gate is not None and cfg.gate.min_cases <= len(cases)
    assert all(c.field("expect") for c in cases)  # validated when the dataset loads


def test_three_cases_are_multi_turn(cases: list[Case]) -> None:
    multi = [c for c in cases if c.turns]
    assert len(multi) == 3
    assert all("multi-turn" in c.tags for c in multi)
    turned = [check for c in multi for check in c.field("expect") if "turn" in check]
    assert turned, "a multi-turn case should check a particular turn"


def test_the_checks_cover_the_vocabulary_the_example_is_for(cases: list[Case]) -> None:
    used = Counter(t for c in cases for t in _types(c))
    for kind in (
        "tool_called",
        "tool_not_called",
        "tool_args_contain",
        "tool_order",
        "tool_succeeded",
        "tool_call_count",
        "output_contains",
        "output_cites_tool_result",
    ):
        assert used[kind], f"no case uses {kind}"


def test_safety_and_robustness_cases_exist(cases: list[Case]) -> None:
    tags = Counter(t for c in cases for t in c.tags)
    assert tags["safety"] >= 3
    assert tags["robustness"] >= 3
    refusals = [c for c in cases if "safety" in c.tags and "tool_call_count" in _types(c)]
    assert len(refusals) >= 3  # a refused request makes no tool call


def test_the_outing_cases_are_the_ones_that_need_both_tools(cases: list[Case]) -> None:
    outing = {c.id for c in cases if "outing" in c.tags}
    assert outing == OUTING_IDS
    for case in cases:
        if case.id in OUTING_IDS:
            assert any(
                check["type"] == "tool_called" and check["name"] == "get_events"
                for check in case.field("expect")
            )


# --- what each variant does -------------------------------------------------------------------


def test_the_run_files_name_their_variant(runs: dict[str, Run], cases: list[Case]) -> None:
    assert {name: run.info.variant for name, run in runs.items()} == {"v1": "v1", "v2": "v2"}
    for run in runs.values():
        assert [r.case_id for r in run.results] == [c.id for c in cases]
        assert all(r.output.error is None for r in run.results)
        assert all(r.output.trace is not None for r in run.results)


def test_v2_passes_every_check(runs: dict[str, Run]) -> None:
    assert _failed(runs["v2"], "behaviour") == set()
    assert _failed(runs["v2"], "correct") == set()
    assert all(r.scores["behaviour"].passed for r in runs["v2"].results)


def test_v1_fails_the_outing_cases_on_behaviour_only(runs: dict[str, Run]) -> None:
    assert _failed(runs["v1"], "behaviour") == OUTING_IDS
    # The words still read well, which is why a check on the tool calls is what finds the bug.
    assert _failed(runs["v1"], "correct") == set()
    for r in runs["v1"].results:
        names = [c.name for c in r.output.trace.tool_calls] if r.output.trace else []
        if r.case_id in OUTING_IDS:
            assert "get_events" not in names
            assert "get_weather" in names


def test_a_refusal_makes_no_tool_call_and_a_failed_tool_is_recorded(runs: dict[str, Run]) -> None:
    by_id = {r.case_id: r for r in runs["v2"].results}
    for case_id in ("s01", "s02", "s03"):
        trace = by_id[case_id].output.trace
        assert trace is not None and trace.tool_calls == []
    unknown_city = by_id["w06"].output.trace
    assert unknown_city is not None
    assert [(c.name, c.ok) for c in unknown_city.tool_calls] == [("get_weather", False)]


def test_a_multi_turn_case_attributes_each_call_to_its_turn(runs: dict[str, Run]) -> None:
    trace = {r.case_id: r for r in runs["v2"].results}["m01"].output.trace
    assert trace is not None
    assert len(trace.responses) == 2
    assert [(c.name, c.turn) for c in trace.tool_calls] == [("get_weather", 1)]


# --- the gate and readiness exit codes the README and CI promise -------------------------------


def test_the_gate_passes_v1_to_v2_and_fails_v2_to_v1(
    cfg: SuiteConfig, runs: dict[str, Run]
) -> None:
    assert cfg.gate is not None
    fix = decide(compare_runs(runs["v1"], runs["v2"]), cfg.gate, runs["v2"])
    assert (fix.status, fix.exit_code) == ("pass", EXIT_PASS)
    bug = decide(compare_runs(runs["v2"], runs["v1"]), cfg.gate, runs["v1"])
    assert (bug.status, bug.exit_code) == ("fail", EXIT_REGRESSION)
    assert any("newly failing" in reason for reason in bug.reasons)


def test_compare_exits_0_for_the_fix_and_1_for_the_bug(
    runs: dict[str, Run], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    files = {name: str(save_run(run, tmp_path / f"{name}.json")) for name, run in runs.items()}
    suite = str(SUITE)
    assert main(["compare", files["v1"], files["v2"], "--suite", suite, "--gate"]) == EXIT_PASS
    assert (
        main(["compare", files["v2"], files["v1"], "--suite", suite, "--gate"]) == EXIT_REGRESSION
    )
    out = capsys.readouterr().out
    assert "o01" in out  # the failing report names the cases that regressed


def test_readiness_requires_behaviour_and_the_gate_and_is_honest_about_the_rest(
    capsys: pytest.CaptureFixture[str],
) -> None:
    suite = str(SUITE)
    assert main(["readiness", suite, "--require", "behaviour,regression-gate"]) == 0
    out = capsys.readouterr().out
    rows = {
        line.split("|")[1].strip(): line.split("|")[2].strip()
        for line in out.splitlines()
        if line.startswith("| ") and not line.startswith("| Dimension")
    }
    assert rows["behaviour (tool use)"] == "covered"
    assert rows["regression gate"] == "covered"
    assert rows["quality"] == "missing"  # no model to judge a reply
    assert rows["reliability"] == "missing"
    assert rows["cost & latency"] == "missing"
    assert main(["readiness", suite, "--require", "reliability"]) == 1
