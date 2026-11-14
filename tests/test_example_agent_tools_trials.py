"""The repeated-trials walkthrough of the offline agent example
(examples/agent-tools/suite-flaky.yaml): the build that forgets the calendar on a seeded 40% of the
outing trials fails the gate at five trials per case, the fixed build does not, and the exit codes
are the ones the README and CI promise.

The runs here are made in-process, so none of this depends on committed run files."""

from __future__ import annotations

import random
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import SuiteConfig, load_config
from evalproof.gate import EXIT_PASS, EXIT_REGRESSION, decide
from evalproof.models import Run
from evalproof.runner import run_suite, save_run

EXAMPLE = Path(__file__).resolve().parent.parent / "examples" / "agent-tools"
SUITE = EXAMPLE / "suite-flaky.yaml"
TRIALS = 5
CHANCE = 0.4

OUTING_IDS = {f"o{n:02d}" for n in range(1, 9)}


@pytest.fixture(scope="module")
def cfg() -> SuiteConfig:
    return load_config(SUITE)


@pytest.fixture(scope="module")
def runs(cfg: SuiteConfig) -> Iterator[dict[str, Run]]:
    """v2 twice and flaky once, five trials each, in-process and offline."""
    path = sys.path.copy()
    try:
        yield {
            "v2": run_suite(cfg, use_cache=False, variant="v2", trials=TRIALS),
            "v2-again": run_suite(cfg, use_cache=False, variant="v2", trials=TRIALS),
            "flaky": run_suite(cfg, use_cache=False, variant="flaky", trials=TRIALS),
        }
    finally:
        sys.path[:] = path
        sys.modules.pop("weather_agent", None)


def _failed_trials(run: Run) -> set[tuple[str, int]]:
    return {(r.case_id, r.trial) for r in run.results if r.scores["behaviour"].passed is False}


def _forgets(case_id: str, trial: int) -> bool:
    return random.Random(f"{case_id}:{trial}").random() < CHANCE


@pytest.fixture(scope="module")
def files(runs: dict[str, Run], tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("agent-k5")
    return {name: str(save_run(run, root / f"{name}.json")) for name, run in runs.items()}


# --- the suite ---------------------------------------------------------------------------------


def test_the_suite_is_suite_yaml_plus_a_flaky_variant(cfg: SuiteConfig) -> None:
    plain = load_config(EXAMPLE / "suite.yaml")
    assert [v.name for v in cfg.variants] == ["v2", "flaky"]
    assert cfg.variant("flaky").vars == {"calendar_for_outings": True, "flaky_calendar": CHANCE}
    # Same dataset, scorers and gate: only the variants differ.
    assert cfg.dataset == plain.dataset
    assert cfg.scorers == plain.scorers
    assert cfg.gate == plain.gate
    assert cfg.gate is not None and "safety" in cfg.gate.must_pass


# --- what each build does over five trials -----------------------------------------------------


def test_the_run_files_hold_five_trials_per_case(runs: dict[str, Run]) -> None:
    for run in runs.values():
        assert run.trials == TRIALS
        assert len(run.results) == len(run.by_case()) * TRIALS
        assert all(len(trials) == TRIALS for trials in run.by_case().values())
        assert all(r.output.error is None for r in run.results)


def test_the_fixed_build_passes_every_trial_and_is_identical_each_time(
    runs: dict[str, Run],
) -> None:
    assert _failed_trials(runs["v2"]) == set()
    assert [r.scores["behaviour"].passed for r in runs["v2"].results] == [
        r.scores["behaviour"].passed for r in runs["v2-again"].results
    ]


def test_the_flaky_build_fails_exactly_the_seeded_outing_trials(runs: dict[str, Run]) -> None:
    expected = {(cid, t) for cid in OUTING_IDS for t in range(TRIALS) if _forgets(cid, t)}
    assert expected, "the seed should pick some trials"
    assert _failed_trials(runs["flaky"]) == expected
    # Only an outing question consults the calendar, so no other case can be flaky.
    assert {cid for cid, _ in _failed_trials(runs["flaky"])} <= OUTING_IDS
    # The words still read well, which is why only a check on the tool calls sees it.
    assert all(r.scores["correct"].passed for r in runs["flaky"].results)


def test_the_flaky_build_is_deterministic(cfg: SuiteConfig, runs: dict[str, Run]) -> None:
    again = run_suite(cfg, use_cache=False, variant="flaky", trials=TRIALS)
    assert _failed_trials(again) == _failed_trials(runs["flaky"])
    assert [r.output.output for r in again.results] == [
        r.output.output for r in runs["flaky"].results
    ]


def test_every_outing_case_is_flaky_and_none_fails_every_trial(runs: dict[str, Run]) -> None:
    failed = _failed_trials(runs["flaky"])
    for cid in OUTING_IDS:
        n_failed = sum(1 for c, _ in failed if c == cid)
        assert 0 < n_failed < TRIALS, f"{cid} should pass some trials and fail others"


def test_one_run_per_case_sees_less_than_five(cfg: SuiteConfig, runs: dict[str, Run]) -> None:
    """A single trial, the same build: the McNemar test cannot tell it from noise, and the
    decision can differ from the five-trial one. That is the point of the example."""
    assert cfg.gate is not None
    baseline = run_suite(cfg, use_cache=False, variant="v2", trials=1)
    candidate = run_suite(cfg, use_cache=False, variant="flaky", trials=1)
    one = compare_runs(baseline, candidate).metrics["behaviour"]
    five = compare_runs(runs["v2"], runs["flaky"]).metrics["behaviour"]
    assert one.test == "exact McNemar"
    assert five.test.startswith("sign-flip permutation")
    assert one.p_value is not None and five.p_value is not None
    assert five.p_value < cfg.gate.alpha < one.p_value


# --- the gate and readiness exit codes the README and CI promise -------------------------------


def test_the_gate_fails_flaky_and_passes_the_fixed_build(
    cfg: SuiteConfig, runs: dict[str, Run]
) -> None:
    assert cfg.gate is not None
    bug = decide(compare_runs(runs["v2"], runs["flaky"]), cfg.gate, runs["flaky"])
    assert (bug.status, bug.exit_code) == ("fail", EXIT_REGRESSION)
    assert any("significant" in reason or "newly failing" in reason for reason in bug.reasons)
    same = decide(compare_runs(runs["v2"], runs["v2-again"]), cfg.gate, runs["v2-again"])
    assert (same.status, same.exit_code) == ("pass", EXIT_PASS)
    fix = decide(compare_runs(runs["flaky"], runs["v2"]), cfg.gate, runs["v2"])
    assert (fix.status, fix.exit_code) == ("pass", EXIT_PASS)


def test_the_gate_reports_power_and_it_changes_nothing(
    cfg: SuiteConfig, runs: dict[str, Run]
) -> None:
    assert cfg.gate is not None
    decision = decide(compare_runs(runs["v2"], runs["flaky"]), cfg.gate, runs["flaky"])
    power = decision.power["behaviour"]
    assert power.n == len(runs["v2"].by_case())
    assert power.mde is not None and power.mde > cfg.gate.max_drop
    assert power.cases_needed is not None and power.cases_needed > power.n


def test_compare_exits_1_for_flaky_and_0_for_the_fixed_build(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    suite = str(SUITE)

    def gate(baseline: str, *candidates: str) -> int:
        return main(
            [
                "compare",
                files[baseline],
                *(files[c] for c in candidates),
                "--suite",
                suite,
                "--gate",
            ]
        )

    assert gate("v2", "flaky") == EXIT_REGRESSION
    out = capsys.readouterr().out
    assert "o01" in out and "flaky" in out.lower()
    assert gate("v2", "v2-again") == EXIT_PASS
    assert gate("flaky", "v2") == EXIT_PASS
    # Several candidates: the exit code is the worst of them.
    assert gate("v2", "v2-again", "flaky") == EXIT_REGRESSION
    assert gate("v2", "v2-again", "v2-again") == EXIT_PASS


def test_readiness_needs_the_trials_in_a_run_file(
    files: dict[str, str], capsys: pytest.CaptureFixture[str]
) -> None:
    suite = str(SUITE)
    # The static check reads the config: one trial per case is "missing".
    assert main(["readiness", suite, "--require", "reliability"]) == 1
    # A run file with five trials per case covers it.
    capsys.readouterr()
    assert main(["readiness", suite, "--run", files["flaky"], "--require", "reliability"]) == 0
    assert "5 trials per case" in capsys.readouterr().out
