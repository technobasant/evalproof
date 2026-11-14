"""What reads a run with repeated trials: the run report, verify, rehash, calibration, readiness
and the importers."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pytest

from evalproof.calibration import calibrate
from evalproof.cli import main
from evalproof.config import load_config
from evalproof.importers import ImportFormatError, import_run
from evalproof.models import Run, Score
from evalproof.readiness import COVERED, MISSING, PARTIAL
from evalproof.rehash import rehash_run
from evalproof.report import calibration_markdown, run_markdown
from evalproof.runner import load_run, run_suite, save_run
from evalproof.trials import case_counts, is_flaky, pooled_rate_ci, trial_outcome
from evalproof.verify import verify_run
from tests.conftest import make_run
from tests.test_importers import _jsonl, _pf_row
from tests.test_readiness import _assess
from tests.test_readiness import _write as _readiness_suite
from tests.test_verify import _write as _verify_suite


def _repeated(
    outcomes: dict[str, list[bool | float | None]],
    *,
    scorer: str = "m",
    errors: Iterable[tuple[str, int]] = (),
    same_output: bool = True,
) -> Run:
    """A run with ``len(trials)`` trials per case, from ``{case: [outcome of trial 0, 1, ...]}``."""
    failed = set(errors)
    k = len(next(iter(outcomes.values())))
    results = []
    for cid, values in outcomes.items():
        for trial, value in enumerate(values):
            errs = {cid} if (cid, trial) in failed else None
            one = make_run({cid: {scorer: value}}, errors=errs).results[0]
            text = one.output.output if same_output else f"{cid}-{trial}"
            one.output = one.output.model_copy(update={"output": text})
            results.append(one.model_copy(update={"trial": trial}))
    info = make_run({"x": {}}).info.model_copy(update={"trials": k})
    return Run(info=info, results=results)


# --- the helpers -------------------------------------------------------------------------------


def test_trial_outcome_and_case_counts() -> None:
    run = _repeated(
        {"a": [True, True, True], "b": [True, False, True], "u": [None, True, False]},
        errors=[("a", 2)],
    )
    a, b, u = run.by_case()["a"], run.by_case()["b"], run.by_case()["u"]
    assert [trial_outcome(r, "m") for r in a] == [True, True, False]  # an error is a failure
    assert [trial_outcome(r, "m") for r in b] == [True, False, True]
    assert [trial_outcome(r, "m") for r in u] == [None, True, False]  # undecided is not counted
    assert trial_outcome(a[0], "other") is None
    assert case_counts(run, "m") == {"a": (2, 3), "b": (2, 3), "u": (1, 2)}
    assert is_flaky(2, 3) and is_flaky(1, 2) and not is_flaky(3, 3) and not is_flaky(0, 3)


def test_a_continuous_scorer_has_no_counts() -> None:
    run = _repeated({"a": [0.2, 0.4], "b": [0.9, 1.0]}, errors=[("a", 0)])
    assert case_counts(run, "m") == {}


def test_the_pooled_interval_resamples_cases_and_is_reproducible() -> None:
    counts = [(3, 3), (0, 3), (2, 3), (1, 3)]
    lo, hi = pooled_rate_ci(counts)
    assert 0.0 <= lo <= 0.5 <= hi <= 1.0
    assert pooled_rate_ci(counts) == (lo, hi)
    assert pooled_rate_ci([(2, 3)]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))


# --- the run report ----------------------------------------------------------------------------


def test_the_report_gives_pass_at_k_pass_hat_k_and_the_flaky_cases() -> None:
    run = _repeated(
        {
            "always": [True, True, True],
            "mostly": [True, False, True],
            "never": [False, False, False],
            "once": [True, False, False],
        },
        same_output=False,
    )
    text = run_markdown(run)
    assert "4 cases × 3 trials = 12 runs" in text
    assert (
        "| Scorer | Decided | Pass rate (95% CI) | pass@3 | pass^3 | Mean | Within-case SD |"
        in text
    )
    row = next(line for line in text.splitlines() if line.startswith("| m |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1] == "12/12"
    assert cells[2].startswith("50.0% [") and cells[2].endswith("]")  # 6 of 12 passes
    assert cells[3] == "75.0%"  # three of four cases passed at least once
    assert cells[4] == "25.0%"  # one case passed every trial
    assert cells[6] == "—"  # a binary scorer has no within-case SD
    assert "### Flaky cases (2)" in text
    assert "- `mostly` · m 2/3" in text and "- `once` · m 1/3" in text
    assert "always" not in text.split("### Flaky cases")[1].split("###")[0]
    assert run_markdown(run) == text  # the bootstrap is seeded


def test_an_errored_trial_is_a_failure_and_an_undecided_one_is_left_out() -> None:
    run = _repeated(
        {"e": [True, True, True], "u": [None, True, True]},
        errors=[("e", 1)],
        same_output=False,
    )
    text = run_markdown(run)
    assert "### Flaky cases (1)" in text and "- `e` · m 2/3" in text  # not `u`: 2/2 decided
    row = next(line for line in text.splitlines() if line.startswith("| m |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[3] == "100.0%" and cells[4] == "50.0%"  # u passed every decided trial; e did not


def test_a_continuous_scorer_reports_the_mean_of_case_means_and_the_within_case_sd() -> None:
    run = _repeated({"a": [0.2, 0.4, 0.6], "b": [1.0, 1.0, 1.0]}, same_output=False)
    row = next(line for line in run_markdown(run).splitlines() if line.startswith("| m |"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[2:5] == ["—", "—", "—"]
    assert cells[5] == "0.700"  # (0.4 + 1.0) / 2
    assert cells[6] == "0.100"  # (0.2 + 0.0) / 2


def test_the_report_says_when_every_case_gave_identical_outputs() -> None:
    same = run_markdown(_repeated({"a": [True, True], "b": [True, False]}, same_output=True))
    assert "All trials identical: the system may be deterministic at these settings" in same
    varied = run_markdown(_repeated({"a": [True, True], "b": [True, False]}, same_output=False))
    assert "All trials identical" not in varied
    assert "All trials identical" not in run_markdown(make_run({"a": {"m": True}}))


def test_a_run_with_no_flaky_case_says_so() -> None:
    text = run_markdown(_repeated({"a": [True, True, True], "b": [False, False, False]}))
    assert "No flaky cases" in text and "### Flaky cases" not in text


def test_failed_trials_are_named_by_trial_and_the_lists_obey_max_cases() -> None:
    run = _repeated({"a": [True, False, False], "b": [False, True, True]}, same_output=False)
    text = run_markdown(run)
    assert "### Failed trials (3)" in text
    assert "- `a` trial 1 ·" in text and "- `a` trial 2 ·" in text and "- `b` trial 0 ·" in text
    short = run_markdown(run, max_cases=1)
    assert "- … and 1 more" in short and "- … and 1 more" in short.split("### Failed")[0]
    off = run_markdown(run, max_cases=0)
    assert "### Failed" not in off and "### Flaky" not in off


def test_a_single_trial_report_is_as_it_was() -> None:
    run = make_run({"a": {"m": True}, "b": {"m": False}})
    text = run_markdown(run)
    assert "2 cases · wall 1.0 s · dataset `h`" in text
    assert "| Scorer | Decided | Pass rate (95% CI) | Mean |" in text
    assert "pass@" not in text and "trial" not in text.lower().replace("trials", "")
    assert "### Failed cases (1)" in text and "- `b` · failed on m" in text


def _flaky_suite(tmp_path: Path, module: str) -> Path:
    modname = f"flakymod_{tmp_path.name}"  # python caches modules, and each test has its own
    cases = [{"id": f"c{i}", "input": f"q{i}"} for i in range(4)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), "utf-8")
    (tmp_path / f"{modname}.py").write_text(module, "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"name: s\ndataset: cases.jsonl\nsystem: {{target: 'python:{modname}:ask'}}\n"
        "scorers: [{name: m, type: contains, params: {any_of: [ok]}}]\n",
        "utf-8",
    )
    return suite


def test_a_seeded_flaky_python_system_reports_its_flakiness(tmp_path: Path) -> None:
    module = (
        "PATTERN = {'c0': 'ppp', 'c1': 'fff', 'c2': 'pfp', 'c3': 'ffp'}\n"
        "def ask(x, ctx):\n"
        "    return 'ok' if PATTERN[ctx.case_id][ctx.trial] == 'p' else 'bad'\n"
    )
    run = run_suite(load_config(_flaky_suite(tmp_path, module)), use_cache=False, trials=3)
    text = run_markdown(run)
    assert "4 cases × 3 trials = 12 runs" in text
    assert "50.0% [" in text and "| 75.0% | 25.0% |" in text
    assert "- `c2` · m 2/3" in text and "- `c3` · m 1/3" in text
    assert "All trials identical" not in text


def test_a_deterministic_python_system_is_reported_as_identical(tmp_path: Path) -> None:
    module = "def ask(x):\n    return 'ok'\n"
    run = run_suite(load_config(_flaky_suite(tmp_path, module)), use_cache=False, trials=3)
    text = run_markdown(run)
    assert "All trials identical" in text and "No flaky cases" in text


def test_the_run_command_prints_the_repeated_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = "def ask(x, ctx):\n    return 'ok' if ctx.trial != 1 else 'bad'\n"
    suite = _flaky_suite(tmp_path, module)
    out = tmp_path / "run.json"
    assert main(["run", str(suite), "--trials", "3", "--out", str(out), "--no-cache"]) == 0
    printed = capsys.readouterr()
    assert "pass@3" in printed.out and "### Flaky cases (4)" in printed.out  # 2 of 3 on each
    assert load_run(out).trials == 3


# --- verify and rehash -------------------------------------------------------------------------


def _verified(tmp_path: Path, trials: int = 3):
    cfg = load_config(_verify_suite(tmp_path))
    return cfg, run_suite(cfg, use_cache=False, trials=trials)


def test_a_run_with_trials_verifies_and_every_trial_is_re_scored(tmp_path: Path) -> None:
    cfg, run = _verified(tmp_path)
    result = verify_run(run, cfg)
    assert result.ok and result.problems == [] and result.rescored == ["m"]
    run.results[7].scores["m"] = Score(value=0.0, passed=False)  # c02, trial 1
    bad = verify_run(run, cfg)
    assert not bad.ok
    assert any("`c02` trial 1 on `m`" in p for p in bad.problems)


def test_a_missing_or_extra_trial_is_a_problem(tmp_path: Path) -> None:
    cfg, run = _verified(tmp_path)
    run.results = [r for r in run.results if not (r.case_id == "c03" and r.trial == 1)]
    problems = verify_run(run, cfg).problems
    assert any("`c03` has trials [0, 2], expected 0..2" in p for p in problems)

    (tmp_path / "again").mkdir()
    cfg, run = _verified(tmp_path / "again")
    run.info.trials = 2
    problems = verify_run(run, cfg).problems
    assert any("trials [0, 1, 2], expected 0..1" in p for p in problems)


def test_a_duplicate_case_and_trial_is_a_problem(tmp_path: Path) -> None:
    cfg, run = _verified(tmp_path)
    run.results.append(run.results[0].model_copy(deep=True))
    problems = verify_run(run, cfg).problems
    assert any("duplicate results for one case and trial" in p for p in problems)

    (tmp_path / "one").mkdir()
    cfg, run = _verified(tmp_path / "one", trials=1)
    run.results.append(run.results[0].model_copy(deep=True))
    assert "the run has duplicate case ids" in verify_run(run, cfg).problems


def test_one_stale_trial_makes_its_case_changed(tmp_path: Path) -> None:
    cfg, run = _verified(tmp_path)
    run.results[4].case_hash = "stale"  # c01, trial 1
    problems = verify_run(run, cfg).problems
    assert any("1 case(s) changed" in p and "`c01`" in p for p in problems)


def test_a_trials_override_does_not_break_verify(tmp_path: Path) -> None:
    suite = _verify_suite(tmp_path)
    out = tmp_path / "run.json"
    assert main(["run", str(suite), "--trials", "3", "--out", str(out), "--no-cache"]) == 0
    assert load_run(out).trials == 3
    assert main(["verify", str(out), str(suite)]) == 0


def test_rehash_restores_the_hashes_of_every_trial_of_a_limited_run(tmp_path: Path) -> None:
    cfg = load_config(_verify_suite(tmp_path))
    fresh = run_suite(cfg, use_cache=False, trials=2, limit=5)
    old = fresh.model_copy(deep=True)
    for r in old.results:
        r.case_hash = r.content_hash = None
    old.info.suite_hash = None
    assert len(old.results) == 10
    out = rehash_run(old, cfg)
    assert [(r.case_id, r.trial, r.case_hash, r.content_hash) for r in out.results] == [
        (r.case_id, r.trial, r.case_hash, r.content_hash) for r in fresh.results
    ]
    assert all(r.case_hash and r.content_hash for r in out.results)


def test_the_rehash_message_counts_cases_and_trials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _verify_suite(tmp_path)
    run = run_suite(load_config(suite), use_cache=False, trials=2)
    for r in run.results:
        r.case_hash = r.content_hash = None
    path = save_run(run, tmp_path / "old.json")
    assert main(["rehash", str(path), str(suite)]) == 0
    assert f"rehashed 12 cases × 2 trials in {path}" in capsys.readouterr().err


# --- calibration -------------------------------------------------------------------------------


def test_calibration_reads_trial_zero_only() -> None:
    # Trial 0 agrees with the labels everywhere; trial 1 disagrees everywhere.
    run = _repeated(
        {"a": [True, False], "b": [False, True], "c": [True, False], "d": [False, True]},
        same_output=False,
    )
    labels = {"a": True, "b": False, "c": True, "d": False}
    rep = calibrate(run, "m", labels=labels, n_boot=200)
    cm = rep.confusion
    assert (cm.tp, cm.tn, cm.fp, cm.fn) == (2, 2, 0, 0)
    assert rep.n_judged == 4 and rep.trials == 2
    assert "Trial 0 of 2" in calibration_markdown(rep)


def test_calibration_of_a_single_trial_run_does_not_mention_trials() -> None:
    run = make_run({"a": {"m": True}, "b": {"m": False}, "c": {"m": True}, "d": {"m": False}})
    rep = calibrate(run, "m", labels={"a": True, "b": False, "c": True, "d": False}, n_boot=200)
    assert rep.trials == 1 and "Trial 0 of" not in calibration_markdown(rep)


def test_calibration_gold_scorer_uses_trial_zero_of_each() -> None:
    run = _repeated({"a": [True, False], "b": [False, True]}, scorer="judge", same_output=False)
    gold = _repeated({"a": [True, True], "b": [False, False]}, scorer="gold", same_output=False)
    for r, g in zip(run.results, gold.results, strict=True):
        r.scores["gold"] = g.scores["gold"]
    rep = calibrate(run, "judge", gold_scorer="gold", n_boot=200)
    assert rep.n_judged == 2 and rep.n_gold == 2
    assert (rep.confusion.tp, rep.confusion.tn) == (1, 1)


# --- readiness ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("trials", "status", "evidence"),
    [
        (1, MISSING, "one trial per case: set `trials: 3` or more"),
        (2, PARTIAL, "two trials show disagreement but not a rate; use 3 or more"),
        (3, COVERED, "3 trials per case"),
        (10, COVERED, "10 trials per case"),
    ],
)
def test_readiness_reliability_follows_the_suite_trials(
    tmp_path: Path, trials: int, status: str, evidence: str
) -> None:
    report = _assess(_readiness_suite(tmp_path, extra=f"trials: {trials}\n"))
    assert report.get("reliability").status == status
    assert report.get("reliability").evidence == evidence


def test_readiness_with_a_run_reads_the_trials_the_run_had(tmp_path: Path) -> None:
    suite = _readiness_suite(tmp_path)  # the suite itself says one trial
    many = _repeated({"c0": [True, True, True]})
    assert _assess(suite, run=many).get("reliability").status == COVERED
    two = _repeated({"c0": [True, True]})
    assert _assess(suite, run=two).get("reliability").status == PARTIAL
    (tmp_path / "three").mkdir()
    suite = _readiness_suite(tmp_path / "three", extra="trials: 3\n")
    assert _assess(suite, run=make_run({"c0": {"m": True}})).get("reliability").status == MISSING


# --- importers ---------------------------------------------------------------------------------


def _repeat_rows(group: str, question: str, passes: list[bool], start: int = 0) -> list[dict]:
    return [
        _pf_row(start + i, question, ok, repeatGroupId=group, repeatIndex=i)
        for i, ok in enumerate(passes)
    ]


def test_epochs_trials_keeps_each_repeat_as_a_trial(tmp_path: Path) -> None:
    rows = _repeat_rows("g", "q", [True, True, False])
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs="trials")
    assert run.info.trials == 3 and run.trials == 3
    assert [r.trial for r in run.results] == [0, 1, 2]
    assert len({r.case_id for r in run.results}) == 1
    assert [r.scores["pass"].passed for r in run.results] == [True, True, False]
    assert len({r.case_hash for r in run.results}) == 1 and run.results[0].case_hash
    assert run.info.environment["epochs"] == "trials"
    assert "pass@3" in run_markdown(run)


def test_epochs_trials_orders_cases_then_trials(tmp_path: Path) -> None:
    rows = _repeat_rows("g1", "first", [True, False]) + _repeat_rows(
        "g2", "second", [False, True], 2
    )
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs="trials")
    assert run.info.trials == 2 and len(run.results) == 4
    assert [r.trial for r in run.results] == [0, 1, 0, 1]
    grouped = run.by_case()
    assert [len(v) for v in grouped.values()] == [2, 2]
    assert [r.scores["pass"].passed for r in list(grouped.values())[1]] == [False, True]


def test_epochs_trials_on_unrepeated_cases_is_one_trial(tmp_path: Path) -> None:
    rows = [_pf_row(i, f"q{i}", i % 2 == 0) for i in range(3)]
    path = _jsonl(tmp_path / "r.jsonl", rows)
    kept = import_run("promptfoo", path, epochs="trials")
    first = import_run("promptfoo", path, epochs="first")
    assert kept.info.trials == 1 and kept.results == first.results


def test_epochs_trials_a_system_error_is_a_failed_trial(tmp_path: Path) -> None:
    rows = _repeat_rows("g", "q", [True, True])
    rows[1].update(error="429 Too Many Requests", success=False, failureReason=2)
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs="trials")
    assert [r.scores["pass"].passed for r in run.results] == [True, False]
    assert run.results[1].output.error


def test_epochs_trials_needs_equal_repeats_and_says_how_to_collapse(tmp_path: Path) -> None:
    rows = _repeat_rows("g1", "first", [True, False]) + _repeat_rows("g2", "second", [True] * 3, 2)
    with pytest.raises(ImportFormatError) as exc:
        import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs="trials")
    message = str(exc.value)
    assert "--epochs trials needs the same number of repeats" in message
    assert "has 2" in message and "has 3" in message and "--epochs mean" in message


def test_the_import_command_takes_epochs_trials(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    good = _jsonl(tmp_path / "good.jsonl", _repeat_rows("g", "q", [True, False, True]))
    out = tmp_path / "run.json"
    assert main(["import", "promptfoo", str(good), "--out", str(out), "--epochs", "trials"]) == 0
    assert load_run(out).trials == 3
    assert "pass@3" in capsys.readouterr().out
    rows = _repeat_rows("g1", "a", [True, True]) + _repeat_rows("g2", "b", [True], 2)
    bad = _jsonl(tmp_path / "bad.jsonl", rows)
    args = ["import", "promptfoo", str(bad), "--out", str(tmp_path / "bad.json")]
    assert main([*args, "--epochs", "trials"]) == 2
    assert "--epochs mean" in capsys.readouterr().err
    assert not (tmp_path / "bad.json").exists()


@pytest.mark.parametrize("epochs", ["mean", "first", "all", "any"])
def test_the_other_epochs_modes_still_collapse_to_one_trial(tmp_path: Path, epochs: str) -> None:
    rows = _repeat_rows("g", "q", [True, True, False])
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs=epochs)
    assert run.info.trials == 1 and len(run.results) == 1 and run.results[0].trial == 0
