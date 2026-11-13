"""The run and compare reports that tell a developer what failed, and the small CLI fixes
that go with them: the label for a prompt file, and `import --quiet`."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

import evalproof.cli as cli
from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, decide
from evalproof.models import Score
from evalproof.report import comparison_markdown, run_markdown
from evalproof.runner import save_run
from tests.conftest import make_run

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "deterministic"
PF = Path(__file__).parent / "fixtures" / "promptfoo" / "sample-export.json"


def _failing_run(n_failed: int = 3, *, errored: bool = False):
    outcomes = {f"ok{i}": {"m": True} for i in range(5)}
    outcomes |= {f"bad{i}": {"m": False} for i in range(n_failed)}
    return make_run(outcomes, errors={"bad0"} if errored else None)


def test_run_report_lists_failed_cases_with_reason() -> None:
    run = _failing_run()
    run.results[5].scores["m"] = Score(value=0.0, passed=False, reason="expected 'refund'")
    md = run_markdown(run)
    assert "### Failed cases (3)" in md
    assert "- `bad0` · failed on m · expected 'refund'" in md
    assert "`ok0`" not in md


def test_run_report_labels_errored_failed_and_undecided() -> None:
    run = make_run(
        {"e": {"m": False}, "f": {"m": False}, "u": {"m": None}, "p": {"m": True}},
        errors={"e"},
    )
    md = run_markdown(run)
    assert "- `e` · errored on m · boom" in md  # the system error outranks the scorer's failure
    assert "- `f` · failed on m" in md
    assert "- `u` · undecided on m" in md
    assert "`p`" not in md


def test_run_report_caps_the_list_and_show_zero_hides_it() -> None:
    run = _failing_run(25)
    md = run_markdown(run)
    assert "### Failed cases (25)" in md
    assert md.count("- `bad") == 10 and "… and 15 more" in md
    assert "- `bad2` " in run_markdown(run, 3) and "… and 22 more" in run_markdown(run, 3)
    assert "Failed cases" not in run_markdown(run, 0)


def test_run_report_clips_a_long_reason_and_has_no_section_when_all_pass() -> None:
    run = _failing_run(1)
    run.results[5].scores["m"] = Score(value=0.0, passed=False, reason="why " * 100)
    line = next(ln for ln in run_markdown(run).splitlines() if "`bad0`" in ln)
    assert line.endswith("…") and len(line.split(" · ", 2)[2]) == 120
    assert "Failed cases" not in run_markdown(make_run({"a": {"m": True}}))


def _compare_pair(regress: int = 4):
    base = make_run({f"c{i}": {"m": True, "s": 0.9} for i in range(10)})
    cand = make_run(
        {f"c{i}": {"m": i >= regress, "s": 0.9 if i >= regress else 0.1} for i in range(10)}
    )
    for res in base.results:
        res.output.output = f"baseline answer for {res.case_id}"
    for res in cand.results:
        res.output.output = f"candidate answer for {res.case_id}"
        if res.scores["m"].passed is False:
            res.scores["m"] = Score(
                value=0.0, passed=False, reason=f"{res.case_id} missed the point"
            )
    return base, cand


def test_compare_lists_regressed_ids_for_every_metric_not_only_gated() -> None:
    base, cand = _compare_pair()
    cmp = compare_runs(base, cand)
    md = comparison_markdown(cmp, None, "m")
    assert "Regressed on **m** (4): `c0`, `c1`, `c2`, `c3`" in md
    assert "Regressed on s (4)" in md  # ungated, plain name


def test_compare_caps_ids_at_ten_per_metric() -> None:
    base = make_run({f"c{i:02d}": {"m": True} for i in range(40)})
    cand = make_run({f"c{i:02d}": {"m": False} for i in range(40)})
    md = comparison_markdown(compare_runs(base, cand), None, "m")
    assert "`c09`" in md and "`c10`" not in md and "and 30 more" in md


def test_compare_shows_outputs_and_reason_for_regressed_cases() -> None:
    base, cand = _compare_pair()
    cmp = compare_runs(base, cand)
    md = comparison_markdown(cmp, None, ["m"], runs=(base, cand), show=2)
    assert md.count("<details>") == 2
    assert "baseline answer for c0" in md and "candidate answer for c0" in md
    assert "**Candidate's m reason:** c0 missed the point" in md
    assert "c2 regressed on m" not in md  # only two examples
    assert comparison_markdown(cmp, None, ["m"], runs=(base, cand), show=0).count("<details>") == 0
    assert "<details>" not in comparison_markdown(cmp, None, ["m"])


def test_compare_examples_cover_only_gated_metrics() -> None:
    base, cand = _compare_pair()
    md = comparison_markdown(compare_runs(base, cand), None, ["s"], runs=(base, cand), show=5)
    assert md.count("<details>") == 4 and "regressed on s" in md and "regressed on m" not in md


def test_compare_examples_clip_long_outputs_and_survive_code_fences() -> None:
    base, cand = _compare_pair()
    base.results[0].output.output = "A" * 1000
    cand.results[0].output.output = "```\nclose the fence\n```"
    md = comparison_markdown(compare_runs(base, cand), None, ["m"], runs=(base, cand), show=1)
    assert "A" * 400 not in md and "A" * 399 + "…" in md
    # The candidate's own fence cannot close ours: ours is longer.
    assert "````\n``` close the fence ```\n````" in md


def test_compare_example_shows_a_system_error_for_an_errored_candidate() -> None:
    base, cand = _compare_pair()
    cand.results[0].output.error = "timeout after 30s"
    md = comparison_markdown(compare_runs(base, cand), None, ["m"], runs=(base, cand), show=1)
    assert "system error: timeout after 30s" in md


def test_gate_that_cannot_decide_says_cannot_gate_not_fail() -> None:
    base = make_run({f"c{i}": {"m": True} for i in range(5)}, dataset_hash="a")
    cand = make_run({f"c{i}": {"m": True} for i in range(5)}, dataset_hash="b")
    cmp = compare_runs(base, cand)
    decision = decide(cmp, GateConfig(metric="m", min_cases=1))
    assert decision.exit_code == EXIT_CANNOT_GATE
    md = comparison_markdown(cmp, decision, "m")
    assert "**Gate: ⚠️ cannot gate** (exit 4)" in md and "❌ fail" not in md


def test_real_regression_still_says_fail() -> None:
    base, cand = _compare_pair(regress=8)
    cmp = compare_runs(base, cand)
    decision = decide(cmp, GateConfig(metric="m", min_cases=5))
    assert "❌ fail" in comparison_markdown(cmp, decision, "m")


def test_compare_cli_show_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    base, cand = _compare_pair()
    b, c = tmp_path / "b.json", tmp_path / "c.json"
    save_run(base, b)
    save_run(cand, c)
    assert main(["compare", str(b), str(c), "--metric", "m", "--min-cases", "5"]) == EXIT_PASS
    assert capsys.readouterr().out.count("<details>") == 4  # all four regressions, default 5
    main(["compare", str(b), str(c), "--metric", "m", "--min-cases", "5", "--show", "1"])
    assert capsys.readouterr().out.count("<details>") == 1
    main(["compare", str(b), str(c), "--metric", "m", "--min-cases", "5", "--show", "0"])
    assert "<details>" not in capsys.readouterr().out


@pytest.fixture
def example(tmp_path: Path) -> Path:
    dst = tmp_path / "deterministic"
    shutil.copytree(
        EXAMPLE, dst, ignore=shutil.ignore_patterns("runs", ".evalproof", "__pycache__")
    )
    return dst


@pytest.fixture
def labels(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """The label `run` hands to the runner. A python target ignores labels, so this watches
    the override instead of reading it back from the run file."""
    seen: list[str | None] = []
    real = cli.run_suite

    def spy(cfg, **kw):
        seen.append(kw["system_override"].label if kw["system_override"] else cfg.system.label)
        return real(cfg, **kw)

    monkeypatch.setattr(cli, "run_suite", spy)
    return seen


def _run_with(example: Path, tmp_path: Path, *extra: str) -> None:
    argv = ["run", str(example / "suite.yaml"), "--out", str(tmp_path / "r.json"), "--quiet"]
    assert main([*argv, *extra]) == EXIT_PASS


def test_prompt_file_without_label_sets_a_telling_label(
    example: Path, tmp_path: Path, labels: list[str | None]
) -> None:
    prompt = tmp_path / "v2-prompt.txt"
    prompt.write_text("Be brief.", "utf-8")
    _run_with(example, tmp_path, "--system-prompt-file", str(prompt))
    assert labels == ["python:router:baseline_route + v2-prompt.txt"]


def test_prompt_file_with_label_keeps_the_label(
    example: Path, tmp_path: Path, labels: list[str | None]
) -> None:
    prompt = tmp_path / "v2-prompt.txt"
    prompt.write_text("Be brief.", "utf-8")
    _run_with(example, tmp_path, "--system-prompt-file", str(prompt), "--label", "mine")
    assert labels == ["mine"]


def test_prompt_file_label_builds_on_the_suite_label(
    example: Path, tmp_path: Path, labels: list[str | None]
) -> None:
    suite = example / "suite.yaml"
    suite.write_text(suite.read_text("utf-8").replace("system:\n", "system:\n  label: router\n"))
    prompt = tmp_path / "p.txt"
    prompt.write_text("x", "utf-8")
    _run_with(example, tmp_path, "--system-prompt-file", str(prompt))
    assert labels == ["router + p.txt"]


def test_no_prompt_file_leaves_the_label_alone(
    example: Path, tmp_path: Path, labels: list[str | None]
) -> None:
    _run_with(example, tmp_path)
    assert labels == [None]


def test_import_quiet_keeps_stdout_clean_but_still_writes_markdown(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = PF
    out, md = tmp_path / "run.json", tmp_path / "report.md"
    base = ["import", "promptfoo", str(src), "--system", "openai:gpt-5", "--out", str(out)]
    assert main([*base, "--markdown", str(md)]) == EXIT_PASS
    loud = capsys.readouterr()
    assert "| " in loud.out
    out.unlink()
    md.unlink()
    assert main([*base, "--markdown", str(md), "--quiet"]) == EXIT_PASS
    quiet = capsys.readouterr()
    assert quiet.out == "" and "wrote" in quiet.err
    assert md.read_text("utf-8").strip() and json.loads(out.read_text("utf-8"))
