"""The supported Python API: what `from evalproof import ...` promises."""

from __future__ import annotations

from pathlib import Path

import evalproof

ROOT = Path(__file__).resolve().parents[1]


def test_every_public_name_resolves() -> None:
    missing = [name for name in evalproof.__all__ if not hasattr(evalproof, name)]
    assert missing == []


def test_version_has_one_source() -> None:
    # pyproject.toml reads the version from _version.py at build time; nothing else
    # may write it down.
    assert evalproof.__version__.count(".") == 2
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version", "readme"]' in pyproject
    assert "\nversion =" not in pyproject


def test_the_readme_example_runs(tmp_path: Path) -> None:
    # The same calls the README's "From Python" section shows, on the committed runs.
    runs = ROOT / "examples" / "support-qa" / "runs"
    baseline = evalproof.load_run(runs / "baseline.json")
    candidate = evalproof.load_run(runs / "candidate.json")
    cmp = evalproof.compare_runs(baseline, candidate)
    decision = evalproof.decide(cmp, evalproof.GateConfig(metric="judge", max_drop=0.03))
    assert decision.exit_code == evalproof.EXIT_PASS
    assert decision.status == "warn"
    report = evalproof.calibrate(baseline, "judge", gold_scorer="answer_key", gold_sample=30)
    assert report.n_gold == 30
    assert "Gate" in evalproof.comparison_markdown(cmp, decision)
