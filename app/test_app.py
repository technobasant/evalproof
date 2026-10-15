"""The app renders, and its defaults reproduce the recorded run's published verdict.

Run with the app's own dependencies, as Streamlit Community Cloud installs them:
``uv pip install -r app/requirements.txt pytest`` then ``python -m pytest app``.
"""

from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("streamlit_app.py"))


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def test_defaults_reproduce_the_recorded_verdict() -> None:
    at = run_app()
    # Calculator defaults are the recorded run: 8 worse, 4 better of 83 → p = 0.388, warn.
    assert "0.388" in [m.value for m in at.metric]
    warnings = [w.value for w in at.warning]
    assert sum("**Gate: warn** (exit 0)" in w for w in warnings) == 2  # calculator + replay
    assert any("12 answers that decided it" in s.value for s in at.subheader)


def test_latency_budget_fails_the_recorded_run_with_exit_3() -> None:
    at = run_app()
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert any("**Gate: fail** (exit 3)" in e.value for e in at.error)
    assert any("p95 latency 4422 ms exceeds 2500 ms" in e.value for e in at.error)


def test_counts_that_cannot_happen_are_refused() -> None:
    at = run_app()
    at.number_input[2].set_value(70).run()  # more regressions than baseline passes
    assert not at.exception
    assert any("cannot exceed" in e.value for e in at.error)
