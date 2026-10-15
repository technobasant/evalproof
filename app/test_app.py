"""The app renders, its defaults reproduce the recorded run's verdict, and each scenario
lands on the gate outcome it is there to show.

Run with the app's own dependencies, as Streamlit Community Cloud installs them:
``uv pip install -r app/requirements.txt pytest`` then ``python -m pytest app``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).with_name("streamlit_app.py"))


def run_app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=120).run()
    assert not at.exception
    return at


def text(at: AppTest) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_defaults_reproduce_the_recorded_verdict() -> None:
    at = run_app()
    # 8 worse, 4 better of 83 → exact McNemar p = 0.388, a warning and not a failure,
    # in the calculator and in the replay of the committed runs alike.
    assert "0.388" in [m.value for m in at.metric]
    assert text(at).count("## Not distinguishable from noise") == 2
    assert "comes up **39%** of the time" in text(at)
    assert "exit 0 (warn)" in text(at)


@pytest.mark.parametrize(
    ("scenario", "headline", "exit_text"),
    [
        ("Same rates, 4× the cases", "## A real regression", "exit 1 (fail)"),
        ("Small but certain", "## A real drop, inside your tolerance", "exit 0 (warn)"),
        ("Twenty cases", "## Too few cases to gate", "exit 4 (fail)"),
    ],
)
def test_each_scenario_shows_its_outcome(scenario: str, headline: str, exit_text: str) -> None:
    at = run_app()
    at.pills[0].set_value(scenario).run()
    assert not at.exception
    assert headline in text(at)
    assert exit_text in text(at)


def test_latency_budget_fails_the_recorded_run_with_exit_3() -> None:
    at = run_app()
    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert "## Over budget; the quality drop is noise" in text(at)
    assert "p95 latency 4422 ms exceeds 2500 ms" in text(at)


def test_a_changed_answer_is_shown_side_by_side() -> None:
    at = run_app()
    assert "q08 · Is there a discount" in text(at)
    assert "**Before: grounded prompt**" in text(at)
    assert "**After: friendly prompt**" in text(at)


def test_a_pushed_library_change_is_picked_up_without_a_reboot() -> None:
    # Community Cloud pulls a push into the running process; the app must re-import the
    # library rather than keep the copy it imported before the push.
    at = run_app()
    before = sys.modules["evalproof"]
    lib = Path(__file__).resolve().parents[1] / "src" / "evalproof"
    src = min(lib.rglob("*.py"), key=lambda p: p.stat().st_mtime_ns)  # not just the newest
    stat = src.stat()
    try:
        os.utime(src, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9 * 3600))
        at.run()
        assert not at.exception
        assert sys.modules["evalproof"] is not before
    finally:
        os.utime(src, ns=(stat.st_atime_ns, stat.st_mtime_ns))
