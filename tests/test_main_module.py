"""``python -m evalproof`` runs the same CLI as the ``evalproof`` command."""

from __future__ import annotations

import subprocess
import sys

from evalproof._version import __version__


def run_module(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "evalproof", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
        check=False,
    )


def test_python_dash_m_evalproof_prints_the_version() -> None:
    done = run_module("--version")
    assert done.returncode == 0
    assert done.stdout.strip() == f"evalproof {__version__}"


def test_python_dash_m_evalproof_exits_with_the_command_return_code() -> None:
    done = run_module("run", "no-such-suite.yaml")
    assert done.returncode == 2  # the usage/config error code the evalproof command gives
    assert "no-such-suite.yaml" in done.stderr
