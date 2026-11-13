"""action.yml: the bash step turns its inputs into the right `evalproof compare` arguments.

The step is run for real under `bash -eo pipefail` (how GitHub runs `shell: bash`), with a
stub `uvx` that records its arguments instead of installing and running evalproof.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ACTION = Path(__file__).resolve().parents[1] / "action.yml"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _step_script() -> str:
    steps = yaml.safe_load(ACTION.read_text("utf-8"))["runs"]["steps"]
    return next(s["run"] for s in steps if s.get("id") == "gate")


def _run_step(
    tmp_path: Path, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the step; return the process and the arguments of each `uvx` call, in order.

    The stub exits with `$STUB_VERIFY_EXIT` (default 0) for `evalproof verify`.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "uvx"
    stub.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >> "$STUB_OUT"\n'
        'printf -- "---\\n" >> "$STUB_OUT"\n'
        '[ "$4" = verify ] && exit "${STUB_VERIFY_EXIT:-0}"\n'
        "exit 0\n",
        "utf-8",
    )
    stub.chmod(0o755)
    base = {
        "EP_BASELINE": "base.json",
        "EP_CANDIDATE": "cand.json",
        "EP_SUITE": "",
        "EP_VERIFY": "true",
        "EP_METRIC": "",
        "EP_MAX_DROP": "",
        "EP_ALPHA": "",
        "EP_SOURCE": "/src",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "GITHUB_OUTPUT": str(tmp_path / "output.txt"),
        "STUB_OUT": str(tmp_path / "argv.txt"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _step_script()],
        env=base | env,
        capture_output=True,
        text=True,
        check=False,
    )
    out = tmp_path / "argv.txt"
    calls: list[list[str]] = []
    if out.exists():
        for chunk in out.read_text("utf-8").split("---\n"):
            if chunk:
                calls.append(chunk.splitlines())
    return done, calls


def _argv_for(tmp_path: Path, **env: str) -> list[str]:
    """The arguments the step passes to `uvx --from <source> evalproof compare`."""
    done, calls = _run_step(tmp_path, **env)
    assert done.returncode == 0, done.stderr
    assert "exit-code=0" in (tmp_path / "output.txt").read_text("utf-8")
    argv = calls[-1]
    assert argv[:4] == ["--from", "/src", "evalproof", "compare"]
    return argv[4:]


def _metrics(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, a in enumerate(argv) if a == "--metric"]


def test_no_metric_passes_none(tmp_path: Path) -> None:
    argv = _argv_for(tmp_path)
    assert "--metric" not in argv and argv[:3] == ["base.json", "cand.json", "--gate"]


def test_one_metric_is_unchanged(tmp_path: Path) -> None:
    assert _metrics(_argv_for(tmp_path, EP_METRIC="judge")) == ["judge"]


@pytest.mark.parametrize(
    "value",
    ["judge,exact", "judge\nexact", " judge , exact ", "judge,\n\n exact,\n", "judge\r\nexact\r\n"],
)
def test_several_metrics_become_repeated_flags(tmp_path: Path, value: str) -> None:
    assert _metrics(_argv_for(tmp_path, EP_METRIC=value)) == ["judge", "exact"]


def test_a_metric_name_with_spaces_stays_one_argument(tmp_path: Path) -> None:
    argv = _argv_for(tmp_path, EP_METRIC="Answer Relevancy, judge", EP_ALPHA="0.01")
    assert _metrics(argv) == ["Answer Relevancy", "judge"]
    assert argv[-2:] == ["--alpha", "0.01"]  # the other inputs still follow


def test_no_suite_means_no_verify(tmp_path: Path) -> None:
    done, calls = _run_step(tmp_path)
    assert done.returncode == 0 and [c[3] for c in calls] == ["compare"]


def test_verify_runs_before_compare_on_the_candidate(tmp_path: Path) -> None:
    done, calls = _run_step(tmp_path, EP_SUITE="evals/suite.yaml")
    assert done.returncode == 0, done.stderr
    assert [c[3] for c in calls] == ["verify", "compare"]
    assert calls[0] == [
        "--from",
        "/src",
        "evalproof",
        "verify",
        "cand.json",
        "evals/suite.yaml",
        "--markdown",
        str(tmp_path / "summary.md"),
    ]
    assert "--suite" in calls[1]  # compare still gets the suite
    assert "exit-code=0" in (tmp_path / "output.txt").read_text("utf-8")


def test_a_failed_verify_skips_compare_and_keeps_its_exit_code(tmp_path: Path) -> None:
    done, calls = _run_step(tmp_path, EP_SUITE="evals/suite.yaml", STUB_VERIFY_EXIT="4")
    assert done.returncode == 4
    assert [c[3] for c in calls] == ["verify"]  # compare never ran
    assert (tmp_path / "output.txt").read_text("utf-8").splitlines() == ["exit-code=4"]


def test_verify_can_be_switched_off(tmp_path: Path) -> None:
    done, calls = _run_step(
        tmp_path, EP_SUITE="evals/suite.yaml", EP_VERIFY="false", STUB_VERIFY_EXIT="4"
    )
    assert done.returncode == 0, done.stderr
    assert [c[3] for c in calls] == ["compare"]


def test_the_verify_input_defaults_to_true() -> None:
    inputs = yaml.safe_load(ACTION.read_text("utf-8"))["inputs"]
    assert inputs["verify-candidate"]["default"] == "true"
