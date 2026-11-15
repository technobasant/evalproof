"""action.yml: the bash steps turn the inputs into the right `evalproof compare` arguments,
and post the report as one pull request comment.

Each step is run for real under `bash -eo pipefail` (how GitHub runs `shell: bash`). A stub
`uvx` records its arguments instead of installing and running evalproof, and a stub `gh`
records its calls instead of reaching GitHub.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ACTION = Path(__file__).resolve().parents[1] / "action.yml"
MARKER = "<!-- evalproof-report -->"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="needs bash, a #!/bin/bash stub on a ':'-separated PATH (not Windows)",
)


def _steps() -> list[dict[str, object]]:
    steps = yaml.safe_load(ACTION.read_text("utf-8"))["runs"]["steps"]
    assert isinstance(steps, list)
    return steps


def _step_script(step_id: str = "gate") -> str:
    return next(str(s["run"]) for s in _steps() if s.get("id") == step_id)


def _run_step(
    tmp_path: Path, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the step; return the process and the arguments of each `uvx` call, in order.

    The stub exits with `$STUB_VERIFY_EXIT` (default 0) for `evalproof verify` and with
    `$STUB_COMPARE_EXIT` (default 0) for `evalproof compare`. It appends one line to the
    `--markdown` file, as the real command does.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "uvx"
    stub.write_text(
        "#!/bin/bash\n"
        'printf "%s\\n" "$@" >> "$STUB_OUT"\n'
        'printf -- "---\\n" >> "$STUB_OUT"\n'
        'prev=""\n'
        'for a in "$@"; do\n'
        '  [ "$prev" = --markdown ] && printf "report of %s\\n" "$4" >> "$a"\n'
        '  prev="$a"\n'
        "done\n"
        '[ "$4" = verify ] && exit "${STUB_VERIFY_EXIT:-0}"\n'
        'exit "${STUB_COMPARE_EXIT:-0}"\n',
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
        "RUNNER_TEMP": str(tmp_path),
        "STUB_OUT": str(tmp_path / "argv.txt"),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _step_script()],
        env=base | env,
        capture_output=True,
        text=True,
        encoding="utf-8",
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


def test_the_gate_keeps_a_copy_of_the_report_for_the_comment_step(tmp_path: Path) -> None:
    done, _ = _run_step(tmp_path)
    assert done.returncode == 0, done.stderr
    # Every step has its own summary file, so the gate step leaves the report in RUNNER_TEMP.
    assert (tmp_path / "evalproof-report.md").read_text("utf-8") == "report of compare\n"


# --- the pull request comment ---------------------------------------------------------------

REPORT = "## evalproof\n\nthe report\n"


def _run_comment(
    tmp_path: Path, *, report: str | None = REPORT, **env: str
) -> tuple[subprocess.CompletedProcess[str], list[list[str]]]:
    """Run the comment step with a stub `gh`; return the process and each `gh` call's arguments.

    The stub keeps the body file it was given (`-F body=@file`) as `gh/body-N.md`, and the token
    as `gh/token-N`. `GH_LIST_IDS` is what the comment lookup prints; `GH_LIST_EXIT` and
    `GH_WRITE_EXIT` make the lookup or the post/update fail.
    """
    bin_dir = tmp_path / "ghbin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "gh"
    log.mkdir(exist_ok=True)
    stub = bin_dir / "gh"
    stub.write_text(
        "#!/bin/bash\n"
        'n=$(( $(cat "$GH_DIR/count" 2>/dev/null || echo 0) + 1 ))\n'
        'echo "$n" > "$GH_DIR/count"\n'
        'printf "%s\\n" "$@" > "$GH_DIR/call-$n.txt"\n'
        'printf "%s" "$GH_TOKEN" > "$GH_DIR/token-$n"\n'
        'for a in "$@"; do\n'
        '  case "$a" in body=@*) cp "${a#body=@}" "$GH_DIR/body-$n.md" ;; esac\n'
        "done\n"
        'case "$*" in\n'
        "  *--paginate*)\n"
        '    [ "${GH_LIST_EXIT:-0}" = 0 ] || { echo "list failed" >&2; exit "$GH_LIST_EXIT"; }\n'
        '    printf "%s" "$GH_LIST_IDS"; exit 0 ;;\n'
        "esac\n"
        'if [ "${GH_WRITE_EXIT:-0}" != 0 ]; then\n'
        '  echo "HTTP 403: Resource not accessible by integration" >&2; exit "$GH_WRITE_EXIT"\n'
        "fi\n"
        "echo '{}'\n",
        "utf-8",
    )
    stub.chmod(0o755)
    if report is not None:
        (tmp_path / "evalproof-report.md").write_text(report, "utf-8")
    base = {
        "EP_COMMENT": "true",
        "EP_PR": "7",
        "EP_REPO": "octo/repo",
        "GH_TOKEN": "token-from-input",
        "GH_DIR": str(log),
        "GH_LIST_IDS": "",
        "RUNNER_TEMP": str(tmp_path),
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
    }
    done = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", _step_script("comment")],
        env=base | env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    calls = [
        (log / f"call-{n}.txt").read_text("utf-8").splitlines()
        for n in range(1, len(list(log.glob("call-*.txt"))) + 1)
    ]
    return done, calls


def _body(tmp_path: Path, n: int) -> str:
    return (tmp_path / "gh" / f"body-{n}.md").read_text("utf-8")


def test_the_comment_inputs_default_to_off_and_the_workflow_token() -> None:
    inputs = yaml.safe_load(ACTION.read_text("utf-8"))["inputs"]
    assert inputs["pr-comment"]["default"] == "false"
    assert inputs["github-token"]["default"] == "${{ github.token }}"


def test_every_run_step_names_its_shell() -> None:
    run_steps = [s for s in _steps() if "run" in s]
    assert [s["id"] for s in run_steps] == ["gate", "comment"]
    assert all(s.get("shell") == "bash" for s in run_steps)


def test_the_comment_step_runs_after_a_failed_gate_and_only_on_a_pull_request() -> None:
    step = next(s for s in _steps() if s.get("id") == "comment")
    condition = str(step["if"])
    assert condition.startswith("always()")
    assert "inputs.pr-comment == 'true'" in condition
    assert "github.event.pull_request.number" in condition


def test_no_comment_by_default(tmp_path: Path) -> None:
    done, calls = _run_comment(tmp_path, EP_COMMENT="false")
    assert done.returncode == 0, done.stderr
    assert calls == []


def test_no_comment_outside_a_pull_request(tmp_path: Path) -> None:
    done, calls = _run_comment(tmp_path, EP_PR="")
    assert done.returncode == 0, done.stderr
    assert calls == []


def test_no_comment_when_no_report_was_written(tmp_path: Path) -> None:
    done, calls = _run_comment(tmp_path, report=None)
    assert done.returncode == 0 and calls == []
    assert "::warning::" not in done.stdout
    done, calls = _run_comment(tmp_path, report="")  # an empty file is no report either
    assert done.returncode == 0 and calls == []


def test_a_first_run_creates_one_comment_with_the_marker(tmp_path: Path) -> None:
    done, calls = _run_comment(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "::warning::" not in done.stdout
    assert len(calls) == 2
    lookup, create = calls
    assert lookup[:2] == ["api", "repos/octo/repo/issues/7/comments"]
    assert "--paginate" in lookup and MARKER in lookup[lookup.index("--jq") + 1]
    assert create[:2] == ["api", "repos/octo/repo/issues/7/comments"]
    assert "-X" not in create and create[2] == "-F" and create[3].startswith("body=@")
    assert _body(tmp_path, 2) == f"{MARKER}\n\n{REPORT}"
    assert (tmp_path / "gh" / "token-2").read_text("utf-8") == "token-from-input"


def test_a_later_run_updates_the_existing_comment(tmp_path: Path) -> None:
    done, calls = _run_comment(tmp_path, GH_LIST_IDS="111\n222\n")
    assert done.returncode == 0, done.stderr
    assert len(calls) == 2
    update = calls[1]
    assert update[:4] == ["api", "-X", "PATCH", "repos/octo/repo/issues/comments/111"]  # the first
    assert _body(tmp_path, 2) == f"{MARKER}\n\n{REPORT}"
    assert "updated" in done.stdout


def test_the_lookup_matches_only_comments_that_carry_the_marker(tmp_path: Path) -> None:
    jq = shutil.which("jq")
    if jq is None:
        pytest.skip("needs jq on PATH")
    _, calls = _run_comment(tmp_path)
    flt = calls[0][calls[0].index("--jq") + 1]
    page = [
        {"id": 1, "body": "looks good"},
        {"id": 2, "body": f"{MARKER}\n\n## evalproof"},
        {"id": 3, "body": None},
        {"id": 4, "body": f"{MARKER} again"},
    ]
    out = subprocess.run(
        [jq, "-r", flt], input=json.dumps(page), capture_output=True, text=True, check=True
    )
    assert out.stdout.split() == ["2", "4"]


def test_a_long_report_is_cut_to_fit_and_says_so(tmp_path: Path) -> None:
    done, _calls = _run_comment(tmp_path, report="x" * 70000 + "\n")
    assert done.returncode == 0, done.stderr
    body = _body(tmp_path, 2)
    assert body.startswith(f"{MARKER}\n\n" + "x" * 65000)
    assert "x" * 65001 not in body
    assert body.rstrip().endswith("The job summary has all of it._")
    assert len(body.encode("utf-8")) < 65536  # what a comment holds


@pytest.mark.parametrize(
    "env", [{"GH_WRITE_EXIT": "1"}, {"GH_LIST_EXIT": "1"}], ids=["post fails", "lookup fails"]
)
def test_a_failed_comment_is_a_warning_and_never_fails_the_step(
    tmp_path: Path, env: dict[str, str]
) -> None:
    done, calls = _run_comment(tmp_path, **env)
    assert done.returncode == 0
    assert "::warning::evalproof could not post the pull request comment" in done.stdout
    assert "pull-requests: write" in done.stdout
    assert len(calls) >= 1


def test_a_regression_stays_a_failure_when_the_comment_cannot_be_posted(tmp_path: Path) -> None:
    gate, _ = _run_step(tmp_path, STUB_COMPARE_EXIT="1")
    assert gate.returncode == 1  # the gate's own exit code
    assert (tmp_path / "output.txt").read_text("utf-8") == "exit-code=1\n"
    # The step that follows still has the report, and failing to post it changes nothing.
    comment, calls = _run_comment(tmp_path, report=None, GH_WRITE_EXIT="1")
    assert comment.returncode == 0
    assert "::warning::" in comment.stdout
    assert len(calls) == 2  # the lookup found nothing; the create was refused
    assert _body(tmp_path, 2) == f"{MARKER}\n\nreport of compare\n"
    assert (tmp_path / "output.txt").read_text("utf-8") == "exit-code=1\n"  # output untouched
