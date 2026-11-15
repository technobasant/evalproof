"""``evalproof init --platform bitbucket``: the pipeline file, the snippet fallback, the refusals.

Like the GitLab tests, the generated pipeline's shell script is run for real under ``sh`` in a
throwaway git repository, with ``python`` and ``evalproof`` stubbed to log their arguments. No
model is called and nothing touches the network.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalproof.cli import main
from evalproof.config import ConfigError
from evalproof.scaffold import (
    ACTION_REF,
    BITBUCKET_CI_FILE,
    BITBUCKET_SNIPPET_FILE,
    bitbucket_ci_target,
    choose_judge,
    init_project,
    render_bitbucket_ci,
    render_ci_text,
)
from tests.test_init import (
    AGENTS_MD,
    BASE_FILES,
    SKILLS,
    Mr,
    _git,
    fake_cli,
    files_under,
    needs_git,
    run_init,
)

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake CLIs are shell scripts")

PIP_VERSION = ACTION_REF.removeprefix("v")
HOSTED = ["--judge", "anthropic:claude-haiku-5-5"]


@pytest.fixture
def bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory that is the whole ``PATH``: no real claude, codex or ollama."""
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d))
    return d


def pipeline(directory: Path, file: str = BITBUCKET_CI_FILE) -> dict[str, Any]:
    loaded: dict[str, Any] = yaml.safe_load((directory / file).read_text(encoding="utf-8"))
    return loaded


def pr_step(directory: Path, file: str = BITBUCKET_CI_FILE) -> dict[str, Any]:
    steps = pipeline(directory, file)["pipelines"]["pull-requests"]["**"]
    assert len(steps) == 1
    step: dict[str, Any] = steps[0]["step"]
    return step


def script_of(step: dict[str, Any]) -> str:
    return "\n".join(step["script"])


# --- files written -----------------------------------------------------------------------


@pytest.mark.parametrize("ci", ["keyless", "api-key", "none"])
def test_bitbucket_writes_its_own_file_and_no_other_workflow(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], ci: str
) -> None:
    code, out = run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", ci, *HOSTED)
    assert code == 0
    pipeline_files = {BITBUCKET_CI_FILE} if ci != "none" else set()
    assert files_under(tmp_path) == BASE_FILES | set(SKILLS) | {AGENTS_MD} | pipeline_files
    for path in files_under(tmp_path):
        assert "@@" not in (tmp_path / path).read_text(encoding="utf-8"), path
    if ci != "none":
        assert f"  wrote          {BITBUCKET_CI_FILE}" in out
        assert "include:" not in out and ".github" not in out and "gitlab" not in out.lower()


@pytest.mark.parametrize("ci", ["keyless", "api-key"])
def test_the_pipeline_runs_on_pull_requests_in_a_python_image_with_the_full_history(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], ci: str
) -> None:
    run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", ci, *HOSTED)
    loaded = pipeline(tmp_path)
    assert loaded["image"] == "python:3.13-slim"
    assert loaded["clone"] == {"depth": "full"}  # the destination branch has to be in the clone
    step = pr_step(tmp_path)
    script = script_of(step)
    assert "BITBUCKET_PR_DESTINATION_BRANCH" in script
    assert f'python -m pip install --quiet "evalproof=={PIP_VERSION}"' in script
    assert "evalproof compare" in script and "--gate" in script
    assert "--junit" not in script  # Bitbucket has no JUnit report to hand it to


def test_the_keyless_pipeline_verifies_before_it_compares_and_has_no_main_step(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    assert set(pipeline(tmp_path)["pipelines"]) == {"pull-requests"}
    script = script_of(pr_step(tmp_path))
    verify = "evalproof verify evals/runs/latest.json evals/suite.yaml"
    assert verify in script and script.index(verify) < script.index("evalproof compare")
    assert "evalproof run" not in script  # keyless makes no model call
    assert "definitions" not in pipeline(tmp_path)  # and has no response cache
    header = (tmp_path / BITBUCKET_CI_FILE).read_text(encoding="utf-8").split("\nimage:")[0]
    assert "evalproof verify" in header and "edited by hand" in header
    assert "--platform bitbucket --ci api-key" in header
    assert "snippet: merge it" not in header  # that wording is only for the snippet file


def test_the_api_key_pipeline_names_the_variable_keeps_a_cache_and_stores_main_runs(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", "api-key", *HOSTED)
    text = (tmp_path / BITBUCKET_CI_FILE).read_text(encoding="utf-8")
    header = text.split("\nimage:")[0]
    assert "repository variable named ANTHROPIC_API_KEY" in header and "Secured" in header
    assert "support.atlassian.com/bitbucket-cloud/docs/variables-and-secrets" in header
    for name in ("BITBUCKET_PR_DESTINATION_BRANCH", "BITBUCKET_PR_ID", "BITBUCKET_BRANCH"):
        assert name in header  # the default variables the file relies on are cited
    loaded = pipeline(tmp_path)
    assert loaded["definitions"]["caches"] == {"evalproof-responses": "evals/.evalproof/cache"}
    step = pr_step(tmp_path)
    assert step["caches"] == ["evalproof-responses"]
    assert 'evalproof run evals/suite.yaml --out "$candidate"' in script_of(step)
    main_steps = loaded["pipelines"]["branches"]["main"]
    assert len(main_steps) == 1
    main_step = main_steps[0]["step"]
    assert main_step["caches"] == ["evalproof-responses"]
    assert "evalproof run evals/suite.yaml --out evalproof-run.json" in script_of(main_step)
    assert main_step["artifacts"] == ["evalproof-run.json"]
    assert "add the repository variable ANTHROPIC_API_KEY, marked Secured" in out
    assert "Repository settings > Pipelines > Repository variables" in out
    assert "in CI: the ANTHROPIC_API_KEY repository variable" in out
    assert "pull requests from forks start no pipeline" in out
    assert "repository secret" not in out and "CI/CD variable" not in out


def test_the_api_key_variable_follows_the_judge_provider(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(
        tmp_path, capsys, "--platform", "bitbucket", "--ci", "api-key", "--judge", "openai:gpt-6"
    )
    assert "repository variable named OPENAI_API_KEY" in (tmp_path / BITBUCKET_CI_FILE).read_text(
        encoding="utf-8"
    )


def test_the_next_steps_for_keyless_name_the_files_to_commit(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    assert f"commit evals/ (including evals/runs/latest.json) and {BITBUCKET_CI_FILE}" in out
    assert "Pipelines must be turned on for the repository" in out
    _, out = run_init(tmp_path / "none", capsys, "--platform", "bitbucket", "--ci", "none")
    assert "Pipelines must be turned on" not in out  # no file, nothing to turn on
    _, out = run_init(tmp_path / "gh", capsys, *HOSTED)  # the GitHub run says nothing of it
    assert "bitbucket" not in out.lower()


def test_the_evals_readme_describes_the_bitbucket_pipeline(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert (
        BITBUCKET_CI_FILE in readme and "pull request" in readme and "destination branch" in readme
    )
    assert ".github" not in readme and "merge request" not in readme
    run_init(tmp_path / "key", capsys, "--platform", "bitbucket", "--ci", "api-key", *HOSTED)
    readme = (tmp_path / "key" / "evals" / "README.md").read_text(encoding="utf-8")
    assert "ANTHROPIC_API_KEY" in readme and "secured repository variable" in readme
    assert "push to `main`" in readme and "artifact" in readme


def test_the_other_platforms_keep_their_ci_text(bin_dir: Path) -> None:
    judge = choose_judge("anthropic:claude-haiku-5-5")
    for ci in ("keyless", "api-key", "none"):
        assert render_ci_text(ci, judge, "api-key", "github", BITBUCKET_CI_FILE) == render_ci_text(
            ci, judge
        )
    assert "bitbucket" not in render_ci_text("api-key", judge, "api-key", "gitlab").lower()


# --- the CLI-judge and platform refusals --------------------------------------------------


@pytest.mark.parametrize("judge", [None, "claude-code:haiku", "codex:gpt-6"])
def test_bitbucket_api_key_refuses_a_coding_agent_cli_before_writing_anything(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], judge: str | None
) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")  # the default judge is claude-code
    args = ["--judge", judge] if judge else []
    argv = ["init", "--dir", str(tmp_path / "p"), "--platform", "bitbucket", "--ci", "api-key"]
    code = main([*argv, *args])
    err = capsys.readouterr().err
    assert code == 2 and "--platform bitbucket --ci api-key" in err and "no Node" in err
    assert "--ci keyless" in err and "anthropic:" in err
    assert not (tmp_path / "p").exists()
    # A hosted API target, or the keyless mode, is fine with the same CLI on PATH.
    assert (
        run_init(tmp_path / "ok", capsys, "--platform", "bitbucket", "--ci", "api-key", *HOSTED)[0]
        == 0
    )
    assert run_init(tmp_path / "kl", capsys, "--platform", "bitbucket", "--ci", "keyless")[0] == 0
    for name in ("ok", "kl"):
        assert (tmp_path / name / BITBUCKET_CI_FILE).is_file()


def test_bitbucket_api_key_refuses_the_oauth_token_route_too(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The token is for a claude-code judge, and a claude-code judge is refused here, so every
    ``--cli-auth oauth-token`` combination ends in an error and nothing is written."""
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    argv = ["init", "--dir", str(tmp_path / "a"), "--platform", "bitbucket", "--ci", "api-key"]
    code = main([*argv, "--judge", "claude-code:haiku", "--cli-auth", "oauth-token"])
    assert code == 2 and "no Node" in capsys.readouterr().err
    code = main([*argv, *HOSTED, "--cli-auth", "oauth-token"])
    assert code == 2 and "works only with a claude-code judge" in capsys.readouterr().err
    assert not (tmp_path / "a").exists()


def test_a_bad_platform_still_names_bitbucket_among_the_choices(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="github, gitlab, bitbucket"):
        init_project(tmp_path, platform="azure")


# --- an existing bitbucket-pipelines.yml --------------------------------------------------

MINE = (
    "image: node:22\npipelines:\n  default:\n    - step:\n        script:\n          - npm test\n"
)


@pytest.mark.parametrize("ci", ["keyless", "api-key"])
def test_an_existing_pipeline_file_is_never_edited_and_the_snippet_is_written(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], ci: str
) -> None:
    (tmp_path / BITBUCKET_CI_FILE).write_text(MINE, encoding="utf-8")
    for extra in ([], ["--force"]):  # not even --force replaces a file someone else wrote
        _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", ci, *HOSTED, *extra)
        assert (tmp_path / BITBUCKET_CI_FILE).read_text(encoding="utf-8") == MINE
        assert (tmp_path / BITBUCKET_SNIPPET_FILE).is_file()
        assert f"{BITBUCKET_SNIPPET_FILE}" in out and "never edits it" in out
        assert f"Merge it into {BITBUCKET_CI_FILE}" in out
    assert files_under(tmp_path) == (
        BASE_FILES | set(SKILLS) | {AGENTS_MD, BITBUCKET_CI_FILE, BITBUCKET_SNIPPET_FILE}
    )


def test_the_snippet_says_how_to_merge_it_and_is_otherwise_the_pipeline(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / BITBUCKET_CI_FILE).write_text(MINE, encoding="utf-8")
    run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", "api-key", *HOSTED)
    snippet = (tmp_path / BITBUCKET_SNIPPET_FILE).read_text(encoding="utf-8")
    assert "bitbucket-pipelines.yml already exists" in snippet and "merge it into" in snippet
    assert "a pattern can appear only once in a file" in snippet
    assert pr_step(tmp_path, BITBUCKET_SNIPPET_FILE)["name"] == "evalproof run and gate"
    # Everything after the merge instructions is the file init would have written.
    fresh = render_bitbucket_ci("api-key", choose_judge(HOSTED[1]))
    assert snippet.endswith(fresh.split("# CI runs the suite itself", 1)[1])
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert BITBUCKET_SNIPPET_FILE in readme and "a snippet to merge into" in readme


def test_the_snippet_fallback_is_also_used_for_the_keyless_pipeline(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / BITBUCKET_CI_FILE).write_text(MINE, encoding="utf-8")
    _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    snippet = (tmp_path / BITBUCKET_SNIPPET_FILE).read_text(encoding="utf-8")
    assert "so this is a" in snippet and "evalproof verify" in script_of(
        pr_step(tmp_path, BITBUCKET_SNIPPET_FILE)
    )
    assert f"commit evals/ (including evals/runs/latest.json) and {BITBUCKET_CI_FILE}" in out


def test_an_existing_pipeline_file_with_ci_none_writes_no_snippet(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / BITBUCKET_CI_FILE).write_text(MINE, encoding="utf-8")
    _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", "--ci", "none")
    assert not (tmp_path / BITBUCKET_SNIPPET_FILE).exists() and "snippet" not in out


def test_a_pipeline_file_written_by_init_follows_the_usual_keep_or_force_rule(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    path = tmp_path / BITBUCKET_CI_FILE
    marked = path.read_text(encoding="utf-8")
    path.write_text(marked + "# my own note\n", encoding="utf-8")  # still carries the marker
    _, out = run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED)
    assert f"  kept (exists)  {BITBUCKET_CI_FILE}" in out
    assert path.read_text(encoding="utf-8").endswith("# my own note\n")
    assert not (tmp_path / BITBUCKET_SNIPPET_FILE).exists()  # it is ours: no snippet
    run_init(tmp_path, capsys, "--platform", "bitbucket", *HOSTED, "--force")
    assert path.read_text(encoding="utf-8") == marked


def test_bitbucket_ci_target_decides_between_the_file_and_the_snippet(tmp_path: Path) -> None:
    assert bitbucket_ci_target(tmp_path) == BITBUCKET_CI_FILE  # nothing there yet
    path = tmp_path / BITBUCKET_CI_FILE
    path.write_text(MINE, encoding="utf-8")
    assert bitbucket_ci_target(tmp_path) == BITBUCKET_SNIPPET_FILE
    path.write_text(render_bitbucket_ci("keyless", choose_judge(HOSTED[1])), encoding="utf-8")
    assert bitbucket_ci_target(tmp_path) == BITBUCKET_CI_FILE
    path.write_bytes(b"\xff\xfe not text")  # unreadable as text: never overwritten
    assert bitbucket_ci_target(tmp_path) == BITBUCKET_SNIPPET_FILE
    path.unlink()
    path.mkdir()  # a directory is not ours either
    assert bitbucket_ci_target(tmp_path) == BITBUCKET_SNIPPET_FILE


# --- the pipeline's script, run for real --------------------------------------------------

DESTINATION = {"BITBUCKET_PR_DESTINATION_BRANCH": "main"}


@pytest.fixture
def bitbucket_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> Callable[..., dict[str, Any]]:
    """Generate the Bitbucket file for a ci mode and return a job for ``Mr``: the script of
    its pull-request step (or its ``main`` step). Init runs on a bare PATH; the script needs the
    real one."""

    def make(ci: str, *, main_step: bool = False) -> dict[str, Any]:
        empty = tmp_path / f"empty-{ci}"
        empty.mkdir(exist_ok=True)
        directory = tmp_path / f"gen-{ci}"
        with monkeypatch.context() as patch:
            patch.setenv("PATH", str(empty))
            run_init(directory, capsys, "--platform", "bitbucket", "--ci", ci, *HOSTED)
        if main_step:
            step = pipeline(directory)["pipelines"]["branches"]["main"][0]["step"]
        else:
            step = pr_step(directory)
        return {"script": step["script"], "variables": {}}

    return make


def widen(job: dict[str, Any], watch: str) -> dict[str, Any]:
    """The same job with a longer ``EVALPROOF_WATCH`` list, the edit the comment asks for."""
    script = script_of({"script": job["script"]})
    assert 'EVALPROOF_WATCH="evals/"' in script
    return {
        **job,
        "script": [script.replace('EVALPROOF_WATCH="evals/"', f'EVALPROOF_WATCH="{watch}"')],
    }


@needs_git
def test_bitbucket_keyless_passes_with_a_notice_when_the_destination_has_no_run_file(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("keyless"), base_has_run=False)
    mr.commit({"evals/runs/latest.json": '{"v": 2}'})
    code, out, log = mr.run(**DESTINATION)
    assert code == 0, out
    assert "main has no evals/runs/latest.json yet" in out
    assert log == []  # nothing to compare with: no install, no verify, no compare


@needs_git
def test_bitbucket_keyless_fails_when_the_branch_removes_the_run_file(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("keyless"), base_has_run=True)
    _git(mr.repo, "rm", "-q", "evals/runs/latest.json")
    _git(mr.repo, "commit", "-q", "-m", "drop the run file")
    code, out, log = mr.run(**DESTINATION)
    assert code == 1 and "evals/runs/latest.json is missing from this branch" in out
    assert log == []


@needs_git
def test_bitbucket_keyless_verifies_then_compares_with_the_baseline_from_the_destination(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("keyless"), base_has_run=True)
    mr.commit({"evals/runs/latest.json": '{"v": 2}', "evals/suite.yaml": "name: s2\n"})
    code, out, log = mr.run(**DESTINATION)
    assert code == 0, out
    assert "evalproof: warning" not in out
    baseline = mr.scratch / "evalproof-baseline.json"
    assert log == [
        f"python -m pip install --quiet evalproof=={PIP_VERSION}",
        "evalproof verify evals/runs/latest.json evals/suite.yaml",
        f"evalproof compare {baseline} evals/runs/latest.json --suite evals/suite.yaml --gate",
    ]
    # The baseline is the destination branch's file, not the branch's own.
    assert baseline.read_text(encoding="utf-8") == '{"v": 1}'


@needs_git
def test_bitbucket_follows_the_destination_branch_it_is_given(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("keyless"), base_has_run=True)
    mr.commit({"evals/runs/latest.json": '{"v": 2}'})
    _git(mr.repo, "push", "-q", "origin", "HEAD:refs/heads/release")  # a second target
    code, out, _ = mr.run(BITBUCKET_PR_DESTINATION_BRANCH="release")
    assert code == 0, out  # baseline taken from release, which holds the v2 file
    assert (mr.scratch / "evalproof-baseline.json").read_text(encoding="utf-8") == '{"v": 2}'
    code, out, _ = mr.run(BITBUCKET_PR_DESTINATION_BRANCH="missing")
    assert code != 0  # a destination that cannot be fetched stops the step


@needs_git
@pytest.mark.parametrize(
    ("env", "ran"),
    [
        ({"STUB_VERIFY_EXIT": "4"}, ["verify"]),  # a run file that is not the suite's: no compare
        ({"STUB_COMPARE_EXIT": "1"}, ["verify", "compare"]),  # a regression
        ({"STUB_COMPARE_EXIT": "3"}, ["verify", "compare"]),  # a budget the run went over
    ],
)
def test_bitbucket_keyless_fails_the_step_with_the_commands_exit_code(
    tmp_path: Path,
    bitbucket_job: Callable[..., dict[str, Any]],
    env: dict[str, str],
    ran: list[str],
) -> None:
    mr = Mr(tmp_path, bitbucket_job("keyless"), base_has_run=True)
    mr.commit({"evals/runs/latest.json": '{"v": 2}'})
    code, out, log = mr.run(**DESTINATION, **env)
    assert code == int(next(iter(env.values()))), out
    assert [line.split()[1] for line in log if line.startswith("evalproof")] == ran


@needs_git
def test_bitbucket_keyless_warns_when_evals_changed_but_the_run_file_did_not(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    job = bitbucket_job("keyless")
    mr = Mr(tmp_path, job, base_has_run=True)
    mr.commit({"evals/cases.jsonl": "{}\n"})
    code, out, _ = mr.run(**DESTINATION)
    assert code == 0, out
    assert "evalproof: warning: these files changed but evals/runs/latest.json did not" in out
    assert "evals/cases.jsonl" in out
    mr.commit({"src/prompt.txt": "v2\n"})
    mr.job = widen(job, "evals/ src/")  # the list is yours to extend
    _, out, _ = mr.run(**DESTINATION)
    assert "evalproof: warning" in out and "src/prompt.txt" in out
    mr.commit({"evals/runs/latest.json": '{"v": 2}'})  # refreshed: nothing to warn about
    _, out, _ = mr.run(**DESTINATION)
    assert "evalproof: warning" not in out


@needs_git
def test_bitbucket_api_key_runs_the_suite_only_when_something_watched_changed(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    job = bitbucket_job("api-key")
    mr = Mr(tmp_path, job, base_has_run=True)
    mr.commit({"README.md": "docs\n"})
    code, out, log = mr.run(**DESTINATION)
    assert code == 0 and "nothing in evals/ changed" in out and log == []
    mr.commit({"src/prompt.txt": "v2\n"})  # not watched until it is on the list
    assert mr.run(**DESTINATION)[2] == []
    mr.job = widen(job, "evals/ src/")
    _, _, log = mr.run(**DESTINATION)
    assert [line.split()[1] for line in log if line.startswith("evalproof")] == ["run", "compare"]


@needs_git
def test_bitbucket_api_key_runs_and_gates_when_the_suite_changed(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("api-key"), base_has_run=True)
    mr.commit({"evals/cases.jsonl": "{}\n"})
    code, out, log = mr.run(**DESTINATION)
    assert code == 0, out
    baseline = mr.scratch / "evalproof-baseline.json"
    candidate = mr.scratch / "evalproof-candidate.json"
    assert log == [
        f"python -m pip install --quiet evalproof=={PIP_VERSION}",
        f"evalproof run evals/suite.yaml --out {candidate} --quiet",
        f"evalproof compare {baseline} {candidate} --suite evals/suite.yaml --gate",
    ]
    code, _, _ = mr.run(**DESTINATION, STUB_COMPARE_EXIT="1")
    assert code == 1  # a regression fails the step


@needs_git
def test_bitbucket_api_key_passes_with_a_notice_when_the_destination_has_no_run_file(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    mr = Mr(tmp_path, bitbucket_job("api-key"), base_has_run=False)
    mr.commit({"evals/cases.jsonl": "{}\n"})
    code, out, log = mr.run(**DESTINATION)
    assert code == 0 and "has no evals/runs/latest.json yet" in out and log == []


@needs_git
def test_the_main_step_runs_the_suite_and_writes_the_artifact_path_it_declares(
    tmp_path: Path, bitbucket_job: Callable[..., dict[str, Any]]
) -> None:
    job = bitbucket_job("api-key", main_step=True)
    mr = Mr(tmp_path, job, base_has_run=True)
    code, out, log = mr.run()
    assert code == 0, out
    assert log == [
        f"python -m pip install --quiet evalproof=={PIP_VERSION}",
        "evalproof run evals/suite.yaml --out evalproof-run.json --quiet",
    ]
    # The artifact path in the pipeline is the file the run command writes.
    assert "--out evalproof-run.json" in script_of({"script": job["script"]})
