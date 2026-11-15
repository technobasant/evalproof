"""``evalproof hook install | uninstall``: the git pre-push hook that runs ``verify``.

Git is the real one, in a temporary repository. The hook script is run for real under ``sh``,
with stubs named ``evalproof`` and ``uvx`` on the PATH that log their arguments, so no model is
called and nothing is fetched.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from evalproof._version import __version__
from evalproof.cli import main
from evalproof.hook import HOOK_MARKER, render_hook

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

posix_sh = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("sh") is None, reason="the stubs are shell scripts"
)

GIT_ENV = {
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@example.com",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}


@pytest.fixture(autouse=True)
def isolated_git(monkeypatch: pytest.MonkeyPatch) -> None:
    """No user or system git config (a global ``core.hooksPath`` would redirect the hook)."""
    for key, value in GIT_ENV.items():
        monkeypatch.setenv(key, value)


def git(repo: Path, *args: str) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **GIT_ENV},
    )
    return done.stdout


@pytest.fixture
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "repo"
    git(tmp_path, "init", "-q", "-b", "main", str(root))
    (root / "evals" / "runs").mkdir(parents=True)
    (root / "evals" / "suite.yaml").write_text("name: s\n", encoding="utf-8")
    (root / "evals" / "runs" / "latest.json").write_text("{}", encoding="utf-8")
    monkeypatch.chdir(root)
    return root


def hook_cmd(capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["hook", *args])
    out, err = capsys.readouterr()
    return code, out, err


def hook_file(repo: Path) -> Path:
    return repo / ".git" / "hooks" / "pre-push"


# --- install ------------------------------------------------------------------------------


def test_install_writes_an_executable_marked_hook(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = hook_cmd(capsys, "install")
    assert code == 0
    path = hook_file(repo)
    text = path.read_text(encoding="utf-8")
    assert text.startswith("#!/bin/sh\n")
    assert f"{HOOK_MARKER} (evalproof {__version__})" in text.splitlines()
    assert "suite=evals/suite.yaml\n" in text and "run=evals/runs/latest.json\n" in text
    assert f"uvx evalproof@{__version__} verify" in text
    assert "@@" not in text and "\r" not in text
    if sys.platform != "win32":
        assert path.stat().st_mode & 0o111 == 0o111
    assert f"installed {path}" in out
    assert "evalproof verify evals/runs/latest.json evals/suite.yaml" in out
    assert "does not exist yet" not in out


def test_install_notes_a_suite_or_run_that_does_not_exist_yet(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "evals" / "runs" / "latest.json").unlink()
    code, out, _ = hook_cmd(capsys, "install")
    assert code == 0
    assert "note: evals/runs/latest.json does not exist yet; the hook skips until it does" in out
    assert "evals/suite.yaml does not exist" not in out


def test_install_outside_a_git_repository_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plain = tmp_path / "plain"
    plain.mkdir()
    monkeypatch.chdir(plain)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    code, out, err = hook_cmd(capsys, "install")
    assert code == 2
    assert "is not inside a git repository" in err
    assert out == ""
    assert list(plain.iterdir()) == []


def test_install_refuses_a_hook_it_did_not_write_unless_forced(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hook_file(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    code, _, err = hook_cmd(capsys, "install")
    assert code == 2
    assert "was not written by evalproof" in err and "--force" in err
    assert path.read_text(encoding="utf-8") == "#!/bin/sh\necho mine\n"
    code, out, _ = hook_cmd(capsys, "install", "--force")
    assert code == 0 and "installed" in out
    assert HOOK_MARKER in path.read_text(encoding="utf-8")


def test_install_again_updates_its_own_hook_without_force(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hook_cmd(capsys, "install")
    code, _, _ = hook_cmd(capsys, "install", "--suite", "evals/other.yaml", "--run", "r.json")
    assert code == 0
    text = hook_file(repo).read_text(encoding="utf-8")
    assert "suite=evals/other.yaml\n" in text and "run=r.json\n" in text
    assert text.count(HOOK_MARKER) == 1


@pytest.mark.skipif(sys.platform == "win32", reason="symbolic links need privileges on Windows")
def test_install_does_not_write_through_a_symbolic_link(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    shared = tmp_path / "shared-hook.sh"
    shared.write_text(f"#!/bin/sh\n{HOOK_MARKER}\n", encoding="utf-8")
    path = hook_file(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(shared)
    code, _, err = hook_cmd(capsys, "install")
    assert code == 2 and "was not written by evalproof" in err
    assert path.is_symlink()
    code, _, _ = hook_cmd(capsys, "install", "--force")
    assert code == 0
    assert not path.is_symlink() and HOOK_MARKER in path.read_text(encoding="utf-8")
    assert shared.read_text(encoding="utf-8") == f"#!/bin/sh\n{HOOK_MARKER}\n"


def test_install_follows_core_hooks_path(repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    git(repo, "config", "core.hooksPath", "custom-hooks")
    code, _, _ = hook_cmd(capsys, "install")
    assert code == 0
    assert HOOK_MARKER in (repo / "custom-hooks" / "pre-push").read_text(encoding="utf-8")
    assert not hook_file(repo).exists()


def test_install_from_a_subdirectory_stores_paths_from_the_repository_root(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    sub = repo / "services" / "api"
    sub.mkdir(parents=True)
    (sub / "suite.yaml").write_text("name: s\n", encoding="utf-8")
    monkeypatch.chdir(sub)
    code, _, _ = hook_cmd(
        capsys, "install", "--suite", "suite.yaml", "--run", "../../evals/runs/latest.json"
    )
    assert code == 0
    text = hook_file(repo).read_text(encoding="utf-8")
    assert "suite=services/api/suite.yaml\n" in text
    assert "run=evals/runs/latest.json\n" in text


def test_install_in_a_linked_worktree_writes_the_shared_hooks_folder(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (repo / "README").write_text("x\n", encoding="utf-8")
    git(repo, "add", "README")
    git(repo, "commit", "-q", "-m", "first")
    linked = tmp_path / "linked"
    git(repo, "worktree", "add", "-q", "-b", "other", str(linked))
    monkeypatch.chdir(linked)
    code, _, _ = hook_cmd(capsys, "install")
    assert code == 0
    assert HOOK_MARKER in hook_file(repo).read_text(encoding="utf-8")


# --- uninstall ----------------------------------------------------------------------------


def test_uninstall_removes_only_its_own_hook(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    hook_cmd(capsys, "install")
    code, out, _ = hook_cmd(capsys, "uninstall")
    assert code == 0 and f"removed {hook_file(repo)}" in out
    assert not hook_file(repo).exists()


def test_uninstall_leaves_a_hook_it_did_not_write(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = hook_file(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\necho mine\n", encoding="utf-8")
    code, _, err = hook_cmd(capsys, "uninstall")
    assert code == 2 and "was not written by evalproof, so it was left in place" in err
    code, _, _ = hook_cmd(capsys, "uninstall", "--force")
    assert code == 2  # --force is for install only
    assert path.read_text(encoding="utf-8") == "#!/bin/sh\necho mine\n"


def test_uninstall_with_no_hook_says_so_and_succeeds(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = hook_cmd(capsys, "uninstall")
    assert code == 0 and "nothing to remove" in out


def test_uninstall_outside_a_git_repository_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    code, _, err = hook_cmd(capsys, "uninstall")
    assert code == 2 and "is not inside a git repository" in err


def test_the_action_must_be_install_or_uninstall(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["hook", "enable"])
    assert exit_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


# --- the script, run for real -------------------------------------------------------------


class HookRun:
    """Runs the installed hook under ``sh`` from the repository root, as git would."""

    def __init__(self, repo: Path, tmp_path: Path) -> None:
        self.repo = repo
        self.bin = tmp_path / "stubs"
        self.bin.mkdir()
        self.log = tmp_path / "stub.log"

    def stub(self, name: str) -> None:
        path = self.bin / name
        path.write_text(
            f'#!/bin/sh\necho "{name} $*" >> "{self.log}"\nexit "${{STUB_EXIT:-0}}"\n',
            encoding="utf-8",
        )
        path.chmod(0o755)

    def run(self, **env: str) -> tuple[int, str, list[str]]:
        self.log.write_text("", encoding="utf-8")
        sh = shutil.which("sh")
        assert sh is not None
        done = subprocess.run(
            [sh, str(hook_file(self.repo)), "origin", "git@example.com:o/r.git"],
            cwd=self.repo,
            capture_output=True,
            text=True,
            env={**os.environ, "PATH": str(self.bin), **env},
            check=False,
        )
        calls = self.log.read_text(encoding="utf-8").splitlines()
        return done.returncode, done.stderr + done.stdout, calls


@pytest.fixture
def installed(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> Callable[..., HookRun]:
    def make(*args: str) -> HookRun:
        assert hook_cmd(capsys, "install", *args)[0] == 0
        return HookRun(repo, tmp_path)

    return make


@posix_sh
def test_the_hook_runs_verify_on_the_run_and_suite(installed: Callable[..., HookRun]) -> None:
    hook = installed()
    hook.stub("evalproof")
    code, output, calls = hook.run()
    assert code == 0 and output == ""
    assert calls == ["evalproof verify evals/runs/latest.json evals/suite.yaml"]


@posix_sh
def test_the_hook_passes_the_verify_exit_code_through_and_says_how_to_go_on(
    installed: Callable[..., HookRun],
) -> None:
    hook = installed()
    hook.stub("evalproof")
    code, output, _ = hook.run(STUB_EXIT="4")
    assert code == 4
    assert "the push is stopped (exit 4)" in output
    assert "evalproof run 'evals/suite.yaml' --out 'evals/runs/latest.json'" in output
    assert "git push --no-verify" in output


@posix_sh
def test_the_hook_falls_back_to_a_pinned_uvx(installed: Callable[..., HookRun]) -> None:
    hook = installed()
    hook.stub("uvx")
    code, _, calls = hook.run()
    assert code == 0
    assert calls == [f"uvx evalproof@{__version__} verify evals/runs/latest.json evals/suite.yaml"]
    code, output, _ = hook.run(STUB_EXIT="1")
    assert code == 1 and "the push is stopped (exit 1)" in output


@posix_sh
def test_the_hook_prefers_evalproof_on_the_path_to_uvx(installed: Callable[..., HookRun]) -> None:
    hook = installed()
    hook.stub("evalproof")
    hook.stub("uvx")
    _, _, calls = hook.run()
    assert [c.split()[0] for c in calls] == ["evalproof"]


@posix_sh
def test_the_hook_skips_when_neither_evalproof_nor_uvx_is_available(
    installed: Callable[..., HookRun],
) -> None:
    hook = installed()
    code, output, calls = hook.run()
    assert code == 0 and calls == []
    assert "pre-push check skipped: neither evalproof nor uvx is on PATH" in output


@posix_sh
@pytest.mark.parametrize("gone", ["evals/suite.yaml", "evals/runs/latest.json"])
def test_the_hook_skips_when_the_suite_or_the_run_is_missing(
    installed: Callable[..., HookRun], repo: Path, gone: str
) -> None:
    hook = installed()
    hook.stub("evalproof")
    (repo / gone).unlink()
    code, output, calls = hook.run(STUB_EXIT="4")
    assert code == 0 and calls == []
    assert f"pre-push check skipped: {gone} does not exist" in output


@posix_sh
def test_the_hook_quotes_paths_with_spaces_and_quotes(
    repo: Path, installed: Callable[..., HookRun]
) -> None:
    awkward = "my evals/it's suite.yaml"
    (repo / "my evals").mkdir()
    (repo / awkward).write_text("name: s\n", encoding="utf-8")
    hook = installed("--suite", awkward)
    hook.stub("evalproof")
    code, _, calls = hook.run()
    assert code == 0
    assert calls == [f"evalproof verify evals/runs/latest.json {awkward}"]


@posix_sh
def test_git_push_is_stopped_by_a_failing_verify_and_goes_through_when_it_passes(
    repo: Path, tmp_path: Path, installed: Callable[..., HookRun]
) -> None:
    hook = installed()
    hook.stub("evalproof")
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", "-b", "main", str(remote))
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "first")
    git(repo, "remote", "add", "origin", str(remote))

    def push(exit_code: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["git", "-C", str(repo), "push", "-q", "origin", "main"],
            capture_output=True,
            text=True,
            env={
                **os.environ,
                **GIT_ENV,
                "PATH": f"{hook.bin}{os.pathsep}{os.environ['PATH']}",
                "STUB_EXIT": exit_code,
            },
            check=False,
        )

    blocked = push("4")
    assert blocked.returncode != 0
    assert "the push is stopped (exit 4)" in blocked.stderr
    assert git(remote, "for-each-ref", "refs/heads").strip() == ""
    assert push("0").returncode == 0
    assert "refs/heads/main" in git(remote, "for-each-ref", "refs/heads")


def test_render_hook_quotes_a_path_for_sh() -> None:
    text = render_hook("it's here/suite.yaml", "run.json")
    assert "suite='it'\"'\"'s here/suite.yaml'\n" in text
