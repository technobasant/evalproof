"""Starting a child in its own process group and killing the whole tree, on both platforms.

Each platform branch is tested with ``sys.platform`` patched, so both run on every machine; the
real kill of a real tree is tested by the callers (the ``command:`` target and the agent CLIs).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Any, cast

import pytest

from evalproof import _process
from evalproof._process import kill_tree, popen_group_kwargs


class FakeProc:
    pid = 4242

    def __init__(self, kill_error: Exception | None = None) -> None:
        self.killed = 0
        self.kill_error = kill_error

    def kill(self) -> None:
        self.killed += 1
        if self.kill_error:
            raise self.kill_error


def as_popen(proc: FakeProc) -> subprocess.Popen[bytes]:
    return cast("subprocess.Popen[bytes]", proc)


def test_windows_starts_a_new_process_group(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 512, raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        assert popen_group_kwargs() == {"creationflags": 512}


def test_posix_starts_a_new_session(monkeypatch: pytest.MonkeyPatch) -> None:
    for platform in ("linux", "darwin"):
        with monkeypatch.context() as patch:
            patch.setattr(sys, "platform", platform)
            assert popen_group_kwargs() == {"start_new_session": True}


def test_windows_kills_the_tree_with_taskkill_then_the_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ran: list[tuple[list[str], dict[str, Any]]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> None:
        ran.append((argv, kwargs))

    monkeypatch.setattr(subprocess, "run", fake_run)
    proc = FakeProc()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        kill_tree(as_popen(proc))
    assert len(ran) == 1
    assert ran[0][0] == ["taskkill", "/F", "/T", "/PID", "4242"]
    assert ran[0][1]["capture_output"] is True and ran[0][1]["check"] is False
    assert ran[0][1]["timeout"] == _process.TASKKILL_TIMEOUT_S
    assert proc.killed == 1  # the fallback, in case taskkill did not end it


@pytest.mark.parametrize(
    "error",
    [
        FileNotFoundError("taskkill"),
        PermissionError("denied"),
        subprocess.TimeoutExpired("taskkill", 10),
    ],
    ids=["missing", "denied", "hung"],
)
def test_a_taskkill_that_fails_is_ignored_and_the_process_is_still_killed(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def failing(argv: list[str], **kwargs: Any) -> None:
        raise error

    monkeypatch.setattr(subprocess, "run", failing)
    proc = FakeProc()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "win32")
        kill_tree(as_popen(proc))
    assert proc.killed == 1


def test_windows_swallows_an_error_from_the_kill_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda argv, **kwargs: None)
    for error in (PermissionError("access denied"), ProcessLookupError(), OSError("gone")):
        proc = FakeProc(kill_error=error)
        with monkeypatch.context() as patch:
            patch.setattr(sys, "platform", "win32")
            kill_tree(as_popen(proc))  # does not raise
        assert proc.killed == 1


def test_posix_kills_the_process_group_and_does_not_run_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signalled: list[tuple[int, int]] = []

    def no_subprocess(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("taskkill is for Windows")

    monkeypatch.setattr(
        os, "killpg", lambda pgid, sig: signalled.append((pgid, sig)), raising=False
    )
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(subprocess, "run", no_subprocess)
    proc = FakeProc()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "linux")
        kill_tree(as_popen(proc))
    assert signalled == [(4242, 9)]


@pytest.mark.parametrize(
    "error", [ProcessLookupError(), PermissionError("not yours"), OSError("other")]
)
def test_posix_swallows_an_error_from_the_kill_itself(
    monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    def failing(pgid: int, sig: int) -> None:
        raise error

    monkeypatch.setattr(os, "killpg", failing, raising=False)
    monkeypatch.setattr(signal, "SIGKILL", 9, raising=False)
    with monkeypatch.context() as patch:
        patch.setattr(sys, "platform", "linux")
        kill_tree(as_popen(FakeProc()))  # does not raise


def test_a_real_child_started_with_the_group_arguments_is_killed_with_its_child() -> None:
    """The two functions together on the real platform: the child starts a grandchild, and
    both are gone after ``kill_tree``."""
    code = (
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        "print('up', flush=True)\n"
        "time.sleep(60)\n"
    )
    with subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        **popen_group_kwargs(),
    ) as proc:
        assert proc.stdout is not None
        assert proc.stdout.readline().strip() == b"up"
        kill_tree(proc)
        # The pipes close only when the grandchild is gone too; without a tree kill this waits
        # for its 60 seconds.
        proc.communicate(timeout=20)
        assert proc.returncode is not None
