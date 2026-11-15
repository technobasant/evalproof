"""Start a child process as the head of its own process group and kill the whole tree.

A child that outlives its parent (``npx`` starting ``node``, a shell script starting a program)
keeps the output pipes open, so reading them never finishes. The platform branch lives here, in
two small functions, so the rest of evalproof calls one thing on every platform. Internal: not
part of the public API.
"""

from __future__ import annotations

import contextlib
import os
import signal
import subprocess
import sys
from typing import Any

TASKKILL_TIMEOUT_S = 10.0  # how long to wait for ``taskkill`` before falling back to ``kill()``


def popen_group_kwargs() -> dict[str, Any]:
    """The ``Popen`` arguments that make the child the head of its own process group: a new
    session on POSIX, a new process group on Windows. ``kill_tree`` needs this."""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def kill_tree(proc: subprocess.Popen[bytes]) -> None:
    """Kill ``proc`` and everything it started. ``proc`` must have been started with
    ``popen_group_kwargs``. A process that is already gone, or one that cannot be signalled, is
    not an error: the caller is cleaning up and has nothing to do about it.

    POSIX sends SIGKILL to the process group. Windows runs ``taskkill /F /T``, which ends the
    process and its children, and then ``proc.kill()`` in case ``taskkill`` was not there or
    failed."""
    if sys.platform == "win32":
        with contextlib.suppress(OSError, subprocess.SubprocessError):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
                timeout=TASKKILL_TIMEOUT_S,
            )
        with contextlib.suppress(OSError):
            proc.kill()
    else:
        with contextlib.suppress(OSError):  # ProcessLookupError and PermissionError are OSErrors
            os.killpg(proc.pid, signal.SIGKILL)
