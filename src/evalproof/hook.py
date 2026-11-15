"""``evalproof hook install | uninstall``: a git pre-push hook that runs ``evalproof verify``.

The hook is a POSIX ``sh`` script (Git for Windows runs it with its own ``sh``). It checks that
the committed run file still matches the suite and its dataset, with no model call, so a run
that has gone stale does not reach the remote. It carries a marker line; ``install`` replaces
only a hook that has the marker (or any hook with ``--force``) and ``uninstall`` removes only
a hook that has it, so a hook someone else wrote is never touched by accident.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

from evalproof._version import __version__

HOOK_NAME = "pre-push"
HOOK_MARKER = "# evalproof pre-push hook"
DEFAULT_SUITE = "evals/suite.yaml"
DEFAULT_RUN = "evals/runs/latest.json"

HOOK_TEMPLATE = """\
#!/bin/sh
@@MARKER@@ (evalproof @@VERSION@@)
# Written by `evalproof hook install`; `evalproof hook uninstall` removes it. Before a push it
# runs `evalproof verify`, which checks that the committed run file matches the suite and its
# dataset without calling a model. When it does not, the push is stopped: refresh the run, or
# skip this check once with `git push --no-verify`.
suite=@@SUITE@@
run=@@RUN@@
for file in "$suite" "$run"; do
    if [ ! -f "$file" ]; then
        echo "evalproof: pre-push check skipped: $file does not exist" >&2
        exit 0
    fi
done
if command -v evalproof >/dev/null 2>&1; then
    evalproof verify "$run" "$suite"
elif command -v uvx >/dev/null 2>&1; then
    uvx evalproof@@@VERSION@@ verify "$run" "$suite"
else
    echo "evalproof: pre-push check skipped: neither evalproof nor uvx is on PATH" >&2
    exit 0
fi
status=$?
if [ "$status" -ne 0 ]; then
    echo "evalproof: the push is stopped (exit $status). Refresh the run with:" \\
        "evalproof run '$suite' --out '$run'; or push without this check: git push --no-verify" >&2
fi
exit "$status"
"""


class HookError(ValueError):
    """A hook command that cannot be carried out (a usage error: exit 2)."""


def _git(directory: Path, *args: str) -> str | None:
    """Standard output of ``git`` run in ``directory``; None when git exits non-zero (here: not
    inside a repository, or no work tree)."""
    try:
        done = subprocess.run(
            ["git", *args],
            cwd=directory,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )
    except FileNotFoundError:
        raise HookError("git is not installed or is not on PATH") from None
    return done.stdout.strip() if done.returncode == 0 else None


def hooks_directory(directory: Path) -> Path:
    """Where git keeps hooks for the repository ``directory`` is in: ``core.hooksPath`` when it
    is set, else the ``hooks`` folder of the git directory (shared by linked worktrees)."""
    found = _git(directory, "rev-parse", "--git-path", "hooks")
    if not found:
        raise HookError(f"{directory} is not inside a git repository")
    path = Path(found)
    return path if path.is_absolute() else directory / path


def _locate(path: str, directory: Path) -> tuple[str, Path]:
    """``path`` (as given, relative to ``directory``) the way the hook will find it, and where
    that is on disk. Git runs a hook from the repository root, so a path inside the repository
    is written relative to it; a path elsewhere stays absolute."""
    top = _git(directory, "rev-parse", "--show-toplevel")
    absolute = Path(os.path.normpath(os.path.join(os.path.realpath(directory), path)))
    if top:
        try:
            return absolute.relative_to(os.path.realpath(top)).as_posix(), absolute
        except ValueError:
            pass
    return absolute.as_posix(), absolute


def render_hook(suite: str, run: str) -> str:
    """The hook script for ``suite`` and ``run`` (paths relative to the repository root)."""
    return (
        HOOK_TEMPLATE.replace("@@MARKER@@", HOOK_MARKER)
        .replace("@@VERSION@@", __version__)
        .replace("@@SUITE@@", shlex.quote(suite))
        .replace("@@RUN@@", shlex.quote(run))
    )


def _is_ours(path: Path) -> bool:
    """True for a regular file that carries the marker. A symbolic link is never ours: writing
    through it would change the file it points to."""
    if path.is_symlink() or not path.is_file():
        return False
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(line.startswith(HOOK_MARKER) for line in text.splitlines())


def install_hook(
    directory: Path,
    *,
    suite: str = DEFAULT_SUITE,
    run: str = DEFAULT_RUN,
    force: bool = False,
    emit: Callable[[str], None] = print,
) -> int:
    """Write the pre-push hook. Refuses to replace a hook without the marker unless ``force``."""
    target = hooks_directory(directory) / HOOK_NAME
    if (target.exists() or target.is_symlink()) and not _is_ours(target) and not force:
        raise HookError(
            f"{target} already exists and was not written by evalproof; add "
            f"`evalproof verify {run} {suite}` to it yourself, or use --force to replace it"
        )
    suite_path, suite_file = _locate(suite, directory)
    run_path, run_file = _locate(run, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        target.unlink()
    target.write_text(render_hook(suite_path, run_path), encoding="utf-8", newline="\n")
    target.chmod(0o755)
    emit(f"installed {target}")
    emit(f"  before each push it runs: evalproof verify {run_path} {suite_path}")
    for shown, file in ((suite_path, suite_file), (run_path, run_file)):
        if not file.is_file():
            emit(f"  note: {shown} does not exist yet; the hook skips until it does")
    emit("  skip it once with `git push --no-verify`; remove it with `evalproof hook uninstall`")
    return 0


def uninstall_hook(directory: Path, *, emit: Callable[[str], None] = print) -> int:
    """Remove the pre-push hook ``install_hook`` wrote. A hook without the marker is left alone."""
    target = hooks_directory(directory) / HOOK_NAME
    if not target.exists() and not target.is_symlink():
        emit(f"no pre-push hook at {target}; nothing to remove")
        return 0
    if not _is_ours(target):
        raise HookError(f"{target} was not written by evalproof, so it was left in place")
    target.unlink()
    emit(f"removed {target}")
    return 0
