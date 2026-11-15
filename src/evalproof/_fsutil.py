"""Filesystem helpers for what Windows does differently from POSIX: a file or directory that
another process (a just-killed child, an antivirus scanner) has open cannot be replaced or
removed for a moment.

The helpers take their ``sleep`` (and the operation itself) as arguments so a test can simulate
a lock that clears after a few attempts without waiting.
"""

from __future__ import annotations

import os
import re
import shutil
import time
from collections.abc import Callable

# A lock held by a just-killed child or a scanner clears within a fraction of a second. Linear
# back-off: 0.025 + 0.05 + 0.075 + 0.1 + 0.125 s, about 0.4 s in all before giving up.
RETRY_ATTEMPTS = 6
RETRY_DELAY = 0.025


def replace_retry(
    src: str | os.PathLike[str],
    dst: str | os.PathLike[str],
    *,
    attempts: int = RETRY_ATTEMPTS,
    delay: float = RETRY_DELAY,
    sleep: Callable[[float], None] = time.sleep,
    replace: Callable[[str, str], None] | None = None,
) -> bool:
    """``os.replace`` that retries on ``PermissionError`` (Windows raises it while another
    handle has the target open). Returns True once replaced and False if every attempt was
    refused; any other ``OSError`` propagates."""
    do_replace = replace if replace is not None else os.replace
    for attempt in range(attempts):
        try:
            do_replace(os.fspath(src), os.fspath(dst))
        except PermissionError:
            if attempt < attempts - 1:
                sleep(delay * (attempt + 1))
            continue
        return True
    return False


def rmtree_retry(
    path: str | os.PathLike[str],
    *,
    attempts: int = RETRY_ATTEMPTS,
    delay: float = RETRY_DELAY,
    sleep: Callable[[float], None] = time.sleep,
    rmtree: Callable[[str], None] | None = None,
) -> bool:
    """Remove a directory tree, retrying a few times (Windows keeps a just-killed child's
    files locked for a moment), then give up silently: a leaked temp dir is not worth failing
    a run. Returns True when the tree is gone."""
    remove = rmtree if rmtree is not None else shutil.rmtree
    for attempt in range(attempts):
        try:
            remove(os.fspath(path))
        except FileNotFoundError:
            return True  # already gone
        except OSError:
            if attempt < attempts - 1:
                sleep(delay * (attempt + 1))
            continue
        return True
    return False


# Characters NTFS refuses in a file name, and the control characters (0-31).
_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def safe_filename(name: str, fallback: str = "run") -> str:
    """``name`` as one file-name component that every OS accepts: each character NTFS refuses
    (``<>:"/\\|?*`` and control characters) becomes ``-``, trailing dots and spaces (which
    Windows silently drops) are removed, and an empty result is ``fallback``."""
    cleaned = _INVALID_FILENAME_CHARS.sub("-", name).rstrip(". ")
    return cleaned or fallback
