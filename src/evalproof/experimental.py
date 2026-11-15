"""The one-line stderr notice for a command that is experimental: said once per process."""

from __future__ import annotations

import sys
import threading

_noticed: set[str] = set()
_lock = threading.Lock()


def experimental_notice(command: str, detail: str) -> None:
    """Say once per process that ``command`` is experimental, and what may change."""
    with _lock:
        if command in _noticed:
            return
        _noticed.add(command)
    print(f"evalproof: {command} is experimental: {detail}", file=sys.stderr, flush=True)
