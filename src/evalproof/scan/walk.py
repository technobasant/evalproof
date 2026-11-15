"""Walk a directory for the files a scan reads, skipping what is not the project's own code."""

from __future__ import annotations

import fnmatch
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "env",
        "__pycache__",
        "dist",
        "build",
        "site-packages",
    }
)
MAX_BYTES = 1_000_000  # a larger file is skipped and counted, not read

PY_SUFFIXES = frozenset({".py"})
TS_SUFFIXES = frozenset({".ts", ".tsx", ".js", ".mjs", ".jsx", ".cjs", ".mts", ".cts"})
YAML_SUFFIXES = frozenset({".yaml", ".yml"})
DATA_SUFFIXES = frozenset({".json", ".jsonl", ".csv"})
PROMPT_SUFFIXES = frozenset({".txt", ".md", ".j2", ".jinja", ".prompt", ".yaml", ".yml"})
READ_SUFFIXES = PY_SUFFIXES | TS_SUFFIXES | YAML_SUFFIXES | DATA_SUFFIXES | PROMPT_SUFFIXES


@dataclass
class Walked:
    files: list[str]  # relative to the root, "/" separated, sorted
    too_large: int


def _normal(pattern: str) -> str:
    pattern = pattern.replace("\\", "/").removeprefix("./")
    return pattern.rstrip("/") if len(pattern) > 1 else pattern


def _excluded(rel: str, name: str, patterns: Sequence[str]) -> bool:
    return any(
        fnmatch.fnmatchcase(rel, p)
        or fnmatch.fnmatchcase(name, p)
        or fnmatch.fnmatchcase(rel + "/", p)
        for p in patterns
    )


def walk(root: Path, excludes: Sequence[str] = ()) -> Walked:
    """The readable files under ``root``. A directory that is hidden (starts with a dot), is in
    ``SKIP_DIRS``, ends in ``.egg-info`` or matches an ``excludes`` glob is not entered; a symlink
    to a directory is not followed."""
    patterns = [_normal(p) for p in excludes if p.strip()]
    files: list[str] = []
    too_large = 0
    for here, dirs, names in os.walk(root, topdown=True, followlinks=False):
        rel_dir = Path(here).relative_to(root).as_posix()
        prefix = "" if rel_dir == "." else rel_dir + "/"
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in SKIP_DIRS
            and not d.startswith(".")
            and not d.endswith(".egg-info")
            and not _excluded(prefix + d, d, patterns)
        )
        for name in sorted(names):
            if os.path.splitext(name)[1].lower() not in READ_SUFFIXES:
                continue
            rel = prefix + name
            if _excluded(rel, name, patterns):
                continue
            try:
                size = os.stat(os.path.join(here, name)).st_size
            except OSError:
                continue
            if size > MAX_BYTES:
                too_large += 1
                continue
            files.append(rel)
    return Walked(files, too_large)
