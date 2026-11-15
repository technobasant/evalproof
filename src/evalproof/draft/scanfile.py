"""Read the JSON ``evalproof scan`` writes, checking it is one this version understands."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evalproof.config import ConfigError
from evalproof.scan.model import FORMAT_VERSION


@dataclass
class ScanFile:
    path: Path
    root: Path  # the directory that was scanned, as a path that exists from here
    findings: list[dict[str, Any]]
    suites: list[dict[str, Any]]
    gaps: list[dict[str, Any]]

    def findings_of(self, kind: str) -> list[dict[str, Any]]:
        return [f for f in self.findings if f.get("kind") == kind]

    def finding_at(self, kind: str, path: str) -> dict[str, Any] | None:
        return next(
            (f for f in self.findings if f.get("kind") == kind and f.get("path") == path), None
        )


def _list_of_dicts(data: dict[str, Any], key: str, where: Path) -> list[dict[str, Any]]:
    value = data.get(key)
    if not isinstance(value, list) or not all(isinstance(v, dict) for v in value):
        raise ConfigError(f"{where}: {key!r} is missing or is not a list of objects")
    return value


def load_scan(path: Path) -> ScanFile:
    """The scan in ``path``. A relative ``root`` in it is taken from the current directory, where
    ``evalproof scan`` was run."""
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"cannot read the scan file {path}: {exc.strerror or exc}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path}: not valid JSON ({exc.msg}); write it with `evalproof scan --out`"
        ) from exc
    if not isinstance(data, dict) or "format_version" not in data:
        raise ConfigError(
            f"{path}: not an evalproof scan file; write it with `evalproof scan --out`"
        )
    if data["format_version"] != FORMAT_VERSION:
        raise ConfigError(
            f"{path}: scan format_version {data['format_version']!r}, but this evalproof reads "
            f"{FORMAT_VERSION}; run `evalproof scan` again"
        )
    root_text = data.get("root")
    if not isinstance(root_text, str) or not root_text:
        raise ConfigError(f"{path}: 'root' is missing")
    root = Path(root_text)
    if not root.is_dir():
        raise ConfigError(
            f"{path}: the scanned directory {root_text!r} does not exist from here; run "
            "`evalproof draft` from the directory you ran `evalproof scan` in"
        )
    return ScanFile(
        path=path,
        root=root,
        findings=_list_of_dicts(data, "findings", path),
        suites=_list_of_dicts(data, "suites", path),
        gaps=_list_of_dicts(data, "gaps", path),
    )
