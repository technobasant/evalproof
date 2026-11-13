"""Golden sets as JSONL: one case per line, stable ids, and a content hash for provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalproof.models import Case


class DatasetError(ValueError):
    pass


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_id(record: dict[str, Any]) -> str:
    """An id derived from the input, so reordering the file never re-keys a case."""
    digest = hashlib.sha256(canonical_json(record.get("input")).encode()).hexdigest()
    return f"case-{digest[:12]}"


def load_dataset(path: str | Path) -> list[Case]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path}:{lineno}: invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict) or "input" not in record:
            raise DatasetError(f"{path}:{lineno}: each line needs an object with an 'input'")
        record.setdefault("id", stable_id(record))
        try:
            case = Case.model_validate(record)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"])
            raise DatasetError(f"{path}:{lineno}: {where}: {first['msg']}") from exc
        if case.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"dataset is empty: {path}")
    return cases


def case_hash(case: Case) -> str:
    """Hash of one case's content (its id, input, reference, tags and extra fields)."""
    return hashlib.sha256(canonical_json(case.model_dump(mode="json")).encode()).hexdigest()[:16]


def content_hash(case: Case) -> str:
    """Hash of a case's content without its tags, which label a case and don't change what the
    system is asked. ``compare`` pairs on this, so adding a tag is not an edit."""
    dump = case.model_dump(mode="json", exclude={"tags"})
    return hashlib.sha256(canonical_json(dump).encode()).hexdigest()[:16]


def dataset_hash(cases: list[Case]) -> str:
    """Hash of the cases as loaded; two runs on different data never compare silently."""
    h = hashlib.sha256()
    for case in sorted(cases, key=lambda c: c.id):
        h.update(canonical_json(case.model_dump(mode="json")).encode())
        h.update(b"\n")
    return h.hexdigest()[:16]
