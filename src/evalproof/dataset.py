"""Golden sets as JSONL: one case per line, stable ids, and a content hash for provenance."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalproof.models import Case, TagFilter


class DatasetError(ValueError):
    pass


def parse_tag_list(values: Sequence[str] | None, option: str) -> list[str]:
    """The tags of a repeatable, comma-separated option (``--tags a,b --tags c``): sorted, with
    blanks and duplicates dropped. ``None`` (the option was not given) is an empty list; an option
    that was given but names no tag, such as ``--tags ""``, is an error."""
    if values is None:
        return []
    tags = sorted({tag.strip() for value in values for tag in value.split(",") if tag.strip()})
    if not tags:
        raise ValueError(f"{option} needs at least one tag")
    return tags


def make_tag_filter(
    tags: Sequence[str] | None, exclude_tags: Sequence[str] | None
) -> TagFilter | None:
    """The filter ``--tags`` and ``--exclude-tags`` ask for; ``None`` when neither was given."""
    include = parse_tag_list(tags, "--tags")
    exclude = parse_tag_list(exclude_tags, "--exclude-tags")
    return TagFilter(include=include, exclude=exclude) if include or exclude else None


def select_cases(cases: list[Case], tag_filter: TagFilter | None) -> list[Case]:
    """The cases ``tag_filter`` selects (all of them without a filter). Raises ``DatasetError``,
    naming the tags the dataset does have, when the filter selects none."""
    if tag_filter is None:
        return cases
    selected = tag_filter.apply(cases)
    if not selected:
        known = sorted({tag for case in cases for tag in case.tags})
        have = f"the dataset's tags are {', '.join(known)}" if known else "no case has a tag"
        raise DatasetError(f"{tag_filter.describe()} selects no case: {have}")
    return selected


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_id(record: dict[str, Any]) -> str:
    """An id derived from the input, so reordering the file never re-keys a case."""
    digest = hashlib.sha256(canonical_json(record.get("input")).encode()).hexdigest()
    return f"case-{digest[:12]}"


def _rename_hint(field: str, meaning: str) -> str:
    return f"(evalproof reads {field!r} as {meaning}; rename the field if it means something else)"


def validate_expect(case: Case) -> None:
    """Check a case's inline ``expect`` checks against the ``checks`` scorer's vocabulary: a list
    of objects, each with a known ``type``, its required fields and fields of the right type.

    Raises ``ValueError`` naming the case and the path of the first problem, such as
    ``expect[2].checks[0]`` for a check inside an ``any_of``. A ``python`` check is only checked
    for looking like ``module:function``; the scorer imports it when the suite runs."""
    expect = (case.model_extra or {}).get("expect")
    if expect is None:
        return
    if not isinstance(expect, list):
        hint = _rename_hint("expect", "the inline checks of the checks scorer")
        raise ValueError(f"case {case.id!r}: expect must be a list of checks {hint}")
    from evalproof.scorers.checks import validate_checks  # lazy: the scorers import this module

    try:
        validate_checks(expect, "expect")
    except ValueError as exc:
        raise ValueError(f"case {case.id!r}: {exc}") from exc


def validate_case_fields(case: Case) -> None:
    """Check the optional agent fields of a case: ``turns`` (a list of strings or objects),
    ``state`` (an object) and ``expect`` (a list of checks). Raises ``ValueError``.

    The names are evalproof's, so a dataset that used one for something else fails here, and the
    message says to rename it."""
    extra = case.model_extra or {}
    turns = extra.get("turns")
    if turns is not None:
        if not isinstance(turns, list):
            hint = _rename_hint("turns", "the user turns that follow 'input'")
            raise ValueError(f"case {case.id!r}: turns must be a list of strings or objects {hint}")
        for i, turn in enumerate(turns):
            if not isinstance(turn, str | dict):
                raise ValueError(f"case {case.id!r}: turns[{i}] must be a string or an object")
    state = extra.get("state")
    if state is not None and not isinstance(state, dict):
        hint = _rename_hint("state", "the initial session state")
        raise ValueError(f"case {case.id!r}: state must be an object {hint}")
    validate_expect(case)


def load_dataset(path: str | Path) -> list[Case]:
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"dataset not found: {path}")
    cases: list[Case] = []
    seen: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
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
        try:
            validate_case_fields(case)
        except ValueError as exc:
            raise DatasetError(f"{path}:{lineno}: {exc}") from exc
        if case.id in seen:
            raise DatasetError(f"{path}:{lineno}: duplicate case id {case.id!r}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise DatasetError(f"dataset is empty: {path}")
    return cases


def write_dataset(cases: Sequence[Case], path: str | Path) -> Path:
    """Write ``cases`` as JSONL, one per line, the form ``load_dataset`` reads: each case without
    its default-valued fields, and with its extra fields (``turns``, ``expect``, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dataset_text(cases), encoding="utf-8", newline="\n")
    return path


def dataset_text(cases: Sequence[Case]) -> str:
    """The JSONL text ``write_dataset`` writes."""
    lines = [
        json.dumps(case.model_dump(mode="json", exclude_defaults=True), ensure_ascii=False)
        for case in cases
    ]
    return "\n".join(lines) + "\n"


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
