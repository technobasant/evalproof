"""What the dataset readers share: the result type, options, and the rules that must not differ
between formats.

A dataset reader turns another tool's TEST CASES (not its results) into evalproof cases, and says
which scorers those cases need, so that a suite can run on them straight away. ``Builder`` applies
the rules once:

* a case id is the source's own id when it has one, else ``<format>-<10 hex of a sha256 over the
  input and what the source expected>``. The hash reads the content, never the line number, so
  reordering the file never re-keys a case. A repeated id gets ``-2``, ``-3`` and a note;
* tags are the source's tags plus ``imported`` and ``from:<format>``;
* a chat input drops its system messages; if what is left is all user messages the first is the
  ``input`` and the rest are ``turns``, otherwise the input is ``{"messages": [...]}``;
* anything the scorers cannot express is never dropped silently: the raw assertion stays on the
  case under ``unmapped`` and the count per type is one summary note.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalproof.config import ScorerConfig
from evalproof.dataset import canonical_json, validate_case_fields
from evalproof.importers._common import (
    ImportFormatError,
    _default_warn,
    parse_json,
    parse_jsonl,
    read_text,
)
from evalproof.models import Case

IMPORTED_TAG = "imported"

# Names a prompt template cannot use for an input key: ``Case.render_vars`` already sets them.
_RESERVED_VARS = frozenset({"id", "input", "reference", "tags"})

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

_LIST_KEYS = (
    "examples",
    "data",
    "items",
    "samples",
    "goldens",
    "test_cases",
    "cases",
    "tests",
    "rows",
    "records",
    "dataset",
)

# One summary line per kind of thing a reader changed, with ``{n}`` the number of cases.
_TALLIES = {
    "system_dropped": "system messages were dropped from {n} case(s) (set the system prompt in "
    "the suite instead)",
    "messages_input": "{n} case(s) keep a conversation that is not user-only as "
    '{{"messages": [...]}}; a system that takes text needs a prompt for it',
    "no_input": "{n} record(s) skipped: no input",
    "scenario_only": "{n} golden(s) skipped: a scenario with no turns is a user simulation, "
    "which has no fixed turns to replay",
    "multi_output": "{n} case(s) had several output keys; the reference is their JSON",
    "structured_expected": "{n} case(s) had a structured expected value; the reference is its JSON",
    "file_vars": "{n} variable value(s) are file:// references, kept as written",
    "first_truth": "{n} case(s) had several ground truths; the first is the reference",
}


@dataclass
class DatasetOptions:
    """What the caller asked for. The four field options apply to the generic ``jsonl`` and
    ``csv`` formats only."""

    warn: Callable[[str], None] = _default_warn
    id_field: str | None = None
    input_fields: tuple[str, ...] = ()
    reference_field: str | None = None
    tags_field: str | None = None

    @property
    def has_fields(self) -> bool:
        return bool(self.id_field or self.input_fields or self.reference_field or self.tags_field)


@dataclass
class ImportedDataset:
    """The cases an import found, the scorers those cases need, and notes for the person
    running it. ``prompt_template`` is the ``{field}`` template that turns a dict input into the
    user message (None when the plain ``{input}`` does). ``detected`` is True when ``auto`` chose
    the format."""

    cases: list[Case]
    scorers: list[ScorerConfig]
    notes: list[str] = field(default_factory=list)
    format: str = ""
    prompt_template: str | None = None
    detected: bool = False
    coverage: dict[str, int] = field(default_factory=dict)
    """How many cases each scorer checks, by scorer name."""

    @property
    def needs_judge(self) -> bool:
        return any(s.type == "llm_judge" for s in self.scorers)

    @property
    def gate_metric(self) -> str | None:
        """The scorer that checks the most cases (the first of them on a tie): the one to gate,
        since a gated metric needs ``min_cases`` paired cases of its own."""
        if not self.scorers:
            return None
        return max(self.scorers, key=lambda s: self.coverage.get(s.name, 0)).name

    def scorer_blocks(self) -> list[dict[str, Any]]:
        """The scorers as the plain mappings of a suite file."""
        blocks = []
        for scorer in self.scorers:
            block: dict[str, Any] = {"name": scorer.name, "type": scorer.type}
            if scorer.params:
                block["params"] = scorer.params
            blocks.append(block)
        return blocks


@dataclass
class Draft:
    """One case while it is being read. ``key`` is whatever the source expected, as the source
    wrote it: it goes into the fallback id and nowhere else."""

    input: str | dict[str, Any]
    reference: str | None = None
    source_id: str | None = None
    tags: list[str] = field(default_factory=list)
    extras: dict[str, Any] = field(default_factory=dict)
    key: Any = None
    turns: list[str | dict[str, Any]] = field(default_factory=list)
    lists: dict[tuple[str, bool], list[str]] = field(default_factory=dict)
    checks: list[dict[str, Any]] = field(default_factory=list)
    rubrics: list[Any] = field(default_factory=list)
    unmapped: list[Any] = field(default_factory=list)
    schema: dict[str, Any] | None = None
    exact: bool = False
    judge: bool = False
    where: str = ""

    def add_list(self, base: str, values: Iterable[str], *, case_sensitive: bool) -> None:
        """Add to the per-case list ``base`` (``accept``, ``require`` or ``pattern``); the
        case-sensitive and the case-insensitive ones are kept apart because the scorer is one or
        the other."""
        bucket = self.lists.setdefault((base, case_sensitive), [])
        for value in values:
            if value not in bucket:
                bucket.append(value)


def jsonable(value: Any) -> Any:
    """``value`` as plain JSON data: a YAML date or any other object becomes its text."""
    return json.loads(json.dumps(value, default=str, ensure_ascii=False))


def text_of_value(value: Any) -> str:
    """A reference as text: a string as it is, a number as its digits, anything else as JSON."""
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return canonical_json(jsonable(value))


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def as_tag_list(value: Any) -> list[str]:
    """Tags from a list, or from one string (split on commas)."""
    if isinstance(value, str):
        return [t.strip() for t in value.split(",") if t.strip()]
    if isinstance(value, list):
        return [str(t) for t in value if isinstance(t, str | int) and str(t).strip()]
    return []


# --- chat messages ----------------------------------------------------------------------------

_ROLES = {"human": "user", "ai": "assistant", "developer": "system"}


def _role(message: dict[str, Any]) -> str:
    raw = message.get("role", message.get("type"))
    return _ROLES.get(str(raw), str(raw)) if raw is not None else "user"


def _message_text(content: Any) -> str | None:
    """The text of a message's content: a string, or the text parts of a list of parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            p.get("text") if isinstance(p, dict) else p
            for p in content
            if isinstance(p, str) or (isinstance(p, dict) and p.get("type") in (None, "text"))
        ]
        if parts and all(isinstance(p, str) for p in parts) and len(parts) == len(content):
            return "\n".join(str(p) for p in parts)
    return None


def looks_like_messages(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(m, dict) and ("content" in m or "text" in m) and ("role" in m or "type" in m)
            for m in value
        )
    )


# --- files ------------------------------------------------------------------------------------


def read_records(path: Path) -> list[dict[str, Any]]:
    """The records of a json, jsonl or csv file, as dicts.

    A ``.json`` file is a list of objects, or an object holding such a list under a usual key
    (``examples``, ``data``, ``items``, ...), or one record. A CSV cell stays text."""
    suffix = path.suffix.lower()
    text = read_text(path)
    if suffix == ".csv":
        return csv_rows(text, path)
    if suffix in (".jsonl", ".ndjson"):
        return parse_jsonl(text, path)
    if suffix == ".json" or text.lstrip()[:1] in ("[", "{"):
        try:
            data = parse_json(text, path)
        except ImportFormatError as first:
            try:
                return parse_jsonl(text, path)
            except ImportFormatError:
                raise first from None  # the JSON message is the more useful one
        return records_of(data, path)
    raise ImportFormatError(
        f"{path}: unsupported file type {suffix or '(none)'}; expected .json, .jsonl or .csv"
    )


def records_of(data: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(data, list):
        if not all(isinstance(r, dict) for r in data):
            raise ImportFormatError(f"{path}: every record must be a JSON object")
        return list(data)
    if isinstance(data, dict):
        for key in _LIST_KEYS:
            value = data.get(key)
            if isinstance(value, list) and value and all(isinstance(r, dict) for r in value):
                return list(value)
        return [data]
    raise ImportFormatError(f"{path}: expected a list of objects")


def csv_rows(text: str, path: Path | str) -> list[dict[str, Any]]:
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        return [
            {k: (v if v is not None else "") for k, v in row.items() if k is not None}
            for row in reader
        ]
    except csv.Error as exc:
        raise ImportFormatError(f"{path}: not valid CSV: {exc}") from exc


def csv_cell(value: Any) -> Any:
    """A CSV cell that holds a JSON object or array as that value; any other cell as it is."""
    if isinstance(value, str) and value.strip()[:1] in ("{", "["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def column_names(path: Path) -> set[str]:
    """The keys across the first records of a json, jsonl or csv file (for ``auto``)."""
    keys: set[str] = set()
    for record in read_records(path)[:50]:
        keys.update(str(k) for k in record)
    return keys


# --- the builder ------------------------------------------------------------------------------


class Builder:
    """Collects the drafts of one import and turns them into an ``ImportedDataset``."""

    def __init__(self, fmt: str, path: Path) -> None:
        self.fmt = fmt
        self.path = path
        self.drafts: list[Draft] = []
        self.notes: list[str] = []
        self.tallies: Counter[str] = Counter()
        self.ignored: Counter[str] = Counter()
        self.unmapped_types: Counter[str] = Counter()

    # reading --------------------------------------------------------------------------------

    def add(self, draft: Draft) -> Draft:
        self.drafts.append(draft)
        return draft

    def tally(self, kind: str, n: int = 1) -> None:
        self.tallies[kind] += n

    def ignore_present(self, record: dict[str, Any], names: Iterable[str]) -> None:
        """Count each of ``names`` that the record holds with a value."""
        for name in names:
            if record.get(name) not in (None, "", [], {}):
                self.ignore(name)

    def ignore(self, name: str, n: int = 1) -> None:
        """A field of the source this importer does not carry over."""
        self.ignored[name] += n

    def unmap(self, draft: Draft, kind: str, raw: Any) -> None:
        """Keep ``raw`` on the case under ``unmapped`` and count it under ``kind``."""
        draft.unmapped.append(jsonable(raw))
        self.unmapped_types[kind] += 1

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)

    def error(self, where: str, message: str) -> ImportFormatError:
        return ImportFormatError(f"{self.path}: {where}: {message}")

    def chat(
        self, messages: list[Any], where: str
    ) -> tuple[str | dict[str, Any], list[Any]] | None:
        """The input and turns of a chat. None when nothing but system messages is left."""
        normalized: list[dict[str, Any]] = []
        for item in messages:
            if isinstance(item, str):
                normalized.append({"role": "user", "content": item})
                continue
            if not isinstance(item, dict):
                raise self.error(where, "a message must be a string or an object")
            message = {k: v for k, v in item.items() if k != "type" or "role" in item}
            message["role"] = _role(item)
            if "content" not in message and "text" in message:
                message["content"] = message.pop("text")
            normalized.append(message)
        kept = [m for m in normalized if m["role"] != "system"]
        if len(kept) != len(normalized):
            self.tally("system_dropped")
        if not kept:
            return None
        texts = [_message_text(m.get("content")) for m in kept]
        if all(m["role"] == "user" for m in kept) and all(t is not None for t in texts):
            strings = [str(t) for t in texts]
            return strings[0], strings[1:]
        self.tally("messages_input")
        return {"messages": jsonable(kept)}, []

    def input_of(self, raw: Any, where: str) -> tuple[str | dict[str, Any], list[Any]] | None:
        """A case input and its turns from whatever the source calls the input: a string, a
        chat (a list of messages, or ``{"messages": [...]}``), or an object of variables.
        None when there is nothing to ask."""
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return None
        if isinstance(raw, str):
            return raw, []
        if isinstance(raw, list):
            if looks_like_messages(raw) or all(isinstance(m, str) for m in raw):
                return self.chat(raw, where) if raw else None
            return {"input": jsonable(raw)}, []
        if isinstance(raw, dict):
            if set(raw) == {"messages"} and isinstance(raw["messages"], list):
                return self.chat(raw["messages"], where)
            return jsonable(raw), []
        return str(raw), []

    # finishing ------------------------------------------------------------------------------

    def _fallback_id(self, draft: Draft) -> str:
        payload = {"input": draft.input, "turns": draft.turns, "expected": draft.key}
        digest = hashlib.sha256(canonical_json(jsonable(payload)).encode()).hexdigest()
        return f"{self.fmt}-{digest[:10]}"

    def _ids(self) -> list[str]:
        wanted = [d.source_id or self._fallback_id(d) for d in self.drafts]
        reserved = set(wanted)
        assigned: set[str] = set()
        ids: list[str] = []
        repeated: list[str] = []
        for cid in wanted:
            final = cid
            if final in assigned:
                n = 2
                while f"{cid}-{n}" in assigned or f"{cid}-{n}" in reserved:
                    n += 1
                final = f"{cid}-{n}"
                repeated.append(cid)
            assigned.add(final)
            ids.append(final)
        if repeated:
            shown = ", ".join(repr(c) for c in list(dict.fromkeys(repeated))[:5])
            self.note(f"{len(repeated)} repeated case id(s) were numbered -2, -3, ...: {shown}")
        return ids

    def finish(self, *, template: str | None = None) -> ImportedDataset:
        if not self.drafts:
            raise ImportFormatError(f"{self.path}: no cases found")
        ids = self._ids()
        names = _FieldNames(self.drafts)
        cases = [self._case(draft, cid, names) for draft, cid in zip(self.drafts, ids, strict=True)]
        scorers = _scorers(self.drafts, names)
        return ImportedDataset(
            cases=cases,
            scorers=scorers,
            notes=self._notes(),
            format=self.fmt,
            prompt_template=template if template is not None else choose_template(cases),
            coverage=_coverage(self.drafts, names),
        )

    def _case(self, draft: Draft, cid: str, names: _FieldNames) -> Case:
        extras = dict(draft.extras)
        for (base, sensitive), values in draft.lists.items():
            if values:
                extras[names.field(base, sensitive)] = values
        if draft.turns:
            extras["turns"] = draft.turns
        if draft.checks:
            extras["expect"] = draft.checks
        if draft.rubrics:
            extras["rubrics"] = draft.rubrics
        if draft.schema is not None:
            extras["schema"] = draft.schema
        if draft.unmapped:
            extras["unmapped"] = draft.unmapped
        tags = list(dict.fromkeys([*draft.tags, IMPORTED_TAG, f"from:{self.fmt}"]))
        record: dict[str, Any] = {
            **extras,
            "id": cid,
            "input": draft.input,
            "reference": draft.reference,
            "tags": tags,
        }
        try:
            case = Case.model_validate(record)
            validate_case_fields(case)
        except ValueError as exc:
            raise ImportFormatError(f"{self.path}: {draft.where or cid}: {exc}") from exc
        return case

    def _notes(self) -> list[str]:
        notes = list(self.notes)
        for kind, n in self.tallies.items():
            notes.append(_TALLIES[kind].format(n=n))
        if self.ignored:
            listing = ", ".join(f"{name} x {n}" for name, n in self.ignored.most_common())
            notes.append(f"fields not carried over: {listing}")
        if self.unmapped_types:
            notes.append(unmapped_summary(self.unmapped_types))
        return notes


def unmapped_summary(counts: Counter[str]) -> str:
    """``12 assertions not imported: javascript x 8, similar x 4`` (largest first)."""
    total = sum(counts.values())
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    listing = ", ".join(f"{name} x {n}" for name, n in ordered)
    noun = "assertion" if total == 1 else "assertions"
    return f"{total} {noun} not imported: {listing} (kept on each case under 'unmapped')"


# --- the scorers a set of drafts needs --------------------------------------------------------

_CONTAINS = ("accept", "require")
_REGEX = ("pattern",)


class _FieldNames:
    """Field and scorer names. A family (contains, regex) has one scorer per case sensitivity,
    and the plain names go to the case-insensitive one, unless it is the only one in use."""

    def __init__(self, drafts: list[Draft]) -> None:
        used = {key for d in drafts for key, values in d.lists.items() if values}
        self.used = used
        self.split = {
            family: any((b, True) in used for b in bases) and any((b, False) in used for b in bases)
            for family, bases in (("contains", _CONTAINS), ("regex", _REGEX))
        }

    def family(self, base: str) -> str:
        return "contains" if base in _CONTAINS else "regex"

    def field(self, base: str, sensitive: bool) -> str:
        return f"{base}_cs" if sensitive and self.split[self.family(base)] else base

    def scorer(self, family: str, sensitive: bool) -> str:
        return f"{family}_cs" if sensitive and self.split[family] else family


def _scorers(drafts: list[Draft], names: _FieldNames) -> list[ScorerConfig]:
    scorers: list[ScorerConfig] = []
    for sensitive in (False, True):
        bases = {b for (b, s) in names.used if s is sensitive and b in _CONTAINS}
        if bases:
            params: dict[str, Any] = {}
            if "accept" in bases:
                params["any_of_field"] = names.field("accept", sensitive)
            if "require" in bases:
                params["all_of_field"] = names.field("require", sensitive)
            if sensitive:
                params["case_sensitive"] = True
            scorers.append(
                ScorerConfig(
                    name=names.scorer("contains", sensitive), type="contains", params=params
                )
            )
        if ("pattern", sensitive) in names.used:
            scorers.append(
                ScorerConfig(
                    name=names.scorer("regex", sensitive),
                    type="regex",
                    params={
                        "any_of_field": names.field("pattern", sensitive),
                        "flags": "" if sensitive else "i",
                    },
                )
            )
    if any(d.exact for d in drafts):
        scorers.insert(0, ScorerConfig(name="exact", type="exact_match"))
    if any(d.schema is not None for d in drafts):
        scorers.append(
            ScorerConfig(name="json_schema", type="json_schema", params={"schema_field": "schema"})
        )
    if any(d.checks for d in drafts):
        scorers.append(
            ScorerConfig(
                name="omits",
                type="checks",
                params={"field": "expect", "on_missing": "skip"},
            )
        )
    if any(d.judge for d in drafts):
        scorers.append(ScorerConfig(name="judge", type="llm_judge"))
    if any(d.rubrics for d in drafts):
        scorers.append(
            ScorerConfig(name="rubrics", type="llm_judge", params={"case_rubrics": True})
        )
    return scorers


def _coverage(drafts: list[Draft], names: _FieldNames) -> dict[str, int]:
    """How many drafts each scorer of ``_scorers`` checks, by scorer name."""
    counts: Counter[str] = Counter()
    for d in drafts:
        hit: set[str] = set()
        for (base, sensitive), values in d.lists.items():
            if values:
                hit.add(names.scorer(names.family(base), sensitive))
        if d.exact:
            hit.add("exact")
        if d.schema is not None:
            hit.add("json_schema")
        if d.checks:
            hit.add("omits")
        if d.judge:
            hit.add("judge")
        if d.rubrics:
            hit.add("rubrics")
        counts.update(hit)
    return dict(counts)


# --- the prompt template ----------------------------------------------------------------------


def choose_template(cases: list[Case]) -> str | None:
    """A ``prompt_template`` for dict inputs: ``{question}`` for one shared key, ``key: {key}``
    lines for several. None for string inputs, and when no key is shared by every case."""
    dicts = [c.input for c in cases if isinstance(c.input, dict)]
    if not dicts or len(dicts) != len(cases):
        return None
    shared = [
        k
        for k in dicts[0]
        if _IDENTIFIER.match(k) and k not in _RESERVED_VARS and all(k in d for d in dicts)
    ]
    if not shared:
        return None
    if len(shared) == 1:
        return f"{{{shared[0]}}}"
    return "\n".join(f"{k}: {{{k}}}" for k in shared)


# --- reader helpers ---------------------------------------------------------------------------


def reject_fields(opts: DatasetOptions, fmt: str) -> None:
    """The ``--*-field`` options belong to the generic ``jsonl`` and ``csv`` readers only."""
    if opts.has_fields:
        raise ImportFormatError(
            f"--id-field, --input-field, --reference-field and --tags-field apply to the jsonl "
            f"and csv formats, not {fmt}"
        )


def first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    """The value of the first of ``keys`` that the record holds with a value, else None."""
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return value
    return None


def source_id(value: Any) -> str | None:
    """A source id as text: a string or a whole number; anything else is no id."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def one_output(builder: Builder, value: Any) -> Any:
    """The expected answer of an ``outputs``/``output`` object: the one value it holds, or, with
    several keys, the object (which becomes the reference as JSON)."""
    if isinstance(value, dict):
        if len(value) == 1:
            return next(iter(value.values()))
        if value:
            builder.tally("multi_output")
            return canonical_json(jsonable(value))
        return None
    return value


# --- registry helpers -------------------------------------------------------------------------


def add_reference(
    builder: Builder,
    draft: Draft,
    value: Any,
    *,
    judge: bool = True,
    exact: bool = False,
) -> None:
    """Set the case's reference from what the source expected. Free text is judged against;
    ``exact`` says the source compares exactly."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return
    if not isinstance(value, str | int | float | bool):
        builder.tally("structured_expected")
    draft.reference = text_of_value(value)
    draft.judge = judge and not exact
    draft.exact = exact
