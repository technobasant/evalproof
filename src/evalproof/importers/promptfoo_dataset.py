"""Read promptfoo test cases as an evalproof dataset (``evalproof import-dataset promptfoo``).

promptfoo keeps its cases as ``tests:`` in ``promptfooconfig.yaml``, or in files the config points
at with ``file://`` (YAML, JSON, JSONL or CSV, relative to the config, globs allowed). This reader
accepts the config and each of those files on its own.

* ``defaultTest`` is inherited: its ``vars`` and ``metadata`` are overridden by the test's own, and
  its ``assert`` list applies to every test, after the test's own.
* ``vars`` become a dict input; ``{{var}}`` inside an assertion value is filled from them.
* In a CSV, plain columns are variables, ``__expected`` and ``__expected1``, ``__expected2``, ...
  hold ``type:value`` (``equals`` when there is no type), ``__description``, ``__metadata:key`` and
  ``__metadata:tags[]`` are read, and ``__metric``, ``__threshold``, ``__prefix`` and ``__suffix``
  are counted as fields not carried over.

What each assertion becomes is in ``map_assertions``. promptfoo checks are all required, so two
assertions on one case are both required here too. An assertion no scorer can express stays on the
case under ``unmapped`` and is counted in the one summary note.

This is written from promptfoo's own documentation of the config, the CSV convention and the
assertion types. The value shape of ``contains-json`` is not confirmed, so one without a schema
object is left unmapped rather than guessed.
"""

from __future__ import annotations

import csv
import glob
import io
import re
from pathlib import Path
from typing import Any

import yaml

from evalproof.importers._common import ImportFormatError, parse_json, parse_jsonl, read_text
from evalproof.importers._dataset import (
    Builder,
    DatasetOptions,
    Draft,
    ImportedDataset,
    as_tag_list,
    csv_rows,
    jsonable,
    slug,
    text_of_value,
)

FORMAT = "promptfoo"

_ASSERTION_TYPES = frozenset(
    {
        "answer-relevance",
        "assert-set",
        "bleu",
        "classifier",
        "contains",
        "contains-all",
        "contains-any",
        "contains-json",
        "contains-sql",
        "contains-xml",
        "context-faithfulness",
        "context-recall",
        "context-relevance",
        "cost",
        "ends-with",
        "equals",
        "f-score",
        "factuality",
        "g-eval",
        "gleu",
        "guardrails",
        "icontains",
        "icontains-all",
        "icontains-any",
        "is-json",
        "is-refusal",
        "is-sql",
        "is-valid-function-call",
        "is-valid-openai-function-call",
        "is-valid-openai-tools-call",
        "is-xml",
        "javascript",
        "latency",
        "levenshtein",
        "llm-rubric",
        "meteor",
        "model-graded-closedqa",
        "model-graded-factuality",
        "moderation",
        "perplexity",
        "perplexity-score",
        "pi",
        "python",
        "regex",
        "rouge-l",
        "rouge-n",
        "ruby",
        "select-best",
        "similar",
        "starts-with",
        "webhook",
        "word-count",
    }
)
_TYPE_ALIASES = {"fn": "javascript", "eval": "javascript", "grade": "llm-rubric"}
_LIST_TYPES = frozenset({"contains-any", "contains-all", "icontains-any", "icontains-all"})
_RUBRIC_TYPES = frozenset({"llm-rubric", "g-eval", "model-graded-closedqa", "factuality"})
_REFERENCE_TYPES = frozenset({"factuality", "model-graded-factuality"})
_VARIABLE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_UNSUPPORTED_REF = re.compile(r"^[a-z]+://", re.I)


def read(path: Path, opts: DatasetOptions) -> ImportedDataset:
    if opts.has_fields:
        raise ImportFormatError(
            "--id-field, --input-field, --reference-field and --tags-field apply to the jsonl "
            "and csv formats, not promptfoo"
        )
    builder = Builder(FORMAT, path)
    suffix = path.suffix.lower()
    text = read_text(path)
    default: dict[str, Any] = {}
    prompt: str | None = None
    if suffix in (".yaml", ".yml"):
        data = _yaml(text, path)
    elif suffix == ".jsonl":
        data = parse_jsonl(text, path)
    elif suffix == ".csv":
        data = [_csv_test(builder, row) for row in csv_rows(text, path)]
    elif suffix == ".json":
        data = parse_json(text, path)
    else:
        raise ImportFormatError(
            f"{path}: expected a promptfooconfig.yaml or a .yaml, .json, .jsonl or .csv test file"
        )
    tests: list[Any]
    if isinstance(data, dict) and _is_config(data):
        tests = _as_list(data.get("tests"))
        default = _default_test(builder, data.get("defaultTest"), path.parent)
        prompt = _single_prompt(data.get("prompts"))
    elif isinstance(data, list):
        tests = data
    elif isinstance(data, dict):
        tests = [data]
    else:
        raise ImportFormatError(f"{path}: expected a promptfoo config or a list of tests")
    items = _expand(builder, tests, path.parent, 0)
    if not items:
        raise ImportFormatError(
            f"{path}: no tests found (a promptfoo config lists them under 'tests:')"
        )
    slugs = [slug(str(t["description"])) if t.get("description") else "" for t in items]
    for position, test in enumerate(items):
        unique = bool(slugs[position]) and slugs.count(slugs[position]) == 1
        _draft(builder, test, default, slugs[position] if unique else None, position)
    return _finish(builder, prompt)


def _is_config(data: dict[str, Any]) -> bool:
    return any(k in data for k in ("tests", "prompts", "providers", "defaultTest"))


def _yaml(text: str, path: Path) -> Any:
    try:
        return yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ImportFormatError(f"{path}: not valid YAML: {exc}") from exc


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


# --- finding the tests ------------------------------------------------------------------------


def _default_test(builder: Builder, value: Any, base: Path) -> dict[str, Any]:
    if isinstance(value, str):
        loaded = _expand(builder, [value], base, 0)
        value = loaded[0] if loaded else {}
    return dict(value) if isinstance(value, dict) else {}


def _expand(builder: Builder, items: list[Any], base: Path, depth: int) -> list[dict[str, Any]]:
    """Every test in ``items``, reading the ``file://`` references (and globs) they hold."""
    tests: list[dict[str, Any]] = []
    for item in items:
        entry = item
        if isinstance(item, dict) and isinstance(item.get("path"), str) and "vars" not in item:
            entry = item["path"]
        if isinstance(entry, dict):
            tests.append(entry)
        elif isinstance(entry, str):
            if depth > 3:
                builder.note(f"test reference {entry!r} nests too deeply and was skipped")
                continue
            loaded = _load_reference(builder, entry, base)
            tests.extend(_expand(builder, loaded, base, depth + 1))
        else:
            builder.note(f"a test that is not an object or a file reference was skipped: {item!r}")
    return tests


def _load_reference(builder: Builder, ref: str, base: Path) -> list[Any]:
    if ref.startswith("file://"):
        target = ref.removeprefix("file://")
    elif _UNSUPPORTED_REF.match(ref):
        builder.note(
            f"test source not imported: {ref} (only local YAML, JSON, JSONL and CSV files)"
        )
        return []
    else:
        target = ref
    target = target.split("#", 1)[0]
    pattern = target if Path(target).is_absolute() else str(base / target)
    paths = (
        sorted(glob.glob(pattern, recursive=True))
        if any(ch in pattern for ch in "*?[")
        else [pattern]
    )
    found: list[Any] = []
    for name in paths:
        file = Path(name)
        suffix = file.suffix.lower()
        if suffix in (".yaml", ".yml", ".json", ".jsonl", ".csv"):
            if not file.exists():
                raise ImportFormatError(f"{ref}: no such file {file}")
            found.extend(_load_file(builder, file))
        else:
            builder.note(
                f"test source not imported: {ref} ({suffix or 'no extension'} files are not read; "
                "only YAML, JSON, JSONL and CSV)"
            )
    return found


def _load_file(builder: Builder, file: Path) -> list[Any]:
    text = read_text(file)
    suffix = file.suffix.lower()
    if suffix == ".csv":
        return [_csv_test(builder, row) for row in csv_rows(text, file)]
    if suffix == ".jsonl":
        return list(parse_jsonl(text, file))
    data = _yaml(text, file) if suffix in (".yaml", ".yml") else parse_json(text, file)
    return _as_list(data)


def _csv_test(builder: Builder, row: dict[str, str]) -> dict[str, Any]:
    """One CSV row as the test object promptfoo reads it from the file."""
    variables: dict[str, Any] = {}
    asserts: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {}
    test: dict[str, Any] = {}
    for column, cell in row.items():
        name = column.strip()
        if name == "__description":
            if cell.strip():
                test["description"] = cell.strip()
        elif re.fullmatch(r"__expected\d*", name):
            parsed = parse_expected(cell)
            if parsed is not None:
                asserts.append(parsed)
        elif name.startswith("__metadata:"):
            key = name.removeprefix("__metadata:")
            if key.endswith("[]"):
                metadata[key[:-2]] = [
                    p.replace("\\,", ",").strip() for p in re.split(r"(?<!\\),", cell) if p.strip()
                ]
            elif cell.strip():
                metadata[key] = cell
        elif name.startswith("__"):
            if cell.strip():
                builder.ignore(f"column {name}")
        else:
            variables[name] = cell
    test["vars"] = variables
    if asserts:
        test["assert"] = asserts
    if metadata:
        test["metadata"] = metadata
    return test


def parse_expected(cell: str) -> dict[str, Any] | None:
    """A CSV ``__expected`` cell as an assertion: ``contains:yes``, ``not-contains:x``,
    ``similar(0.8):x``, ``contains-any:a,b``; a cell with no known type is ``equals``."""
    text = cell.strip()
    if not text:
        return None
    if text in _ASSERTION_TYPES or (text.startswith("not-") and text[4:] in _ASSERTION_TYPES):
        return {"type": text}
    match = re.match(r"^(not-)?([a-z][a-z0-9-]*)(?:\(([0-9.]+)\))?:(.*)$", text, re.S)
    if match:
        kind = _TYPE_ALIASES.get(match.group(2), match.group(2))
        if kind in _ASSERTION_TYPES:
            assertion: dict[str, Any] = {"type": f"{match.group(1) or ''}{kind}"}
            value: Any = match.group(4).strip()
            if kind in _LIST_TYPES:
                value = [v.strip() for v in next(csv.reader(io.StringIO(value)), [])]
            assertion["value"] = value
            if match.group(3):
                assertion["threshold"] = float(match.group(3))
            return assertion
    if text.startswith("file://"):
        return {"type": "javascript", "value": text}
    return {"type": "equals", "value": text}


# --- one test ---------------------------------------------------------------------------------


def _draft(
    builder: Builder, test: dict[str, Any], default: dict[str, Any], source_id: str | None, pos: int
) -> None:
    where = f"tests[{pos}]"
    variables: dict[str, Any] = {}
    for source in (default, test):
        raw_vars = source.get("vars")
        if isinstance(raw_vars, dict):
            variables.update({str(k): v for k, v in raw_vars.items()})
        elif raw_vars is not None:
            builder.note(f"{where}: vars given as a file reference are not read")
    for value in variables.values():
        if isinstance(value, str) and value.startswith("file://"):
            builder.tally("file_vars")
    metadata: dict[str, Any] = {}
    for source in (default, test):
        if isinstance(source.get("metadata"), dict):
            metadata.update(source["metadata"])
    assertions = [*_as_list(default.get("assert")), *_as_list(test.get("assert"))]
    draft = Draft(
        input=jsonable(variables),
        source_id=source_id,
        tags=as_tag_list(metadata.pop("tags", None)),
        key=jsonable(assertions),
        where=where,
    )
    if metadata:
        draft.extras["metadata"] = jsonable(metadata)
    for extra in ("provider", "providers", "prompts", "options", "providerOutput"):
        if extra in test:
            builder.ignore(extra)
    map_assertions(builder, draft, assertions, variables)
    builder.add(draft)


def _flatten(builder: Builder, draft: Draft, assertions: list[Any]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for item in assertions:
        if not isinstance(item, dict):
            builder.unmap(draft, "invalid assertion", item)
        elif item.get("type") == "assert-set" and item.get("threshold") is None:
            flat.extend(_flatten(builder, draft, _as_list(item.get("assert"))))
        else:
            flat.append(item)
    return flat


class _Unresolved(Exception):
    pass


def _resolve(value: Any, variables: dict[str, Any]) -> Any:
    """``value`` with ``{{var}}`` filled from the test's vars. Raises ``_Unresolved`` when a
    template is left that the vars cannot fill, or the value is a file."""
    if isinstance(value, list):
        return [_resolve(v, variables) for v in value]
    if not isinstance(value, str):
        return value
    if value.startswith("file://"):
        raise _Unresolved

    def fill(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in variables:
            raise _Unresolved
        return text_of_value(variables[name])

    filled = _VARIABLE.sub(fill, value)
    if "{{" in filled or "{%" in filled:
        raise _Unresolved
    return filled


def _strings(value: Any) -> list[str] | None:
    """A list of strings from a list value, or a comma-separated string; None if it is neither."""
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list) and all(isinstance(v, str | int | float) for v in value):
        return [text_of_value(v) for v in value]
    return None


def _scalar(value: Any) -> str | None:
    if isinstance(value, str | int | float) and not isinstance(value, bool):
        return text_of_value(value)
    return None


class _Groups:
    """What the string assertions of one case ask for, by case sensitivity."""

    def __init__(self) -> None:
        self.singles: dict[bool, list[str]] = {True: [], False: []}
        self.anys: dict[bool, list[tuple[dict[str, Any], list[str]]]] = {True: [], False: []}
        self.alls: dict[bool, list[str]] = {True: [], False: []}
        self.patterns: dict[bool, list[tuple[dict[str, Any], str]]] = {True: [], False: []}


def map_assertions(
    builder: Builder, draft: Draft, assertions: list[Any], variables: dict[str, Any]
) -> None:
    """Turn promptfoo assertions into case fields (see the table in the import guide).

    ==========================================  ==============================================
    equals                                      ``reference`` + ``exact_match``
    contains, icontains, contains-any           ``accept`` + ``contains``
    contains-all (and ``contains`` x 2 or more) ``require`` + ``contains``
    starts-with, regex                          ``pattern`` + ``regex`` (``^prefix``)
    not-contains, not-icontains, not-regex, ... an ``output_omits`` or ``not`` check in
                                                ``expect`` + ``checks``
    is-json, contains-json with a schema        ``schema`` + ``json_schema``
    llm-rubric, g-eval, model-graded-closedqa,  ``rubrics`` + ``llm_judge`` (``case_rubrics``)
    factuality
    anything else                               ``unmapped``, counted
    ==========================================  ==============================================
    """
    groups = _Groups()
    for assertion in _flatten(builder, draft, assertions):
        kind = str(assertion.get("type") or "")
        if assertion.get("transform") or assertion.get("contextTransform"):
            builder.unmap(draft, f"{kind} (with transform)", assertion)
            continue
        weight = assertion.get("weight")
        if isinstance(weight, int | float) and not isinstance(weight, bool) and weight == 0:
            builder.unmap(draft, f"{kind} (weight 0, informational)", assertion)
            continue
        try:
            value = _resolve(assertion.get("value"), variables)
        except _Unresolved:
            builder.unmap(draft, f"{kind} (value from a template or file)", assertion)
            continue
        negate = kind.startswith("not-")
        base = kind.removeprefix("not-")
        if not _map_one(builder, draft, assertion, kind, base, negate, value, groups):
            builder.unmap(draft, kind or "(no type)", assertion)
    _apply_groups(builder, draft, groups)


def _map_one(
    builder: Builder,
    draft: Draft,
    assertion: dict[str, Any],
    kind: str,
    base: str,
    negate: bool,
    value: Any,
    groups: _Groups,
) -> bool:
    """Map one assertion; False when nothing can express it."""
    sensitive = base in ("contains", "contains-all", "contains-any", "regex", "starts-with")
    text = _scalar(value)
    if base == "equals":
        if text is None:
            return False
        if negate:
            draft.checks.append(_not_regex(f"\\A{re.escape(text)}\\Z", ""))
            return True
        if draft.reference is not None:
            return False
        draft.reference, draft.exact = text, True
        return True
    if base in ("contains", "icontains"):
        if text is None:
            return False
        if negate:
            draft.checks.append(_omits(text, sensitive))
        else:
            groups.singles[sensitive].append(text)
        return True
    if base in ("contains-any", "icontains-any", "contains-all", "icontains-all"):
        values = _strings(value)
        if not values:
            return False
        if negate:
            if "all" in base:
                return False
            draft.checks.extend(_omits(v, sensitive) for v in values)
        elif "all" in base:
            groups.alls[sensitive].extend(values)
        else:
            groups.anys[sensitive].append((assertion, values))
        return True
    if base in ("starts-with", "regex"):
        if text is None:
            return False
        pattern = f"^{re.escape(text)}" if base == "starts-with" else text
        try:
            re.compile(pattern)
        except re.error:
            return False
        if negate:
            draft.checks.append(_not_regex(pattern, ""))
        else:
            groups.patterns[True].append((assertion, pattern))
        return True
    if base in ("is-json", "contains-json"):
        if negate or not isinstance(value, dict) or not value or draft.schema is not None:
            return False
        draft.schema = jsonable(value)
        return True
    if base in _RUBRIC_TYPES and not negate:
        values = _strings(value) if isinstance(value, list) else ([value] if text else None)
        if not values:
            return False
        for rubric in values:
            draft.rubrics.append(
                f"The response is factually consistent with this reference: {rubric}"
                if base in _REFERENCE_TYPES
                else rubric
            )
        if assertion.get("threshold") is not None or assertion.get("weight") not in (None, 1):
            builder.ignore("assertion threshold or weight")
        return True
    return False


def _omits(text: str, sensitive: bool) -> dict[str, Any]:
    check: dict[str, Any] = {"type": "output_omits", "text": text}
    if sensitive:
        check["case_sensitive"] = True
    return check


def _not_regex(pattern: str, flags: str) -> dict[str, Any]:
    check: dict[str, Any] = {"type": "output_regex", "pattern": pattern}
    if flags:
        check["flags"] = flags
    return {"type": "not", "check": check}


def _apply_groups(builder: Builder, draft: Draft, groups: _Groups) -> None:
    for sensitive in (True, False):
        singles = groups.singles[sensitive]
        anys = groups.anys[sensitive]
        require = list(groups.alls[sensitive])
        accept: list[str] = []
        if anys:
            accept = anys[0][1]
            for raw, _ in anys[1:]:
                builder.unmap(draft, f"{raw.get('type')} (second any-of group)", raw)
        if len(singles) == 1 and not accept:
            accept = singles
        else:
            require.extend(singles)
        if accept:
            draft.add_list("accept", accept, case_sensitive=sensitive)
        if require:
            draft.add_list("require", require, case_sensitive=sensitive)
    for raw, pattern in groups.patterns[True]:
        if draft.lists.get(("pattern", True)):
            builder.unmap(draft, f"{raw.get('type')} (second pattern)", raw)
        else:
            draft.add_list("pattern", [pattern], case_sensitive=True)


# --- the prompt -------------------------------------------------------------------------------


def _single_prompt(prompts: Any) -> str | None:
    """The one inline prompt of the config as a ``{var}`` template, or None (no prompt, several
    prompts, a file, a chat in JSON, or a template that uses more than plain ``{{var}}``)."""
    items = _as_list(prompts)
    if len(items) != 1:
        return None
    prompt = items[0]
    if isinstance(prompt, dict):
        prompt = prompt.get("raw")
    if not isinstance(prompt, str) or _UNSUPPORTED_REF.match(prompt):
        return None
    if prompt.lstrip()[:1] in ("[", "{") or not (" " in prompt or "{{" in prompt):
        return None
    pieces = _VARIABLE.split(prompt)
    literal = pieces[0::2]
    if any("{{" in p or "{%" in p for p in literal):
        return None
    out = []
    for position, piece in enumerate(pieces):
        out.append(f"{{{piece}}}" if position % 2 else piece.replace("{", "{{").replace("}", "}}"))
    return "".join(out)


def _finish(builder: Builder, prompt: str | None) -> ImportedDataset:
    """The dataset, with the config's prompt as the template when every case has its variables."""
    result = builder.finish()
    if prompt is not None:
        names = set(re.findall(r"(?<!\{)\{([A-Za-z_][A-Za-z0-9_]*)\}(?!\})", prompt))
        if all(isinstance(c.input, dict) and names <= set(c.input) for c in result.cases):
            result.prompt_template = prompt
    return result
