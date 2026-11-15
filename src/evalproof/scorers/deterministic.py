"""Scorers that need no model: string, pattern, schema and numeric checks."""

from __future__ import annotations

import json
import math
import re
from typing import Any

from evalproof.models import Case, Score, SystemOutput

NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" .!?;:\"'`")


def _case_list(case: Case, field: str | None) -> list[str]:
    if not field:
        return []
    value = case.field(field)
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def _binary(ok: bool, reason: str | None = None) -> Score:
    return Score(value=1.0 if ok else 0.0, passed=ok, reason=reason)


class ExactMatch:
    def __init__(self, name: str, field: str = "reference", normalize: bool = True) -> None:
        self.name, self.field, self.norm = name, field, normalize

    def score(self, case: Case, out: SystemOutput) -> Score:
        expected = case.field(self.field)
        if expected is None:
            return Score(value=None, reason=f"case has no {self.field!r}")
        got, want = out.output, str(expected)
        if self.norm:
            got, want = normalize(got), normalize(want)
        return _binary(got == want)


class Contains:
    """Substring checks. ``all_of`` must all appear; ``any_of`` needs at least one."""

    def __init__(
        self,
        name: str,
        any_of: list[str] | None = None,
        all_of: list[str] | None = None,
        any_of_field: str | None = None,
        all_of_field: str | None = None,
        case_sensitive: bool = False,
    ) -> None:
        self.name = name
        self.any_of, self.all_of = any_of or [], all_of or []
        self.any_of_field, self.all_of_field = any_of_field, all_of_field
        self.case_sensitive = case_sensitive

    def score(self, case: Case, out: SystemOutput) -> Score:
        text = out.output if self.case_sensitive else out.output.lower()

        def fold(s: str) -> str:
            return s if self.case_sensitive else s.lower()

        any_of = self.any_of + _case_list(case, self.any_of_field)
        all_of = self.all_of + _case_list(case, self.all_of_field)
        if not any_of and not all_of:
            return Score(value=None, reason="nothing to look for")
        missing = [s for s in all_of if fold(s) not in text]
        any_ok = not any_of or any(fold(s) in text for s in any_of)
        ok = not missing and any_ok
        reason = None if ok else f"missing {missing}" if missing else "none of any_of found"
        return _binary(ok, reason)


class Regex:
    """``pattern``/``any_of_field`` must match; ``none_of``/``none_of_field`` must not."""

    def __init__(
        self,
        name: str,
        pattern: str | None = None,
        any_of_field: str | None = None,
        none_of: list[str] | None = None,
        none_of_field: str | None = None,
        flags: str = "i",
    ) -> None:
        self.name = name
        self.patterns = [pattern] if pattern else []
        self.any_of_field = any_of_field
        self.none_of = none_of or []
        self.none_of_field = none_of_field
        self.flags = (re.IGNORECASE if "i" in flags else 0) | (re.MULTILINE if "m" in flags else 0)
        self.flags |= re.DOTALL if "s" in flags else 0

    def score(self, case: Case, out: SystemOutput) -> Score:
        accept = self.patterns + _case_list(case, self.any_of_field)
        reject = self.none_of + _case_list(case, self.none_of_field)
        if not accept:
            return Score(value=None, reason="no accept pattern")
        hit = next((p for p in accept if re.search(p, out.output, self.flags)), None)
        bad = next((p for p in reject if re.search(p, out.output, self.flags)), None)
        ok = hit is not None and bad is None
        reason = None if ok else f"rejected by {bad!r}" if bad else "no accept pattern matched"
        return _binary(ok, reason)


def _check(instance: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """A deliberately small JSON Schema subset: type, enum, required, properties, items."""
    errors: list[str] = []
    types = {
        "object": dict,
        "array": list,
        "string": str,
        "boolean": bool,
        "null": type(None),
    }
    expected = schema.get("type")
    if expected == "number":
        if not isinstance(instance, int | float) or isinstance(instance, bool):
            return [f"{path}: expected number"]
    elif expected == "integer":
        if not isinstance(instance, int) or isinstance(instance, bool):
            return [f"{path}: expected integer"]
    elif expected in types and not isinstance(instance, types[expected]):
        return [f"{path}: expected {expected}"]
    if "enum" in schema and instance not in schema["enum"]:
        errors.append(f"{path}: {instance!r} not in enum")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                errors.append(f"{path}: missing required {key!r}")
        for key, sub in schema.get("properties", {}).items():
            if key in instance:
                errors.extend(_check(instance[key], sub, f"{path}.{key}"))
    if isinstance(instance, list) and "items" in schema:
        for i, item in enumerate(instance):
            errors.extend(_check(item, schema["items"], f"{path}[{i}]"))
    return errors


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply that may wrap it in prose or a code fence."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found")


class JsonSchema:
    def __init__(
        self, name: str, schema: dict[str, Any] | None = None, schema_field: str | None = None
    ) -> None:
        self.name, self.schema, self.schema_field = name, schema, schema_field

    def score(self, case: Case, out: SystemOutput) -> Score:
        schema = self.schema or (case.field(self.schema_field) if self.schema_field else None)
        if not schema:
            return Score(value=None, reason="no schema")
        try:
            data = extract_json(out.output)
        except ValueError:
            return _binary(False, "output is not JSON")
        errors = _check(data, schema)
        return _binary(not errors, "; ".join(errors[:3]) or None)


class NumericTolerance:
    def __init__(
        self,
        name: str,
        field: str = "reference",
        abs_tol: float = 0.0,
        rel_tol: float = 0.0,
        pick: str = "first",
    ) -> None:
        if pick not in {"first", "last", "any"}:
            raise ValueError("pick must be first, last or any")
        self.name, self.field, self.abs_tol, self.rel_tol, self.pick = (
            name,
            field,
            abs_tol,
            rel_tol,
            pick,
        )

    def score(self, case: Case, out: SystemOutput) -> Score:
        raw = case.field(self.field)
        if raw is None:
            return Score(value=None, reason=f"case has no {self.field!r}")
        expected = float(str(raw).replace(",", ""))
        found = [float(n.replace(",", "")) for n in NUMBER.findall(out.output)]
        if not found:
            return _binary(False, "no number in output")
        candidates = {"first": found[:1], "last": found[-1:], "any": found}[self.pick]
        ok = any(
            math.isclose(x, expected, rel_tol=self.rel_tol, abs_tol=self.abs_tol)
            for x in candidates
        )
        return _binary(ok, None if ok else f"expected {expected}, found {candidates}")


def _number(value: Any) -> float | None:
    """A fact's value as a float: a number, or a numeric string such as ``"1,234"``."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError:
            return None
    else:
        return None
    return number if math.isfinite(number) else None


class Facts:
    """Numbers and phrases an answer must contain. ``field`` lists numbers (or ``{value, rel_tol?,
    abs_tol?, required?, label?}``), each of which must match some number in the output;
    ``mention_field`` lists strings that must appear. The value is the share found; the case
    passes when every required fact and every mention is found."""

    def __init__(
        self,
        name: str,
        field: str = "facts",
        mention_field: str = "must_mention",
        rel_tol: float = 0.0,
        abs_tol: float = 0.0,
        normalize: bool = True,
    ) -> None:
        if rel_tol < 0 or abs_tol < 0:
            raise ValueError("rel_tol and abs_tol must be >= 0")
        self.name, self.field, self.mention_field = name, field, mention_field
        self.rel_tol, self.abs_tol, self.norm = rel_tol, abs_tol, normalize

    def _facts(self, case: Case) -> list[tuple[float, float, float, bool, str]]:
        """The case's facts as ``(value, rel_tol, abs_tol, required, label)``."""
        raw = case.field(self.field)
        if raw is None:
            return []
        items = raw if isinstance(raw, list) else [raw]
        facts = []
        for item in items:
            spec = item if isinstance(item, dict) else {"value": item}
            unknown = [
                k for k in spec if k not in {"value", "rel_tol", "abs_tol", "required", "label"}
            ]
            if unknown:
                raise ValueError(f"case {case.id!r}: fact has unknown key {str(unknown[0])!r}")
            value = _number(spec.get("value"))
            if value is None:
                raise ValueError(
                    f"case {case.id!r}: fact value {spec.get('value')!r} is not a number"
                )
            tols = []
            for key, default in (("rel_tol", self.rel_tol), ("abs_tol", self.abs_tol)):
                tol = spec.get(key, default)
                if isinstance(tol, bool) or not isinstance(tol, int | float) or tol < 0:
                    raise ValueError(f"case {case.id!r}: fact {key} must be a number >= 0")
                tols.append(float(tol))
            required = spec.get("required", True)
            if not isinstance(required, bool):
                raise ValueError(f"case {case.id!r}: fact 'required' must be true or false")
            label = str(spec.get("label") or f"{value:g}")
            facts.append((value, tols[0], tols[1], required, label))
        return facts

    def score(self, case: Case, out: SystemOutput) -> Score:
        facts = self._facts(case)
        mentions = _case_list(case, self.mention_field)
        if not facts and not mentions:
            return Score(value=None, reason=f"case has no {self.field} or {self.mention_field}")
        found = [float(n.replace(",", "")) for n in NUMBER.findall(out.output)]
        missing: list[str] = []
        matched = 0
        required_ok = True
        for value, rel_tol, abs_tol, required, label in facts:
            if any(math.isclose(x, value, rel_tol=rel_tol, abs_tol=abs_tol) for x in found):
                matched += 1
            else:
                missing.append(label if required else f"{label} (optional)")
                required_ok = required_ok and not required
        text = normalize(out.output) if self.norm else out.output
        for mention in mentions:
            needle = normalize(mention) if self.norm else mention
            if needle in text:
                matched += 1
            else:
                missing.append(repr(mention))
                required_ok = False
        total = len(facts) + len(mentions)
        return Score(
            value=matched / total,
            passed=required_ok,
            reason=f"missing: {', '.join(missing)}" if missing else None,
        )
