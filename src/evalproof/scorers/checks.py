"""The ``checks`` scorer: assertions on what an agent did (its tool calls) and said.

A check is a small object with a ``type``. They come from the scorer's ``checks`` param (applied
to every case) and from each case's ``expect`` field (or the field named by ``field``). The
vocabulary is the ``_VOCAB`` table; ``validate_checks`` checks a list of them without running
anything, so a typo fails when the suite or the dataset loads, not halfway through a run.
"""

from __future__ import annotations

import difflib
import json
import math
import re
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal, InvalidOperation
from typing import Any

from evalproof.config import ConfigError
from evalproof.models import Case, Score, SystemOutput, ToolCall
from evalproof.scorers.python_scorer import FN_SPEC, resolve_fn

# type -> (required keys, optional keys); ``type`` itself is always allowed.
_VOCAB: dict[str, tuple[tuple[str, ...], tuple[str, ...]]] = {
    "tool_called": (("name",), ("args", "args_match", "min", "max", "turn")),
    "tool_not_called": (("name",), ("args", "args_match", "turn")),
    "tool_args_contain": (("name", "arg", "value"), ("turn",)),
    "forbidden_arg_values": (("arg", "values"), ("name", "turn")),
    "tool_order": (("calls",), ("match", "turn")),
    "tool_succeeded": ((), ("name", "turn")),
    "tool_call_count": ((), ("min", "max", "name", "turn")),
    "rounds": ((), ("min", "max")),
    "output_contains": (("text",), ("all", "case_sensitive", "turn")),
    "output_omits": (("text",), ("case_sensitive", "turn")),
    "output_regex": (("pattern",), ("flags", "turn")),
    "output_cites_tool_result": (
        ("name",),
        ("min_matches", "rel_tol", "strict", "turn"),
    ),
    "any_of": (("checks",), ()),
    "not": (("check",), ()),
    "python": (("fn",), ()),
}
CHECK_TYPES = tuple(_VOCAB)

# The checks that read ``trace``: without one they fail with "no trace".
TRACE_TYPES = frozenset(
    {
        "tool_called",
        "tool_not_called",
        "tool_args_contain",
        "forbidden_arg_values",
        "tool_order",
        "tool_succeeded",
        "tool_call_count",
        "rounds",
        "output_cites_tool_result",
    }
)

ON_MISSING = ("pass", "skip")
MODES = ("all", "fraction")
ORDER_MATCHES = ("subsequence", "exact", "any_order")
ARGS_MATCHES = ("exact", "contains")
REGEX_FLAGS = "ims"

# --- validation -------------------------------------------------------------------------------


def _is_num(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _int_from(low: int) -> Callable[[Any], str | None]:
    def rule(value: Any) -> str | None:
        ok = isinstance(value, int) and not isinstance(value, bool) and value >= low
        return None if ok else f"must be an integer >= {low}"

    return rule


def _text_rule(value: Any) -> str | None:
    one = isinstance(value, str) and bool(value)
    many = isinstance(value, list) and bool(value) and all(isinstance(v, str) and v for v in value)
    return None if one or many else "must be a non-empty string or a list of non-empty strings"


def _string_rule(value: Any) -> str | None:
    return None if isinstance(value, str) and value else "must be a non-empty string"


def _bool_rule(value: Any) -> str | None:
    return None if isinstance(value, bool) else "must be true or false"


def _enum_rule(options: tuple[str, ...]) -> Callable[[Any], str | None]:
    def rule(value: Any) -> str | None:
        return None if value in options else f"must be one of {', '.join(options)}"

    return rule


def _pattern_rule(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return "must be a non-empty string"
    try:
        re.compile(value)
    except re.error as exc:
        return f"is not a valid regular expression ({exc})"
    return None


def _flags_rule(value: Any) -> str | None:
    ok = isinstance(value, str) and all(ch in REGEX_FLAGS for ch in value)
    return None if ok else f"must be a string of the letters {', '.join(REGEX_FLAGS)}"


def _tol_rule(value: Any) -> str | None:
    ok = _is_num(value) and math.isfinite(value) and value >= 0
    return None if ok else "must be a number >= 0"


def _fn_rule(value: Any) -> str | None:
    ok = isinstance(value, str) and FN_SPEC.fullmatch(value) is not None
    return None if ok else f"must be 'module:function', got {value!r}"


def _object_rule(value: Any) -> str | None:
    return None if isinstance(value, dict) else "must be an object"


def _values_rule(value: Any) -> str | None:
    return None if isinstance(value, list) and value else "must be a non-empty list"


_RULES: dict[str, Callable[[Any], str | None]] = {
    "name": _string_rule,
    "arg": _string_rule,
    "args": _object_rule,
    "args_match": _enum_rule(ARGS_MATCHES),
    "min": _int_from(0),
    "max": _int_from(0),
    "turn": _int_from(0),
    "values": _values_rule,
    "text": _text_rule,
    "all": _bool_rule,
    "case_sensitive": _bool_rule,
    "strict": _bool_rule,
    "pattern": _pattern_rule,
    "flags": _flags_rule,
    "match": _enum_rule(ORDER_MATCHES),
    "min_matches": _int_from(1),
    "rel_tol": _tol_rule,
    "fn": _fn_rule,
}


def _hint(bad: str, valid: Iterable[str], label: str) -> str:
    options = list(valid)
    close = difflib.get_close_matches(bad, options, n=1)
    return f"did you mean {close[0]!r}?" if close else f"valid {label}: {', '.join(options)}"


def _validate_shape(item: Any, where: str) -> None:
    if not isinstance(item, dict):
        raise ValueError(f"{where} must be an object with a 'type'")
    if not isinstance(item.get("type"), str) or not item["type"]:
        raise ValueError(f"{where} needs a 'type' naming the check")


def validate_checks(items: Any, path: str) -> None:
    """Check a list of checks without running them. Raises ``ValueError`` naming the first
    problem by its path, e.g. ``expect[2].checks[0].min must be an integer >= 0``.

    Every item is checked for its shape (an object with a string ``type``) before any is checked
    for its vocabulary, so a malformed list reports its shape problem first."""
    if not isinstance(items, list):
        raise ValueError(f"{path} must be a list of checks")
    for i, item in enumerate(items):
        _validate_shape(item, f"{path}[{i}]")
    for i, item in enumerate(items):
        _validate_check(item, f"{path}[{i}]")


def _validate_check(item: dict[str, Any], where: str) -> None:
    kind = item["type"]
    if kind not in _VOCAB:
        raise ValueError(f"{where} has unknown type {kind!r} ({_hint(kind, _VOCAB, 'types')})")
    required, optional = _VOCAB[kind]
    valid = ["type", *required, *optional]
    for key in item:
        if key not in valid:
            raise ValueError(
                f"{where} has unknown key {str(key)!r} ({_hint(str(key), valid, 'keys')})"
            )
    for key in required:
        if key not in item:
            raise ValueError(f"{where} needs {key!r} ({kind})")
    if kind in ("tool_call_count", "rounds") and "min" not in item and "max" not in item:
        raise ValueError(f"{where} needs 'min' or 'max' ({kind})")
    for key, value in item.items():
        rule = _RULES.get(key)
        problem = rule(value) if rule else None
        if problem:
            raise ValueError(f"{where}.{key} {problem}")
    low, high = item.get("min"), item.get("max")
    if low is not None and high is not None and low > high:
        raise ValueError(f"{where} has min greater than max")
    if kind == "tool_order":
        _validate_calls(item["calls"], f"{where}.calls")
    elif kind == "any_of":
        children = item["checks"]
        if not isinstance(children, list) or not children:
            raise ValueError(f"{where}.checks must be a non-empty list of checks")
        validate_checks(children, f"{where}.checks")
    elif kind == "not":
        child = item["check"]
        _validate_shape(child, f"{where}.check")
        _validate_check(child, f"{where}.check")


def _validate_calls(calls: Any, where: str) -> None:
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{where} must be a non-empty list of tool names or objects")
    for i, entry in enumerate(calls):
        at = f"{where}[{i}]"
        if isinstance(entry, str):
            if not entry:
                raise ValueError(f"{at} must be a tool name or an object with a 'name'")
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"{at} must be a tool name or an object with a 'name'")
        valid = ["name", "args", "args_match"]
        for key in entry:
            if key not in valid:
                raise ValueError(
                    f"{at} has unknown key {str(key)!r} ({_hint(str(key), valid, 'keys')})"
                )
        if "name" not in entry:
            raise ValueError(f"{at} needs 'name'")
        for key, value in entry.items():
            problem = _RULES[key](value)
            if problem:
                raise ValueError(f"{at}.{key} {problem}")


def validate_checks_params(params: dict[str, Any], path: str = "params") -> None:
    """Check the ``checks`` scorer's params without running anything. Raises ``ValueError``."""
    known = ("checks", "field", "on_missing", "mode")
    for key in params:
        if key not in known:
            raise ValueError(
                f"{path} has unknown key {str(key)!r} ({_hint(str(key), list(known), 'keys')})"
            )
    if "checks" in params:
        validate_checks(params["checks"], f"{path}.checks")
    if "field" in params and (not isinstance(params["field"], str) or not params["field"]):
        raise ValueError(f"{path}.field must be the name of a case field")
    for key, options in (("on_missing", ON_MISSING), ("mode", MODES)):
        if key in params and params[key] not in options:
            raise ValueError(f"{path}.{key} must be one of {', '.join(options)}")


def _python_specs(checks: list[dict[str, Any]]) -> Iterator[str]:
    for check in checks:
        if check["type"] == "python":
            yield check["fn"]
        elif check["type"] == "any_of":
            yield from _python_specs(check["checks"])
        elif check["type"] == "not":
            yield from _python_specs([check["check"]])


# --- matching helpers -------------------------------------------------------------------------


def _j(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)


def _equal(a: Any, b: Any) -> bool:
    """Deep equality where numbers compare as floats (``1 == 1.0``) and a bool is not a number."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if _is_num(a) and _is_num(b):
        if isinstance(a, int) and isinstance(b, int):
            return a == b
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple) and isinstance(b, list | tuple):
        return len(a) == len(b) and all(_equal(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def _subset(expected: Any, actual: Any, contains: bool) -> bool:
    """``expected`` is a recursive subset of ``actual``: every expected key is present and
    matches, a list is equal (or, with ``contains``, has every expected element), a string is
    equal (or, with ``contains``, a substring)."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            k in actual and _subset(v, actual[k], contains) for k, v in expected.items()
        )
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return False
        if contains:
            return all(any(_subset(e, a, contains) for a in actual) for e in expected)
        return _equal(expected, actual)
    if contains and isinstance(expected, str):
        return isinstance(actual, str) and expected in actual
    return _equal(expected, actual)


def _holds(actual: Any, value: Any) -> bool:
    """``actual`` contains ``value``: a substring of a string, a member of a list, otherwise
    equal."""
    if isinstance(actual, str):
        return isinstance(value, str) and value in actual
    if isinstance(actual, list):
        return any(_equal(value, element) for element in actual)
    return _equal(value, actual)


def _range(low: int, high: int | None) -> str:
    if high is None:
        return f"at least {low}"
    if low == high:
        return f"exactly {low}"
    return f"at most {high}" if low == 0 else f"{low} to {high}"


def _within(n: int, low: int, high: int | None) -> bool:
    return n >= low and (high is None or n <= high)


# --- numbers in a reply and in tool results ---------------------------------------------------

# A number as a reply writes it: optional minus, thousands commas, decimals, trailing percent. It
# may not start inside a word or after a dot, so "Q3" and "v2.1" yield nothing and "3-5" is 3, 5.
_REPLY_NUMBER = re.compile(r"(?<![\w.])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_STRING_NUMBER = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_YEARS = range(1900, 2101)


@dataclass(frozen=True)
class _Number:
    value: float
    exact: Decimal
    places: int  # digits after the decimal point, as written

    @property
    def trivial(self) -> bool:
        """0, 1 and years show up in results as flags, counts and dates, not as findings."""
        return self.value in (0.0, 1.0) or (self.value.is_integer() and int(self.value) in _YEARS)


def _parse_number(token: str) -> _Number | None:
    text = token.strip().removesuffix("%").replace(",", "")
    try:
        exact = Decimal(text)
        value = float(exact)
    except (InvalidOperation, ValueError, OverflowError):
        return None
    if not math.isfinite(value):
        return None
    exponent = exact.as_tuple().exponent
    return _Number(value, exact, -exponent if isinstance(exponent, int) and exponent < 0 else 0)


def _reply_numbers(reply: str) -> list[_Number]:
    found = (_parse_number(m.group()) for m in _REPLY_NUMBER.finditer(reply))
    return [n for n in found if n is not None]


def _leaf_numbers(value: Any) -> Iterator[_Number]:
    """The numeric leaves of a tool result: numbers, and strings that are wholly a number."""
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        number = _parse_number(repr(value))
        if number is not None:
            yield number
    elif isinstance(value, str):
        if _STRING_NUMBER.fullmatch(value.strip()):
            number = _parse_number(value)
            if number is not None:
                yield number
    elif isinstance(value, dict):
        for item in value.values():
            yield from _leaf_numbers(item)
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _leaf_numbers(item)


def _cites(reply: _Number, leaf: _Number, rel_tol: float) -> bool:
    """The reply states the leaf: equal within ``rel_tol``, or the leaf rounded to the number of
    decimal places the reply wrote."""
    if math.isclose(reply.value, leaf.value, rel_tol=rel_tol, abs_tol=0.0):
        return True
    try:
        rounded = leaf.exact.quantize(Decimal(1).scaleb(-reply.places), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return False
    return rounded == reply.exact


# --- evaluation -------------------------------------------------------------------------------


class _Undecided(Exception):
    """A check could not be run (its function raised): the whole score is undecided."""


@dataclass(frozen=True)
class _Verdict:
    ok: bool
    why: str = ""
    # The check could not look at what it needs (no trace, no rounds, no reply for the turn). It
    # fails, and ``not`` keeps it failed: "not called" cannot be concluded from a missing trace.
    missing: bool = False


_PASS = _Verdict(True)


def _fail(why: str) -> _Verdict:
    return _Verdict(False, why)


def _missing(why: str) -> _Verdict:
    return _Verdict(False, why, missing=True)


def _label(check: dict[str, Any]) -> str:
    """How a failed check is named in a reason, e.g. ``tool_called query args {"id": 3}``."""
    kind = check["type"]
    parts = [kind]
    if kind in ("tool_called", "tool_not_called"):
        parts.append(check["name"])
        if check.get("args"):
            parts.append(f"args {_j(check['args'])}")
    elif kind == "tool_args_contain":
        parts.append(f"{check['name']} {check['arg']} contains {_j(check['value'])}")
    elif kind == "forbidden_arg_values":
        parts.append(f"{check['arg']} in {_j(check['values'])}")
        if check.get("name"):
            parts.append(f"for {check['name']}")
    elif kind == "tool_order":
        names = [e if isinstance(e, str) else e["name"] for e in check["calls"]]
        parts.append(f"{check.get('match', 'subsequence')} {_j(names)}")
    elif kind in ("tool_succeeded", "tool_call_count") and check.get("name"):
        parts.append(check["name"])
    elif kind in ("output_contains", "output_omits"):
        parts.append(_j(check["text"]))
    elif kind == "output_regex":
        parts.append(_j(check["pattern"]))
    elif kind == "output_cites_tool_result":
        parts.append(check["name"])
    elif kind == "not":
        parts.append(f"[{_label(check['check'])}]")
    elif kind == "python":
        parts.append(check["fn"])
    if "turn" in check:
        parts.append(f"(turn {check['turn']})")
    return " ".join(parts)


def _entry(spec: str | dict[str, Any]) -> tuple[str, dict[str, Any] | None, bool]:
    if isinstance(spec, str):
        return spec, None, False
    return spec["name"], spec.get("args"), spec.get("args_match") == "contains"


def _call_matches(spec: str | dict[str, Any], call: ToolCall) -> bool:
    name, args, contains = _entry(spec)
    return call.name == name and (not args or _subset(args, call.args, contains))


def _assignable(want: list[str | dict[str, Any]], calls: list[ToolCall]) -> bool:
    """Every wanted call can be given its own actual call (a bipartite matching, so a loose
    entry cannot use up the call a stricter one needs)."""
    owner: dict[int, int] = {}

    def place(w: int, seen: set[int]) -> bool:
        for ci, call in enumerate(calls):
            if ci in seen or not _call_matches(want[w], call):
                continue
            seen.add(ci)
            if ci not in owner or place(owner[ci], seen):
                owner[ci] = w
                return True
        return False

    return all(place(w, set()) for w in range(len(want)))


class _Evaluation:
    """One case's output, against which checks run."""

    def __init__(
        self, case: Case, out: SystemOutput, resolve: Callable[[str], Callable[..., Any]]
    ) -> None:
        self.case, self.out, self._resolve = case, out, resolve

    def run(self, check: dict[str, Any]) -> _Verdict:
        kind = check["type"]
        if kind in TRACE_TYPES and self.out.trace is None:
            return _missing("no trace")
        return getattr(self, f"_{kind}")(check)  # type: ignore[no-any-return]

    def _calls(self, check: dict[str, Any], name: str | None = None) -> list[ToolCall]:
        """The trace's tool calls, narrowed to a turn and a tool name when the check says so."""
        assert self.out.trace is not None
        turn = check.get("turn")
        return [
            c
            for c in self.out.trace.tool_calls
            if (turn is None or c.turn == turn) and (name is None or c.name == name)
        ]

    def _reply(self, check: dict[str, Any]) -> str | None:
        """The text a check reads: the final output, or the reply to turn ``turn``."""
        turn: int | None = check.get("turn")
        if turn is None:
            return self.out.output
        trace = self.out.trace
        if trace is None or turn >= len(trace.responses):
            return None
        return trace.responses[turn]

    # -- tool checks --

    def _tool_called(self, c: dict[str, Any]) -> _Verdict:
        named = self._calls(c, c["name"])
        want = c.get("args")
        contains = c.get("args_match") == "contains"
        hits = [x for x in named if not want or _subset(want, x.args, contains)]
        high = c.get("max")
        # At least once by default; ``max: 0`` alone means "never" rather than an empty range.
        low = c.get("min", 1 if high is None else min(1, high))
        if _within(len(hits), low, high):
            return _PASS
        if not named:
            return _fail("not called")
        if want and not hits:
            seen = "; ".join(_j(x.args) for x in named[:3])
            return _fail(f"called {len(named)} time(s), none with matching args (saw {seen})")
        return _fail(f"called {len(hits)} time(s), expected {_range(low, high)}")

    def _tool_not_called(self, c: dict[str, Any]) -> _Verdict:
        want = c.get("args")
        contains = c.get("args_match") == "contains"
        hits = [x for x in self._calls(c, c["name"]) if not want or _subset(want, x.args, contains)]
        return _fail(f"called {len(hits)} time(s)") if hits else _PASS

    def _tool_args_contain(self, c: dict[str, Any]) -> _Verdict:
        named = self._calls(c, c["name"])
        if any(c["arg"] in x.args and _holds(x.args[c["arg"]], c["value"]) for x in named):
            return _PASS
        if not named:
            return _fail("not called")
        seen = "; ".join(_j(x.args.get(c["arg"])) for x in named[:3])
        return _fail(f"no call has it (saw {seen})")

    def _forbidden_arg_values(self, c: dict[str, Any]) -> _Verdict:
        for call in self._calls(c, c.get("name")):
            if c["arg"] not in call.args:
                continue
            actual = call.args[c["arg"]]
            for value in c["values"]:
                if _equal(actual, value) or _holds(actual, value):
                    return _fail(f"{call.name} called with {c['arg']}={_j(actual)}")
        return _PASS

    def _tool_order(self, c: dict[str, Any]) -> _Verdict:
        want: list[str | dict[str, Any]] = c["calls"]
        calls = self._calls(c)
        match = c.get("match", "subsequence")
        if match == "exact":
            ok = len(want) == len(calls) and all(
                _call_matches(w, x) for w, x in zip(want, calls, strict=True)
            )
        elif match == "any_order":
            ok = len(want) <= len(calls) and _assignable(want, calls)
        else:
            i = 0
            for call in calls:
                if i < len(want) and _call_matches(want[i], call):
                    i += 1
            ok = i == len(want)
        return _PASS if ok else _fail(f"the calls were {_j([x.name for x in calls])}")

    def _tool_succeeded(self, c: dict[str, Any]) -> _Verdict:
        failed = [x for x in self._calls(c, c.get("name")) if not x.ok]
        if not failed:
            return _PASS
        shown = "; ".join(f"{x.name}: {x.error}" if x.error else x.name for x in failed[:3])
        return _fail(f"{len(failed)} call(s) failed ({shown})")

    def _tool_call_count(self, c: dict[str, Any]) -> _Verdict:
        n = len(self._calls(c, c.get("name")))
        low, high = c.get("min", 0), c.get("max")
        if _within(n, low, high):
            return _PASS
        return _fail(f"{n} call(s), expected {_range(low, high)}")

    def _rounds(self, c: dict[str, Any]) -> _Verdict:
        assert self.out.trace is not None
        rounds = self.out.trace.rounds
        if rounds is None:
            return _missing("the trace does not report rounds")
        low, high = c.get("min", 0), c.get("max")
        if _within(rounds, low, high):
            return _PASS
        return _fail(f"{rounds} round(s), expected {_range(low, high)}")

    # -- text checks --

    def _no_reply(self, c: dict[str, Any]) -> _Verdict:
        return _missing(f"no reply for turn {c['turn']}")

    def _output_contains(self, c: dict[str, Any]) -> _Verdict:
        reply = self._reply(c)
        if reply is None:
            return self._no_reply(c)
        needles = [c["text"]] if isinstance(c["text"], str) else c["text"]
        fold = (lambda s: s) if c.get("case_sensitive") else str.lower
        missing = [n for n in needles if fold(n) not in fold(reply)]
        if c.get("all", True):
            return _fail(f"missing {_j(missing)}") if missing else _PASS
        return _fail(f"none of {_j(needles)} found") if len(missing) == len(needles) else _PASS

    def _output_omits(self, c: dict[str, Any]) -> _Verdict:
        reply = self._reply(c)
        if reply is None:
            return self._no_reply(c)
        needles = [c["text"]] if isinstance(c["text"], str) else c["text"]
        fold = (lambda s: s) if c.get("case_sensitive") else str.lower
        found = [n for n in needles if fold(n) in fold(reply)]
        return _fail(f"found {_j(found)}") if found else _PASS

    def _output_regex(self, c: dict[str, Any]) -> _Verdict:
        reply = self._reply(c)
        if reply is None:
            return self._no_reply(c)
        flags = 0
        for letter, flag in (("i", re.IGNORECASE), ("m", re.MULTILINE), ("s", re.DOTALL)):
            if letter in c.get("flags", ""):
                flags |= flag
        return _PASS if re.search(c["pattern"], reply, flags) else _fail("no match")

    def _output_cites_tool_result(self, c: dict[str, Any]) -> _Verdict:
        reply = self._reply(c)
        if reply is None:
            return self._no_reply(c)
        named = self._calls(c, c["name"])
        if not named:
            return _fail("not called")
        strict = bool(c.get("strict", False))
        leaves: dict[Decimal, _Number] = {}
        for call in named:
            for leaf in _leaf_numbers(call.result):
                if strict or not leaf.trivial:
                    leaves.setdefault(leaf.exact.normalize(), leaf)
        if not leaves:
            return _fail("its results have no numbers to cite (0, 1 and years are ignored)")
        said = _reply_numbers(reply)
        rel_tol = float(c.get("rel_tol", 1e-6))
        cited = sum(any(_cites(n, leaf, rel_tol) for n in said) for leaf in leaves.values())
        need = c.get("min_matches", 1)
        if cited >= need:
            return _PASS
        return _fail(
            f"the reply cites {cited} of {len(leaves)} numbers in the results, need {need}"
        )

    # -- combinators and custom checks --

    def _any_of(self, c: dict[str, Any]) -> _Verdict:
        failures = []
        all_missing = True
        for child in c["checks"]:
            verdict = self.run(child)
            if verdict.ok:
                return _PASS
            all_missing = all_missing and verdict.missing
            failures.append(f"{_label(child)}: {verdict.why}")
        why = "none passed (" + "; ".join(failures) + ")"
        return _missing(why) if all_missing else _fail(why)

    def _not(self, c: dict[str, Any]) -> _Verdict:
        inner = self.run(c["check"])
        if inner.missing:  # nothing to negate: the inner check never looked
            return inner
        return _fail("the inner check passed") if inner.ok else _PASS

    def _python(self, c: dict[str, Any]) -> _Verdict:
        function = self._resolve(c["fn"])
        try:
            result = function(self.case, self.out)
        except Exception as exc:
            raise _Undecided(f"check error: {type(exc).__name__}: {exc}") from exc
        reason: str | None = None
        if isinstance(result, Score):
            if result.passed is None:
                raise _Undecided("check error: the python check returned an undecided Score")
            ok, reason = result.passed, result.reason
        elif isinstance(result, tuple):
            if len(result) != 2:
                raise _Undecided(
                    f"check error: the python check returned a tuple of {len(result)} items; "
                    "expected (bool, reason)"
                )
            ok, reason = bool(result[0]), None if result[1] is None else str(result[1])
        else:
            ok = bool(result)
        return _PASS if ok else _fail(reason or "returned a falsy value")


class ChecksScorer:
    """Scores a case on how many of its checks pass.

    ``checks`` apply to every case; the case's own checks come from its ``field`` (``expect``).
    ``mode: all`` scores 1.0 or 0.0; ``fraction`` scores the share of checks that pass, and
    ``passed`` still means all of them did. A case with no checks at all is scored by
    ``on_missing``: ``pass`` passes it, ``skip`` leaves it undecided. A case whose system errored
    fails every check. A ``python`` check that raises leaves the whole score undecided.
    """

    def __init__(
        self,
        name: str,
        checks: list[dict[str, Any]] | None = None,
        field: str = "expect",
        on_missing: str = "pass",
        mode: str = "all",
        *,
        search_path: str | None = None,
    ) -> None:
        validate_checks_params(
            {"checks": checks or [], "field": field, "on_missing": on_missing, "mode": mode}
        )
        self.name, self.field = name, field
        self.on_missing, self.mode = on_missing, mode
        self.checks = list(checks or [])
        self._search_path = search_path
        self._functions: dict[str, Callable[..., Any]] = {}
        for spec in _python_specs(self.checks):  # a bad function fails when the scorer is built
            self._function(spec)

    def _function(self, spec: str) -> Callable[..., Any]:
        if spec not in self._functions:
            self._functions[spec] = resolve_fn(
                spec, self._search_path, f"scorer {self.name!r} python check"
            )
        return self._functions[spec]

    def _checks_for(self, case: Case) -> list[dict[str, Any]]:
        own = case.field(self.field)
        if own is None:
            return list(self.checks)
        validate_checks(own, self.field)
        return [*self.checks, *own]

    def score(self, case: Case, out: SystemOutput) -> Score:
        try:
            checks = self._checks_for(case)
        except ValueError as exc:
            return Score(value=None, passed=None, reason=f"invalid checks: {exc}")
        if not checks:
            if self.on_missing == "skip":
                return Score(value=None, passed=None, reason="no checks")
            return Score(value=1.0, passed=True, reason="no checks")
        if out.error:
            return Score(value=0.0, passed=False, reason="system error")
        evaluation = _Evaluation(case, out, self._resolve)
        failed: list[str] = []
        try:
            for check in checks:
                verdict = evaluation.run(check)
                if not verdict.ok:
                    failed.append(f"{_label(check)}: {verdict.why}")
        except _Undecided as exc:
            return Score(value=None, passed=None, reason=str(exc))
        if self.mode == "fraction":
            value = (len(checks) - len(failed)) / len(checks)
        else:
            value = 0.0 if failed else 1.0
        return Score(value=value, passed=not failed, reason="; ".join(failed) or None)

    def _resolve(self, spec: str) -> Callable[..., Any]:
        try:
            return self._function(spec)
        except ConfigError as exc:  # a case's own python check names a function that is not there
            raise _Undecided(f"check error: ConfigError: {exc}") from exc
