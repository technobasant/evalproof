"""The ``checks`` scorer: assertions on an agent's tool calls and replies, and the validation of
the check vocabulary at suite load and at dataset load."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.config import ConfigError, ScorerConfig, SuiteConfig, TargetConfig, load_config
from evalproof.dataset import DatasetError, load_dataset
from evalproof.models import Case, Score, SystemOutput, ToolCall, Trace
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import ChecksScorer, make_scorer
from evalproof.verify import verify_run


def call(name: str, args: dict[str, Any] | None = None, **kw: Any) -> ToolCall:
    return ToolCall(name=name, args=args or {}, **kw)


SEARCH = call(
    "search",
    {"q": "weather in Paris", "filters": {"lang": "en", "tags": ["a", "b"]}, "limit": 5},
    result={"hits": 3},
)
BOOK_PARIS = call("book", {"city": "Paris", "nights": 2}, result={"price": 1234.56})
BOOK_ROME = call("book", {"city": "Rome", "nights": 3.0}, ok=False, error="sold out", turn=1)
CALLS = [SEARCH, BOOK_PARIS, BOOK_ROME]
REPLIES = ["It is 20 degrees in Paris.", "Rome is sold out."]


def make_out(
    calls: list[ToolCall] | None = None,
    replies: list[str] | None = None,
    *,
    rounds: int | None = None,
    trace: bool = True,
    error: str | None = None,
) -> SystemOutput:
    replies = REPLIES if replies is None else replies
    return SystemOutput(
        output=replies[-1] if replies else "",
        error=error,
        trace=Trace(responses=replies, tool_calls=CALLS if calls is None else calls, rounds=rounds)
        if trace
        else None,
    )


CASE = Case(id="c1", input="q")


def run(checks: list[dict[str, Any]], out: SystemOutput | None = None, **params: Any) -> Score:
    return ChecksScorer("c", checks=checks, **params).score(CASE, out or make_out())


def passes(check: dict[str, Any], out: SystemOutput | None = None) -> bool:
    score = run([check], out)
    assert score.passed is not None, score.reason
    return score.passed


# --- tool_called and argument matching --------------------------------------------------------


def test_tool_called_by_name_and_count() -> None:
    assert passes({"type": "tool_called", "name": "search"})
    assert passes({"type": "tool_called", "name": "book", "min": 2})
    assert passes({"type": "tool_called", "name": "book", "min": 2, "max": 2})
    assert not passes({"type": "tool_called", "name": "book", "min": 3})
    assert not passes({"type": "tool_called", "name": "book", "max": 1})
    assert passes({"type": "tool_called", "name": "cancel", "min": 0, "max": 0})


def test_tool_called_with_only_max_zero_means_never() -> None:
    assert passes({"type": "tool_called", "name": "cancel", "max": 0})
    assert not passes({"type": "tool_called", "name": "search", "max": 0})


def test_not_does_not_turn_a_missing_trace_or_reply_into_a_pass() -> None:
    no_trace = make_out(trace=False)
    for inner in (
        {"type": "tool_called", "name": "delete"},
        {"type": "rounds", "max": 3},
        {"type": "output_contains", "text": "x", "turn": 5},
        {"type": "any_of", "checks": [{"type": "tool_called", "name": "delete"}]},
    ):
        score = run([{"type": "not", "check": inner}], no_trace)
        assert score.passed is False, inner
    # A rounds-less trace is missing evidence too; a real failure inside `not` still passes.
    assert not passes({"type": "not", "check": {"type": "rounds", "max": 3}}, make_out())
    assert passes({"type": "not", "check": {"type": "tool_called", "name": "delete"}})


def test_a_failed_tool_called_names_what_it_wanted() -> None:
    score = run([{"type": "tool_called", "name": "cancel", "args": {"id": 3}}])
    assert score.reason == 'tool_called cancel args {"id": 3}: not called'
    score = run([{"type": "tool_called", "name": "book", "max": 1}])
    assert score.reason == "tool_called book: called 2 time(s), expected exactly 1"  # min is 1
    score = run([{"type": "tool_called", "name": "book", "min": 0, "max": 1}])
    assert score.reason == "tool_called book: called 2 time(s), expected at most 1"
    score = run([{"type": "tool_called", "name": "book", "min": 5}])
    assert score.reason == "tool_called book: called 2 time(s), expected at least 5"
    score = run([{"type": "tool_called", "name": "book", "args": {"city": "Oslo"}}])
    assert score.reason is not None
    assert "none with matching args" in score.reason and '"city": "Paris"' in score.reason


@pytest.mark.parametrize(
    ("args", "extra", "expected"),
    [
        ({"limit": 5}, {}, True),
        ({"limit": 5.0}, {}, True),  # numbers compare as floats
        ({"limit": 6}, {}, False),
        ({"limit": "5"}, {}, False),
        ({"limit": True}, {}, False),  # a bool is not a number
        ({"q": "weather in Paris"}, {}, True),
        ({"q": "Paris"}, {}, False),  # strings are exact...
        ({"q": "Paris"}, {"args_match": "contains"}, True),  # ...unless contains
        ({"q": "Oslo"}, {"args_match": "contains"}, False),
        ({"filters": {"lang": "en"}}, {}, True),  # nested dicts are subsets
        ({"filters": {"lang": "fr"}}, {}, False),
        ({"filters": {"lang": "en", "tags": ["a", "b"]}}, {}, True),
        ({"filters": {"tags": ["a"]}}, {}, False),  # lists must be equal...
        ({"filters": {"tags": ["b", "a"]}}, {}, False),
        ({"filters": {"tags": ["a"]}}, {"args_match": "contains"}, True),  # ...unless contains
        ({"filters": {"tags": ["a", "c"]}}, {"args_match": "contains"}, False),
        ({"filters": {"missing": 1}}, {}, False),
        ({"absent": 1}, {}, False),
        ({"filters": "en"}, {}, False),
        ({}, {}, True),
    ],
)
def test_args_match_by_recursive_subset(
    args: dict[str, Any], extra: dict[str, Any], expected: bool
) -> None:
    assert passes({"type": "tool_called", "name": "search", "args": args, **extra}) is expected


def test_args_numbers_are_floats_and_a_float_arg_matches_an_int() -> None:
    assert passes({"type": "tool_called", "name": "book", "args": {"nights": 3}, "turn": 1})
    assert passes({"type": "tool_called", "name": "book", "args": {"nights": 2.0}, "turn": 0})


def test_contains_matching_recurses_into_lists_of_objects() -> None:
    out = make_out([call("pick", {"items": [{"id": 1, "tag": "red"}, {"id": 2, "tag": "blue"}]})])
    wanted = {"items": [{"tag": "blue"}]}
    assert passes(
        {"type": "tool_called", "name": "pick", "args": wanted, "args_match": "contains"}, out
    )
    assert not passes({"type": "tool_called", "name": "pick", "args": wanted}, out)


# --- turn -------------------------------------------------------------------------------------


def test_turn_keeps_only_the_calls_of_that_turn() -> None:
    assert passes({"type": "tool_called", "name": "book", "turn": 0, "min": 1, "max": 1})
    assert passes({"type": "tool_called", "name": "book", "turn": 1, "args": {"city": "Rome"}})
    assert not passes({"type": "tool_called", "name": "book", "turn": 1, "args": {"city": "Paris"}})
    assert not passes({"type": "tool_called", "name": "search", "turn": 1})
    assert passes({"type": "tool_not_called", "name": "search", "turn": 1})
    assert passes({"type": "tool_call_count", "turn": 1, "min": 1, "max": 1})
    assert passes({"type": "tool_succeeded", "turn": 0})
    assert not passes({"type": "tool_succeeded", "turn": 1})
    assert "(turn 1)" in (run([{"type": "tool_succeeded", "turn": 1}]).reason or "")


def test_turn_on_a_text_check_reads_that_reply() -> None:
    assert passes({"type": "output_contains", "text": "degrees", "turn": 0})
    assert not passes({"type": "output_contains", "text": "degrees"})  # the final reply
    assert passes({"type": "output_contains", "text": "sold out", "turn": 1})
    assert passes({"type": "output_omits", "text": "sold out", "turn": 0})
    assert passes({"type": "output_regex", "pattern": r"\d+ degrees", "turn": 0})


@pytest.mark.parametrize("trace", [True, False])
def test_a_turn_with_no_reply_fails_with_that_reason(trace: bool) -> None:
    for check in (
        {"type": "output_contains", "text": "x", "turn": 5},
        {"type": "output_omits", "text": "x", "turn": 5},
        {"type": "output_regex", "pattern": "x", "turn": 5},
    ):
        score = run([check], make_out(trace=trace))
        assert score.passed is False and score.reason is not None
        assert score.reason.endswith("no reply for turn 5")


# --- the other tool checks --------------------------------------------------------------------


def test_tool_not_called() -> None:
    assert passes({"type": "tool_not_called", "name": "cancel"})
    assert not passes({"type": "tool_not_called", "name": "search"})
    assert passes({"type": "tool_not_called", "name": "book", "args": {"city": "Oslo"}})
    assert not passes({"type": "tool_not_called", "name": "book", "args": {"city": "Paris"}})
    assert run([{"type": "tool_not_called", "name": "search"}]).reason == (
        "tool_not_called search: called 1 time(s)"
    )


def test_tool_args_contain() -> None:
    assert passes({"type": "tool_args_contain", "name": "search", "arg": "q", "value": "Paris"})
    assert not passes({"type": "tool_args_contain", "name": "search", "arg": "q", "value": "Oslo"})
    list_arg = {"type": "tool_args_contain", "name": "search", "arg": "filters"}
    assert not passes({**list_arg, "value": "en"})  # a dict arg is compared for equality
    out = make_out([call("tag", {"tags": ["a", "b"], "n": 2, "flag": True})])
    assert passes({"type": "tool_args_contain", "name": "tag", "arg": "tags", "value": "b"}, out)
    assert not passes(
        {"type": "tool_args_contain", "name": "tag", "arg": "tags", "value": "c"}, out
    )
    assert passes({"type": "tool_args_contain", "name": "tag", "arg": "n", "value": 2.0}, out)
    assert passes({"type": "tool_args_contain", "name": "tag", "arg": "flag", "value": True}, out)
    assert not passes({"type": "tool_args_contain", "name": "tag", "arg": "flag", "value": 1}, out)
    assert not passes({"type": "tool_args_contain", "name": "tag", "arg": "gone", "value": 1}, out)
    missing = run([{"type": "tool_args_contain", "name": "nope", "arg": "q", "value": "x"}])
    assert missing.reason is not None and missing.reason.endswith("not called")


def test_forbidden_arg_values() -> None:
    out = make_out(
        [
            call("sql", {"query": "DROP TABLE users", "mode": "ro"}),
            call("tag", {"tags": ["x", "y"]}),
        ]
    )
    assert not passes({"type": "forbidden_arg_values", "arg": "query", "values": ["DROP"]}, out)
    assert passes({"type": "forbidden_arg_values", "arg": "query", "values": ["SELECT"]}, out)
    assert not passes({"type": "forbidden_arg_values", "arg": "mode", "values": ["rw", "ro"]}, out)
    assert not passes(
        {"type": "forbidden_arg_values", "arg": "tags", "values": ["y"]}, out
    )  # member
    assert not passes({"type": "forbidden_arg_values", "arg": "tags", "values": [["x", "y"]]}, out)
    # scoped to a tool: the DROP is in `sql`, not in `tag`
    assert passes(
        {"type": "forbidden_arg_values", "arg": "query", "values": ["DROP"], "name": "tag"}, out
    )
    assert passes({"type": "forbidden_arg_values", "arg": "other", "values": ["x"]}, out)
    reason = run([{"type": "forbidden_arg_values", "arg": "query", "values": ["DROP"]}], out).reason
    assert reason == (
        'forbidden_arg_values query in ["DROP"]: sql called with query="DROP TABLE users"'
    )


@pytest.mark.parametrize(
    ("match", "calls", "expected"),
    [
        ("subsequence", ["search", "book"], True),
        ("subsequence", ["search", "book", "book"], True),
        ("subsequence", ["book", "search"], False),  # wrong order
        ("subsequence", ["search", "search"], False),
        ("exact", ["search", "book", "book"], True),
        ("exact", ["search", "book"], False),  # others present
        ("exact", ["book", "search", "book"], False),
        ("any_order", ["book", "search"], True),
        ("any_order", ["book", "book", "search"], True),
        ("any_order", ["search", "search"], False),  # a multiset: search was called once
        ("any_order", ["cancel"], False),
    ],
)
def test_tool_order(match: str, calls: list[str], expected: bool) -> None:
    assert passes({"type": "tool_order", "calls": calls, "match": match}) is expected


def test_tool_order_default_is_a_subsequence_and_entries_can_carry_args() -> None:
    assert passes({"type": "tool_order", "calls": ["search", "book"]})
    with_args = [
        {"name": "search", "args": {"limit": 5}},
        {"name": "book", "args": {"city": "Rome"}},
    ]
    assert passes({"type": "tool_order", "calls": with_args})
    assert not passes({"type": "tool_order", "calls": [{"name": "book", "args": {"city": "Oslo"}}]})
    contains = [{"name": "search", "args": {"q": "Paris"}, "args_match": "contains"}]
    assert passes({"type": "tool_order", "calls": contains})
    assert not passes({"type": "tool_order", "calls": [{"name": "search", "args": {"q": "Paris"}}]})
    # a loose entry cannot use up the call a stricter one needs
    both = [{"name": "book"}, {"name": "book", "args": {"city": "Paris"}}]
    assert passes({"type": "tool_order", "calls": both, "match": "any_order"})
    assert run([{"type": "tool_order", "calls": ["book", "search"]}]).reason == (
        'tool_order subsequence ["book", "search"]: the calls were ["search", "book", "book"]'
    )


def test_tool_order_with_a_turn_looks_at_that_turn() -> None:
    assert passes({"type": "tool_order", "calls": ["book"], "match": "exact", "turn": 1})
    assert not passes({"type": "tool_order", "calls": ["search"], "match": "exact", "turn": 1})


def test_tool_succeeded() -> None:
    assert not passes({"type": "tool_succeeded"})
    assert run([{"type": "tool_succeeded"}]).reason == (
        "tool_succeeded: 1 call(s) failed (book: sold out)"
    )
    assert passes({"type": "tool_succeeded", "name": "search"})
    assert not passes({"type": "tool_succeeded", "name": "book"})
    assert passes({"type": "tool_succeeded", "name": "nothing"})  # no calls, none failed
    assert passes({"type": "tool_succeeded"}, make_out([SEARCH]))


def test_tool_call_count() -> None:
    assert passes({"type": "tool_call_count", "min": 3, "max": 3})
    assert not passes({"type": "tool_call_count", "max": 2})
    assert not passes({"type": "tool_call_count", "min": 4})
    assert passes({"type": "tool_call_count", "name": "book", "min": 2, "max": 2})
    assert not passes({"type": "tool_call_count", "name": "book", "max": 1})
    assert run([{"type": "tool_call_count", "max": 2}]).reason == (
        "tool_call_count: 3 call(s), expected at most 2"
    )


def test_rounds() -> None:
    out = make_out(rounds=3)
    assert passes({"type": "rounds", "min": 1, "max": 3}, out)
    assert not passes({"type": "rounds", "max": 2}, out)
    assert (
        run([{"type": "rounds", "max": 2}], out).reason == "rounds: 3 round(s), expected at most 2"
    )
    unknown = run([{"type": "rounds", "max": 2}], make_out())
    assert unknown.passed is False
    assert unknown.reason == "rounds: the trace does not report rounds"


# --- text checks ------------------------------------------------------------------------------


def test_output_contains() -> None:
    assert passes({"type": "output_contains", "text": "SOLD OUT"})  # case-insensitive by default
    assert not passes({"type": "output_contains", "text": "SOLD OUT", "case_sensitive": True})
    assert passes({"type": "output_contains", "text": ["rome", "sold"]})
    assert not passes({"type": "output_contains", "text": ["rome", "paris"]})
    assert passes({"type": "output_contains", "text": ["rome", "paris"], "all": False})
    assert not passes({"type": "output_contains", "text": ["oslo", "paris"], "all": False})
    assert run([{"type": "output_contains", "text": ["rome", "paris"]}]).reason == (
        'output_contains ["rome", "paris"]: missing ["paris"]'
    )
    assert run([{"type": "output_contains", "text": ["a", "b"], "all": False}]).reason == (
        'output_contains ["a", "b"]: none of ["a", "b"] found'
    )


def test_output_omits() -> None:
    assert passes({"type": "output_omits", "text": "error"})
    assert not passes({"type": "output_omits", "text": "SOLD"})
    assert passes({"type": "output_omits", "text": "SOLD", "case_sensitive": True})
    assert not passes({"type": "output_omits", "text": ["error", "sold"]})
    assert run([{"type": "output_omits", "text": ["error", "sold"]}]).reason == (
        'output_omits ["error", "sold"]: found ["sold"]'
    )


def test_output_regex() -> None:
    assert passes({"type": "output_regex", "pattern": r"Rome is \w+ out"})
    assert not passes({"type": "output_regex", "pattern": "rome"})  # case-sensitive by default
    assert passes({"type": "output_regex", "pattern": "rome", "flags": "i"})
    assert passes({"type": "output_regex", "pattern": "^Rome", "flags": "m"})
    two_lines = make_out(replies=["first\nsecond"])
    assert not passes({"type": "output_regex", "pattern": "first.second"}, two_lines)
    assert passes({"type": "output_regex", "pattern": "first.second", "flags": "s"}, two_lines)
    assert (
        run([{"type": "output_regex", "pattern": "xyz"}]).reason == 'output_regex "xyz": no match'
    )


# --- output_cites_tool_result -----------------------------------------------------------------


def cites(reply: str, result: Any, *, name: str = "stats", turn: int = 0, **options: Any) -> Score:
    out = make_out([call(name, {}, result=result, turn=turn)], [reply])
    return run([{"type": "output_cites_tool_result", "name": "stats", **options}], out)


@pytest.mark.parametrize(
    ("reply", "result", "options", "expected"),
    [
        ("Revenue was 1,234.56 last week.", {"revenue": 1234.56}, {}, True),
        ("Revenue was 1234.56.", {"revenue": 1234.56}, {}, True),
        ("Revenue was 1,235 last week.", {"revenue": 1234.56}, {}, True),  # the leaf, rounded
        ("Revenue was 1,234.6.", {"revenue": 1234.56}, {}, True),
        ("Revenue was 1,234.5.", {"revenue": 1234.56}, {}, False),  # 1234.56 rounds to .6
        ("Revenue was about 1.2k.", {"revenue": 1234.56}, {}, False),
        ("Churn is 12.5%.", {"churn": 12.5}, {}, True),  # a trailing percent
        ("Churn is 12.5%.", {"churn": "12.5%"}, {}, True),  # a string that is a number
        ("Churn is 13%.", {"churn": 12.5}, {}, True),  # 12.5 rounds half up to 13
        ("Churn is 12%.", {"churn": 12.5}, {}, False),
        ("Delta is -4.5 points.", {"delta": -4.5}, {}, True),  # a minus sign
        ("Delta is 4.5 points.", {"delta": -4.5}, {}, False),  # no sign, no match
        ("Total 1,000,000.", {"total": 1000000}, {}, True),
        ("Total 999.", {"total": 1000}, {}, False),
        ("They sold 42 units.", {"rows": [{"units": 42}, {"units": 7}]}, {}, True),  # walks lists
        ("They sold 7 units.", {"a": {"b": [{"c": "7"}]}}, {}, True),  # nested, a string leaf
        ("They sold 8 units.", {"a": {"b": [{"c": "7"}]}}, {}, False),
        ("Value 0.30 here.", {"v": 0.1 + 0.2}, {}, True),  # float noise is no problem
        ("Value 100 here.", {"v": 100.00000001}, {"rel_tol": 1e-6}, True),
        ("Value 101 here.", {"v": 100}, {"rel_tol": 0.02}, True),  # within rel_tol
        ("Value 103 here.", {"v": 100}, {"rel_tol": 0.02}, False),
        ("They sold 42.", {"units": 42, "other": 17}, {"min_matches": 2}, False),
        ("They sold 42 and 17.", {"units": 42, "other": 17}, {"min_matches": 2}, True),
        ("They sold 42 and 42.", {"units": 42, "dup": 42}, {"min_matches": 2}, False),  # distinct
        ("Region Q3 and v2.1", {"n": 3}, {}, False),  # numbers inside words do not count
        ("Between 3-5 days.", {"n": 5}, {}, True),  # a range is two numbers, not 3 and -5
    ],
)
def test_output_cites_tool_result(
    reply: str, result: Any, options: dict[str, Any], expected: bool
) -> None:
    assert cites(reply, result, **options).passed is expected


def test_cites_ignores_flags_counts_and_years_unless_strict() -> None:
    result = {"ok": 1, "none": 0, "year": 2024, "count": 1900, "edge": 2100, "big": 2101}
    assert cites("It is 1 and 0 in 2024, 1900, 2100.", result).passed is False
    assert cites("It is 1 and 0 in 2024, 1900, 2100.", result).reason == (
        "output_cites_tool_result stats: the reply cites 0 of 1 numbers in the results, need 1"
    )  # only 2101 is left to cite
    assert cites("It is 2101.", result).passed is True
    assert cites("It is 1.", result, strict=True).passed is True
    assert cites("It is 0.", result, strict=True).passed is True
    assert cites("It is 2024.", result, strict=True).passed is True
    assert cites("It is about 2025.", {"year": 2024.5}).passed is True  # not an integer year
    assert cites("It is 2024.", {"year": 2024.5}).passed is False  # 2024.5 rounds to 2025
    assert cites("Nothing.", {"only": 1}).reason is not None
    assert "no numbers to cite" in (cites("It is 1.", {"only": 1}).reason or "")
    assert "no numbers to cite" in (cites("x", {"text": "none", "flag": True}).reason or "")


def test_cites_uses_only_the_named_tool_and_the_turn() -> None:
    out = make_out(
        [call("stats", result={"n": 55}), call("other", result={"n": 99}, turn=1)],
        ["Got 55.", "Got 99."],
    )
    base = {"type": "output_cites_tool_result", "name": "stats"}
    assert passes(base, make_out(out.trace.tool_calls, ["Got 55."]))  # type: ignore[union-attr]
    assert not passes({**base, "name": "other"}, make_out(out.trace.tool_calls, ["Got 55."]))  # type: ignore[union-attr]
    missing = run([{**base, "name": "nope"}], out)
    assert missing.reason == "output_cites_tool_result nope: not called"
    # turn: the reply to turn 0 is checked against turn 0's calls
    assert passes({**base, "turn": 0}, out)
    assert not passes(
        {**base, "name": "other", "turn": 1}, make_out(out.trace.tool_calls, ["Got 55.", "Got 7."])
    )  # type: ignore[union-attr]
    assert passes({**base, "name": "other", "turn": 1}, out)


# --- any_of, not, python ----------------------------------------------------------------------


def test_any_of_and_not_nest() -> None:
    sold = {"type": "output_contains", "text": "sold out"}
    nope = {"type": "tool_called", "name": "cancel"}
    assert passes({"type": "any_of", "checks": [nope, sold]})
    assert not passes(
        {"type": "any_of", "checks": [nope, {"type": "output_contains", "text": "zz"}]}
    )
    assert passes({"type": "not", "check": nope})
    assert not passes({"type": "not", "check": sold})
    assert passes({"type": "not", "check": {"type": "not", "check": sold}})
    deep = {
        "type": "any_of",
        "checks": [
            nope,
            {
                "type": "not",
                "check": {
                    "type": "any_of",
                    "checks": [nope, {"type": "tool_called", "name": "zz"}],
                },
            },
        ],
    }
    assert passes(deep)
    assert run([{"type": "not", "check": sold}]).reason == (
        'not [output_contains "sold out"]: the inner check passed'
    )
    reason = run(
        [{"type": "any_of", "checks": [nope, {"type": "tool_called", "name": "zz"}]}]
    ).reason
    assert (
        reason == "any_of: none passed (tool_called cancel: not called; tool_called zz: not called)"
    )


def test_a_not_over_a_missing_trace_still_fails() -> None:
    # "no trace" is a failure of the inner check, so `not` would pass it; the trace-based check
    # types are not negated into a pass without evidence.
    score = run([{"type": "tool_not_called", "name": "x"}], make_out(trace=False))
    assert score.passed is False and score.reason == "tool_not_called x: no trace"


def test_python_check() -> None:
    assert passes({"type": "python", "fn": "tests.scorer_fns:long_enough"})
    assert not passes({"type": "python", "fn": "tests.scorer_fns:short"})
    reason = run([{"type": "python", "fn": "tests.scorer_fns:short"}]).reason
    assert reason == "python tests.scorer_fns:short: returned a falsy value"
    bad = run([{"type": "python", "fn": "tests.scorer_fns:with_reason"}])
    assert bad.reason == "python tests.scorer_fns:with_reason: the reply never says ok"
    assert passes(
        {"type": "python", "fn": "tests.scorer_fns:with_reason"}, make_out(replies=["ok"])
    )
    assert passes({"type": "not", "check": {"type": "python", "fn": "tests.scorer_fns:short"}})
    sc = run([{"type": "python", "fn": "tests.scorer_fns:returns_score"}])
    assert sc.passed is False and "a hand-built score" in (sc.reason or "")


def test_a_python_check_that_raises_leaves_the_whole_score_undecided() -> None:
    score = run(
        [
            {"type": "tool_called", "name": "cancel"},  # fails
            {"type": "python", "fn": "tests.scorer_fns:raises"},
        ]
    )
    assert score.value is None and score.passed is None
    assert score.reason == "check error: ValueError: boom"
    odd = run([{"type": "python", "fn": "tests.scorer_fns:three_items"}])
    assert odd.value is None and (odd.reason or "").startswith("check error: the python check")
    undecided = run([{"type": "python", "fn": "tests.scorer_fns:undecided_score"}])
    assert undecided.value is None and "undecided Score" in (undecided.reason or "")


def test_a_python_check_that_cannot_be_imported_fails_the_scorer_build() -> None:
    with pytest.raises(ConfigError, match=r"cannot import 'tests\.nowhere'"):
        ChecksScorer("c", checks=[{"type": "python", "fn": "tests.nowhere:fn"}])
    nested = {
        "type": "not",
        "check": {"type": "any_of", "checks": [{"type": "python", "fn": "tests.scorer_fns:gone"}]},
    }
    with pytest.raises(ConfigError, match=r"tests\.scorer_fns\.gone is not callable"):
        ChecksScorer("c", checks=[nested])


def test_a_case_level_python_check_that_cannot_be_imported_is_undecided() -> None:
    case = Case.model_validate(
        {"id": "c", "input": "q", "expect": [{"type": "python", "fn": "tests.nowhere:fn"}]}
    )
    score = ChecksScorer("c").score(case, make_out())
    assert score.value is None and score.passed is None
    assert (score.reason or "").startswith("check error: ConfigError: scorer 'c' python check")


# --- the score --------------------------------------------------------------------------------


def test_all_mode_is_one_or_zero_and_the_reason_joins_the_failures() -> None:
    good = {"type": "tool_called", "name": "search"}
    bad1 = {"type": "tool_called", "name": "cancel"}
    bad2 = {"type": "output_contains", "text": "zzz"}
    ok = run([good, good])
    assert (ok.value, ok.passed, ok.reason) == (1.0, True, None)
    score = run([good, bad1, bad2])
    assert (score.value, score.passed) == (0.0, False)
    assert score.reason == 'tool_called cancel: not called; output_contains "zzz": missing ["zzz"]'


def test_fraction_mode_scores_the_share_and_still_fails_unless_all_pass() -> None:
    good = {"type": "tool_called", "name": "search"}
    bad = {"type": "tool_called", "name": "cancel"}
    score = run([good, good, bad, good], mode="fraction")
    assert (score.value, score.passed) == (0.75, False)
    assert run([good, good], mode="fraction").passed is True
    assert run([bad], mode="fraction").value == 0.0


def test_a_system_error_fails_every_check() -> None:
    score = run([{"type": "output_omits", "text": "x"}], make_out(error="boom"))
    assert (score.value, score.passed, score.reason) == (0.0, False, "system error")
    fraction = run([{"type": "output_omits", "text": "x"}], make_out(error="boom"), mode="fraction")
    assert fraction.value == 0.0 and fraction.passed is False


def test_no_trace_fails_trace_checks_but_text_checks_still_work() -> None:
    out = make_out(trace=False)
    for check in (
        {"type": "tool_called", "name": "search"},
        {"type": "tool_not_called", "name": "x"},
        {"type": "tool_args_contain", "name": "a", "arg": "b", "value": 1},
        {"type": "forbidden_arg_values", "arg": "a", "values": [1]},
        {"type": "tool_order", "calls": ["a"]},
        {"type": "tool_succeeded"},
        {"type": "tool_call_count", "max": 5},
        {"type": "rounds", "max": 5},
        {"type": "output_cites_tool_result", "name": "a"},
    ):
        score = run([check], out)
        assert score.passed is False and (score.reason or "").endswith("no trace"), check
    assert run([{"type": "output_contains", "text": "sold"}], out).passed is True
    assert run([{"type": "output_regex", "pattern": "Rome"}], out).passed is True
    assert run([{"type": "python", "fn": "tests.scorer_fns:untyped"}], out).passed is True


def test_no_checks_passes_or_skips() -> None:
    plain = run([])
    assert (plain.value, plain.passed, plain.reason) == (1.0, True, "no checks")
    skipped = run([], on_missing="skip")
    assert (skipped.value, skipped.passed, skipped.reason) == (None, None, "no checks")


def test_suite_checks_and_the_cases_own_checks_both_apply() -> None:
    scorer = ChecksScorer("c", checks=[{"type": "tool_called", "name": "search"}])
    own = Case.model_validate(
        {"id": "c", "input": "q", "expect": [{"type": "output_contains", "text": "zzz"}]}
    )
    score = scorer.score(own, make_out())
    assert score.passed is False and score.reason == 'output_contains "zzz": missing ["zzz"]'
    assert scorer.score(CASE, make_out()).passed is True  # a case with none gets the suite's
    only_own = ChecksScorer("c", on_missing="skip")
    assert only_own.score(CASE, make_out()).passed is None
    assert only_own.score(own, make_out()).passed is False


def test_a_case_field_other_than_expect() -> None:
    scorer = ChecksScorer("c", field="asserts")
    case = Case.model_validate(
        {"id": "c", "input": "q", "asserts": [{"type": "tool_called", "name": "search"}]}
    )
    assert scorer.score(case, make_out()).passed is True
    assert (
        scorer.score(Case.model_validate({"id": "c", "input": "q", "expect": 3}), make_out()).reason
        == "no checks"
    )
    broken = Case.model_validate({"id": "c", "input": "q", "asserts": [{"type": "nope"}]})
    score = scorer.score(broken, make_out())
    assert score.value is None and score.passed is None
    assert score.reason is not None and score.reason.startswith(
        "invalid checks: asserts[0] has unknown type"
    )


# --- validation: the pure list validator ------------------------------------------------------


@pytest.mark.parametrize(
    ("checks", "message"),
    [
        ("tool_called", "p must be a list of checks"),
        (["tool_called"], "p[0] must be an object with a 'type'"),
        ([{"name": "x"}], "p[0] needs a 'type' naming the check"),
        ([{"type": ""}], "p[0] needs a 'type' naming the check"),
        (
            [{"type": "tool_cald", "name": "x"}],
            "p[0] has unknown type 'tool_cald' (did you mean 'tool_called'?)",
        ),
        ([{"type": "zzz"}], "p[0] has unknown type 'zzz' (valid types: tool_called, "),
        ([{"type": "tool_called"}], "p[0] needs 'name' (tool_called)"),
        (
            [{"type": "tool_called", "name": "x", "nmae": 1}],
            "p[0] has unknown key 'nmae' (did you mean 'name'?)",
        ),
        (
            [{"type": "tool_called", "name": "x", "bogus": 1}],
            "p[0] has unknown key 'bogus' (valid keys: type, name, args, args_match, min, max, "
            "turn)",
        ),
        ([{"type": "rounds", "max": 2, "turn": 0}], "p[0] has unknown key 'turn'"),
        ([{"type": "any_of", "checks": [], "turn": 1}], "p[0] has unknown key 'turn'"),
        ([{"type": "tool_called", "name": 3}], "p[0].name must be a non-empty string"),
        ([{"type": "tool_called", "name": ""}], "p[0].name must be a non-empty string"),
        ([{"type": "tool_called", "name": "x", "args": [1]}], "p[0].args must be an object"),
        (
            [{"type": "tool_called", "name": "x", "args_match": "fuzzy"}],
            "p[0].args_match must be one of exact, contains",
        ),
        ([{"type": "tool_called", "name": "x", "min": -1}], "p[0].min must be an integer >= 0"),
        ([{"type": "tool_called", "name": "x", "min": 1.5}], "p[0].min must be an integer >= 0"),
        ([{"type": "tool_called", "name": "x", "max": True}], "p[0].max must be an integer >= 0"),
        (
            [{"type": "tool_called", "name": "x", "min": 3, "max": 1}],
            "p[0] has min greater than max",
        ),
        ([{"type": "tool_called", "name": "x", "turn": "1"}], "p[0].turn must be an integer >= 0"),
        ([{"type": "tool_called", "name": "x", "turn": -2}], "p[0].turn must be an integer >= 0"),
        (
            [{"type": "tool_args_contain", "name": "x", "arg": "a"}],
            "p[0] needs 'value' (tool_args_contain)",
        ),
        (
            [{"type": "tool_args_contain", "name": "x", "value": 1}],
            "p[0] needs 'arg' (tool_args_contain)",
        ),
        (
            [{"type": "forbidden_arg_values", "arg": "a", "values": []}],
            "p[0].values must be a non-empty list",
        ),
        (
            [{"type": "forbidden_arg_values", "arg": "a", "values": "x"}],
            "p[0].values must be a non-empty list",
        ),
        ([{"type": "tool_order"}], "p[0] needs 'calls' (tool_order)"),
        (
            [{"type": "tool_order", "calls": []}],
            "p[0].calls must be a non-empty list of tool names or objects",
        ),
        (
            [{"type": "tool_order", "calls": ["a", 3]}],
            "p[0].calls[1] must be a tool name or an object with a 'name'",
        ),
        ([{"type": "tool_order", "calls": ["a", {"args": {}}]}], "p[0].calls[1] needs 'name'"),
        (
            [{"type": "tool_order", "calls": [{"name": "a", "arg": {}}]}],
            "p[0].calls[0] has unknown key 'arg'",
        ),
        (
            [{"type": "tool_order", "calls": [{"name": "a", "args": 1}]}],
            "p[0].calls[0].args must be an object",
        ),
        (
            [{"type": "tool_order", "calls": ["a"], "match": "loose"}],
            "p[0].match must be one of subsequence, exact, any_order",
        ),
        ([{"type": "tool_call_count"}], "p[0] needs 'min' or 'max' (tool_call_count)"),
        ([{"type": "rounds"}], "p[0] needs 'min' or 'max' (rounds)"),
        ([{"type": "output_contains"}], "p[0] needs 'text' (output_contains)"),
        (
            [{"type": "output_contains", "text": ""}],
            "p[0].text must be a non-empty string or a list of non-empty strings",
        ),
        (
            [{"type": "output_contains", "text": ["a", 2]}],
            "p[0].text must be a non-empty string or a list of non-empty strings",
        ),
        (
            [{"type": "output_contains", "text": "a", "all": "yes"}],
            "p[0].all must be true or false",
        ),
        (
            [{"type": "output_omits", "text": "a", "case_sensitive": 1}],
            "p[0].case_sensitive must be true or false",
        ),
        (
            [{"type": "output_regex", "pattern": "("}],
            "p[0].pattern is not a valid regular expression",
        ),
        (
            [{"type": "output_regex", "pattern": "a", "flags": "x"}],
            "p[0].flags must be a string of the letters i, m, s",
        ),
        (
            [{"type": "output_cites_tool_result", "name": "a", "min_matches": 0}],
            "p[0].min_matches must be an integer >= 1",
        ),
        (
            [{"type": "output_cites_tool_result", "name": "a", "rel_tol": -1}],
            "p[0].rel_tol must be a number >= 0",
        ),
        (
            [{"type": "output_cites_tool_result", "name": "a", "strict": "no"}],
            "p[0].strict must be true or false",
        ),
        ([{"type": "any_of", "checks": []}], "p[0].checks must be a non-empty list of checks"),
        (
            [{"type": "any_of", "checks": {"type": "x"}}],
            "p[0].checks must be a non-empty list of checks",
        ),
        (
            [{"type": "any_of", "checks": [{"type": "tool_called", "name": "a"}, {"type": "zzz"}]}],
            "p[0].checks[1] has unknown type 'zzz'",
        ),
        (
            [{"type": "any_of", "checks": [{"type": "tool_called", "name": "a"}, 5]}],
            "p[0].checks[1] must be an object with a 'type'",
        ),
        ([{"type": "not", "check": "x"}], "p[0].check must be an object with a 'type'"),
        (
            [{"type": "not", "check": {"type": "tool_called"}}],
            "p[0].check needs 'name' (tool_called)",
        ),
        (
            [{"type": "not", "check": {"type": "any_of", "checks": [{"type": "rounds"}]}}],
            "p[0].check.checks[0] needs 'min' or 'max' (rounds)",
        ),
        ([{"type": "python"}], "p[0] needs 'fn' (python)"),
        (
            [{"type": "python", "fn": "no_colon"}],
            "p[0].fn must be 'module:function', got 'no_colon'",
        ),
        ([{"type": "python", "fn": "a:b:c"}], "p[0].fn must be 'module:function', got 'a:b:c'"),
        ([{"type": "tool_called", "name": "a"}, {"type": "tool_called"}], "p[1] needs 'name'"),
    ],
)
def test_validate_checks_reports_the_path_of_the_problem(checks: Any, message: str) -> None:
    from evalproof.scorers.checks import validate_checks

    with pytest.raises(ValueError) as caught:
        validate_checks(checks, "p")
    assert str(caught.value).startswith(message)


def test_the_shapes_of_all_items_are_reported_before_their_vocabulary() -> None:
    from evalproof.scorers.checks import validate_checks

    with pytest.raises(ValueError, match=r"^p\[1\] must be an object"):
        validate_checks([{"type": "zzz"}, "tool_called"], "p")


def test_every_check_type_in_the_vocabulary_has_a_valid_form() -> None:
    from evalproof.scorers.checks import CHECK_TYPES, validate_checks

    forms = [
        {
            "type": "tool_called",
            "name": "a",
            "args": {"x": 1},
            "args_match": "contains",
            "min": 0,
            "max": 2,
            "turn": 0,
        },
        {"type": "tool_not_called", "name": "a", "args": {}, "args_match": "exact", "turn": 1},
        {"type": "tool_args_contain", "name": "a", "arg": "x", "value": None, "turn": 0},
        {"type": "forbidden_arg_values", "arg": "x", "values": [1, "a"], "name": "a", "turn": 0},
        {
            "type": "tool_order",
            "calls": ["a", {"name": "b", "args": {"k": 1}, "args_match": "contains"}],
            "match": "exact",
            "turn": 0,
        },
        {"type": "tool_succeeded", "name": "a", "turn": 0},
        {"type": "tool_call_count", "min": 1, "max": 2, "name": "a", "turn": 0},
        {"type": "rounds", "min": 1, "max": 4},
        {
            "type": "output_contains",
            "text": ["a", "b"],
            "all": False,
            "case_sensitive": True,
            "turn": 0,
        },
        {"type": "output_omits", "text": "a", "case_sensitive": False, "turn": 0},
        {"type": "output_regex", "pattern": r"\d+", "flags": "ims", "turn": 0},
        {
            "type": "output_cites_tool_result",
            "name": "a",
            "min_matches": 2,
            "rel_tol": 0.01,
            "strict": True,
            "turn": 0,
        },
        {"type": "any_of", "checks": [{"type": "rounds", "max": 1}]},
        {"type": "not", "check": {"type": "rounds", "max": 1}},
        {"type": "python", "fn": "pkg.mod:func"},
    ]
    assert [f["type"] for f in forms] == list(CHECK_TYPES)
    validate_checks(forms, "p")


# --- validation at suite load -----------------------------------------------------------------


def _suite_file(tmp_path: Path, params: str) -> Path:
    path = tmp_path / "suite.yaml"
    path.write_text(
        "name: s\ndataset: d.jsonl\nsystem: {target: 'python:a:b'}\n"
        f"scorers:\n  - {{name: behaviour, type: checks, params: {params}}}\n",
        "utf-8",
    )
    return path


@pytest.mark.parametrize(
    ("params", "message"),
    [
        (
            "{checks: [{type: tool_cald, name: x}]}",
            r"params\.checks\[0\] has unknown type 'tool_cald' \(did you mean 'tool_called'\?\)",
        ),
        (
            "{checks: [{type: tool_called, name: x}, {type: rounds}]}",
            r"params\.checks\[1\] needs 'min' or 'max' \(rounds\)",
        ),
        (
            "{checks: [{type: any_of, checks: [{type: output_regex, pattern: '('}]}]}",
            r"params\.checks\[0\]\.checks\[0\]\.pattern is not a valid regular expression",
        ),
        ("{checks: {type: tool_called}}", r"params\.checks must be a list of checks"),
        ("{on_missing: maybe}", r"params\.on_missing must be one of pass, skip"),
        ("{mode: most}", r"params\.mode must be one of all, fraction"),
        ("{field: 3}", r"params\.field must be the name of a case field"),
        ("{filed: expect}", r"params has unknown key 'filed' \(did you mean 'field'\?\)"),
    ],
)
def test_checks_params_are_validated_when_the_suite_loads(
    tmp_path: Path, params: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=r"suite\.yaml: scorers\.0: " + message):
        load_config(_suite_file(tmp_path, params))


def test_good_checks_params_load(tmp_path: Path) -> None:
    cfg = load_config(
        _suite_file(
            tmp_path,
            "{checks: [{type: tool_called, name: x}], field: asserts, on_missing: skip, "
            "mode: fraction}",
        )
    )
    assert cfg.scorers[0].type == "checks" and cfg.scorers[0].params["mode"] == "fraction"
    assert load_config(_suite_file(tmp_path, "{}")).scorers[0].params == {}


def test_make_scorer_builds_a_checks_scorer_and_names_it_in_errors(tmp_path: Path) -> None:
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="python:a:b"),
        scorers=[ScorerConfig(name="b", type="checks", params={"mode": "fraction"})],
        base_dir=str(tmp_path),
    )
    scorer = make_scorer(suite.scorers[0], suite)
    assert isinstance(scorer, ChecksScorer) and scorer.mode == "fraction"
    pyfn = {"checks": [{"type": "python", "fn": "tests.gone:fn"}]}
    broken = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="python:a:b"),
        scorers=[ScorerConfig(name="mine", type="checks", params=pyfn)],
    )
    with pytest.raises(
        ConfigError, match=r"scorer 'mine' python check: cannot import 'tests\.gone'"
    ):
        make_scorer(broken.scorers[0], broken)


def test_a_python_check_resolves_from_the_suite_directory(tmp_path: Path) -> None:
    (tmp_path / "my_check_mod.py").write_text(
        "def cited(case, out):\n    return 'ok' in out.output\n", "utf-8"
    )
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="python:a:b"),
        scorers=[
            ScorerConfig(
                name="b",
                type="checks",
                params={"checks": [{"type": "python", "fn": "my_check_mod:cited"}]},
            )
        ],
        base_dir=str(tmp_path),
    )
    scorer = make_scorer(suite.scorers[0], suite)
    assert scorer.score(CASE, SystemOutput(output="ok")).passed is True
    assert scorer.score(CASE, SystemOutput(output="no")).passed is False


# --- validation at dataset load ---------------------------------------------------------------


def _dataset(tmp_path: Path, *rows: dict[str, Any]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return path


@pytest.mark.parametrize(
    ("expect", "message"),
    [
        (
            [{"type": "tool_called", "name": "x"}, {"type": "tool_cald"}, {"type": "rounds"}],
            r"expect\[1\] has unknown type 'tool_cald' \(did you mean 'tool_called'\?\)",
        ),
        (
            [
                {"type": "tool_called", "name": "x"},
                {"type": "tool_called", "name": "x"},
                {
                    "type": "any_of",
                    "checks": [
                        {"type": "tool_called", "name": "a"},
                        {"type": "rounds", "min": "1"},
                    ],
                },
            ],
            r"expect\[2\]\.checks\[1\]\.min must be an integer >= 0",
        ),
        (
            [
                {"type": "tool_called", "name": "x"},
                {"type": "tool_called", "name": "x"},
                {"type": "any_of", "checks": [{"type": "tool_called"}]},
            ],
            r"expect\[2\]\.checks\[0\] needs 'name' \(tool_called\)",
        ),
        (
            [{"type": "not", "check": {"type": "python", "fn": "bad"}}],
            r"expect\[0\]\.check\.fn must be 'module:function', got 'bad'",
        ),
        (
            [{"type": "output_contains", "text": "a", "cases": True}],
            r"expect\[0\] has unknown key 'cases'",
        ),
        (
            [{"type": "tool_called", "name": "x"}, "tool_called"],
            r"expect\[1\] must be an object with a 'type'",
        ),
        ([{"name": "x"}], r"expect\[0\] needs a 'type' naming the check"),
    ],
)
def test_expect_is_validated_against_the_vocabulary_at_dataset_load(
    tmp_path: Path, expect: list[Any], message: str
) -> None:
    path = _dataset(
        tmp_path, {"id": "ok", "input": "q"}, {"id": "a", "input": "q", "expect": expect}
    )
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2: case 'a': " + message):
        load_dataset(path)


def test_a_valid_expect_loads_and_a_python_check_is_only_checked_for_its_form(
    tmp_path: Path,
) -> None:
    expect = [
        {"type": "tool_called", "name": "x"},
        {"type": "python", "fn": "not_installed_anywhere:fn"},  # imported when the suite runs
    ]
    (case,) = load_dataset(_dataset(tmp_path, {"id": "a", "input": "q", "expect": expect}))
    assert case.expect == expect


@pytest.mark.parametrize(
    ("row", "message"),
    [
        (
            {"state": "CA"},
            "case 'a': state must be an object (evalproof reads 'state' as the initial session "
            "state; rename the field if it means something else)",
        ),
        (
            {"turns": "later"},
            "case 'a': turns must be a list of strings or objects (evalproof reads 'turns' as "
            "the user turns that follow 'input'; rename the field if it means something else)",
        ),
        (
            {"expect": "x"},
            "case 'a': expect must be a list of checks (evalproof reads 'expect' as the inline "
            "checks of the checks scorer; rename the field if it means something else)",
        ),
    ],
)
def test_a_wrong_typed_agent_field_says_to_rename_it(
    tmp_path: Path, row: dict[str, Any], message: str
) -> None:
    path = _dataset(tmp_path, {"id": "a", "input": "q", **row})
    with pytest.raises(DatasetError) as caught:
        load_dataset(path)
    assert str(caught.value) == f"{path}:1: {message}"


# --- verify -----------------------------------------------------------------------------------


AGENT = """
def agent(question):
    n = int(question["n"])
    value = (n + 2) * 10
    calls = [{"name": "lookup", "args": {"id": n}, "result": {"value": value}}]
    if n % 2 == 0:
        calls.append({"name": "notify", "args": {"id": n}})
    return {"output": f"The value is {value}.", "trace": {"tool_calls": calls}}
"""


def _agent_suite(tmp_path: Path) -> Path:
    rows = [
        {
            "id": f"c{n}",
            "input": {"n": n},
            "expect": [{"type": "tool_called", "name": "notify"}] if n % 2 == 0 else [],
        }
        for n in range(6)
    ]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    (tmp_path / "agent_mod.py").write_text(AGENT, "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:agent_mod:agent'}\n"
        "scorers:\n"
        "  - name: behaviour\n    type: checks\n    params:\n      mode: fraction\n"
        "      checks:\n"
        "        - {type: tool_called, name: lookup}\n"
        "        - {type: output_cites_tool_result, name: lookup}\n",
        "utf-8",
    )
    return suite


def test_a_checks_run_scores_and_verify_re_scores_it(tmp_path: Path) -> None:
    cfg = load_config(_agent_suite(tmp_path))
    run = run_suite(cfg, use_cache=False)
    assert all(r.scores["behaviour"].passed for r in run.results)
    run_path = save_run(run, tmp_path / "run.json")

    result = verify_run(load_run(run_path), cfg)
    assert result.ok, result.problems
    assert result.rescored == ["behaviour"] and result.not_rescored == []

    data = json.loads(run_path.read_text("utf-8"))
    data["results"][2]["scores"]["behaviour"] = {"value": 0.5, "passed": False}
    run_path.write_text(json.dumps(data), "utf-8")
    bad = verify_run(load_run(run_path), cfg)
    assert not bad.ok
    assert any("`c2` on `behaviour`" in p for p in bad.problems)


def test_a_failing_case_is_scored_from_its_stored_trace(tmp_path: Path) -> None:
    suite = _agent_suite(tmp_path)
    # c1 is odd, so the agent never calls notify; the case now demands it
    rows = [json.loads(line) for line in (tmp_path / "cases.jsonl").read_text("utf-8").splitlines()]
    rows[1]["expect"] = [{"type": "tool_called", "name": "notify"}]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    cfg = load_config(suite)
    run = run_suite(cfg, use_cache=False)
    score = run.by_id()["c1"].scores["behaviour"]
    assert score.passed is False and score.value == 2 / 3
    assert score.reason == "tool_called notify: not called"
    assert verify_run(run, cfg).ok
