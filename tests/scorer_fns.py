"""Functions the python scorer and the ``python`` check tests point at (``tests.scorer_fns:fn``)."""

from __future__ import annotations

from typing import Any

from evalproof.models import Case, Score, SystemOutput

NOT_CALLABLE = 3


def long_enough(case: Case, out: SystemOutput) -> bool:
    return len(out.output.split()) >= 3


def short(case: Case, out: SystemOutput) -> bool:
    return len(out.output.split()) < 3


def with_reason(case: Case, out: SystemOutput) -> tuple[bool, str]:
    ok = "ok" in out.output
    return ok, "" if ok else "the reply never says ok"


def with_no_reason(case: Case, out: SystemOutput) -> tuple[bool, None]:
    return "ok" in out.output, None


def length_share(case: Case, out: SystemOutput) -> float:
    return min(len(out.output) / 10, 1.0)


def exact_one(case: Case, out: SystemOutput) -> int:
    return 1


def returns_score(case: Case, out: SystemOutput) -> Score:
    return Score(value=0.25, passed=False, reason="a hand-built score")


def raises(case: Case, out: SystemOutput) -> bool:
    raise ValueError("boom")


def returns_none(case: Case, out: SystemOutput) -> None:
    return None


def returns_text(case: Case, out: SystemOutput) -> str:
    return "yes"


def too_big(case: Case, out: SystemOutput) -> float:
    return 1.5


def not_a_number(case: Case, out: SystemOutput) -> float:
    return float("nan")


def three_items(case: Case, out: SystemOutput) -> tuple[bool, str, int]:
    return True, "x", 1


def bad_pair(case: Case, out: SystemOutput) -> tuple[str, str]:
    return "yes", "reason"


def undecided_score(case: Case, out: SystemOutput) -> Score:
    return Score(value=None)


def case_aware(case: Case, out: SystemOutput) -> bool:
    return case.id in out.output


def untyped(case: Any, out: Any) -> bool:
    return True
