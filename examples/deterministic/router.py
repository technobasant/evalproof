"""A simple support-ticket router: the system under test for the offline example.

Four versions of the same function, chosen to exercise every branch of the gate:

* ``baseline_route``   — the current production rules.
* ``regressed_route``  — someone "tidied" the billing keywords. A real regression.
* ``noisy_route``      — adds one keyword; two tickets get worse. Beyond the drop
                         threshold, but two cases out of sixty is not evidence.
* ``refactored_route`` — the same rules restructured. Behaviour is identical.
"""

from __future__ import annotations

BASELINE_RULES: list[tuple[str, tuple[str, ...]]] = [
    (
        "billing",
        ("refund", "invoice", "charge", "payment", "billing", "card", "price", "subscription"),
    ),
    (
        "account",
        (
            "password",
            "login",
            "log in",
            "2fa",
            "two-factor",
            "email address",
            "username",
            "account",
        ),
    ),
    ("technical", ("crash", "error", "bug", "slow", "freez", "not loading", "sync", "broken")),
    ("shipping", ("delivery", "package", "tracking", "courier", "shipped", "arrived", "parcel")),
]


def _route(ticket: str, rules: list[tuple[str, tuple[str, ...]]]) -> str:
    text = ticket.lower()
    for queue, keywords in rules:
        if any(k in text for k in keywords):
            return queue
    return "other"


def baseline_route(ticket: str) -> str:
    return _route(ticket, BASELINE_RULES)


def regressed_route(ticket: str) -> str:
    rules = [("billing", ("invoice", "billing")), *BASELINE_RULES[1:]]
    return _route(ticket, rules)


def noisy_route(ticket: str) -> str:
    technical = ("technical", (*BASELINE_RULES[2][1], "update"))
    rules = [BASELINE_RULES[0], BASELINE_RULES[1], technical, BASELINE_RULES[3]]
    return _route(ticket, rules)


def refactored_route(ticket: str) -> str:
    text = ticket.lower()
    matches = (queue for queue, words in BASELINE_RULES if any(w in text for w in words))
    return next(matches, "other")
