"""A tiny agent with two tools, and no model: the system under test for the agent example.

It reads a message, picks tools with keyword rules, calls them, and writes a reply from what the
tools returned. Everything is deterministic and offline, so the suite runs in CI on every commit.
``TraceRecorder`` records what it did (the tool calls, their arguments and results, the replies),
which is what the ``checks`` scorer reads.

The tools:

* ``get_weather(city, day)``: a made-up forecast table.
* ``get_events(day)``: a made-up calendar.

The variant arrives as ``ctx.vars``. ``v1`` sets ``calendar_for_outings: false``, a flag that was
left off: asked whether to go on a picnic or a hike, the agent then answers from the forecast alone
and never looks at the calendar. The reply still reads well, which is why a check on the tool
calls, not on the words, is what finds it. ``v2`` sets the flag, so the agent looks at both.

``flaky`` (in ``suite-flaky.yaml``) sets ``flaky_calendar: 0.4``: on an outing question it forgets
the calendar in about 40% of the trials, so a case fails some runs and passes others, which one run
per case cannot show and ``--trials 5`` can. The coin is ``random.Random(f"{case_id}:{trial}")``, so
which trial forgets is fixed: the same on every machine and every run, and the example stays
deterministic and offline.
"""

from __future__ import annotations

import difflib
import random
import re
from collections.abc import Callable
from typing import Any

from evalproof import RunContext
from evalproof.adapters import TraceRecorder

# --- the two tools: fixed tables standing in for a forecast service and a calendar ---------------

# city -> (name, temperature today in Celsius, condition, chance of rain today in percent)
FORECASTS: dict[str, tuple[str, int, str, int]] = {
    "paris": ("Paris", 18, "cloudy", 40),
    "berlin": ("Berlin", 11, "rainy", 80),
    "lisbon": ("Lisbon", 24, "sunny", 5),
    "oslo": ("Oslo", 2, "snowy", 70),
    "tokyo": ("Tokyo", 21, "clear", 10),
    "nairobi": ("Nairobi", 26, "partly cloudy", 15),
}
# day -> how much warmer it gets, and how much likelier rain, than today
DAYS: dict[str, int] = {"today": 0, "tomorrow": 1, "saturday": 2, "sunday": 3, "monday": 4}
CALENDAR: dict[str, list[tuple[str, int]]] = {
    "today": [("Team sync", 14)],
    "tomorrow": [("Dentist", 10), ("Design review", 15)],
    "saturday": [],
    "sunday": [("Family lunch", 12)],
    "monday": [("Standup", 9), ("Planning", 11), ("Budget review", 14)],
}


def get_weather(city: str, day: str) -> dict[str, Any]:
    entry = FORECASTS.get(city.lower())
    if entry is None:
        raise LookupError(f"no forecast for {city}")
    name, temp_c, condition, rain = entry
    shift = DAYS[day]
    return {
        "city": name,
        "day": day,
        "temp_c": temp_c + shift,
        "condition": condition,
        "rain_chance_pct": min(100, rain + 5 * shift),
    }


def get_events(day: str) -> dict[str, Any]:
    events = [{"title": title, "start_hour": hour} for title, hour in CALENDAR[day]]
    return {"day": day, "events": events, "count": len(events)}


# --- reading a message ---------------------------------------------------------------------------

WEATHER = {
    "weather",
    "forecast",
    "temperature",
    "rain",
    "raining",
    "umbrella",
    "sunny",
    "snow",
    "cold",
    "warm",
    "hot",
}
CALENDAR_WORDS = {
    "calendar",
    "schedule",
    "meetings",
    "meeting",
    "events",
    "agenda",
    "appointments",
    "busy",
    "free",
}
OUTING = {"picnic", "hike", "hiking", "barbecue", "bbq", "outdoor", "outdoors", "bike", "cycling"}
DESTRUCTIVE = {"delete", "erase", "wipe", "remove", "cancel"}
ACTIVITY = {
    "picnic": "a picnic",
    "hike": "a hike",
    "hiking": "a hike",
    "barbecue": "a barbecue",
    "bbq": "a barbecue",
    "outdoor": "being outside",
    "outdoors": "being outside",
    "bike": "a bike ride",
    "cycling": "a bike ride",
}
VOCABULARY = WEATHER | CALENDAR_WORDS | OUTING | DESTRUCTIVE | set(DAYS) | set(FORECASTS)
NOT_A_CITY = {"the", "my", "a", "an", "this", "that", "case"}


def _canonical(word: str) -> str | None:
    """A word from the vocabulary, allowing a typo in a word of five letters or more."""
    if word in VOCABULARY:
        return word
    if len(word) < 5:
        return None
    close = difflib.get_close_matches(word, sorted(VOCABULARY), n=1, cutoff=0.85)
    return close[0] if close else None


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())
    return {c for c in (_canonical(w) for w in words) if c}


def _intent(words: set[str]) -> str:
    if words & DESTRUCTIVE:
        return "unsafe"
    if words & OUTING:
        return "outing"
    weather, calendar = bool(words & WEATHER), bool(words & CALENDAR_WORDS)
    if weather and calendar:
        return "briefing"
    if weather:
        return "weather"
    if calendar:
        return "calendar"
    return "other"


def _city(words: set[str], text: str) -> str | None:
    """A known city, else the word after "in" (which the weather tool will not know)."""
    for word in sorted(words):
        if word in FORECASTS:
            return FORECASTS[word][0]
    match = re.search(r"\bin\s+([a-z]+)", text.lower())
    if match and match.group(1) not in NOT_A_CITY and _canonical(match.group(1)) is None:
        return match.group(1).title()
    return None


def _when(day: str) -> str:
    return day if day in ("today", "tomorrow") else f"on {day.title()}"


# --- the agent -----------------------------------------------------------------------------------


def _call(
    rec: TraceRecorder, name: str, tool: Callable[..., dict[str, Any]], **args: Any
) -> dict[str, Any] | None:
    """Call a tool and record the call, failed or not."""
    try:
        result = tool(**args)
    except LookupError as exc:
        rec.tool(name, args, ok=False, error=str(exc))
        return None
    rec.tool(name, args, result=result)
    return result


def _weather_line(forecast: dict[str, Any]) -> str:
    return (
        f"{forecast['city']} {_when(forecast['day'])}: {forecast['temp_c']}°C and "
        f"{forecast['condition']}, {forecast['rain_chance_pct']}% chance of rain."
    )


def _calendar_line(day: str, calendar: dict[str, Any]) -> str:
    if not calendar["events"]:
        return f"Nothing on your calendar {_when(day)}."
    listed = ", ".join(f"{e['title']} at {e['start_hour']}:00" for e in calendar["events"])
    count = calendar["count"]
    return f"You have {count} event{'s' if count != 1 else ''} {_when(day)}: {listed}."


def _verdict(activity: str, forecast: dict[str, Any], calendar: dict[str, Any] | None) -> str:
    if forecast["rain_chance_pct"] > 30:
        return f"Not great for {activity}: take an umbrella or pick another day."
    if calendar and calendar["events"]:
        first = calendar["events"][0]
        return (
            f"The weather suits {activity}, but you have {first['title']} at "
            f"{first['start_hour']}:00, so plan around it."
        )
    return f"Good conditions for {activity}."


def _turn(rec: TraceRecorder, text: str, memory: dict[str, Any], check_calendar: bool) -> None:
    """Handle one user message: pick the tools, call them, reply. ``memory`` carries the city and
    the day from one turn to the next, and the intent of a question that was left unanswered."""
    rec.round()
    words = _keywords(text)
    intent = _intent(words)
    city = _city(words, text) or memory["city"]
    day = next((w for w in sorted(words) if w in DAYS), None) or memory["day"]
    if intent == "other" and memory["pending"] and (city != memory["city"] or day != memory["day"]):
        intent = memory["pending"]  # the answer to "which city?"
    memory["city"], memory["day"], memory["pending"] = city, day, None
    day = day or "today"

    if intent == "unsafe":
        rec.reply("I can't delete or change anything; I can only read your calendar and forecasts.")
    elif intent == "other":
        rec.reply("I can only help with the weather and your calendar.")
    elif intent in ("weather", "briefing", "outing") and city is None:
        memory["pending"] = intent
        rec.reply("Which city should I check?")
    else:
        wants_weather = intent in ("weather", "briefing", "outing") and city is not None
        # The planted bug of v1: an outing is judged on the forecast alone.
        outing_calendar = intent == "outing" and check_calendar
        wants_calendar = intent in ("calendar", "briefing") or outing_calendar
        forecast = (
            _call(rec, "get_weather", get_weather, city=city, day=day) if wants_weather else None
        )
        calendar = _call(rec, "get_events", get_events, day=day) if wants_calendar else None
        if wants_weather and forecast is None:
            rec.reply(f"I don't have a forecast for {city}.")
            return
        parts: list[str] = []
        if forecast is not None:
            parts.append(_weather_line(forecast))
        if calendar is not None:
            parts.append(_calendar_line(day, calendar))
        if intent == "outing" and forecast is not None:
            activity = ACTIVITY[next(w for w in sorted(words) if w in OUTING)]
            parts.append(_verdict(activity, forecast, calendar))
        rec.reply(" ".join(parts))


def _forgets_calendar(ctx: RunContext) -> bool:
    """The ``flaky`` variant's coin: true for a fixed 40% of (case, trial) pairs, whatever the
    machine, because the generator is seeded by the pair and by nothing else."""
    chance = float(ctx.vars.get("flaky_calendar", 0.0))
    return chance > 0 and random.Random(f"{ctx.case_id}:{ctx.trial}").random() < chance


def run(question: str, ctx: RunContext) -> dict[str, Any]:
    """The system under test: one conversation, the case input then each later turn."""
    check_calendar = bool(ctx.vars.get("calendar_for_outings", True)) and not _forgets_calendar(ctx)
    memory: dict[str, Any] = {"city": ctx.state.get("home_city"), "day": None, "pending": None}
    rec = TraceRecorder()
    for message in [question, *ctx.turns]:
        _turn(rec, str(message), memory, check_calendar)
    return dict(rec.result())
