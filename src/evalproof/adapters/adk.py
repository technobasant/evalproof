"""Google ADK: ``from_events(events)`` turns the events of a run into an evalproof trace.

SDK fields read (names confirmed in the SDK source, not from memory):

* google-adk==2.14.1 google/adk/events/event.py:97 ``Event`` (a ``LlmResponse``), :113 ``author``,
  :305 ``is_final_response()``; the aliases at :106-107 (camelCase, ``populate_by_name``) are why
  a dumped event with ``usageMetadata`` or ``functionCall`` keys is read too.
* google-adk==2.14.1 google/adk/models/llm_response.py:58 ``model_version``, :61 ``content``,
  :71 ``partial``, :138 ``usage_metadata``, :211 ``get_function_calls``, :220
  ``get_function_responses``.
* google-genai==3.3.0 google/genai/types.py:2676 ``Content.parts``, :2681 ``Content.role``,
  :2375 ``Part.function_call``, :2379 ``Part.function_response``, :2387 ``Part.text``, :2391
  ``Part.thought``; :1880-1888 ``FunctionCall.id``, ``.args``, ``.name``; :2098-2106
  ``FunctionResponse.id``, ``.name``, ``.response``.
* google-genai==3.3.0 google/genai/types.py:8680 ``GenerateContentResponseUsageMetadata.
  cached_content_token_count``, :8684 ``candidates_token_count``, :8692 ``prompt_token_count``.
* A tool failure is a response dict with an ``error`` key: google-adk==2.14.1
  google/adk/tools/function_tool.py:385, :528 (``{"error": ...}``) and :646 (the SDK's own test,
  ``response.get("error")``).

ADK runs return events, not messages. A turn boundary is an event authored by ``user`` that
carries text (a user turn); without one every call is turn 0.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from evalproof.adapters._common import AgentResult, Ledger, count, get
from evalproof.models import ModelUsage, Trace


def _parts(event: Any) -> list[Any]:
    parts = get(get(event, "content"), "parts")
    return list(parts) if isinstance(parts, list | tuple) else []


def _is_final(event: Any, calls: bool, responses: bool) -> bool:
    method = getattr(event, "is_final_response", None)
    if callable(method):
        try:
            return bool(method())
        except Exception:  # a method that cannot run on this object: judge from the parts
            pass
    return not calls and not responses and not get(event, "partial", default=False)


def _error_of(response: Any) -> str | None:
    error = get(response, "error")
    return str(error) if error else None


def from_events(events: Iterable[Any]) -> AgentResult:
    """Build ``{"output", "trace"}`` from ADK events (objects or dicts), in order.

    Calls and responses pair by id, or by order and name when an id is missing. The reply of a
    turn is the text of its last final-response event. Usage is summed per model over the
    complete (not ``partial``) events that report it; ``rounds`` counts the model responses.
    """
    ledger = Ledger()
    turn = 0
    active = False  # an event of the agent since the last user turn began
    replies: dict[int, str] = {}
    usage: dict[str | None, ModelUsage] = {}
    rounds = 0
    for event in events:
        if get(event, "partial", default=False):
            continue
        author = get(event, "author", default="")
        parts = _parts(event)
        calls = [c for p in parts if (c := get(p, "function_call", "functionCall")) is not None]
        responses = [
            r for p in parts if (r := get(p, "function_response", "functionResponse")) is not None
        ]
        text = "".join(
            t
            for p in parts
            if isinstance(t := get(p, "text"), str) and not get(p, "thought", default=False)
        )
        is_user = author == "user"
        if is_user and not responses:
            if active and text:
                turn += 1
                active = False
            continue
        if not is_user:
            active = True
        for c in calls:
            ledger.call(get(c, "name"), get(c, "args"), get(c, "id"), turn)
        for r in responses:
            response = get(r, "response")
            error = _error_of(response)
            ledger.result(
                get(r, "id"),
                get(r, "name"),
                response,
                ok=error is None,
                error=error,
                turn=turn,
            )
        if is_user:
            continue
        meta = get(event, "usage_metadata", "usageMetadata")
        role = get(get(event, "content"), "role")
        if not responses and (meta is not None or calls or text or role == "model"):
            rounds += 1
        if meta is not None:
            model = get(event, "model_version", "modelVersion")
            entry = usage.setdefault(model, ModelUsage(role="agent", model=model))
            entry.tokens_in += count(get(meta, "prompt_token_count", "promptTokenCount"))
            entry.tokens_out += count(get(meta, "candidates_token_count", "candidatesTokenCount"))
            entry.cached_tokens += count(
                get(meta, "cached_content_token_count", "cachedContentTokenCount")
            )
        if text and _is_final(event, bool(calls), bool(responses)):
            replies[turn] = text
    responses_by_turn = [replies.get(i, "") for i in range(max(replies) + 1)] if replies else []
    trace = Trace(
        responses=responses_by_turn,
        tool_calls=ledger.calls,
        rounds=rounds or None,
        usage=list(usage.values()),
    )
    return {"output": responses_by_turn[-1] if responses_by_turn else "", "trace": trace}


__all__ = ["from_events"]
