"""The one thing evalproof needs from a model: send messages, get text and token counts back."""

from __future__ import annotations

import os
import random
import time
from collections.abc import Callable
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel

from evalproof.config import GenerationParams

T = TypeVar("T")

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}


class ProviderError(RuntimeError):
    pass


class TransientError(ProviderError):
    """A failure the provider knows is temporary (a CLI reporting a rate limit), so
    ``with_retries`` tries again. Anything else a provider raises is final."""


class Message(BaseModel):
    role: str  # "system" | "user" | "assistant"
    content: str


class ChatResponse(BaseModel):
    text: str
    model: str
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    cached: bool = False


class Provider(Protocol):
    name: str
    model: str

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse: ...


def with_retries(
    fn: Callable[[], T],
    *,
    attempts: int = 4,
    base_delay: float = 0.5,
    sleep: Callable[[float], None] = time.sleep,
    rng: random.Random | None = None,
) -> T:
    """Retry transport errors, retryable HTTP statuses and ``TransientError`` with jittered
    exponential backoff."""
    rng = rng or random.Random(0)
    last: Exception | None = None
    for attempt in range(attempts):
        try:
            return fn()
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code not in RETRY_STATUS:
                raise ProviderError(
                    f"HTTP {exc.response.status_code}: {exc.response.text[:300]}"
                ) from exc
            last = exc
        except (httpx.TransportError, TransientError) as exc:
            last = exc
        if attempt < attempts - 1:
            sleep(base_delay * (2**attempt) * (0.5 + rng.random()))
    raise ProviderError(f"giving up after {attempts} attempts: {last}") from last


def merge_headers(defaults: dict[str, str], extra: dict[str, str] | None) -> dict[str, str]:
    """Suite headers win over a provider's own. Names are lowercased on both sides, because
    HTTP header names are case-insensitive and ``Anthropic-Version`` must replace
    ``anthropic-version`` rather than ride along beside it."""
    merged = {k.lower(): v for k, v in defaults.items()}
    merged.update({k.lower(): v for k, v in (extra or {}).items()})
    return merged


def read_key(
    provider: str,
    env: str | None,
    *,
    required: bool,
    api_key: str | None = None,
) -> str:
    """The API key from an explicit argument or the environment, never from a suite file.

    Returns ``""`` when there is none and none is required; a required key that is missing
    is an error naming the variable to set."""
    key = api_key or (os.environ.get(env) if env else None) or ""
    if not key and required:
        raise ProviderError(f"{provider}: set {env} to use this provider")
    return key


def split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Providers that take the system prompt as a separate field need it pulled out."""
    system = "\n\n".join(m.content for m in messages if m.role == "system") or None
    return system, [m for m in messages if m.role != "system"]
