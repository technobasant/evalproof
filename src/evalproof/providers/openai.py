"""OpenAI-compatible Chat Completions over plain HTTP, driven by a profile per provider.

``OpenAICompatibleProvider`` is the one implementation behind ``openai``, ``groq``,
``openrouter``, ``vllm`` and the rest of the table in ``profiles.py``; ``OpenAIProvider`` is
the ``openai`` profile under its older name. A key, where one is needed, comes from the
environment (``api_key_env`` in the suite picks the variable) and goes only into the request
header.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from evalproof.config import GenerationParams
from evalproof.providers.base import (
    ChatResponse,
    Message,
    ProviderError,
    merge_headers,
    read_key,
    with_retries,
)
from evalproof.providers.profiles import PROFILES, Profile


def message_text(content: Any) -> str:
    """The reply text. Some servers (Mistral's reasoning models) return a list of typed parts
    instead of a string; keep the text parts and drop the rest."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            p["text"] for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
        )
    return ""


class OpenAICompatibleProvider:
    def __init__(
        self,
        profile: Profile,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.profile = profile
        self.name = profile.name  # part of the cache key, so it is the profile, not "openai"
        self.model = model
        url = base_url or profile.base_url
        if not url:
            raise ProviderError(f"{profile.name}: set base_url in the suite to use this provider")
        self.base_url = url.rstrip("/")
        # A variable the suite names explicitly must exist, or a typo would fail as a 401.
        key = read_key(
            profile.name,
            api_key_env or profile.api_key_env,
            required=profile.key_required or api_key_env is not None,
            api_key=api_key,
        )
        defaults = {"content-type": "application/json"}
        if key:
            if profile.auth_header == "authorization":
                defaults["authorization"] = f"Bearer {key}"
            else:
                defaults[profile.auth_header] = key
        self._headers = merge_headers(defaults, headers)
        self._client = client or httpx.Client(timeout=timeout_s)

    def payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            self.profile.max_tokens_field: params.max_tokens,
        }
        # Local servers get the defaults, so a local run is repeatable without any setup.
        # Hosted ones get only what the suite set: reasoning models reject a temperature
        # other than their own default, and an unset value should not become a 400.
        explicit = params.model_fields_set
        if self.profile.local or "temperature" in explicit:
            body["temperature"] = params.temperature
        if params.seed is not None and (self.profile.local or "seed" in explicit):
            body[self.profile.seed_field] = params.seed
        body.update(params.extra)
        return body

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        body = self.payload(messages, params)

        def call() -> httpx.Response:
            resp = self._client.post(
                f"{self.base_url}/chat/completions", json=body, headers=self._headers
            )
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = with_retries(call)
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name}: no choices in response: {str(data)[:300]}")
        text = message_text((choices[0].get("message") or {}).get("content"))
        finish = choices[0].get("finish_reason")
        if finish == "content_filter":
            raise ProviderError(f"{self.name}: response blocked by the content filter")
        if finish == "length" and not text:
            raise ProviderError(
                f"{self.name}: max tokens reached before any text; raise params.max_tokens"
            )
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=data.get("model", self.model),
            tokens_in=int(usage.get("prompt_tokens") or 0),
            tokens_out=int(usage.get("completion_tokens") or 0),
            latency_ms=latency_ms,
        )


class OpenAIProvider(OpenAICompatibleProvider):
    """The ``openai`` profile, kept under the name earlier versions exported."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(
            PROFILES["openai"],
            model,
            base_url=base_url,
            timeout_s=timeout_s,
            client=client,
            api_key=api_key,
            api_key_env=api_key_env,
            headers=headers,
        )
