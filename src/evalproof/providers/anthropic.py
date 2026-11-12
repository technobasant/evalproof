"""Anthropic Messages API over plain HTTP. Optional: needs ANTHROPIC_API_KEY."""

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
    split_system,
    with_retries,
)

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"  # override per suite with ``headers: {anthropic-version: ...}``


class AnthropicProvider:
    name = "anthropic"

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
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        key = read_key(
            "anthropic", api_key_env or "ANTHROPIC_API_KEY", required=True, api_key=api_key
        )
        self._headers = merge_headers(
            {
                "x-api-key": key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
            headers,
        )
        self._client = client or httpx.Client(timeout=timeout_s)

    def payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        system, rest = split_system(messages)
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": params.max_tokens,
            "messages": [m.model_dump() for m in rest],
        }
        # Current Claude models answer a temperature (or top_p, top_k) with HTTP 400, so it
        # goes out only when the suite asks for it. There is no seed parameter at all.
        if "temperature" in params.model_fields_set:
            body["temperature"] = params.temperature
        if system:
            body["system"] = system
        body.update(params.extra)
        return body

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        body = self.payload(messages, params)

        def call() -> httpx.Response:
            resp = self._client.post(
                f"{self.base_url}/v1/messages", json=body, headers=self._headers
            )
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = with_retries(call)
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        stop_reason = data.get("stop_reason")
        if stop_reason == "refusal":
            details = data.get("stop_details")
            category = details.get("category") if isinstance(details, dict) else None
            raise ProviderError(f"anthropic: refused ({category or 'no category'})")
        if stop_reason == "max_tokens" and not text:
            # Adaptive thinking spends output tokens inside max_tokens, so a small budget can
            # be used up before the first word of the answer.
            raise ProviderError(
                "anthropic: max_tokens reached before any text; raise params.max_tokens"
            )
        usage = data.get("usage") or {}
        return ChatResponse(
            text=text,
            model=data.get("model", self.model),
            tokens_in=int(usage.get("input_tokens") or 0),
            tokens_out=int(usage.get("output_tokens") or 0),
            latency_ms=latency_ms,
        )
