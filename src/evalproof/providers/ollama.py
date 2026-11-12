"""Ollama, the default provider: local, free, and reproducible with a fixed seed."""

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
    with_retries,
)

DEFAULT_BASE_URL = "http://127.0.0.1:11434"


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        client: httpx.Client | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.model = model
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/")
        self._headers = merge_headers({}, headers)  # e.g. for an Ollama behind a proxy
        self._client = client or httpx.Client(timeout=timeout_s)

    def payload(self, messages: list[Message], params: GenerationParams) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": params.temperature,
            "num_predict": params.max_tokens,
        }
        if params.seed is not None:
            options["seed"] = params.seed
        options.update(params.extra)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
            "stream": False,
            "options": options,
        }
        if params.think is not None:
            body["think"] = params.think
        return body

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        body = self.payload(messages, params)

        def call() -> httpx.Response:
            resp = self._client.post(f"{self.base_url}/api/chat", json=body, headers=self._headers)
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        resp = with_retries(call)
        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()
        message = data.get("message") or {}
        if "content" not in message:
            raise ProviderError(f"ollama: unexpected response shape: {str(data)[:300]}")
        return ChatResponse(
            text=message["content"],
            model=data.get("model", self.model),
            tokens_in=int(data.get("prompt_eval_count") or 0),
            tokens_out=int(data.get("eval_count") or 0),
            latency_ms=latency_ms,
        )
