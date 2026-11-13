"""Systems under test: an LLM prompt, a Python function, or an HTTP endpoint.

A system failure is recorded on the case, not raised: one timeout should not throw
away an hour of judging. Errored cases score as failures and are listed in reports.
"""

from __future__ import annotations

import importlib
import string
import sys
import time
from collections.abc import Callable
from typing import Any, Protocol

import httpx

from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import ConfigError, TargetConfig
from evalproof.models import Case, SystemOutput
from evalproof.providers import (
    Message,
    Provider,
    ProviderError,
    make_provider,
    response_cost,
    with_retries,
)


class System(Protocol):
    name: str

    def run(self, case: Case) -> SystemOutput: ...


def render(template: str, variables: dict[str, Any]) -> str:
    """``str.format`` with a clear error naming the missing field."""
    missing = [
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field and field.split(".")[0].split("[")[0] not in variables
    ]
    if missing:
        raise ConfigError(f"prompt template references unknown fields: {sorted(set(missing))}")
    return template.format_map(variables)


class LLMSystem:
    def __init__(self, cfg: TargetConfig, provider: Provider) -> None:
        self.cfg = cfg
        self.provider = provider
        self.name = cfg.label or cfg.target

    def messages(self, case: Case) -> list[Message]:
        msgs: list[Message] = []
        if self.cfg.system_prompt:
            msgs.append(Message(role="system", content=self.cfg.system_prompt))
        msgs.append(
            Message(role="user", content=render(self.cfg.prompt_template, case.render_vars()))
        )
        return msgs

    def run(self, case: Case) -> SystemOutput:
        try:
            resp = self.provider.chat(self.messages(case), self.cfg.params)
        except ProviderError as exc:
            return SystemOutput(output="", error=str(exc))
        return SystemOutput(
            output=resp.text,
            latency_ms=resp.latency_ms,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cost_usd=response_cost(resp, self.cfg.pricing),
            cached=resp.cached,
        )


def _coerce(result: Any) -> tuple[str, list[str]]:
    if isinstance(result, str):
        return result, []
    if isinstance(result, dict) and "output" in result:
        return str(result["output"]), [str(x) for x in result.get("retrieved") or []]
    raise TypeError("a system must return a string or a dict with an 'output' key")


class PythonSystem:
    """``python:package.module:function`` — the function receives the case input.

    ``search_path`` (the suite's directory) is put on ``sys.path`` so a suite can ship
    its system next to it and still run from any working directory.
    """

    def __init__(self, spec: str, search_path: str | None = None) -> None:
        self.name = spec
        if search_path and search_path not in sys.path:
            sys.path.insert(0, search_path)
        try:
            _, module_name, func_name = spec.split(":", 2)
        except ValueError as exc:
            raise ConfigError(
                f"python target must be 'python:<module>:<function>', got {spec!r}"
            ) from exc
        module = importlib.import_module(module_name)
        func = getattr(module, func_name, None)
        if not callable(func):
            raise ConfigError(f"{module_name}.{func_name} is not callable")
        self.func: Callable[[Any], Any] = func

    def run(self, case: Case) -> SystemOutput:
        start = time.perf_counter()
        try:
            output, retrieved = _coerce(self.func(case.input))
        except Exception as exc:
            return SystemOutput(output="", error=f"{type(exc).__name__}: {exc}")
        return SystemOutput(
            output=output, retrieved=retrieved, latency_ms=(time.perf_counter() - start) * 1000
        )


class HttpSystem:
    """POSTs ``{"id", "input"}``; expects JSON ``{"output", "retrieved"?}`` or plain text."""

    def __init__(self, cfg: TargetConfig, client: httpx.Client | None = None) -> None:
        self.cfg = cfg
        self.name = cfg.label or cfg.target
        self._client = client or httpx.Client(timeout=cfg.timeout_s)

    def run(self, case: Case) -> SystemOutput:
        def call() -> httpx.Response:
            resp = self._client.post(self.cfg.target, json={"id": case.id, "input": case.input})
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        try:
            resp = with_retries(call)
            ctype = resp.headers.get("content-type", "")
            output, retrieved = _coerce(resp.json() if "json" in ctype else resp.text)
        except (ProviderError, TypeError, ValueError) as exc:
            return SystemOutput(output="", error=str(exc))
        return SystemOutput(
            output=output, retrieved=retrieved, latency_ms=(time.perf_counter() - start) * 1000
        )


def make_system(
    cfg: TargetConfig,
    cache: ResponseCache | None = None,
    client: httpx.Client | None = None,
    search_path: str | None = None,
) -> System:
    kind = cfg.kind
    if kind == "python":
        return PythonSystem(cfg.target, search_path=search_path)
    if kind == "http":
        return HttpSystem(cfg, client=client)
    provider: Provider = make_provider(cfg, client=client)
    if cache is not None:
        provider = CachedProvider(provider, cache, base_url=cfg.base_url)
    return LLMSystem(cfg, provider)
