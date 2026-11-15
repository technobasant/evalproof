"""An HTTP client evalproof creates is closed; one the caller passes in is left open."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from evalproof.config import ScorerConfig, SuiteConfig, TargetConfig
from evalproof.providers import AnthropicProvider, OllamaProvider, OpenAICompatibleProvider
from evalproof.providers.base import close_resource
from evalproof.providers.profiles import PROFILES
from evalproof.runner import environment, run_suite
from evalproof.systems import HttpSystem
from tests.conftest import mock_client


def _ok(req: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"version": "0", "models": []})


def _providers(client: httpx.Client | None) -> list[object]:
    return [
        OllamaProvider("m", client=client),
        AnthropicProvider("m", api_key="k", client=client),
        OpenAICompatibleProvider(PROFILES["openai"], "m", api_key="k", client=client),
        HttpSystem(TargetConfig(target="http://localhost:1/x"), client=client),
    ]


def test_a_provider_closes_the_client_it_made(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    for provider in _providers(None):
        client = provider._client  # type: ignore[attr-defined]
        assert not client.is_closed
        provider.close()  # type: ignore[attr-defined]
        assert client.is_closed, type(provider).__name__


def test_a_provider_leaves_a_client_that_was_passed_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    client = mock_client(_ok)
    for provider in _providers(client):
        provider.close()  # type: ignore[attr-defined]
        assert not client.is_closed, type(provider).__name__
    client.close()


def test_close_resource_skips_what_has_no_close() -> None:
    close_resource(object())  # a coding-agent CLI provider holds nothing to close


def _suite(tmp_path: Path) -> SuiteConfig:
    (tmp_path / "cases.jsonl").write_text('{"id": "c0", "input": "q"}\n', encoding="utf-8")
    return SuiteConfig(
        name="s",
        dataset=str(tmp_path / "cases.jsonl"),
        system=TargetConfig(target="ollama:m"),
        scorers=[ScorerConfig(name="has", type="contains", params={"any_of": ["a"]})],
        cache_dir=str(tmp_path / "cache"),
        base_dir=str(tmp_path),
    )


def test_run_suite_does_not_close_the_client_it_was_given(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/chat":
            return httpx.Response(200, json={"model": "m", "message": {"content": "a"}})
        return _ok(req)

    client = mock_client(handler)
    run_suite(_suite(tmp_path), client=client)
    assert not client.is_closed
    client.close()


def test_run_suite_closes_the_client_its_own_provider_made(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[httpx.Client] = []
    real = httpx.Client

    def spy(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(
            lambda req: (
                httpx.Response(200, json={"model": "m", "message": {"content": "a"}})
                if req.url.path == "/api/chat"
                else _ok(req)
            )
        )
        made.append(real(*args, **kwargs))  # type: ignore[arg-type]
        return made[-1]

    monkeypatch.setattr(httpx, "Client", spy)
    run_suite(_suite(tmp_path))
    assert made  # the provider and the environment lookup each made one
    assert all(c.is_closed for c in made)


def test_the_environment_lookup_closes_its_one_shot_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    made: list[httpx.Client] = []
    real = httpx.Client

    def spy(*args: object, **kwargs: object) -> httpx.Client:
        kwargs["transport"] = httpx.MockTransport(_ok)
        made.append(real(*args, **kwargs))  # type: ignore[arg-type]
        return made[-1]

    monkeypatch.setattr(httpx, "Client", spy)
    env = environment(_suite(tmp_path))
    assert env["ollama_version"] == "0"
    assert len(made) == 1 and made[0].is_closed
