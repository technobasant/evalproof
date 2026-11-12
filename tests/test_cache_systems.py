from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from evalproof.cache import CachedProvider, ResponseCache, cache_key
from evalproof.config import ConfigError, GenerationParams, Pricing, TargetConfig
from evalproof.models import Case
from evalproof.providers import ChatResponse, Message, ProviderError
from evalproof.systems import HttpSystem, LLMSystem, PythonSystem, make_system, render
from tests.conftest import ScriptedProvider, mock_client

MSGS = [Message(role="user", content="q")]


def test_cache_key_depends_on_everything_that_changes_a_reply() -> None:
    base = cache_key("ollama", "m", None, MSGS, GenerationParams())
    assert base == cache_key("ollama", "m", None, MSGS, GenerationParams())
    assert base != cache_key("ollama", "m2", None, MSGS, GenerationParams())
    assert base != cache_key("ollama", "m", None, MSGS, GenerationParams(seed=1))
    assert base != cache_key("ollama", "m", "http://x", MSGS, GenerationParams())
    assert base != cache_key(
        "ollama", "m", None, [Message(role="user", content="r")], GenerationParams()
    )


def test_cache_keys_of_existing_ollama_calls_are_pinned() -> None:
    """Computed on the release before provider profiles. If these change, every cached local
    run misses: change GenerationParams defaults or the key's inputs only on purpose."""
    msgs = [Message(role="system", content="be brief"), Message(role="user", content="hi")]
    assert GenerationParams().model_dump(mode="json") == {
        "temperature": 0.0,
        "seed": 0,
        "max_tokens": 512,
        "think": False,
        "extra": {},
    }
    assert (
        cache_key("ollama", "qwen3:8b", None, msgs, GenerationParams())
        == "08d0b7fc706e8781e04a18eecd99fc97d56614a35c65b9985a5cf1439c2f7037"
    )
    tuned = GenerationParams(seed=7, max_tokens=64, think=True, extra={"top_p": 0.9})
    assert (
        cache_key("ollama", "qwen3:8b", "http://127.0.0.1:11434", msgs, tuned)
        == "4b9256f81e2adceb39d49211bb7b81274f6b0d35b82ed028bae8566094826404"
    )


def test_cache_round_trip_and_torn_file(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    resp = ChatResponse(text="t", model="m", tokens_in=3, tokens_out=1, latency_ms=42.0)
    cache.put("ab" + "0" * 62, resp)
    hit = cache.get("ab" + "0" * 62)
    assert hit is not None and hit.cached and hit.latency_ms == 42.0 and hit.text == "t"
    assert cache.get("cd" + "0" * 62) is None
    (tmp_path / "ef").mkdir()
    (tmp_path / "ef" / ("ef" + "0" * 62 + ".json")).write_text("{not json", "utf-8")
    assert cache.get("ef" + "0" * 62) is None
    assert cache.stats()["entries"] == 2


def test_cached_provider_hits_and_misses(tmp_path: Path) -> None:
    inner = ScriptedProvider(["a", "b"])
    cp = CachedProvider(inner, ResponseCache(tmp_path))
    first = cp.chat(MSGS, GenerationParams())
    second = cp.chat(MSGS, GenerationParams())
    assert first.text == second.text == "a"
    assert not first.cached and second.cached
    assert (cp.hits, cp.misses, len(inner.calls)) == (1, 1, 1)


def test_render() -> None:
    assert render("Q: {question}", {"question": "why"}) == "Q: why"
    with pytest.raises(ConfigError, match="unknown fields"):
        render("{nope}", {"question": "x"})


def test_llm_system_builds_messages_and_cost() -> None:
    provider = ScriptedProvider(["answer"])
    cfg = TargetConfig(
        target="ollama:m",
        system_prompt="facts",
        prompt_template="Q: {question}",
        pricing=Pricing(input_per_mtok=1.0, output_per_mtok=2.0),
    )
    out = LLMSystem(cfg, provider).run(Case(id="c", input={"question": "why"}))
    assert out.output == "answer"
    assert out.cost_usd == pytest.approx((10 * 1 + 5 * 2) / 1_000_000)
    roles = [m.role for m in provider.calls[0]]
    assert roles == ["system", "user"] and provider.calls[0][1].content == "Q: why"


def test_llm_system_records_provider_errors() -> None:
    class Failing(ScriptedProvider):
        def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
            raise ProviderError("down")

    out = LLMSystem(TargetConfig(target="ollama:m"), Failing([])).run(Case(id="c", input="q"))
    assert out.error == "down" and out.output == ""


@pytest.fixture
def sut_module(tmp_path: Path) -> Path:
    (tmp_path / "toy_sut.py").write_text(
        textwrap.dedent(
            """
            def echo(x):
                return f"echo {x}"
            def rag(x):
                return {"output": "a", "retrieved": ["d1", 2]}
            def broken(x):
                raise RuntimeError("nope")
            def bad_type(x):
                return 42
            NOT_CALLABLE = 1
            """
        ),
        "utf-8",
    )
    yield tmp_path
    sys.modules.pop("toy_sut", None)


def test_python_system(sut_module: Path) -> None:
    case = Case(id="c", input="hi")
    assert (
        PythonSystem("python:toy_sut:echo", search_path=str(sut_module)).run(case).output
        == "echo hi"
    )
    rag = PythonSystem("python:toy_sut:rag", search_path=str(sut_module)).run(case)
    assert rag.retrieved == ["d1", "2"]
    assert "RuntimeError" in (PythonSystem("python:toy_sut:broken").run(case).error or "")
    assert "TypeError" in (PythonSystem("python:toy_sut:bad_type").run(case).error or "")
    with pytest.raises(ConfigError):
        PythonSystem("python:toy_sut:NOT_CALLABLE")
    with pytest.raises(ConfigError):
        PythonSystem("python:toy_sut")


def test_http_system_json_and_text() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        if body["input"] == "text":
            return httpx.Response(200, text="plain")
        return httpx.Response(200, json={"output": "ok", "retrieved": ["d"]})

    sys_ = HttpSystem(TargetConfig(target="http://sut/answer"), client=mock_client(handler))
    assert sys_.run(Case(id="a", input="json")).retrieved == ["d"]
    assert sys_.run(Case(id="b", input="text")).output == "plain"
    failing = HttpSystem(
        TargetConfig(target="http://sut/x"),
        client=mock_client(lambda r: httpx.Response(404, text="gone")),
    )
    assert failing.run(Case(id="c", input="q")).error


def test_make_system_kinds(tmp_path: Path, sut_module: Path) -> None:
    cache = ResponseCache(tmp_path / "c")
    assert isinstance(make_system(TargetConfig(target="ollama:m"), cache=cache), LLMSystem)
    assert isinstance(make_system(TargetConfig(target="https://x")), HttpSystem)
    assert isinstance(
        make_system(TargetConfig(target="python:toy_sut:echo"), search_path=str(sut_module)),
        PythonSystem,
    )
