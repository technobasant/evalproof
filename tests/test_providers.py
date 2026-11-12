from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
import pytest

from evalproof.config import GenerationParams, TargetConfig
from evalproof.models import Case
from evalproof.providers import (
    LLM_PROVIDERS,
    PROFILES,
    AnthropicProvider,
    Message,
    OllamaProvider,
    OpenAIProvider,
    ProviderError,
    make_provider,
    parse_target,
    with_retries,
)
from evalproof.systems import LLMSystem
from tests.conftest import mock_client

MSGS = [Message(role="system", content="be brief"), Message(role="user", content="hi")]


@pytest.fixture(autouse=True)
def _no_ambient_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's real keys must not decide whether these tests pass."""
    for env in {p.api_key_env for p in PROFILES.values() if p.api_key_env} | {"ANTHROPIC_API_KEY"}:
        monkeypatch.delenv(env, raising=False)


def test_parse_target() -> None:
    assert parse_target("ollama:qwen3:8b") == ("ollama", "qwen3:8b")
    for bad in ("ollama", "ollama:"):
        with pytest.raises(ProviderError):
            parse_target(bad)


def test_ollama_request_and_response() -> None:
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.url.path == "/api/chat"
        return httpx.Response(
            200,
            json={
                "model": "qwen3:8b",
                "message": {"role": "assistant", "content": "hello"},
                "prompt_eval_count": 12,
                "eval_count": 3,
            },
        )

    p = OllamaProvider("qwen3:8b", client=mock_client(handler))
    resp = p.chat(MSGS, GenerationParams(seed=7, max_tokens=64, think=False))
    assert resp.text == "hello" and resp.tokens_in == 12 and resp.tokens_out == 3
    assert seen["stream"] is False and seen["think"] is False
    assert seen["options"] == {"temperature": 0.0, "num_predict": 64, "seed": 7}


def test_ollama_bad_shape() -> None:
    p = OllamaProvider("m", client=mock_client(lambda r: httpx.Response(200, json={"x": 1})))
    with pytest.raises(ProviderError):
        p.chat(MSGS, GenerationParams())


def test_anthropic_pulls_system_prompt(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.headers["x-api-key"] == "k"
        assert req.headers["anthropic-version"] == "2023-06-01"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "content": [{"type": "text", "text": "ok"}],
                "usage": {"input_tokens": 5, "output_tokens": 2},
            },
        )

    resp = AnthropicProvider("m", client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert resp.text == "ok" and resp.tokens_in == 5
    assert seen["system"] == "be brief"
    assert seen["messages"] == [{"role": "user", "content": "hi"}]


def test_openai_request(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        assert req.headers["authorization"] == "Bearer k"
        return httpx.Response(
            200,
            json={
                "model": "m",
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 1},
            },
        )

    resp = OpenAIProvider("m", client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert resp.text == "ok" and resp.tokens_out == 1
    # Hosted models get only what the suite set; reasoning models reject a temperature.
    assert seen == {
        "model": "m",
        "messages": [m.model_dump() for m in MSGS],
        "max_completion_tokens": 512,
    }


def test_openai_no_choices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    p = OpenAIProvider("m", client=mock_client(lambda r: httpx.Response(200, json={})))
    with pytest.raises(ProviderError):
        p.chat(MSGS, GenerationParams())


def test_missing_keys() -> None:
    with pytest.raises(
        ProviderError, match="anthropic: set ANTHROPIC_API_KEY to use this provider"
    ):
        AnthropicProvider("m")
    with pytest.raises(ProviderError, match="openai: set OPENAI_API_KEY to use this provider"):
        OpenAIProvider("m")


def test_retries_then_succeeds() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json={"message": {"content": "ok"}})

    p = OllamaProvider("m", client=mock_client(handler))
    sleeps: list[float] = []
    import evalproof.providers.ollama as mod

    orig = mod.with_retries
    mod.with_retries = lambda fn: orig(fn, sleep=sleeps.append)  # type: ignore[assignment]
    try:
        assert p.chat(MSGS, GenerationParams()).text == "ok"
    finally:
        mod.with_retries = orig
    assert calls["n"] == 3 and len(sleeps) == 2 and sleeps[1] > sleeps[0] * 0.5


def test_non_retryable_status_raises() -> None:
    p = OllamaProvider("m", client=mock_client(lambda r: httpx.Response(400, text="bad model")))
    with pytest.raises(ProviderError, match="400"):
        p.chat(MSGS, GenerationParams())


def test_transport_errors_give_up() -> None:
    def boom() -> int:
        raise httpx.ConnectError("down")

    with pytest.raises(ProviderError, match="giving up"):
        with_retries(boom, attempts=3, sleep=lambda s: None)


def test_make_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    assert make_provider(TargetConfig(target="ollama:m")).name == "ollama"
    assert make_provider(TargetConfig(target="anthropic:m")).name == "anthropic"
    assert make_provider(TargetConfig(target="openai:m")).name == "openai"


def test_unknown_provider_lists_every_valid_name() -> None:
    with pytest.raises(ProviderError, match="unknown provider 'nope'") as err:
        make_provider(TargetConfig(target="nope:m"))
    assert all(name in str(err.value) for name in LLM_PROVIDERS)
    assert {"ollama", "anthropic", "openai", "groq", "openai-compatible"} <= LLM_PROVIDERS


OK_REPLY = {
    "model": "m",
    "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
}


def _row(
    name: str,
    url: str,
    *,
    suite_url: str | None = None,
    env: str | None = None,
    auth: str = "authorization",
    max_field: str = "max_tokens",
    seed_field: str = "seed",
    local: bool = False,
) -> tuple[object, ...]:
    return name, url, env, auth, max_field, seed_field, suite_url or "", local


# The shipped table written out by hand, so a changed URL or field name fails a test.
# name, request URL, key variable (None: a key is optional and none is set), auth header,
# max-token field, seed field, base_url the suite must give, local.
PROFILE_CASES = [
    _row("openai", "https://api.openai.com/v1/chat/completions", env="OPENAI_API_KEY",
         max_field="max_completion_tokens"),
    _row("azure", "https://res.openai.azure.com/openai/v1/chat/completions",
         suite_url="https://res.openai.azure.com/openai/v1/", env="AZURE_OPENAI_API_KEY",
         auth="api-key", max_field="max_completion_tokens"),
    _row("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
         env="GEMINI_API_KEY"),
    _row("mistral", "https://api.mistral.ai/v1/chat/completions", env="MISTRAL_API_KEY",
         seed_field="random_seed"),
    _row("groq", "https://api.groq.com/openai/v1/chat/completions", env="GROQ_API_KEY",
         max_field="max_completion_tokens"),
    _row("together", "https://api.together.ai/v1/chat/completions", env="TOGETHER_API_KEY"),
    _row("fireworks", "https://api.fireworks.ai/inference/v1/chat/completions",
         env="FIREWORKS_API_KEY"),
    _row("deepseek", "https://api.deepseek.com/chat/completions", env="DEEPSEEK_API_KEY"),
    _row("xai", "https://api.x.ai/v1/chat/completions", env="XAI_API_KEY"),
    _row("openrouter", "https://openrouter.ai/api/v1/chat/completions", env="OPENROUTER_API_KEY"),
    _row("cerebras", "https://api.cerebras.ai/v1/chat/completions", env="CEREBRAS_API_KEY",
         max_field="max_completion_tokens"),
    _row("openai-compatible", "https://gw.example.com/v1/chat/completions",
         suite_url="https://gw.example.com/v1"),
    _row("vllm", "http://127.0.0.1:8000/v1/chat/completions", local=True),
    _row("lmstudio", "http://127.0.0.1:1234/v1/chat/completions", local=True),
    _row("llamacpp", "http://127.0.0.1:9931/v1/chat/completions", local=True),
]  # fmt: skip


def test_every_profile_is_covered() -> None:
    assert {c[0] for c in PROFILE_CASES} == set(PROFILES)
    assert all(p.docs_url.startswith("https://") for p in PROFILES.values())


@pytest.mark.parametrize(
    ("name", "url", "env", "auth", "max_field", "seed_field", "suite_url", "local"),
    PROFILE_CASES,
    ids=[str(c[0]) for c in PROFILE_CASES],
)
def test_profile_request_shape(
    name: str,
    url: str,
    env: str | None,
    auth: str,
    max_field: str,
    seed_field: str,
    suite_url: str,
    local: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if env:
        monkeypatch.setenv(env, "k")
    requests: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        requests.append(req)
        return httpx.Response(200, json=OK_REPLY)

    cfg = TargetConfig(target=f"{name}:vendor/model:1", base_url=suite_url or None)
    provider = make_provider(cfg, client=mock_client(handler))
    assert provider.name == name and provider.model == "vendor/model:1"
    resp = provider.chat(MSGS, GenerationParams())
    provider.chat(MSGS, GenerationParams(temperature=0.2, seed=3))
    default, explicit = requests

    assert str(default.url) == url
    assert resp.text == "ok" and resp.tokens_in == 4 and resp.tokens_out == 1
    if env is None:  # a key is optional here, and none was set
        assert "authorization" not in default.headers and "api-key" not in default.headers
    elif auth == "authorization":
        assert default.headers["authorization"] == "Bearer k" and "api-key" not in default.headers
    else:
        assert default.headers[auth] == "k" and "authorization" not in default.headers

    body = json.loads(default.content)
    assert body["model"] == "vendor/model:1" and body["messages"][-1]["content"] == "hi"
    assert body[max_field] == 512
    assert not ({"max_tokens", "max_completion_tokens"} - {max_field}) & body.keys()
    if local:  # reproducible by default: the defaults are sent
        assert body["temperature"] == 0.0 and body[seed_field] == 0
    else:  # hosted: only what the suite set
        assert not {"temperature", "seed", "random_seed"} & body.keys()
    set_body = json.loads(explicit.content)
    assert set_body["temperature"] == 0.2 and set_body[seed_field] == 3


def test_extra_params_reach_the_body(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    seen: dict[str, object] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=OK_REPLY)

    cfg = TargetConfig(
        target="openrouter:anthropic/claude-sonnet-4.5",
        params=GenerationParams(extra={"reasoning": {"effort": "low"}}),
    )
    make_provider(cfg, client=mock_client(handler)).chat(MSGS, cfg.params)
    assert seen["model"] == "anthropic/claude-sonnet-4.5" and seen["reasoning"] == {"effort": "low"}


@pytest.mark.parametrize("name", [n for n, p in PROFILES.items() if p.key_required])
def test_missing_key_names_the_variable(name: str) -> None:
    env = PROFILES[name].api_key_env
    cfg = TargetConfig(target=f"{name}:m", base_url="https://x.example.com/v1")
    with pytest.raises(ProviderError, match=f"^{name}: set {env} to use this provider$"):
        make_provider(cfg)


@pytest.mark.parametrize("name", ["azure", "openai-compatible"])
def test_profiles_without_a_default_url_need_base_url(name: str) -> None:
    with pytest.raises(ProviderError, match=f"{name}: set base_url"):
        make_provider(TargetConfig(target=f"{name}:m"))


def test_api_key_env_overrides_the_profile_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEAM_GROQ_KEY", "team")
    monkeypatch.setenv("GROQ_API_KEY", "default")
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=OK_REPLY)

    for target, base_url in [("groq:m", None), ("openai-compatible:m", "http://gw/v1")]:
        cfg = TargetConfig(target=target, base_url=base_url, api_key_env="TEAM_GROQ_KEY")
        make_provider(cfg, client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert [r.headers["authorization"] for r in seen] == ["Bearer team", "Bearer team"]


def test_api_key_env_that_is_not_set_is_an_error_even_for_optional_keys() -> None:
    cfg = TargetConfig(target="vllm:m", api_key_env="NOT_SET_ANYWHERE")
    with pytest.raises(ProviderError, match="vllm: set NOT_SET_ANYWHERE"):
        make_provider(cfg)


def test_local_profile_sends_a_key_when_one_is_set(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "k")
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json=OK_REPLY)

    make_provider(TargetConfig(target="vllm:m"), client=mock_client(handler)).chat(
        MSGS, GenerationParams()
    )
    assert seen[0].headers["authorization"] == "Bearer k"


def test_headers_are_merged_into_every_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    seen: dict[str, httpx.Request] = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen[req.url.path] = req
        if req.url.path == "/api/chat":
            return httpx.Response(200, json={"message": {"content": "ok"}})
        if req.url.path == "/v1/messages":
            return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})
        return httpx.Response(200, json=OK_REPLY)

    hdrs = {"X-Route": "eu", "Anthropic-Version": "2026-10-21"}
    for target in ("ollama:m", "anthropic:m", "openai:m"):
        cfg = TargetConfig(target=target, headers=hdrs)
        make_provider(cfg, client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert len(seen) == 3
    assert all(r.headers["x-route"] == "eu" for r in seen.values())
    # The suite's version replaces the default instead of riding along beside it.
    assert seen["/v1/messages"].headers.get_list("anthropic-version") == ["2026-10-21"]
    assert seen["/v1/messages"].headers["x-api-key"] == "k"


def test_anthropic_api_key_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_GATEWAY_KEY", "gw")
    seen: list[httpx.Request] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return httpx.Response(200, json={"content": [{"type": "text", "text": "ok"}]})

    cfg = TargetConfig(target="anthropic:m", api_key_env="CLAUDE_GATEWAY_KEY")
    make_provider(cfg, client=mock_client(handler)).chat(MSGS, GenerationParams())
    assert seen[0].headers["x-api-key"] == "gw"


def _chat_reply(finish: str, content: object) -> httpx.Response:
    reply = {"choices": [{"message": {"content": content}, "finish_reason": finish}]}
    return httpx.Response(200, json=reply)


def test_openai_compatible_finish_reasons(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def chat(finish: str, content: object) -> str:
        client = mock_client(lambda r: _chat_reply(finish, content))
        return (
            make_provider(TargetConfig(target="groq:m"), client=client)
            .chat(MSGS, GenerationParams())
            .text
        )

    with pytest.raises(ProviderError, match="groq: max tokens reached before any text"):
        chat("length", "")
    with pytest.raises(ProviderError, match=r"raise params\.max_tokens"):
        chat("length", None)
    with pytest.raises(ProviderError, match="groq: response blocked by the content filter"):
        chat("content_filter", "")
    assert chat("length", "partial answer") == "partial answer"  # truncated is still an answer
    assert chat("stop", "done") == "done"
    # Some servers return typed parts; the text parts are the answer.
    parts = [
        {"type": "thinking", "thinking": "hm"},
        {"type": "text", "text": "a"},
        "?",
        {"text": "b"},
    ]
    assert chat("stop", parts) == "ab"
    assert chat("stop", 42) == ""


def test_openai_compatible_error_names_the_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "k")
    client = mock_client(lambda r: httpx.Response(200, json={}))
    with pytest.raises(ProviderError, match=r"^together: no choices"):
        make_provider(TargetConfig(target="together:m"), client=client).chat(
            MSGS, GenerationParams()
        )


def test_ollama_still_sends_temperature_and_seed_by_default() -> None:
    options = OllamaProvider("m").payload(MSGS, GenerationParams())["options"]
    assert options == {"temperature": 0.0, "num_predict": 512, "seed": 0}


def _anthropic_payload(params: GenerationParams) -> dict[str, object]:
    return AnthropicProvider("m", api_key="k").payload(MSGS, params)


def test_anthropic_sends_temperature_only_when_set() -> None:
    default = _anthropic_payload(GenerationParams())
    assert default == {
        "model": "m",
        "max_tokens": 512,
        "system": "be brief",
        "messages": [{"role": "user", "content": "hi"}],
    }
    assert _anthropic_payload(GenerationParams(temperature=0.3))["temperature"] == 0.3
    # A suite that sets temperature: 0 explicitly means it, and gets it.
    assert _anthropic_payload(GenerationParams(temperature=0.0))["temperature"] == 0.0
    # Anthropic has no seed, whether or not the suite sets one.
    assert "seed" not in _anthropic_payload(GenerationParams(seed=7))


def test_anthropic_extra_carries_thinking_and_effort() -> None:
    extra = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}}
    body = _anthropic_payload(GenerationParams(max_tokens=4096, extra=extra))
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {"effort": "low"} and body["max_tokens"] == 4096
    assert "temperature" not in body


def _anthropic_chat(reply: Mapping[str, object]) -> str:
    client = mock_client(lambda r: httpx.Response(200, json=reply))
    return AnthropicProvider("m", api_key="k", client=client).chat(MSGS, GenerationParams()).text


def test_anthropic_refusal_is_an_error_with_its_category() -> None:
    with pytest.raises(ProviderError, match=r"^anthropic: refused \(cyber\)$"):
        _anthropic_chat(
            {
                "content": [{"type": "text", "text": "I can't"}],
                "stop_reason": "refusal",
                "stop_details": {"type": "refusal", "category": "cyber"},
            }
        )
    with pytest.raises(ProviderError, match=r"^anthropic: refused \(no category\)$"):
        _anthropic_chat({"content": [], "stop_reason": "refusal"})
    with pytest.raises(ProviderError, match=r"refused \(no category\)"):
        _anthropic_chat(
            {"content": [], "stop_reason": "refusal", "stop_details": {"category": None}}
        )


def test_anthropic_max_tokens_before_any_text_is_an_error() -> None:
    thinking = [{"type": "thinking", "thinking": "hmm"}]
    with pytest.raises(ProviderError, match="max_tokens reached before any text; raise params"):
        _anthropic_chat({"content": thinking, "stop_reason": "max_tokens"})
    # Cut off mid-answer is still an answer.
    cut = {"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}
    assert _anthropic_chat(cut) == "partial"
    assert (
        _anthropic_chat({"content": [{"type": "text", "text": "x"}], "stop_reason": "end_turn"})
        == "x"
    )


def test_provider_errors_become_case_errors() -> None:
    refusal = {"content": [], "stop_reason": "refusal", "stop_details": {"category": "bio"}}
    client = mock_client(lambda r: httpx.Response(200, json=refusal))
    system = LLMSystem(
        TargetConfig(target="anthropic:m"), AnthropicProvider("m", api_key="k", client=client)
    )
    out = system.run(Case(id="c", input="q"))
    assert out.output == "" and out.error == "anthropic: refused (bio)"
