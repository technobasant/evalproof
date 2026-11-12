"""OpenAI-compatible services as data: one HTTP shape, one row per provider.

Most hosted and local model servers accept the OpenAI Chat Completions request, so evalproof
speaks it once and describes each provider by the few things that differ: the base URL, where
the key lives, how it is sent, what the token-limit field is called, and whether the server
is local. A profile is a row, not code, so adding a provider is a reviewable one-line change.
Every base URL and key variable below was read from the provider's own documentation
(``docs_url``) on 2026-11-12; none came from memory.

Not here, by design: AWS Bedrock and Google Vertex AI sign requests with cloud credentials.
Reach them through a gateway's OpenAI-compatible endpoint with ``openai-compatible``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Profile:
    """How to talk to one OpenAI-compatible service.

    ``base_url`` of ``None`` means the suite must supply it (Azure is per-resource).
    ``api_key_env`` of ``None`` means no key is read unless the suite names a variable with
    ``api_key_env``. ``auth_header`` of ``"authorization"`` sends ``Bearer <key>``; any other
    name sends the raw key under that header, as Azure's ``api-key`` does.
    ``local`` servers get temperature and seed on every call, so a local run is repeatable
    by default; hosted ones get them only when the suite sets them, because hosted reasoning
    models reject a non-default temperature.
    """

    name: str
    base_url: str | None
    api_key_env: str | None
    key_required: bool
    docs_url: str
    auth_header: str = "authorization"
    max_tokens_field: str = "max_tokens"
    seed_field: str = "seed"  # Mistral calls it random_seed
    local: bool = False


_PROFILES = [
    Profile(
        "openai",
        "https://api.openai.com/v1",
        "OPENAI_API_KEY",
        True,
        "https://developers.openai.com/api/reference/overview",
        max_tokens_field="max_completion_tokens",  # max_tokens is rejected by o-series and gpt-5
    ),
    Profile(
        "azure",
        None,  # https://<resource>.openai.azure.com/openai/v1
        "AZURE_OPENAI_API_KEY",
        True,
        "https://learn.microsoft.com/en-us/azure/foundry/openai/api-version-lifecycle",
        auth_header="api-key",
        max_tokens_field="max_completion_tokens",
    ),
    Profile(
        "gemini",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "GEMINI_API_KEY",
        True,
        "https://aistudio.google.com/docs/openai",
    ),
    Profile(
        "mistral",
        "https://api.mistral.ai/v1",
        "MISTRAL_API_KEY",
        True,
        "https://docs.mistral.ai/api/",
        seed_field="random_seed",
    ),
    Profile(
        "groq",
        "https://api.groq.com/openai/v1",
        "GROQ_API_KEY",
        True,
        "https://console.groq.com/docs/openai",
        max_tokens_field="max_completion_tokens",  # max_tokens is documented as deprecated
    ),
    Profile(
        "together",
        "https://api.together.ai/v1",
        "TOGETHER_API_KEY",
        True,
        "https://docs.together.ai/docs/openai-api-compatibility",
    ),
    Profile(
        "fireworks",
        "https://api.fireworks.ai/inference/v1",
        "FIREWORKS_API_KEY",
        True,
        "https://docs.fireworks.ai/getting-started/quickstart",
    ),
    Profile(
        "deepseek",
        "https://api.deepseek.com",  # no /v1: the docs give the bare host
        "DEEPSEEK_API_KEY",
        True,
        "https://api-docs.deepseek.com/",
    ),
    Profile(
        "xai",
        "https://api.x.ai/v1",
        "XAI_API_KEY",
        True,
        "https://docs.x.ai/docs/api-reference",
    ),
    Profile(
        "openrouter",
        "https://openrouter.ai/api/v1",
        "OPENROUTER_API_KEY",
        True,
        "https://openrouter.ai/docs/quickstart",
    ),
    Profile(
        "cerebras",
        "https://api.cerebras.ai/v1",
        "CEREBRAS_API_KEY",
        True,
        "https://inference-docs.cerebras.ai/resources/openai",
        max_tokens_field="max_completion_tokens",  # max_tokens is only an alias for it
    ),
    Profile(
        "vllm",
        "http://127.0.0.1:8000/v1",
        "VLLM_API_KEY",
        False,
        "https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/",
        local=True,
    ),
    Profile(
        "lmstudio",
        "http://127.0.0.1:1234/v1",
        None,  # no key unless authentication is switched on; then set api_key_env
        False,
        "https://lmstudio.ai/docs/developer/openai-compat",
        local=True,
    ),
    Profile(
        "llamacpp",
        # llama-server's default port moved from 8080 to 9931 on 2026-10-09; for an older
        # build set base_url (or start it with --port 8080).
        "http://127.0.0.1:9931/v1",
        "LLAMA_API_KEY",
        False,
        "https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md",
        local=True,
    ),
    Profile(
        # Any other server or gateway. Not treated as local: it may front a hosted
        # reasoning model, so set temperature and seed in the suite when you want them sent.
        "openai-compatible",
        None,
        None,
        False,
        "https://developers.openai.com/api/reference/overview",
    ),
]

PROFILES: dict[str, Profile] = {p.name: p for p in _PROFILES}
