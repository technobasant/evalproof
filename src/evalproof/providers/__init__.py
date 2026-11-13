"""Provider registry. A target string ``<provider>:<model>`` picks the implementation."""

from __future__ import annotations

import httpx

from evalproof.config import TargetConfig
from evalproof.providers.agent_cli import (
    CLI_PROVIDERS,
    AgentCLIProvider,
    ClaudeCodeProvider,
    CodexProvider,
)
from evalproof.providers.agent_cli_experimental import (
    EXPERIMENTAL_PROVIDERS,
    CopilotProvider,
    CursorProvider,
    OpenCodeProvider,
)
from evalproof.providers.anthropic import AnthropicProvider
from evalproof.providers.base import (
    ChatResponse,
    Message,
    Provider,
    ProviderError,
    TransientError,
    response_cost,
    with_retries,
)
from evalproof.providers.ollama import OllamaProvider
from evalproof.providers.openai import OpenAICompatibleProvider, OpenAIProvider
from evalproof.providers.profiles import PROFILES, Profile

__all__ = [
    "CLI_PROVIDERS",
    "EXPERIMENTAL_PROVIDERS",
    "PROFILES",
    "AgentCLIProvider",
    "AnthropicProvider",
    "ChatResponse",
    "ClaudeCodeProvider",
    "CodexProvider",
    "CopilotProvider",
    "CursorProvider",
    "Message",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "OpenAIProvider",
    "OpenCodeProvider",
    "Profile",
    "Provider",
    "ProviderError",
    "TransientError",
    "make_provider",
    "parse_target",
    "response_cost",
    "with_retries",
]

# The experimental CLI providers join the registry here, not in ``agent_cli``: that module is
# imported by the experimental one, so registering from there would be a circular import. Every
# user of ``CLI_PROVIDERS`` (``make_provider``, ``cli_facts``, ``doctor``) imports this package
# first, so the dict is complete before any of them reads it.
CLI_PROVIDERS.update(EXPERIMENTAL_PROVIDERS)

LLM_PROVIDERS = {"ollama", "anthropic", *PROFILES, *CLI_PROVIDERS}


def parse_target(target: str) -> tuple[str, str]:
    """``"ollama:qwen3:8b"`` → ``("ollama", "qwen3:8b")``. Model names may contain colons."""
    if ":" not in target:
        raise ProviderError(f"target {target!r} must look like '<provider>:<model>'")
    provider, model = target.split(":", 1)
    if not model:
        raise ProviderError(f"target {target!r} has no model")
    return provider, model


def make_provider(cfg: TargetConfig, client: httpx.Client | None = None) -> Provider:
    """Native adapters first (``ollama``, ``anthropic``), then the coding-agent CLIs
    (``claude-code``, ``codex``, and the experimental ``opencode``, ``cursor``, ``copilot``),
    then the OpenAI-compatible profiles."""
    provider, model = parse_target(cfg.target)
    if provider == "ollama":
        return OllamaProvider(
            model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            client=client,
            headers=cfg.headers,
        )
    if provider == "anthropic":
        return AnthropicProvider(
            model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            client=client,
            api_key_env=cfg.api_key_env,
            headers=cfg.headers,
        )
    if provider in CLI_PROVIDERS:
        return CLI_PROVIDERS[provider](
            model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            api_key_env=cfg.api_key_env,
            headers=cfg.headers,
        )
    if provider in PROFILES:
        return OpenAICompatibleProvider(
            PROFILES[provider],
            model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            client=client,
            api_key_env=cfg.api_key_env,
            headers=cfg.headers,
        )
    raise ProviderError(f"unknown provider {provider!r}; expected one of {sorted(LLM_PROVIDERS)}")
