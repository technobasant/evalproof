"""Live check of the CLI providers: one tiny real call per installed CLI.

Skipped unless ``EVALPROOF_LIVE=1``, because it spends a little of your subscription quota and
needs ``claude`` and ``codex`` installed and logged in. Run it by hand before a release:

    EVALPROOF_LIVE=1 uv run python -m pytest tests/test_live_agent_cli.py -v

Override the models with ``EVALPROOF_LIVE_CLAUDE_MODEL`` (default ``haiku``) and
``EVALPROOF_LIVE_CODEX_MODEL`` (default ``gpt-6-luna``, a model a ChatGPT plan could use when
the spec was written).
"""

from __future__ import annotations

import os
import shutil

import pytest

from evalproof.config import GenerationParams
from evalproof.providers import ClaudeCodeProvider, CodexProvider, Message
from evalproof.providers.agent_cli import AgentCLIProvider

pytestmark = pytest.mark.skipif(
    os.environ.get("EVALPROOF_LIVE") != "1", reason="set EVALPROOF_LIVE=1 to call the real CLIs"
)

PROMPT = [Message(role="user", content="Reply with exactly the one word: PONG")]


def check(provider: AgentCLIProvider) -> None:
    resp = provider.chat(PROMPT, GenerationParams())
    assert "PONG" in resp.text.upper()
    assert resp.tokens_out > 0 and resp.latency_ms > 0
    assert resp.model  # claude reports the resolved id; codex echoes the request
    fingerprint = provider.fingerprint()
    assert fingerprint["cli_version"] and fingerprint["isolation"]


@pytest.mark.skipif(shutil.which("claude") is None, reason="claude is not installed")
def test_live_claude_code() -> None:
    check(ClaudeCodeProvider(os.environ.get("EVALPROOF_LIVE_CLAUDE_MODEL", "haiku")))


@pytest.mark.skipif(shutil.which("codex") is None, reason="codex is not installed")
def test_live_codex() -> None:
    check(CodexProvider(os.environ.get("EVALPROOF_LIVE_CODEX_MODEL", "gpt-6-luna")))
