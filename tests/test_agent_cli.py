"""The claude-code and codex providers, against fake binaries (no real CLI, no model call)."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

import pytest

from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import GenerationParams, TargetConfig
from evalproof.providers import (
    LLM_PROVIDERS,
    ClaudeCodeProvider,
    CodexProvider,
    Message,
    ProviderError,
    make_provider,
)
from evalproof.providers.agent_cli import NEUTRAL_SYSTEM, cli_facts, render_prompt
from tests.fakecli import (
    FakeCLI,
    claude_array,
    claude_failure,
    claude_ok,
    claude_result,
    codex_failure,
    codex_ok,
    install,
    reply,
)

USER = [Message(role="user", content="What is 2+2?")]
PARAMS = GenerationParams()
BYPASS = ("--dangerously-skip-permissions", "--yolo", "--full-auto", "danger-full-access")


def no_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    return install(
        tmp_path, monkeypatch, "claude", version="2.1.322 (Claude Code)", responses=[claude_ok()]
    )


@pytest.fixture
def codex(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    return install(
        tmp_path, monkeypatch, "codex", version="codex-cli 0.167.2", responses=[codex_ok()]
    )


def claude_provider(**kwargs: object) -> ClaudeCodeProvider:
    return ClaudeCodeProvider("haiku", sleep=no_sleep, **kwargs)  # type: ignore[arg-type]


def codex_provider(**kwargs: object) -> CodexProvider:
    return CodexProvider("gpt-6-luna", sleep=no_sleep, **kwargs)  # type: ignore[arg-type]


# -- command lines -------------------------------------------------------------------------


def test_claude_command_line_and_stdin(claude: FakeCLI) -> None:
    resp = claude_provider().chat(USER, PARAMS)
    call = claude.last()
    system_file = call["argv"][-1]
    assert call["argv"] == [
        "-p",
        "--output-format",
        "json",
        "--no-session-persistence",
        "--model",
        "haiku",
        "--setting-sources",
        "",
        "--tools",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--system-prompt-file",
        system_file,
    ]
    assert not any(flag in call["argv"] for flag in BYPASS)
    assert "What is 2+2?" not in " ".join(call["argv"])  # the prompt goes on stdin
    assert call["stdin"] == "What is 2+2?"
    assert call["env"]["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert (resp.text, resp.model) == ("OK", "claude-haiku-5-5")
    assert (resp.tokens_in, resp.tokens_out) == (2 + 692 + 10, 4)
    assert resp.latency_ms > 0 and not resp.cached


def test_codex_command_line_and_stdin(codex: FakeCLI) -> None:
    resp = codex_provider().chat(USER, PARAMS)
    call = codex.last()
    instructions = next(a for a in call["argv"] if a.startswith("model_instructions_file="))
    assert instructions.split("=", 1)[1].endswith("system.txt")
    assert call["argv"] == [
        "exec",
        "--json",
        "--ephemeral",
        "--skip-git-repo-check",
        "-s",
        "read-only",
        "--ignore-user-config",
        "--ignore-rules",
        "-m",
        "gpt-6-luna",
        "-c",
        "model_reasoning_effort=low",
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        instructions,
        "--disable",
        "shell_tool",
        "--disable",
        "plugins",
        "--disable",
        "apps",
        "--disable",
        "hooks",
        "--disable",
        "multi_agent",
        "-",
    ]
    assert not any(flag in call["argv"] for flag in BYPASS)
    assert call["stdin"] == "What is 2+2?"
    assert (resp.text, resp.model) == ("OK", "gpt-6-luna")  # codex reports no model: the request
    assert (resp.tokens_in, resp.tokens_out) == (16150, 104)


@pytest.mark.parametrize(
    ("think", "extra", "flag"),
    [
        (False, {}, "model_reasoning_effort=low"),
        (True, {}, "model_reasoning_effort=medium"),
        (False, {"reasoning_effort": "high"}, "model_reasoning_effort=high"),
        (True, {"reasoning_effort": "xhigh"}, "model_reasoning_effort=xhigh"),
        (None, {}, None),
    ],
)
def test_codex_reasoning_effort(
    codex: FakeCLI, think: bool | None, extra: dict[str, str], flag: str | None
) -> None:
    codex_provider().chat(USER, GenerationParams(think=think, extra=extra))
    effort = [a for a in codex.last()["argv"] if a.startswith("model_reasoning_effort=")]
    assert effort == ([flag] if flag else [])


def test_codex_rejects_an_unknown_reasoning_effort(codex: FakeCLI) -> None:
    with pytest.raises(ProviderError, match="reasoning_effort must be one of"):
        codex_provider().chat(USER, GenerationParams(extra={"reasoning_effort": "extreme"}))
    assert codex.calls() == []


def test_other_extra_keys_are_never_passed_on(claude: FakeCLI, codex: FakeCLI) -> None:
    params = GenerationParams(extra={"dangerously_skip_permissions": True, "args": ["--yolo"]})
    claude_provider().chat(USER, params)
    codex_provider().chat(USER, params)
    for fake in (claude, codex):
        argv = fake.last()["argv"]
        assert not any(flag in argv for flag in BYPASS)
        assert not any("dangerously" in a for a in argv)


# -- working directory ---------------------------------------------------------------------


def test_cwd_is_a_fresh_temp_dir_removed_afterwards(claude: FakeCLI) -> None:
    claude_provider().chat(USER, PARAMS)
    call = claude.last()
    cwd = Path(call["cwd"]).resolve()
    assert cwd.name.startswith("evalproof-")
    assert cwd != Path.cwd().resolve()
    assert cwd.parent == Path(tempfile.gettempdir()).resolve()
    assert call["cwd_files"] == ["system.txt"]  # nothing of the user's project is there
    assert not cwd.exists()


def test_cwd_is_removed_after_an_error_too(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_failure(404, "There's an issue with the selected model")],
    )
    with pytest.raises(ProviderError):
        claude_provider().chat(USER, PARAMS)
    assert not Path(fake.last()["cwd"]).exists()


# -- environment ---------------------------------------------------------------------------


def test_key_variables_are_stripped_by_default(
    claude: FakeCLI, codex: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "OPENAI_API_KEY",
        "CODEX_API_KEY",
        "CLAUDE_CODE_OAUTH_TOKEN",
    ):
        monkeypatch.setenv(var, "synthetic-" + var)
    claude_provider().chat(USER, PARAMS)
    codex_provider().chat(USER, PARAMS)
    seen = claude.last()["env"]
    assert seen["ANTHROPIC_API_KEY"] is None and seen["ANTHROPIC_AUTH_TOKEN"] is None
    # CI uses a subscription token this way, so it is left alone.
    assert seen["CLAUDE_CODE_OAUTH_TOKEN"] == "synthetic-CLAUDE_CODE_OAUTH_TOKEN"
    seen = codex.last()["env"]
    assert seen["OPENAI_API_KEY"] is None and seen["CODEX_API_KEY"] is None


def test_api_key_env_is_passed_under_the_cli_key_name(
    claude: FakeCLI, codex: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic-team-key")
    claude_provider(api_key_env="MY_TEAM_KEY").chat(USER, PARAMS)
    codex_provider(api_key_env="MY_TEAM_KEY").chat(USER, PARAMS)
    assert claude.last()["env"]["ANTHROPIC_API_KEY"] == "synthetic-team-key"
    assert codex.last()["env"]["CODEX_API_KEY"] == "synthetic-team-key"


def test_api_key_env_must_be_set(claude: FakeCLI) -> None:
    with pytest.raises(ProviderError, match="set MISSING_KEY"):
        claude_provider(api_key_env="MISSING_KEY")


def test_a_key_in_a_cli_message_is_masked(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic-team-key")
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_failure(401, "bad key synthetic-team-key")],
    )
    with pytest.raises(ProviderError) as err:
        claude_provider(api_key_env="MY_TEAM_KEY").chat(USER, PARAMS)
    assert "synthetic-team-key" not in str(err.value) and "***" in str(err.value)


def test_base_url_and_headers_do_not_apply() -> None:
    with pytest.raises(ProviderError, match="do not apply"):
        ClaudeCodeProvider("haiku", base_url="http://x")
    with pytest.raises(ProviderError, match="do not apply"):
        CodexProvider("m", headers={"a": "b"})


# -- messages ------------------------------------------------------------------------------


def test_system_message_goes_to_a_file(claude: FakeCLI, codex: FakeCLI) -> None:
    messages = [
        Message(role="system", content="Answer in French."),
        Message(role="user", content="Hello"),
    ]
    claude_provider().chat(messages, PARAMS)
    codex_provider().chat(messages, PARAMS)
    for fake in (claude, codex):
        call = fake.last()
        assert call["system"] == "Answer in French."
        assert call["stdin"] == "Hello"
        assert "French" not in " ".join(call["argv"])


def test_no_system_message_still_replaces_the_agent_prompt(claude: FakeCLI) -> None:
    claude_provider().chat(USER, PARAMS)
    assert claude.last()["system"] == NEUTRAL_SYSTEM


def test_multi_turn_transcript() -> None:
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello."),
        Message(role="user", content="Again?"),
    ]
    assert render_prompt(messages) == "User: Hi\n\nAssistant: Hello.\n\nUser: Again?"
    assert render_prompt([Message(role="user", content="only")]) == "only"
    with pytest.raises(ProviderError, match="no user message"):
        render_prompt([])


def test_multi_turn_goes_to_stdin(codex: FakeCLI) -> None:
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello."),
        Message(role="user", content="Again?"),
    ]
    codex_provider().chat(messages, PARAMS)
    assert codex.last()["stdin"] == "User: Hi\n\nAssistant: Hello.\n\nUser: Again?"


# -- parsing: claude -----------------------------------------------------------------------


def test_claude_single_object_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[reply(json.dumps(claude_result("single")))],
    )
    assert claude_provider().chat(USER, PARAMS).text == "single"


def test_claude_is_error_even_when_subtype_says_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_failure(404, "There's an issue with the selected model (nope).")],
    )
    with pytest.raises(ProviderError) as err:
        claude_provider().chat(USER, PARAMS)
    assert "selected model" in str(err.value) and "404" in str(err.value)


def test_claude_error_exit_with_no_result_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[reply("not json", stderr="boom", exit=1)],
    )
    with pytest.raises(ProviderError, match="boom"):
        claude_provider().chat(USER, PARAMS)


def test_claude_output_without_a_result_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[reply('[{"type":"system"}]', stderr="warned")],
    )
    with pytest.raises(ProviderError, match="no result event"):
        claude_provider().chat(USER, PARAMS)


def test_claude_resolved_model_is_the_one_with_most_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_ok(model="claude-haiku-5-5", extra_models={"claude-opus-5-5": 900})],
    )
    assert claude_provider().chat(USER, PARAMS).model == "claude-opus-5-5"


def test_claude_falls_back_to_the_requested_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(tmp_path, monkeypatch, "claude", version="v", responses=[claude_ok(model=None)])
    assert claude_provider().chat(USER, PARAMS).model == "haiku"


def test_claude_result_without_text(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    result = claude_result()
    del result["result"]
    install(tmp_path, monkeypatch, "claude", version="v", responses=[reply(claude_array(result))])
    with pytest.raises(ProviderError, match="no text"):
        claude_provider().chat(USER, PARAMS)


# -- parsing: codex ------------------------------------------------------------------------


def test_codex_jsonl_with_noise_lines(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[codex_ok("hello", noise=True)],
    )
    assert codex_provider().chat(USER, PARAMS).text == "hello"


def test_codex_turn_failed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[codex_failure("The 'nope' model is not supported with a ChatGPT account.")],
    )
    with pytest.raises(ProviderError, match="is not supported"):
        codex_provider().chat(USER, PARAMS)


def test_codex_command_execution_means_isolation_broke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events = [
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {"type": "command_execution", "command": "ls", "exit_code": 0},
        },
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}},
    ]
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[reply("\n".join(json.dumps(e) for e in events))],
    )
    with pytest.raises(ProviderError, match="isolation broken"):
        codex_provider().chat(USER, PARAMS)


def test_codex_failure_exit_without_events_and_without_a_message(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path, monkeypatch, "codex", version="v", responses=[reply("", stderr="oops", exit=2)]
    )
    with pytest.raises(ProviderError, match="oops"):
        codex_provider().chat(USER, PARAMS)
    install(tmp_path, monkeypatch, "codex", version="v", responses=[reply("{}\n")])
    with pytest.raises(ProviderError, match="no agent message"):
        codex_provider().chat(USER, PARAMS)


def test_error_messages_are_clipped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[codex_failure("x" * 2000)],
    )
    with pytest.raises(ProviderError) as err:
        codex_provider().chat(USER, PARAMS)
    assert len(str(err.value)) < 600


# -- process handling ----------------------------------------------------------------------


def _gone(pid: int, wait_s: float = 3.0) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_kills_the_whole_process_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[reply(sleep=30, spawn_child=True)],
    )
    provider = ClaudeCodeProvider("haiku", timeout_s=1.0, sleep=no_sleep)
    start = time.monotonic()
    with pytest.raises(ProviderError, match="timed out after 1s"):
        provider.chat(USER, PARAMS)
    assert time.monotonic() - start < 10
    call = fake.last()
    assert _gone(call["pid"]) and _gone(call["child_pid"])
    assert len(fake.calls()) == 1  # a timeout is not retried
    assert not Path(call["cwd"]).exists()


def test_missing_binary_names_the_install_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ProviderError, match="npm install -g @anthropic-ai/claude-code"):
        claude_provider().chat(USER, PARAMS)
    with pytest.raises(ProviderError, match="npm install -g @openai/codex"):
        codex_provider().chat(USER, PARAMS)
    assert cli_facts(TargetConfig(target="codex:m")) == {}  # best effort: no raise


# -- retries -------------------------------------------------------------------------------


def test_claude_retries_overloaded_but_not_model_not_found(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_failure(529, "Overloaded"), claude_ok("after retry")],
    )
    assert claude_provider().chat(USER, PARAMS).text == "after retry"
    assert len(fake.calls()) == 2

    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_failure(404, "model not found"), claude_ok()],
    )
    (fake.calls_path).unlink(missing_ok=True)
    with pytest.raises(ProviderError, match="model not found"):
        claude_provider().chat(USER, PARAMS)
    assert len(fake.calls()) == 1


def test_persistent_overload_gives_up(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="v",
        responses=[claude_failure(529, "Overloaded")],
    )
    with pytest.raises(ProviderError, match="giving up after 4 attempts"):
        claude_provider().chat(USER, PARAMS)
    assert len(fake.calls()) == 4


def test_codex_retries_a_rate_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[codex_failure("Rate limit reached, try later"), codex_ok("second")],
    )
    assert codex_provider().chat(USER, PARAMS).text == "second"
    assert len(fake.calls()) == 2


def test_codex_does_not_retry_other_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "codex",
        version="v",
        responses=[codex_failure("not supported"), codex_ok()],
    )
    with pytest.raises(ProviderError):
        codex_provider().chat(USER, PARAMS)
    assert len(fake.calls()) == 1


# -- fingerprint and cache -----------------------------------------------------------------


def test_fingerprint(claude: FakeCLI, codex: FakeCLI) -> None:
    assert claude_provider().fingerprint() == {
        "cli": "claude-code",
        "cli_version": "2.1.322 (Claude Code)",
        "isolation": "claude-code/1",
        "sampling": "cli-default",
    }
    assert codex_provider().fingerprint() == {
        "cli": "codex",
        "cli_version": "codex-cli 0.167.2",
        "isolation": "codex/1",
        "sampling": "cli-default",
    }
    assert cli_facts(TargetConfig(target="claude-code:haiku"))["cli_version"].startswith("2.1")
    assert cli_facts(TargetConfig(target="ollama:qwen3")) == {}
    assert not claude.calls() and not codex.calls()  # a version lookup is not a model call


def test_cache_salt_changes_with_the_cli_version(claude: FakeCLI) -> None:
    provider = claude_provider()
    before = provider.cache_salt(PARAMS)
    claude.set_version("2.1.400 (Claude Code)")
    after = provider.cache_salt(PARAMS)
    assert before != after and after["cli_version"] == "2.1.400 (Claude Code)"


def test_codex_cache_salt_includes_the_reasoning_effort(codex: FakeCLI) -> None:
    provider = codex_provider()
    low = provider.cache_salt(GenerationParams(think=False))
    medium = provider.cache_salt(GenerationParams(think=True))
    default = provider.cache_salt(GenerationParams(think=None))
    assert len({str(low), str(medium), str(default)}) == 3
    assert low["reasoning_effort"] == "low" and default["reasoning_effort"] == "default"


def test_a_version_change_is_a_cache_miss(claude: FakeCLI, tmp_path: Path) -> None:
    cached = CachedProvider(claude_provider(), ResponseCache(tmp_path / "cache"))
    first = cached.chat(USER, PARAMS)
    again = cached.chat(USER, PARAMS)
    assert not first.cached and again.cached and len(claude.calls()) == 1
    claude.set_version("2.1.400 (Claude Code)")
    cached.chat(USER, PARAMS)
    assert len(claude.calls()) == 2


# -- registry ------------------------------------------------------------------------------


def test_make_provider_builds_the_cli_providers(claude: FakeCLI) -> None:
    assert isinstance(make_provider(TargetConfig(target="claude-code:haiku")), ClaudeCodeProvider)
    codex_p = make_provider(TargetConfig(target="codex:gpt-6-luna"))
    assert isinstance(codex_p, CodexProvider) and codex_p.model == "gpt-6-luna"
    assert {"claude-code", "codex"} <= LLM_PROVIDERS
    with pytest.raises(ProviderError, match="do not apply"):
        make_provider(TargetConfig(target="codex:m", base_url="http://x"))
