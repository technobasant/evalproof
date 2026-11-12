"""The opencode, cursor and copilot providers, against fake binaries (no real CLI or model)."""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import evalproof.providers.agent_cli_experimental as experimental
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import GenerationParams, TargetConfig
from evalproof.doctor import installed_experimental, run_doctor
from evalproof.providers import (
    CLI_PROVIDERS,
    LLM_PROVIDERS,
    CopilotProvider,
    CursorProvider,
    Message,
    OpenCodeProvider,
    ProviderError,
    make_provider,
)
from evalproof.providers.agent_cli import NEUTRAL_SYSTEM, cli_facts
from evalproof.providers.agent_cli_experimental import (
    ARGV_PROMPT_LIMIT,
    ExperimentalCLIProvider,
    frame_prompt,
)
from evalproof.providers.base import TransientError
from tests.fakecli import FakeCLI, claude_auth, claude_ok, codex_ok, install, reply
from tests.fakecli_experimental import (
    ENV_KEYS,
    cursor_about,
    cursor_ok,
    cursor_result,
    install_experimental,
    only_fakes_on_path,
    opencode_events,
    opencode_failure,
    opencode_ok,
    respond,
)

USER = [Message(role="user", content="What is 2+2?")]
PARAMS = GenerationParams()
BYPASS = {
    "--dangerously-skip-permissions",
    "--yolo",
    "--full-auto",
    "danger-full-access",
    "--allow-all",
    "--allow-all-tools",
    "--allow-all-paths",
    "--allow-all-urls",
    "--allow-tool",
    "--force",
    "-f",
}

SPECS: dict[str, tuple[type[ExperimentalCLIProvider], str, str]] = {
    "opencode": (OpenCodeProvider, "ollama/qwen3:8b", "1.4.0"),
    "cursor": (CursorProvider, "auto", "2026.02.27-e7d2ef6"),
    "copilot": (CopilotProvider, "gpt-5", "GitHub Copilot CLI 1.0.4"),
}
# The CLI's own key variable, and the variables of other tools that must be left alone.
KEY_VARS = {
    "opencode": ("OPENCODE_API_KEY", ["ANTHROPIC_API_KEY"]),
    "cursor": ("CURSOR_API_KEY", []),
    "copilot": ("COPILOT_GITHUB_TOKEN", ["GH_TOKEN", "GITHUB_TOKEN"]),
}


def no_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_KEYS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(experimental, "_warned", set())  # the warning is once per process


def ok_reply(name: str, text: str = "OK") -> dict[str, Any]:
    if name == "opencode":
        return opencode_ok(text)
    if name == "cursor":
        return cursor_ok(text)
    return reply(text + "\n")


@dataclass
class Case:
    name: str
    fake: FakeCLI
    cls: type[ExperimentalCLIProvider]
    model: str
    version: str

    def provider(self, **kwargs: Any) -> ExperimentalCLIProvider:
        return self.cls(self.model, sleep=no_sleep, **kwargs)

    def set_replies(self, *responses: dict[str, Any]) -> None:
        respond(self.fake, *responses)


def make_case(name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    cls, model, version = SPECS[name]
    fake = install_experimental(
        tmp_path,
        monkeypatch,
        cls.binary,
        version=version,
        responses=[ok_reply(name)],
        status_args=["about"] if name == "cursor" else None,
        status_stdout=cursor_about("someone@example.invalid"),
    )
    return Case(name, fake, cls, model, version)


@pytest.fixture(params=list(SPECS))
def cli(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return make_case(request.param, tmp_path, monkeypatch)


@pytest.fixture
def opencode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return make_case("opencode", tmp_path, monkeypatch)


@pytest.fixture
def cursor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return make_case("cursor", tmp_path, monkeypatch)


@pytest.fixture
def copilot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Case:
    return make_case("copilot", tmp_path, monkeypatch)


def same_path(a: str, b: str) -> bool:
    return Path(a).resolve() == Path(b).resolve()  # macOS: /var is /private/var


# -- the warning -----------------------------------------------------------------------------


def test_the_first_use_prints_one_warning_per_provider(
    opencode: Case, cursor: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    provider = opencode.provider()
    provider.chat(USER, PARAMS)
    provider.chat(USER, PARAMS)
    cursor.provider().chat(USER, PARAMS)
    err = capsys.readouterr().err
    assert err.count("evalproof: opencode provider is experimental") == 1
    assert err.count("evalproof: cursor provider is experimental") == 1
    assert "copilot" not in err


def test_the_warning_is_once_even_from_threads(capsys: pytest.CaptureFixture[str]) -> None:
    threads = [
        threading.Thread(target=experimental.warn_experimental, args=("copilot",)) for _ in range(8)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert capsys.readouterr().err == "evalproof: copilot provider is experimental\n"


def test_looking_up_the_fingerprint_does_not_warn(
    cli: Case, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.provider().fingerprint()
    assert capsys.readouterr().err == ""


# -- command lines -----------------------------------------------------------------------------


def test_opencode_command_line_and_stdin(opencode: Case) -> None:
    resp = opencode.provider().chat(USER, PARAMS)
    call = opencode.fake.last()
    assert call["argv"] == [
        "run",
        "--standalone",
        "--archived",
        "--format",
        "json",
        "--agent",
        "judge",
        "-m",
        "ollama/qwen3:8b",
    ]
    assert not BYPASS & set(call["argv"])
    assert "What is 2+2?" not in " ".join(call["argv"])  # the prompt goes on stdin
    assert call["stdin"] == "What is 2+2?"
    env = call["env"]
    assert env["OPENCODE_DISABLE_PROJECT_CONFIG"] == "1"
    assert env["OPENCODE_CONFIG_PROJECT_DISABLE"] == "1"
    assert env["OPENCODE_DISABLE_CLAUDE_CODE"] == "1"
    assert env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] == "1"
    assert call["config_dir_files"] == []  # the config dir is empty, so no global config is read
    assert not Path(env["OPENCODE_CONFIG_DIR"]).exists()  # and removed with the call
    assert (resp.text, resp.model) == ("OK", "ollama/qwen3:8b")
    assert (resp.tokens_in, resp.tokens_out) == (478, 8)
    assert resp.latency_ms > 0 and not resp.cached


def test_cursor_command_line_and_prompt(cursor: Case) -> None:
    resp = cursor.provider().chat(USER, PARAMS)
    call = cursor.fake.last()
    argv = call["argv"]
    assert argv[:-1] == [
        "-p",
        "--output-format",
        "json",
        "--mode",
        "ask",
        "--trust",
        "--model",
        "auto",
        "--workspace",
        argv[-2],
    ]
    assert same_path(argv[-2], call["cwd"])  # the workspace is the empty directory
    assert argv[-1] == frame_prompt(NEUTRAL_SYSTEM, "What is 2+2?")
    assert not BYPASS & set(argv)
    assert "--api-key" not in argv and "--system-prompt" not in argv
    assert call["stdin"] == ""  # no open pipe: the prompt is an argument
    assert (resp.text, resp.model) == ("OK", "auto")
    assert (resp.tokens_in, resp.tokens_out) == (120 + 30 + 4, 7)


def test_copilot_command_line_and_prompt(copilot: Case) -> None:
    resp = copilot.provider().chat(USER, PARAMS)
    call = copilot.fake.last()
    assert call["argv"] == [
        "-s",
        "--no-auto-update",
        "--no-ask-user",
        "--available-tools=",
        "--model",
        "gpt-5",
        "-p",
        frame_prompt(NEUTRAL_SYSTEM, "What is 2+2?"),
    ]
    assert not BYPASS & set(call["argv"])
    assert not any(a.startswith(("--allow", "--yolo")) for a in call["argv"])
    assert call["stdin"] == ""
    assert (resp.text, resp.model) == ("OK", "gpt-5")
    assert (resp.tokens_in, resp.tokens_out) == (0, 0)  # text output has no usage


def test_no_command_line_ever_has_a_bypass_flag(cli: Case) -> None:
    cli.provider().chat(USER, PARAMS)
    assert not BYPASS & set(cli.fake.last()["argv"])
    doctor_view = cli.provider().command(cli.cls.binary, "<tmp>/system.txt", PARAMS)
    assert not BYPASS & set(doctor_view)


# -- the system text -----------------------------------------------------------------------------


def test_opencode_carries_the_system_text_in_its_agent_config(opencode: Case) -> None:
    system = Message(role="system", content='Réponds en français — merci. {not} "json"')
    opencode.provider().chat([system, *USER], PARAMS)
    call = opencode.fake.last()
    config = json.loads(call["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config == {
        "agent": {
            "judge": {
                "mode": "primary",
                "prompt": system.content,
                "tools": {"*": False},
                "permission": {"*": "deny"},
            }
        }
    }
    assert call["stdin"] == "What is 2+2?"  # the system text is not repeated on stdin


def test_opencode_without_a_system_message_uses_the_neutral_one(opencode: Case) -> None:
    opencode.provider().chat(USER, PARAMS)
    config = json.loads(opencode.fake.last()["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["agent"]["judge"]["prompt"] == NEUTRAL_SYSTEM


@pytest.mark.parametrize("name", ["cursor", "copilot"])
def test_cursor_and_copilot_put_the_system_text_in_front_of_the_prompt(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(name, tmp_path, monkeypatch)
    system = Message(role="system", content="Reply in French.")
    user = Message(role="user", content="--help")  # must not become a flag
    case.provider().chat([system, user], PARAMS)
    call = case.fake.last()
    sent = call["argv"][-1]
    assert sent == frame_prompt("Reply in French.", "--help")
    assert not sent.startswith("-") and "Reply in French." in sent
    assert call["stdin"] == ""
    assert "--help" not in call["argv"][:-1]


def test_frame_prompt_never_starts_with_a_dash() -> None:
    assert not frame_prompt("-x", "-y").startswith("-")


# -- size guards -----------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["cursor", "copilot"])
def test_a_prompt_too_long_for_argv_is_refused_not_sent(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = make_case(name, tmp_path, monkeypatch)
    big = [Message(role="user", content="x" * (ARGV_PROMPT_LIMIT + 1))]
    with pytest.raises(ProviderError, match=f"over {ARGV_PROMPT_LIMIT} bytes"):
        case.provider().chat(big, PARAMS)
    assert case.fake.calls() == []  # refused up front, and not retried
    just_under = [Message(role="user", content="x" * (ARGV_PROMPT_LIMIT - 1000))]
    assert case.provider().chat(just_under, PARAMS).text == "OK"


def test_opencode_takes_a_long_prompt_on_stdin_but_not_a_huge_system_text(opencode: Case) -> None:
    long_input = [Message(role="user", content="y" * 300_000)]
    assert opencode.provider().chat(long_input, PARAMS).text == "OK"
    assert len(opencode.fake.last()["stdin"]) == 300_000
    modest = Message(role="system", content="s" * 90_000)
    assert opencode.provider().chat([modest, *USER], PARAMS).text == "OK"
    huge = Message(role="system", content="s" * (ARGV_PROMPT_LIMIT + 1))
    opencode.set_replies(opencode_ok())
    with pytest.raises(ProviderError, match="system prompt is over"):
        opencode.provider().chat([huge, *USER], PARAMS)
    assert opencode.fake.calls() == []


# -- working directory ---------------------------------------------------------------------------


def test_the_working_directory_is_fresh_empty_and_removed(cli: Case) -> None:
    provider = cli.provider()
    provider.chat(USER, PARAMS)
    provider.chat(USER, PARAMS)
    first, second = cli.fake.calls()
    assert first["cwd_files"] == [] and second["cwd_files"] == []
    assert first["cwd"] != second["cwd"]
    assert not same_path(first["cwd"], str(Path.cwd()))
    assert not Path(first["cwd"]).exists() and not Path(second["cwd"]).exists()
    assert not Path(first["cwd"]).parent.exists()  # nothing of the temp root is left


def test_the_working_directory_is_removed_after_a_failure(cli: Case) -> None:
    cli.set_replies(reply("", stderr="boom: something broke", exit=1))
    with pytest.raises(ProviderError, match="boom: something broke"):
        cli.provider().chat(USER, PARAMS)
    call = cli.fake.last()
    assert not Path(call["cwd"]).exists() and not Path(call["cwd"]).parent.exists()
    if cli.name == "opencode":
        assert not Path(call["env"]["OPENCODE_CONFIG_DIR"]).exists()


# -- keys ----------------------------------------------------------------------------------------


def test_the_cli_own_key_variable_is_stripped_and_other_tools_are_left_alone(
    cli: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_var, keep = KEY_VARS[cli.name]
    monkeypatch.setenv(key_var, "synthetic-own-key")
    for var in keep:
        monkeypatch.setenv(var, f"synthetic-{var}")
    cli.provider().chat(USER, PARAMS)
    seen = cli.fake.last()["env"]
    assert seen[key_var] is None
    for var in keep:  # a key for the model's own provider, or a general GitHub token
        assert seen[var] == f"synthetic-{var}"


def test_api_key_env_is_passed_under_the_cli_key_name(
    cli: Case, monkeypatch: pytest.MonkeyPatch
) -> None:
    key_var, _ = KEY_VARS[cli.name]
    assert cli.cls.key_var == key_var
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic-team-key")
    monkeypatch.setenv(key_var, "synthetic-ambient-key")  # the named key wins over the ambient one
    cli.provider(api_key_env="MY_TEAM_KEY").chat(USER, PARAMS)
    call = cli.fake.last()
    assert call["env"][key_var] == "synthetic-team-key"
    assert "synthetic-team-key" not in json.dumps(call["argv"])  # never on a command line
    assert "synthetic-team-key" not in (call["env"]["OPENCODE_CONFIG_CONTENT"] or "")


def test_api_key_env_must_be_set(cli: Case) -> None:
    with pytest.raises(ProviderError, match="set MISSING_KEY"):
        cli.provider(api_key_env="MISSING_KEY")


def test_a_key_in_a_cli_message_is_masked(cli: Case, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic-team-key")
    cli.set_replies(reply("", stderr="bad key synthetic-team-key", exit=1))
    with pytest.raises(ProviderError) as err:
        cli.provider(api_key_env="MY_TEAM_KEY").chat(USER, PARAMS)
    assert "synthetic-team-key" not in str(err.value) and "***" in str(err.value)


def test_base_url_and_headers_do_not_apply() -> None:
    for cls, model, _ in SPECS.values():
        with pytest.raises(ProviderError, match="do not apply"):
            cls(model, base_url="http://x")
        with pytest.raises(ProviderError, match="do not apply"):
            cls(model, headers={"a": "b"})


# -- process handling ----------------------------------------------------------------------------


def _gone(pid: int, wait_s: float = 3.0) -> bool:
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


def test_timeout_kills_the_whole_process_group(cli: Case) -> None:
    cli.set_replies(reply(sleep=30, spawn_child=True))
    provider = cli.provider(timeout_s=1.0)
    start = time.monotonic()
    with pytest.raises(ProviderError, match="timed out after 1s"):
        provider.chat(USER, PARAMS)
    assert time.monotonic() - start < 10
    call = cli.fake.last()
    assert _gone(call["pid"]) and _gone(call["child_pid"])
    assert len(cli.fake.calls()) == 1  # a timeout is not retried
    assert not Path(call["cwd"]).exists() and not Path(call["cwd"]).parent.exists()


def test_missing_binary_names_the_install_command(
    cli: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    with pytest.raises(ProviderError, match=re.escape(cli.cls.install_hint)):
        cli.provider().chat(USER, PARAMS)
    assert cli_facts(TargetConfig(target=f"{cli.name}:{cli.model}")) == {}  # best effort


def test_a_plain_failure_is_not_retried(cli: Case) -> None:
    cli.set_replies(reply("", stderr="quota exceeded for this account", exit=1), ok_reply(cli.name))
    with pytest.raises(ProviderError, match="quota exceeded") as err:
        cli.provider().chat(USER, PARAMS)
    assert not isinstance(err.value, TransientError)
    assert len(cli.fake.calls()) == 1


# -- opencode output -----------------------------------------------------------------------------


def test_opencode_tokens_add_cache_and_reasoning_and_steps_add_up(opencode: Case) -> None:
    opencode.set_replies(
        opencode_ok(
            "A", input_tokens=478, output_tokens=8, reasoning=20, cache_read=100, cache_write=5
        )
    )
    resp = opencode.provider().chat(USER, PARAMS)
    assert (resp.tokens_in, resp.tokens_out) == (478 + 100 + 5, 8 + 20)
    two_steps = opencode_events("A", input_tokens=10, output_tokens=1)
    two_steps += opencode_events("B", input_tokens=20, output_tokens=2)
    opencode.set_replies(reply(two_steps))
    resp = opencode.provider().chat(USER, PARAMS)
    assert (resp.text, resp.tokens_in, resp.tokens_out) == ("AB", 30, 3)


def test_opencode_ignores_lines_that_are_not_events(opencode: Case) -> None:
    noisy = "Loading...\n[1, 2]\n{not json\n" + opencode_events("fine") + "\ntrailing text\n"
    opencode.set_replies(reply(noisy))
    assert opencode.provider().chat(USER, PARAMS).text == "fine"


def test_opencode_error_events(opencode: Case) -> None:
    opencode.set_replies(
        opencode_failure("provider.no-route", "no route for ollama/nope"), opencode_ok()
    )
    with pytest.raises(ProviderError, match="no route for ollama/nope") as err:
        opencode.provider().chat(USER, PARAMS)
    assert not isinstance(err.value, TransientError)
    assert len(opencode.fake.calls()) == 1  # not retried

    # An error event with exit status 0 is still an error.
    opencode.set_replies(opencode_failure("unknown", "the model said no", exit=0))
    with pytest.raises(ProviderError, match="the model said no"):
        opencode.provider().chat(USER, PARAMS)


def test_opencode_does_not_retry_a_usage_limit_but_retries_an_overload(opencode: Case) -> None:
    opencode.set_replies(
        opencode_failure("provider.free-limit", "Free usage limit reached", status=429),
        opencode_ok(),
    )
    with pytest.raises(ProviderError, match=r"Free usage limit reached \(HTTP 429\)") as err:
        opencode.provider().chat(USER, PARAMS)
    assert "giving up" not in str(err.value) and len(opencode.fake.calls()) == 1

    opencode.set_replies(
        opencode_failure("overloaded", "Overloaded", status=529), opencode_ok("after retry")
    )
    assert opencode.provider().chat(USER, PARAMS).text == "after retry"
    assert len(opencode.fake.calls()) == 2

    opencode.set_replies(opencode_failure("rate-limit", "slow down", status=429))
    with pytest.raises(ProviderError, match="giving up after 4 attempts"):
        opencode.provider().chat(USER, PARAMS)
    assert len(opencode.fake.calls()) == 4


def test_opencode_tool_use_means_the_isolation_broke(opencode: Case) -> None:
    tool = {"type": "tool_use", "part": {"type": "tool", "tool": "bash", "state": {}}}
    opencode.set_replies(reply(opencode_events("done", extra=[tool])), opencode_ok())
    with pytest.raises(ProviderError, match="opencode ran a tool; isolation broken"):
        opencode.provider().chat(USER, PARAMS)
    assert len(opencode.fake.calls()) == 1


def test_opencode_without_text_or_with_a_bad_exit(opencode: Case) -> None:
    opencode.set_replies(reply('{"type":"step_start"}\n'))
    with pytest.raises(ProviderError, match="no text in the output"):
        opencode.provider().chat(USER, PARAMS)
    opencode.set_replies(reply("just words\n"))
    with pytest.raises(ProviderError, match="no text in the output: just words"):
        opencode.provider().chat(USER, PARAMS)
    opencode.set_replies(reply("", stderr="crashed hard", exit=2))
    with pytest.raises(ProviderError, match="crashed hard"):
        opencode.provider().chat(USER, PARAMS)
    opencode.set_replies(reply("", exit=3))
    with pytest.raises(ProviderError, match="exited with status 3"):
        opencode.provider().chat(USER, PARAMS)


# -- cursor output -------------------------------------------------------------------------------


def test_cursor_result_without_usage_has_zero_tokens(cursor: Case) -> None:
    cursor.set_replies(cursor_ok("hello", with_usage=False))
    resp = cursor.provider().chat(USER, PARAMS)
    assert (resp.text, resp.tokens_in, resp.tokens_out) == ("hello", 0, 0)


def test_cursor_accepts_an_event_array_and_terminal_noise(cursor: Case) -> None:
    events = [
        {"type": "system", "subtype": "init"},
        json.loads(cursor_result("from an array")),
    ]
    cursor.set_replies(reply(json.dumps(events)))
    assert cursor.provider().chat(USER, PARAMS).text == "from an array"
    noisy = "\x1b[2K\x1b[Gthinking...\n" + cursor_result("after noise") + "\x1b[0m\n"
    cursor.set_replies(reply(noisy))
    assert cursor.provider().chat(USER, PARAMS).text == "after noise"


def test_cursor_is_error_result_is_a_failure_that_is_not_retried(cursor: Case) -> None:
    cursor.set_replies(
        reply(cursor_result("model not available for this plan", is_error=True)), cursor_ok()
    )
    with pytest.raises(ProviderError, match="model not available for this plan") as err:
        cursor.provider().chat(USER, PARAMS)
    assert not isinstance(err.value, TransientError) and len(cursor.fake.calls()) == 1


def test_cursor_auth_required_is_an_error_with_its_own_message(cursor: Case) -> None:
    text = "Error: Authentication required. Run `cursor-agent login` or set CURSOR_API_KEY."
    cursor.set_replies(reply("", stderr=text + "\n", exit=1), cursor_ok())
    with pytest.raises(ProviderError, match="Authentication required"):
        cursor.provider().chat(USER, PARAMS)
    assert len(cursor.fake.calls()) == 1


@pytest.mark.parametrize(
    "stdout",
    [
        '{"unexpected": "shape"}\n',
        "just some text, not JSON\n",
        "",
        '{"type": "result", "result": {"nested": 1}}\n',
        "[1, 2, 3]\n",
        '{"type": "result", "result": "cut off mid-object\n',
    ],
)
def test_cursor_unexpected_output_says_what_was_expected(cursor: Case, stdout: str) -> None:
    cursor.set_replies(reply(stdout))
    with pytest.raises(ProviderError) as err:
        cursor.provider().chat(USER, PARAMS)
    message = str(err.value)
    assert "unexpected output from cursor-agent" in message
    assert '"result"' in message and "--output-format json" in message
    assert len(message) < 600
    assert len(cursor.fake.calls()) == 1


def test_cursor_unexpected_output_shows_only_the_start_of_a_long_one(cursor: Case) -> None:
    cursor.set_replies(reply("z" * 5000))
    with pytest.raises(ProviderError) as err:
        cursor.provider().chat(USER, PARAMS)
    assert "z" * 200 in str(err.value) and "z" * 201 not in str(err.value)


# -- copilot output --------------------------------------------------------------------------------


def test_copilot_text_is_stripped_and_has_no_token_counts(copilot: Case) -> None:
    copilot.set_replies(reply("\n  The answer is 4.\n\n"))
    resp = copilot.provider().chat(USER, PARAMS)
    assert resp.text == "The answer is 4."
    assert (resp.tokens_in, resp.tokens_out) == (0, 0)


def test_copilot_empty_reply_and_bad_exit(copilot: Case) -> None:
    copilot.set_replies(reply("\n", stderr="nothing to say"))
    with pytest.raises(ProviderError, match="empty reply: nothing to say"):
        copilot.provider().chat(USER, PARAMS)
    copilot.set_replies(reply("", exit=1))
    with pytest.raises(ProviderError, match="exited with status 1"):
        copilot.provider().chat(USER, PARAMS)
    assert len(copilot.fake.calls()) == 1


# -- fingerprint, cache, registry ------------------------------------------------------------------


def test_fingerprint(cli: Case) -> None:
    assert cli.provider().fingerprint() == {
        "cli": cli.name,
        "cli_version": cli.version,
        "isolation": f"{cli.name}/0",
        "sampling": "cli-default",
    }
    facts = cli_facts(TargetConfig(target=f"{cli.name}:{cli.model}"))
    assert facts["cli_version"] == cli.version and facts["isolation"] == f"{cli.name}/0"
    assert cli.fake.calls() == []  # a version lookup is not a model call


def test_cache_salt_changes_with_the_cli_version(cli: Case) -> None:
    provider = cli.provider()
    before = provider.cache_salt(PARAMS)
    cli.fake.set_version("9.9.9")
    after = provider.cache_salt(PARAMS)
    assert before != after and after["cli_version"] == "9.9.9"


def test_a_version_change_is_a_cache_miss(cli: Case, tmp_path: Path) -> None:
    cached = CachedProvider(cli.provider(), ResponseCache(tmp_path / "cache"))
    first = cached.chat(USER, PARAMS)
    again = cached.chat(USER, PARAMS)
    assert not first.cached and again.cached and len(cli.fake.calls()) == 1
    cli.fake.set_version("9.9.9")
    cached.chat(USER, PARAMS)
    assert len(cli.fake.calls()) == 2


def test_make_provider_builds_the_experimental_providers(cli: Case) -> None:
    made = make_provider(TargetConfig(target=f"{cli.name}:{cli.model}"))
    assert isinstance(made, cli.cls) and made.model == cli.model
    assert {"opencode", "cursor", "copilot"} <= LLM_PROVIDERS
    assert {"opencode", "cursor", "copilot"} <= set(CLI_PROVIDERS)
    with pytest.raises(ProviderError, match="do not apply"):
        make_provider(TargetConfig(target=f"{cli.name}:m", base_url="http://x"))


# -- evalproof doctor ------------------------------------------------------------------------------


@pytest.fixture
def stable_clis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_ok()],
        auth_stdout=claude_auth(),
    )
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="codex-cli 0.167.2",
        responses=[codex_ok()],
        auth_stdout="Logged in using ChatGPT\n",
    )


def doctor(suite: str | None = None) -> tuple[int, str]:
    out = io.StringIO()
    return run_doctor(suite, out), out.getvalue()


def write_suite(tmp_path: Path, system: str, judge: str | None, extra: str = "") -> str:
    judge_block = f"judge: {{target: '{judge}'{extra}}}\n" if judge else ""
    text = (
        f"name: t\ndataset: d.jsonl\nsystem: {{target: '{system}'}}\n{judge_block}"
        "scorers: [{name: m, type: exact_match}]\n"
    )
    path = tmp_path / "suite.yaml"
    path.write_text(text, "utf-8")
    return str(path)


def section(text: str, header: str) -> str:
    """The lines of one target's block: from its header to the next header."""
    lines = text.splitlines()
    start = lines.index(header)
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.startswith("  "):
            break
        block.append(line)
    return "\n".join(block)


def test_doctor_without_a_suite_lists_the_experimental_clis_that_are_installed(
    stable_clis: None, opencode: Case, cursor: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)  # no copilot on PATH
    assert installed_experimental() == ["opencode:<model>", "cursor:<model>"]
    code, text = doctor()
    assert code == 0 and text.rstrip().endswith("ok")
    assert "\nopencode\n" in f"\n{text}" and "\ncursor\n" in f"\n{text}"
    assert "copilot" not in text
    block = section(text, "opencode")
    assert str(opencode.fake.path) in block and "version: 1.4.0" in block
    assert "auth: not checked (OpenCode logs in per model provider)" in block
    assert "isolation: opencode/0" in block
    assert "--archived" in block and "--standalone" in block
    assert "experimental provider" in block
    assert "OPENCODE_DISABLE_PROJECT_CONFIG=1" in block
    block = section(text, "cursor")
    assert "version: 2026.02.27-e7d2ef6" in block
    assert "auth: logged in (account not shown)" in block
    assert "isolation: cursor/0" in block and "--mode ask" in block
    assert "someone@example.invalid" not in text  # never an account
    # A free check: nothing was asked of a model.
    assert opencode.fake.calls() == [] and cursor.fake.calls() == []


def test_doctor_lists_copilot_with_auth_not_checked(
    stable_clis: None, copilot: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    code, text = doctor()
    assert code == 0
    block = section(text, "copilot")
    assert "auth: not checked" in block and "isolation: copilot/0" in block
    assert "--available-tools=" in block and "--no-ask-user" in block
    assert "opencode" not in text and "cursor" not in text
    assert copilot.fake.calls() == []


def test_doctor_lists_nothing_extra_when_no_experimental_cli_is_installed(
    stable_clis: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    assert installed_experimental() == []
    code, text = doctor()
    assert code == 0
    assert "opencode" not in text and "cursor" not in text and "copilot" not in text


def test_doctor_reports_cursor_when_not_logged_in(
    stable_clis: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_experimental(
        tmp_path,
        monkeypatch,
        "cursor-agent",
        version="2026.02.27-e7d2ef6",
        responses=[cursor_ok()],
        status_args=["about"],
        status_stdout=cursor_about("Not logged in"),
    )
    only_fakes_on_path(tmp_path, monkeypatch)
    code, text = doctor()
    assert code == 1 and "not ready" in text
    assert "auth: not logged in; run `cursor-agent login` or set CURSOR_API_KEY" in text


def test_doctor_does_not_guess_when_about_has_no_account_line(
    stable_clis: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_experimental(
        tmp_path,
        monkeypatch,
        "cursor-agent",
        version="2026.02.27-e7d2ef6",
        responses=[cursor_ok()],
        status_args=["about"],
        status_stdout="About Cursor CLI\nCLI Version 1\n",
    )
    only_fakes_on_path(tmp_path, monkeypatch)
    code, text = doctor()
    assert code == 0
    assert "auth: not checked (no account line in `about`, exit 0)" in text


def test_doctor_reads_experimental_targets_from_a_suite(
    stable_clis: None,
    opencode: Case,
    copilot: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    suite = write_suite(tmp_path, "copilot:gpt-5", "opencode:ollama/qwen3:8b")
    code, text = doctor(suite)
    assert code == 0
    assert "copilot:gpt-5 (system)" in text and "opencode:ollama/qwen3:8b (judge)" in text
    assert "claude-code" not in text  # a suite names what to check


def test_doctor_names_the_key_variable_for_an_experimental_api_key_env(
    stable_clis: None,
    cursor: Case,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic")
    suite = write_suite(tmp_path, "python:x:y", "cursor:auto", ", api_key_env: MY_TEAM_KEY")
    code, text = doctor(suite)
    assert code == 0
    assert "$MY_TEAM_KEY is passed to the CLI as CURSOR_API_KEY" in text
    assert "auth: not checked (key from api_key_env is passed as CURSOR_API_KEY)" in text
    assert "synthetic" not in text


def test_doctor_notes_a_key_variable_it_will_strip(
    stable_clis: None, copilot: Case, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "synthetic-token")
    _, text = doctor()
    assert "COPILOT_GITHUB_TOKEN is set and is removed" in text
    assert "synthetic-token" not in text


def test_doctor_fails_on_a_suite_target_whose_binary_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, text = doctor(write_suite(tmp_path, "copilot:gpt-5", None))
    assert code == 1 and "npm install -g @github/copilot" in text and "not ready" in text
