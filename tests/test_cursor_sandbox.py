"""The macOS sandbox around cursor-agent and ``evalproof doctor --cursor-canary``.

No real ``cursor-agent`` and no ``sandbox-exec`` run here: the CLI is a fake binary (see
``fakecli_experimental``), and the platform and the sandbox binary are patched.
"""

from __future__ import annotations

import io
import json
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

import evalproof.providers.agent_cli_experimental as experimental
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.cli import main
from evalproof.config import GenerationParams, TargetConfig
from evalproof.doctor import (
    CANARY_MODEL,
    CANARY_SKILL_PATH,
    experimental_warnings,
    parse_canary_reply,
    run_cursor_canary,
    run_doctor,
)
from evalproof.providers.agent_cli import cli_facts
from evalproof.providers.agent_cli_experimental import (
    CURSOR_SANDBOX_DENY,
    CursorProvider,
    cursor_home,
    cursor_sandbox_mode,
    cursor_sandbox_profile,
    cursor_sandbox_state,
    sandbox_exec_available,
)
from evalproof.providers.base import Message, ProviderError
from tests.fakecli import FakeCLI, reply
from tests.fakecli_experimental import (
    ENV_KEYS,
    cursor_about,
    cursor_ok,
    cursor_stream,
    install_experimental,
    only_fakes_on_path,
    respond,
)

PARAMS = GenerationParams()
USER = [Message(role="user", content="What is 2+2?")]
VERIFIED_PROFILE = (
    '(version 1)(allow default)(deny file-read* (subpath "/H/.claude") (subpath "/H/.agents") '
    '(subpath "/H/.codex") (subpath "/H/.gemini") (subpath "/H/.cursor/skills") '
    '(subpath "/H/.cursor/skills-cursor") (subpath "/H/.cursor/rules") '
    '(subpath "/H/.cursor/plugins") (subpath "/H/.cursor/agents"))'
)


def no_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ENV_KEYS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("EVALPROOF_CURSOR_SANDBOX", raising=False)
    monkeypatch.setattr(experimental, "_warned", set())


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    return install_experimental(
        tmp_path,
        monkeypatch,
        "cursor-agent",
        version="2026.11.11-synthetic",
        responses=[cursor_ok()],
        status_args=["about"],
        status_stdout=cursor_about("someone@example.invalid"),
    )


def sandbox_available(monkeypatch: pytest.MonkeyPatch, available: bool) -> None:
    monkeypatch.setattr(experimental, "sandbox_exec_available", lambda: available)


def provider(**kwargs: Any) -> CursorProvider:
    return CursorProvider("auto", sleep=no_sleep, **kwargs)


# -- the profile -----------------------------------------------------------------------------------


def test_the_profile_is_the_verified_one() -> None:
    assert cursor_sandbox_profile("/H") == VERIFIED_PROFILE


def test_the_profile_escapes_quotes_and_backslashes_in_the_home_path() -> None:
    profile = cursor_sandbox_profile('/tmp/a"b\\c')
    assert '(subpath "/tmp/a\\"b\\\\c/.claude")' in profile
    assert '(subpath "/tmp/a\\"b\\\\c/.cursor/skills-cursor")' in profile
    # No unescaped quote inside a path: every quote is a literal delimiter or escaped.
    assert profile.count('\\"') == len(CURSOR_SANDBOX_DENY)


def test_the_deny_list_is_relative_and_names_no_one() -> None:
    assert len(CURSOR_SANDBOX_DENY) == 9 and len(set(CURSOR_SANDBOX_DENY)) == 9
    assert all(not p.startswith("/") and ".." not in p for p in CURSOR_SANDBOX_DENY)
    assert {".claude", ".agents", ".codex", ".gemini"} <= set(CURSOR_SANDBOX_DENY)
    assert "basant" not in cursor_sandbox_profile("/H")


def test_the_home_is_resolved_through_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real)
    monkeypatch.setenv("HOME", str(link))
    assert cursor_home() == str(real.resolve())


# -- the switch ------------------------------------------------------------------------------------


def test_the_mode_is_on_by_default_and_off_when_asked(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cursor_sandbox_mode() == "on"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "")
    assert cursor_sandbox_mode() == "on"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", " off ")
    assert cursor_sandbox_mode() == "off"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "on")
    assert cursor_sandbox_mode() == "on"


def test_any_other_value_is_an_error_naming_the_allowed_ones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "maybe")
    with pytest.raises(ProviderError, match=r"EVALPROOF_CURSOR_SANDBOX must be one of on, off, "):
        cursor_sandbox_mode()
    with pytest.raises(ProviderError, match="got 'maybe'"):
        provider()


def test_the_sandbox_needs_macos_and_sandbox_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    present = tmp_path / "sandbox-exec"
    present.write_text("#!/bin/sh\n", "utf-8")
    monkeypatch.setattr(experimental, "SANDBOX_EXEC", str(present))
    monkeypatch.setattr(sys, "platform", "darwin")
    assert sandbox_exec_available()
    monkeypatch.setattr(sys, "platform", "linux")
    assert not sandbox_exec_available()
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(experimental, "SANDBOX_EXEC", str(tmp_path / "missing"))
    assert not sandbox_exec_available()


def test_the_state_follows_the_variable_and_the_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_available(monkeypatch, True)
    assert cursor_sandbox_state() == "sandboxed"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    assert cursor_sandbox_state() == "off"
    sandbox_available(monkeypatch, False)
    assert cursor_sandbox_state() == "unavailable"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "on")
    assert cursor_sandbox_state() == "unavailable"


# -- the command and the isolation tag -------------------------------------------------------------


def test_a_sandboxed_command_is_the_old_one_behind_sandbox_exec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_available(monkeypatch, True)
    wrapped = provider().build_command("cursor-agent", "/w", PARAMS)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    plain = provider().build_command("cursor-agent", "/w", PARAMS)
    assert plain[0] == "cursor-agent" and "--mode" in plain
    assert wrapped[:2] == [experimental.SANDBOX_EXEC, "-p"]
    assert wrapped[2] == cursor_sandbox_profile(cursor_home())
    assert wrapped[3:] == plain


def test_no_wrapper_when_off_or_off_macos_or_without_sandbox_exec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox_available(monkeypatch, True)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    assert provider().build_command("cursor-agent", "/w", PARAMS)[0] == "cursor-agent"
    monkeypatch.delenv("EVALPROOF_CURSOR_SANDBOX")
    sandbox_available(monkeypatch, False)  # not macOS
    assert provider().build_command("cursor-agent", "/w", PARAMS)[0] == "cursor-agent"
    monkeypatch.undo()
    # The real check, on a platform that is macOS but has no sandbox-exec.
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(experimental, "SANDBOX_EXEC", str(tmp_path / "missing"))
    p = provider()
    assert not p.sandboxed and p.build_command("cursor-agent", "/w", PARAMS)[0] == "cursor-agent"


def test_the_command_doctor_prints_has_no_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_available(monkeypatch, True)
    command = provider().command("cursor-agent", "<tmp>/system.txt", PARAMS)
    assert command[0] == "cursor-agent" and "sandbox-exec" not in " ".join(command)
    assert command[-2:] == ["--workspace", "<tmp>/cwd"]


def test_the_isolation_tag_is_cursor_3_with_the_sandbox_and_cursor_2_without(
    fake: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox_available(monkeypatch, True)
    assert provider().isolation == "cursor/3"
    assert provider().fingerprint()["isolation"] == "cursor/3"
    assert cli_facts(TargetConfig(target="cursor:auto"))["isolation"] == "cursor/3"
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    assert provider().isolation == "cursor/2"
    assert provider().fingerprint()["isolation"] == "cursor/2"
    assert CursorProvider.isolation == "cursor/2"
    assert fake.calls() == []


def test_the_cache_key_differs_between_sandboxed_and_not(
    fake: FakeCLI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox_available(monkeypatch, True)
    sandboxed = provider().cache_salt(PARAMS)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    plain = provider().cache_salt(PARAMS)
    assert sandboxed["isolation"] == "cursor/3" and plain["isolation"] == "cursor/2"
    assert sandboxed != plain
    # A reply cached without the sandbox is not served to a sandboxed run.
    cache = ResponseCache(tmp_path / "cache")
    assert not CachedProvider(provider(), cache).chat(USER, PARAMS).cached
    monkeypatch.delenv("EVALPROOF_CURSOR_SANDBOX")
    assert not CachedProvider(provider(), cache).chat(USER, PARAMS).cached
    assert len(fake.calls()) == 2


def test_a_call_runs_the_cli_behind_sandbox_exec(
    fake: FakeCLI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A stand-in for /usr/bin/sandbox-exec: it records its arguments (NUL-separated, the prompt
    # has newlines), drops ``-p PROFILE`` and runs the rest, as the real one does once the
    # profile is applied.
    stand_in = tmp_path / "fake-sandbox-exec"
    stand_in.write_text('#!/bin/sh\nprintf \'%s\\0\' "$@" > "$0.argv"\nshift 2\nexec "$@"\n')
    stand_in.chmod(stand_in.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(experimental, "SANDBOX_EXEC", str(stand_in))
    sandbox_available(monkeypatch, True)
    assert provider().chat(USER, PARAMS).text == "OK"
    seen = Path(f"{stand_in}.argv").read_bytes().decode().split("\0")[:-1]
    assert seen[0] == "-p" and seen[1] == cursor_sandbox_profile(cursor_home())
    assert Path(seen[2]).name == "cursor-agent"
    assert seen[-1].endswith("What is 2+2?")  # the prompt is still the last argument
    call = fake.last()  # the CLI itself got its usual arguments
    assert call["argv"][:3] == ["-p", "--output-format", "stream-json"]
    assert call["argv"] == seen[3:]


# -- a failing sandbox -----------------------------------------------------------------------------


def test_a_sandbox_exec_failure_points_at_the_switch(
    fake: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    text = "sandbox-exec: sandbox_apply: Operation not permitted"
    respond(fake, reply("", stderr=text + "\n", exit=71), cursor_ok())
    with pytest.raises(ProviderError) as err:
        provider().chat(USER, PARAMS)
    message = str(err.value)
    assert text in message and "EVALPROOF_CURSOR_SANDBOX=off" in message
    assert len(fake.calls()) == 1  # not retried


def test_other_failures_get_no_sandbox_hint(fake: FakeCLI, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    respond(fake, reply("", stderr="Error: Authentication required.\n", exit=1))
    with pytest.raises(ProviderError) as err:
        provider().chat(USER, PARAMS)
    assert "Authentication required" in str(err.value)
    assert "EVALPROOF_CURSOR_SANDBOX" not in str(err.value)


# -- extra files in the workspace ------------------------------------------------------------------


def test_extra_workspace_files_are_written_next_to_the_permission_file(
    fake: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    extra = {".claude/skills/x/SKILL.md": "---\nname: x\n---\n", "note.txt": "hi"}
    provider(extra_workspace_files=extra).chat(USER, PARAMS)
    tree = fake.last()["cwd_tree"]
    assert sorted(tree) == [".claude/skills/x/SKILL.md", ".cursor/cli.json", "note.txt"]
    assert tree[".claude/skills/x/SKILL.md"] == extra[".claude/skills/x/SKILL.md"]
    provider().chat(USER, PARAMS)  # none by default
    assert list(fake.last()["cwd_tree"]) == [".cursor/cli.json"]


@pytest.mark.parametrize("path", ["/etc/x", "../x", "a/../../x"])
def test_extra_workspace_files_must_stay_inside_the_workspace(
    fake: FakeCLI, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    with pytest.raises(ProviderError, match="must be inside the workspace"):
        provider(extra_workspace_files={path: "x"}).chat(USER, PARAMS)
    assert fake.calls() == []


# -- the doctor warning ----------------------------------------------------------------------------


def write_suite(tmp_path: Path, judge: str) -> str:
    text = (
        "name: t\ndataset: d.jsonl\nsystem: {target: 'python:x:y'}\n"
        f"judge: {{target: '{judge}'}}\n"
        "scorers: [{name: m, type: exact_match}]\n"
    )
    path = tmp_path / "suite.yaml"
    path.write_text(text, "utf-8")
    return str(path)


def doctor_text(suite: str) -> tuple[int, str]:
    out = io.StringIO()
    return run_doctor(suite, out), out.getvalue()


def test_the_sandboxed_warning_names_the_folders_and_the_user_rules_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_available(monkeypatch, True)
    (text,) = experimental_warnings("cursor", "auto", [])
    assert text.startswith("cursor-agent runs in a macOS sandbox that blocks reads of your skill")
    for folder in ("~/.claude", "~/.agents", "~/.codex", "~/.gemini", "~/.cursor/skills"):
        assert folder in text
    assert "skills-cursor, rules, plugins, agents" in text
    assert "User Rules are still sent with every prompt" in text
    assert "Cursor Settings > Rules" in text
    assert "no option turns this off" not in text


def test_the_unsandboxed_warning_on_macos_says_the_sandbox_would_block_them(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sandbox_available(monkeypatch, True)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    (text,) = experimental_warnings("cursor", "auto", [])
    assert text.startswith("cursor-agent puts the names and descriptions of your skills")
    assert "no option turns this off" in text
    assert "on macOS, EVALPROOF_CURSOR_SANDBOX=on (the default) blocks them" in text
    assert "no sandbox is available" not in text


def test_the_warning_elsewhere_says_there_is_no_sandbox(monkeypatch: pytest.MonkeyPatch) -> None:
    sandbox_available(monkeypatch, False)
    for value in ("", "off"):
        monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", value)
        (text,) = experimental_warnings("cursor", "auto", [])
        assert "no option turns this off" in text and "no sandbox is available here" in text
        assert "(the default) blocks them" not in text


def test_doctor_reports_the_sandbox_in_its_cursor_block(
    fake: FakeCLI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    sandbox_available(monkeypatch, True)
    code, text = doctor_text(write_suite(tmp_path, "cursor:auto"))
    assert code == 0
    assert "isolation: cursor/3" in text and "runs in a macOS sandbox" in text
    assert "sandbox-exec" not in text and "(subpath" not in text  # no profile in the output
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    _, text = doctor_text(write_suite(tmp_path, "cursor:auto"))
    assert "isolation: cursor/2" in text and "EVALPROOF_CURSOR_SANDBOX=on" in text
    assert fake.calls() == []  # still a free check


def test_doctor_reports_an_invalid_sandbox_value(
    fake: FakeCLI, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    only_fakes_on_path(tmp_path, monkeypatch)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "maybe")
    code, text = doctor_text(write_suite(tmp_path, "cursor:auto"))
    assert code == 1
    assert "EVALPROOF_CURSOR_SANDBOX must be one of on, off, got 'maybe'" in text


# -- the canary ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "skills", "rules", "seen"),
    [
        ("SKILLS: evalproof-canary\nRULES: NONE", [], [], True),
        ("SKILLS: evalproof-canary, deploy, review\nRULES: NONE", ["deploy", "review"], [], True),
        ("SKILLS: NONE\nRULES: NONE", [], [], False),
        (
            "SKILLS: deploy\nRULES: House style, No emojis",
            ["deploy"],
            ["House style", "No emojis"],
            False,
        ),
        # markdown, a Line prefix and descriptions after a name
        (
            "**SKILLS:** `evalproof-canary` (Canary skill used by evalproof doctor.), "
            "`plugin:run`\n**RULES:** none.",
            ["plugin:run"],
            [],
            True,
        ),
        ("Line 1: SKILLS: evalproof-canary\nLine 2: RULES: Style", [], ["Style"], True),
        # a list over several lines, and a preamble before the labels
        (
            "Here you go.\nSKILLS:\n- evalproof-canary\n- /deploy\nRULES:\n- Style\n",
            ["deploy"],
            ["Style"],
            True,
        ),
        # no labels: the first line is the skills and the second the rules
        ("evalproof-canary, deploy\nStyle", ["deploy"], ["Style"], True),
        ("NONE\nNONE", [], [], False),
        ("", [], [], False),
        # only a rules label: what precedes it is the skills
        ("evalproof-canary\nRULES: Style", [], ["Style"], True),
        ("skill: EVALPROOF-CANARY", [], [], True),
    ],
)
def test_parse_canary_reply(text: str, skills: list[str], rules: list[str], seen: bool) -> None:
    assert parse_canary_reply(text) == (skills, rules, seen)


@pytest.fixture
def canary_env(fake: FakeCLI, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    return fake


def canary(fake: FakeCLI, text: str | None = None, **kwargs: Any) -> tuple[int, str]:
    if text is not None:
        respond(fake, cursor_ok(text, **kwargs))
    out = io.StringIO()
    return run_cursor_canary(out=out), out.getvalue()


def test_the_canary_alone_is_a_pass(canary_env: FakeCLI) -> None:
    code, text = canary(canary_env, "SKILLS: evalproof-canary\nRULES: NONE")
    assert code == 0
    assert "canary skill seen: yes" in text and "other skills seen: none" in text
    assert "user rules seen (titles only): none" in text
    assert "ok: only the canary skill reached the prompt" in text
    assert "best effort" in text and "one real cursor-agent call" in text
    assert f"model {CANARY_MODEL}" in text
    assert len(canary_env.calls()) == 1  # exactly one call


def test_the_canary_call_goes_through_the_providers_normal_path(canary_env: FakeCLI) -> None:
    canary(canary_env, "SKILLS: evalproof-canary\nRULES: NONE")
    call = canary_env.last()
    argv = call["argv"]
    assert argv[argv.index("--model") + 1] == CANARY_MODEL
    assert argv[argv.index("--mode") + 1] == "ask" and "--trust" in argv
    tree = call["cwd_tree"]
    assert sorted(tree) == [".claude/skills/evalproof-canary/SKILL.md", ".cursor/cli.json"]
    assert CANARY_SKILL_PATH == ".claude/skills/evalproof-canary/SKILL.md"
    skill = tree[CANARY_SKILL_PATH]
    assert skill.startswith("---\nname: evalproof-canary\n")
    assert "description: Canary skill used by evalproof doctor.\n---" in skill
    prompt = argv[-1]
    assert "Do not use any tool" in prompt and "SKILLS:" in prompt and "user_rule" in prompt
    assert "titles only, never the contents" in prompt
    # the deny-all permission file is the provider's, untouched
    assert json.loads(tree[".cursor/cli.json"])["permissions"]["allow"] == []


def test_another_skill_next_to_the_canary_is_a_leak(canary_env: FakeCLI) -> None:
    code, text = canary(canary_env, "SKILLS: evalproof-canary, deploy, plugin:review\nRULES: NONE")
    assert code == 1
    assert "canary skill seen: yes" in text
    assert "other skills seen: deploy, plugin:review" in text
    assert "leak: 2 skill(s) besides the canary reached the prompt" in text
    assert "ok:" not in text


def test_a_leak_with_the_sandbox_off_says_how_to_block_it(
    canary_env: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    sandbox_available(monkeypatch, True)
    _, text = canary(canary_env, "SKILLS: evalproof-canary, deploy\nRULES: NONE")
    assert "set EVALPROOF_CURSOR_SANDBOX=on to block them on macOS" in text
    sandbox_available(monkeypatch, False)
    _, text = canary(canary_env, "SKILLS: evalproof-canary, deploy\nRULES: NONE")
    assert "set EVALPROOF_CURSOR_SANDBOX=on" not in text


def test_no_canary_is_not_a_pass_even_with_no_other_skill(canary_env: FakeCLI) -> None:
    code, text = canary(canary_env, "SKILLS: NONE\nRULES: NONE")
    assert code == 1
    assert "canary skill seen: no" in text and "not conclusive" in text
    assert "ok:" not in text


def test_user_rule_titles_are_listed_and_contents_are_never_printed(canary_env: FakeCLI) -> None:
    reply_text = (
        "SKILLS: evalproof-canary\nRULES: House style, No emojis\n"
        "The first rule says: always answer in pirate speak SECRET-RULE-BODY"
    )
    code, text = canary(canary_env, reply_text)
    assert code == 0  # user rules are reported, not failed on
    assert "user rules seen (titles only): House style, No emojis" in text
    assert "Cursor Settings > Rules" in text
    assert "SECRET-RULE-BODY" not in text and "pirate" not in text


def test_long_rule_titles_and_long_lists_are_cut(canary_env: FakeCLI) -> None:
    titles = ", ".join(f"rule-{i}-" + "x" * 100 for i in range(25))
    _, text = canary(canary_env, f"SKILLS: evalproof-canary\nRULES: {titles}")
    line = next(ln for ln in text.splitlines() if ln.startswith("user rules seen"))
    assert "x" * 70 not in line and "…" in line and "and 5 more" in line


def test_a_failed_call_has_no_verdict(canary_env: FakeCLI) -> None:
    respond(canary_env, reply("", stderr="Error: model not available\n", exit=1))
    code, text = canary(canary_env)
    assert code == 4
    assert "canary call failed" in text and "model not available" in text
    assert "no verdict" in text and "canary skill seen" not in text


def test_a_tool_call_in_the_canary_turn_is_refused_by_the_providers_guard(
    canary_env: FakeCLI,
) -> None:
    call = {"readToolCall": {"args": {"path": "/etc/hosts"}}}
    respond(
        canary_env, reply(cursor_stream("SKILLS: evalproof-canary\nRULES: NONE", tool_calls=[call]))
    )
    code, text = canary(canary_env)
    assert code == 4 and "cursor used a tool (readToolCall); isolation broken" in text


def test_doctor_cursor_canary_command(
    canary_env: FakeCLI, capsys: pytest.CaptureFixture[str]
) -> None:
    respond(canary_env, cursor_ok("SKILLS: evalproof-canary\nRULES: NONE"))
    assert main(["doctor", "--cursor-canary", "--model", "composer-test"]) == 0
    assert "model composer-test" in capsys.readouterr().out
    argv = canary_env.last()["argv"]
    assert argv[argv.index("--model") + 1] == "composer-test"
    assert main(["doctor", "--cursor-canary"]) == 0
    argv = canary_env.last()["argv"]
    assert argv[argv.index("--model") + 1] == "composer-2.5-fast"


def test_doctor_cursor_canary_exit_codes_through_the_command(
    canary_env: FakeCLI, capsys: pytest.CaptureFixture[str]
) -> None:
    respond(canary_env, cursor_ok("SKILLS: evalproof-canary, other\nRULES: NONE"))
    assert main(["doctor", "--cursor-canary"]) == 1
    respond(canary_env, reply("", stderr="boom\n", exit=1))
    assert main(["doctor", "--cursor-canary"]) == 4
    capsys.readouterr()


def test_doctor_rejects_flag_combinations_that_make_no_sense(
    canary_env: FakeCLI, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor", "--model", "m"]) == 2
    assert "--model only applies with --cursor-canary" in capsys.readouterr().err
    assert main(["doctor", write_suite(tmp_path, "cursor:auto"), "--cursor-canary"]) == 2
    assert "does not take a SUITE" in capsys.readouterr().err
    assert canary_env.calls() == []  # neither spent a call


def test_the_help_says_the_canary_costs_a_call(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main(["doctor", "--help"])
    text = " ".join(capsys.readouterr().out.split())
    assert "ONE real cursor-agent call" in text and "Cursor quota" in text
    assert "model-reported" in text and "composer-2.5-fast" in text


def test_an_invalid_sandbox_value_is_a_usage_error_for_the_canary(
    canary_env: FakeCLI, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "maybe")
    assert main(["doctor", "--cursor-canary"]) == 2
    assert "EVALPROOF_CURSOR_SANDBOX must be one of on, off" in capsys.readouterr().err
    assert canary_env.calls() == []
