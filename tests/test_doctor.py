"""``evalproof doctor`` against fake binaries: it reports, and it never calls a model."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.doctor import run_doctor
from tests.fakecli import FakeCLI, claude_auth, claude_ok, codex_ok, install


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture(autouse=True)
def no_experimental_clis(monkeypatch: pytest.MonkeyPatch) -> None:
    # A machine that has OpenCode, Cursor or Copilot installed must not change what these see;
    # the experimental providers have their own doctor tests in test_agent_cli_experimental.py.
    monkeypatch.setattr("evalproof.doctor.installed_experimental", lambda: [])


@pytest.fixture
def both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeCLI, FakeCLI]:
    claude = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_ok()],
        auth_stdout=claude_auth(),
    )
    codex = install(
        tmp_path,
        monkeypatch,
        "codex",
        version="codex-cli 0.167.2",
        responses=[codex_ok()],
        auth_stdout="Logged in using ChatGPT\n",
    )
    return claude, codex


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


def test_doctor_without_a_suite_checks_both_clis(both: tuple[FakeCLI, FakeCLI]) -> None:
    code, text = doctor()
    assert code == 0
    claude_path, codex_path = (str(f.path) for f in both)
    assert claude_path in text and codex_path in text
    assert "version: 2.1.322 (Claude Code)" in text and "version: codex-cli 0.167.2" in text
    assert "auth: authMethod=claude.ai apiProvider=firstParty subscriptionType=team" in text
    assert "auth: Logged in using ChatGPT" in text
    assert "isolation: claude-code/1" in text and "isolation: codex/1" in text
    assert "--tools ''" in text and "--disable shell_tool" in text
    assert "--dangerously-skip-permissions" not in text
    assert text.rstrip().endswith("ok")
    # A free check: nothing was asked of a model.
    assert not both[0].calls() and not both[1].calls()


def test_doctor_does_not_print_the_account(both: tuple[FakeCLI, FakeCLI]) -> None:
    _, text = doctor()
    assert "someone@example.invalid" not in text and "Synthetic Org" not in text


def test_doctor_mentions_a_key_variable_it_will_strip(
    both: tuple[FakeCLI, FakeCLI], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic")
    _, text = doctor()
    assert "ANTHROPIC_API_KEY is set and is removed" in text
    assert "OPENAI_API_KEY is set and is removed" in text
    assert "synthetic" not in text.replace("is set and is removed", "")


def test_doctor_reads_a_suite_and_names_the_roles(
    both: tuple[FakeCLI, FakeCLI], tmp_path: Path
) -> None:
    suite = write_suite(tmp_path, "python:x:y", "claude-code:haiku")
    code, text = doctor(suite)
    assert code == 0
    assert "claude-code:haiku (judge)" in text and "codex" not in text
    suite = write_suite(tmp_path, "codex:gpt-6-luna", "codex:gpt-6-luna")
    _, text = doctor(suite)
    assert text.count("codex:gpt-6-luna (system, judge)") == 1  # one entry for the same target


def test_doctor_with_no_cli_target(tmp_path: Path) -> None:
    code, text = doctor(write_suite(tmp_path, "ollama:m", "ollama:j"))
    assert code == 0 and "nothing to check" in text


def test_doctor_names_the_key_variable_for_api_key_env(
    both: tuple[FakeCLI, FakeCLI], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MY_TEAM_KEY", "synthetic")
    suite = write_suite(tmp_path, "python:x:y", "claude-code:haiku", ", api_key_env: MY_TEAM_KEY")
    code, text = doctor(suite)
    assert code == 0 and "$MY_TEAM_KEY is passed to the CLI as ANTHROPIC_API_KEY" in text
    monkeypatch.delenv("MY_TEAM_KEY")
    code, text = doctor(suite)
    assert code == 1 and "set MY_TEAM_KEY" in text


def test_doctor_fails_on_a_missing_binary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    code, text = doctor()
    assert code == 1
    assert "npm install -g @anthropic-ai/claude-code" in text
    assert "npm install -g @openai/codex" in text
    assert "not ready" in text


def test_doctor_fails_when_not_logged_in(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        tmp_path,
        monkeypatch,
        "codex",
        version="codex-cli 0.167.2",
        responses=[codex_ok()],
        auth_stdout="Not logged in\n",
        auth_exit=1,
    )
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_ok()],
        auth_stdout='{"loggedIn": false}',
        auth_exit=1,
    )
    code, text = doctor()
    assert code == 1
    assert "auth: Not logged in" in text and "claude auth login" in text


def test_doctor_command(
    both: tuple[FakeCLI, FakeCLI], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["doctor"]) == 0
    assert "claude-code" in capsys.readouterr().out
    assert main(["doctor", write_suite(tmp_path, "codex:m", None)]) == 0
    assert main(["doctor", str(tmp_path / "missing.yaml")]) == 2
