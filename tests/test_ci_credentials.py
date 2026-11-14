"""CI credentials, stated plainly: the README table, what ``init`` prints and writes for each
target, ``init --cli-auth oauth-token`` (Claude Code only) and what ``doctor`` says about the
credential each target would use. None of it reads a secret: only the names of variables."""

from __future__ import annotations

import io
import os
import re
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalproof.cli import main
from evalproof.config import ConfigError, TargetConfig
from evalproof.doctor import credential_lines, run_doctor
from evalproof.providers.agent_cli import CLAUDE_OAUTH_ENV, CLI_PROVIDERS, AgentCLIProvider
from evalproof.scaffold import Judge, init_project, requirement_lines
from tests.fakecli import FakeCLI, claude_auth, claude_ok, codex_ok, install

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = Path(".github/workflows/evalproof.yml")
TOKEN_VALUE = "synthetic-credential-value-do-not-print"

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake CLIs are shell scripts")

CREDENTIAL_VARS = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    CLAUDE_OAUTH_ENV,
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "EVALPROOF_CLI_AUTH",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in CREDENTIAL_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("evalproof.doctor.installed_experimental", lambda: [])


@pytest.fixture
def bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory that is the whole ``PATH``: no real claude, codex or ollama."""
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d))
    return d


def run_init(directory: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["init", "--dir", str(directory), *args])
    return code, capsys.readouterr().out


def run_step(directory: Path) -> dict[str, Any]:
    wf = yaml.safe_load((directory / WORKFLOW).read_text(encoding="utf-8"))
    steps = wf["jobs"]["gate"]["steps"]
    step: dict[str, Any] = next(s for s in steps if s.get("name") == "Run the suite")
    return step


# --- README ------------------------------------------------------------------------------------


def _readme_section() -> str:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    match = re.search(r"^## What it needs to run\n(.*?)^## ", text, re.S | re.M)
    assert match, "README has no 'What it needs to run' section"
    return match.group(1)


def test_readme_has_the_credentials_section_near_the_top() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert text.index("## What it needs to run") < text.index("## Use cases")
    assert text.index("## Install") < text.index("## What it needs to run")


def test_readme_table_has_a_row_per_target_family() -> None:
    section = _readme_section()
    rows = [line for line in section.splitlines() if line.startswith("|")]
    header, _rule, *body = rows  # the rule line is `|---|---|...`
    assert [c.strip() for c in header.strip("|").split("|")] == [
        "Target",
        "Credential variable",
        "Local use",
        "CI use",
        "Where allowed",
    ]
    families = [row.split("|")[1].strip() for row in body]
    assert len(families) == 5
    for needle in ("No model", "Local models", "Hosted APIs", "Claude Code", "Codex"):
        assert any(needle in family for family in families), needle


def test_readme_states_the_credential_rules_plainly() -> None:
    section = " ".join(_readme_section().split())  # the prose wraps
    for needle in (
        "Keyless mode needs no secrets",
        "ANTHROPIC_API_KEY",
        CLAUDE_OAUTH_ENV,
        "claude setup-token",
        "own repository",
        "OPENAI_API_KEY",
        "trusted private infrastructure",
        "never on public repos",
        "Fork pull requests never receive secrets",
        "EVALPROOF_CLI_AUTH=api-key",
        "--cli-auth oauth-token",
    ):
        assert needle in section, needle


# --- init: the requirement it prints -----------------------------------------------------------


def _requirement(out: str) -> list[str]:
    head, _, tail = out.partition("what it needs to run:\n")
    assert tail, "init printed no requirement block"
    assert head.rstrip().splitlines()[-1].startswith("  ")  # it follows the `next:` steps
    return [line.strip() for line in tail.splitlines() if line.strip()]


def test_init_prints_the_requirement_last_for_a_login_judge_with_keyless_ci(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_init(tmp_path, capsys, "--judge", "claude-code:haiku")
    assert code == 0 and out.rstrip().splitlines()[-3] == "what it needs to run:"
    lines = _requirement(out)
    assert lines[0].startswith("locally: your Claude Code login") and "no API key" in lines[0]
    assert lines[1].startswith("in CI: no secrets") and len(lines) == 2


def test_init_prints_the_secret_a_claude_code_api_key_workflow_needs(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "claude-code:haiku")
    lines = _requirement(out)
    assert "the ANTHROPIC_API_KEY repository secret, with EVALPROOF_CLI_AUTH=api-key" in lines[1]
    assert any("never receive secrets" in line for line in lines)


def test_init_prints_the_codex_limit_for_a_codex_judge(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "codex:gpt-6-luna")
    lines = _requirement(out)
    assert lines[0].startswith("locally: your Codex login")
    assert "the OPENAI_API_KEY repository secret" in lines[1]
    assert "trusted private infrastructure" in lines[1] and "public repositories" in lines[1]


def test_init_prints_the_key_variable_for_a_hosted_judge(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "anthropic:claude-haiku-5-5")
    lines = _requirement(out)
    assert lines[0] == "locally: ANTHROPIC_API_KEY in your environment"
    assert lines[1] == "in CI: the ANTHROPIC_API_KEY repository secret"
    _, out = run_init(tmp_path / "keyless", capsys, "--judge", "openai:gpt-5")
    assert _requirement(out)[0] == "locally: OPENAI_API_KEY in your environment"


def test_init_prints_a_local_server_for_a_local_judge_and_nothing_for_no_ci(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--ci", "none", "--judge", "ollama:qwen3:8b")
    lines = _requirement(out)
    assert lines[0].startswith("locally: a local model server (ollama) running")
    assert lines[1] == "in CI: nothing set up: no workflow was written"


def test_requirement_lines_name_variables_only() -> None:
    judge = Judge("claude-code:haiku", "test")
    text = "\n".join(requirement_lines(judge, "api-key", "oauth-token"))
    assert CLAUDE_OAUTH_ENV in text and "claude setup-token" in text
    assert TOKEN_VALUE not in text


# --- init --cli-auth oauth-token ---------------------------------------------------------------


def test_oauth_token_workflow_passes_the_token_in_login_mode(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = run_init(
        tmp_path,
        capsys,
        "--ci",
        "api-key",
        "--cli-auth",
        "oauth-token",
        "--judge",
        "claude-code:haiku",
    )
    assert code == 0
    run = run_step(tmp_path)
    assert run["env"] == {CLAUDE_OAUTH_ENV: f"${{{{ secrets.{CLAUDE_OAUTH_ENV} }}}}"}
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    assert "EVALPROOF_CLI_AUTH" not in text and "ANTHROPIC_API_KEY" not in text
    assert "own" in text and "API key" in text  # the comment says where the token may be used
    assert "@@" not in text and "#@@" not in text
    steps = yaml.safe_load(text)["jobs"]["gate"]["steps"]
    assert any(s.get("name") == "Install claude" for s in steps)  # still installs the CLI


def test_oauth_token_init_prints_the_own_ci_warning_and_the_secret(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(
        tmp_path,
        capsys,
        "--ci",
        "api-key",
        "--cli-auth",
        "oauth-token",
        "--judge",
        "claude-code:haiku",
    )
    assert "warning: CLAUDE_CODE_OAUTH_TOKEN" in out and "your own repository and CI only" in out
    assert "add the repository secret CLAUDE_CODE_OAUTH_TOKEN" in out
    assert "ANTHROPIC_API_KEY" not in out
    lines = _requirement(out)
    assert lines[1].startswith(f"in CI: the {CLAUDE_OAUTH_ENV} repository secret")
    assert "your own repository and CI only" in lines[1]


def test_oauth_token_evals_readme_names_the_token_not_the_key(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(
        tmp_path,
        capsys,
        "--ci",
        "api-key",
        "--cli-auth",
        "oauth-token",
        "--judge",
        "claude-code:haiku",
    )
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert f"`{CLAUDE_OAUTH_ENV}` repository secret" in readme
    assert "EVALPROOF_CLI_AUTH=api-key" not in readme and "claude setup-token" in readme
    assert "your own repository and CI" in readme


def test_the_default_cli_auth_changes_nothing_in_the_workflow(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path / "a", capsys, "--ci", "api-key", "--judge", "claude-code:haiku")
    run_init(
        tmp_path / "b",
        capsys,
        "--ci",
        "api-key",
        "--cli-auth",
        "api-key",
        "--judge",
        "claude-code:haiku",
    )
    assert (tmp_path / "a" / WORKFLOW).read_text("utf-8") == (tmp_path / "b" / WORKFLOW).read_text(
        "utf-8"
    )
    assert run_step(tmp_path / "a")["env"] == {
        "EVALPROOF_CLI_AUTH": "api-key",
        "ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}",
    }


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["--ci", "api-key", "--judge", "codex:gpt-6-luna"], "claude-code judge"),
        (["--ci", "api-key", "--judge", "anthropic:claude-haiku-5-5"], "claude-code judge"),
        (["--ci", "api-key", "--judge", "ollama:qwen3:8b"], "claude-code judge"),
        (["--ci", "keyless", "--judge", "claude-code:haiku"], "needs --ci api-key"),
        (["--ci", "none", "--judge", "claude-code:haiku"], "needs --ci api-key"),
    ],
)
def test_oauth_token_is_a_usage_error_for_anything_but_claude_code_with_api_key_ci(
    tmp_path: Path,
    bin_dir: Path,
    capsys: pytest.CaptureFixture[str],
    args: list[str],
    message: str,
) -> None:
    code = main(["init", "--dir", str(tmp_path / "p"), "--cli-auth", "oauth-token", *args])
    captured = capsys.readouterr()
    assert code == 2
    assert message in captured.err and captured.out == ""
    assert not (tmp_path / "p").exists()  # refused before anything was written


def test_the_codex_refusal_says_codex_has_no_token_option(tmp_path: Path, bin_dir: Path) -> None:
    with pytest.raises(ConfigError, match="Codex and every other target have no token option"):
        init_project(
            tmp_path,
            judge="codex:gpt-6-luna",
            ci="api-key",
            cli_auth="oauth-token",
            emit=lambda _line: None,
        )


def test_an_unknown_cli_auth_is_refused(tmp_path: Path, bin_dir: Path) -> None:
    with pytest.raises(SystemExit) as exc:  # argparse rejects it before init runs
        main(["init", "--dir", str(tmp_path), "--cli-auth", "token"])
    assert exc.value.code == 2
    with pytest.raises(ConfigError, match="--cli-auth must be one of api-key, oauth-token"):
        init_project(tmp_path, cli_auth="token", emit=lambda _line: None)


def test_login_mode_keeps_the_oauth_token_and_drops_the_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What the oauth-token workflow relies on: with no EVALPROOF_CLI_AUTH the CLI still gets
    the token, and a stray API key is not passed."""
    monkeypatch.setenv(CLAUDE_OAUTH_ENV, TOKEN_VALUE)
    monkeypatch.setenv("ANTHROPIC_API_KEY", TOKEN_VALUE)
    env = _provider("claude-code").child_env()
    assert env[CLAUDE_OAUTH_ENV] == TOKEN_VALUE and "ANTHROPIC_API_KEY" not in env


# --- doctor ------------------------------------------------------------------------------------


def _provider(kind: str, cfg: TargetConfig | None = None) -> AgentCLIProvider:
    cfg = cfg or TargetConfig(target=f"{kind}:m")
    return CLI_PROVIDERS[kind](
        "m", base_url=cfg.base_url, timeout_s=cfg.timeout_s, api_key_env=cfg.api_key_env,
        headers=cfg.headers,
    )  # fmt: skip


@pytest.fixture
def both(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[FakeCLI, FakeCLI]:
    claude = install(
        tmp_path, monkeypatch, "claude", version="2.1.322 (Claude Code)",
        responses=[claude_ok()], auth_stdout=claude_auth(),
    )  # fmt: skip
    codex = install(
        tmp_path, monkeypatch, "codex", version="codex-cli 0.167.2",
        responses=[codex_ok()], auth_stdout="Logged in using ChatGPT\n",
    )  # fmt: skip
    return claude, codex


def doctor(suite: str | None = None) -> tuple[int, str]:
    out = io.StringIO()
    return run_doctor(suite, out), out.getvalue()


def _section(text: str, kind: str) -> str:
    """The doctor lines of one target: from its header to the next header or the summary."""
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line == kind)
    end = next((i for i in range(start + 1, len(lines)) if not lines[i].startswith(" ")), None)
    return "\n".join(lines[start:end])


def test_doctor_reports_login_only_for_a_login_target(both: tuple[FakeCLI, FakeCLI]) -> None:
    code, text = doctor()
    assert code == 0
    claude, codex = _section(text, "claude-code"), _section(text, "codex")
    assert "credential: your claude-code login on this machine" in claude
    assert "in CI: login only, not available in CI" in claude
    assert (
        "map a repository secret to ANTHROPIC_API_KEY and set EVALPROOF_CLI_AUTH=api-key" in claude
    )
    assert f"{CLAUDE_OAUTH_ENV} from `claude setup-token`" in claude
    assert "your own repository and CI only" in claude
    assert "credential: your codex login on this machine" in codex
    assert "in CI: login only, not available in CI" in codex
    assert "map a repository secret to OPENAI_API_KEY" in codex
    assert "trusted private infrastructure" in codex and "public repositories" in codex
    assert CLAUDE_OAUTH_ENV not in codex  # Codex has no token option


def test_doctor_reports_a_key_as_usable_in_ci_in_api_key_mode(
    both: tuple[FakeCLI, FakeCLI], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALPROOF_CLI_AUTH", "api-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", TOKEN_VALUE)
    _, text = doctor()
    claude, codex = _section(text, "claude-code"), _section(text, "codex")
    assert "credential: ANTHROPIC_API_KEY; set here (EVALPROOF_CLI_AUTH=api-key)" in claude
    assert "in CI: usable; map a repository secret to ANTHROPIC_API_KEY" in claude
    assert "credential: OPENAI_API_KEY or CODEX_API_KEY; not set here" in codex
    assert "in CI: usable;" in codex
    assert TOKEN_VALUE not in text


def test_doctor_reports_the_oauth_token_for_claude_code_only_in_its_own_ci(
    both: tuple[FakeCLI, FakeCLI], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(CLAUDE_OAUTH_ENV, TOKEN_VALUE)
    _, text = doctor()
    claude = _section(text, "claude-code")
    assert f"credential: {CLAUDE_OAUTH_ENV}; set here, and login mode leaves it in place" in claude
    assert "usable from a secret, in your own repository and CI only" in claude
    assert CLAUDE_OAUTH_ENV not in _section(text, "codex")
    assert TOKEN_VALUE not in text


def test_doctor_reports_the_api_key_env_a_suite_names(
    both: tuple[FakeCLI, FakeCLI], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ndataset: d.jsonl\nsystem: {target: 'python:x:y'}\n"
        "judge: {target: 'claude-code:haiku', api_key_env: MY_TEAM_KEY}\n"
        "scorers: [{name: m, type: exact_match}]\n",
        "utf-8",
    )
    monkeypatch.setenv("MY_TEAM_KEY", TOKEN_VALUE)  # a target with an unset api_key_env fails
    _, text = doctor(str(suite))
    assert "credential: $MY_TEAM_KEY (the suite's api_key_env), passed as ANTHROPIC_API_KEY" in text
    assert "in CI: usable; map a repository secret to MY_TEAM_KEY" in text
    assert TOKEN_VALUE not in text


def test_doctor_keeps_its_summary_and_exit_code(both: tuple[FakeCLI, FakeCLI]) -> None:
    code, text = doctor()
    assert code == 0 and text.rstrip().endswith("ready: claude-code, codex")
    assert "warning" not in text and "problem" not in text


@pytest.mark.parametrize("kind", ["claude-code", "codex"])
@pytest.mark.parametrize("mode", [None, "login", "api-key"])
def test_credential_lines_check_names_and_never_read_a_value(
    kind: str, mode: str | None, monkeypatch: pytest.MonkeyPatch
) -> None:
    for var in CREDENTIAL_VARS:
        monkeypatch.setenv(var, TOKEN_VALUE)
    if mode:
        monkeypatch.setenv("EVALPROOF_CLI_AUTH", mode)
    else:
        monkeypatch.delenv("EVALPROOF_CLI_AUTH")
    monkeypatch.setenv("MY_TEAM_KEY", TOKEN_VALUE)
    reads: list[str] = []

    class Spy(dict[str, str]):  # records every value read; `name in environ` is not a read
        def __getitem__(self, key: str) -> str:
            reads.append(key)
            return super().__getitem__(key)

        def get(self, key: str, default: Any = None) -> Any:
            reads.append(key)
            return super().get(key, default)

    cfg = TargetConfig(target=f"{kind}:m")
    provider = _provider(kind, cfg)  # built first: the auth mode is read here, a value of none
    with_key = TargetConfig(target=f"{kind}:m", api_key_env="MY_TEAM_KEY")
    keyed = _provider(kind, with_key)
    monkeypatch.setattr(os, "environ", Spy(os.environ))
    lines = [*credential_lines(kind, cfg, provider), *credential_lines(kind, with_key, keyed)]
    assert reads == []
    assert lines and TOKEN_VALUE not in "\n".join(lines)
