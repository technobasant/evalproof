"""``evalproof init``: write a starter suite, a CI workflow and an agent skill into a repository.

Everything written comes from ``evalproof/templates`` (package data) with a few ``@@NAME@@``
markers filled in. Nothing is overwritten without ``--force`` and nothing is written outside
the target directory. The only commands run are zero-cost ones (``--version``,
``codex debug models``): no model call is made.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from evalproof.config import ConfigError
from evalproof.providers import LLM_PROVIDERS, PROFILES, parse_target
from evalproof.providers.agent_cli import CLAUDE_OAUTH_ENV

# The tag of the composite action (``action.yml``) the generated workflows use. The
# ``evalproof`` version that ``run`` is pinned to in the api-key variant is the same release.
ACTION_REF = "v0.7.0"

CI_MODES = ("keyless", "api-key", "none")

# What the api-key workflow hands a coding-agent CLI: its API key (the default), or, for Claude
# Code only, the token ``claude setup-token`` prints.
CLI_AUTHS = ("api-key", "oauth-token")

ANTHROPIC_DEFAULT = "anthropic:claude-haiku-5-5"
OLLAMA_DEFAULT = "ollama:qwen3:8b"
CODEX_PLACEHOLDER = "codex:<model>"


@dataclass(frozen=True)
class AgentCLI:
    """A coding-agent CLI that evalproof can use as a judge or system."""

    binary: str
    label: str
    npm_package: str
    key_env: str  # the API-key variable CI maps its secret to


AGENT_CLIS = {
    "claude-code": AgentCLI(
        "claude", "Claude Code", "@anthropic-ai/claude-code", "ANTHROPIC_API_KEY"
    ),
    "codex": AgentCLI("codex", "Codex", "@openai/codex", "OPENAI_API_KEY"),
}


@dataclass(frozen=True)
class Judge:
    """The model a starter suite uses for both the system and the judge, and why."""

    target: str
    reason: str
    note: str | None = None  # a YAML comment for the suite, when the target needs a hand edit
    warning: str | None = None  # printed after the file list
    api_key_env: str | None = None

    @property
    def provider(self) -> str:
        return self.target.split(":", 1)[0]


# --- zero-cost probes -------------------------------------------------------------------


def _capture(argv: list[str], timeout: float = 20.0) -> str | None:
    """stdout of a command, or None when it cannot run, times out or fails."""
    try:
        done = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def cli_version(binary_path: str) -> str | None:
    """The number in ``<cli> --version``: ``2.1.322 (Claude Code)``, ``codex-cli 0.167.2``."""
    out = _capture([binary_path, "--version"], timeout=15.0)
    match = re.search(r"\d+\.\d+\.\d+(?:-[0-9A-Za-z.]+)?", out or "")
    return match.group(0) if match else None


def codex_model(binary_path: str) -> str | None:
    """A cheap listed model from ``codex debug models`` (a catalogue dump, not a model call).

    Prefers the lowest-priority listed model described as fast, which is the judge-sized one;
    falls back to the first listed model. None when the catalogue cannot be read.
    """
    out = _capture([binary_path, "debug", "models"])
    try:
        models = json.loads(out or "")["models"]
    except (ValueError, KeyError, TypeError):
        return None
    listed = [
        m
        for m in models
        if isinstance(m, dict) and m.get("visibility") == "list" and isinstance(m.get("slug"), str)
    ]

    def priority(m: dict[str, object]) -> float:
        p = m.get("priority")
        return float(p) if isinstance(p, int | float) else float("inf")

    listed.sort(key=priority)
    fast = [m for m in listed if "fast" in str(m.get("description", "")).lower()]
    chosen = fast or listed
    return str(chosen[0]["slug"]) if chosen else None


# --- the judge ---------------------------------------------------------------------------


def choose_judge(explicit: str | None) -> Judge:
    """``--judge`` as given, else the first of claude, codex, ollama found on ``PATH``,
    else a hosted Anthropic model that needs an API key."""
    if explicit:
        provider, _ = parse_target(explicit)
        if provider not in AGENT_CLIS and provider not in LLM_PROVIDERS:
            known = sorted({*AGENT_CLIS, *LLM_PROVIDERS})
            raise ConfigError(f"--judge: unknown provider {provider!r}; expected one of {known}")
        return Judge(
            explicit,
            "from --judge",
            api_key_env="ANTHROPIC_API_KEY" if provider == "anthropic" else None,
        )
    claude = shutil.which("claude")
    if claude:
        return Judge(
            "claude-code:haiku",
            "claude is on PATH, so evals use your Claude Code login and need no API key",
        )
    codex = shutil.which("codex")
    if codex:
        model = codex_model(codex)
        if model:
            return Judge(
                f"codex:{model}",
                f"codex is on PATH, so evals use your Codex login and need no API key; "
                f"{model} comes from `codex debug models`",
            )
        return Judge(
            CODEX_PLACEHOLDER,
            "codex is on PATH, so evals use your Codex login and need no API key",
            note='TODO: replace <model> with a model name from "codex debug models"',
            warning="could not read `codex debug models`; set the codex model in evals/suite.yaml",
        )
    if shutil.which("ollama"):
        return Judge(
            OLLAMA_DEFAULT,
            f"ollama is on PATH, so evals run locally with no API key; `ollama pull "
            f"{OLLAMA_DEFAULT.split(':', 1)[1]}` first",
        )
    return Judge(
        ANTHROPIC_DEFAULT,
        "no claude, codex or ollama on PATH, so this uses a hosted model that needs "
        "ANTHROPIC_API_KEY",
        api_key_env="ANTHROPIC_API_KEY",
    )


# --- templates ---------------------------------------------------------------------------


def template(name: str) -> str:
    return (files("evalproof") / "templates" / name).read_text(encoding="utf-8")


_MARKER = re.compile(r"@@([A-Z_]+)@@")


def fill(text: str, values: dict[str, str], flags: frozenset[str] = frozenset()) -> str:
    """Replace ``@@NAME@@`` markers; keep ``#@@IF flag`` ... ``#@@END`` blocks only for set flags.

    A line that is nothing but a marker whose value is empty is dropped, so optional lines
    leave no blank behind. A marker without a value is a bug in the template.
    """
    kept: list[str] = []
    skipping = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#@@IF "):
            skipping = stripped.removeprefix("#@@IF ") not in flags
            continue
        if stripped == "#@@END":
            skipping = False
            continue
        if skipping:
            continue
        if _MARKER.fullmatch(stripped) and not values.get(stripped.strip("@")):
            continue
        kept.append(_MARKER.sub(lambda m: values[m.group(1)], line))
    return "\n".join(kept) + "\n"


# --- the files ---------------------------------------------------------------------------


def _key_env(judge: Judge) -> str:
    """The variable a CI secret is mapped to for this judge."""
    if judge.provider in AGENT_CLIS:
        return AGENT_CLIS[judge.provider].key_env
    if judge.api_key_env:
        return judge.api_key_env
    profile = PROFILES.get(judge.provider)
    if profile is None or profile.local or not profile.api_key_env:
        raise ConfigError(
            f"--ci api-key needs a hosted judge with an API key, and {judge.target!r} has none "
            "(a local model cannot run on a CI runner). Pass --judge with a hosted target, "
            "or use --ci keyless."
        )
    return profile.api_key_env


def render_suite(judge: Judge) -> str:
    line = f"  api_key_env: {judge.api_key_env}" if judge.api_key_env else ""
    local = judge.provider in ("ollama", "vllm", "lmstudio", "llamacpp")
    note = f"  # {judge.note}" if judge.note else ""
    return fill(
        template("suite.yaml"),
        {
            "TARGET": judge.target,
            "TARGET_NOTE": note,
            "API_KEY_LINE": line,
            "CONCURRENCY": "1" if local else "4",
        },
    )


def _ci_secret(judge: Judge, cli_auth: str) -> str:
    """The repository secret the api-key workflow reads for this judge."""
    return CLAUDE_OAUTH_ENV if cli_auth == "oauth-token" else _key_env(judge)


def render_ci_text(ci: str, judge: Judge, cli_auth: str = "api-key") -> str:
    if ci == "keyless":
        return (
            "Commit `evals/runs/latest.json` with your change. CI makes no model calls: "
            "`.github/workflows/evalproof.yml` compares that file with the one on the base "
            "branch and fails the pull request on a regression that is both large enough to "
            "matter and statistically significant (exit 1; exit 3 over a latency or cost "
            "budget; exit 4 when it cannot decide). Before comparing it runs `evalproof verify`, "
            "which fails the job when `latest.json` does not match the suite and the cases in "
            "`cases.jsonl` (a stale file, a `--limit` run, an edited case). It also warns when "
            "`evals/` changed but `latest.json` did not."
        )
    if ci == "api-key":
        text = (
            "Commit `evals/runs/latest.json` on the main branch to set the baseline. "
            "`.github/workflows/evalproof.yml` runs the suite on a pull request that changes "
            "`evals/` (add the paths that hold your prompt to `WATCH` in the workflow) with the "
            f"`{_ci_secret(judge, cli_auth)}` repository secret and compares the result with "
            "that file; "
            "it fails on a regression that is both large enough to matter and statistically "
            "significant (exit 1; exit 3 over a budget; exit 4 when it cannot decide). Pull "
            "requests from forks get no secrets, so the workflow runs nothing for them. "
            "Responses are cached between runs."
        )
        if cli_auth == "oauth-token":
            text += (
                f" The run step passes `{CLAUDE_OAUTH_ENV}` from that secret and sets no "
                "`EVALPROOF_CLI_AUTH`, so the CLI signs in with the token; locally it keeps "
                "using your login. The token comes from `claude setup-token`: use it only in "
                "your own repository and CI, and use an API key for a secret shared across a "
                "team or organisation."
            )
        elif judge.provider in AGENT_CLIS:
            text += (
                " The run step sets `EVALPROOF_CLI_AUTH=api-key`, so the CLI uses that key "
                "instead of a login; locally it keeps using your login."
            )
        return text
    return (
        "No CI workflow was written. To gate by hand, run the suite on the base and on your "
        "branch and compare: `uvx evalproof compare base.json candidate.json "
        "--suite evals/suite.yaml --gate`."
    )


def render_workflow(ci: str, judge: Judge, pinned: str | None, cli_auth: str = "api-key") -> str:
    if ci == "keyless":
        return fill(template("workflow-keyless.yml"), {"ACTION_REF": ACTION_REF})
    cli = AGENT_CLIS.get(judge.provider)
    oauth = cli_auth == "oauth-token"
    key_var = _key_env(judge)
    values = {
        "ACTION_REF": ACTION_REF,
        "PIP_VERSION": ACTION_REF.removeprefix("v"),
        "KEY_VAR": key_var,
        "CLI_BINARY": cli.binary if cli else "",
        "NPM_SPEC": "",
        "CLI_PIN_NOTE": "",
    }
    if cli:
        if pinned:
            values["NPM_SPEC"] = f"{cli.npm_package}@{pinned}"
            values["CLI_PIN_NOTE"] = (
                f"      # `evalproof init` found {cli.binary} {pinned} and pinned it."
            )
        else:
            values["NPM_SPEC"] = f"{cli.npm_package}@latest"
            values["CLI_PIN_NOTE"] = (
                f"      # `evalproof init` did not find {cli.binary}, so this installs the "
                "latest:\n      # replace latest with the version your run file records."
            )
    flags = {"cli"} if cli else set()
    if oauth:
        flags.add("oauth")  # the token replaces both the API key and EVALPROOF_CLI_AUTH
    else:
        flags.add("key")
        if cli:
            flags.add("cliauth")
    return fill(template("workflow-api-key.yml"), values, frozenset(flags))


def build_files(
    judge: Judge, ci: str, skill: bool, pinned: str | None, cli_auth: str = "api-key"
) -> list[tuple[str, str]]:
    """Every file to write, as (path relative to the target directory, content)."""
    out = [
        ("evals/suite.yaml", render_suite(judge)),
        ("evals/cases.jsonl", template("cases.jsonl")),
        (
            "evals/README.md",
            fill(template("evals-readme.md"), {"CI_TEXT": render_ci_text(ci, judge, cli_auth)}),
        ),
        ("evals/.gitignore", template("gitignore")),
    ]
    if ci != "none":
        out.append(
            (".github/workflows/evalproof.yml", render_workflow(ci, judge, pinned, cli_auth))
        )
    if skill:
        body = template("SKILL.md")
        # Two copies, not a symlink, so a Windows checkout works.
        out.append((".agents/skills/evalproof/SKILL.md", body))
        out.append((".claude/skills/evalproof/SKILL.md", body))
    return out


def _hosted_key(judge: Judge) -> str | None:
    """The environment variable a hosted judge reads its API key from; None for a local one."""
    if judge.api_key_env:
        return judge.api_key_env
    profile = PROFILES.get(judge.provider)
    if profile is None or profile.local:
        return None
    return profile.api_key_env


def requirement_lines(judge: Judge, ci: str, cli_auth: str = "api-key") -> list[str]:
    """What the starter suite needs to run, locally and in CI, for the chosen judge.

    The same facts as the README's "What it needs to run" table, for the one target ``init``
    chose. Names variables only; it reads none."""
    cli = AGENT_CLIS.get(judge.provider)
    key = _hosted_key(judge)
    if cli:
        local = f"your {cli.label} login (`{cli.binary}` on PATH, logged in); no API key"
    elif key:
        local = f"{key} in your environment"
    else:
        local = f"a local model server ({judge.provider}) running, with the model pulled"
    if ci == "keyless":
        remote = "no secrets: keyless mode makes no model calls in CI, it compares run files"
    elif ci == "none":
        remote = "nothing set up: no workflow was written"
    elif cli_auth == "oauth-token":
        remote = (
            f"the {CLAUDE_OAUTH_ENV} repository secret, from `claude setup-token`; "
            "your own repository and CI only (use an API key for a secret shared across a "
            "team or organisation)"
        )
    elif cli:
        remote = f"the {cli.key_env} repository secret, with EVALPROOF_CLI_AUTH=api-key"
        if judge.provider == "codex":
            remote += (
                "; a personal Codex login belongs only on trusted private infrastructure, "
                "never on public repositories or for open source work"
            )
    else:
        remote = f"the {_key_env(judge)} repository secret"
    lines = [f"locally: {local}", f"in CI: {remote}"]
    if ci == "api-key":
        lines.append("pull requests from forks never receive secrets, so the workflow skips them")
    return lines


def init_project(
    directory: Path,
    *,
    judge: str | None = None,
    ci: str = "keyless",
    skill: bool = True,
    force: bool = False,
    cli_auth: str = "api-key",
    emit: Callable[[str], None] = print,
) -> int:
    """Write the starter files into ``directory``; returns the exit code (always 0)."""
    if ci not in CI_MODES:
        raise ConfigError(f"--ci must be one of {', '.join(CI_MODES)}, got {ci!r}")
    if cli_auth not in CLI_AUTHS:
        raise ConfigError(f"--cli-auth must be one of {', '.join(CLI_AUTHS)}, got {cli_auth!r}")
    chosen = choose_judge(judge)
    if cli_auth == "oauth-token":
        if ci != "api-key":
            raise ConfigError(
                "--cli-auth oauth-token changes the secret the api-key workflow passes, so it "
                f"needs --ci api-key (got --ci {ci})"
            )
        if chosen.provider != "claude-code":
            raise ConfigError(
                f"--cli-auth oauth-token works only with a claude-code judge, and the judge is "
                f"{chosen.target!r}. Pass --judge claude-code:<model>, or leave --cli-auth out "
                "to use an API key (Codex and every other target have no token option)."
            )
    # An api-key workflow needs the key variable and, for a CLI judge, its pinned version.
    pinned = None
    if ci == "api-key":
        _key_env(chosen)  # fail before writing anything if there is no key to map
        cli = AGENT_CLIS.get(chosen.provider)
        found = shutil.which(cli.binary) if cli else None
        pinned = cli_version(found) if found else None
    files_to_write = build_files(chosen, ci, skill, pinned, cli_auth)

    emit(f"evalproof init: judge {chosen.target} ({chosen.reason})")
    root = directory.resolve()
    for rel, content in files_to_write:
        path = root / rel
        if path.exists() and not force:
            emit(f"  kept (exists)  {rel}")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        emit(f"  wrote          {rel}")

    if chosen.warning:
        emit(f"note: {chosen.warning}")
    if cli_auth == "oauth-token":
        emit(
            f"warning: {CLAUDE_OAUTH_ENV} is a Claude login token from `claude setup-token`. "
            "Use it in your own repository and CI only; for a secret shared across a team or "
            "organisation use an API key (leave --cli-auth out)."
        )
    cli = AGENT_CLIS.get(chosen.provider)
    if ci == "api-key" and cli:
        if pinned:
            emit(f"note: the workflow installs {cli.binary} {pinned}, the version found here")
        else:
            emit(f"note: {cli.binary} was not found, so the workflow installs the latest; pin it")
    emit("")
    emit("next:")
    emit("  uvx evalproof run evals/suite.yaml --out evals/runs/latest.json")
    if ci == "api-key":
        secret = _ci_secret(chosen, cli_auth)
        emit(f"  add the repository secret {secret}, then commit evals/ and .github/")
    else:
        emit("  commit evals/ (including evals/runs/latest.json) with your change")
    emit("")
    emit("what it needs to run:")
    for line in requirement_lines(chosen, ci, cli_auth):
        emit(f"  {line}")
    return 0
