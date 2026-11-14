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

# The tag of the composite action (``action.yml``) the generated workflows use. The
# ``evalproof`` version that ``run`` is pinned to in the api-key variant is the same release.
ACTION_REF = "v0.5.2"

CI_MODES = ("keyless", "api-key", "none")

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


def render_ci_text(ci: str, judge: Judge) -> str:
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
            f"`{_key_env(judge)}` repository secret and compares the result with that file; "
            "it fails on a regression that is both large enough to matter and statistically "
            "significant (exit 1; exit 3 over a budget; exit 4 when it cannot decide). Pull "
            "requests from forks get no secrets, so the workflow runs nothing for them. "
            "Responses are cached between runs."
        )
        if judge.provider in AGENT_CLIS:
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


def render_workflow(ci: str, judge: Judge, pinned: str | None) -> str:
    if ci == "keyless":
        return fill(template("workflow-keyless.yml"), {"ACTION_REF": ACTION_REF})
    cli = AGENT_CLIS.get(judge.provider)
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
    return fill(
        template("workflow-api-key.yml"), values, frozenset({"cli"}) if cli else frozenset()
    )


def build_files(judge: Judge, ci: str, skill: bool, pinned: str | None) -> list[tuple[str, str]]:
    """Every file to write, as (path relative to the target directory, content)."""
    out = [
        ("evals/suite.yaml", render_suite(judge)),
        ("evals/cases.jsonl", template("cases.jsonl")),
        (
            "evals/README.md",
            fill(template("evals-readme.md"), {"CI_TEXT": render_ci_text(ci, judge)}),
        ),
        ("evals/.gitignore", template("gitignore")),
    ]
    if ci != "none":
        out.append((".github/workflows/evalproof.yml", render_workflow(ci, judge, pinned)))
    if skill:
        body = template("SKILL.md")
        # Two copies, not a symlink, so a Windows checkout works.
        out.append((".agents/skills/evalproof/SKILL.md", body))
        out.append((".claude/skills/evalproof/SKILL.md", body))
    return out


def init_project(
    directory: Path,
    *,
    judge: str | None = None,
    ci: str = "keyless",
    skill: bool = True,
    force: bool = False,
    emit: Callable[[str], None] = print,
) -> int:
    """Write the starter files into ``directory``; returns the exit code (always 0)."""
    if ci not in CI_MODES:
        raise ConfigError(f"--ci must be one of {', '.join(CI_MODES)}, got {ci!r}")
    chosen = choose_judge(judge)
    # An api-key workflow needs the key variable and, for a CLI judge, its pinned version.
    pinned = None
    if ci == "api-key":
        _key_env(chosen)  # fail before writing anything if there is no key to map
        cli = AGENT_CLIS.get(chosen.provider)
        found = shutil.which(cli.binary) if cli else None
        pinned = cli_version(found) if found else None
    files_to_write = build_files(chosen, ci, skill, pinned)

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
        emit(f"  add the repository secret {_key_env(chosen)}, then commit evals/ and .github/")
    else:
        emit("  commit evals/ (including evals/runs/latest.json) with your change")
    return 0
