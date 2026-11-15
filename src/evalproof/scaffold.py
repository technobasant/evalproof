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
import textwrap
from collections.abc import Callable
from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path

from evalproof.config import ConfigError
from evalproof.dataset import dataset_text
from evalproof.importers import ImportedDataset, import_cases
from evalproof.providers import LLM_PROVIDERS, PROFILES, parse_target
from evalproof.providers.agent_cli import CLAUDE_OAUTH_ENV

# The tag of the composite action (``action.yml``) the generated workflows use. The
# ``evalproof`` version that ``run`` is pinned to in the api-key variant is the same release.
ACTION_REF = "v1.2.0"

CI_MODES = ("keyless", "api-key", "none")

# Where the generated CI lives. GitLab gets a file of its own that the user includes from
# ``.gitlab-ci.yml``; ``init`` never edits that file.
PLATFORMS = ("github", "gitlab")
GITLAB_CI_FILE = ".gitlab-ci.evalproof.yml"

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
            encoding="utf-8",
            errors="replace",
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


def render_ci_text(
    ci: str, judge: Judge, cli_auth: str = "api-key", platform: str = "github"
) -> str:
    gitlab = platform == "gitlab"
    if gitlab:
        file = f"`{GITLAB_CI_FILE}` (included from `.gitlab-ci.yml`)"
        request, base = "merge request", "target branch"
    else:
        file, request, base = "`.github/workflows/evalproof.yml`", "pull request", "base branch"
    if ci == "keyless":
        return (
            "Commit `evals/runs/latest.json` with your change. CI makes no model calls: "
            f"{file} compares that file with the one on the {base} "
            f"and fails the {request} on a regression that is both large enough to "
            "matter and statistically significant (exit 1; exit 3 over a latency or cost "
            "budget; exit 4 when it cannot decide). Before comparing it runs `evalproof verify`, "
            "which fails the job when `latest.json` does not match the suite and the cases in "
            "`cases.jsonl` (a stale file, a `--limit` run, an edited case). It also warns when "
            "`evals/` changed but `latest.json` did not."
        )
    if ci == "api-key":
        secret = _ci_secret(judge, cli_auth)
        if gitlab:
            return (
                "Commit `evals/runs/latest.json` on the main branch to set the baseline. "
                f"{file} runs the suite on a merge request that changes `evals/` (add the paths "
                "that hold your prompt to `EVALPROOF_WATCH` in the job) with the "
                f"`{secret}` CI/CD variable and compares the result with that file; "
                "it fails on a regression that is both large enough to matter and statistically "
                "significant (exit 1; exit 3 over a budget; exit 4 when it cannot decide). Merge "
                "requests from forks get no variables, so the job runs nothing for them. "
                "Responses are cached between runs."
            )
        text = (
            "Commit `evals/runs/latest.json` on the main branch to set the baseline. "
            f"{file} runs the suite on a pull request that changes "
            "`evals/` (add the paths that hold your prompt to `WATCH` in the workflow) with the "
            f"`{secret}` repository secret and compares the result with "
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


def render_gitlab_ci(ci: str, judge: Judge) -> str:
    """``.gitlab-ci.evalproof.yml``: the same gate as the GitHub workflow, on merge requests.

    Both variants run in ``python:3.13-slim`` and install evalproof from PyPI at the release
    this file was written by. A coding-agent CLI judge is refused before this is called
    (``init_project``): the image has no Node to install the CLI with.
    """
    values = {"PIP_VERSION": ACTION_REF.removeprefix("v")}
    if ci == "keyless":
        return fill(template("workflow-gitlab-keyless.yml"), values)
    values["KEY_VAR"] = _key_env(judge)
    return fill(template("workflow-gitlab-api-key.yml"), values)


# --- the imported suite ------------------------------------------------------------------


def render_import_suite(
    dataset: ImportedDataset, judge: Judge, *, source: str, dataset_path: str = "cases.jsonl"
) -> str:
    """A suite for imported cases: exactly the scorers the import needs, and the ``judge`` block
    only when one of them is an ``llm_judge``. Raises ``ConfigError`` when no scorer fits."""
    if not dataset.scorers:
        raise ConfigError(
            "no scorer fits these cases: they carry no expected answer and no assertion that "
            "maps to one, so there is nothing to check an answer against. Write the cases "
            "without a suite and add scorers by hand (docs: guides/import-test-cases)"
        )
    blocks = []
    for scorer in dataset.scorers:
        checked = dataset.coverage.get(scorer.name)
        lines = [f"  # checks {checked} of {len(dataset.cases)} case(s)"] if checked else []
        lines += [f"  - name: {scorer.name}", f"    type: {scorer.type}"]
        if scorer.params:
            lines.append(f"    params: {json.dumps(scorer.params)}")
        blocks.append("\n".join(lines))
    template_line = (
        f"  prompt_template: {json.dumps(dataset.prompt_template)}"
        if dataset.prompt_template
        else ""
    )
    local = judge.provider in ("ollama", "vllm", "lmstudio", "llamacpp")
    notes = "\n".join(f"# {line}" for note in dataset.notes for line in _wrap(note))
    return fill(
        template("suite-import.yaml"),
        {
            "SOURCE": source,
            "FORMAT": dataset.format,
            "NOTES": f"#\n# Import notes:\n{notes}" if notes else "",
            "NAME": "imported",
            "DATASET": json.dumps(dataset_path),
            "CONCURRENCY": "1" if local else "4",
            "TARGET": judge.target,
            "TARGET_NOTE": f"  # {judge.note}" if judge.note else "",
            "API_KEY_LINE": f"  api_key_env: {judge.api_key_env}" if judge.api_key_env else "",
            "TEMPLATE_LINE": template_line,
            "SCORERS": "\n".join(blocks),
            "METRIC": dataset.gate_metric or dataset.scorers[0].name,
        },
        frozenset({"judge"}) if dataset.needs_judge else frozenset(),
    )


def _wrap(note: str, width: int = 94) -> list[str]:
    """``note`` as comment lines no longer than ``width`` (the first begins with ``-``)."""
    lines = textwrap.wrap(note, width=width, initial_indent="- ", subsequent_indent="  ")
    return lines or [note]


# --- the agent files ---------------------------------------------------------------------

# The evalproof section of the repository's AGENTS.md sits between these two markers. ``init``
# replaces what is between them and never touches the rest of the file.
AGENTS_FILE = "AGENTS.md"
AGENTS_BEGIN = "<!-- evalproof:begin -->"
AGENTS_END = "<!-- evalproof:end -->"
AGENTS_HEADING = "# Agent instructions"

# Two copies of the skill, not a symlink, so a Windows checkout works.
SKILL_FILES = (".agents/skills/evalproof/SKILL.md", ".claude/skills/evalproof/SKILL.md")


def agents_block() -> str:
    """The evalproof section, from the begin marker to the end marker, with LF line ends."""
    text = template(AGENTS_FILE).replace("\r\n", "\n").strip()
    if not (text.startswith(AGENTS_BEGIN) and text.endswith(AGENTS_END)):
        raise RuntimeError("templates/AGENTS.md must start and end with the evalproof markers")
    return text


def merge_agents_md(existing: str | None) -> tuple[str, str]:
    """Put the evalproof section into the text of an AGENTS.md.

    ``existing`` is None for a file that does not exist. Returns the new text and what was done:
    ``created`` (a heading and the section), ``updated`` (the text between the markers was
    replaced), ``appended`` (the file had no markers, so the section follows its content after
    one blank line) or ``unchanged``. Text outside the markers is never changed, byte for byte,
    and the section uses the newline style the file already has. Damaged markers (one without
    the other, repeated, or in the wrong order) raise ``ConfigError``: nothing is guessed.
    """
    block = agents_block()
    if not existing:
        return f"{AGENTS_HEADING}\n\n{block}\n", "created"
    nl = "\r\n" if existing.count("\r\n") * 2 > existing.count("\n") else "\n"
    block = block.replace("\n", nl)
    begins, ends = existing.count(AGENTS_BEGIN), existing.count(AGENTS_END)
    if begins == 0 and ends == 0:
        trailing = existing[len(existing.rstrip("\r\n")) :].count("\n")
        return existing + nl * max(0, 2 - trailing) + block + nl, "appended"
    if begins == 1 and ends == 1:
        start, stop = existing.index(AGENTS_BEGIN), existing.index(AGENTS_END)
        if start < stop:
            merged = existing[:start] + block + existing[stop + len(AGENTS_END) :]
            return merged, ("unchanged" if merged == existing else "updated")
    raise ConfigError(
        f"{AGENTS_FILE} has damaged evalproof markers: {begins} x {AGENTS_BEGIN} and "
        f"{ends} x {AGENTS_END}. Each must appear once, the begin marker first. Fix or remove "
        "them by hand. Nothing was written."
    )


def plan_agents_md(root: Path) -> tuple[str, str]:
    """The new text of ``root/AGENTS.md`` and what writing it does, without writing."""
    path = root / AGENTS_FILE
    existing = None
    if path.exists():
        try:
            existing = path.read_bytes().decode("utf-8")  # bytes, so a CRLF file stays CRLF
        except (OSError, UnicodeDecodeError) as exc:
            raise ConfigError(
                f"cannot edit {AGENTS_FILE}: it is not readable UTF-8 text ({exc})"
            ) from exc
    return merge_agents_md(existing)


_AGENTS_VERB = {
    "created": "wrote",
    "appended": "appended",
    "updated": "updated",
    "unchanged": "unchanged",
}


def write_agents_md(root: Path, plan: tuple[str, str], emit: Callable[[str], None]) -> None:
    """Write what ``plan_agents_md`` returned and say what was done."""
    text, action = plan
    if action != "unchanged":
        root.mkdir(parents=True, exist_ok=True)
        # newline="": the text already has the file's own line ends, so none are translated.
        (root / AGENTS_FILE).write_text(text, encoding="utf-8", newline="")
    emit(f"  {_AGENTS_VERB[action]:<15}{AGENTS_FILE}")


def skill_files() -> list[tuple[str, str]]:
    """The skill, once per place coding agents look for it."""
    body = template("SKILL.md")
    return [(rel, body) for rel in SKILL_FILES]


def _write_files(
    root: Path, files: list[tuple[str, str]], force: bool, emit: Callable[[str], None]
) -> list[str]:
    """Write each file that is missing (or all with ``force``); returns the ones kept."""
    kept = []
    for rel, content in files:
        path = root / rel
        if path.exists() and not force:
            emit(f"  kept (exists)  {rel}")
            kept.append(rel)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")
        emit(f"  wrote          {rel}")
    return kept


def build_files(
    judge: Judge,
    ci: str,
    skill: bool,
    pinned: str | None,
    cli_auth: str = "api-key",
    platform: str = "github",
    imported: tuple[ImportedDataset, str] | None = None,
) -> list[tuple[str, str]]:
    """Every file to write, as (path relative to the target directory, content).

    ``imported`` is an import and the name of the file it came from: the cases and the suite's
    scorers then come from it instead of the three sample cases.

    ``AGENTS.md`` is not in the list: it is merged into an existing file, not written whole
    (``plan_agents_md``)."""
    ci_text = render_ci_text(ci, judge, cli_auth, platform)
    if imported is None:
        suite_text, cases_text = render_suite(judge), template("cases.jsonl")
    else:
        suite_text = render_import_suite(imported[0], judge, source=imported[1])
        cases_text = dataset_text(imported[0].cases)
    out = [
        ("evals/suite.yaml", suite_text),
        ("evals/cases.jsonl", cases_text),
        ("evals/README.md", fill(template("evals-readme.md"), {"CI_TEXT": ci_text})),
        ("evals/.gitignore", template("gitignore")),
    ]
    if ci != "none":
        if platform == "gitlab":
            out.append((GITLAB_CI_FILE, render_gitlab_ci(ci, judge)))
        else:
            out.append(
                (".github/workflows/evalproof.yml", render_workflow(ci, judge, pinned, cli_auth))
            )
    if skill:
        out.extend(skill_files())
    return out


def _hosted_key(judge: Judge) -> str | None:
    """The environment variable a hosted judge reads its API key from; None for a local one."""
    if judge.api_key_env:
        return judge.api_key_env
    profile = PROFILES.get(judge.provider)
    if profile is None or profile.local:
        return None
    return profile.api_key_env


def requirement_lines(
    judge: Judge, ci: str, cli_auth: str = "api-key", platform: str = "github"
) -> list[str]:
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
    elif platform == "gitlab":
        remote = f"the {_key_env(judge)} CI/CD variable, masked and not protected"
    else:
        remote = f"the {_key_env(judge)} repository secret"
    lines = [f"locally: {local}", f"in CI: {remote}"]
    if ci == "api-key":
        if platform == "gitlab":
            lines.append(
                "merge requests from forks never receive the variable, so the job skips them"
            )
        else:
            lines.append(
                "pull requests from forks never receive secrets, so the workflow skips them"
            )
    return lines


def _init_agent_files(
    directory: Path,
    *,
    judge: str | None,
    ci: str,
    skill: bool,
    force: bool,
    cli_auth: str,
    platform: str,
    agents_md: bool,
    emit: Callable[[str], None],
) -> int:
    """``init --agent-files-only``: the AGENTS.md section and the skill, and nothing else."""
    if judge is not None:
        raise ConfigError("--agent-files-only writes no suite, so --judge has nothing to set")
    if ci != "keyless" or platform != "github" or cli_auth != "api-key":
        raise ConfigError(
            "--agent-files-only writes no CI workflow, so it cannot be combined with "
            "--ci, --platform or --cli-auth"
        )
    if not agents_md and not skill:
        raise ConfigError(
            "--agent-files-only with both --no-agents-md and --no-skill has nothing to write"
        )
    root = directory.resolve()
    agents_plan = plan_agents_md(root) if agents_md else None

    emit("evalproof init: agent files only (no suite, no CI workflow)")
    files = skill_files() if skill else []
    kept = _write_files(root, files, force, emit)
    if agents_plan:
        write_agents_md(root, agents_plan, emit)
    for rel, content in files:
        if rel in kept:
            on_disk = (root / rel).read_bytes().decode("utf-8", "replace").replace("\r\n", "\n")
            if on_disk != content:
                emit(f"note: {rel} differs from this version of evalproof; --force replaces it")
    return 0


def init_project(
    directory: Path,
    *,
    judge: str | None = None,
    ci: str = "keyless",
    skill: bool = True,
    force: bool = False,
    cli_auth: str = "api-key",
    platform: str = "github",
    agents_md: bool = True,
    agent_files_only: bool = False,
    from_file: Path | None = None,
    from_format: str = "auto",
    emit: Callable[[str], None] = print,
) -> int:
    """Write the starter files into ``directory``; returns the exit code (always 0).

    ``agents_md`` also puts the evalproof section into the repository's ``AGENTS.md``.
    ``agent_files_only`` writes only the agent files (that section and the skill) and nothing else.
    ``from_file`` is a file of test cases from another tool (``from_format`` says which, ``auto``
    reads it off the file): ``evals/cases.jsonl`` is those cases and ``evals/suite.yaml`` has
    the scorers they need, in place of the three sample cases. Nothing is written when the file
    cannot be imported.
    """
    if from_file is not None and agent_files_only:
        raise ConfigError(
            "--from imports test cases into a suite, and --agent-files-only writes no suite: "
            "use one or the other"
        )
    if agent_files_only:
        return _init_agent_files(
            directory,
            judge=judge,
            ci=ci,
            skill=skill,
            force=force,
            cli_auth=cli_auth,
            platform=platform,
            agents_md=agents_md,
            emit=emit,
        )
    if ci not in CI_MODES:
        raise ConfigError(f"--ci must be one of {', '.join(CI_MODES)}, got {ci!r}")
    if cli_auth not in CLI_AUTHS:
        raise ConfigError(f"--cli-auth must be one of {', '.join(CLI_AUTHS)}, got {cli_auth!r}")
    if platform not in PLATFORMS:
        raise ConfigError(f"--platform must be one of {', '.join(PLATFORMS)}, got {platform!r}")
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
    if platform == "gitlab" and ci == "api-key" and chosen.provider in AGENT_CLIS:
        raise ConfigError(
            f"--platform gitlab --ci api-key runs in python:3.13-slim, which has no Node to "
            f"install the {AGENT_CLIS[chosen.provider].label} CLI with, and the judge is "
            f"{chosen.target!r}. Pass --judge with a hosted API target such as "
            f"{ANTHROPIC_DEFAULT}, or use --ci keyless."
        )
    # An api-key workflow needs the key variable and, for a CLI judge, its pinned version.
    pinned = None
    if ci == "api-key":
        _key_env(chosen)  # fail before writing anything if there is no key to map
        cli = AGENT_CLIS.get(chosen.provider)
        found = shutil.which(cli.binary) if cli else None
        pinned = cli_version(found) if found else None
    imported = None
    notes: list[str] = []
    if from_file is not None:
        imp = import_cases(from_format, from_file, warn=notes.append)
        imported = (replace(imp, notes=[*imp.notes, *notes]), from_file.name)
    files_to_write = build_files(chosen, ci, skill, pinned, cli_auth, platform, imported)
    root = directory.resolve()
    # Merge into AGENTS.md before writing anything, so damaged markers stop init with no change.
    agents_plan = plan_agents_md(root) if agents_md else None

    emit(f"evalproof init: judge {chosen.target} ({chosen.reason})")
    if imported is not None:
        dataset, name = imported
        how = "detected" if dataset.detected else "format"
        emit(f"imported {len(dataset.cases)} case(s) from {name} ({how}: {dataset.format})")
        for note in dataset.notes:
            emit(f"  note: {note}")
    _write_files(root, files_to_write, force, emit)
    if agents_plan:
        write_agents_md(root, agents_plan, emit)

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
    gitlab = platform == "gitlab"
    if ci == "api-key":
        secret = _ci_secret(chosen, cli_auth)
        if gitlab:
            emit(
                f"  add the CI/CD variable {secret} (masked, not protected), "
                f"then commit evals/ and {GITLAB_CI_FILE}"
            )
        else:
            emit(f"  add the repository secret {secret}, then commit evals/ and .github/")
    elif gitlab and ci == "keyless":
        emit(
            f"  commit evals/ (including evals/runs/latest.json) and {GITLAB_CI_FILE} "
            "with your change"
        )
    else:
        emit("  commit evals/ (including evals/runs/latest.json) with your change")
    if gitlab and ci != "none":
        emit(f"  {GITLAB_CI_FILE} runs only when .gitlab-ci.yml includes it. init never edits")
        emit("  that file, so add these two lines to it (create it if you have none):")
        emit("    include:")
        emit(f"      - local: {GITLAB_CI_FILE}")
    emit("")
    emit("what it needs to run:")
    for line in requirement_lines(chosen, ci, cli_auth, platform):
        emit(f"  {line}")
    return 0
