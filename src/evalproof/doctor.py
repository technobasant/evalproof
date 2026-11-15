"""``evalproof doctor``: is each coding-agent CLI target in a suite ready to use?

Free by construction: it looks up the binary, runs ``--version`` and the CLI's own auth-status
command, and never sends a prompt. The auth check prints only fields that identify the kind of
login (not an account, email or organisation).

The one exception is ``--cursor-canary`` (``run_cursor_canary``): it makes a single real
``cursor-agent`` call, on request, to see what reaches the prompt.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import sys
from typing import TextIO

from evalproof.config import GenerationParams, TargetConfig, load_config
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, EXIT_REGRESSION
from evalproof.providers import parse_target
from evalproof.providers.agent_cli import (
    AUTH_ENV,
    CLAUDE_OAUTH_ENV,
    CLI_PROVIDERS,
    AgentCLIProvider,
    cli_version,
)
from evalproof.providers.agent_cli_experimental import (
    CURSOR_SANDBOX_ENV,
    EXPERIMENTAL_PROVIDERS,
    CursorProvider,
    cursor_sandbox_state,
)
from evalproof.providers.base import Message, ProviderError

# What to check when no suite is given: one target per CLI, so the model is only a placeholder.
DEFAULT_TARGETS = ("claude-code:<model>", "codex:<model>")


def installed_experimental() -> list[str]:
    """The experimental providers whose binary is on ``PATH``: the ones worth listing when no
    suite says what to check."""
    return [
        f"{name}:<model>"
        for name, cls in EXPERIMENTAL_PROVIDERS.items()
        if shutil.which(cls.binary) is not None
    ]


def suite_targets(path: str | None) -> list[tuple[TargetConfig, list[str]]]:
    """The CLI targets of a suite (or claude-code, codex and whichever experimental CLIs are on
    ``PATH`` when there is no suite), each with the roles that use it."""
    if path is None:
        return [(TargetConfig(target=t), []) for t in (*DEFAULT_TARGETS, *installed_experimental())]
    cfg = load_config(path)
    found: dict[str, tuple[TargetConfig, list[str]]] = {}
    for role, target in (("system", cfg.system), ("judge", cfg.judge)):
        if target is None or target.kind not in CLI_PROVIDERS:
            continue
        key = f"{target.target}|{target.api_key_env}"
        found.setdefault(key, (target, []))[1].append(role)
    return list(found.values())


CURSOR_SKILLS_WARNING = (
    "cursor-agent puts the names and descriptions of your skills (~/.claude/skills, "
    "~/.cursor/skills, ~/.codex/skills, ~/.agents/skills, plugin skills) in every "
    "prompt; no option turns this off, so they are sent to Cursor and count as input "
    "tokens on each call"
)
CURSOR_SANDBOXED_WARNING = (
    "cursor-agent runs in a macOS sandbox that blocks reads of your skill and rule folders "
    "(~/.claude, ~/.agents, ~/.codex, ~/.gemini, ~/.cursor/skills, skills-cursor, rules, "
    "plugins, agents); your Cursor account's User Rules are still sent with every prompt (they "
    "come from Cursor's servers): clear them in Cursor Settings > Rules for a clean run"
)


def cursor_warning(state: str) -> str:
    """The cursor limit to report, given ``cursor_sandbox_state()``."""
    if state == "sandboxed":
        return CURSOR_SANDBOXED_WARNING
    if state == "off":
        return (
            f"{CURSOR_SKILLS_WARNING}; on macOS, {CURSOR_SANDBOX_ENV}=on (the default) blocks them"
        )
    return (
        f"{CURSOR_SKILLS_WARNING}; no sandbox is available here (it needs macOS with "
        "/usr/bin/sandbox-exec)"
    )


def experimental_warnings(kind: str, model: str, roles: list[str]) -> list[str]:
    """Known limits of an experimental CLI that a user should hear about before a run."""
    warnings: list[str] = []
    if kind == "cursor":
        warnings.append(cursor_warning(cursor_sandbox_state()))
    if kind == "copilot" and model == "auto" and "judge" in roles:
        warnings.append(
            "copilot:auto as the judge: Copilot picks the model per call and the run records "
            "every model it saw, so two runs often differ and compare and gate refuse them as "
            "a judge change (exit 4) unless the gate sets allow_judge_change; name a model if "
            "your plan allows one"
        )
    return warnings


def credential_lines(kind: str, cfg: TargetConfig, provider: AgentCLIProvider) -> list[str]:
    """Which credential the target would use here, and whether that works in CI.

    Looks only at the names of environment variables (``name in os.environ``), never at a
    value, so nothing secret can be printed or logged. A login is a file on this machine, so it
    is never available in CI; a key or token can be, from a secret."""
    secret = provider.strip_vars[0]  # the variable a CI secret is mapped to
    ci_key = f"map a repository secret to {secret} and set {AUTH_ENV}=api-key"
    if cfg.api_key_env:
        env = cfg.api_key_env
        return [
            f"credential: ${env} (the suite's api_key_env), passed as {provider.key_var}",
            f"in CI: usable; map a repository secret to {env}",
        ]
    if provider.auth_mode == "api-key":
        present = [v for v in provider.strip_vars if v in os.environ]
        have = (
            f"{', '.join(present)}; set here"
            if present
            else (f"{' or '.join(provider.strip_vars)}; not set here")
        )
        return [f"credential: {have} ({AUTH_ENV}=api-key)", f"in CI: usable; {ci_key}"]
    if kind == "claude-code" and CLAUDE_OAUTH_ENV in os.environ:
        return [
            f"credential: {CLAUDE_OAUTH_ENV}; set here, and login mode leaves it in place",
            "in CI: usable from a secret, in your own repository and CI only; for a secret "
            f"shared across a team or organisation, {ci_key}",
        ]
    login = f"credential: your {kind} login on this machine"
    use_key = f"in CI: login only, not available in CI; use an API key ({ci_key})"
    if kind == "claude-code":
        use_key += (
            f", or, in your own repository and CI only, {CLAUDE_OAUTH_ENV} from "
            "`claude setup-token`"
        )
    elif kind == "codex":
        use_key += (
            "; a personal Codex login belongs only on trusted private infrastructure, never "
            "on public repositories or for open source work"
        )
    return [login, use_key]


def check_target(cfg: TargetConfig, roles: list[str], out: TextIO) -> bool:
    """Print what is known about one target; True when it is usable."""
    kind, model = parse_target(cfg.target)
    # Without a suite the model is a placeholder, so only the CLI is named.
    out.write(f"{cfg.target} ({', '.join(roles)})\n" if roles else f"{kind}\n")
    try:
        provider = CLI_PROVIDERS[kind](
            model,
            base_url=cfg.base_url,
            timeout_s=cfg.timeout_s,
            api_key_env=cfg.api_key_env,
            headers=cfg.headers,
        )
        binary = provider.binary_path()
    except ProviderError as exc:
        out.write(f"  problem: {exc}\n")
        return False
    out.write(f"  binary: {binary}\n")
    ok = True
    try:
        out.write(f"  version: {cli_version(binary)}\n")
    except ProviderError as exc:
        out.write(f"  problem: {exc}\n")
        ok = False
    key_vars = [v for v in provider.strip_vars if v in os.environ]
    # A key reaches the CLI when the suite names one, or in api-key mode when one is set.
    key_in_use = bool(cfg.api_key_env) or (provider.auth_mode == "api-key" and bool(key_vars))
    auth = provider.auth_status()
    out.write(f"  auth: {auth.detail}\n")
    # With a key in use the CLI need not be logged in, so a failed login check is not a problem.
    ok = ok and (auth.ok or key_in_use)
    command = provider.command(provider.binary, "<tmp>/system.txt", GenerationParams())
    out.write(f"  isolation: {provider.isolation}\n")
    out.write(f"  flags: {shlex.join(command[1:])}\n")
    source = f"from {AUTH_ENV}" if os.environ.get(AUTH_ENV, "").strip() else f"{AUTH_ENV} not set"
    out.write(f"  auth mode: {provider.auth_mode} ({source})\n")
    if kind in EXPERIMENTAL_PROVIDERS:
        out.write("  note: experimental provider; its isolation is less proven (see the README)\n")
        if provider.extra_env:
            out.write(f"  env: {' '.join(f'{k}={v}' for k, v in provider.extra_env.items())}\n")
        for line in experimental_warnings(kind, model, roles):
            out.write(f"  warning: {line}\n")
    if cfg.api_key_env:
        out.write(f"  key: ${cfg.api_key_env} is passed to the CLI as {provider.key_var}\n")
    elif provider.auth_mode == "api-key":
        if key_vars:
            out.write(f"  key: {', '.join(key_vars)} is passed to the CLI\n")
        else:
            out.write(
                f"  warning: {AUTH_ENV}=api-key but {' and '.join(provider.strip_vars)} "
                f"is not set, so the {kind} CLI has no key to use\n"
            )
    elif key_vars:
        out.write(
            f"  note: {', '.join(key_vars)} is set and is removed for this call, so the "
            f"{kind} login is used (name the variable in api_key_env, or set "
            f"{AUTH_ENV}=api-key, to use a key)\n"
        )
    for line in credential_lines(kind, cfg, provider):
        out.write(f"  {line}\n")
    return ok


def run_doctor(suite: str | None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    targets = suite_targets(suite)
    if not targets:
        out.write("no coding-agent CLI targets in this suite; nothing to check\n")
        return 0
    results = [check_target(cfg, roles, out) for cfg, roles in targets]
    if suite is None:
        # A survey of the machine: fine when any CLI is usable; the others are listed.
        names = [parse_target(cfg.target)[0] for cfg, _ in targets]
        ready = [n for n, ok in zip(names, results, strict=True) if ok]
        missing = [n for n, ok in zip(names, results, strict=True) if not ok]
        if ready:
            out.write(f"ready: {', '.join(ready)}\n")
            if missing:
                out.write(f"not ready (unused unless a suite names it): {', '.join(missing)}\n")
            return 0
        out.write("not ready: no coding-agent CLI is usable; fix the problems above\n")
        return 1
    out.write("ok\n" if all(results) else "not ready: fix the problems above\n")
    return 0 if all(results) else 1


# -- the cursor canary -----------------------------------------------------------------------------

CANARY_MODEL = "composer-2.5-fast"
CANARY_TIMEOUT_S = 180.0
CANARY_SKILL_NAME = "evalproof-canary"
# Where cursor-agent looks for project skills, inside the call's own workspace.
CANARY_SKILL_PATH = f".claude/skills/{CANARY_SKILL_NAME}/SKILL.md"
CANARY_SKILL = (
    f"---\nname: {CANARY_SKILL_NAME}\ndescription: Canary skill used by evalproof doctor.\n---\n\n"
    "This skill exists only so that evalproof doctor can tell whether a skill reaches the "
    "prompt. Never use it.\n"
)
CANARY_PROMPT = (
    "Answer from the context you were given. Do not use any tool and do not read any file. "
    "Reply with exactly two lines and nothing else.\n"
    "Line 1: `SKILLS:` followed by the name of every skill listed in your context, separated by "
    "commas, or NONE.\n"
    "Line 2: `RULES:` followed by the title of every user_rule block in your context, separated "
    "by commas (titles only, never the contents), or NONE."
)
CANARY_SHOWN_ITEMS = 20
CANARY_SHOWN_CHARS = 60

_CANARY_LABEL = re.compile(
    r"(?im)^[^\w\n]*(?:line\s*[12]\b[^\w\n]*)?(skills?|(?:user[_ ]?)?rules?)\b[^:\n]*:"
)
_CANARY_NAME = re.compile(r"[\w][\w.\-/]*(?::[\w][\w.\-/]*)*")


_CANARY_BULLET = re.compile(r"^\s*(?:[-*•>]|\d+[.)])\s+\S")
_CANARY_NONE = re.compile(r"none(?![\w\-/:]|\.\w)", re.IGNORECASE)


def _canary_items(section: str, *, whole: bool = False) -> list[str]:
    """Names from one answer section: the label line split on commas and semicolons, then any
    bulleted lines under it (other lines are prose and are ignored); bullets, quotes and a
    trailing full stop removed, ``NONE`` dropped. A skill keeps only its name (its first word,
    and text in brackets is cut, so a description does not count); with ``whole`` an item is
    kept as written, for rule titles, which have spaces."""
    first, *rest = section.split("\n")
    if not whole:
        first = re.sub(r"\([^)]*\)", "", first)  # a description in brackets may hold commas
    parts = re.split(r"[,;]", first) + [ln for ln in rest if _CANARY_BULLET.match(ln)]
    items: list[str] = []
    for part in parts:
        part = re.sub(r"^[\s\-*•>`'\"/]+(?:\d+[.)]\s+)?", "", part)
        if _CANARY_NONE.match(part):
            continue
        if whole:
            item = part.strip().strip("`'\"*").rstrip(".:").strip()
        else:
            found = _CANARY_NAME.match(part)
            item = found.group(0).rstrip(".:") if found else ""
        if item:
            items.append(item)
    return items


def parse_canary_reply(text: str) -> tuple[list[str], list[str], bool]:
    """The skill names, the user-rule titles and whether the canary was named, from a reply that
    should have two lines (``SKILLS: ...`` and ``RULES: ...``). Lenient: the labels may carry
    markdown or a ``Line 1:`` prefix, a list may run over several bulleted lines, text before a
    lone ``RULES:`` label counts as the skills, and with no label at all the first line is the
    skills and the second the rules."""
    labelled: dict[str, re.Match[str]] = {}
    for found in _CANARY_LABEL.finditer(text):
        key = "skills" if found.group(1).lower().startswith("skill") else "rules"
        labelled.setdefault(key, found)
    if not labelled:
        lines = [ln for ln in text.splitlines() if ln.strip()]
        skills = lines[0] if lines else ""
        rules = lines[1] if len(lines) > 1 else ""
    else:
        ordered = sorted(labelled.items(), key=lambda item: item[1].start())
        sections = {
            key: text[match.end() : ordered[i + 1][1].start() if i + 1 < len(ordered) else None]
            for i, (key, match) in enumerate(ordered)
        }
        skills, rules = sections.get("skills", ""), sections.get("rules", "")
        if "skills" not in sections:
            skills = text[: labelled["rules"].start()]
    seen = CANARY_SKILL_NAME in skills.lower()
    names = [n for n in _canary_items(skills) if n.lower() != CANARY_SKILL_NAME]
    return names, _canary_items(rules, whole=True), seen


def _shown(items: list[str]) -> str:
    shown = [
        i if len(i) <= CANARY_SHOWN_CHARS else i[: CANARY_SHOWN_CHARS - 1] + "…" for i in items
    ]
    more = len(shown) - CANARY_SHOWN_ITEMS
    text = ", ".join(shown[:CANARY_SHOWN_ITEMS])
    return text + (f" and {more} more" if more > 0 else "")


def run_cursor_canary(
    model: str | None = None, out: TextIO | None = None, *, timeout_s: float = CANARY_TIMEOUT_S
) -> int:
    """One real ``cursor-agent`` call, through the provider's normal path (the same flags, the
    same sandbox, the same tool guard), with a canary skill in the call's workspace. The model is
    asked which skills and user rules it can see: the canary must be there, any other skill is a
    leak, and the titles of user rules are listed. The answer is the model's own report, so the
    result is best effort. Rule contents are never printed.

    Exit 0 when the canary is seen and no other skill is named, 1 otherwise, 4 when the call
    failed and there is no verdict."""
    out = out or sys.stdout
    model = model or CANARY_MODEL
    provider = CursorProvider(
        model,
        timeout_s=timeout_s,
        extra_workspace_files={CANARY_SKILL_PATH: CANARY_SKILL},
    )
    state = cursor_sandbox_state()
    out.write(f"cursor canary: model {model}, isolation {provider.isolation}, sandbox {state}\n")
    out.write("note: this makes one real cursor-agent call and uses your Cursor quota\n")
    try:
        reply = provider.chat([Message(role="user", content=CANARY_PROMPT)], GenerationParams())
    except ProviderError as exc:
        out.write(f"canary call failed: {exc}\n")
        out.write("no verdict: the call did not return an answer to check\n")
        return EXIT_CANNOT_GATE
    leaked, rules, seen = parse_canary_reply(reply.text)
    out.write(f"canary skill seen: {'yes' if seen else 'no'}\n")
    out.write(f"other skills seen: {_shown(leaked) if leaked else 'none'}\n")
    out.write(f"user rules seen (titles only): {_shown(rules) if rules else 'none'}\n")
    if rules:
        out.write(
            "user rules come from your Cursor account; clear them in Cursor Settings > Rules\n"
        )
    if not seen:
        out.write(
            "not conclusive: the model did not report the canary skill, so what it listed "
            "cannot be trusted\n"
        )
    elif leaked:
        hint = f"; set {CURSOR_SANDBOX_ENV}=on to block them on macOS" if state == "off" else ""
        out.write(f"leak: {len(leaked)} skill(s) besides the canary reached the prompt{hint}\n")
    else:
        out.write("ok: only the canary skill reached the prompt\n")
    out.write(
        "note: the answer is what the model says about its own context, so this is a best "
        "effort check, not proof\n"
    )
    return EXIT_PASS if seen and not leaked else EXIT_REGRESSION
