"""``evalproof doctor``: is each coding-agent CLI target in a suite ready to use?

Free by construction: it looks up the binary, runs ``--version`` and the CLI's own auth-status
command, and never sends a prompt. The auth check prints only fields that identify the kind of
login (not an account, email or organisation).
"""

from __future__ import annotations

import os
import shlex
import shutil
import sys
from typing import TextIO

from evalproof.config import GenerationParams, TargetConfig, load_config
from evalproof.providers import parse_target
from evalproof.providers.agent_cli import AUTH_ENV, CLI_PROVIDERS, cli_version
from evalproof.providers.agent_cli_experimental import EXPERIMENTAL_PROVIDERS
from evalproof.providers.base import ProviderError

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
    return ok


def run_doctor(suite: str | None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    targets = suite_targets(suite)
    if not targets:
        out.write("no coding-agent CLI targets in this suite; nothing to check\n")
        return 0
    results = [check_target(cfg, roles, out) for cfg, roles in targets]
    out.write("ok\n" if all(results) else "not ready: fix the problems above\n")
    return 0 if all(results) else 1
