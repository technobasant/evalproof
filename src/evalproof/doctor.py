"""``evalproof doctor``: is each coding-agent CLI target in a suite ready to use?

Free by construction: it looks up the binary, runs ``--version`` and the CLI's own auth-status
command, and never sends a prompt. The auth check prints only fields that identify the kind of
login (not an account, email or organisation).
"""

from __future__ import annotations

import os
import shlex
import sys
from typing import TextIO

from evalproof.config import GenerationParams, TargetConfig, load_config
from evalproof.providers import parse_target
from evalproof.providers.agent_cli import CLI_PROVIDERS, cli_version
from evalproof.providers.base import ProviderError

# What to check when no suite is given: one target per CLI, so the model is only a placeholder.
DEFAULT_TARGETS = ("claude-code:<model>", "codex:<model>")


def suite_targets(path: str | None) -> list[tuple[TargetConfig, list[str]]]:
    """The CLI targets of a suite (or both CLIs when there is no suite), each with the roles
    that use it."""
    if path is None:
        return [(TargetConfig(target=t), []) for t in DEFAULT_TARGETS]
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
    auth = provider.auth_status()
    out.write(f"  auth: {auth.detail}\n")
    ok = ok and auth.ok
    command = provider.command(provider.binary, "<tmp>/system.txt", GenerationParams())
    out.write(f"  isolation: {provider.isolation}\n")
    out.write(f"  flags: {shlex.join(command[1:])}\n")
    if cfg.api_key_env:
        out.write(f"  key: ${cfg.api_key_env} is passed to the CLI as {provider.key_var}\n")
    else:
        stripped = [v for v in provider.strip_vars if v in os.environ]
        if stripped:
            out.write(
                f"  note: {', '.join(stripped)} is set and is removed for this call, so the "
                f"{kind} login is used (name the variable in api_key_env to use a key)\n"
            )
    return ok


def run_doctor(suite: str | None, out: TextIO | None = None) -> int:
    out = out or sys.stdout
    targets = suite_targets(suite)
    if not targets:
        out.write("no claude-code or codex targets in this suite; nothing to check\n")
        return 0
    results = [check_target(cfg, roles, out) for cfg, roles in targets]
    out.write("ok\n" if all(results) else "not ready: fix the problems above\n")
    return 0 if all(results) else 1
