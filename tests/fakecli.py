"""Fake ``claude`` and ``codex`` binaries for the tests: no real CLI runs, no model is called.

``install`` writes a small Python script into ``tmp_path/bin`` and puts that directory first on
``PATH``. On a model call the script records its argv, cwd, stdin, the system-prompt file it was
given and the key environment variables, then prints a canned output. Canned outputs follow the
shapes in the 0.4.0 spec (real CLI output, trimmed, with synthetic values).
"""

from __future__ import annotations

import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from evalproof.providers.agent_cli import cli_version

ENV_KEYS = [
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "CLAUDE_CODE_OAUTH_TOKEN",
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
]

SCRIPT = """#!{python}
import json, os, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(__file__)
cfg = json.load(open(os.path.join(here, name + ".config.json")))
args = sys.argv[1:]
if args == ["--version"]:
    print(cfg["version"])
    sys.exit(0)
if args[:2] in (["auth", "status"], ["login", "status"]):
    sys.stdout.write(cfg["auth"]["stdout"])
    sys.exit(cfg["auth"]["exit"])

calls_path = os.path.join(here, name + ".calls.jsonl")
index = sum(1 for _ in open(calls_path)) if os.path.exists(calls_path) else 0
system_file = None
for i, arg in enumerate(args):
    if arg == "--system-prompt-file":
        system_file = args[i + 1]
    if arg.startswith("model_instructions_file="):
        system_file = arg.split("=", 1)[1]
record = {{
    "argv": args,
    "cwd": os.getcwd(),
    "cwd_files": sorted(os.listdir(os.getcwd())),
    "stdin": sys.stdin.read(),
    "system": open(system_file).read() if system_file else None,
    "env": {{k: os.environ.get(k) for k in cfg["env_keys"]}},
    "pid": os.getpid(),
}}
reply = cfg["responses"][min(index, len(cfg["responses"]) - 1)]
if reply.get("spawn_child"):
    child = subprocess.Popen(["sleep", "60"])
    record["child_pid"] = child.pid
with open(calls_path, "a") as fh:
    fh.write(json.dumps(record) + "\\n")
if reply.get("sleep"):
    time.sleep(reply["sleep"])
sys.stdout.write(reply.get("stdout", ""))
sys.stderr.write(reply.get("stderr", ""))
sys.exit(reply.get("exit", 0))
"""


@dataclass
class FakeCLI:
    path: Path
    name: str

    @property
    def calls_path(self) -> Path:
        return self.path.parent / f"{self.name}.calls.jsonl"

    def calls(self) -> list[dict[str, Any]]:
        if not self.calls_path.exists():
            return []
        lines = self.calls_path.read_text("utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def last(self) -> dict[str, Any]:
        return self.calls()[-1]

    def set_version(self, version: str) -> None:
        config_path = self.path.parent / f"{self.name}.config.json"
        config = json.loads(config_path.read_text("utf-8"))
        config["version"] = version
        config_path.write_text(json.dumps(config), "utf-8")
        cli_version.cache_clear()


def reply(
    stdout: str = "",
    *,
    stderr: str = "",
    exit: int = 0,
    sleep: float = 0,
    spawn_child: bool = False,
) -> dict[str, Any]:
    return {
        "stdout": stdout,
        "stderr": stderr,
        "exit": exit,
        "sleep": sleep,
        "spawn_child": spawn_child,
    }


def install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    version: str,
    responses: list[dict[str, Any]],
    auth_stdout: str = "",
    auth_exit: int = 0,
) -> FakeCLI:
    """Write the fake ``name`` binary and put its directory first on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(SCRIPT.format(python=sys.executable), "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    config = {
        "version": version,
        "responses": responses,
        "auth": {"stdout": auth_stdout, "exit": auth_exit},
        "env_keys": ENV_KEYS,
    }
    (bin_dir / f"{name}.config.json").write_text(json.dumps(config), "utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    cli_version.cache_clear()
    return FakeCLI(path, name)


# -- canned outputs ----------------------------------------------------------------------


def claude_result(
    text: str = "OK",
    *,
    model: str | None = "claude-haiku-5-5",
    is_error: bool = False,
    api_error_status: int | None = None,
    extra_models: dict[str, int] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": "result",
        "subtype": "success",  # stays "success" on errors too
        "is_error": is_error,
        "result": text,
        "stop_reason": "end_turn",
        "num_turns": 1,
        "api_error_status": api_error_status,
        "usage": {
            "input_tokens": 2,
            "cache_creation_input_tokens": 692,
            "cache_read_input_tokens": 10,
            "output_tokens": 4,
        },
    }
    usage_by_model: dict[str, Any] = {}
    if model:
        usage_by_model[model] = {"inputTokens": 2, "outputTokens": 4}
    for other, out_tokens in (extra_models or {}).items():
        usage_by_model[other] = {"inputTokens": 1, "outputTokens": out_tokens}
    if usage_by_model:
        result["modelUsage"] = usage_by_model
    return result


def claude_array(result: dict[str, Any]) -> str:
    """The 2.1.x shape: an array of events, the result last."""
    events = [
        {"type": "system", "subtype": "init", "claude_code_version": "2.1.322", "tools": []},
        {"type": "assistant", "message": {"model": "claude-haiku-5-5"}},
        {"type": "rate_limit_event"},
        result,
    ]
    return json.dumps(events)


def claude_ok(text: str = "OK", **kwargs: Any) -> dict[str, Any]:
    return reply(claude_array(claude_result(text, **kwargs)))


def claude_failure(status: int, message: str) -> dict[str, Any]:
    result = claude_result(message, model=None, is_error=True, api_error_status=status)
    return reply(claude_array(result), stderr='[claude-code:error] {"query_source":"sdk"}', exit=1)


def codex_events(text: str = "OK", *, noise: bool = False) -> str:
    lines = [
        '{"type":"thread.started","thread_id":"01a1d0fc-synthetic"}',
        '{"type":"turn.started"}',
        json.dumps(
            {
                "type": "item.completed",
                "item": {"id": "item_0", "type": "agent_message", "text": text},
            }
        ),
        json.dumps(
            {
                "type": "turn.completed",
                "usage": {
                    "input_tokens": 16150,
                    "cached_input_tokens": 12160,
                    "output_tokens": 104,
                    "reasoning_output_tokens": 86,
                },
            }
        ),
    ]
    if noise:
        lines.insert(1, "Reading additional input from stdin...")
        lines.insert(3, "WARN something unrelated {not json")
    return "\n".join(lines) + "\n"


def codex_ok(text: str = "OK", *, noise: bool = False) -> dict[str, Any]:
    return reply(codex_events(text, noise=noise))


def codex_failure(message: str, *, exit: int = 1) -> dict[str, Any]:
    lines = [
        '{"type":"thread.started","thread_id":"t"}',
        '{"type":"turn.started"}',
        json.dumps({"type": "error", "message": message}),
        json.dumps({"type": "turn.failed", "error": {"message": message}}),
    ]
    return reply(
        "\n".join(lines) + "\n", stderr="Reading additional input from stdin...\n", exit=exit
    )


def claude_auth() -> str:
    # The real status also holds an email and an organisation; the fake includes synthetic ones
    # so a test can prove they are not printed.
    return json.dumps(
        {
            "loggedIn": True,
            "authMethod": "claude.ai",
            "apiProvider": "firstParty",
            "subscriptionType": "team",
            "email": "someone@example.invalid",
            "orgName": "Synthetic Org",
        }
    )
