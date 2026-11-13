"""Fake ``opencode``, ``cursor-agent`` and ``copilot`` binaries: no real CLI runs, no model call.

A sibling of ``fakecli`` (a separate module so the two do not collide). The script records its
argv, cwd, the files in it (with their text), stdin, the environment variables the test asks for
and the contents of the directory ``OPENCODE_CONFIG_DIR`` names, then prints a canned output. A
reply may also name text for the file ``--usage-output-file`` points at, as Copilot writes one.
Besides ``--version`` it answers one configurable status command (``cursor-agent about``). Canned
outputs follow the shapes seen from the real CLIs (OpenCode against a local ollama model, Cursor's
stream-json and Copilot's JSONL on 2026-11-13), trimmed, with synthetic values.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from evalproof.providers.agent_cli import cli_version
from tests.fakecli import FakeCLI, reply

ENV_KEYS = [
    "OPENCODE_API_KEY",
    "OPENCODE_CONFIG_CONTENT",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_DISABLE_PROJECT_CONFIG",
    "OPENCODE_CONFIG_PROJECT_DISABLE",
    "OPENCODE_DISABLE_CLAUDE_CODE",
    "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS",
    "ANTHROPIC_API_KEY",
    "CURSOR_API_KEY",
    "COPILOT_GITHUB_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
]

SCRIPT = """#!{python}
import json, os, subprocess, sys, time

def tree(root):
    found = {{}}
    for base, _, files in os.walk(root):
        for f in files:
            path = os.path.join(base, f)
            found[os.path.relpath(path, root)] = open(path).read()
    return found


here = os.path.dirname(os.path.abspath(__file__))
name = os.path.basename(__file__)
cfg = json.load(open(os.path.join(here, name + ".config.json")))
args = sys.argv[1:]
if args == ["--version"]:
    print(cfg["version"])
    sys.exit(0)
if cfg["status"] and args == cfg["status"]["args"]:
    sys.stdout.write(cfg["status"]["stdout"])
    sys.exit(cfg["status"]["exit"])

calls_path = os.path.join(here, name + ".calls.jsonl")
index = sum(1 for _ in open(calls_path)) if os.path.exists(calls_path) else 0
config_dir = os.environ.get("OPENCODE_CONFIG_DIR")
usage_path = args[args.index("--usage-output-file") + 1] if "--usage-output-file" in args else None
record = {{
    "argv": args,
    "cwd": os.getcwd(),
    "cwd_files": sorted(os.listdir(os.getcwd())),
    "cwd_tree": tree(os.getcwd()),
    "stdin": sys.stdin.read(),
    "env": {{k: os.environ.get(k) for k in cfg["env_keys"]}},
    "config_dir_files": sorted(os.listdir(config_dir)) if config_dir else None,
    "pid": os.getpid(),
    "usage_path": usage_path,
}}
reply = cfg["responses"][min(index, len(cfg["responses"]) - 1)]
if reply.get("spawn_child"):
    child = subprocess.Popen(["sleep", "60"])
    record["child_pid"] = child.pid
with open(calls_path, "a") as fh:
    fh.write(json.dumps(record) + "\\n")
if reply.get("usage_file") is not None and usage_path:
    with open(usage_path, "w") as fh:
        fh.write(reply["usage_file"])
if reply.get("sleep"):
    time.sleep(reply["sleep"])
sys.stdout.write(reply.get("stdout", ""))
sys.stderr.write(reply.get("stderr", ""))
sys.exit(reply.get("exit", 0))
"""


def install_experimental(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    version: str,
    responses: list[dict[str, Any]],
    status_args: list[str] | None = None,
    status_stdout: str = "",
    status_exit: int = 0,
) -> FakeCLI:
    """Write the fake ``name`` binary and put its directory first on PATH."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    path = bin_dir / name
    path.write_text(SCRIPT.format(python=sys.executable), "utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    status = (
        {"args": status_args, "stdout": status_stdout, "exit": status_exit} if status_args else None
    )
    config = {"version": version, "responses": responses, "status": status, "env_keys": ENV_KEYS}
    (bin_dir / f"{name}.config.json").write_text(json.dumps(config), "utf-8")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    cli_version.cache_clear()
    return FakeCLI(path, name)


def respond(fake: FakeCLI, *responses: dict[str, Any]) -> None:
    """Replace the canned replies of an installed fake and forget its recorded calls."""
    config_path = fake.path.parent / f"{fake.name}.config.json"
    config = json.loads(config_path.read_text("utf-8"))
    config["responses"] = list(responses)
    config_path.write_text(json.dumps(config), "utf-8")
    fake.calls_path.unlink(missing_ok=True)


def only_fakes_on_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Hide every real binary: ``PATH`` is the fakes' directory and nothing else."""
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))


# -- canned outputs ----------------------------------------------------------------------


def opencode_events(
    text: str = "OK",
    *,
    input_tokens: int = 478,
    output_tokens: int = 8,
    reasoning: int = 0,
    cache_read: int = 0,
    cache_write: int = 0,
    extra: list[dict[str, Any]] | None = None,
) -> str:
    sid = "ses_synthetic0001"
    lines: list[dict[str, Any]] = [
        {"type": "step_start", "sessionID": sid, "part": {"type": "step-start"}},
        {
            "type": "text",
            "sessionID": sid,
            "part": {"type": "text", "text": text, "time": {"start": 1, "end": 2}},
        },
        *(extra or []),
        {
            "type": "step_finish",
            "sessionID": sid,
            "part": {
                "type": "step-finish",
                "reason": "stop",
                "cost": 0,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "reasoning": reasoning,
                    "cache": {"read": cache_read, "write": cache_write},
                },
            },
        },
    ]
    return "".join(json.dumps(line) + "\n" for line in lines)


def opencode_ok(text: str = "OK", **kwargs: Any) -> dict[str, Any]:
    return reply(opencode_events(text, **kwargs))


def opencode_failure(
    kind: str, message: str, *, status: int | None = None, exit: int = 1
) -> dict[str, Any]:
    error: dict[str, Any] = {"type": kind, "message": message}
    if status is not None:
        error["status"] = status
    line = json.dumps({"type": "error", "sessionID": "ses_synthetic0001", "error": error})
    return reply(line + "\n", exit=exit)


def cursor_result(text: str = "OK", *, with_usage: bool = True, is_error: bool = False) -> str:
    result: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": is_error,
        "duration_ms": 1200,
        "duration_api_ms": 1100,
        "result": text,
        "session_id": "synthetic-session",
        "request_id": "synthetic-request",
    }
    if with_usage:
        result["usage"] = {
            "inputTokens": 120,
            "outputTokens": 7,
            "cacheReadTokens": 30,
            "cacheWriteTokens": 4,
        }
    return json.dumps(result) + "\n"


def cursor_stream(
    text: str = "OK",
    *,
    with_usage: bool = True,
    is_error: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
) -> str:
    """The ``stream-json`` events of one turn: init, the prompt, thinking, the reply, any tool
    calls (each a ``{kind: {...}}`` object), the result."""
    lines: list[dict[str, Any]] = [
        {
            "type": "system",
            "subtype": "init",
            "apiKeySource": "login",
            "cwd": "/synthetic/workspace",
            "model": "Synthetic Model",
            "permissionMode": "default",
        },
        {"type": "user", "message": {"role": "user", "content": [{"type": "text", "text": "q"}]}},
        {"type": "thinking", "subtype": "delta", "text": "hm"},
        {"type": "thinking", "subtype": "completed"},
    ]
    for call in tool_calls or []:
        lines.append({"type": "tool_call", "subtype": "started", "tool_call": call})
        lines.append({"type": "tool_call", "subtype": "completed", "tool_call": call})
    lines.append(
        {
            "type": "assistant",
            "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
        }
    )
    lines.append(json.loads(cursor_result(text, with_usage=with_usage, is_error=is_error)))
    return "".join(json.dumps(line) + "\n" for line in lines)


def cursor_ok(text: str = "OK", **kwargs: Any) -> dict[str, Any]:
    return reply(cursor_stream(text, **kwargs))


def cursor_about(email: str = "Not logged in") -> str:
    # The real command prints spinner escape codes first (seen on the test machine).
    return (
        "\x1b[2K\x1b[GLoading...\n\x1b[2K\x1b[1A\x1b[2K\x1b[GAbout Cursor CLI\n\n"
        "CLI Version         2026.02.27-e7d2ef6\nModel               Auto\n"
        f"OS                  darwin (arm64)\nUser Email          {email}\n"
    )


def copilot_usage(**by_model: tuple[int, int]) -> str:
    """The ``--usage-output-file`` JSON: ``model=(input, output)`` per model."""
    metrics = {
        model: {
            "requests": {"count": 1, "cost": 1},
            "usage": {
                "inputTokens": tokens_in,
                "outputTokens": tokens_out,
                "cacheReadTokens": 0,
                "cacheWriteTokens": 0,
                "reasoningTokens": 0,
            },
        }
        for model, (tokens_in, tokens_out) in by_model.items()
    }
    return json.dumps({"sessionId": "synthetic", "modelMetrics": metrics})


def copilot_events(
    text: str = "OK",
    *,
    model: str | None = "gpt-6-luna",
    requested: str = "auto",
    tool_requests: list[dict[str, Any]] | None = None,
    extra: list[dict[str, Any]] | None = None,
) -> str:
    """The ``--output-format json`` JSONL of one turn, as Copilot prints it."""
    message: dict[str, Any] = {"content": text, "toolRequests": tool_requests or []}
    if model is not None:
        message["model"] = model
    lines: list[dict[str, Any]] = [
        {"type": "session.start", "data": {"selectedModel": requested}},
        {"type": "session.info", "data": {"infoType": "tools", "message": "Disabled tools: bash"}},
        {
            "type": "session.info",
            "data": {
                "infoType": "warning",
                "message": 'Unknown tool name in the tool allowlist: "evalproof-no-tools"',
            },
        },
        {"type": "session.auto_mode_resolved", "data": {"chosenModel": model}},
        *(extra or []),
        {"type": "assistant.message", "data": message},
        {"type": "result", "exitCode": 0, "usage": {"premiumRequests": 1}},
    ]
    return "".join(json.dumps({**line, "id": f"e{i}"}) + "\n" for i, line in enumerate(lines))


def copilot_ok(
    text: str = "OK", *, model: str | None = "gpt-6-luna", tokens: tuple[int, int] | None = (13, 7)
) -> dict[str, Any]:
    """A reply with the events and, unless ``tokens`` is ``None``, the usage file."""
    out = reply(copilot_events(text, model=model))
    if tokens is not None:
        out["usage_file"] = copilot_usage(**{model or "synthetic": tokens})
    return out


def copilot_failure(message: str, *, exit: int = 1) -> dict[str, Any]:
    event = {"type": "session.error", "data": {"errorType": "query", "message": message}}
    return reply(json.dumps(event) + "\n", exit=exit)
