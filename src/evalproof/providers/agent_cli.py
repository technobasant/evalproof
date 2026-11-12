"""Coding-agent CLIs as providers: ``claude-code:<model>`` and ``codex:<model>``.

A team that already has Claude Code or Codex logged in can run the judge, or the system under
test, on that login with no API key. evalproof runs your own ``claude`` or ``codex`` binary as
a subprocess and reads its JSON output. It never reads a token file or a keychain entry.

A coding agent is a different thing from a model endpoint: left alone it loads the project's
``CLAUDE.md`` or ``AGENTS.md``, your personal memory, plugins, MCP servers and skills into the
prompt, and it can run commands. A judge that read your project instructions is not a judge,
so isolation is a requirement here, not an option: every call runs in a fresh empty
directory, with the agent's own system prompt replaced and its tools, settings, MCP servers
and skills switched off. No permission-bypass flag is ever passed; tools are removed instead
of being allowed.

The CLIs offer no temperature, seed or token-limit control, so those parameters are ignored
and the run records a fingerprint (CLI, version, isolation scheme) that ``compare`` and the
gate check.
"""

from __future__ import annotations

import contextlib
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from evalproof.config import GenerationParams, TargetConfig
from evalproof.providers.base import (
    ChatResponse,
    Message,
    ProviderError,
    TransientError,
    read_key,
    split_system,
    with_retries,
)

NEUTRAL_SYSTEM = "You are a careful assistant. Answer the user's message."
MAX_ERROR_CHARS = 500
REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
VERSION_TIMEOUT_S = 30.0
AUTH_TIMEOUT_S = 30.0

# How the CLI authenticates. ``login`` (the default) uses the subscription login and removes the
# CLI's API-key variables from the child's environment; ``api-key`` leaves them, so a CI job
# can hand the CLI a key from a secret. Either way it is the same model, so the mode is not part
# of the fingerprint or the cache key.
AUTH_ENV = "EVALPROOF_CLI_AUTH"
AUTH_MODES = ("login", "api-key")

# The codex release the isolation flags were verified against.
CODEX_TESTED_VERSION = "codex-cli 0.167.2"
# The only item types a codex turn may contain when it used no tool.
CODEX_ALLOWED_ITEMS = frozenset({"agent_message", "reasoning", "error", "todo_list"})


def cli_auth_mode() -> str:
    """The auth mode from ``EVALPROOF_CLI_AUTH``: ``login`` when it is unset or empty."""
    value = os.environ.get(AUTH_ENV, "").strip()
    if not value:
        return "login"
    if value not in AUTH_MODES:
        raise ProviderError(f"{AUTH_ENV} must be one of {', '.join(AUTH_MODES)}, got {value!r}")
    return value


def clip(text: str, limit: int = MAX_ERROR_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render_prompt(messages: list[Message]) -> str:
    """The non-system messages as the one string a CLI takes on stdin.

    A single user message goes as it is. Several turns become a transcript of ``User:`` and
    ``Assistant:`` blocks separated by blank lines, ending with the last turn."""
    if not messages:
        raise ProviderError("no user message to send")
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    labels = {"user": "User", "assistant": "Assistant"}
    return "\n\n".join(f"{labels.get(m.role, m.role.title())}: {m.content}" for m in messages)


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """Kill the child and everything it started (it runs in its own session)."""
    try:
        if sys.platform == "win32":  # pragma: no cover - POSIX only in CI
            proc.kill()
        else:
            os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_process(
    cmd: list[str],
    *,
    stdin: bytes,
    cwd: str,
    env: dict[str, str],
    timeout_s: float,
) -> tuple[int, str, str]:
    """Run ``cmd`` with ``stdin`` as its whole input (then EOF) and return the exit code, stdout
    and stderr. On a timeout, or any other interruption, the whole process group is killed and
    the exception is re-raised, so nothing outlives the call."""
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        env=env,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(stdin, timeout=timeout_s)
    except BaseException:
        _kill_group(proc)
        with contextlib.suppress(subprocess.TimeoutExpired, OSError, ValueError):
            proc.communicate(timeout=5)
        raise
    return (
        proc.returncode,
        out.decode("utf-8", errors="replace"),
        err.decode("utf-8", errors="replace"),
    )


@functools.cache
def cli_version(binary: str) -> str:
    """The first line of ``<binary> --version``, once per binary path per process.

    It is a local command, never a model call."""
    try:
        proc = subprocess.run(
            [binary, "--version"],
            capture_output=True,
            stdin=subprocess.DEVNULL,
            timeout=VERSION_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ProviderError(f"`{binary} --version` failed: {exc}") from exc
    lines = proc.stdout.decode("utf-8", errors="replace").strip().splitlines()
    if proc.returncode != 0 or not lines:
        raise ProviderError(f"`{binary} --version` exited {proc.returncode} with no version")
    return lines[0].strip()


def _int(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


@dataclass
class AuthStatus:
    ok: bool
    detail: str


class AgentCLIProvider:
    """What every CLI provider shares: the binary, an isolated working directory, the prompt on
    stdin, a clean environment, a bounded subprocess and a fingerprint. A subclass supplies the
    command line, the output parser and the auth check."""

    name: str
    binary: str
    install_hint: str
    isolation: str  # bump when the flag set changes: it is part of the cache key and the gate
    key_var: str  # the variable the CLI reads its API key from
    strip_vars: tuple[str, ...]  # key variables removed in login mode (the subscription login)
    key_alias: ClassVar[str | None] = None  # in api-key mode, copied into key_var when unset
    extra_env: ClassVar[dict[str, str]]

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        timeout_s: float = 120.0,
        api_key: str | None = None,
        api_key_env: str | None = None,
        headers: dict[str, str] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if base_url or headers:
            raise ProviderError(
                f"{self.name}: base_url and headers do not apply to a CLI provider; remove them"
            )
        self.model = model
        self.timeout_s = timeout_s
        self._sleep = sleep
        self.auth_mode = cli_auth_mode()
        self._api_key = read_key(
            self.name, api_key_env, required=api_key_env is not None, api_key=api_key
        )

    # -- what a subclass supplies ------------------------------------------------------

    def command(self, binary: str, system_file: str, params: GenerationParams) -> list[str]:
        raise NotImplementedError

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        raise NotImplementedError

    def auth_status(self) -> AuthStatus:
        raise NotImplementedError

    def salt_extra(self, params: GenerationParams) -> dict[str, str]:
        """Request settings that change the reply and so belong in the cache key."""
        return {}

    # -- shared behaviour ----------------------------------------------------------------

    def binary_path(self) -> str:
        path = shutil.which(self.binary)
        if path is None:
            raise ProviderError(
                f"{self.name}: `{self.binary}` was not found on PATH; "
                f"install it with `{self.install_hint}`"
            )
        return path

    def child_env(self) -> dict[str, str]:
        """The caller's environment. In ``login`` mode the CLI's API-key variables are removed,
        so the CLI uses the subscription login; in ``api-key`` mode they stay. A key named by
        ``api_key_env`` goes in under the CLI's own name in either mode and wins."""
        if self.auth_mode == "login":
            env = {k: v for k, v in os.environ.items() if k not in self.strip_vars}
        else:
            env = dict(os.environ)
            # A key set under the vendor's usual name also reaches a CLI that reads another one.
            alias = self.key_alias
            if alias and not env.get(self.key_var) and env.get(alias):
                env[self.key_var] = env[alias]
        env.update(self.extra_env)
        if self._api_key:
            env[self.key_var] = self._api_key
        return env

    def fingerprint(self) -> dict[str, str]:
        return {
            "cli": self.name,
            "cli_version": cli_version(self.binary_path()),
            "isolation": self.isolation,
            "sampling": "cli-default",
        }

    def cache_salt(self, params: GenerationParams) -> dict[str, object]:
        """A CLI upgrade or a change to the flag set must not be answered from the cache."""
        salt: dict[str, object] = dict(self.fingerprint())
        salt.update(self.salt_extra(params))
        return salt

    def fail(self, message: str, *, transient: bool = False) -> ProviderError:
        if self._api_key:
            message = message.replace(self._api_key, "***")
        cls = TransientError if transient else ProviderError
        return cls(f"{self.name}: {clip(message)}")

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        return with_retries(lambda: self._once(messages, params), sleep=self._sleep)

    def _once(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        binary = self.binary_path()
        system, rest = split_system(messages)
        prompt = render_prompt(rest)
        env = self.child_env()
        # A fresh empty directory: the project's CLAUDE.md / AGENTS.md and rules must not reach
        # the prompt. Removed afterwards, also when the call fails.
        workdir = tempfile.mkdtemp(prefix="evalproof-")
        try:
            system_file = Path(workdir) / "system.txt"
            system_file.write_text(system or NEUTRAL_SYSTEM, encoding="utf-8")
            cmd = self.command(binary, str(system_file), params)
            start = time.perf_counter()
            try:
                code, out, err = run_process(
                    cmd,
                    stdin=prompt.encode("utf-8"),
                    cwd=workdir,
                    env=env,
                    timeout_s=self.timeout_s,
                )
            except subprocess.TimeoutExpired:
                raise self.fail(f"timed out after {self.timeout_s:g}s") from None
            except OSError as exc:
                raise self.fail(f"could not run {binary}: {exc.strerror or exc}") from exc
            latency_ms = (time.perf_counter() - start) * 1000
            return self.parse(code, out, err, latency_ms)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ClaudeCodeProvider(AgentCLIProvider):
    """``claude -p`` with its tools, settings, MCP servers and skills switched off (verified:
    0 tools, 0 MCP servers, 0 skills, a CLAUDE.md canary in the project not seen). ``--bare``
    would skip even more but needs an API key, which defeats the point."""

    name = "claude-code"
    binary = "claude"
    install_hint = "npm install -g @anthropic-ai/claude-code"
    isolation = "claude-code/1"
    key_var = "ANTHROPIC_API_KEY"
    strip_vars = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
    extra_env: ClassVar[dict[str, str]] = {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"}

    TRANSIENT_STATUS: ClassVar[frozenset[int]] = frozenset({429, 500, 502, 503, 529})

    def command(self, binary: str, system_file: str, params: GenerationParams) -> list[str]:
        return [
            binary,
            "-p",
            "--output-format",
            "json",
            "--no-session-persistence",
            "--model",
            self.model,
            "--setting-sources",
            "",
            "--tools",
            "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--system-prompt-file",
            system_file,
        ]

    @staticmethod
    def _result_event(stdout: str) -> dict[str, Any] | None:
        """The ``result`` event: the output is an array of events (older versions printed a
        single object)."""
        try:
            data = json.loads(stdout)
        except ValueError:
            return None
        events = data if isinstance(data, list) else [data]
        for event in reversed(events):
            if isinstance(event, dict) and event.get("type") == "result":
                return event
        return None

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        result = self._result_event(stdout)
        # ``subtype`` stays "success" on errors, so only ``is_error`` says whether it failed.
        if result is not None and result.get("is_error"):
            status = result.get("api_error_status")
            message = str(result.get("result") or "").strip() or stderr.strip() or "request failed"
            if status:
                message += f" (HTTP {status})"
            raise self.fail(message, transient=status in self.TRANSIENT_STATUS)
        if returncode != 0:
            raise self.fail(stderr.strip() or stdout.strip() or f"exited with status {returncode}")
        if result is None:
            raise self.fail(f"no result event in the output: {stderr.strip() or stdout.strip()}")
        text = result.get("result")
        if not isinstance(text, str):
            raise self.fail("the result event has no text")
        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}
        tokens_in = sum(
            _int(usage.get(k))
            for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
        )
        return ChatResponse(
            text=text,
            model=self._resolved_model(result),
            tokens_in=tokens_in,
            tokens_out=_int(usage.get("output_tokens")),
            latency_ms=latency_ms,
        )

    def _resolved_model(self, result: dict[str, Any]) -> str:
        """The model id the CLI actually used: the ``modelUsage`` key with the most output."""
        by_model = result.get("modelUsage")
        if not isinstance(by_model, dict) or not by_model:
            return self.model

        def produced(name: str) -> int:
            entry = by_model[name]
            return _int(entry.get("outputTokens")) if isinstance(entry, dict) else 0

        return str(max(by_model, key=produced))

    def auth_status(self) -> AuthStatus:
        code, out, _ = _quick(
            [self.binary_path(), "auth", "status"], self.child_env(), AUTH_TIMEOUT_S
        )
        try:
            data = json.loads(out)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            return AuthStatus(False, f"`claude auth status` gave no JSON (exit {code})")
        # Only these three fields: the full status also holds an email and an organisation.
        shown = {k: data[k] for k in ("authMethod", "apiProvider", "subscriptionType") if k in data}
        detail = " ".join(f"{k}={v}" for k, v in shown.items()) or "logged in"
        if code != 0 or data.get("loggedIn") is False:
            return AuthStatus(False, f"not logged in; run `claude auth login` ({detail})")
        return AuthStatus(True, detail)


class CodexProvider(AgentCLIProvider):
    """``codex exec`` read-only, with the shell and other tools disabled, project docs, user
    config and skills ignored and the agent's base instructions replaced (verified: canary not
    seen, no command execution).

    Codex still offers a few tools that no flag removes in codex-cli 0.167.2: a JavaScript
    isolate (``exec``: no file system, no network), sub-agents that get the same restricted
    tools, and a question to the user that exec mode cannot answer. So the guard is an
    allowlist: any item in the output that is not a message, reasoning, an error or a plan means
    a tool was used, the call fails and the reply is discarded."""

    name = "codex"
    binary = "codex"
    install_hint = "npm install -g @openai/codex"
    isolation = "codex/1"
    key_var = "CODEX_API_KEY"
    strip_vars = ("OPENAI_API_KEY", "CODEX_API_KEY")
    # `codex exec` reads CODEX_API_KEY; CI secrets are usually named OPENAI_API_KEY.
    key_alias = "OPENAI_API_KEY"
    extra_env: ClassVar[dict[str, str]] = {}

    def reasoning_effort(self, params: GenerationParams) -> str | None:
        """``params.extra["reasoning_effort"]``, else ``think`` (false: low, true: medium).
        ``think: null`` leaves the flag out and the model's default applies."""
        chosen = params.extra.get("reasoning_effort")
        if chosen is not None:
            if chosen not in REASONING_EFFORTS:
                raise ProviderError(
                    f"codex: reasoning_effort must be one of {', '.join(REASONING_EFFORTS)}, "
                    f"got {chosen!r}"
                )
            return str(chosen)
        if params.think is None:
            return None
        return "medium" if params.think else "low"

    def salt_extra(self, params: GenerationParams) -> dict[str, str]:
        return {"reasoning_effort": self.reasoning_effort(params) or "default"}

    def command(self, binary: str, system_file: str, params: GenerationParams) -> list[str]:
        effort = self.reasoning_effort(params)
        return [
            binary,
            "exec",
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--ignore-user-config",
            "--ignore-rules",
            "-m",
            self.model,
            *(["-c", f"model_reasoning_effort={effort}"] if effort else []),
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            f"model_instructions_file={system_file}",
            "-c",
            "skills.include_instructions=false",
            "-c",
            "skills.bundled.enabled=false",
            "--disable",
            "shell_tool",
            "--disable",
            "plugins",
            "--disable",
            "apps",
            "--disable",
            "hooks",
            "--disable",
            "multi_agent",
            "--disable",
            "sleep_tool",
            "--disable",
            "view_image",
            "-",
        ]

    @staticmethod
    def _events(stdout: str) -> list[dict[str, Any]]:
        """JSONL; lines that are not JSON objects are skipped."""
        events: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict):
                events.append(event)
        return events

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        if "Unknown feature flag" in stderr:
            found = re.search(r"Unknown feature flag:\s*(\S+)", stderr)
            name = f" ({found.group(1)})" if found else ""
            raise self.fail(
                f"this codex version lacks a feature evalproof disables{name}; evalproof was "
                f"tested with {CODEX_TESTED_VERSION}, so upgrade codex ({self.install_hint})"
            )
        events = self._events(stdout)
        for event in events:
            item = event.get("item")
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if isinstance(kind, str) and kind not in CODEX_ALLOWED_ITEMS:
                # The reply text is not read: a turn that used a tool is not isolated.
                raise self.fail(f"codex used a tool ({kind}); isolation broken")
        failure = self._failure(events)
        if failure is not None:
            lowered = failure.lower()
            raise self.fail(failure, transient="rate limit" in lowered or "overloaded" in lowered)
        if returncode != 0:
            raise self.fail(stderr.strip() or stdout.strip() or f"exited with status {returncode}")
        text: str | None = None
        usage: dict[str, Any] = {}
        for event in events:
            item = event.get("item")
            if (
                event.get("type") == "item.completed"
                and isinstance(item, dict)
                and item.get("type") == "agent_message"
                and isinstance(item.get("text"), str)
            ):
                text = item["text"]
            elif event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                usage = event["usage"]
        if text is None:
            raise self.fail(f"no agent message in the output: {stderr.strip() or stdout.strip()}")
        return ChatResponse(
            text=text,
            model=self.model,  # codex does not report the model it used
            tokens_in=_int(usage.get("input_tokens")),
            tokens_out=_int(usage.get("output_tokens")),
            latency_ms=latency_ms,
        )

    @staticmethod
    def _failure(events: list[dict[str, Any]]) -> str | None:
        """The message of a ``turn.failed`` or ``error`` event; ``turn.failed`` wins."""
        message: str | None = None
        for event in events:
            kind = event.get("type")
            if kind not in ("turn.failed", "error"):
                continue
            error = event.get("error")
            found = (error.get("message") if isinstance(error, dict) else None) or event.get(
                "message"
            )
            text = str(found) if found else "the turn failed"
            if kind == "turn.failed" or message is None:
                message = text
        return message

    def auth_status(self) -> AuthStatus:
        code, out, err = _quick(
            [self.binary_path(), "login", "status"], self.child_env(), AUTH_TIMEOUT_S
        )
        lines = [ln.strip() for ln in (out + "\n" + err).splitlines() if ln.strip()]
        line = clip(lines[0], 200) if lines else f"no output (exit {code})"
        return AuthStatus(code == 0, line)


def _quick(cmd: list[str], env: dict[str, str], timeout_s: float) -> tuple[int, str, str]:
    """A short local command (never a model call), with stdin closed."""
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 1, "", f"{type(exc).__name__}: {exc}"
    return (
        proc.returncode,
        proc.stdout.decode("utf-8", errors="replace"),
        proc.stderr.decode("utf-8", errors="replace"),
    )


CLI_PROVIDERS: dict[str, type[AgentCLIProvider]] = {
    ClaudeCodeProvider.name: ClaudeCodeProvider,
    CodexProvider.name: CodexProvider,
}


def cli_facts(cfg: TargetConfig) -> dict[str, str]:
    """The fingerprint of a CLI target for the run artifact, or ``{}`` for any other target.

    Best effort and free: it asks the binary for its version, never for a model reply."""
    cls = CLI_PROVIDERS.get(cfg.kind)
    if cls is None:
        return {}
    try:
        return cls(cfg.target.split(":", 1)[1], timeout_s=cfg.timeout_s).fingerprint()
    except ProviderError:
        return {}
