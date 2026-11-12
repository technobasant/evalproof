"""Experimental coding-agent CLI providers: ``opencode:``, ``cursor:`` and ``copilot:``.

Same idea and the same shared behaviour as ``claude-code`` and ``codex`` (see ``agent_cli``): your
own logged-in binary, run as a subprocess in a fresh empty directory, with API-key variables
removed from its environment, a bounded runtime and a fingerprint in the run artifact. They are
experimental because their isolation is less proven. What was and was not verified is in the
README ("Experimental providers"). Each prints one warning per process on first use.

Three things differ from the stable pair, and each is a consequence of what the CLI offers:

* ``cursor`` and ``copilot`` have no system-prompt option. The system text goes in front of the
  prompt instead, under a fixed heading, so it is an instruction in the user turn, not a
  replacement of the agent's own system prompt.
* The same two take the prompt as an argument, not on stdin: the Cursor docs only show an
  argument and nothing says it reads stdin, and ``copilot -p`` is documented as taking the
  prompt text. The prompt is therefore visible in ``ps`` while the call runs, and a prompt over
  ``ARGV_PROMPT_LIMIT`` bytes is refused instead of failing in ``execve``. stdin is closed.
* ``opencode`` has no system-prompt flag either, but its agent config has a ``prompt`` field, and
  the config can be given inline in an environment variable, so the system text travels there.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, ClassVar

from evalproof.config import GenerationParams
from evalproof.providers.agent_cli import (
    AUTH_TIMEOUT_S,
    NEUTRAL_SYSTEM,
    AgentCLIProvider,
    AuthStatus,
    _int,
    _quick,
    render_prompt,
    run_process,
)
from evalproof.providers.base import ChatResponse, Message, split_system

# A Linux process cannot take one argument or environment string longer than 128 KiB; stay well
# below it. The same bound caps the OpenCode config, which travels in one environment variable.
ARGV_PROMPT_LIMIT = 100_000

_warned: set[str] = set()
_warn_lock = threading.Lock()


def warn_experimental(name: str) -> None:
    """One line on stderr, once per process and provider, the first time it is used."""
    with _warn_lock:
        if name in _warned:
            return
        _warned.add(name)
    print(f"evalproof: {name} provider is experimental", file=sys.stderr, flush=True)


def frame_prompt(system: str, prompt: str) -> str:
    """For a CLI with no system-prompt option: the system text in front of the prompt. The heading
    also keeps the argument from starting with ``-``, which an argument parser would read as a
    flag."""
    return f"Instructions for your reply:\n{system}\n\nMessage to answer:\n{prompt}"


def _json_lines(stdout: str) -> list[dict[str, Any]]:
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


class ExperimentalCLIProvider(AgentCLIProvider):
    """The base class's calling convention with three extra switches: a per-call environment, the
    prompt in argv instead of stdin, and the system text folded into the prompt."""

    prompt_in_argv: ClassVar[bool] = False
    system_in_prompt: ClassVar[bool] = False
    extra_env: ClassVar[dict[str, str]] = {}

    def build_command(self, binary: str, workdir: str, params: GenerationParams) -> list[str]:
        raise NotImplementedError

    def with_prompt(self, command: list[str], prompt: str) -> list[str]:
        """Where the prompt goes in argv (only for ``prompt_in_argv`` providers)."""
        return [*command, prompt]

    def call_env(self, system: str, aux_dir: str) -> dict[str, str]:
        """Environment that differs per call. ``aux_dir`` is an empty directory that is removed
        with the call."""
        return {}

    def command(self, binary: str, system_file: str, params: GenerationParams) -> list[str]:
        # The base class and ``evalproof doctor`` ask for the command with a system file; none of
        # these CLIs reads one, so only the directory it would be in is used.
        return self.build_command(binary, os.path.dirname(system_file), params)

    def _once(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        warn_experimental(self.name)
        binary = self.binary_path()
        system, rest = split_system(messages)
        system = system or NEUTRAL_SYSTEM
        prompt = render_prompt(rest)
        if self.system_in_prompt:
            prompt = frame_prompt(system, prompt)
        if self.prompt_in_argv and len(prompt.encode("utf-8")) > ARGV_PROMPT_LIMIT:
            raise self.fail(
                f"the prompt is over {ARGV_PROMPT_LIMIT} bytes and {self.binary} takes it as an "
                "argument; shorten the system prompt or the input"
            )
        env = self.child_env()
        # One temp root, two empty directories: the working directory the CLI sees, and a spare
        # one for config. Removed afterwards, also when the call fails.
        root = tempfile.mkdtemp(prefix="evalproof-")
        try:
            cwd = Path(root) / "cwd"
            aux = Path(root) / "aux"
            cwd.mkdir()
            aux.mkdir()
            env.update(self.call_env(system, str(aux)))
            cmd = self.build_command(binary, str(cwd), params)
            if self.prompt_in_argv:
                cmd = self.with_prompt(cmd, prompt)
                stdin = b""  # EOF at once: never an open pipe
            else:
                stdin = prompt.encode("utf-8")
            start = time.perf_counter()
            try:
                code, out, err = run_process(
                    cmd, stdin=stdin, cwd=str(cwd), env=env, timeout_s=self.timeout_s
                )
            except subprocess.TimeoutExpired:
                raise self.fail(f"timed out after {self.timeout_s:g}s") from None
            except OSError as exc:
                raise self.fail(f"could not run {binary}: {exc.strerror or exc}") from exc
            latency_ms = (time.perf_counter() - start) * 1000
            return self.parse(code, out, err, latency_ms)
        finally:
            shutil.rmtree(root, ignore_errors=True)


class OpenCodeProvider(ExperimentalCLIProvider):
    """``opencode run`` with a custom agent that has every tool off and every permission denied,
    and with project config, global config dir and Claude-Code compatibility files switched off.

    Verified on local ollama models only: tools off (no tool event), a canary ``AGENTS.md`` and
    ``CLAUDE.md`` in the project not seen. ``opencode run`` is agentic and permissive by default
    (it ran a shell command unasked), so the agent config is not optional."""

    name = "opencode"
    binary = "opencode"
    install_hint = "npm install -g opencode-ai"
    isolation = "opencode/0"
    # OpenCode has no single login: the credential belongs to the model's provider (``anthropic/``,
    # ``openai/``, ``github-copilot/`` ...), and a user who names that provider in the model id
    # chose to use it, so its key variables are left alone (stripping ANTHROPIC_API_KEY would
    # break ``opencode:anthropic/...`` in CI, where the key is the only credential). The one
    # variable that is OpenCode's own is ``OPENCODE_API_KEY`` (its hosted Zen/Go service; the
    # name is in the binary, no docs page was found), so that one is stripped like the other
    # CLIs' keys and ``api_key_env`` maps to it.
    key_var = "OPENCODE_API_KEY"
    strip_vars = ("OPENCODE_API_KEY",)
    # The verified recipe: DISABLE_PROJECT_CONFIG alone stops the project's AGENTS.md and
    # CLAUDE.md (verified), the others close the remaining project and Claude-Code paths.
    extra_env: ClassVar[dict[str, str]] = {
        "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
        "OPENCODE_CONFIG_PROJECT_DISABLE": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE": "1",
        "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "1",
    }

    TRANSIENT_STATUS: ClassVar[frozenset[int]] = frozenset({429, 500, 502, 503, 529})
    # A usage limit with a retry-after of hours, not a blip.
    NOT_TRANSIENT_TYPES: ClassVar[frozenset[str]] = frozenset({"provider.free-limit"})

    def build_command(self, binary: str, workdir: str, params: GenerationParams) -> list[str]:
        # The prompt is on stdin (``opencode run`` with no message argument reads it; verified).
        return [
            binary,
            "run",
            "--standalone",  # a private server: the user's running OpenCode is not touched
            "--archived",  # keeps the session out of the user's session lists
            "--format",
            "json",
            "--agent",
            "judge",
            "-m",
            self.model,
        ]

    def call_env(self, system: str, aux_dir: str) -> dict[str, str]:
        config = {
            "agent": {
                "judge": {
                    "mode": "primary",
                    "prompt": system,
                    "tools": {"*": False},
                    "permission": {"*": "deny"},
                }
            }
        }
        content = json.dumps(config, ensure_ascii=False)
        if len(content.encode("utf-8")) > ARGV_PROMPT_LIMIT:
            raise self.fail(
                f"the system prompt is over {ARGV_PROMPT_LIMIT} bytes and opencode takes it in an "
                "environment variable; shorten it"
            )
        # OPENCODE_CONFIG_DIR is an empty directory, as in the verified recipe. A canary
        # AGENTS.md in a global config dir (XDG_CONFIG_HOME) was read by default and not read
        # with this recipe, on ollama/qwen3:8b.
        return {"OPENCODE_CONFIG_CONTENT": content, "OPENCODE_CONFIG_DIR": aux_dir}

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        events = _json_lines(stdout)
        for event in events:
            # With every tool off there are none (verified); one means the agent config did not
            # take, so the isolation is broken, as with codex's ``command_execution``.
            if event.get("type") == "tool_use":
                raise self.fail("opencode ran a tool; isolation broken")
        for event in events:
            if event.get("type") == "error":
                self._raise_error(event)
        if returncode != 0:
            raise self.fail(stderr.strip() or stdout.strip() or f"exited with status {returncode}")
        texts: list[str] = []
        tokens_in = tokens_out = 0
        for event in events:
            part = event.get("part")
            if not isinstance(part, dict):
                continue
            if event.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
            elif event.get("type") == "step_finish" and isinstance(part.get("tokens"), dict):
                tokens = part["tokens"]
                cache = tokens.get("cache")
                cache = cache if isinstance(cache, dict) else {}
                # Summed per field name, not checked against a provider's own accounting:
                # in = fresh + cache read + cache write; out = output + reasoning.
                tokens_in += _int(tokens.get("input")) + _int(cache.get("read"))
                tokens_in += _int(cache.get("write"))
                tokens_out += _int(tokens.get("output")) + _int(tokens.get("reasoning"))
        if not texts:
            raise self.fail(f"no text in the output: {stderr.strip() or stdout.strip()}")
        return ChatResponse(
            text="".join(texts),
            # The run output has no model id (only ``opencode session export`` does).
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
        )

    def _raise_error(self, event: dict[str, Any]) -> None:
        error = event.get("error")
        error = error if isinstance(error, dict) else {}
        message = str(error.get("message") or event.get("message") or "the request failed")
        status = error.get("status")
        if status:
            message += f" (HTTP {status})"
        transient = (
            status in self.TRANSIENT_STATUS and error.get("type") not in self.NOT_TRANSIENT_TYPES
        )
        raise self.fail(message, transient=transient)

    def auth_status(self) -> AuthStatus:
        # ``opencode auth list`` names the stored logins, but a local model (ollama) needs none and
        # the credential that matters is the one for the provider in the model id, so there is no
        # zero-cost command that says whether this target will work.
        return AuthStatus(True, "not checked (OpenCode logs in per model provider)")


_ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ABOUT_EMAIL = re.compile(r"^\s*User Email\s+(?P<value>.+?)\s*$", re.MULTILINE)


class CursorProvider(ExperimentalCLIProvider):
    """``cursor-agent -p`` in ask mode (read-only Q&A) in an empty workspace.

    Not verified: Cursor was not logged in on the test machine, so no call succeeded and the
    output parser follows the CLI's source, leniently. The empty workspace is the only isolation:
    there is no flag to ignore ``.cursor/rules``, ``AGENTS.md`` or ``CLAUDE.md``, and whether
    ``--mode ask`` really has no tools is not known."""

    name = "cursor"
    binary = "cursor-agent"
    install_hint = "curl https://cursor.com/install -fsS | bash"
    isolation = "cursor/0"
    key_var = "CURSOR_API_KEY"
    strip_vars = ("CURSOR_API_KEY",)
    prompt_in_argv = True
    system_in_prompt = True

    def build_command(self, binary: str, workdir: str, params: GenerationParams) -> list[str]:
        # The key travels in the environment, never as ``--api-key``: an argument is visible in
        # ``ps`` (a running Cursor worker was seen with its key on its command line).
        return [
            binary,
            "-p",
            "--output-format",
            "json",
            "--mode",
            "ask",
            "--trust",
            "--model",
            self.model,
            "--workspace",
            workdir,
        ]

    @staticmethod
    def _result_object(stdout: str) -> dict[str, Any] | None:
        """The result object: the whole output as JSON, else the last line that is. An array of
        events is accepted too (the ``stream-json`` shape)."""
        text = _ANSI.sub("", stdout).strip()
        candidates = [text, *reversed(text.splitlines())]
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate.startswith(("{", "[")):
                continue
            try:
                data = json.loads(candidate)
            except ValueError:
                continue
            for item in reversed(data) if isinstance(data, list) else [data]:
                if isinstance(item, dict) and (item.get("type") == "result" or "result" in item):
                    return item
        return None

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        result = self._result_object(stdout)
        if result is not None and result.get("is_error"):
            message = result.get("result") or result.get("error") or result.get("message")
            raise self.fail(str(message or stderr.strip() or "request failed"))
        if returncode != 0:
            # The one failure seen: exit 1, empty stdout, a plain-text line on stderr.
            raise self.fail(stderr.strip() or stdout.strip() or f"exited with status {returncode}")
        if result is None or not isinstance(result.get("result"), str):
            shown = (stdout.strip() or stderr.strip() or "no output")[:200]
            raise self.fail(
                "unexpected output from cursor-agent: expected one JSON object with a string "
                f'"result" (--output-format json); got: {shown}'
            )
        usage = result.get("usage")
        usage = usage if isinstance(usage, dict) else {}

        def count(*names: str) -> int:
            return sum(_int(usage.get(n)) for n in names)

        return ChatResponse(
            text=result["result"],
            model=self.model,  # the json result carries no model id
            tokens_in=count("inputTokens", "cacheReadTokens", "cacheWriteTokens"),
            tokens_out=count("outputTokens"),
            latency_ms=latency_ms,
        )

    def auth_status(self) -> AuthStatus:
        if self._api_key:
            return AuthStatus(
                True, f"not checked (key from api_key_env is passed as {self.key_var})"
            )
        # ``cursor-agent status`` is the obvious command but it hung for over two minutes on the
        # test machine; ``about`` returns at once and has a "User Email" line. The email itself is
        # never printed.
        code, out, _ = _quick([self.binary_path(), "about"], self.child_env(), AUTH_TIMEOUT_S)
        match = _ABOUT_EMAIL.search(_ANSI.sub("", out))
        if match is None:
            return AuthStatus(True, f"not checked (no account line in `about`, exit {code})")
        if match.group("value").lower().startswith("not logged in"):
            return AuthStatus(
                False, f"not logged in; run `{self.binary} login` or set {self.key_var}"
            )
        return AuthStatus(True, "logged in (account not shown)")


class CopilotProvider(ExperimentalCLIProvider):
    """``copilot -p`` in silent mode with an empty tool allow-list, in an empty directory.

    Not verified: Copilot CLI was not installed on the test machine, so every flag here comes from
    GitHub's documentation. The reply is plain text (the JSONL schema is undocumented), so token
    counts are 0."""

    name = "copilot"
    binary = "copilot"
    install_hint = "npm install -g @github/copilot"
    isolation = "copilot/0"
    # COPILOT_GITHUB_TOKEN wins over GH_TOKEN and GITHUB_TOKEN (documented precedence). Only it is
    # stripped: the other two are the general GitHub tokens that ``gh`` and CI set for other
    # reasons, and removing them would break unrelated tools' habits more than it protects.
    key_var = "COPILOT_GITHUB_TOKEN"
    strip_vars = ("COPILOT_GITHUB_TOKEN",)
    prompt_in_argv = True
    system_in_prompt = True

    def build_command(self, binary: str, workdir: str, params: GenerationParams) -> list[str]:
        return [
            binary,
            "-s",  # silent: the response only
            "--no-auto-update",  # the version in the fingerprint is the version that runs
            "--no-ask-user",  # never wait for an answer in a scripted run
            # The most restrictive documented tool option: an allow-list ("the model can use only
            # the listed tools") with nothing in it. Whether the CLI takes an empty list was not
            # tried. No ``--allow-tool`` or ``--allow-all`` flag is ever passed, so a tool that
            # needs approval cannot get it in a non-interactive run.
            "--available-tools=",
            "--model",
            self.model,
        ]

    def with_prompt(self, command: list[str], prompt: str) -> list[str]:
        return [*command, "-p", prompt]

    def parse(self, returncode: int, stdout: str, stderr: str, latency_ms: float) -> ChatResponse:
        if returncode != 0:
            raise self.fail(stderr.strip() or stdout.strip() or f"exited with status {returncode}")
        text = stdout.strip()
        if not text:
            raise self.fail(f"empty reply: {stderr.strip() or 'no output'}")
        return ChatResponse(
            text=text,
            model=self.model,
            tokens_in=0,  # text output carries no usage
            tokens_out=0,
            latency_ms=latency_ms,
        )

    def auth_status(self) -> AuthStatus:
        # No documented command reports the login without starting the interactive UI.
        return AuthStatus(True, "not checked")


EXPERIMENTAL_PROVIDERS: dict[str, type[AgentCLIProvider]] = {
    OpenCodeProvider.name: OpenCodeProvider,
    CursorProvider.name: CursorProvider,
    CopilotProvider.name: CopilotProvider,
}
