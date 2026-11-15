"""Systems under test: an LLM prompt, a Python function, a program, or an HTTP endpoint.

A system failure is recorded on the case, not raised: one timeout should not throw
away an hour of judging. Errored cases score as failures and are listed in reports.

A python, http or command system may be an agent. It then returns, besides its ``output``, a
``trace`` of what it did (see ``evalproof.models.Trace``) and optionally its token counts and cost;
the runner fills any token or cost figure it leaves out from the trace's usage (``_finish``).
"""

from __future__ import annotations

import contextlib
import copy
import importlib
import inspect
import json
import os
import shutil
import string
import subprocess
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from evalproof._process import kill_tree, popen_group_kwargs
from evalproof.cache import CachedProvider, ResponseCache, trial_scope
from evalproof.config import (
    ConfigError,
    GenerationParams,
    Pricing,
    TargetConfig,
    command_argv,
    trial_params,
)
from evalproof.models import Case, SystemOutput, Trace
from evalproof.providers import (
    CLI_PROVIDERS,
    CacheMiss,
    Message,
    Provider,
    ProviderError,
    make_provider,
    response_cost,
    with_retries,
)
from evalproof.providers.base import close_resource


@dataclass(frozen=True)
class RunContext:
    """What a python system may ask for beside the case input: declare a parameter named
    ``ctx`` and the system is called as ``func(input, ctx=ctx)``.

    ``turns`` are the user turns after ``input`` (turn 0); ``state`` is the case's initial session
    state; ``variant`` and ``vars`` are the suite variant being run and its variables; ``trial`` is
    the repeat index, 0 to ``trials - 1`` (always 0 when the suite runs each case once). Each
    call gets its own copies of the mutable parts.
    """

    case_id: str
    input: str | dict[str, Any]
    turns: list[str | dict[str, Any]] = field(default_factory=list)
    state: dict[str, Any] = field(default_factory=dict)
    variant: str | None = None
    vars: dict[str, Any] = field(default_factory=dict)
    trial: int = 0


def _context(
    case: Case, variant: str | None, variant_vars: dict[str, Any], trial: int = 0
) -> RunContext:
    return RunContext(
        case_id=case.id,
        input=case.input,
        turns=copy.deepcopy(case.turns),
        state=copy.deepcopy(case.state),
        variant=variant,
        vars=copy.deepcopy(variant_vars),
        trial=trial,
    )


class System(Protocol):
    name: str

    def run(self, case: Case, *, trial: int = 0) -> SystemOutput: ...


def render(template: str, variables: dict[str, Any]) -> str:
    """``str.format`` with a clear error naming the missing field."""
    missing = [
        field
        for _, field, _, _ in string.Formatter().parse(template)
        if field and field.split(".")[0].split("[")[0] not in variables
    ]
    if missing:
        raise ConfigError(f"prompt template references unknown fields: {sorted(set(missing))}")
    return template.format_map(variables)


class LLMSystem:
    """A model prompt. A case with ``turns`` is a conversation: turn 0 is the case input, each
    later turn follows the model's previous reply, and the output is the last reply."""

    def __init__(
        self, cfg: TargetConfig, provider: Provider, *, vars: dict[str, Any] | None = None
    ) -> None:
        self.cfg = cfg
        self.provider = provider
        self.name = cfg.label or cfg.target
        self.vars = dict(vars or {})  # the variant's, rendered into the prompt template

    def close(self) -> None:
        close_resource(self.provider)

    def _user(self, case: Case, **turn: Any) -> Message:
        variables = {**self.vars, **case.render_vars(**turn)}
        return Message(role="user", content=render(self.cfg.prompt_template, variables))

    def messages(self, case: Case) -> list[Message]:
        msgs: list[Message] = []
        if self.cfg.system_prompt:
            msgs.append(Message(role="system", content=self.cfg.system_prompt))
        msgs.append(self._user(case))
        return msgs

    def run(self, case: Case, *, trial: int = 0) -> SystemOutput:
        """Trial ``t`` of a case: the same prompt, with the seed shifted by ``t`` (when the suite
        sets one) and the call keyed as that trial's in the response cache."""
        params = trial_params(self.cfg.params, trial)
        with trial_scope(trial):
            if case.turns:
                return self._converse(case, params)
            return self._single(case, params)

    def _single(self, case: Case, params: GenerationParams) -> SystemOutput:
        try:
            resp = self.provider.chat(self.messages(case), params)
        except CacheMiss:
            raise  # a cache-only run stops; it is not a failure of this case
        except ProviderError as exc:
            return SystemOutput(output="", error=str(exc))
        return SystemOutput(
            output=resp.text,
            latency_ms=resp.latency_ms,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            cost_usd=response_cost(resp, self.cfg.pricing),
            cached=resp.cached,
        )

    def _converse(self, case: Case, params: GenerationParams) -> SystemOutput:
        """Send the turns in order, each after the model's previous reply, through the same
        provider (and cache) as a single call. The system prompt goes once, at the start."""
        msgs = self.messages(case)
        replies: list[str] = []
        tokens_in = tokens_out = 0
        cost = latency = 0.0
        cached = True
        for turn in [None, *case.turns]:
            if turn is not None:
                msgs = [
                    *msgs,
                    Message(role="assistant", content=replies[-1]),
                    self._user(case, turn=turn),
                ]
            try:
                resp = self.provider.chat(msgs, params)
            except CacheMiss:
                raise
            except ProviderError as exc:
                # What the earlier turns cost is still spent, so budgets see it.
                return SystemOutput(
                    output="",
                    error=str(exc),
                    latency_ms=latency,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost_usd=cost,
                    trace=Trace(responses=replies, rounds=len(replies)),
                )
            replies.append(resp.text)
            tokens_in += resp.tokens_in
            tokens_out += resp.tokens_out
            cost += response_cost(resp, self.cfg.pricing)
            latency += resp.latency_ms
            cached = cached and resp.cached
        return SystemOutput(
            output=replies[-1],
            latency_ms=latency,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost,
            cached=cached,
            trace=Trace(responses=replies, rounds=len(replies)),
        )


@dataclass
class _Result:
    """What a python, http or command system returned, before token and cost figures are
    settled."""

    output: str
    retrieved: list[str] = field(default_factory=list)
    trace: Trace | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    cost_usd: float | None = None


def _number(result: dict[str, Any], key: str, kind: type[int] | type[float]) -> Any:
    value = result.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"'{key}' must be a number, got {type(value).__name__}")
    return kind(value)


def _coerce(result: Any) -> _Result:
    """A system returns a string, or a dict with ``output`` and optionally ``retrieved``,
    ``trace`` (a dict or a ``Trace``), ``tokens_in``, ``tokens_out`` and ``cost_usd``."""
    if isinstance(result, str):
        return _Result(result)
    if isinstance(result, dict) and "output" in result:
        trace = result.get("trace")
        if trace is not None and not isinstance(trace, Trace):
            if not isinstance(trace, dict):
                raise TypeError("'trace' must be a dict or a Trace")
            try:
                trace = Trace.model_validate(trace)
            except ValidationError as exc:
                first = exc.errors(include_url=False)[0]
                where = ".".join(str(p) for p in first["loc"])
                raise ValueError(f"invalid 'trace': {where}: {first['msg']}") from exc
        return _Result(
            output=str(result["output"]),
            retrieved=[str(x) for x in result.get("retrieved") or []],
            trace=trace,
            tokens_in=_number(result, "tokens_in", int),
            tokens_out=_number(result, "tokens_out", int),
            cost_usd=_number(result, "cost_usd", float),
        )
    raise TypeError("a system must return a string or a dict with an 'output' key")


def _finish(result: _Result, latency_ms: float, pricing: Pricing) -> SystemOutput:
    """Settle the token and cost figures a system left out, from its trace's usage.

    A figure the system set is kept. Otherwise tokens are the sums over ``trace.usage``; the cost
    is the sum of the usage costs that are set, or, when none is, the target's ``pricing`` applied
    to the tokens (zero without a price). So budgets see an agent's spend."""
    usage = result.trace.usage if result.trace else []
    tokens_in = (
        result.tokens_in if result.tokens_in is not None else sum(u.tokens_in for u in usage)
    )
    tokens_out = (
        result.tokens_out if result.tokens_out is not None else sum(u.tokens_out for u in usage)
    )
    if result.cost_usd is not None:
        cost = result.cost_usd
    else:
        costs = [u.cost_usd for u in usage if u.cost_usd is not None]
        cost = sum(costs) if costs else pricing.cost(tokens_in, tokens_out)
    return SystemOutput(
        output=result.output,
        retrieved=result.retrieved,
        latency_ms=latency_ms,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost,
        trace=result.trace,
    )


def _accepts_ctx(func: Callable[..., Any]) -> bool:
    """True when ``func`` has a parameter named ``ctx`` that can be passed by keyword."""
    try:
        param = inspect.signature(func).parameters.get("ctx")
    except (TypeError, ValueError):  # a builtin or extension without a signature
        return False
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class PythonSystem:
    """``python:package.module:function`` — the function receives the case input, and a
    ``RunContext`` as ``ctx=`` when it has a parameter of that name.

    ``search_path`` (the suite's directory) is put on ``sys.path`` so a suite can ship
    its system next to it and still run from any working directory. ``pricing`` prices the
    tokens of a result that reports no cost; ``variant`` and ``vars`` reach the function through
    its context.
    """

    def __init__(
        self,
        spec: str,
        search_path: str | None = None,
        *,
        pricing: Pricing | None = None,
        variant: str | None = None,
        vars: dict[str, Any] | None = None,
    ) -> None:
        self.name = spec
        self.pricing = pricing or Pricing()
        self.variant = variant
        self.vars = dict(vars or {})
        if search_path and search_path not in sys.path:
            sys.path.insert(0, search_path)
        try:
            _, module_name, func_name = spec.split(":", 2)
        except ValueError as exc:
            raise ConfigError(
                f"python target must be 'python:<module>:<function>', got {spec!r}"
            ) from exc
        module = importlib.import_module(module_name)
        func = getattr(module, func_name, None)
        if not callable(func):
            raise ConfigError(f"{module_name}.{func_name} is not callable")
        self.func: Callable[..., Any] = func
        self._wants_ctx = _accepts_ctx(func)

    def run(self, case: Case, *, trial: int = 0) -> SystemOutput:
        start = time.perf_counter()
        try:
            if self._wants_ctx:
                result = self.func(case.input, ctx=_context(case, self.variant, self.vars, trial))
            else:
                result = self.func(case.input)
            coerced = _coerce(result)
        except Exception as exc:
            return SystemOutput(output="", error=f"{type(exc).__name__}: {exc}")
        return _finish(coerced, (time.perf_counter() - start) * 1000, self.pricing)


def request_body(
    case: Case,
    *,
    variant: str | None,
    vars: dict[str, Any],
    trials: int,
    trial: int,
) -> dict[str, Any]:
    """What an http or command system is sent for one run: ``id`` and ``input``, and when the case
    or variant has them ``turns``, ``state``, ``variant`` and ``vars``; ``trial`` when each case
    runs more than once."""
    body: dict[str, Any] = {"id": case.id, "input": case.input}
    for key, value in (
        ("turns", case.turns),
        ("state", case.state),
        ("variant", variant),
        ("vars", vars),
    ):
        if value:
            body[key] = value
    if trials > 1:
        body["trial"] = trial
    return body


class HttpSystem:
    """POSTs ``{"id", "input"}`` and, when the case or variant has them, ``turns``, ``state``,
    ``variant`` and ``vars``; with more than one trial per case, also ``trial``. Expects JSON
    ``{"output", "retrieved"?, "trace"?, "tokens_in"?, "tokens_out"?, "cost_usd"?}`` or plain
    text."""

    def __init__(
        self,
        cfg: TargetConfig,
        client: httpx.Client | None = None,
        *,
        variant: str | None = None,
        vars: dict[str, Any] | None = None,
        trials: int = 1,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.label or cfg.target
        self.variant = variant
        self.vars = dict(vars or {})
        self.trials = trials  # the body carries ``trial`` only when a case runs more than once
        self._owns_client = client is None
        self._client = httpx.Client(timeout=cfg.timeout_s) if client is None else client

    def close(self) -> None:
        """Close the HTTP client this created; a client passed in stays its owner's to close."""
        if self._owns_client:
            self._client.close()

    def _body(self, case: Case, trial: int = 0) -> dict[str, Any]:
        return request_body(
            case, variant=self.variant, vars=self.vars, trials=self.trials, trial=trial
        )

    def run(self, case: Case, *, trial: int = 0) -> SystemOutput:
        body = self._body(case, trial)

        def call() -> httpx.Response:
            resp = self._client.post(self.cfg.target, json=body)
            resp.raise_for_status()
            return resp

        start = time.perf_counter()
        try:
            resp = with_retries(call)
            ctype = resp.headers.get("content-type", "")
            coerced = _coerce(resp.json() if "json" in ctype else resp.text)
        except (ProviderError, TypeError, ValueError) as exc:
            return SystemOutput(output="", error=str(exc))
        return _finish(coerced, (time.perf_counter() - start) * 1000, self.cfg.pricing)


# -- the command target ----------------------------------------------------------------------------

STDERR_TAIL_CHARS = 300  # how much of a failed command's stderr goes into the case's error
REAP_TIMEOUT_S = 5.0  # how long to wait for a killed command's pipes to close


def resolve_program(program: str, base_dir: str | None = None) -> str:
    """The file a ``command:`` target runs, or a ``ConfigError`` naming the program.

    A program with a path separator (``/``, and ``\\`` on Windows) is a path, relative to the
    suite's directory ``base_dir``. A bare name is looked up on ``PATH``, where ``shutil.which``
    adds the extensions Windows runs (``npx`` finds ``npx.cmd``)."""
    separators = [c for c in (os.sep, os.altsep) if c]
    if any(c in program for c in separators):
        path = os.path.abspath(os.path.join(base_dir or ".", program))
        found = shutil.which(path)
        if found is None:
            raise ConfigError(f"command target: {program!r} is not an executable file ({path})")
    else:
        found = shutil.which(program)
        if found is None:
            raise ConfigError(f"command target: program {program!r} not found on PATH")
    return os.path.abspath(found)  # the command runs elsewhere, so a relative path would break


class CommandSystem:
    """``command:<program> [args...]``: runs the program once per case (and per trial), in the
    suite's directory. It reads the same JSON an http system is sent from stdin, and prints either
    plain text or JSON ``{"output", "retrieved"?, "trace"?, "tokens_in"?, "tokens_out"?,
    "cost_usd"?}``. A failure (an exit code, a timeout, a program that will not start) is recorded
    on the case. No shell is used: the command line comes from the suite file, so it is trusted as
    much as a ``python:`` target."""

    def __init__(
        self,
        cfg: TargetConfig,
        base_dir: str | None = None,
        *,
        variant: str | None = None,
        vars: dict[str, Any] | None = None,
        trials: int = 1,
    ) -> None:
        self.cfg = cfg
        self.name = cfg.label or cfg.target
        self.variant = variant
        self.vars = dict(vars or {})
        self.trials = trials  # the input carries ``trial`` only when a case runs more than once
        self.cwd = os.path.abspath(base_dir) if base_dir else None
        try:
            argv = command_argv(cfg.target)
        except ValueError as exc:  # a target set after validation, such as ``--system``
            raise ConfigError(str(exc)) from exc
        self.argv = [resolve_program(argv[0], self.cwd), *argv[1:]]

    def _execute(self, stdin: bytes) -> tuple[int, bytes, bytes] | None:
        """Run the command and return its exit code, stdout and stderr, or None when it timed
        out (and was killed). A command still running when this is interrupted is killed too."""
        with subprocess.Popen(
            self.argv,
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **popen_group_kwargs(),
        ) as proc:
            try:
                stdout, stderr = proc.communicate(stdin, timeout=self.cfg.timeout_s)
            except subprocess.TimeoutExpired:
                kill_tree(proc)
                # Reap it and drain the pipes; if something outside its group still holds one,
                # leaving the ``with`` block closes it.
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.communicate(timeout=REAP_TIMEOUT_S)
                return None
            except BaseException:
                kill_tree(proc)
                raise
            return proc.returncode, stdout, stderr

    def run(self, case: Case, *, trial: int = 0) -> SystemOutput:
        body = request_body(
            case, variant=self.variant, vars=self.vars, trials=self.trials, trial=trial
        )
        try:
            stdin = json.dumps(body, ensure_ascii=False).encode("utf-8")
        except ValueError as exc:  # e.g. a lone surrogate, which UTF-8 cannot hold
            return SystemOutput(output="", error=f"cannot encode the case for the command: {exc}")
        start = time.perf_counter()
        try:
            done = self._execute(stdin)
        except OSError as exc:
            return SystemOutput(output="", error=f"cannot run {self.argv[0]}: {exc}")
        if done is None:
            return SystemOutput(output="", error=f"command timed out after {self.cfg.timeout_s:g}s")
        code, stdout, stderr = done
        if code != 0:
            tail = stderr.decode("utf-8", errors="replace").strip()[-STDERR_TAIL_CHARS:]
            message = f"command exited with code {code}"
            return SystemOutput(output="", error=f"{message}: {tail}" if tail else message)
        latency_ms = (time.perf_counter() - start) * 1000
        text = stdout.decode("utf-8", errors="replace")
        try:
            coerced = _coerce(_parse_stdout(text))
        except (TypeError, ValueError) as exc:
            return SystemOutput(output="", error=str(exc))
        return _finish(coerced, latency_ms, self.cfg.pricing)


def _parse_stdout(text: str) -> Any:
    """A JSON object is the result as an http system's JSON reply is; anything else, a number or
    a sentence, is the answer as plain text."""
    try:
        value = json.loads(text)
    except ValueError:
        return text.strip()
    return value if isinstance(value, dict) else text.strip()


def check_turns_supported(cfg: TargetConfig, cases: Iterable[Case]) -> None:
    """Multi-turn cases run on a python, http, command or direct model target. A coding-agent CLI
    takes one prompt per call, so it is refused up front rather than run on the first turn alone."""
    if cfg.kind not in CLI_PROVIDERS:
        return
    for case in cases:
        if case.turns:
            raise ConfigError(
                "multi-turn cases need a python, http, command or direct model target: "
                f"case {case.id!r} has turns, and {cfg.target} is a coding-agent CLI"
            )


def check_variant_vars(
    cfg: TargetConfig, variant: str, variant_vars: dict[str, Any], cases: Iterable[Case]
) -> None:
    """A variant's vars are template fields of a model prompt, next to the case's own fields, so
    a name in both is ambiguous. (A python, http or command system gets them apart, in
    ``ctx.vars``, the request body or the command's input, so nothing can clash there.)"""
    if not variant_vars or cfg.kind in ("python", "http", "command"):
        return
    for case in cases:
        fields = set(case.render_vars())
        for turn in case.turns:
            fields |= set(case.render_vars(turn=turn))
        clash = sorted(set(variant_vars) & fields)
        if clash:
            raise ConfigError(
                f"variant {variant!r}: vars {clash} are also fields of case {case.id!r}; "
                "rename the var or the field"
            )


def make_system(
    cfg: TargetConfig,
    cache: ResponseCache | None = None,
    client: httpx.Client | None = None,
    search_path: str | None = None,
    *,
    variant: str | None = None,
    vars: dict[str, Any] | None = None,
    trials: int = 1,
    cache_only: bool = False,
) -> System:
    kind = cfg.kind
    if kind == "python":
        return PythonSystem(
            cfg.target, search_path=search_path, pricing=cfg.pricing, variant=variant, vars=vars
        )
    if kind == "http":
        return HttpSystem(cfg, client=client, variant=variant, vars=vars, trials=trials)
    if kind == "command":
        return CommandSystem(cfg, base_dir=search_path, variant=variant, vars=vars, trials=trials)
    provider: Provider = make_provider(cfg, client=client)
    if cache is not None:
        provider = CachedProvider(
            provider, cache, base_url=cfg.base_url, cache_only=cache_only, role="system"
        )
    return LLMSystem(cfg, provider, vars=vars)
