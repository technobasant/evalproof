"""Systems under test: an LLM prompt, a Python function, or an HTTP endpoint.

A system failure is recorded on the case, not raised: one timeout should not throw
away an hour of judging. Errored cases score as failures and are listed in reports.

A python or http system may be an agent. It then returns, besides its ``output``, a ``trace`` of
what it did (see ``evalproof.models.Trace``) and optionally its token counts and cost; the runner
fills any token or cost figure it leaves out from the trace's usage (``_finish``).
"""

from __future__ import annotations

import copy
import importlib
import inspect
import string
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx
from pydantic import ValidationError

from evalproof.cache import CachedProvider, ResponseCache, trial_scope
from evalproof.config import (
    ConfigError,
    GenerationParams,
    Pricing,
    TargetConfig,
    trial_params,
)
from evalproof.models import Case, SystemOutput, Trace
from evalproof.providers import (
    CLI_PROVIDERS,
    Message,
    Provider,
    ProviderError,
    make_provider,
    response_cost,
    with_retries,
)


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
    """What a python or http system returned, before token and cost figures are settled."""

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
        self._client = client or httpx.Client(timeout=cfg.timeout_s)

    def _body(self, case: Case, trial: int = 0) -> dict[str, Any]:
        body: dict[str, Any] = {"id": case.id, "input": case.input}
        for key, value in (
            ("turns", case.turns),
            ("state", case.state),
            ("variant", self.variant),
            ("vars", self.vars),
        ):
            if value:
                body[key] = value
        if self.trials > 1:
            body["trial"] = trial
        return body

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


def check_turns_supported(cfg: TargetConfig, cases: Iterable[Case]) -> None:
    """Multi-turn cases run on a python, http or direct model target. A coding-agent CLI takes
    one prompt per call, so it is refused up front rather than run on the first turn alone."""
    if cfg.kind not in CLI_PROVIDERS:
        return
    for case in cases:
        if case.turns:
            raise ConfigError(
                "multi-turn cases need a python, http or direct model target: "
                f"case {case.id!r} has turns, and {cfg.target} is a coding-agent CLI"
            )


def check_variant_vars(
    cfg: TargetConfig, variant: str, variant_vars: dict[str, Any], cases: Iterable[Case]
) -> None:
    """A variant's vars are template fields of a model prompt, next to the case's own fields, so
    a name in both is ambiguous. (A python or http system gets them apart, in ``ctx.vars`` or
    the request body, so nothing can clash there.)"""
    if not variant_vars or cfg.kind in ("python", "http"):
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
) -> System:
    kind = cfg.kind
    if kind == "python":
        return PythonSystem(
            cfg.target, search_path=search_path, pricing=cfg.pricing, variant=variant, vars=vars
        )
    if kind == "http":
        return HttpSystem(cfg, client=client, variant=variant, vars=vars, trials=trials)
    provider: Provider = make_provider(cfg, client=client)
    if cache is not None:
        provider = CachedProvider(provider, cache, base_url=cfg.base_url)
    return LLMSystem(cfg, provider, vars=vars)
