"""The trace contract: a system may report what it did, and the runner settles tokens and cost."""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import FrozenInstanceError
from pathlib import Path

import httpx
import pytest

import evalproof
from evalproof.config import Pricing, TargetConfig, load_config
from evalproof.dataset import case_hash
from evalproof.models import Case, ModelUsage, SystemOutput, ToolCall, Trace
from evalproof.runner import load_run, run_suite, save_run
from evalproof.systems import HttpSystem, PythonSystem, RunContext, _coerce, _finish
from tests.conftest import mock_client

ROOT = Path(__file__).resolve().parents[1]

AGENTS = """
def old(x):
    return f"old {x}"

def with_ctx(x, ctx):
    seen = [ctx.case_id, ctx.input, ctx.turns, ctx.state, ctx.variant, ctx.vars, ctx.trial]
    return {"output": "|".join(str(s) for s in seen)}

def kw_only(x, *, ctx=None):
    return f"kw {ctx.case_id}"

def swallows(x, **kwargs):
    assert "ctx" not in kwargs
    return "kwargs ok"

def agent(x):
    return {
        "output": "done",
        "retrieved": ["d1"],
        "trace": {
            "responses": ["done"],
            "tool_calls": [{"name": "search", "args": {"q": x}, "result": [1, 2]}],
            "rounds": 2,
            "usage": [
                {"model": "m", "tokens_in": 100, "tokens_out": 10, "cost_usd": 0.5},
                {"role": "judge", "tokens_in": 20, "tokens_out": 5, "cost_usd": 0.25},
            ],
        },
    }

def mutates_state(x, ctx):
    ctx.state["nested"]["n"] += 1
    ctx.turns.append("extra")
    return "mutated"

def bad_trace(x):
    return {"output": "o", "trace": {"tool_calls": [{"args": {}}]}}

def bad_trace_type(x):
    return {"output": "o", "trace": "nope"}

def bad_tokens(x):
    return {"output": "o", "tokens_in": "many"}
"""


@pytest.fixture
def agents(tmp_path: Path):
    (tmp_path / "trace_agents.py").write_text(textwrap.dedent(AGENTS), "utf-8")
    yield tmp_path
    sys.modules.pop("trace_agents", None)


def _py(name: str, path: Path, **kw) -> PythonSystem:
    return PythonSystem(f"python:trace_agents:{name}", search_path=str(path), **kw)


def test_models_default_to_empty() -> None:
    call = ToolCall(name="t")
    assert (call.args, call.result, call.ok, call.error, call.id, call.turn, call.latency_ms) == (
        {},
        None,
        True,
        None,
        None,
        0,
        None,
    )
    usage = ModelUsage()
    assert (usage.role, usage.model, usage.tokens_in, usage.cached_tokens, usage.cost_usd) == (
        "agent",
        None,
        0,
        0,
        None,
    )
    trace = Trace()
    assert trace.responses == [] and trace.tool_calls == [] and trace.rounds is None
    assert trace.usage == [] and trace.meta == {}
    assert SystemOutput(output="x").trace is None


def test_old_run_files_load_without_a_trace_or_variant() -> None:
    run = load_run(ROOT / "examples" / "support-qa" / "runs" / "baseline.json")
    assert run.results and all(r.output.trace is None for r in run.results)
    assert run.info.variant is None
    # And a hand-written pre-0.6 output (no trace key at all) validates as it always did.
    out = SystemOutput.model_validate({"output": "a", "latency_ms": 1.0, "tokens_in": 3})
    assert out.trace is None and out.tokens_in == 3


def test_a_trace_survives_a_run_file(agents: Path, tmp_path: Path) -> None:
    (tmp_path / "cases.jsonl").write_text(json.dumps({"id": "a", "input": "q"}) + "\n", "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:trace_agents:agent'}\n"
        "scorers: [{name: c, type: contains, params: {any_of: [done]}}]\n",
        "utf-8",
    )
    run = run_suite(load_config(suite), use_cache=False)
    path = save_run(run, tmp_path / "run.json")
    out = load_run(path).results[0].output
    assert out.trace is not None
    assert out.trace.tool_calls[0].name == "search" and out.trace.tool_calls[0].args == {"q": "q"}
    assert out.trace.rounds == 2 and out.retrieved == ["d1"]
    assert (out.tokens_in, out.tokens_out, out.cost_usd) == (120, 15, 0.75)


def test_runcontext_is_public_and_frozen() -> None:
    assert "RunContext" in evalproof.__all__ and evalproof.RunContext is RunContext
    for name in ("Trace", "ToolCall", "ModelUsage"):
        assert name in evalproof.__all__ and hasattr(evalproof, name)
    ctx = RunContext(case_id="c", input="i")
    assert (ctx.turns, ctx.state, ctx.variant, ctx.vars, ctx.trial) == ([], {}, None, {}, 0)
    with pytest.raises(FrozenInstanceError):
        ctx.case_id = "other"  # type: ignore[misc]


def test_a_function_without_ctx_is_called_with_the_input_only(agents: Path) -> None:
    case = Case(id="c", input="hi")
    assert _py("old", agents).run(case).output == "old hi"
    # ``**kwargs`` alone does not opt in: only a parameter named ctx does.
    assert _py("swallows", agents).run(case).output == "kwargs ok"


def test_a_function_with_ctx_gets_the_context(agents: Path) -> None:
    case = Case.model_validate(
        {"id": "c", "input": "hi", "turns": ["next"], "state": {"user": "u1"}}
    )
    system = _py("with_ctx", agents, variant="v2", vars={"model": "m"})
    out = system.run(case)
    assert out.error is None
    assert out.output == "c|hi|['next']|{'user': 'u1'}|v2|{'model': 'm'}|0"
    # A keyword-only ctx works too, and without variants the context is empty.
    assert _py("kw_only", agents).run(case).output == "kw c"
    bare = _py("with_ctx", agents).run(Case(id="b", input="x")).output
    assert bare == "b|x|[]|{}|None|{}|0"


def test_a_system_cannot_change_the_case_through_its_context(agents: Path) -> None:
    case = Case.model_validate(
        {"id": "c", "input": "i", "turns": ["t"], "state": {"nested": {"n": 0}}}
    )
    before = case_hash(case)
    system = _py("mutates_state", agents)
    assert system.run(case).output == "mutated"
    assert system.run(case).output == "mutated"
    assert case.state == {"nested": {"n": 0}} and case.turns == ["t"]
    assert case_hash(case) == before


def test_coerce_accepts_the_extended_dict() -> None:
    plain = _coerce("text")
    assert (plain.output, plain.retrieved, plain.trace) == ("text", [], None)
    full = _coerce(
        {
            "output": 5,
            "retrieved": ["a", 2],
            "trace": {"responses": ["5"]},
            "tokens_in": 7,
            "tokens_out": 3.0,
            "cost_usd": 1,
        }
    )
    assert full.output == "5" and full.retrieved == ["a", "2"]
    assert full.trace == Trace(responses=["5"])
    assert (full.tokens_in, full.tokens_out, full.cost_usd) == (7, 3, 1.0)
    given = Trace(rounds=1)
    assert _coerce({"output": "o", "trace": given}).trace is given
    with pytest.raises(TypeError, match="string or a dict"):
        _coerce(42)
    with pytest.raises(TypeError, match="'tokens_in' must be a number"):
        _coerce({"output": "o", "tokens_in": True})


def test_a_malformed_result_is_recorded_on_the_case(agents: Path) -> None:
    case = Case(id="c", input="q")
    out = _py("bad_trace", agents).run(case)
    assert out.output == "" and out.trace is None
    assert out.error is not None and "invalid 'trace'" in out.error and "name" in out.error
    assert "'trace' must be a dict" in (_py("bad_trace_type", agents).run(case).error or "")
    assert "'tokens_in' must be a number" in (_py("bad_tokens", agents).run(case).error or "")


def _result(**kw) -> object:
    return _coerce({"output": "o", **kw})


def test_rollup_sums_usage_when_the_system_sets_nothing() -> None:
    usage = [
        {"tokens_in": 100, "tokens_out": 10, "cost_usd": 0.5},
        {"tokens_in": 20, "tokens_out": 5, "cost_usd": 0.25},
        {"tokens_in": 1, "tokens_out": 1},  # no cost: it adds tokens, not dollars
    ]
    out = _finish(_result(trace={"usage": usage}), 12.0, Pricing())  # type: ignore[arg-type]
    assert (out.tokens_in, out.tokens_out) == (121, 16)
    assert out.cost_usd == pytest.approx(0.75)
    assert out.latency_ms == 12.0


def test_rollup_prices_the_tokens_when_no_usage_has_a_cost() -> None:
    pricing = Pricing(input_per_mtok=2.0, output_per_mtok=10.0)
    usage = [{"tokens_in": 1_000_000, "tokens_out": 100_000}, {"tokens_in": 500_000}]
    out = _finish(_result(trace={"usage": usage}), 0.0, pricing)  # type: ignore[arg-type]
    assert (out.tokens_in, out.tokens_out) == (1_500_000, 100_000)
    assert out.cost_usd == pytest.approx(1.5 * 2.0 + 0.1 * 10.0)
    # Without a price the cost is zero, not missing.
    free = _finish(_result(trace={"usage": usage}), 0.0, Pricing())  # type: ignore[arg-type]
    assert free.cost_usd == 0.0


def test_rollup_keeps_what_the_system_set() -> None:
    usage = [{"tokens_in": 100, "tokens_out": 10, "cost_usd": 0.5}]
    pricing = Pricing(input_per_mtok=1.0)
    only_cost = _finish(_result(trace={"usage": usage}, cost_usd=9.0), 0.0, pricing)  # type: ignore[arg-type]
    assert (only_cost.tokens_in, only_cost.tokens_out, only_cost.cost_usd) == (100, 10, 9.0)
    only_tokens = _finish(_result(trace={"usage": usage}, tokens_in=3, tokens_out=0), 0.0, pricing)  # type: ignore[arg-type]
    assert (only_tokens.tokens_in, only_tokens.tokens_out) == (3, 0)
    assert only_tokens.cost_usd == pytest.approx(0.5)  # the usage cost, not the price
    # Tokens given without a trace are priced; a result with neither costs nothing.
    priced = _finish(_result(tokens_in=1_000_000), 0.0, pricing)  # type: ignore[arg-type]
    assert priced.cost_usd == pytest.approx(1.0)
    bare = _finish(_result(), 0.0, pricing)  # type: ignore[arg-type]
    assert (bare.tokens_in, bare.tokens_out, bare.cost_usd) == (0, 0, 0.0)


def test_a_python_agent_is_priced_from_the_suite_target(agents: Path) -> None:
    cfg = TargetConfig(target="python:trace_agents:agent", pricing=Pricing(input_per_mtok=1.0))
    # The agent's usage carries costs, so the price is ignored; the figures are the usage sums.
    system = PythonSystem(cfg.target, search_path=str(agents), pricing=cfg.pricing)
    out = system.run(Case(id="c", input="q"))
    assert (out.tokens_in, out.tokens_out, out.cost_usd) == (120, 15, 0.75)


def test_http_body_adds_fields_only_when_they_are_set() -> None:
    bodies: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, json={"output": "ok"})

    cfg = TargetConfig(target="http://sut/answer")
    plain = HttpSystem(cfg, client=mock_client(handler))
    plain.run(Case(id="a", input="q"))
    assert bodies[-1] == {"id": "a", "input": "q"}

    rich = HttpSystem(cfg, client=mock_client(handler), variant="v2", vars={"model": "m"})
    case = Case.model_validate(
        {"id": "b", "input": "q", "turns": ["t1", {"k": 1}], "state": {"s": 1}}
    )
    rich.run(case)
    assert bodies[-1] == {
        "id": "b",
        "input": "q",
        "turns": ["t1", {"k": 1}],
        "state": {"s": 1},
        "variant": "v2",
        "vars": {"model": "m"},
    }
    # Empty values stay out: a variant with no vars sends only its name.
    named = HttpSystem(cfg, client=mock_client(handler), variant="v1")
    named.run(Case(id="c", input="q"))
    assert bodies[-1] == {"id": "c", "input": "q", "variant": "v1"}


def test_http_parses_the_extended_response() -> None:
    reply = {
        "output": "answer",
        "retrieved": ["d"],
        "trace": {
            "responses": ["answer"],
            "tool_calls": [{"name": "lookup", "args": {"id": 1}, "ok": False, "error": "404"}],
            "usage": [{"tokens_in": 10, "tokens_out": 4, "cost_usd": 0.01}],
        },
    }
    cfg = TargetConfig(target="http://sut/answer", pricing=Pricing(input_per_mtok=100.0))
    system = HttpSystem(cfg, client=mock_client(lambda r: httpx.Response(200, json=reply)))
    out = system.run(Case(id="c", input="q"))
    assert out.error is None and out.retrieved == ["d"]
    assert out.trace is not None and out.trace.tool_calls[0].ok is False
    assert (out.tokens_in, out.tokens_out, out.cost_usd) == (10, 4, 0.01)
    # A response whose trace is malformed is an errored case, not a crash.
    broken = {"output": "x", "trace": {"usage": [{"tokens_in": "lots"}]}}
    bad = HttpSystem(cfg, client=mock_client(lambda r: httpx.Response(200, json=broken)))
    assert "invalid 'trace'" in (bad.run(Case(id="c", input="q")).error or "")
