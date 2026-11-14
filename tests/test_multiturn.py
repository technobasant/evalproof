"""Multi-turn cases: the ``turns`` / ``state`` / ``expect`` case fields, and how each target
family runs a conversation."""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import httpx
import pytest

from evalproof.cache import CachedProvider, ResponseCache, cache_key
from evalproof.config import ConfigError, GenerationParams, Pricing, TargetConfig, load_config
from evalproof.dataset import DatasetError, load_dataset
from evalproof.models import Case
from evalproof.providers import ChatResponse, Message, ProviderError
from evalproof.runner import run_suite
from evalproof.systems import HttpSystem, LLMSystem, PythonSystem, check_turns_supported
from tests.conftest import ScriptedProvider, mock_client

# --- dataset validation -----------------------------------------------------------------------


def _dataset(tmp_path: Path, *rows: dict[str, object]) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return path


def test_valid_agent_fields_load_and_are_readable(tmp_path: Path) -> None:
    (case,) = load_dataset(
        _dataset(
            tmp_path,
            {
                "id": "a",
                "input": "first",
                "turns": ["second", {"q": "third"}],
                "state": {"user": "u"},
                "expect": [{"type": "tool_called", "name": "x"}],
            },
        )
    )
    assert case.turns == ["second", {"q": "third"}]
    assert case.state == {"user": "u"}
    assert case.expect == [{"type": "tool_called", "name": "x"}]
    plain = Case(id="p", input="x")
    assert (plain.turns, plain.state, plain.expect) == ([], {}, [])


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"turns": "later"}, "case 'a': turns must be a list"),
        ({"turns": ["ok", 3]}, "case 'a': turns[1] must be a string or an object"),
        ({"state": ["x"]}, "case 'a': state must be an object"),
        ({"expect": {"type": "x"}}, "case 'a': expect must be a list"),
        ({"expect": [{"type": "x"}, "tool_called"]}, "case 'a': expect[1] must be an object"),
        ({"expect": [{"name": "x"}]}, "case 'a': expect[0] needs a 'type'"),
        ({"expect": [{"type": 3}]}, "case 'a': expect[0] needs a 'type'"),
    ],
)
def test_bad_agent_fields_name_the_case_and_the_line(
    tmp_path: Path, row: dict[str, object], message: str
) -> None:
    path = _dataset(tmp_path, {"id": "z", "input": "ok"}, {"id": "a", "input": "q", **row})
    with pytest.raises(DatasetError, match=r"cases\.jsonl:2: " + message.replace("[", r"\[")):
        load_dataset(path)


def test_the_checks_vocabulary_is_enforced_at_load(tmp_path: Path) -> None:
    good = [{"type": "tool_called", "name": "x"}]
    (case,) = load_dataset(_dataset(tmp_path, {"id": "a", "input": "q", "expect": good}))
    assert case.expect == good
    with pytest.raises(DatasetError, match=r"expect\[0\] has unknown type '\?'"):
        load_dataset(_dataset(tmp_path, {"id": "b", "input": "q", "expect": [{"type": "?"}]}))


# --- direct model targets ---------------------------------------------------------------------


def _llm(provider: ScriptedProvider, **kw: object) -> LLMSystem:
    cfg = TargetConfig(target="ollama:m", **kw)  # type: ignore[arg-type]
    return LLMSystem(cfg, provider)


def test_turns_run_in_order_with_the_history_resent() -> None:
    provider = ScriptedProvider(["r0", "r1", "r2"])
    system = _llm(provider, system_prompt="be brief")
    case = Case.model_validate({"id": "c", "input": "u0", "turns": ["u1", "u2"]})
    out = system.run(case)

    assert [[(m.role, m.content) for m in call] for call in provider.calls] == [
        [("system", "be brief"), ("user", "u0")],
        [("system", "be brief"), ("user", "u0"), ("assistant", "r0"), ("user", "u1")],
        [
            ("system", "be brief"),
            ("user", "u0"),
            ("assistant", "r0"),
            ("user", "u1"),
            ("assistant", "r1"),
            ("user", "u2"),
        ],
    ]
    assert out.output == "r2" and out.error is None
    assert out.trace is not None and out.trace.responses == ["r0", "r1", "r2"]
    assert out.trace.rounds == 3 and out.trace.tool_calls == []


def test_turns_sum_tokens_cost_and_latency() -> None:
    provider = ScriptedProvider(["a", "b", "c"])  # 10 in, 5 out, 3 ms per call
    pricing = Pricing(input_per_mtok=1.0, output_per_mtok=2.0)
    out = _llm(provider, pricing=pricing).run(
        Case.model_validate({"id": "c", "input": "q", "turns": ["x", "y"]})
    )
    assert (out.tokens_in, out.tokens_out, out.latency_ms) == (30, 15, 9.0)
    assert out.cost_usd == pytest.approx(3 * (10 * 1.0 + 5 * 2.0) / 1_000_000)
    assert out.cached is False


def test_every_turn_is_rendered_with_the_case_vars() -> None:
    provider = ScriptedProvider(["a", "b", "c"])
    system = _llm(provider, prompt_template="[{lang}] {input}")
    case = Case.model_validate(
        {"id": "c", "input": "one", "lang": "en", "turns": ["two", {"input": "three"}]}
    )
    system.run(case)
    users = [[m.content for m in call if m.role == "user"] for call in provider.calls]
    assert users[0] == ["[en] one"]
    assert users[1] == ["[en] one", "[en] two"]
    # A dict turn supplies its fields the way a dict input does; here ``input`` is the dict.
    assert users[2][-1] == "[en] {'input': 'three'}"


def test_dict_turns_fill_template_fields() -> None:
    provider = ScriptedProvider(["a", "b"])
    system = _llm(provider, prompt_template="Q: {question}")
    case = Case.model_validate(
        {"id": "c", "input": {"question": "q0"}, "turns": [{"question": "q1"}]}
    )
    system.run(case)
    assert [m.content for m in provider.calls[1] if m.role == "user"] == ["Q: q0", "Q: q1"]


def test_an_error_on_any_turn_ends_the_case() -> None:
    class FailsSecond(ScriptedProvider):
        def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
            if len(self.calls) == 1:
                self.calls.append(messages)
                raise ProviderError("down on turn 1")
            return super().chat(messages, params)

    provider = FailsSecond(["a", "b", "c"])
    out = _llm(provider).run(Case.model_validate({"id": "c", "input": "q", "turns": ["x", "y"]}))
    assert out.error == "down on turn 1" and out.output == ""
    assert len(provider.calls) == 2  # turn 2 was never sent
    # What turn 0 cost is kept, so budgets see it.
    assert out.tokens_in == 10 and out.tokens_out == 5
    assert out.trace is not None and out.trace.responses == ["a"]


def test_multi_turn_calls_go_through_the_cache(tmp_path: Path) -> None:
    inner = ScriptedProvider(["a", "b"])
    system = _llm(inner)
    system.provider = CachedProvider(inner, ResponseCache(tmp_path))
    case = Case.model_validate({"id": "c", "input": "q", "turns": ["x"]})
    first = system.run(case)
    assert not first.cached and len(inner.calls) == 2
    again = system.run(case)
    assert again.cached and len(inner.calls) == 2  # both turns were hits
    assert again.output == first.output and again.trace == first.trace
    # Same first turn, different second turn: turn 0 is a hit, only the new turn is called.
    other = ScriptedProvider(["fresh"])
    system.provider = CachedProvider(other, ResponseCache(tmp_path))
    changed = system.run(Case.model_validate({"id": "c", "input": "q", "turns": ["y"]}))
    assert len(other.calls) == 1 and changed.output == "fresh"
    assert [(m.role, m.content) for m in other.calls[0]] == [
        ("user", "q"),
        ("assistant", "a"),  # the cached reply to turn 0, not a new one
        ("user", "y"),
    ]


def test_single_turn_cache_keys_are_unchanged(tmp_path: Path) -> None:
    class Ollama(ScriptedProvider):
        name = "ollama"
        model = "qwen3:8b"

    cache = ResponseCache(tmp_path)
    inner = Ollama(["hello"])
    cfg = TargetConfig(target="ollama:qwen3:8b", system_prompt="be brief")
    system = LLMSystem(cfg, CachedProvider(inner, cache))
    out = system.run(Case(id="c", input="hi"))
    assert out.output == "hello" and out.trace is None
    # The same key tests/test_cache_systems.py pins for these messages, computed before the
    # trace and turns work: a one-turn case must keep hitting entries cached by older versions.
    pinned = "08d0b7fc706e8781e04a18eecd99fc97d56614a35c65b9985a5cf1439c2f7037"
    assert cache.get(pinned) is not None
    msgs = [Message(role="system", content="be brief"), Message(role="user", content="hi")]
    assert cache_key("ollama", "qwen3:8b", None, msgs, GenerationParams()) == pinned
    assert [(m.role, m.content) for m in inner.calls[0]] == [("system", "be brief"), ("user", "hi")]


# --- the other target families ----------------------------------------------------------------

AGENT = """
def chat(x, ctx):
    replies = [x, *ctx.turns]
    text = [str(r) for r in replies]
    return {"output": " | ".join(text), "trace": {"responses": text}}
"""


def test_python_system_receives_turns_and_state(tmp_path: Path) -> None:
    (tmp_path / "mt_agent.py").write_text(textwrap.dedent(AGENT), "utf-8")
    try:
        system = PythonSystem("python:mt_agent:chat", search_path=str(tmp_path))
        out = system.run(Case.model_validate({"id": "c", "input": "a", "turns": ["b", "c"]}))
        assert out.output == "a | b | c" and out.trace is not None
        assert out.trace.responses == ["a", "b", "c"]
    finally:
        sys.modules.pop("mt_agent", None)


def test_http_system_sends_turns_and_state() -> None:
    seen: list[dict] = []

    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"output": "ok"})

    system = HttpSystem(TargetConfig(target="http://sut/chat"), client=mock_client(handler))
    system.run(Case.model_validate({"id": "c", "input": "a", "turns": ["b"], "state": {"k": 1}}))
    assert seen[0]["turns"] == ["b"] and seen[0]["state"] == {"k": 1}


@pytest.mark.parametrize(
    "target", ["claude-code:haiku", "codex:gpt-5", "opencode:m", "cursor:m", "copilot:m"]
)
def test_cli_agent_targets_refuse_multi_turn_cases(tmp_path: Path, target: str) -> None:
    _dataset(
        tmp_path,
        {"id": "one", "input": "q"},
        {"id": "chat", "input": "q", "turns": ["more"]},
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"name: s\ndataset: cases.jsonl\nsystem: {{target: '{target}'}}\n"
        "scorers: [{name: c, type: contains, params: {any_of: [x]}}]\n",
        "utf-8",
    )
    with pytest.raises(
        ConfigError, match=r"multi-turn cases need a python, http or direct model target"
    ) as exc:
        run_suite(load_config(suite), use_cache=False)
    assert "'chat'" in str(exc.value) and target in str(exc.value)


def test_the_turns_check_leaves_single_turn_cases_and_other_targets_alone() -> None:
    single = [Case(id="one", input="q")]
    multi = [Case.model_validate({"id": "chat", "input": "q", "turns": ["more"]})]
    check_turns_supported(TargetConfig(target="claude-code:haiku"), single)  # nothing to refuse
    for target in ("ollama:m", "http://sut/chat", "python:mod:fn"):
        check_turns_supported(TargetConfig(target=target), multi)
