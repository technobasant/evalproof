"""Repeated trials: the config, the runner, how a trial reaches each kind of system, and what a
trial does to the response cache and to the seed a provider sends."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalproof.cache import CachedProvider, ResponseCache, trial_scope
from evalproof.cli import main
from evalproof.config import (
    MAX_TRIALS,
    ConfigError,
    GenerationParams,
    TargetConfig,
    load_config,
    suite_fingerprint,
    trial_params,
)
from evalproof.models import Case, CaseResult, Run
from evalproof.providers import ChatResponse, Message, make_provider
from evalproof.providers.anthropic import AnthropicProvider
from evalproof.providers.ollama import OllamaProvider
from evalproof.providers.openai import OpenAICompatibleProvider
from evalproof.providers.profiles import PROFILES
from evalproof.runner import load_run, run_suite, save_run
from evalproof.systems import HttpSystem, LLMSystem
from tests.conftest import ScriptedProvider, make_run, mock_client

CTX_SYSTEM = "def ask(x, ctx):\n    return f'{ctx.case_id}:{ctx.trial}'\n"


def _suite(
    tmp_path: Path,
    *,
    system: str | None = None,
    module: str = CTX_SYSTEM,
    extra: str = "",
    cases: int = 2,
    scorers: str = "[{name: m, type: contains, params: {any_of: [':']}}]",
) -> Path:
    # A module name of its own: python caches imported modules, and the tests' ``ask`` differ.
    modname = f"trialmod_{tmp_path.name}"
    system = system or f"{{target: 'python:{modname}:ask'}}"
    rows = [{"id": f"c{i}", "input": f"q{i}"} for i in range(cases)]
    (tmp_path / "cases.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), "utf-8")
    (tmp_path / f"{modname}.py").write_text(module, "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        f"name: s\ndataset: cases.jsonl\nsystem: {system}\nscorers: {scorers}\n{extra}", "utf-8"
    )
    return suite


# --- config and CLI ----------------------------------------------------------------------------


def test_trials_default_to_one_and_are_bounded(tmp_path: Path) -> None:
    assert load_config(_suite(tmp_path)).trials == 1
    assert load_config(_suite(tmp_path, extra="trials: 5\n")).trials == 5
    assert load_config(_suite(tmp_path, extra=f"trials: {MAX_TRIALS}\n")).trials == MAX_TRIALS
    for bad in (0, -1, MAX_TRIALS + 1):
        with pytest.raises(ConfigError):
            load_config(_suite(tmp_path, extra=f"trials: {bad}\n"))


def test_a_misspelt_trials_key_gets_the_did_you_mean_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(_suite(tmp_path, extra="trails: 3\n"))
    assert "trials" in str(exc.value)


def test_trials_are_not_part_of_the_suite_fingerprint(tmp_path: Path) -> None:
    one = load_config(_suite(tmp_path))
    many = load_config(_suite(tmp_path, extra="trials: 7\n"))
    assert suite_fingerprint(one) == suite_fingerprint(many)


@pytest.mark.parametrize("bad", ["0", "101", "-3", "two", "1.5"])
def test_run_trials_flag_outside_one_to_a_hundred_is_a_usage_error(
    tmp_path: Path, bad: str
) -> None:
    suite = _suite(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["run", str(suite), "--trials", bad, "--out", str(tmp_path / "r.json")])
    assert exc.value.code == 2


def test_run_trials_flag_overrides_the_suite_and_prints_the_plan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path, extra="trials: 2\n")
    out = tmp_path / "r.json"
    assert main(["run", str(suite), "--trials", "3", "--out", str(out), "--no-cache"]) == 0
    assert "2 cases × 3 trials = 6 runs" in capsys.readouterr().err
    run = load_run(out)
    assert run.info.trials == 3 and len(run.results) == 6


def test_the_plan_line_is_left_out_for_one_trial_and_for_quiet(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _suite(tmp_path)
    assert main(["run", str(suite), "--out", str(tmp_path / "a.json"), "--no-cache"]) == 0
    assert "trials =" not in capsys.readouterr().err
    argv = ["run", str(suite), "--trials", "3", "--quiet", "--out", str(tmp_path / "b.json")]
    assert main([*argv, "--no-cache"]) == 0
    assert "trials =" not in capsys.readouterr().err


def test_the_suite_trials_apply_without_the_flag(tmp_path: Path) -> None:
    run = run_suite(load_config(_suite(tmp_path, extra="trials: 2\n")), use_cache=False)
    assert run.info.trials == 2 and run.trials == 2 and len(run.results) == 4
    with pytest.raises(ConfigError):
        run_suite(load_config(_suite(tmp_path)), use_cache=False, trials=0)


# --- k = 1 and old files -----------------------------------------------------------------------


def test_one_trial_only_adds_trial_fields_to_the_run_file(tmp_path: Path) -> None:
    run = run_suite(load_config(_suite(tmp_path)), use_cache=False)
    path = save_run(run, tmp_path / "run.json")
    doc = json.loads(path.read_text("utf-8"))
    assert doc["info"]["trials"] == 1
    assert [r["trial"] for r in doc["results"]] == [0, 0]
    for r in doc["results"]:
        del r["trial"]
    del doc["info"]["trials"]
    assert Run.model_validate(doc) == run  # nothing else changed


def test_an_old_run_file_loads_as_trial_zero_of_one(tmp_path: Path) -> None:
    run = run_suite(load_config(_suite(tmp_path)), use_cache=False)
    doc = json.loads(run.model_dump_json())
    del doc["info"]["trials"]
    for r in doc["results"]:
        del r["trial"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(doc), "utf-8")
    old = load_run(path)
    assert old.trials == 1 and old.info.trials == 1
    assert [r.trial for r in old.results] == [0, 0]
    assert list(old.by_id()) == ["c0", "c1"]


def _result(case_id: str, trial: int, text: str = "x") -> CaseResult:
    return CaseResult.model_validate(
        {"case_id": case_id, "trial": trial, "output": {"output": text}, "scores": {}}
    )


def test_by_id_is_the_first_trial_zero_and_by_case_groups_in_trial_order() -> None:
    results = [
        _result("a", 0, "first"),
        _result("a", 2),
        _result("b", 0),
        _result("a", 1),
        _result("a", 0, "dupe"),
        _result("z", 1),  # a case with no trial 0 has no single-result view
    ]
    run = make_run({"a": {"m": True}}).model_copy(update={"results": results})
    by_id = run.by_id()
    assert list(by_id) == ["a", "b"]
    assert by_id["a"].output.output == "first"  # the first trial-0 result wins
    grouped = run.by_case()
    assert [r.trial for r in grouped["a"]] == [0, 0, 1, 2]
    assert [r.trial for r in grouped["b"]] == [0] and [r.trial for r in grouped["z"]] == [1]


# --- how a trial reaches each system -----------------------------------------------------------


def test_a_python_system_sees_each_trial_in_ctx(tmp_path: Path) -> None:
    run = run_suite(load_config(_suite(tmp_path)), use_cache=False, trials=3)
    assert [(r.case_id, r.trial, r.output.output) for r in run.results] == [
        (f"c{c}", t, f"c{c}:{t}") for c in range(2) for t in range(3)
    ]


def test_results_stay_in_case_then_trial_order_under_concurrency(tmp_path: Path) -> None:
    slow_first = (
        "import time\n"
        "def ask(x, ctx):\n"
        "    time.sleep(0.02 * (3 - ctx.trial))  # later trials finish first\n"
        "    return f'{ctx.case_id}:{ctx.trial}'\n"
    )
    suite = _suite(tmp_path, module=slow_first, extra="concurrency: 6\n", cases=3)
    seen: list[tuple[int, int]] = []
    run = run_suite(
        load_config(suite),
        use_cache=False,
        trials=3,
        progress=lambda i, n, res: seen.append((i, n)),
    )
    assert [(r.case_id, r.trial) for r in run.results] == [
        (f"c{c}", t) for c in range(3) for t in range(3)
    ]
    assert [i for i, _ in seen] == list(range(1, 10)) and {n for _, n in seen} == {9}


def test_every_job_runs_through_the_pool(tmp_path: Path) -> None:
    started = time.perf_counter()
    nap = "import time\ndef ask(x, ctx):\n    time.sleep(0.1)\n    return 'a:b'\n"
    suite = _suite(tmp_path, module=nap, extra="concurrency: 8\n", cases=2)
    run = run_suite(load_config(suite), use_cache=False, trials=4)
    assert len(run.results) == 8
    assert time.perf_counter() - started < 0.7  # 8 sleeps of 0.1 s in parallel, not in a row


def test_an_http_body_carries_trial_only_with_more_than_one(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, text="ok:ok")

    case = Case(id="c", input="q")
    one = HttpSystem(TargetConfig(target="http://sut/x"), client=mock_client(handler))
    one.run(case)
    assert bodies == [{"id": "c", "input": "q"}]  # byte-identical to 0.6

    bodies.clear()
    many = HttpSystem(TargetConfig(target="http://sut/x"), client=mock_client(handler), trials=3)
    for t in range(3):
        many.run(case, trial=t)
    assert [b["trial"] for b in bodies] == [0, 1, 2]
    assert all(set(b) == {"id", "input", "trial"} for b in bodies)


def test_an_http_suite_run_sends_trial_from_the_second_trial_on(tmp_path: Path) -> None:
    bodies: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(req.content))
        return httpx.Response(200, text="a:b")

    suite = _suite(tmp_path, system="{target: 'http://sut/x'}", cases=1)
    cfg = load_config(suite)
    run_suite(cfg, use_cache=False, client=mock_client(handler))
    assert all("trial" not in b for b in bodies)
    bodies.clear()
    run_suite(cfg, use_cache=False, client=mock_client(handler), trials=2)
    assert [b["trial"] for b in bodies] == [0, 1]


# --- the cache and the seed --------------------------------------------------------------------

MSGS = [Message(role="system", content="be brief"), Message(role="user", content="What is 2+2?")]


def _any_handler(req: httpx.Request) -> httpx.Response:
    url = str(req.url)
    if "api/chat" in url:
        return httpx.Response(
            200,
            json={
                "message": {"content": "hi"},
                "model": "m",
                "prompt_eval_count": 1,
                "eval_count": 1,
            },
        )
    if "v1/messages" in url:
        return httpx.Response(
            200,
            json={
                "content": [{"type": "text", "text": "hi"}],
                "model": "m",
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "stop_reason": "end_turn",
            },
        )
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
            "model": "m",
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        },
    )


class _KeyOnly:
    """A provider with a real one's name, model and cache salt, and a canned reply: the cache key
    of a coding-agent CLI without starting the CLI."""

    def __init__(self, real: Any) -> None:
        self.name, self.model, self.cache_salt = real.name, real.model, real.cache_salt

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        return ChatResponse(text="hi", model="m", tokens_in=1, tokens_out=1, latency_ms=1.0)


def _providers() -> dict[str, tuple[Any, GenerationParams]]:
    client = mock_client(_any_handler)
    return {
        "ollama": (OllamaProvider("qwen3:1.7b", client=client), GenerationParams()),
        "openai-hosted-default": (
            OpenAICompatibleProvider(PROFILES["openai"], "gpt-4o-mini", api_key="k", client=client),
            GenerationParams(),
        ),
        "openai-hosted-explicit": (
            OpenAICompatibleProvider(PROFILES["openai"], "gpt-4o-mini", api_key="k", client=client),
            GenerationParams(temperature=0.2, seed=7),
        ),
        "vllm-local": (
            OpenAICompatibleProvider(
                PROFILES["vllm"], "m", base_url="http://localhost:8000/v1", client=client
            ),
            GenerationParams(),
        ),
        "anthropic": (
            AnthropicProvider("claude-sonnet-4-5", api_key="k", client=client),
            GenerationParams(),
        ),
        "claude-code": (
            _KeyOnly(make_provider(TargetConfig(target="claude-code:sonnet"))),
            GenerationParams(),
        ),
        "codex": (_KeyOnly(make_provider(TargetConfig(target="codex:gpt-5"))), GenerationParams()),
    }


# Computed on evalproof 0.6, before trials existed, for the same calls. A trial-0 call must keep
# these keys: a changed key makes every cached run miss. See ``test_trial_zero_keys_are_pinned``.
TRIAL_ZERO_KEYS = {
    "ollama": "5f749d27988861905482f45d7e05ab29db118471d54656fb877284ff1881babc",
    "openai-hosted-default": "f84e6808f773a27f5957358f4fa0d05dd8ebe355f98b382bf2cff6639f44da10",
    "openai-hosted-explicit": "052a9e88d601cb10a9f0d2416a2368a87233d851c6433102029c1a011fd7c491",
    "vllm-local": "c671aacefd8a817613774369994e93fd3a5b93ea187cbec3d0b76ebc29c78d4a",
    "anthropic": "daf07306b4a4a370b1d3b8ba7f6e8e1c312cf32dd05535e23fe0a5b26d01eec8",
    "claude-code": "f9981d4e253aacf88169a695cab43bb364fe028ff1c8493c3965aa34e6803b44",
    "codex": "ef7d59884737cbcd99938433adc882e02a100cc1cc72124dbc11180ee306902e",
}


def _key(provider: Any, params: GenerationParams, trial: int, tmp_path: Path) -> str:
    """The cache key of one call, as ``LLMSystem.run`` makes it for ``trial``: the file name
    the cached reply is stored under."""
    cache_dir = tmp_path / f"cache-{trial}-{id(provider)}"
    cached = CachedProvider(provider, ResponseCache(cache_dir), base_url=None)
    with trial_scope(trial):
        cached.chat(MSGS, trial_params(params, trial))
    return next(cache_dir.glob("*/*.json")).stem


@pytest.mark.parametrize("name", sorted(TRIAL_ZERO_KEYS))
def test_trial_zero_keys_are_pinned(name: str, tmp_path: Path) -> None:
    provider, params = _providers()[name]
    assert _key(provider, params, 0, tmp_path) == TRIAL_ZERO_KEYS[name]


@pytest.mark.parametrize("name", sorted(TRIAL_ZERO_KEYS))
def test_later_trials_have_keys_of_their_own(name: str, tmp_path: Path) -> None:
    provider, params = _providers()[name]
    keys = [_key(provider, params, t, tmp_path) for t in range(4)]
    assert len(set(keys)) == 4 and keys[0] == TRIAL_ZERO_KEYS[name]


def test_outside_a_trial_scope_the_key_is_trial_zeros(tmp_path: Path) -> None:
    provider, params = _providers()["ollama"]
    cache_dir = tmp_path / "plain"
    CachedProvider(provider, ResponseCache(cache_dir), base_url=None).chat(MSGS, params)
    assert next(cache_dir.glob("*/*.json")).stem == TRIAL_ZERO_KEYS["ollama"]


def test_a_trial_is_cached_on_its_own_and_hit_again(tmp_path: Path) -> None:
    inner = ScriptedProvider(["a", "b", "c"])
    cached = CachedProvider(inner, ResponseCache(tmp_path))
    system = LLMSystem(TargetConfig(target="ollama:m"), cached)
    case = Case(id="c", input="q")
    first = [system.run(case, trial=t).output for t in range(3)]
    assert first == ["a", "b", "c"] and len(inner.calls) == 3  # three calls, no sharing
    again = [system.run(case, trial=t) for t in range(3)]
    assert [o.output for o in again] == first and all(o.cached for o in again)
    assert len(inner.calls) == 3


def test_a_conversation_is_salted_per_trial_too(tmp_path: Path) -> None:
    inner = ScriptedProvider(lambda msgs: f"r{len(msgs)}")
    cached = CachedProvider(inner, ResponseCache(tmp_path))
    system = LLMSystem(TargetConfig(target="ollama:m"), cached)
    case = Case(id="c", input="q", turns=["more"])
    system.run(case, trial=0)
    calls = len(inner.calls)
    system.run(case, trial=1)
    assert len(inner.calls) == 2 * calls  # no turn of trial 1 reused one of trial 0
    system.run(case, trial=1)
    assert len(inner.calls) == 2 * calls  # and a repeat of trial 1 is all hits


def test_no_cache_runs_every_trial_for_real(tmp_path: Path) -> None:
    suite = _suite(tmp_path, system="{target: 'ollama:m'}", cases=2)
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/chat"):
            calls.append(1)
        return _any_handler(req)

    run = run_suite(load_config(suite), use_cache=False, client=mock_client(handler), trials=3)
    assert len(calls) == 6 and len(run.results) == 6


def test_a_cached_run_makes_one_call_per_case_and_trial_and_none_the_second_time(
    tmp_path: Path,
) -> None:
    suite = _suite(tmp_path, system="{target: 'ollama:m'}", cases=2, extra="cache_dir: cache\n")
    calls: list[int] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path.endswith("/chat"):
            calls.append(1)
        return _any_handler(req)

    cfg = load_config(suite)
    run_suite(cfg, client=mock_client(handler), trials=3)
    assert len(calls) == 6
    again = run_suite(cfg, client=mock_client(handler), trials=3)
    assert len(calls) == 6 and all(r.output.cached for r in again.results)


def test_trial_params_shift_the_seed_and_keep_what_was_explicit() -> None:
    default = GenerationParams()
    assert trial_params(default, 0) is default
    shifted = trial_params(default, 3)
    assert shifted.seed == 3 and "seed" not in shifted.model_fields_set
    explicit = GenerationParams(temperature=0.2, seed=7)
    later = trial_params(explicit, 2)
    assert later.seed == 9 and later.model_fields_set == {"temperature", "seed"}
    no_seed = GenerationParams(seed=None)
    assert trial_params(no_seed, 5) is no_seed


def _bodies(
    tmp_path: Path, system: str, monkeypatch: pytest.MonkeyPatch, trials: int = 3
) -> list[dict[str, Any]]:
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    suite = _suite(tmp_path, system=system, cases=1)
    seen: list[dict[str, Any]] = []

    def handler(req: httpx.Request) -> httpx.Response:
        if req.method == "POST":
            seen.append(json.loads(req.content))
        return _any_handler(req)

    run_suite(load_config(suite), use_cache=False, client=mock_client(handler), trials=trials)
    return seen


def test_a_hosted_profile_without_an_explicit_seed_sends_none_in_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = _bodies(tmp_path, "{target: 'openai:gpt-4o-mini'}", monkeypatch)
    assert len(bodies) == 3
    for body in bodies:
        assert "seed" not in body and "temperature" not in body


def test_a_hosted_profile_sends_its_explicit_seed_plus_the_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = "{target: 'openai:gpt-4o-mini', params: {temperature: 0.2, seed: 7}}"
    bodies = _bodies(tmp_path, system, monkeypatch)
    assert [b["seed"] for b in bodies] == [7, 8, 9]
    assert all(b["temperature"] == 0.2 for b in bodies)


def test_a_local_profile_sends_seed_plus_the_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system = "{target: 'vllm:m', base_url: 'http://localhost:8000/v1'}"
    bodies = _bodies(tmp_path, system, monkeypatch)
    assert [b["seed"] for b in bodies] == [0, 1, 2]


def test_ollama_sends_seed_plus_the_trial_in_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = _bodies(tmp_path, "{target: 'ollama:m', params: {seed: 4}}", monkeypatch)
    assert [b["options"]["seed"] for b in bodies] == [4, 5, 6]


def test_anthropic_sends_no_seed_and_nothing_new_in_any_trial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = _bodies(tmp_path, "{target: 'anthropic:claude-sonnet-4-5'}", monkeypatch)
    assert len(bodies) == 3
    assert all(b == bodies[0] for b in bodies)  # the same request in every trial
    assert all("seed" not in b and "trial" not in b for b in bodies)


def test_one_trial_sends_the_same_request_as_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = _bodies(tmp_path, "{target: 'openai:gpt-4o-mini'}", monkeypatch, trials=1)
    assert len(bodies) == 1
    assert set(bodies[0]) == {"model", "messages", PROFILES["openai"].max_tokens_field}


def test_a_model_target_never_gets_a_trial_argument() -> None:
    """The provider protocol is as it was: a custom provider written for 0.6 still works."""

    class Old:
        name, model = "old", "m"

        def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
            return ChatResponse(text="ok", model="m", tokens_in=1, tokens_out=1, latency_ms=1.0)

    system = LLMSystem(TargetConfig(target="ollama:m"), Old())
    case = Case(id="c", input="q")
    assert system.run(case, trial=0).output == system.run(case, trial=2).output == "ok"
