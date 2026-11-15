"""``--cache-only``: answer every model call from the response cache, or stop.

No model runs here. The providers are scripted stubs, an Ollama mock transport that fails the test
if it is reached, and a fake coding-agent binary whose call log shows what was executed.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evalproof.cache import CachedProvider, ResponseCache, case_scope
from evalproof.cli import main
from evalproof.config import (
    ConfigError,
    GenerationParams,
    ScorerConfig,
    SuiteConfig,
    TargetConfig,
    load_config,
)
from evalproof.gate import EXIT_PASS, EXIT_USAGE
from evalproof.models import Case, SystemOutput
from evalproof.providers import CacheMiss, Message, ProviderError
from evalproof.providers.agent_cli import cli_version
from evalproof.providers.base import TransientError
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import LLMJudge, RubricSetJudge
from evalproof.systems import LLMSystem
from tests.conftest import ScriptedProvider, mock_client
from tests.fakecli import FakeCLI, claude_ok, install

MSGS = [Message(role="user", content="q")]
PARAMS = GenerationParams()
VERDICT = '{"reason": "fine", "pass": true, "score": 1}'


class CountingCache(ResponseCache):
    """A cache that counts lookups, to show a miss is looked up once and never retried."""

    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.gets = 0

    def get(self, key: str):  # type: ignore[no-untyped-def]
        self.gets += 1
        return super().get(key)


def warm(tmp_path: Path, replies: list[str] | None = None) -> ResponseCache:
    """A cache holding the reply to ``MSGS``, put there by a normal (not cache-only) call."""
    cache = ResponseCache(tmp_path)
    CachedProvider(ScriptedProvider(replies or ["cached"]), cache).chat(MSGS, PARAMS)
    return cache


# -- the cache wrapper ----------------------------------------------------------------------------


def test_a_hit_is_served_and_the_inner_provider_is_never_called(tmp_path: Path) -> None:
    cache = warm(tmp_path)
    inner = ScriptedProvider(["fresh"])
    wrapper = CachedProvider(inner, cache, cache_only=True)
    resp = wrapper.chat(MSGS, PARAMS)
    assert resp.text == "cached" and resp.cached
    assert inner.calls == [] and (wrapper.hits, wrapper.misses) == (1, 0)


def test_a_miss_raises_a_final_error_and_calls_nothing(tmp_path: Path) -> None:
    inner = ScriptedProvider(["fresh"])
    cache = CountingCache(tmp_path)
    with pytest.raises(CacheMiss) as err:
        CachedProvider(inner, cache, cache_only=True).chat(MSGS, PARAMS)
    assert inner.calls == [] and cache.gets == 1
    assert isinstance(err.value, ProviderError) and not isinstance(err.value, TransientError)
    text = str(err.value)
    assert "--cache-only" in text and "no model was called" in text
    assert "the model (scripted:m)" in text and "for case" not in text
    assert list(tmp_path.glob("*/*.json")) == []  # nothing was written either


def test_a_miss_names_the_case_and_trial_and_the_role(tmp_path: Path) -> None:
    wrapper = CachedProvider(
        ScriptedProvider(["x"]), ResponseCache(tmp_path), cache_only=True, role="judge"
    )
    with case_scope("refund-7", 2), pytest.raises(CacheMiss) as err:
        wrapper.chat(MSGS, PARAMS)
    assert "for case 'refund-7' (trial 2)" in str(err.value)
    assert "from the judge (scripted:m)" in str(err.value)
    with case_scope("refund-8"), pytest.raises(CacheMiss, match=r"case 'refund-8' \(trial 0\)"):
        wrapper.chat(MSGS, PARAMS)


def test_without_the_flag_a_miss_still_calls_the_model(tmp_path: Path) -> None:
    inner = ScriptedProvider(["fresh"])
    resp = CachedProvider(inner, ResponseCache(tmp_path)).chat(MSGS, PARAMS)
    assert resp.text == "fresh" and len(inner.calls) == 1


# -- the system and the judge stop the run instead of recording the miss -----------------------


def test_a_system_miss_propagates_and_is_not_recorded_on_the_case(tmp_path: Path) -> None:
    inner = ScriptedProvider(["x"])
    provider = CachedProvider(inner, ResponseCache(tmp_path), cache_only=True, role="system")
    system = LLMSystem(TargetConfig(target="ollama:m"), provider)
    with pytest.raises(CacheMiss, match="no model was called"):
        system.run(Case(id="c", input="q"))
    turns = Case.model_validate({"id": "t", "input": "u0", "turns": ["u1"]})
    with pytest.raises(CacheMiss, match="no model was called"):
        system.run(turns)
    assert inner.calls == []


def test_a_judge_miss_propagates_and_is_not_retried(tmp_path: Path) -> None:
    inner = ScriptedProvider([VERDICT])
    cache = CountingCache(tmp_path)
    judge = LLMJudge(
        "j",
        CachedProvider(inner, cache, cache_only=True, role="judge"),
        PARAMS,
        max_retries=3,
    )
    with pytest.raises(CacheMiss, match="from the judge"):
        judge.score(Case(id="c", input="q", reference="a"), SystemOutput(output="a"))  # type: ignore[call-arg]
    assert inner.calls == [] and cache.gets == 1


def test_a_rubric_set_judge_miss_propagates(tmp_path: Path) -> None:
    inner = ScriptedProvider([VERDICT])
    judge = RubricSetJudge(
        "rs",
        CachedProvider(inner, ResponseCache(tmp_path), cache_only=True, role="judge"),
        PARAMS,
        rubrics=["Be brief."],
    )
    with pytest.raises(CacheMiss, match="from the judge"):
        judge.score(Case(id="c", input="q"), SystemOutput(output="a"))
    assert inner.calls == []


# -- run_suite ------------------------------------------------------------------------------------


def ollama_suite(tmp_path: Path, *, judge: bool = False, n: int = 3) -> SuiteConfig:
    rows = [{"id": f"c{i}", "input": f"question {i}", "reference": "a"} for i in range(n)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return SuiteConfig(
        name="s",
        dataset=str(tmp_path / "cases.jsonl"),
        system=TargetConfig(target="ollama:m"),
        judge=TargetConfig(target="ollama:judge-m") if judge else None,
        scorers=[ScorerConfig(name="j", type="llm_judge")]
        if judge
        else [ScorerConfig(name="has", type="contains", params={"any_of_field": "reference"})],
        cache_dir=str(tmp_path / "cache"),
        base_dir=str(tmp_path),
    )


def reply_for(model: str) -> dict[str, object]:
    text = VERDICT if model == "judge-m" else "a"
    return {
        "model": model,
        "message": {"content": text},
        "prompt_eval_count": 5,
        "eval_count": 5,
    }


def recording_client(seen: list[str], *, allow: bool = True) -> httpx.Client:
    def handler(req: httpx.Request) -> httpx.Response:
        seen.append(req.url.path)
        if not allow:
            raise AssertionError(f"--cache-only reached the server: {req.url}")
        if req.url.path == "/api/chat":
            return httpx.Response(200, json=reply_for(json.loads(req.content)["model"]))
        return httpx.Response(200, json={"version": "0", "models": []})

    return mock_client(handler)


def test_a_fully_cached_suite_reruns_from_the_cache_without_touching_the_server(
    tmp_path: Path,
) -> None:
    cfg = ollama_suite(tmp_path, judge=True)
    first: list[str] = []
    baseline = run_suite(cfg, client=recording_client(first))
    assert first.count("/api/chat") == 6  # three system calls, three judge calls
    again: list[str] = []
    run = run_suite(cfg, client=recording_client(again, allow=False), cache_only=True)
    assert again == []  # no chat call, and no version or digest lookup either
    assert [r.scores["j"].passed for r in run.results] == [True] * 3
    assert all(r.output.cached for r in run.results)
    assert run.info.suite_hash == baseline.info.suite_hash
    assert "ollama_version" not in run.info.environment


def test_a_cache_only_run_says_so_in_its_environment(tmp_path: Path) -> None:
    cfg = ollama_suite(tmp_path)
    baseline = run_suite(cfg, client=recording_client([]))
    assert "cache_only" not in baseline.info.environment  # an ordinary run has no such key
    run = run_suite(cfg, client=recording_client([], allow=False), cache_only=True)
    assert run.info.environment["cache_only"] == "true"
    assert load_run(save_run(run, tmp_path / "r.json")).info.environment["cache_only"] == "true"


def test_a_system_miss_stops_the_run_naming_the_case(tmp_path: Path) -> None:
    cfg = ollama_suite(tmp_path)
    run_suite(cfg, client=recording_client([]))
    with (tmp_path / "cases.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"id": "new-case", "input": "unseen", "reference": "a"}) + "\n")
    seen: list[str] = []
    with pytest.raises(CacheMiss, match=r"case 'new-case' \(trial 0\).*from the system"):
        run_suite(cfg, client=recording_client(seen, allow=False), cache_only=True)
    assert seen == []


def test_a_judge_miss_stops_the_run_and_the_model_is_not_called(tmp_path: Path) -> None:
    cfg = ollama_suite(tmp_path, judge=True)
    run_suite(cfg, client=recording_client([]))
    changed = cfg.model_copy(
        update={"scorers": [ScorerConfig(name="j", type="llm_judge", params={"rubric": "New."})]}
    )
    seen: list[str] = []  # the system replies are cached; the judge's new rubric is not
    with pytest.raises(CacheMiss, match=r"case 'c0' \(trial 0\).*from the judge"):
        run_suite(changed, client=recording_client(seen, allow=False), cache_only=True)
    assert seen == []


def test_a_miss_in_a_threaded_run_still_stops_it(tmp_path: Path) -> None:
    cfg = ollama_suite(tmp_path, n=4).model_copy(update={"concurrency": 3})
    with pytest.raises(CacheMiss, match="no model was called"):
        run_suite(cfg, client=recording_client([], allow=False), cache_only=True)


def test_cache_only_needs_the_cache(tmp_path: Path) -> None:
    cfg = ollama_suite(tmp_path)
    with pytest.raises(ConfigError, match="cannot be combined with use_cache=False"):
        run_suite(cfg, use_cache=False, cache_only=True)


@pytest.mark.parametrize("target", ["python:app:answer", "http://127.0.0.1:9/answer"])
def test_a_system_that_is_not_a_model_is_refused_because_its_code_would_run(
    tmp_path: Path, target: str
) -> None:
    cfg = ollama_suite(tmp_path).model_copy(update={"system": TargetConfig(target=target)})
    with pytest.raises(ConfigError, match="--cache-only needs a model system"):
        run_suite(cfg, cache_only=True)


# -- the command ----------------------------------------------------------------------------------


@pytest.fixture
def claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    return install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_ok(VERDICT)],
    )


def write_suite(tmp_path: Path, n: int = 3) -> Path:
    rows = [{"id": f"c{i}", "input": f"question {i}", "reference": "a"} for i in range(n)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\ncache_dir: cache\n"
        "system: {target: 'claude-code:haiku'}\njudge: {target: 'claude-code:haiku'}\n"
        "scorers:\n  - {name: judge, type: llm_judge}\n",
        "utf-8",
    )
    return suite


def test_run_cache_only_exits_zero_on_a_full_hit_and_runs_no_binary_call(
    tmp_path: Path, claude: FakeCLI
) -> None:
    suite = write_suite(tmp_path)
    assert main(["run", str(suite), "--out", str(tmp_path / "a.json"), "--quiet"]) == EXIT_PASS
    calls = len(claude.calls())
    assert calls == 6
    again = tmp_path / "b.json"
    assert main(["run", str(suite), "--cache-only", "--out", str(again), "--quiet"]) == EXIT_PASS
    assert len(claude.calls()) == calls  # only the version lookups, which are not model calls
    assert again.exists()
    environment = lambda p: json.loads(p.read_text("utf-8"))["info"]["environment"]  # noqa: E731
    assert environment(again)["cache_only"] == "true"
    assert "cache_only" not in environment(tmp_path / "a.json")


def test_run_cache_only_exits_two_on_a_miss_and_writes_nothing(
    tmp_path: Path, claude: FakeCLI, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = write_suite(tmp_path)
    assert main(["run", str(suite), "--out", str(tmp_path / "a.json"), "--quiet"]) == EXIT_PASS
    calls = len(claude.calls())
    write_suite(tmp_path, n=4)  # one case the cache has never seen
    out = tmp_path / "b.json"
    code = main(["run", str(suite), "--cache-only", "--out", str(out), "--quiet"])
    assert code == EXIT_USAGE == 2
    err = capsys.readouterr().err
    assert "evalproof: error: --cache-only: no cached reply for case 'c3' (trial 0)" in err
    assert "no model was called" in err
    assert len(claude.calls()) == calls and not out.exists()


def test_cache_only_and_no_cache_exclude_each_other(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = write_suite(tmp_path)
    for command in (["run", str(suite)], ["pairwise", "a.json", "b.json", "--suite", str(suite)]):
        with pytest.raises(SystemExit) as exit_info:
            main([*command, "--cache-only", "--no-cache"])
        assert exit_info.value.code == 2
        assert "not allowed with argument" in capsys.readouterr().err


def test_a_python_system_is_refused_on_the_command_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = write_suite(tmp_path)
    code = main(["run", str(suite), "--system", "python:app:answer", "--cache-only", "--quiet"])
    assert code == EXIT_USAGE
    assert "--cache-only needs a model system" in capsys.readouterr().err


def test_pairwise_cache_only_serves_hits_and_stops_on_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = install(
        tmp_path,
        monkeypatch,
        "claude",
        version="2.1.322 (Claude Code)",
        responses=[claude_ok('{"winner": "A"}')],
    )
    suite = write_suite(tmp_path)
    suite.write_text(
        suite.read_text("utf-8").replace(
            "{name: judge, type: llm_judge}",
            "{name: has, type: contains, params: {any_of_field: reference, case_sensitive: false}}",
        ),
        "utf-8",
    )
    # The cases' answers are the fake's reply; the pairwise judge reads two of them per case.
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert main(["run", str(suite), "--out", str(a), "--quiet"]) == EXIT_PASS
    assert main(["run", str(suite), "--out", str(b), "--quiet"]) == EXIT_PASS
    argv = ["pairwise", str(a), str(b), "--suite", str(suite), "--limit", "2"]
    calls = len(fake.calls())
    # Nothing has been judged pairwise yet, so the first judge call misses.
    assert main([*argv, "--cache-only"]) == EXIT_USAGE
    err = capsys.readouterr().err
    assert "--cache-only: no cached reply for case" in err and "from the judge" in err
    assert len(fake.calls()) == calls
    # Fill the cache with a normal pairwise run; then the same command answers from it alone.
    assert main(argv) == EXIT_PASS
    filled = len(fake.calls())
    assert filled > calls
    assert main([*argv, "--cache-only"]) == EXIT_PASS
    assert len(fake.calls()) == filled


def test_a_cache_only_run_without_the_binary_still_records_the_cli_facts(
    tmp_path: Path, claude: FakeCLI, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole point of the mode: replay a run on a machine that lacks the CLI."""
    suite = write_suite(tmp_path)
    cfg = load_config(suite)
    run_suite(cfg)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    cli_version.cache_clear()
    run = run_suite(cfg, cache_only=True)
    env = run.info.environment
    assert all(r.scores["judge"].passed for r in run.results)
    for role in ("system", "judge"):
        assert env[f"{role}_cli"] == "claude-code"
        assert env[f"{role}_cli_version"] == "unavailable"
        assert env[f"{role}_isolation"] == "claude-code/1"
        assert env[f"{role}_sampling"] == "cli-default"
