"""Cost a CLI reports (Claude Code's ``total_cost_usd``) counts when no price is set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof.cache import CachedProvider, ResponseCache
from evalproof.compare import run_stats
from evalproof.config import GenerationParams, Pricing, load_config
from evalproof.providers import ChatResponse, ClaudeCodeProvider, CodexProvider, Message
from evalproof.providers import response_cost as response_cost_from_package
from evalproof.providers.base import response_cost
from evalproof.runner import run_suite
from tests.fakecli import FakeCLI, claude_array, claude_ok, claude_result, codex_ok, install, reply

USER = [Message(role="user", content="hi")]
PARAMS = GenerationParams()
VERDICT = '{"pass": true, "score": 1.0, "reason": "fine"}'


def _no_sleep(_: float) -> None:
    return None


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)


def _claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **kw: object) -> FakeCLI:
    out = claude_ok(VERDICT, **kw)  # type: ignore[arg-type]
    return install(
        tmp_path, monkeypatch, "claude", version="2.1.322 (Claude Code)", responses=[out]
    )


def test_the_claude_provider_reads_total_cost_usd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude(tmp_path, monkeypatch)
    resp = ClaudeCodeProvider("haiku", sleep=_no_sleep).chat(USER, PARAMS)
    assert resp.cost_usd == pytest.approx(0.0123)


@pytest.mark.parametrize("cost", [None, -1.0])
def test_a_missing_or_nonsense_cost_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cost: float | None
) -> None:
    _claude(tmp_path, monkeypatch, cost_usd=cost)
    assert ClaudeCodeProvider("haiku", sleep=_no_sleep).chat(USER, PARAMS).cost_usd is None


def test_a_cost_that_is_not_a_number_is_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = claude_result("OK")
    result["total_cost_usd"] = "free"
    out = reply(claude_array(result))
    install(tmp_path, monkeypatch, "claude", version="2.1.322 (Claude Code)", responses=[out])
    assert ClaudeCodeProvider("haiku", sleep=_no_sleep).chat(USER, PARAMS).cost_usd is None


def test_codex_reports_no_cost(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    install(tmp_path, monkeypatch, "codex", version="codex-cli 0.167.2", responses=[codex_ok()])
    assert CodexProvider("gpt-6-luna", sleep=_no_sleep).chat(USER, PARAMS).cost_usd is None


def test_response_cost_prefers_a_configured_price_over_the_reported_cost() -> None:
    resp = ChatResponse(text="x", model="m", tokens_in=1_000_000, tokens_out=0, cost_usd=0.5)
    assert response_cost(resp, Pricing()) == 0.5  # no price set: the CLI's figure
    assert response_cost(resp, Pricing(input_per_mtok=3.0)) == 3.0  # a price wins
    assert response_cost(resp, Pricing(output_per_mtok=3.0)) == 0.0  # any non-zero price wins
    unreported = resp.model_copy(update={"cost_usd": None})
    assert response_cost(unreported, Pricing()) == 0.0
    assert response_cost_from_package is response_cost


def _suite(tmp_path: Path, pricing: str = "") -> Path:
    cases = [{"id": f"c{i}", "input": f"q{i}"} for i in range(3)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\n"
        f"system: {{target: 'claude-code:haiku'{pricing}}}\n"
        f"judge: {{target: 'claude-code:haiku'{pricing}}}\n"
        "scorers: [{name: j, type: llm_judge}]\n",
        "utf-8",
    )
    return suite


def test_a_run_adds_up_the_cli_reported_cost_for_system_and_judge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude(tmp_path, monkeypatch)
    run = run_suite(load_config(_suite(tmp_path)), use_cache=False)
    assert all(r.output.cost_usd == pytest.approx(0.0123) for r in run.results)
    assert all(r.scores["j"].judge_cost_usd == pytest.approx(0.0123) for r in run.results)
    assert run_stats(run).cost_usd == pytest.approx(3 * 2 * 0.0123)


def test_a_configured_price_replaces_the_reported_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude(tmp_path, monkeypatch)
    pricing = ", pricing: {input_per_mtok: 1.0, output_per_mtok: 1.0}"
    run = run_suite(load_config(_suite(tmp_path, pricing)), use_cache=False)
    tokens = 2 + 692 + 10 + 4  # what the fake CLI reports
    expected = tokens / 1_000_000
    assert run.results[0].output.cost_usd == pytest.approx(expected)
    assert run.results[0].scores["j"].judge_cost_usd == pytest.approx(expected)


def test_a_cached_reply_keeps_the_cost_of_the_original_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cli = _claude(tmp_path, monkeypatch)
    cached = CachedProvider(
        ClaudeCodeProvider("haiku", sleep=_no_sleep), ResponseCache(tmp_path / "cache")
    )
    first, again = cached.chat(USER, PARAMS), cached.chat(USER, PARAMS)
    assert not first.cached and again.cached and len(cli.calls()) == 1
    assert again.cost_usd == first.cost_usd == pytest.approx(0.0123)


def test_a_second_run_from_the_cache_reports_the_same_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _claude(tmp_path, monkeypatch)
    cfg = load_config(_suite(tmp_path))
    first = run_suite(cfg)
    second = run_suite(cfg)
    assert all(r.output.cached for r in second.results)
    assert run_stats(second).cost_usd == pytest.approx(run_stats(first).cost_usd)
