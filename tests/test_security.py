"""SECURITY.md promises that API keys never reach run files, caches or reports."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evalproof.config import load_config
from evalproof.report import run_markdown
from evalproof.runner import run_suite, save_run
from tests.conftest import mock_client

SECRET = "sk-test-0123456789-do-not-leak"


def _reply(request: httpx.Request) -> httpx.Response:
    question = json.loads(request.content)["messages"][-1]["content"]
    if request.url.path.endswith("/chat/completions"):
        if request.url.host.endswith("azure.com"):  # Azure takes the raw key in `api-key`
            assert request.headers["api-key"] == SECRET
            assert "authorization" not in request.headers
        else:
            assert request.headers["authorization"] == f"Bearer {SECRET}"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": question.upper()}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 3},
            },
        )
    assert request.headers["x-api-key"] == SECRET
    return httpx.Response(
        200,
        json={
            "content": [{"type": "text", "text": question.upper()}],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        },
    )


@pytest.mark.parametrize(
    ("target", "env", "base_url"),
    [
        ("openai:gpt-test", "OPENAI_API_KEY", None),
        ("anthropic:claude-test", "ANTHROPIC_API_KEY", None),
        ("groq:llama-3.3-70b-versatile", "GROQ_API_KEY", None),
        ("openrouter:anthropic/claude-sonnet-4.5", "OPENROUTER_API_KEY", None),
        ("azure:my-deployment", "AZURE_OPENAI_API_KEY", "https://res.openai.azure.com/openai/v1"),
        ("openai-compatible:m", "TEAM_GATEWAY_KEY", "https://gateway.example.com/v1"),
    ],
)
def test_api_keys_are_not_written_anywhere(
    target: str, env: str, base_url: str | None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(env, SECRET)
    (tmp_path / "cases.jsonl").write_text(
        "\n".join(
            json.dumps({"id": f"c{i}", "input": {"question": f"q{i}"}, "reference": f"Q{i}"})
            for i in range(3)
        )
        + "\n",
        "utf-8",
    )
    system = [f"  target: {target}", "  prompt_template: '{question}'"]
    if base_url:
        system.append(f"  base_url: {base_url}")
    if target.startswith("openai-compatible"):
        system.append(f"  api_key_env: {env}")  # the suite names the variable, never the value
    (tmp_path / "suite.yaml").write_text(
        "name: leak\ndataset: cases.jsonl\nsystem:\n"
        + "\n".join(system)
        + "\nscorers:\n  - {name: exact, type: exact_match}\ncache_dir: cache\n",
        "utf-8",
    )
    run = run_suite(load_config(tmp_path / "suite.yaml"), client=mock_client(_reply))
    assert [r.scores["exact"].passed for r in run.results] == [True, True, True]

    save_run(run, tmp_path / "run.json")
    (tmp_path / "report.md").write_text(run_markdown(run), "utf-8")
    written = [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix != ".yaml"]
    assert any(p.parent.parent.name == "cache" for p in written)  # the cache was used
    leaked = [p for p in written if SECRET in p.read_text("utf-8")]
    assert leaked == []
