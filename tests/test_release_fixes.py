"""Regression tests for the issues found in the 0.3.0 pre-release review."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.cache import cache_key
from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig, GenerationParams
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, EXIT_USAGE, decide
from evalproof.importers import ImportFormatError
from evalproof.providers.anthropic import AnthropicProvider
from evalproof.providers.base import Message
from evalproof.providers.openai import OpenAICompatibleProvider
from evalproof.providers.profiles import PROFILES

FX = Path(__file__).parent / "fixtures"
INSPECT = FX / "inspect" / "log_formats.json"


def _row(idx: int, q: str, ok: bool, *, prompt: str = "tmpl", **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "testIdx": idx,
        "vars": {"q": q},
        "success": ok,
        "score": 1.0 if ok else 0.0,
        "provider": {"id": "prov", "label": ""},
        "prompt": {"raw": f"rendered {q}", "label": prompt},
        "response": {"output": "out"},
        "testCase": {"vars": {"q": q}},
        "gradingResult": {"pass": ok, "componentResults": []},
        "failureReason": 0 if ok else 1,
    }
    row.update(extra)
    return row


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return path


# --- promptfoo: prompts that share a long prefix stay two systems ---------------------------


def test_promptfoo_long_prompts_with_a_shared_prefix_stay_apart(tmp_path: Path) -> None:
    stem = "You are a careful support agent. Answer from the policy text only and " * 2
    rows = [
        _row(0, "a", True, prompt=stem + "be brief.", promptIdx=0),
        _row(0, "a", False, prompt=stem + "be thorough.", promptIdx=1),
    ]
    path = _jsonl(tmp_path / "p.jsonl", rows)
    with pytest.raises(ImportFormatError, match="2 systems") as err:
        import_run("promptfoo", path)
    names = [s for s in str(err.value).split("'") if s.startswith("prov · ")]
    assert len(names) == 2 and names[0] != names[1]
    run = import_run("promptfoo", path, system=names[0])
    assert len(run.results) == 1  # one case, not a blend of both prompts


def test_promptfoo_two_prompts_with_one_label_is_an_error(tmp_path: Path) -> None:
    rows = [_row(0, "a", True, promptIdx=0), _row(0, "a", False, promptIdx=1)]
    with pytest.raises(ImportFormatError, match="share the label"):
        import_run("promptfoo", _jsonl(tmp_path / "same.jsonl", rows))


def test_promptfoo_identical_tests_warn(tmp_path: Path) -> None:
    rows = [_row(0, "same", True), _row(1, "same", False)]
    warnings: list[str] = []
    import_run("promptfoo", _jsonl(tmp_path / "d.jsonl", rows), warn=warnings.append)
    assert any("repeatIndex" in w for w in warnings)


def test_promptfoo_null_test_index_does_not_crash(tmp_path: Path) -> None:
    rows = [_row(0, "a", True), _row(1, "b", True)]
    rows[0]["testIdx"] = None
    assert len(import_run("promptfoo", _jsonl(tmp_path / "n.jsonl", rows)).results) == 2


# --- repeats: a case with a good repeat is not an error under any/mean ----------------------


@pytest.mark.parametrize(("epochs", "is_error"), [("any", False), ("mean", False), ("all", True)])
def test_partial_repeat_errors(tmp_path: Path, epochs: str, is_error: bool) -> None:
    rows = [
        _row(0, "q", True, repeatGroupId="g", repeatIndex=0),
        _row(1, "q", False, repeatGroupId="g", repeatIndex=1, error="timeout", failureReason=2),
    ]
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs=epochs)
    assert (run.results[0].output.error is not None) is is_error


# --- Inspect: per-run message ids do not change the dataset hash ----------------------------


def test_inspect_message_ids_do_not_change_the_dataset_hash(tmp_path: Path) -> None:
    doc = json.loads(INSPECT.read_text("utf-8"))
    paths = []
    for tag in ("a", "b"):
        for i, sample in enumerate(doc["samples"]):
            sample["input"] = [{"id": f"{tag}{i}", "role": "user", "content": f"q{i}"}]
        path = tmp_path / f"{tag}.json"
        path.write_text(json.dumps(doc), "utf-8")
        paths.append(path)
    a, b = (import_run("inspect", p) for p in paths)
    assert a.info.dataset_hash == b.info.dataset_hash


# --- gate: an empty metric name never passes -------------------------------------------------


def test_unvalidated_gate_with_no_metric_cannot_gate(tmp_path: Path) -> None:
    run = import_run("promptfoo", _jsonl(tmp_path / "x.jsonl", [_row(0, "a", True)]))
    cmp = compare_runs(run, run)
    decision = decide(cmp, GateConfig.model_construct(metric=None, metrics=[], min_cases=1))
    assert decision.exit_code == EXIT_CANNOT_GATE


def test_cli_empty_metric_is_a_usage_error(tmp_path: Path) -> None:
    rows = [_row(i, f"q{i}", True) for i in range(5)]
    src = _jsonl(tmp_path / "x.jsonl", rows)
    out = tmp_path / "run.json"
    assert main(["import", "promptfoo", str(src), "--out", str(out)]) == EXIT_PASS
    (tmp_path / "cases.jsonl").write_text('{"id": "c", "input": "i"}\n', "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:m:f'}\n"
        "scorers: [{name: pass, type: exact_match}]\n"
        "gate: {metric: pass, min_cases: 1}\n",
        "utf-8",
    )
    base = ["compare", str(out), str(out), "--suite", str(suite), "--gate"]
    assert main([*base, "--metric", ""]) == EXIT_USAGE
    assert main(base) == EXIT_PASS


# --- cache: a hosted request that now omits temperature gets a new key ---------------------


def _key(provider: Any, params: GenerationParams) -> str:
    msgs = [Message(role="user", content="hi")]
    return cache_key(provider.name, provider.model, None, msgs, params, provider.cache_salt(params))


def test_cache_salt_only_where_the_request_changed() -> None:
    msgs = [Message(role="user", content="hi")]
    default, explicit = GenerationParams(), GenerationParams(temperature=0.0, seed=0)
    hosted = OpenAICompatibleProvider(PROFILES["openai"], "m", api_key="k")
    local = OpenAICompatibleProvider(PROFILES["vllm"], "m")
    claude = AnthropicProvider("m", api_key="k")
    for p in (hosted, claude):
        old = cache_key(p.name, p.model, None, msgs, default)
        assert _key(p, default) != old  # pre-0.3.0 answers came from temperature 0
        assert _key(p, explicit) == cache_key(p.name, p.model, None, msgs, explicit)
    assert local.cache_salt(default) is None
    assert _key(local, default) == cache_key(local.name, local.model, None, msgs, default)
