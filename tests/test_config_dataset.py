from __future__ import annotations

from pathlib import Path

import pytest

from evalproof.config import ConfigError, Pricing, SuiteConfig, TargetConfig, load_config
from evalproof.dataset import DatasetError, dataset_hash, load_dataset, stable_id

SUITE = """
name: demo
dataset: cases.jsonl
system:
  target: ollama:qwen3:1.7b
  system_prompt_file: prompt.txt
  params: {temperature: 0, seed: 3}
judge:
  target: ollama:qwen3:8b
scorers:
  - {name: key, type: regex, params: {any_of_field: accept}}
  - {name: judge, type: llm_judge}
calibration: {judge_scorer: judge, gold_scorer: key, gold_sample: 10}
gate: {metric: judge, max_drop: 0.05}
"""


def test_load_config_resolves_files(tmp_path: Path) -> None:
    (tmp_path / "prompt.txt").write_text("You are helpful.", "utf-8")
    (tmp_path / "suite.yaml").write_text(SUITE, "utf-8")
    cfg = load_config(tmp_path / "suite.yaml")
    assert cfg.system.system_prompt == "You are helpful."
    assert cfg.system.params.seed == 3 and cfg.system.kind == "ollama"
    assert cfg.resolve("cases.jsonl") == tmp_path / "cases.jsonl"
    # a rooted path without a drive lands on the suite's drive on Windows
    assert cfg.resolve("/abs/x") == Path(tmp_path.anchor, "abs", "x")
    assert cfg.gate is not None and cfg.gate.alpha == 0.05


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ("scorers:\n  - {name: a, type: regex}\n  - {name: a, type: regex}\n", "unique"),
        ("scorers:\n  - {name: j, type: llm_judge}\n", "needs a 'judge'"),
        ("scorers:\n  - {name: a, type: regex}\ngate: {metric: zzz}\n", "not a scorer"),
        (
            "scorers:\n  - {name: a, type: regex}\ncalibration: {judge_scorer: a}\n",
            "exactly one",
        ),
        (
            "scorers:\n  - {name: a, type: regex}\ncalibration: {judge_scorer: b, labels: l}\n",
            "not a scorer",
        ),
        ("scorers:\n  - {name: a, type: regex}\nconcurrency: 0\n", "concurrency"),
    ],
)
def test_config_validation(tmp_path: Path, patch: str, message: str) -> None:
    text = "name: x\ndataset: d.jsonl\nsystem: {target: 'ollama:m'}\n" + patch
    (tmp_path / "s.yaml").write_text(text, "utf-8")
    with pytest.raises(ConfigError, match=message):
        load_config(tmp_path / "s.yaml")


def test_config_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "missing.yaml")
    (tmp_path / "bad.yaml").write_text("a: [unclosed", "utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(tmp_path / "bad.yaml")
    (tmp_path / "list.yaml").write_text("- 1\n", "utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        load_config(tmp_path / "list.yaml")


def test_target_kinds_and_pricing() -> None:
    assert TargetConfig(target="https://x").kind == "http"
    assert TargetConfig(target="python:a:b").kind == "python"
    assert Pricing(input_per_mtok=3, output_per_mtok=15).cost(1_000, 100) == pytest.approx(0.0045)
    assert SuiteConfig.model_fields["cache_dir"].default == ".evalproof/cache"


def test_load_dataset(write_jsonl) -> None:
    path = write_jsonl(
        "d.jsonl",
        [
            {"id": "a", "input": "q1", "reference": "r", "accept": ["x"]},
            {"input": {"question": "q2"}},
        ],
    )
    cases = load_dataset(path)
    assert cases[0].field("accept") == ["x"] and cases[0].field("missing") is None
    assert cases[1].id == stable_id({"input": {"question": "q2"}})
    assert cases[1].render_vars()["question"] == "q2"
    assert cases[0].render_vars()["input"] == "q1"
    assert dataset_hash(cases) == dataset_hash(list(reversed(cases)))


def test_dataset_errors(write_jsonl, tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="not found"):
        load_dataset(tmp_path / "nope.jsonl")
    with pytest.raises(DatasetError, match="duplicate"):
        load_dataset(
            write_jsonl("dup.jsonl", [{"id": "a", "input": "1"}, {"id": "a", "input": "2"}])
        )
    with pytest.raises(DatasetError, match=r":1: input"):
        load_dataset(write_jsonl("badtype.jsonl", [{"id": "a", "input": 1}]))
    with pytest.raises(DatasetError, match="needs an object"):
        load_dataset(write_jsonl("noinput.jsonl", [{"id": "a"}]))
    (tmp_path / "bad.jsonl").write_text("{oops\n", "utf-8")
    with pytest.raises(DatasetError, match="invalid JSON"):
        load_dataset(tmp_path / "bad.jsonl")
    (tmp_path / "empty.jsonl").write_text("# only a comment\n\n", "utf-8")
    with pytest.raises(DatasetError, match="empty"):
        load_dataset(tmp_path / "empty.jsonl")
