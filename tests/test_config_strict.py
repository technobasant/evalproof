"""Suite files reject unknown keys, and say which key and what was probably meant."""

from __future__ import annotations

from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.config import ConfigError, GenerationParams, load_config
from evalproof.scaffold import (
    ANTHROPIC_DEFAULT,
    CODEX_PLACEHOLDER,
    OLLAMA_DEFAULT,
    Judge,
    render_suite,
)
from tests.conftest import make_run

ROOT = Path(__file__).resolve().parents[1]
BASE = (
    "name: s\ndataset: cases.jsonl\n"
    "system: {{target: 'python:a:b'{system}}}\n"
    "scorers: [{{name: m, type: contains{scorer}}}]\n"
    "{extra}"
)


def _load(tmp_path: Path, *, system: str = "", scorer: str = "", extra: str = "") -> str:
    """Load a suite built from the pieces; return the ConfigError text."""
    suite = tmp_path / "suite.yaml"
    suite.write_text(BASE.format(system=system, scorer=scorer, extra=extra), "utf-8")
    with pytest.raises(ConfigError) as err:
        load_config(suite)
    return str(err.value)


@pytest.mark.parametrize("suite", sorted((ROOT / "examples").rglob("suite*.yaml")), ids=str)
def test_every_example_suite_loads(suite: Path) -> None:
    assert load_config(suite).scorers


@pytest.mark.parametrize(
    "judge",
    [
        Judge("claude-code:haiku", "x"),
        Judge("codex:gpt-5", "x"),
        Judge(CODEX_PLACEHOLDER, "x"),
        Judge(OLLAMA_DEFAULT, "x"),
        Judge(ANTHROPIC_DEFAULT, "x", api_key_env="ANTHROPIC_API_KEY"),
    ],
    ids=lambda j: j.target,
)
def test_the_suite_template_loads_for_every_judge(tmp_path: Path, judge: Judge) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(render_suite(judge), "utf-8")
    cfg = load_config(suite)
    assert cfg.gate is not None


def test_an_unknown_system_key_suggests_the_right_one(tmp_path: Path) -> None:
    msg = _load(tmp_path, system=", system_promt: hi")
    where = f"{tmp_path / 'suite.yaml'}: system.system_promt"
    assert msg == f"{where}: unknown key (did you mean 'system_prompt'?)"


@pytest.mark.parametrize(
    ("kwargs", "where", "suggestion"),
    [
        ({"system": ", params: {temprature: 1}"}, "system.params.temprature", "temperature"),
        (
            {"system": ", pricing: {input_per_mtoks: 1}"},
            "system.pricing.input_per_mtoks",
            "input_per_mtok",
        ),
        ({"system": ", timeout: 5"}, "system.timeout", "timeout_s"),
        (
            {"system": ", system_prompt_fle: p.txt"},
            "system.system_prompt_fle",
            "system_prompt_file",
        ),
        ({"scorer": ", parms: {}"}, "scorers.0.parms", "params"),
        ({"extra": "gates: {metric: m}\n"}, "gates", "gate"),
        ({"extra": "gate: {metric: m, min_pass_rat: 0.5}\n"}, "gate.min_pass_rat", "min_pass_rate"),
        (
            {"extra": "gate: {metric: m, max_new_failure: 0}\n"},
            "gate.max_new_failure",
            "max_new_failures",
        ),
        (
            {"extra": "judge: {target: 'python:a:b', systm_prompt: x}\n"},
            "judge.systm_prompt",
            "system_prompt",
        ),
    ],
)
def test_every_model_forbids_unknown_keys(
    tmp_path: Path, kwargs: dict[str, str], where: str, suggestion: str
) -> None:
    msg = _load(tmp_path, **kwargs)
    assert f": {where}: unknown key (did you mean '{suggestion}'?)" in msg


def test_a_calibration_typo_is_caught(tmp_path: Path) -> None:
    msg = _load(tmp_path, extra="calibration: {judge_scorer: m, labelz: l.jsonl}\n")
    assert "calibration.labelz: unknown key" in msg


def test_a_key_with_no_close_match_lists_the_valid_ones(tmp_path: Path) -> None:
    msg = _load(tmp_path, extra="gate: {metric: m, zzz: 1}\n")
    assert "gate.zzz: unknown key (valid keys: metric, metrics, max_drop" in msg


def test_every_problem_is_listed_and_no_url_leaks(tmp_path: Path) -> None:
    msg = _load(tmp_path, system=", systm_prompt: a, timeout: 5", scorer=", parms: {}")
    assert len(msg.splitlines()) == 3
    assert "https://" not in msg and "pydantic" not in msg
    assert all(line.startswith(str(tmp_path / "suite.yaml")) for line in msg.splitlines())


def test_a_type_error_reads_as_path_where_what(tmp_path: Path) -> None:
    msg = _load(tmp_path, extra="seed: lots\n")
    assert msg.startswith(f"{tmp_path / 'suite.yaml'}: seed: Input should be a valid integer")
    assert "https://" not in msg


def test_a_rule_the_suite_breaks_has_no_where(tmp_path: Path) -> None:
    msg = _load(tmp_path, extra="gate: {metric: nope}\n")
    assert msg == f"{tmp_path / 'suite.yaml'}: gate.metric 'nope' is not a scorer name"


def test_generation_params_extra_is_still_the_escape_hatch(tmp_path: Path) -> None:
    assert GenerationParams(extra={"top_k": 5}).extra == {"top_k": 5}
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: c.jsonl\nsystem: {target: 'python:a:b', params: {extra: {top_k: 5}}}\n"
        "scorers: [{name: m, type: contains}]\n",
        "utf-8",
    )
    assert load_config(suite).system.params.extra == {"top_k": 5}


def test_a_bad_gate_override_on_the_command_line_is_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run = make_run({f"c{i}": {"m": True} for i in range(12)})
    path = tmp_path / "r.json"
    path.write_text(run.model_dump_json(), "utf-8")
    code = main(["compare", str(path), str(path), "--metric", "m", "--min-pass-rate", "90"])
    assert code == 2
    err = capsys.readouterr().err
    assert "between 0 and 1" in err and "https://" not in err
