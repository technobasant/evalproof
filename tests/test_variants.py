"""Variants: one suite run several ways (a flag, a model, a prompt) over the same cases."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import (
    ConfigError,
    SuiteConfig,
    TargetConfig,
    load_config,
    merge_system,
    suite_fingerprint,
)
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_USAGE
from evalproof.models import Case
from evalproof.report import run_markdown
from evalproof.runner import load_run, run_suite, save_run
from evalproof.systems import LLMSystem, check_variant_vars
from tests.conftest import ScriptedProvider

# --- fingerprints -----------------------------------------------------------------------------

# Two suites without variants and the fingerprints evalproof computed for them before variants
# existed (on main, at 62d60ed). A run file committed back then must keep verifying, so these
# values may only change on purpose, with a note in the changelog.
SUITE_A = {
    "name": "s",
    "dataset": "d.jsonl",
    "system": {
        "target": "ollama:qwen3:8b",
        "system_prompt": "be brief",
        "prompt_template": "Q: {question}",
        "params": {"temperature": 0.2, "seed": 3, "max_tokens": 64},
        "base_url": "http://127.0.0.1:11434",
    },
    "judge": {"target": "anthropic:claude-x", "system_prompt": "judge"},
    "scorers": [
        {"name": "exact", "type": "exact_match"},
        {"name": "j", "type": "llm_judge", "params": {"rubric": "ok"}},
    ],
    "seed": 5,
}
SUITE_B = {
    "name": "s2",
    "dataset": "d.jsonl",
    "system": {"target": "python:app:answer"},
    "scorers": [{"name": "contains", "type": "contains", "params": {"field": "accept"}}],
}
FINGERPRINT_A = "f7b78c98ede66cd1"
FINGERPRINT_B = "b1389da836c93768"


def test_a_suite_without_variants_keeps_its_fingerprint() -> None:
    assert suite_fingerprint(SuiteConfig.model_validate(SUITE_A)) == FINGERPRINT_A
    assert suite_fingerprint(SuiteConfig.model_validate(SUITE_B)) == FINGERPRINT_B


def test_declaring_variants_does_not_change_the_plain_fingerprint() -> None:
    plain = SuiteConfig.model_validate(SUITE_A)
    with_variants = SuiteConfig.model_validate(
        {**SUITE_A, "variants": [{"name": "warm", "vars": {"tone": "warm"}}]}
    )
    assert suite_fingerprint(with_variants) == suite_fingerprint(plain) == FINGERPRINT_A


def test_each_variant_has_its_own_fingerprint() -> None:
    cfg = SuiteConfig.model_validate(
        {
            **SUITE_A,
            "variants": [
                {"name": "plain"},  # no override at all: still a different run from the base
                {"name": "warm", "vars": {"tone": "warm"}},
                {"name": "cool", "vars": {"tone": "cool"}},
                {"name": "cool2", "vars": {"tone": "cool"}},
                {"name": "hot", "system": {"params": {"temperature": 0.9}}},
            ],
        }
    )
    prints = {name: suite_fingerprint(cfg, variant=name) for name in (v.name for v in cfg.variants)}
    assert len(set(prints.values()) | {FINGERPRINT_A}) == 6
    assert suite_fingerprint(cfg, variant="warm") == prints["warm"]  # deterministic
    assert suite_fingerprint(cfg, cfg.variant_system("warm"), "warm") == prints["warm"]


# --- merging a variant's system over the suite's ----------------------------------------------


def test_params_and_pricing_merge_key_by_key_and_the_rest_is_replaced() -> None:
    base = TargetConfig.model_validate(
        {
            "target": "ollama:m",
            "system_prompt": "base prompt",
            "params": {"temperature": 0.2, "max_tokens": 64},
            "pricing": {"input_per_mtok": 1.0, "output_per_mtok": 2.0},
            "headers": {"x-a": "1", "x-b": "2"},
        }
    )
    merged = merge_system(
        base,
        {
            "target": "ollama:other",
            "params": {"temperature": 0.9},
            "pricing": {"output_per_mtok": 3.0},
            "headers": {"x-c": "3"},
        },
    )
    assert merged.target == "ollama:other"
    assert merged.system_prompt == "base prompt"  # not named, so kept
    assert (merged.params.temperature, merged.params.max_tokens) == (0.9, 64)
    assert merged.pricing is not None
    assert (merged.pricing.input_per_mtok, merged.pricing.output_per_mtok) == (1.0, 3.0)
    assert merged.headers == {"x-c": "3"}  # replaced, not merged
    assert base.params.temperature == 0.2  # the suite's own system is left alone


def test_variant_system_is_the_suite_system_when_there_is_no_override() -> None:
    cfg = SuiteConfig.model_validate(
        {**SUITE_B, "variants": [{"name": "a", "vars": {"k": 1}}, {"name": "b"}]}
    )
    assert cfg.variant_system(None) is cfg.system
    assert cfg.variant_system("a") is cfg.system
    assert cfg.variant("a").vars == {"k": 1}


# --- validation -------------------------------------------------------------------------------


def _suite_file(tmp_path: Path, variants: str, extra: str = "") -> Path:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'ollama:m', system_prompt: base}\n"
        "scorers: [{name: c, type: contains, params: {any_of: [x]}}]\n" + extra + variants,
        "utf-8",
    )
    return suite


@pytest.mark.parametrize(
    ("variants", "message"),
    [
        ("variants: [{name: a}, {name: a}]\n", r"variant names must be unique, got \['a', 'a'\]"),
        ("variants: [{name: 'Has Space'}]\n", "variant name 'Has Space' must use only"),
        ("variants: [{name: ''}]\n", "must use only"),
        ("variants: [{name: a, colour: red}]\n", r"variants\.0\.colour: unknown key"),
        ("variants: [{name: a, system: {taget: 'ollama:x'}}]\n", "variant 'a'.*taget"),
        ("variants: [{name: a, system: {params: {temperature: hot}}}]\n", "variant 'a'"),
    ],
)
def test_bad_variants_are_refused_at_load(tmp_path: Path, variants: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        load_config(_suite_file(tmp_path, variants))


def test_vars_must_be_plain_json(tmp_path: Path) -> None:
    cfg = SuiteConfig.model_validate({**SUITE_B, "variants": [{"name": "a", "vars": {"n": [1]}}]})
    assert cfg.variant("a").vars == {"n": [1]}
    with pytest.raises(ValueError, match="plain JSON"):
        SuiteConfig.model_validate({**SUITE_B, "variants": [{"name": "a", "vars": {"x": {1, 2}}}]})


def test_an_unknown_variant_lists_the_real_ones() -> None:
    cfg = SuiteConfig.model_validate({**SUITE_B, "variants": [{"name": "a"}, {"name": "b"}]})
    with pytest.raises(ConfigError, match=r"unknown variant 'c'; the suite's variants: a, b"):
        cfg.variant("c")
    with pytest.raises(ConfigError, match="none defined"):
        SuiteConfig.model_validate(SUITE_B).variant("a")


def test_a_variant_system_takes_prompt_text_from_a_file(tmp_path: Path) -> None:
    (tmp_path / "formal.txt").write_text("Be formal.", "utf-8")
    suite = _suite_file(
        tmp_path, "variants: [{name: formal, system: {system_prompt_file: formal.txt}}]\n"
    )
    cfg = load_config(suite)
    assert cfg.variant_system("formal").system_prompt == "Be formal."
    assert cfg.system.system_prompt == "base"


# --- model targets: vars are template fields --------------------------------------------------


def test_vars_fill_the_prompt_template_of_a_model_target() -> None:
    provider = ScriptedProvider(["ok"])
    cfg = TargetConfig(target="ollama:m", prompt_template="[{tone}] {input}")
    LLMSystem(cfg, provider, vars={"tone": "warm"}).run(Case(id="c", input="hello"))
    assert provider.calls[0][-1].content == "[warm] hello"


def test_a_var_that_is_also_a_case_field_is_refused_for_a_model_target() -> None:
    cfg = TargetConfig(target="ollama:m", prompt_template="{tone} {input}")
    clashing = [Case.model_validate({"id": "c1", "input": "x", "tone": "own"})]
    with pytest.raises(ConfigError, match=r"variant 'warm': vars \['tone'\] .* case 'c1'"):
        check_variant_vars(cfg, "warm", {"tone": "warm"}, clashing)
    check_variant_vars(cfg, "warm", {"tone": "warm"}, [Case(id="c2", input="x")])  # no clash
    check_variant_vars(cfg, "warm", {}, clashing)  # nothing to clash with
    # A python or http system gets the two apart, so the same names are fine there.
    check_variant_vars(TargetConfig(target="python:a:b"), "warm", {"tone": "w"}, clashing)
    check_variant_vars(TargetConfig(target="http://sut/x"), "warm", {"tone": "w"}, clashing)


# --- running ----------------------------------------------------------------------------------

APP = """
def ask(x, ctx):
    text = x["q"] if not ctx.vars.get("shout") else x["q"].upper()
    return {"output": f"{text}|{ctx.variant}|{ctx.vars.get('shout')}"}
"""

VARIANTS = (
    "variants:\n"
    "  - {name: base}\n"
    "  - {name: loud, vars: {shout: true}}\n"
    "  - {name: hot, vars: {shout: false}, system: {params: {temperature: 0.9}}}\n"
)


def _runnable(tmp_path: Path, variants: str = VARIANTS) -> Path:
    cases = [{"id": f"c{i:02d}", "input": {"q": f"q{i}"}, "accept": [f"q{i}"]} for i in range(12)]
    (tmp_path / "cases.jsonl").write_text("\n".join(json.dumps(c) for c in cases) + "\n", "utf-8")
    (tmp_path / "variant_app.py").write_text(APP, "utf-8")
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: s\ndataset: cases.jsonl\nsystem: {target: 'python:variant_app:ask'}\n"
        "scorers: [{name: m, type: contains,\n"
        "  params: {any_of_field: accept, case_sensitive: true}}]\n"
        "gate: {metric: m, min_cases: 10}\n" + variants,
        "utf-8",
    )
    return suite


@pytest.fixture(autouse=True)
def _forget_app_module() -> object:
    yield
    sys.modules.pop("variant_app", None)


def test_a_variant_run_is_stamped_and_reaches_the_system(tmp_path: Path) -> None:
    cfg = load_config(_runnable(tmp_path))
    run = run_suite(cfg, use_cache=False, variant="loud")
    assert run.info.variant == "loud"
    assert run.info.suite_hash == suite_fingerprint(cfg, variant="loud")
    assert run.results[0].output.output == "Q0|loud|True"
    plain = run_suite(cfg, use_cache=False)
    assert plain.info.variant is None and plain.info.suite_hash == suite_fingerprint(cfg)
    assert plain.results[0].output.output == "q0|None|None"
    assert "variant `loud`" in run_markdown(run)
    assert "· variant" not in run_markdown(plain).splitlines()[0]


def test_two_variants_of_one_suite_compare(tmp_path: Path) -> None:
    cfg = load_config(_runnable(tmp_path))
    base = run_suite(cfg, use_cache=False, variant="base")
    loud = run_suite(cfg, use_cache=False, variant="loud")
    cmp = compare_runs(base, loud)
    assert cmp.comparable and cmp.same_dataset
    assert len(cmp.paired_ids) == 12 and not cmp.only_baseline and not cmp.only_candidate
    m = cmp.metrics["m"]
    assert (m.baseline, m.candidate) == (1.0, 0.0)  # shouting fails the case-sensitive match
    assert len(m.regressed) == 12
    # And a variant compares with the suite run as written.
    assert compare_runs(run_suite(cfg, use_cache=False), loud).comparable


def test_the_run_is_checked_against_the_variant_it_names(tmp_path: Path) -> None:
    suite = _runnable(tmp_path)
    cfg = load_config(suite)
    save_run(run_suite(cfg, use_cache=False, variant="loud"), tmp_path / "loud.json")
    assert main(["verify", str(tmp_path / "loud.json"), str(suite)]) == 0
    # Change what the variant sets and the file no longer matches the suite.
    suite.write_text(suite.read_text("utf-8").replace("{shout: true}", "{shout: 1}"), "utf-8")
    assert main(["verify", str(tmp_path / "loud.json"), str(suite)]) == EXIT_CANNOT_GATE


def test_a_run_of_a_variant_the_suite_dropped_does_not_verify(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    suite = _runnable(tmp_path)
    save_run(run_suite(load_config(suite), use_cache=False, variant="hot"), tmp_path / "hot.json")
    suite.write_text(suite.read_text("utf-8").split("variants:")[0], "utf-8")
    assert main(["verify", str(tmp_path / "hot.json"), str(suite)]) == EXIT_CANNOT_GATE
    assert "variant `hot`, which the suite does not define" in capsys.readouterr().out


def test_an_old_run_without_a_variant_still_loads_and_verifies(tmp_path: Path) -> None:
    suite = _runnable(tmp_path)
    run = run_suite(load_config(suite), use_cache=False)
    path = save_run(run, tmp_path / "old.json")
    data = json.loads(path.read_text("utf-8"))
    del data["info"]["variant"]  # as a run file written before variants had no such key
    path.write_text(json.dumps(data), "utf-8")
    assert load_run(path).info.variant is None
    assert main(["verify", str(path), str(suite)]) == 0


# --- the command line -------------------------------------------------------------------------


def test_run_writes_one_file_per_chosen_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = _runnable(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(suite), "--variant", "base", "--variant", "loud", "--quiet"]) == 0
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == ["base.json", "loud.json"]
    assert load_run(tmp_path / "runs" / "loud.json").info.variant == "loud"
    assert main(["compare", "runs/base.json", "runs/loud.json"]) in (0, 1)


def test_all_variants_runs_every_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    suite = _runnable(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["run", str(suite), "--all-variants", "--quiet"]) == 0
    assert sorted(p.name for p in (tmp_path / "runs").iterdir()) == [
        "base.json",
        "hot.json",
        "loud.json",
    ]


def test_one_variant_may_be_written_where_asked(tmp_path: Path) -> None:
    suite = _runnable(tmp_path)
    out = tmp_path / "chosen.json"
    assert main(["run", str(suite), "--variant", "hot", "--out", str(out), "--quiet"]) == 0
    assert load_run(out).info.variant == "hot"


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--variant", "nope"], "unknown variant 'nope'; the suite's variants: base, loud, hot"),
        (["--variant", "base", "--all-variants"], "use --variant or --all-variants, not both"),
        (["--all-variants", "--out", "x.json"], "--out names one file but several variants run"),
    ],
)
def test_bad_variant_flags_are_usage_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    suite = _runnable(tmp_path)
    assert main(["run", str(suite), "--quiet", *extra]) == EXIT_USAGE
    assert message in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_all_variants_needs_variants(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = _runnable(tmp_path, variants="")
    assert main(["run", str(suite), "--all-variants", "--quiet"]) == EXIT_USAGE
    assert "the suite defines no variants" in capsys.readouterr().err
