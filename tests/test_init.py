"""``evalproof init``: files written, judge choice, generated workflows and skill.

Coding-agent CLIs are faked with small shell scripts on ``PATH`` that log their arguments, so
the tests can also assert that init only ever asks them for a version or a model list.
The generated workflow's shell steps are run for real, in a throwaway git repository.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest
import yaml

from evalproof.cli import build_parser, main
from evalproof.config import ConfigError, load_config
from evalproof.dataset import load_dataset
from evalproof.models import SystemOutput
from evalproof.providers.base import ProviderError
from evalproof.scaffold import (
    ACTION_REF,
    AGENT_CLIS,
    build_files,
    choose_judge,
    cli_version,
    fill,
    init_project,
    template,
)
from evalproof.scorers import make_scorer

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="fake CLIs are shell scripts")

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = Path(".github/workflows/evalproof.yml")
SKILLS = (".agents/skills/evalproof/SKILL.md", ".claude/skills/evalproof/SKILL.md")
BASE_FILES = {"evals/suite.yaml", "evals/cases.jsonl", "evals/README.md", "evals/.gitignore"}

CODEX_MODELS = {
    "models": [
        {"slug": "gpt-6.1-sol", "visibility": "list", "priority": 1, "description": "Workhorse."},
        {"slug": "gpt-reserve", "visibility": "hide", "priority": 6, "description": "Fast."},
        {
            "slug": "gpt-6-luna",
            "visibility": "list",
            "priority": 7,
            "description": "Fast and cheap.",
        },
        {"slug": "gpt-5.6-luna", "visibility": "list", "priority": 10, "description": "Fast."},
    ]
}


def fake_cli(
    bin_dir: Path, name: str, version: str, *, models: dict[str, Any] | None = None
) -> None:
    """An executable ``name`` that answers ``--version`` (and ``debug models`` when given a
    catalogue) and logs every call to ``calls.log``. Anything else exits 99: a model call."""
    bin_dir.mkdir(parents=True, exist_ok=True)
    script = f"""#!/bin/sh
echo "{name} $*" >> "{bin_dir}/calls.log"
case "$1" in
  --version) echo '{version}'; exit 0 ;;
"""
    if models is not None:
        script += f"  debug) printf '%s\\n' '{json.dumps(models)}'; exit 0 ;;\n"
    script += "esac\nexit 99\n"
    path = bin_dir / name
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def calls(bin_dir: Path) -> list[str]:
    log = bin_dir / "calls.log"
    return log.read_text(encoding="utf-8").splitlines() if log.exists() else []


@pytest.fixture
def bin_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty directory that is the whole ``PATH``: no real claude, codex or ollama."""
    d = tmp_path / "bin"
    d.mkdir()
    monkeypatch.setenv("PATH", str(d))
    return d


def run_init(directory: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["init", "--dir", str(directory), *args])
    return code, capsys.readouterr().out


def files_under(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


# --- files written -----------------------------------------------------------------------


@pytest.mark.parametrize("ci", ["keyless", "api-key", "none"])
def test_writes_the_expected_files_for_each_ci_mode(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], ci: str
) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    out_dir = tmp_path / "proj"
    code, out = run_init(out_dir, capsys, "--ci", ci)
    assert code == 0
    expected = BASE_FILES | set(SKILLS) | ({WORKFLOW.as_posix()} if ci != "none" else set())
    assert files_under(out_dir) == expected
    for rel in expected:  # one line per file, in the order written
        assert f"  wrote          {rel}" in out
    assert "judge claude-code:haiku" in out
    for path in files_under(out_dir):
        assert "@@" not in (out_dir / path).read_text(encoding="utf-8"), path


@pytest.mark.parametrize("ci", ["keyless", "api-key"])
def test_generated_workflow_is_valid_yaml(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], ci: str
) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    run_init(tmp_path, capsys, "--ci", ci)
    wf = yaml.safe_load((tmp_path / WORKFLOW).read_text(encoding="utf-8"))
    assert wf["name"] == "evalproof"
    # YAML 1.1 reads the key `on` as a boolean. Neither workflow filters by path: a skipped
    # workflow leaves a required check pending, so the api-key one decides inside the job.
    assert wf[True] == {"pull_request": None}
    assert wf["permissions"] == {"contents": "read"}
    steps = wf["jobs"]["gate"]["steps"]
    assert steps[0]["uses"].startswith("actions/checkout@")
    assert steps[0]["with"] == {"fetch-depth": 0, "persist-credentials": False}
    assert steps[-1]["uses"] == f"technobasant/evalproof@{ACTION_REF}"


def test_the_gate_step_passes_inputs_the_action_declares(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    wf = yaml.safe_load((tmp_path / WORKFLOW).read_text(encoding="utf-8"))
    given = wf["jobs"]["gate"]["steps"][-1]["with"]
    declared = yaml.safe_load((ROOT / "action.yml").read_text(encoding="utf-8"))["inputs"]
    assert set(given) <= set(declared)
    assert {k for k, v in declared.items() if v["required"]} <= set(given)
    assert given["suite"] == "evals/suite.yaml"


def test_the_keyless_gate_verifies_the_candidate_against_the_suite(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    wf = yaml.safe_load(text)
    assert wf["jobs"]["gate"]["steps"][-1]["with"]["verify-candidate"] == "true"
    header = text.split("name: evalproof")[0]  # the honest limits, up front
    assert "evalproof verify" in header and "edited by hand" in header
    assert "--ci api-key" in header


def test_the_written_suite_loads_and_scores_its_own_cases(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    run_init(tmp_path, capsys)
    cfg = load_config(tmp_path / "evals" / "suite.yaml")
    assert cfg.judge is not None
    assert cfg.system.target == cfg.judge.target == "claude-code:haiku"
    assert cfg.system.system_prompt and cfg.system.prompt_template == "{question}"
    assert [s.type for s in cfg.scorers] == ["contains", "llm_judge"]
    assert cfg.gate is not None and cfg.gate.gated == ["judge"]
    cases = load_dataset(cfg.resolve(cfg.dataset))
    assert len(cases) == 3
    contains = make_scorer(cfg.scorers[0], cfg)
    for case in cases:  # the reference answer satisfies the deterministic scorer
        assert contains.score(case, SystemOutput(output=case.reference or "")).passed


def test_a_second_run_keeps_every_file(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    before = {p: (tmp_path / p).read_bytes() for p in files_under(tmp_path)}
    code, out = run_init(tmp_path, capsys)
    assert code == 0
    assert "  wrote " not in out
    assert out.count("kept (exists)") == len(before)
    assert {p: (tmp_path / p).read_bytes() for p in files_under(tmp_path)} == before


def test_an_edited_file_survives_unless_forced(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    suite = tmp_path / "evals" / "suite.yaml"
    original = suite.read_text(encoding="utf-8")
    suite.write_text("name: mine\n", encoding="utf-8")
    _, out = run_init(tmp_path, capsys)
    assert "kept (exists)  evals/suite.yaml" in out
    assert suite.read_text(encoding="utf-8") == "name: mine\n"
    code, out = run_init(tmp_path, capsys, "--force")
    assert code == 0
    assert "wrote          evals/suite.yaml" in out and "kept" not in out
    assert suite.read_text(encoding="utf-8") == original


def test_writes_nothing_outside_the_target_directory(
    tmp_path: Path,
    bin_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    run_init(tmp_path / "elsewhere" / "proj", capsys)
    assert list(cwd.iterdir()) == []
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bin", "cwd", "elsewhere"]
    assert [p.name for p in (tmp_path / "elsewhere").iterdir()] == ["proj"]


def test_default_directory_is_the_working_directory(
    tmp_path: Path,
    bin_dir: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["init", "--ci", "none", "--no-skill"]) == 0
    assert files_under(tmp_path) - {"bin/calls.log"} == BASE_FILES


def test_no_skill_and_ci_none_skip_those_files(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--no-skill", "--ci", "none")
    assert files_under(tmp_path) == BASE_FILES


# --- the skill ---------------------------------------------------------------------------


def test_both_skill_copies_are_identical_regular_files_with_valid_frontmatter(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    copies = [tmp_path / rel for rel in SKILLS]
    assert not any(p.is_symlink() for p in copies)  # copies, so a Windows checkout works
    assert copies[0].read_bytes() == copies[1].read_bytes()
    text = copies[0].read_text(encoding="utf-8")
    match = re.match(r"---\n(.*?)\n---\n", text, re.DOTALL)
    assert match is not None
    meta = yaml.safe_load(match.group(1))
    assert set(meta) == {"name", "description"}
    assert meta["name"] == "evalproof"
    assert meta["description"].startswith("Use when") and meta["description"].endswith(".")
    assert "prompt" in meta["description"] and "regression" in meta["description"]
    assert len(text.splitlines()) < 120
    lowered = text.lower()
    for rule in ("never edit", "never weaken", "never delete", "on exit 1, 3 or 4, stop"):
        assert rule in lowered, rule


def test_the_skill_uses_commands_and_flags_that_exist(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    text = (tmp_path / SKILLS[0]).read_text(encoding="utf-8")
    parser = build_parser()
    parser.parse_args(["run", "evals/suite.yaml", "--out", "x.json"])
    parser.parse_args(["compare", "a.json", "b.json", "--suite", "evals/suite.yaml", "--gate"])
    for fragment in (
        "evalproof run evals/suite.yaml --out evals/.evalproof/candidate.json",
        "--suite evals/suite.yaml --gate",
        "evalproof run evals/suite.yaml --out evals/runs/latest.json",
    ):
        assert fragment in text


# --- the judge ---------------------------------------------------------------------------


def test_claude_is_preferred_over_codex_and_ollama(bin_dir: Path) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=CODEX_MODELS)
    fake_cli(bin_dir, "ollama", "ollama version 0.1")
    judge = choose_judge(None)
    assert judge.target == "claude-code:haiku" and "claude is on PATH" in judge.reason
    assert calls(bin_dir) == []  # nothing needed to be asked


def test_codex_model_comes_from_the_catalogue(bin_dir: Path) -> None:
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=CODEX_MODELS)
    fake_cli(bin_dir, "ollama", "ollama version 0.1")
    judge = choose_judge(None)
    assert judge.target == "codex:gpt-6-luna"  # fast, listed, the lowest priority of those
    assert "codex debug models" in judge.reason
    assert calls(bin_dir) == ["codex debug models"]


def test_codex_falls_back_to_the_first_listed_model_when_none_is_fast(bin_dir: Path) -> None:
    models = {"models": [{"slug": "a", "visibility": "hide"}, {"slug": "b", "visibility": "list"}]}
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=models)
    assert choose_judge(None).target == "codex:b"


@pytest.mark.parametrize("catalogue", [None, {"models": []}, {"nope": 1}])
def test_codex_without_a_readable_catalogue_gets_a_placeholder(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str], catalogue: Any
) -> None:
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=catalogue)
    judge = choose_judge(None)
    assert judge.target == "codex:<model>" and judge.note and judge.warning
    code, out = run_init(tmp_path, capsys)
    assert code == 0 and "note: could not read `codex debug models`" in out
    suite = (tmp_path / "evals" / "suite.yaml").read_text(encoding="utf-8")
    assert suite.count("# TODO: replace <model>") == 2  # system and judge
    loaded = load_config(tmp_path / "evals" / "suite.yaml")
    assert loaded.judge is not None and loaded.judge.target == "codex:<model>"


def test_ollama_when_it_is_the_only_cli(bin_dir: Path, tmp_path: Path) -> None:
    fake_cli(bin_dir, "ollama", "ollama version 0.1")
    judge = choose_judge(None)
    assert judge.target == "ollama:qwen3:8b" and "ollama pull qwen3:8b" in judge.reason
    suite = build_files(judge, "keyless", False, None)[0][1]
    assert "concurrency: 1" in suite and "api_key_env" not in suite


def test_hosted_anthropic_when_nothing_is_installed(bin_dir: Path) -> None:
    judge = choose_judge(None)
    assert judge.target == "anthropic:claude-haiku-5-5"
    assert judge.api_key_env == "ANTHROPIC_API_KEY" and "ANTHROPIC_API_KEY" in judge.reason
    suite = build_files(judge, "keyless", False, None)[0][1]
    assert suite.count("  api_key_env: ANTHROPIC_API_KEY") == 2  # system and judge


def test_explicit_judge_wins_and_is_checked(bin_dir: Path) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    assert choose_judge("openai:gpt-5").reason == "from --judge"
    assert choose_judge("anthropic:claude-sonnet-4-5").api_key_env == "ANTHROPIC_API_KEY"
    assert choose_judge("codex:gpt-6-luna").provider == "codex"
    with pytest.raises(ConfigError, match="unknown provider 'nope'"):
        choose_judge("nope:model")
    with pytest.raises(ProviderError, match="must look like"):
        choose_judge("justamodel")


def test_init_never_calls_a_model(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=CODEX_MODELS)
    run_init(tmp_path / "a", capsys, "--ci", "api-key")
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    run_init(tmp_path / "b", capsys, "--ci", "api-key")
    allowed = {"codex debug models", "codex --version", "claude --version"}
    assert set(calls(bin_dir)) <= allowed and calls(bin_dir)


def test_cli_version_reads_the_number(bin_dir: Path) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2")
    fake_cli(bin_dir, "broken", "no version here")
    assert cli_version(str(bin_dir / "claude")) == "2.1.322"
    assert cli_version(str(bin_dir / "codex")) == "0.167.2"
    assert cli_version(str(bin_dir / "broken")) is None
    assert cli_version(str(bin_dir / "missing")) is None


# --- the api-key workflow ----------------------------------------------------------------


def _steps(path: Path) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = yaml.safe_load(path.read_text(encoding="utf-8"))["jobs"]["gate"][
        "steps"
    ]
    return steps


def test_api_key_workflow_pins_the_cli_found_at_init_time(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(bin_dir, "claude", "2.1.322 (Claude Code)")
    run_init(tmp_path, capsys, "--ci", "api-key")
    steps = _steps(tmp_path / WORKFLOW)
    install = next(s for s in steps if s.get("name") == "Install claude")
    assert install["run"] == "npm install -g @anthropic-ai/claude-code@2.1.322"
    run = next(s for s in steps if s.get("name") == "Run the suite")
    assert run["env"] == {
        "EVALPROOF_CLI_AUTH": "api-key",
        "ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}",
    }
    assert 'uvx evalproof@0.7.1 run evals/suite.yaml --out "$RUNNER_TEMP/' in run["run"]
    assert "judge_cli_version" in (tmp_path / WORKFLOW).read_text(encoding="utf-8")


def test_api_key_workflow_for_codex_sets_the_auth_mode_and_the_openai_key(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(bin_dir, "codex", "codex-cli 0.167.2", models=CODEX_MODELS)
    _, out = run_init(tmp_path, capsys, "--ci", "api-key")
    steps = _steps(tmp_path / WORKFLOW)
    install = next(s for s in steps if s.get("name") == "Install codex")
    assert install["run"] == "npm install -g @openai/codex@0.167.2"
    run = next(s for s in steps if s.get("name") == "Run the suite")
    assert run["env"] == {
        "EVALPROOF_CLI_AUTH": "api-key",
        "OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}",
    }
    assert "the workflow installs codex 0.167.2" in out
    assert "add the repository secret OPENAI_API_KEY" in out


def test_api_key_workflow_installs_latest_when_the_cli_is_missing(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, out = run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "claude-code:sonnet")
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    assert "npm install -g @anthropic-ai/claude-code@latest" in text
    assert "did not find claude" in text and "installs the latest" in out
    yaml.safe_load(text)


def test_api_key_workflow_for_a_hosted_api_judge_has_no_cli_steps(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    steps = _steps(tmp_path / WORKFLOW)
    assert not any("npm install" in str(s.get("run", "")) for s in steps)
    run = next(s for s in steps if s.get("name") == "Run the suite")
    assert run["env"] == {"OPENAI_API_KEY": "${{ secrets.OPENAI_API_KEY }}"}  # no CLI: no mode
    assert "evals/suite.yaml --out" in run["run"] and "pyyaml" not in run["run"]


def test_api_key_workflow_for_the_default_hosted_judge(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """No agent CLI on PATH: the starter judge is a hosted Anthropic model and CI uses its key."""
    run_init(tmp_path, capsys, "--ci", "api-key")
    steps = _steps(tmp_path / WORKFLOW)
    run = next(s for s in steps if s.get("name") == "Run the suite")
    assert run["env"] == {"ANTHROPIC_API_KEY": "${{ secrets.ANTHROPIC_API_KEY }}"}
    assert "uvx evalproof@0.7.1 run evals/suite.yaml --out" in run["run"]
    assert not any("npm install" in str(s.get("run", "")) for s in steps)
    assert "api_key_env: ANTHROPIC_API_KEY" in (tmp_path / "evals" / "suite.yaml").read_text(
        encoding="utf-8"
    )


def test_api_key_ci_refuses_a_local_judge_before_writing_anything(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fake_cli(bin_dir, "ollama", "ollama version 0.1")
    assert main(["init", "--dir", str(tmp_path / "p"), "--ci", "api-key"]) == 2
    assert "needs a hosted judge" in capsys.readouterr().err
    assert not (tmp_path / "p").exists()


@pytest.mark.parametrize(
    ("binary", "version", "judge", "key_var"),
    [
        ("claude", "2.1.322 (Claude Code)", "claude-code:sonnet", "ANTHROPIC_API_KEY"),
        ("codex", "codex-cli 0.167.2", "codex:gpt-6-luna", "OPENAI_API_KEY"),
    ],
)
def test_api_key_workflow_has_no_ci_suite_overlay(
    tmp_path: Path,
    bin_dir: Path,
    capsys: pytest.CaptureFixture[str],
    binary: str,
    version: str,
    judge: str,
    key_var: str,
) -> None:
    """A CLI judge is switched to its key by one env var, not by a rewritten copy of the suite."""
    fake_cli(bin_dir, binary, version, models=CODEX_MODELS)
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", judge)
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    run = next(s for s in _steps(tmp_path / WORKFLOW) if s.get("name") == "Run the suite")
    assert run["env"]["EVALPROOF_CLI_AUTH"] == "api-key"
    assert run["env"][key_var] == f"${{{{ secrets.{key_var} }}}}"
    assert ".suite.ci.yaml" not in text and "pyyaml" not in text and "<<'PY'" not in text
    assert 'run evals/suite.yaml --out "$RUNNER_TEMP/' in run["run"]
    written = {p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*") if p.is_file()}
    assert not any("suite.ci" in name for name in written)
    # The suite is the one a developer runs on a login: it names no key.
    cfg = load_config(tmp_path / "evals" / "suite.yaml")
    assert cfg.judge is not None and cfg.judge.api_key_env is None


def test_the_evals_readme_explains_the_cli_auth_mode_for_a_cli_judge(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "claude-code:sonnet")
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert "EVALPROOF_CLI_AUTH=api-key" in readme and "overlay" not in readme
    run_init(tmp_path / "hosted", capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    assert "EVALPROOF_CLI_AUTH" not in (tmp_path / "hosted" / "evals" / "README.md").read_text(
        encoding="utf-8"
    )


def test_api_key_workflow_does_not_filter_by_path(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    assert "paths:" not in text.split("permissions:")[0]
    header = text.split("name: evalproof")[0]
    assert "the run step fails" not in header and "forks get no secrets" in header.replace(
        "\n# ", " "
    )


def test_api_key_workflow_decides_inside_the_job_and_skips_forks(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    steps = _steps(tmp_path / WORKFLOW)
    names = [s.get("name") or s.get("uses", "") for s in steps]
    assert names.index("Check whether anything watched changed") == 1  # right after checkout
    changed = steps[1]
    assert changed["id"] == "changed" and changed["env"]["WATCH"] == "evals/"
    not_fork = "github.event.pull_request.head.repo.full_name == github.repository"
    for name in ("Run the suite", "Restore the response cache", "Compare with the baseline"):
        step = next(s for s in steps if s.get("name", "").startswith(name))
        assert "steps.baseline.outputs.found == 'true'" in step["if"], name
        assert not_fork in step["if"], name  # no secrets for a fork, so no run and no gate
    baseline = next(s for s in steps if s.get("name", "").startswith("Take the baseline"))
    assert baseline["if"] == "steps.changed.outputs.changed == 'true'"
    skip = next(s for s in steps if s.get("name") == "Skip pull requests from forks")
    assert "head.repo.full_name != github.repository" in skip["if"]
    assert "no secrets" in skip["run"] and "::notice" in skip["run"]


def test_api_key_workflow_caches_responses_with_a_pinned_action(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    path = tmp_path / WORKFLOW
    step = next(s for s in _steps(path) if s.get("name") == "Restore the response cache")
    assert re.fullmatch(r"actions/cache/restore@[0-9a-f]{40}", step["uses"])  # a commit
    text = path.read_text("utf-8")
    assert re.search(r"actions/cache/restore@[0-9a-f]{40} # v\d+\.\d+\.\d+", text)
    assert re.search(r"actions/cache/save@[0-9a-f]{40} # v\d+\.\d+\.\d+", text)
    assert step["with"]["path"] == "evals/.evalproof/cache"
    assert step["with"]["key"] == "evalproof-${{ hashFiles('evals/**') }}"
    assert step["with"]["restore-keys"].strip() == "evalproof-"
    # The path is where the suite's default cache_dir lands: next to the suite.
    cfg = load_config(tmp_path / "evals" / "suite.yaml")
    assert (Path("evals") / cfg.cache_dir).as_posix() == step["with"]["path"]
    names = [s.get("name", "") for s in _steps(path)]
    assert names.index("Restore the response cache") < names.index("Run the suite")
    # Saved before the gate, so a pull request that fails the gate keeps what it paid for.
    save = next(s for s in _steps(path) if s.get("name") == "Save the response cache")
    assert names.index("Run the suite") < names.index("Save the response cache")
    assert names.index("Save the response cache") < names.index(
        "Compare with the baseline and gate"
    )
    assert save["with"]["path"] == step["with"]["path"]
    assert save["with"]["key"] == "${{ steps.cache.outputs.cache-primary-key }}"
    assert step["id"] == "cache" and "cache-hit != 'true'" in save["if"]


# --- the keyless workflow, run for real ---------------------------------------------------

needs_git = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("bash") is None, reason="needs git and bash"
)


def _git(repo: Path, *args: str) -> str:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.com",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.com",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
    }
    done = subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True, env=env
    )
    return done.stdout


class Pr:
    """A repository whose ``main`` is ``origin/main`` and whose HEAD is a pull-request branch."""

    def __init__(self, tmp_path: Path, steps: list[dict[str, Any]], base_has_run: bool) -> None:
        self.repo = tmp_path / "repo"
        self.repo.mkdir()
        self.steps = steps
        _git(self.repo, "init", "-q", "-b", "main")
        (self.repo / "evals" / "runs").mkdir(parents=True)
        (self.repo / "evals" / "suite.yaml").write_text("name: s\n", encoding="utf-8")
        (self.repo / "src").mkdir()
        (self.repo / "src" / "prompt.txt").write_text("v1\n", encoding="utf-8")
        if base_has_run:
            (self.repo / "evals" / "runs" / "latest.json").write_text('{"v": 1}', encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "base")
        _git(self.repo, "update-ref", "refs/remotes/origin/main", "HEAD")
        _git(self.repo, "checkout", "-q", "-b", "feature")

    def commit(self, files: dict[str, str]) -> None:
        for rel, content in files.items():
            path = self.repo / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        _git(self.repo, "add", "-A")
        _git(self.repo, "commit", "-q", "-m", "change")

    def run_step(self, name: str, tmp_path: Path) -> tuple[int, str, dict[str, str]]:
        step = next(s for s in self.steps if s.get("name", "").startswith(name))
        env_decl = step.get("env", {})
        out_file = tmp_path / "github_output"
        out_file.write_text("", encoding="utf-8")
        env = {
            **os.environ,
            "BASE_REF": "main",
            "BASELINE": str(tmp_path / "baseline.json"),
            "GITHUB_OUTPUT": str(out_file),
            **{k: v for k, v in env_decl.items() if "${{" not in v},  # WATCH keeps its default
        }
        done = subprocess.run(
            ["bash", "-eo", "pipefail", "-c", step["run"]],
            cwd=self.repo,
            capture_output=True,
            text=True,
            env=env,
            check=False,
        )
        outputs = dict(
            line.split("=", 1) for line in out_file.read_text(encoding="utf-8").splitlines()
        )
        return done.returncode, done.stdout, outputs


@pytest.fixture
def keyless_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> list[dict[str, Any]]:
    """The keyless workflow's steps. Init runs with no agent CLI on PATH, but the steps
    themselves need the real PATH (git, bash), so the override is undone afterwards."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    with monkeypatch.context() as patch:
        patch.setenv("PATH", str(empty))
        run_init(tmp_path / "gen", capsys)
    return _steps(tmp_path / "gen" / WORKFLOW)


@needs_git
def test_keyless_passes_with_a_notice_when_the_base_has_no_run_file(
    tmp_path: Path, keyless_steps: list[dict[str, Any]]
) -> None:
    pr = Pr(tmp_path, keyless_steps, base_has_run=False)
    pr.commit({"evals/runs/latest.json": '{"v": 2}'})
    code, out, outputs = pr.run_step("Take the baseline", tmp_path)
    assert code == 0 and outputs == {"found": "false"}
    assert "::notice title=evalproof::main has no evals/runs/latest.json yet" in out
    gate = keyless_steps[-1]
    assert gate["if"] == "steps.baseline.outputs.found == 'true'"  # so the gate is skipped


@needs_git
def test_keyless_takes_the_baseline_from_the_base_branch(
    tmp_path: Path, keyless_steps: list[dict[str, Any]]
) -> None:
    pr = Pr(tmp_path, keyless_steps, base_has_run=True)
    pr.commit({"evals/runs/latest.json": '{"v": 2}'})
    code, out, outputs = pr.run_step("Take the baseline", tmp_path)
    assert code == 0 and outputs == {"found": "true"} and "::error" not in out
    assert (tmp_path / "baseline.json").read_text(encoding="utf-8") == '{"v": 1}'  # not the PR's


@needs_git
def test_keyless_fails_when_the_pr_removes_the_run_file(
    tmp_path: Path, keyless_steps: list[dict[str, Any]]
) -> None:
    pr = Pr(tmp_path, keyless_steps, base_has_run=True)
    _git(pr.repo, "rm", "-q", "evals/runs/latest.json")
    _git(pr.repo, "commit", "-q", "-m", "drop it")
    code, out, _ = pr.run_step("Take the baseline", tmp_path)
    assert code == 1 and "evals/runs/latest.json is missing from this branch" in out


@needs_git
@pytest.mark.parametrize(
    ("changes", "stale"),
    [
        ({"evals/cases.jsonl": "{}\n"}, True),
        ({"evals/suite.yaml": "name: t\n"}, True),
        ({"evals/cases.jsonl": "{}\n", "evals/runs/latest.json": '{"v": 2}'}, False),
        ({"evals/runs/latest.json": '{"v": 2}'}, False),
        ({"src/prompt.txt": "v2\n"}, False),  # outside WATCH until the user adds it
        ({"README.md": "hello\n"}, False),
    ],
)
def test_keyless_warns_when_evals_changed_but_the_run_file_did_not(
    tmp_path: Path,
    keyless_steps: list[dict[str, Any]],
    changes: dict[str, str],
    stale: bool,
) -> None:
    pr = Pr(tmp_path, keyless_steps, base_has_run=True)
    pr.commit(changes)
    code, out, _ = pr.run_step("Warn when the run file is stale", tmp_path)
    assert code == 0  # a warning, never a failure
    assert ("::warning title=evalproof run file is stale::" in out) is stale
    if stale:
        assert next(iter(f for f in changes if f != "evals/runs/latest.json")) in out


@needs_git
def test_keyless_warning_follows_the_watch_list(
    tmp_path: Path, keyless_steps: list[dict[str, Any]]
) -> None:
    step = next(s for s in keyless_steps if s.get("name", "").startswith("Warn when"))
    assert step["env"]["WATCH"] == "evals/"
    step["env"]["WATCH"] = "evals/ src/"  # what a user does to cover their prompt
    pr = Pr(tmp_path, keyless_steps, base_has_run=True)
    pr.commit({"src/prompt.txt": "v2\n"})
    _, out, _ = pr.run_step("Warn when the run file is stale", tmp_path)
    assert "src/prompt.txt" in out and "::warning" in out


@pytest.fixture
def api_key_steps(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> list[dict[str, Any]]:
    run_init(tmp_path / "gen", capsys, "--ci", "api-key", "--judge", "openai:gpt-5")
    return _steps(tmp_path / "gen" / WORKFLOW)


@needs_git
@pytest.mark.parametrize(
    ("changes", "changed"),
    [
        ({"evals/cases.jsonl": "{}\n"}, True),
        ({"evals/suite.yaml": "name: t\n"}, True),
        ({"evals/runs/latest.json": '{"v": 2}'}, False),  # CI makes the run itself
        ({"src/prompt.txt": "v2\n"}, False),  # outside WATCH until the user adds it
        ({"README.md": "hello\n"}, False),
    ],
)
def test_api_key_runs_the_suite_only_when_something_watched_changed(
    tmp_path: Path,
    api_key_steps: list[dict[str, Any]],
    changes: dict[str, str],
    changed: bool,
) -> None:
    pr = Pr(tmp_path, api_key_steps, base_has_run=True)
    pr.commit(changes)
    code, out, outputs = pr.run_step("Check whether anything watched changed", tmp_path)
    assert code == 0 and outputs == {"changed": "true" if changed else "false"}
    assert ("::notice title=evalproof::Nothing in evals/ changed" in out) is not changed


@needs_git
def test_api_key_change_check_follows_the_watch_list(
    tmp_path: Path, api_key_steps: list[dict[str, Any]]
) -> None:
    step = next(s for s in api_key_steps if s.get("id") == "changed")
    step["env"]["WATCH"] = "evals/ src/"
    pr = Pr(tmp_path, api_key_steps, base_has_run=True)
    pr.commit({"src/prompt.txt": "v2\n"})
    _, _, outputs = pr.run_step("Check whether anything watched changed", tmp_path)
    assert outputs == {"changed": "true"}


@needs_git
def test_api_key_baseline_notice_when_the_base_has_no_run_file(
    tmp_path: Path, api_key_steps: list[dict[str, Any]]
) -> None:
    pr = Pr(tmp_path, api_key_steps, base_has_run=False)
    pr.commit({"evals/cases.jsonl": "{}\n"})
    code, out, outputs = pr.run_step("Take the baseline", tmp_path)
    assert code == 0 and outputs == {"found": "false"} and "::notice" in out


# --- plumbing ----------------------------------------------------------------------------


def test_fill_handles_blocks_optional_lines_and_missing_values() -> None:
    text = "a: @@X@@\n@@OPT@@\n#@@IF on\n  kept\n#@@END\n#@@IF off\n  dropped\n#@@END\nz\n"
    assert fill(text, {"X": "1", "OPT": ""}, frozenset({"on"})) == "a: 1\n  kept\nz\n"
    with pytest.raises(KeyError):
        fill("@@MISSING@@ here", {})


def test_template_files_ship_as_package_data() -> None:
    for name in (
        "suite.yaml",
        "cases.jsonl",
        "evals-readme.md",
        "gitignore",
        "workflow-keyless.yml",
        "workflow-api-key.yml",
        "SKILL.md",
    ):
        assert template(name).strip()
    assert set(AGENT_CLIS) == {"claude-code", "codex"}


def test_evals_readme_names_the_command_that_writes_the_run_file(
    tmp_path: Path, bin_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys)
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert "uvx evalproof run evals/suite.yaml --out evals/runs/latest.json" in readme
    assert "no model calls" in readme and "latest.json" in readme
    assert textwrap.dedent(readme).count("\n\n") >= 2


def test_bad_ci_value_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="--ci must be one of"):
        init_project(tmp_path, ci="sometimes")
    with pytest.raises(SystemExit):
        main(["init", "--dir", str(tmp_path), "--ci", "sometimes"])
