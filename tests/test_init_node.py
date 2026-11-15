"""``evalproof init --node``: a Node.js system, run by the suite as a ``command:`` target.

Nothing here calls a model. The one test that starts ``node`` runs the generated program through
``CommandSystem`` with the helper from this repository's ``npm/`` directory linked in as the
``evalproof`` package.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from evalproof.cli import build_parser, main
from evalproof.config import ConfigError, TargetConfig, load_config
from evalproof.dataset import load_dataset
from evalproof.models import Case
from evalproof.scaffold import init_project
from evalproof.systems import CommandSystem

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="symlinks and fake PATH")

ROOT = Path(__file__).resolve().parents[1]
NPM = ROOT / "npm"
BASE_FILES = {"evals/suite.yaml", "evals/cases.jsonl", "evals/README.md", "evals/.gitignore"}
NODE_FILES = BASE_FILES | {"evals/system.mjs"}
WORKFLOW = ".github/workflows/evalproof.yml"


DATASETS = ROOT / "tests" / "fixtures" / "datasets"
NODE = shutil.which("node")


@pytest.fixture(autouse=True)
def no_agent_clis(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A PATH with node and nothing else: the judge is the hosted default and init finds no
    claude, codex or ollama to probe."""
    bin_dir = tmp_path_factory.mktemp("bin")
    if NODE:
        (bin_dir / "node").symlink_to(NODE)
    monkeypatch.setenv("PATH", str(bin_dir))


def run_init(directory: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str]:
    code = main(["init", "--dir", str(directory), "--no-agents-md", "--no-skill", *args])
    return code, capsys.readouterr().out


def files_under(directory: Path) -> set[str]:
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def test_node_writes_a_system_file_and_a_suite_that_runs_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = run_init(tmp_path, capsys, "--node", "--ci", "none")
    assert code == 0
    assert files_under(tmp_path) == NODE_FILES
    assert "  wrote          evals/system.mjs" in out
    assert "npm install --save-dev evalproof" in out
    for rel in NODE_FILES:
        assert "@@" not in (tmp_path / rel).read_text(encoding="utf-8"), rel
    cfg = load_config(tmp_path / "evals" / "suite.yaml")
    assert cfg.system.kind == "command"
    assert cfg.system.target == "command:node system.mjs"
    assert cfg.system.system_prompt is None
    assert cfg.judge is not None and cfg.judge.target == "anthropic:claude-haiku-5-5"
    assert [s.type for s in cfg.scorers] == ["contains", "llm_judge"]
    assert len(load_dataset(cfg.resolve(cfg.dataset))) == 3


def test_the_default_init_is_unchanged(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_init(tmp_path, capsys, "--ci", "none")
    assert files_under(tmp_path) == BASE_FILES
    suite = (tmp_path / "evals" / "suite.yaml").read_text(encoding="utf-8")
    assert "system_prompt" in suite and "command:" not in suite.replace("# ", "")
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert "system.mjs" not in readme
    assert "\n\n\n" not in readme


def test_the_system_file_imports_the_helper_and_makes_no_network_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--node", "--ci", "none")
    text = (tmp_path / "evals" / "system.mjs").read_text(encoding="utf-8")
    assert 'import { runCommand } from "evalproof";' in text
    assert "await runCommand(" in text
    for forbidden in ("fetch(", "http://", "https://", "require("):
        assert forbidden not in text


def test_the_readme_says_how_to_install_the_helper(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--node", "--ci", "keyless")
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert "npm install --save-dev evalproof" in readme
    assert "command:npx tsx system.ts" in readme
    assert "npm ci" not in readme  # a keyless workflow runs no program
    assert "\n\n\n" not in readme


@pytest.mark.skipif(NODE is None, reason="node is not installed")
def test_the_generated_program_runs_through_the_command_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--node", "--ci", "none")
    modules = tmp_path / "node_modules"
    modules.mkdir()
    (modules / "evalproof").symlink_to(NPM, target_is_directory=True)
    evals = tmp_path / "evals"
    target = TargetConfig(target="command:node system.mjs", timeout_s=60)
    case = Case(id="pro-price", input={"question": "How much does Pro cost per month?"})

    # as written, the placeholder throws, and its message reaches the case
    out = CommandSystem(target, base_dir=str(evals)).run(case)
    assert out.error is not None
    assert "exited with code 1" in out.error
    assert "replace askModel() with a call into your own code" in out.error

    # once edited, the answer reaches the case: the question is the case's `question`
    source = (evals / "system.mjs").read_text(encoding="utf-8")
    edited = re.sub(
        r"async function askModel\(question\) \{.*?\n\}",
        "async function askModel(question) {\n  return `echo: ${question}`;\n}",
        source,
        flags=re.DOTALL,
    )
    assert edited != source
    (evals / "system.mjs").write_text(edited, encoding="utf-8")
    out = CommandSystem(target, base_dir=str(evals)).run(case)
    assert out.error is None
    assert out.output == "echo: How much does Pro cost per month?"
    out = CommandSystem(target, base_dir=str(evals)).run(Case(id="s", input="plain text"))
    assert out.output == "echo: plain text"


def test_an_edited_system_file_survives_a_second_init(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--node", "--ci", "none")
    (tmp_path / "evals" / "system.mjs").write_text("// mine\n", encoding="utf-8")
    _, out = run_init(tmp_path, capsys, "--node", "--ci", "none")
    assert "kept (exists)  evals/system.mjs" in out
    assert (tmp_path / "evals" / "system.mjs").read_text(encoding="utf-8") == "// mine\n"


def test_an_api_key_workflow_sets_up_node_and_installs_before_the_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--node", "--ci", "api-key")
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    steps = yaml.safe_load(text)["jobs"]["gate"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    setup = next(i for i, u in enumerate(uses) if u.startswith("actions/setup-node@"))
    assert re.fullmatch(r"actions/setup-node@[0-9a-f]{40}", uses[setup])
    assert steps[setup]["with"] == {"node-version": "24", "package-manager-cache": False}
    names = [s.get("name", "") for s in steps]
    install, run = names.index("Install the npm dependencies"), names.index("Run the suite")
    assert setup < install < run
    assert steps[install]["run"] == "npm ci"
    readme = (tmp_path / "evals" / "README.md").read_text(encoding="utf-8")
    assert "commit `package-lock.json`" in readme


def test_the_default_api_key_workflow_has_no_node_steps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_init(tmp_path, capsys, "--ci", "api-key")
    text = (tmp_path / WORKFLOW).read_text(encoding="utf-8")
    assert "setup-node" not in text and "npm ci" not in text


@pytest.mark.parametrize("platform", ["gitlab", "bitbucket"])
def test_node_with_an_api_key_workflow_off_github_is_refused(tmp_path: Path, platform: str) -> None:
    with pytest.raises(ConfigError, match=r"has no Node to run evals/system\.mjs with"):
        init_project(tmp_path, ci="api-key", platform=platform, node=True, emit=lambda _: None)
    assert files_under(tmp_path) == set()


@pytest.mark.parametrize("platform", ["gitlab", "bitbucket"])
def test_node_with_a_keyless_workflow_off_github_is_fine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], platform: str
) -> None:
    code, _ = run_init(tmp_path, capsys, "--node", "--platform", platform)
    assert code == 0
    assert "evals/system.mjs" in files_under(tmp_path)


def test_node_with_agent_files_only_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="use one or the other"):
        init_project(tmp_path, node=True, agent_files_only=True, emit=lambda _: None)
    assert files_under(tmp_path) == set()


def test_node_with_from_keeps_the_imported_cases_and_scorers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = DATASETS / "promptfoo" / "tests.jsonl"
    code, out = run_init(tmp_path, capsys, "--node", "--ci", "none", "--from", str(source))
    assert code == 0, out
    assert "evals/system.mjs" in files_under(tmp_path)
    cfg = load_config(tmp_path / "evals" / "suite.yaml")
    assert cfg.system.target == "command:node system.mjs"
    assert cfg.system.system_prompt is None
    assert [s.name for s in cfg.scorers] == ["exact", "contains"]
    assert cfg.judge is None  # these scorers need no model
    assert len(load_dataset(cfg.resolve(cfg.dataset))) > 0


def test_the_flag_is_on_the_parser_and_off_by_default() -> None:
    parser = build_parser()
    assert parser.parse_args(["init"]).node is False
    assert parser.parse_args(["init", "--node"]).node is True
