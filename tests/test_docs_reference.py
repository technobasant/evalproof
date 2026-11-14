"""The reference pages in docs/ stay in step with the code they describe.

Each test reads a page and compares it with the models, the parser or the constants it documents,
so adding a flag, a config key or a run-file field without documenting it fails here.
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
from pathlib import Path
from typing import get_args

import pytest
from pydantic import BaseModel

from evalproof.cli import EXIT_INTERRUPTED, build_parser
from evalproof.config import (
    CalibrationConfig,
    GateConfig,
    GenerationParams,
    Pricing,
    ScorerConfig,
    ScorerType,
    SuiteConfig,
    TargetConfig,
    VariantConfig,
    load_config,
)
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
)
from evalproof.models import (
    CaseResult,
    ModelUsage,
    Run,
    RunInfo,
    Score,
    SystemOutput,
    ToolCall,
    Trace,
)
from evalproof.scorers import _SIMPLE as SIMPLE_SCORERS
from evalproof.scorers import ChecksScorer, PythonScorer

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

pytestmark = pytest.mark.skipif(not DOCS.is_dir(), reason="docs/ is not in this checkout")


def _page(name: str) -> str:
    return (DOCS / name).read_text(encoding="utf-8")


def _sections(page: str) -> dict[str, str]:
    """The text under each ``##`` or ``###`` heading of a page, by heading."""
    parts = re.split(r"(?m)^#{2,3} ", page)
    return {part.split("\n", 1)[0].strip(): part for part in parts[1:]}


def _row(section: str, key: str) -> str | None:
    """The table row of ``key`` in a section: the line that starts ``| `key` |``."""
    prefix = f"| `{key}` |"
    return next((line for line in section.splitlines() if line.startswith(prefix)), None)


def _default_texts(default: object) -> list[str]:
    """How a default may be written in a table: the text the page must show for it."""
    if isinstance(default, bool):
        return ["true" if default else "false"]
    if isinstance(default, int | float | str):
        return [str(default)]
    return []


# --- configuration ------------------------------------------------------------------------------

CONFIG_SECTIONS: dict[str, type[BaseModel]] = {
    "Suite keys": SuiteConfig,
    "System and judge": TargetConfig,
    "Generation parameters": GenerationParams,
    "Pricing": Pricing,
    "Scorers": ScorerConfig,
    "Calibration": CalibrationConfig,
    "Gate": GateConfig,
    "Variants": VariantConfig,
}


@pytest.mark.parametrize("heading", CONFIG_SECTIONS)
def test_configuration_page_lists_every_field_with_its_default(heading: str) -> None:
    model = CONFIG_SECTIONS[heading]
    sections = _sections(_page("reference/configuration.md"))
    assert heading in sections, f"configuration.md has no '## {heading}' section"
    problems = []
    for name, field in model.model_fields.items():
        row = _row(sections[heading], name)
        if row is None:
            problems.append(f"{model.__name__}.{name} has no table row")
            continue
        for text in _default_texts(field.default):
            if f"`{text}`" not in row:
                problems.append(f"{model.__name__}.{name}: the row does not show `{text}`")
    assert not problems, "\n".join(problems)


def test_configuration_page_lists_the_loader_prompt_file_keys() -> None:
    section = _sections(_page("reference/configuration.md"))["System and judge"]
    for key in ("system_prompt_file", "prompt_template_file"):
        assert _row(section, key), f"no row for {key}"


def test_configuration_page_lists_every_scorer_type_and_parameter() -> None:
    section = _sections(_page("reference/configuration.md"))["Scorers"]
    for scorer_type in get_args(ScorerType):
        assert f"`{scorer_type}`" in section, f"scorer type {scorer_type} is not on the page"
    classes: dict[str, type] = {
        **SIMPLE_SCORERS,
        "checks": ChecksScorer,
        "python": PythonScorer,
    }
    problems = []
    for scorer_type, cls in classes.items():
        for param in inspect.signature(cls.__init__).parameters:
            if param in ("self", "name", "search_path"):
                continue
            rows = [
                line
                for line in section.splitlines()
                if line.startswith("|") and f"`{scorer_type}`" in line and f"| `{param}` |" in line
            ]
            if not rows:
                problems.append(f"{scorer_type}.{param} has no parameter row")
    for param in ("rubric", "rubric_file", "max_retries"):
        if not any(
            "`llm_judge`" in line and f"| `{param}` |" in line for line in section.splitlines()
        ):
            problems.append(f"llm_judge.{param} has no parameter row")
    assert not problems, "\n".join(problems)


def test_configuration_example_loads(tmp_path: Path) -> None:
    page = _page("reference/configuration.md")
    block = re.search(r"```yaml title=\"suite.yaml\"\n(.*?)```", page, re.S)
    assert block, "configuration.md has no annotated suite"
    text = block.group(1)
    (tmp_path / "suite.yaml").write_text(text, encoding="utf-8")
    for file_name in re.findall(r"_file:\s*(\S+)", text):
        (tmp_path / file_name).write_text("text", encoding="utf-8")
    cfg = load_config(tmp_path / "suite.yaml")
    assert cfg.gate is not None and cfg.gate.gated == ["judge"]


# --- commands -----------------------------------------------------------------------------------


def _subcommands() -> dict[str, argparse.ArgumentParser]:
    parser = build_parser()
    action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return dict(action.choices)


def test_commands_page_has_a_section_for_every_subcommand() -> None:
    sections = _sections(_page("reference/commands.md"))
    documented = {h.removeprefix("evalproof ") for h in sections if h.startswith("evalproof ")}
    assert documented == set(_subcommands())


@pytest.mark.parametrize("command", sorted(_subcommands()))
def test_commands_page_lists_every_argument_and_flag(command: str) -> None:
    section = _sections(_page("reference/commands.md"))[f"evalproof {command}"]
    problems = []
    for action in _subcommands()[command]._actions:
        if isinstance(action, argparse._HelpAction):
            continue
        if not action.option_strings:
            if _row(section, action.dest.upper()) is None:
                problems.append(f"argument {action.dest} has no row")
            continue
        flag = next(o for o in action.option_strings if o.startswith("--"))
        row = _row(section, flag)
        if row is None:
            problems.append(f"{flag} has no row")
            continue
        if action.default not in (None, False) and f"`{action.default}`" not in row:
            problems.append(f"{flag}: the row does not show the default `{action.default}`")
        for choice in action.choices or ():
            if f"`{choice}`" not in row:
                problems.append(f"{flag}: the row does not list the choice `{choice}`")
    assert not problems, "\n".join(problems)


# --- run file -----------------------------------------------------------------------------------

RUN_FILE_SECTIONS: dict[str, type[BaseModel]] = {
    "Run": Run,
    "Run info": RunInfo,
    "Case result": CaseResult,
    "System output": SystemOutput,
    "Score": Score,
    "Trace": Trace,
    "Tool call": ToolCall,
    "Model usage": ModelUsage,
}


@pytest.mark.parametrize("heading", RUN_FILE_SECTIONS)
def test_run_file_page_lists_every_field(heading: str) -> None:
    model = RUN_FILE_SECTIONS[heading]
    sections = _sections(_page("reference/run-file.md"))
    assert heading in sections, f"run-file.md has no '## {heading}' section"
    missing = [name for name in model.model_fields if _row(sections[heading], name) is None]
    assert not missing, f"{model.__name__} fields with no row: {missing}"


def test_run_file_example_is_a_current_run() -> None:
    page = _page("reference/run-file.md")
    block = re.search(r"```json[^\n]*\n(.*?)```", page, re.S)
    assert block, "run-file.md has no example"
    data = json.loads(block.group(1))
    Run.model_validate(data)  # a valid run
    result = data["results"][0]
    assert set(data["info"]) == set(RunInfo.model_fields)
    assert set(result) == set(CaseResult.model_fields)
    assert set(result["output"]) == set(SystemOutput.model_fields)
    assert {key for score in result["scores"].values() for key in score} == set(Score.model_fields)


# --- exit codes ---------------------------------------------------------------------------------


def test_exit_codes_page_matches_the_constants() -> None:
    assert (EXIT_PASS, EXIT_REGRESSION, EXIT_USAGE, EXIT_BUDGET, EXIT_CANNOT_GATE) == (
        0,
        1,
        2,
        3,
        4,
    )
    assert EXIT_INTERRUPTED == 130
    page = _page("reference/exit-codes.md")
    for code, name in [
        (0, "pass"),
        (1, "regression"),
        (2, "usage error"),
        (3, "budget breach"),
        (4, "cannot gate"),
        (130, "interrupted"),
    ]:
        assert re.search(rf"(?m)^\| {code} \| {name} \|", page), f"no row for exit {code}"


# --- URLs ---------------------------------------------------------------------------------------

# Every page of the 0.7.0 docs. No page may lose its URL: it stays where it was, or it becomes the
# index of a section at the same URL (a/b.md -> a/b/index.md).
PAGES_0_7_0 = [
    "index.md",
    "about/architecture.md",
    "about/changelog.md",
    "about/decisions.md",
    "about/limits.md",
    "about/measurements.md",
    "about/project.md",
    "about/related-tools.md",
    "about/roadmap.md",
    "adr/0001-judge-is-a-classifier.md",
    "adr/0002-local-first-providers.md",
    "adr/0003-gate-on-significance-not-delta.md",
    "adr/0004-agent-clis-as-providers.md",
    "getting-started/credentials.md",
    "getting-started/install.md",
    "getting-started/quickstart.md",
    "getting-started/use-cases.md",
    "guides/adapters.md",
    "guides/ci.md",
    "guides/import.md",
    "guides/providers.md",
    "guides/python.md",
    "guides/readiness.md",
    "guides/repeated-trials.md",
    "guides/testing-agents.md",
    "reference/api.md",
    "reference/commands.md",
    "reference/concepts.md",
]


@pytest.mark.parametrize("page", PAGES_0_7_0)
def test_every_0_7_0_url_still_resolves(page: str) -> None:
    same_url = (DOCS / page.removesuffix(".md") / "index.md").exists()  # a/b.md -> a/b/index.md
    assert (DOCS / page).exists() or same_url, f"{page} has lost its URL"


# --- troubleshooting and glossary ---------------------------------------------------------------


def test_troubleshooting_has_at_least_fifteen_entries_with_a_cause_and_a_fix() -> None:
    page = _page("guides/troubleshooting.md")
    entries = [part for part in re.split(r"(?m)^### ", page)[1:]]
    assert len(entries) >= 15
    for entry in entries:
        title = entry.split("\n", 1)[0]
        assert "**Cause:**" in entry and "**Fix:**" in entry, f"{title!r} lacks a cause or a fix"


def test_glossary_defines_every_term() -> None:
    page = _page("glossary.md")
    terms = [
        "ACC",
        "Calibration",
        "Case",
        "Flaky case",
        "Gate",
        "Gold label",
        "Judge",
        "MDE",
        "Majority flip",
        "McNemar",
        "pass@k",
        "pass^k",
        "PPI",
        "Readiness",
        "Run",
        "Sign-flip test",
        "Suite",
        "Trial",
        "Variant",
    ]
    missing = [term for term in terms if f"| {term} |" not in page]
    assert not missing, f"glossary terms with no row: {missing}"
