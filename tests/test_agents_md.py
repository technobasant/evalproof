"""``evalproof init`` and the evalproof section of a repository's ``AGENTS.md``.

The section sits between two markers. ``init`` replaces what is between them, appends the section
when the file has no markers, and never changes a byte outside them. These tests use no shell
scripts, so they run on Windows as well (tests/test_init.py does not).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pytest

from evalproof.cli import build_parser, main
from evalproof.scaffold import (
    AGENTS_BEGIN,
    AGENTS_END,
    AGENTS_HEADING,
    agents_block,
    merge_agents_md,
    template,
)

SKILLS = (".agents/skills/evalproof/SKILL.md", ".claude/skills/evalproof/SKILL.md")
BLOCK = agents_block()


@pytest.fixture(autouse=True)
def empty_path(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch) -> None:
    """No claude, codex or ollama on PATH, so init picks its hosted default and probes nothing."""
    monkeypatch.setenv("PATH", str(tmp_path_factory.mktemp("empty-path")))


def init(directory: Path, capsys: pytest.CaptureFixture[str], *args: str) -> tuple[int, str, str]:
    code = main(["init", "--dir", str(directory), *args])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def files_under(directory: Path) -> set[str]:
    if not directory.exists():
        return set()
    return {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}


def crlf(text: str) -> str:
    return text.replace("\n", "\r\n")


def has_bare_lf(data: bytes) -> bool:
    return re.search(rb"(?<!\r)\n", data) is not None


# --- the template ------------------------------------------------------------------------


def test_the_template_is_one_marked_section() -> None:
    raw = template("AGENTS.md")
    assert raw.count(AGENTS_BEGIN) == 1 and raw.count(AGENTS_END) == 1
    assert raw.strip().startswith(AGENTS_BEGIN) and raw.strip().endswith(AGENTS_END)
    assert raw.index(AGENTS_BEGIN) < raw.index(AGENTS_END)
    assert raw.replace("\r\n", "\n").strip() == BLOCK
    assert "@@" not in BLOCK
    assert 150 <= len(BLOCK.splitlines()) <= 400


def test_the_template_names_only_real_commands_and_flags() -> None:
    """A command in a code span or block is a real subcommand, and its flags exist on it."""
    parser = build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    flags = {
        name: {o for a in p._actions for o in a.option_strings if o.startswith("--")}
        for name, p in sub.choices.items()
    }
    code = re.findall(r"```.*?```|`[^`\n]+`", BLOCK, flags=re.S)
    seen = 0
    for span in code:
        for line in span.splitlines() or [span]:
            for m in re.finditer(r"\bevalproof ([a-z][a-z-]*)((?: [^|;&\n`]*)?)", line):
                name, rest = m.group(1), m.group(2)
                assert name in sub.choices, f"{name!r} is not an evalproof command: {line!r}"
                seen += 1
                for flag in re.findall(r"(?<![\w-])(--[a-z][a-z-]*)", rest):
                    assert flag in flags[name], f"{flag} is not an option of `{name}`: {line!r}"
    assert seen > 20


# --- creating, replacing and appending ---------------------------------------------------


def test_init_creates_agents_md_with_a_heading_and_the_section(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = init(tmp_path, capsys)
    assert code == 0
    assert "  wrote          AGENTS.md" in out
    data = (tmp_path / "AGENTS.md").read_bytes()
    assert data.decode("utf-8") == f"{AGENTS_HEADING}\n\n{BLOCK}\n"
    assert b"\r" not in data


def test_a_second_run_changes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    init(tmp_path, capsys)
    before = (tmp_path / "AGENTS.md").read_bytes()
    code, out, _ = init(tmp_path, capsys)
    assert code == 0
    assert "  unchanged      AGENTS.md" in out
    assert (tmp_path / "AGENTS.md").read_bytes() == before


def test_the_section_between_the_markers_is_replaced_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    head = "# My repo\n\nRules that are mine.\n\n"
    tail = "\n\n## After\n\nMore of my rules.  \n\ttabbed\n"
    stale = f"{AGENTS_BEGIN}\nan older section\n{AGENTS_END}"
    (tmp_path / "AGENTS.md").write_bytes((head + stale + tail).encode("utf-8"))
    code, out, _ = init(tmp_path, capsys)
    assert code == 0
    assert "  updated        AGENTS.md" in out
    assert (tmp_path / "AGENTS.md").read_bytes().decode("utf-8") == head + BLOCK + tail
    # and the result is stable
    _, out, _ = init(tmp_path, capsys)
    assert "  unchanged      AGENTS.md" in out


@pytest.mark.parametrize("ending", ["", "\n", "\n\n", "\n\n\n"])
def test_a_file_without_markers_gets_the_section_appended_untouched(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], ending: str
) -> None:
    mine = "# Mine\n\nUse tabs\there.  \nTrailing spaces stay.   " + ending
    (tmp_path / "AGENTS.md").write_bytes(mine.encode("utf-8"))
    code, out, _ = init(tmp_path, capsys)
    assert code == 0
    assert "  appended       AGENTS.md" in out
    data = (tmp_path / "AGENTS.md").read_bytes()
    assert data.startswith(mine.encode("utf-8"))  # byte for byte
    appended = data[len(mine.encode("utf-8")) :].decode("utf-8")
    assert appended.endswith(BLOCK + "\n")
    # one blank line between the old content and the section
    gap = appended[: -len(BLOCK + "\n")]
    assert gap == "\n" * max(0, 2 - ending.count("\n"))
    assert (mine + gap).endswith("\n\n")


def test_a_crlf_file_stays_crlf_when_appended_to_and_when_replaced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = crlf("# Mine\n\nWindows rules.\n")
    path = tmp_path / "AGENTS.md"
    path.write_bytes(mine.encode("utf-8"))
    init(tmp_path, capsys)
    data = path.read_bytes()
    assert data.startswith(mine.encode("utf-8"))
    assert not has_bare_lf(data)
    assert data.decode("utf-8").endswith(crlf(BLOCK) + "\r\n")
    # Run again: unchanged. Then replace a stale section in a CRLF file.
    _, out, _ = init(tmp_path, capsys)
    assert "  unchanged      AGENTS.md" in out
    head, tail = crlf("# Mine\n\n"), crlf("\n\nAfter\n")
    path.write_bytes((head + AGENTS_BEGIN + crlf("\nold\n") + AGENTS_END + tail).encode("utf-8"))
    init(tmp_path, capsys)
    data = path.read_bytes()
    assert not has_bare_lf(data)
    assert data.decode("utf-8") == head + crlf(BLOCK) + tail


def test_a_file_with_lf_ends_stays_lf(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "AGENTS.md").write_bytes(b"# Mine\nunix\n")
    init(tmp_path, capsys)
    assert b"\r" not in (tmp_path / "AGENTS.md").read_bytes()


def test_force_never_overwrites_the_users_agents_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = "# Mine\n\nKeep this.\n"
    (tmp_path / "AGENTS.md").write_bytes(mine.encode("utf-8"))
    code, out, _ = init(tmp_path, capsys, "--force")
    assert code == 0
    assert "appended       AGENTS.md" in out
    assert (tmp_path / "AGENTS.md").read_bytes().startswith(mine.encode("utf-8"))
    # forced again: the section is current, so the file does not change
    before = (tmp_path / "AGENTS.md").read_bytes()
    init(tmp_path, capsys, "--force")
    assert (tmp_path / "AGENTS.md").read_bytes() == before


def test_merge_agents_md_without_a_file_or_with_an_empty_one() -> None:
    created = (f"{AGENTS_HEADING}\n\n{BLOCK}\n", "created")
    assert merge_agents_md(None) == created
    assert merge_agents_md("") == created


def test_no_agents_md_leaves_the_file_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = init(tmp_path / "fresh", capsys, "--no-agents-md")
    assert code == 0 and "AGENTS.md" not in out
    assert "AGENTS.md" not in files_under(tmp_path / "fresh")
    assert "evals/suite.yaml" in files_under(tmp_path / "fresh")
    mine = b"# Mine\n"
    (tmp_path / "AGENTS.md").write_bytes(mine)
    init(tmp_path, capsys, "--no-agents-md")
    assert (tmp_path / "AGENTS.md").read_bytes() == mine


# --- damaged markers and unreadable files ------------------------------------------------

DAMAGED = {
    "begin only": f"# Mine\n\n{AGENTS_BEGIN}\nhalf a section\n",
    "end only": f"# Mine\n\n{AGENTS_END}\n",
    "end before begin": f"{AGENTS_END}\nbackwards\n{AGENTS_BEGIN}\n",
    "two begins": f"{AGENTS_BEGIN}\na\n{AGENTS_BEGIN}\nb\n{AGENTS_END}\n",
    "two sections": f"{AGENTS_BEGIN}\na\n{AGENTS_END}\n{AGENTS_BEGIN}\nb\n{AGENTS_END}\n",
}


@pytest.mark.parametrize("extra", [[], ["--agent-files-only"]])
@pytest.mark.parametrize("case", sorted(DAMAGED))
def test_damaged_markers_stop_init_with_exit_2_and_nothing_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], case: str, extra: list[str]
) -> None:
    mine = DAMAGED[case].encode("utf-8")
    (tmp_path / "AGENTS.md").write_bytes(mine)
    code, out, err = init(tmp_path, capsys, *extra)
    assert code == 2
    assert "damaged evalproof markers" in err and "Nothing was written" in err
    assert "wrote" not in out
    assert (tmp_path / "AGENTS.md").read_bytes() == mine
    assert files_under(tmp_path) == {"AGENTS.md"}  # not even the suite


def test_a_file_that_is_not_utf8_is_refused_and_left_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = b"# Mine \xff\xfe not utf-8\n"
    (tmp_path / "AGENTS.md").write_bytes(mine)
    code, _, err = init(tmp_path, capsys)
    assert code == 2 and "UTF-8" in err
    assert (tmp_path / "AGENTS.md").read_bytes() == mine
    assert files_under(tmp_path) == {"AGENTS.md"}


# --- --agent-files-only ------------------------------------------------------------------


def test_agent_files_only_writes_the_section_and_the_skill_and_nothing_else(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out, _ = init(tmp_path, capsys, "--agent-files-only")
    assert code == 0
    assert files_under(tmp_path) == {"AGENTS.md", *SKILLS}
    assert "agent files only" in out and "judge" not in out and "next:" not in out
    for rel in ("AGENTS.md", *SKILLS):
        assert f"  wrote          {rel}" in out
    assert (tmp_path / "AGENTS.md").read_bytes().decode("utf-8").count(AGENTS_BEGIN) == 1


def test_agent_files_only_twice_is_stable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    init(tmp_path, capsys, "--agent-files-only")
    before = {p: (tmp_path / p).read_bytes() for p in files_under(tmp_path)}
    code, out, _ = init(tmp_path, capsys, "--agent-files-only")
    assert code == 0 and "  unchanged      AGENTS.md" in out
    assert out.count("kept (exists)") == 2 and "note:" not in out
    assert {p: (tmp_path / p).read_bytes() for p in files_under(tmp_path)} == before


def test_agent_files_only_with_no_skill_or_no_agents_md(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    init(tmp_path / "a", capsys, "--agent-files-only", "--no-skill")
    assert files_under(tmp_path / "a") == {"AGENTS.md"}
    init(tmp_path / "b", capsys, "--agent-files-only", "--no-agents-md")
    assert files_under(tmp_path / "b") == set(SKILLS)


def test_agent_files_only_keeps_a_changed_skill_unless_forced(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mine = tmp_path / SKILLS[0]
    mine.parent.mkdir(parents=True)
    mine.write_bytes(b"my edited skill\n")
    code, out, _ = init(tmp_path, capsys, "--agent-files-only")
    assert code == 0
    assert f"kept (exists)  {SKILLS[0]}" in out
    assert f"note: {SKILLS[0]} differs" in out and "--force" in out
    assert f"note: {SKILLS[1]}" not in out
    assert mine.read_bytes() == b"my edited skill\n"
    assert (tmp_path / SKILLS[1]).is_file()
    code, out, _ = init(tmp_path, capsys, "--agent-files-only", "--force")
    assert code == 0 and "note:" not in out
    assert mine.read_bytes().decode("utf-8") == template("SKILL.md")


def test_a_kept_skill_with_crlf_ends_is_not_reported_as_different(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for rel in SKILLS:
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(crlf(template("SKILL.md")).encode("utf-8"))
    _, out, _ = init(tmp_path, capsys, "--agent-files-only")
    assert "note:" not in out


@pytest.mark.parametrize(
    "args",
    [
        ["--judge", "claude-code:haiku"],
        ["--ci", "none"],
        ["--ci", "api-key"],
        ["--platform", "gitlab"],
        ["--cli-auth", "oauth-token"],
        ["--no-agents-md", "--no-skill"],
    ],
)
def test_agent_files_only_rejects_options_it_cannot_honour(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str]
) -> None:
    target = tmp_path / "proj"
    code, _, err = init(target, capsys, "--agent-files-only", *args)
    assert code == 2
    assert "--agent-files-only" in err
    assert files_under(target) == set()


def test_no_agents_md_and_no_skill_without_agent_files_only_is_fine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _, _ = init(tmp_path, capsys, "--no-agents-md", "--no-skill", "--ci", "none")
    assert code == 0
    assert "AGENTS.md" not in files_under(tmp_path)
    assert not any(rel in files_under(tmp_path) for rel in SKILLS)
    assert "evals/suite.yaml" in files_under(tmp_path)


def test_the_skill_points_to_the_agents_md_section() -> None:
    skill = template("SKILL.md")
    assert "AGENTS.md" in skill and "evalproof section" in skill
    assert len(skill.splitlines()) < 120
