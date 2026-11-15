"""``evalproof draft``: inputs for what a scan found untested. Every repository is built in
``tmp_path``; the model is a scripted fake and the network is never reached."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from evalproof import experimental
from evalproof.cli import main
from evalproof.config import ConfigError, GenerationParams, TargetConfig
from evalproof.dataset import load_dataset
from evalproof.draft import DraftOptions, draft_id, run_draft
from evalproof.draft.ask import (
    Request,
    ask,
    build_messages,
    key_of,
    parse_reply,
    strip_fence,
)
from evalproof.draft.pytests import harvest
from evalproof.draft.scanfile import load_scan
from evalproof.providers import Message, ProviderError
from evalproof.scan import report_json, scan_path
from tests.conftest import ScriptedProvider
from tests.test_scan import CHECKS, LLM, MULTI, TOOLS, cases, shop, suite, write

Fake = tuple[ScriptedProvider, list[TargetConfig]]
Replies = str | list[str] | Callable[[list[Message]], str]
Install = Callable[[Replies], Fake]


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> Install:
    """Replace the provider ``draft`` builds with a scripted one, and record the targets."""

    def install(replies: Replies) -> Fake:
        provider = ScriptedProvider([replies] if isinstance(replies, str) else replies)
        targets: list[TargetConfig] = []

        def make(target: TargetConfig) -> ScriptedProvider:
            targets.append(target)
            return provider

        monkeypatch.setattr("evalproof.draft.core.make_provider", make)
        return provider, targets

    return install


@pytest.fixture(autouse=True)
def no_real_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """A test that forgets to install a fake fails instead of reaching a model."""

    def refuse(target: TargetConfig) -> ScriptedProvider:
        raise AssertionError(f"a real provider was built for {target.target}")

    monkeypatch.setattr("evalproof.draft.core.make_provider", refuse)


def scan_file(root: Path, where: Path) -> Path:
    path = where / "scan.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report_json(scan_path(root, scan_ref=str(path))), encoding="utf-8")
    return path


def repo(tmp_path: Path, **files: str | None) -> tuple[Path, Path]:
    """A shop repository under ``tmp_path/repo`` and its scan file under ``tmp_path/out``."""
    root = shop(tmp_path / "repo", **files)
    return root, scan_file(root, tmp_path / "out")


def options(scan: Path, out: Path, **extra: object) -> DraftOptions:
    return DraftOptions(scan=scan, out=out, **extra)  # type: ignore[arg-type]


def rows(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def reply(*inputs: object) -> str:
    return json.dumps(list(inputs))


TOOL_REPO = {
    "app/tools.py": TOOLS,
    "evals/support.yaml": suite("python:app.tools:x", CHECKS),
}


# --- harvesting pytest files -------------------------------------------------------------------

IMPORTS = {
    "answer": "app/llm.py::answer",
    "app.llm": "app/llm.py",
    "llm": "app/llm.py",
}


def harvested(tmp_path: Path, body: str, imports: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    write(tmp_path, {"tests/test_llm.py": body})
    return harvest(tmp_path, "tests/test_llm.py", IMPORTS if imports is None else imports)


def test_an_exact_assertion_on_a_direct_call_is_a_reference(tmp_path: Path) -> None:
    found = harvested(tmp_path, 'def test_a():\n    assert answer("Refund window?") == "30 days"\n')
    [case] = found.cases
    assert (case.input, case.reference, case.source) == (
        "Refund window?",
        "30 days",
        "tests/test_llm.py::test_a:2",
    )
    assert case.fields() == {"reference": "30 days"}
    assert found.tests == 1


def test_an_assertion_on_a_variable_and_a_reversed_comparison(tmp_path: Path) -> None:
    body = (
        "def test_a():\n"
        '    result = answer("one")\n'
        '    assert result == "1"\n'
        "def test_b():\n"
        '    assert "2" == answer("two")\n'
    )
    got = {c.input: c.reference for c in harvested(tmp_path, body).cases}
    assert got == {"one": "1", "two": "2"}


def test_in_is_accept_and_several_are_merged_for_one_input(tmp_path: Path) -> None:
    body = (
        "def test_a():\n"
        '    r = answer("q")\n'
        '    assert "30 days" in r\n'
        '    assert "refund" in r and "30 days" in r\n'
    )
    [case] = harvested(tmp_path, body).cases
    assert case.accept == ["30 days", "refund"]
    assert case.fields() == {"accept": ["30 days", "refund"]}


def test_reshaping_methods_do_not_change_the_expected_text(tmp_path: Path) -> None:
    body = (
        "def test_a():\n"
        '    assert answer("q").lower() == "yes"\n'
        '    assert "ok" in answer("r").strip().upper()\n'
    )
    got = {c.input: c.fields() for c in harvested(tmp_path, body).cases}
    assert got == {"q": {"reference": "yes"}, "r": {"accept": ["ok"]}}


def test_re_search_is_a_pattern_and_keeps_its_flags(tmp_path: Path) -> None:
    body = (
        "import re\n"
        "def test_a():\n"
        '    assert re.search(r"\\d+ days", answer("a"))\n'
        '    r = answer("b")\n'
        '    assert re.search(r"days", r, re.I | re.M) is not None\n'
        '    assert re.search(r"days", r, re.IGNORECASE) \n'
        '    assert re.search(r"x", answer("c"), flags_unknown)\n'
        '    assert re.match(r"x", answer("d"))\n'
    )
    got = {c.input: c.pattern for c in harvested(tmp_path, body).cases}
    assert got == {"a": "\\d+ days", "b": "(?im)days"}


def test_the_input_is_the_first_literal_or_the_named_keyword(tmp_path: Path) -> None:
    body = (
        "def test_a():\n"
        '    assert answer(question="named") == "1"\n'
        '    assert answer("first", "second") == "2"\n'
        '    assert answer(mode="x") == "3"\n'
        '    assert answer(mode="x", other="y") == "4"\n'
        '    assert answer(user, ) == "5"\n'
        '    assert answer(f"{user}") == "6"\n'
        '    assert answer(build("z")) == "7"\n'
    )
    got = {c.input: c.reference for c in harvested(tmp_path, body).cases}
    assert got == {"named": "1", "first": "2", "x": "3"}


def test_module_calls_and_awaits_are_followed(tmp_path: Path) -> None:
    body = (
        "async def test_a():\n"
        '    assert await llm.answer("one") == "1"\n'
        '    assert app.llm.answer("two") == "2"\n'
        '    x = await answer("three")\n'
        '    assert x == "3"\n'
    )
    got = {c.input: c.reference for c in harvested(tmp_path, body).cases}
    assert got == {"one": "1", "two": "2", "three": "3"}


def test_a_call_to_something_else_or_an_assertion_that_is_not_a_shape_is_ignored(
    tmp_path: Path,
) -> None:
    body = (
        "def test_a():\n"
        '    assert helper("x") == "1"\n'
        '    assert answer("q") != "2"\n'
        '    assert len(answer("r")) > 3\n'
        '    assert answer("s") == other\n'
        '    assert answer("t") in ("a", "b")\n'
        '    assert 1 < answer("u") < 3\n'
        '    assert answer("v") == "1" or answer("v") == "2"\n'
        "def helper_not_a_test():\n"
        '    assert answer("w") == "1"\n'
    )
    assert harvested(tmp_path, body).cases == []


def test_a_name_assigned_after_the_assertion_is_not_used(tmp_path: Path) -> None:
    body = 'def test_a():\n    assert r == "1"\n    r = answer("late")\n'
    assert harvested(tmp_path, body).cases == []


def test_two_different_exact_answers_keep_the_first_and_warn(tmp_path: Path) -> None:
    body = (
        "def test_a():\n"
        '    assert answer("q") == "1"\n'
        "def test_b():\n"
        '    assert answer("q") == "2"\n'
        '    assert answer("q") == "1"\n'
    )
    found = harvested(tmp_path, body)
    [case] = found.cases
    assert (case.reference, case.source) == ("1", "tests/test_llm.py::test_a:2")
    assert len(found.warnings) == 1 and "two different exact answers" in found.warnings[0]


def test_a_test_file_that_does_not_parse_is_skipped_with_a_warning(tmp_path: Path) -> None:
    found = harvested(tmp_path, "def (:\n")
    assert found.cases == [] and "could not be read as Python" in found.warnings[0]
    gone = harvest(tmp_path, "tests/missing.py", IMPORTS)
    assert gone.cases == [] and "tests/missing.py" in gone.warnings[0]


def test_a_test_in_a_class_is_read_too(tmp_path: Path) -> None:
    body = 'class TestX:\n    def test_a(self):\n        assert answer("q") == "1"\n'
    [case] = harvested(tmp_path, body).cases
    assert case.source == "tests/test_llm.py::test_a:3"


# --- asking a model ----------------------------------------------------------------------------


def test_parse_reply_takes_a_json_array_of_strings() -> None:
    assert parse_reply('["a", " b ", ""]', multi_turn=False) == [["a"], ["b"]]
    assert parse_reply('```json\n["a"]\n```', multi_turn=False) == [["a"]]
    assert parse_reply('[["a", "b", ""], ["only one"]]', multi_turn=True) == [["a", "b"]]
    for bad in ("", "no", "{}", "[]", "[1]", '[["a"]]', '["a", ["b"]]', "[", "null"):
        assert parse_reply(bad, multi_turn=False) is None, bad
    assert parse_reply('["a"]', multi_turn=True) is None
    assert parse_reply("[" * 100000, multi_turn=False) is None


def test_strip_fence_only_removes_a_whole_fence() -> None:
    assert strip_fence('```\n["a"]\n```') == '["a"]'
    assert strip_fence('text ```["a"]```') == 'text ```["a"]```'


def test_key_of_ignores_case_spacing_and_a_final_stop() -> None:
    assert key_of(["What  is X?"]) == key_of(["what is x"])
    assert key_of(["a", "b"]) != key_of(["a b"])


def test_the_prompt_asks_for_inputs_only_and_shows_a_few_existing_ones() -> None:
    request = Request("few_cases", "few_cases s", "Brief here.", 4)
    system, user = build_messages(request, [f"existing {i}" for i in range(30)])
    assert system.role == "system" and "never write the application's answers" in system.content
    assert "Brief here." in user.content and "array of 4 strings" in user.content
    assert "- existing 0" in user.content and "- existing 11" in user.content
    assert "existing 12" not in user.content
    multi = build_messages(Request("x", "x", "b", 2, multi_turn=True), [])[1].content
    assert "2 conversations" in multi and "Inputs the test set already has" not in multi


def test_ask_retries_once_with_the_bad_reply_in_the_conversation() -> None:
    provider = ScriptedProvider(["not json", '["good"]'])
    parsed, problem = ask(provider, GenerationParams(), Request("g", "g", "b", 1), [])
    assert (parsed, problem) == ([["good"]], None)
    assert len(provider.calls) == 2
    retry = provider.calls[1]
    assert [m.role for m in retry] == ["system", "user", "assistant", "user"]
    assert retry[2].content == "not json"


def test_ask_gives_up_after_one_retry() -> None:
    provider = ScriptedProvider(["nope"])
    parsed, problem = ask(provider, GenerationParams(), Request("g", "g", "b", 1), [])
    assert parsed is None and "not a JSON array" in str(problem)
    assert len(provider.calls) == 2


def test_ask_gives_the_reply_room_for_the_inputs_asked_for() -> None:
    seen: list[int] = []

    class Spy(ScriptedProvider):
        def chat(self, messages: list[Message], params: GenerationParams):  # type: ignore[no-untyped-def,override]
            seen.append(params.max_tokens)
            return super().chat(messages, params)

    ask(Spy(['["a"]']), GenerationParams(max_tokens=50), Request("g", "g", "b", 1), [])
    ask(Spy(['["a"]']), GenerationParams(max_tokens=50), Request("g", "g", "b", 30), [])
    assert seen[0] >= 1024 and seen[1] >= 160 * 30


# --- drafting from tests, no model -------------------------------------------------------------

PYTEST_REPO = {
    "tests/test_llm.py": (
        "import re\n"
        "from app.llm import answer\n"
        "def test_window():\n"
        '    assert answer("What is the refund window?") == "30 days"\n'
        "def test_contact():\n"
        '    r = answer("How do I reach you?")\n'
        '    assert "email" in r and re.search(r"[a-z]+@", r)\n'
        "def test_known():\n"
        '    assert answer("question 0") == "x"\n'
    )
}


def test_tests_become_draft_cases_with_the_assertion_they_made(tmp_path: Path) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    out = tmp_path / "drafts" / "d.jsonl"
    result = run_draft(options(scan, out, from_tests_only=True))
    assert result.from_pytest == 2 and result.wrote == out
    assert [n for n in result.notes if "question 0" in n or "already in a suite" in n]
    by_input = {r["input"]: r for r in rows(out)}
    assert set(by_input) == {"What is the refund window?", "How do I reach you?"}
    window = by_input["What is the refund window?"]
    assert window["reference"] == "30 days"
    assert window["tags"] == ["draft", "gap:test_cases_not_imported", "from:pytest"]
    assert window["source"] == "tests/test_llm.py::test_window:4"
    assert window["draft_for"] == {"gap": "test_cases_not_imported", "path": "tests/test_llm.py"}
    assert window["id"] == draft_id(["What is the refund window?"])
    contact = by_input["How do I reach you?"]
    assert contact["accept"] == ["email"] and contact["pattern"] == "[a-z]+@"
    assert "drafted_by" not in window


def test_drafts_from_tests_load_as_a_dataset(tmp_path: Path) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    out = tmp_path / "d.jsonl"
    run_draft(options(scan, out, from_tests_only=True))
    loaded = load_dataset(out)
    assert {c.id for c in loaded} == {r["id"] for r in rows(out)}
    case = next(c for c in loaded if c.input == "What is the refund window?")
    assert case.reference == "30 days" and "draft" in case.tags


def test_from_tests_only_never_builds_a_provider(tmp_path: Path) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    # the autouse fixture makes make_provider raise, so reaching it would fail this test
    out = tmp_path / "d.jsonl"
    assert main(["draft", "--scan", str(scan), "--out", str(out), "--from-tests-only"]) == 0
    result = run_draft(options(scan, tmp_path / "e.jsonl", from_tests_only=True))
    assert result.drafted_by is None and len(result.cases) == 2


def test_ids_are_a_hash_of_the_input_and_its_turns() -> None:
    assert draft_id(["a"]) == draft_id(["a"])
    assert draft_id(["a"]) != draft_id(["a", "b"])
    assert draft_id(["a"]).startswith("draft-") and len(draft_id(["a"])) == len("draft-") + 10


def test_max_cases_caps_the_pytest_drafts(tmp_path: Path) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    result = run_draft(options(scan, tmp_path / "d.jsonl", from_tests_only=True, max_cases=1))
    assert len(result.cases) == 1
    assert any("--max-cases" in n for n in result.notes)


def test_a_pytest_file_the_gap_filter_excludes_is_not_read(tmp_path: Path) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    result = run_draft(
        options(scan, tmp_path / "d.jsonl", from_tests_only=True, files=["tests/other.py"])
    )
    assert result.cases == [] and result.wrote is None
    result = run_draft(
        options(scan, tmp_path / "d.jsonl", from_tests_only=True, files=["tests/test_llm.py"])
    )
    assert len(result.cases) == 2


# --- drafting with a model ---------------------------------------------------------------------


def test_a_tool_gap_gets_inputs_and_no_expected_answer(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    provider, targets = fake(reply("Please refund order 17", "I want my money back", "refund?"))
    out = tmp_path / "d.jsonl"
    result = run_draft(
        options(scan, out, target="ollama:qwen3", gaps=["tool_unchecked"], per_gap=3)
    )
    assert [t.target for t in targets] == ["ollama:qwen3"]
    assert result.by_gap == {"tool_unchecked refund": 3} and result.drafted_by == "ollama:qwen3"
    prompt = provider.calls[0][1].content
    assert "`refund`" in prompt and "Refund it." in prompt and "order_id: str" in prompt
    got = rows(out)
    assert [r["input"] for r in got] == [
        "Please refund order 17",
        "I want my money back",
        "refund?",
    ]
    for row in got:
        assert row["tags"] == ["draft", "gap:tool_unchecked", "from:model"]
        assert row["draft_for"] == {"gap": "tool_unchecked", "tool": "refund"}
        assert row["drafted_by"] == "ollama:qwen3"
        for forbidden in ("reference", "accept", "require", "pattern", "schema", "expect"):
            assert forbidden not in row
    assert {c.id for c in load_dataset(out)} == {r["id"] for r in got}


def test_a_multi_turn_gap_writes_the_later_messages_as_turns(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **{"app/chat.py": MULTI})
    fake(json.dumps([["Hi", "Actually, never mind", "Thanks"], ["Where is my order?", "And now?"]]))
    out = tmp_path / "d.jsonl"
    run_draft(options(scan, out, target="m", gaps=["no_multi_turn_cases"], per_gap=2))
    got = rows(out)
    assert got[0]["input"] == "Hi" and got[0]["turns"] == ["Actually, never mind", "Thanks"]
    assert got[1]["turns"] == ["And now?"]
    assert got[0]["tags"] == ["draft", "gap:no_multi_turn_cases", "from:model"]
    assert got[0]["draft_for"] == {"gap": "no_multi_turn_cases", "path": "app/chat.py", "line": 3}
    assert all(c.turns for c in load_dataset(out))


def test_an_uncovered_file_prompt_carries_the_function_it_found(
    tmp_path: Path, fake: Install
) -> None:
    code = (
        "import anthropic\n"
        "client = anthropic.Anthropic()\n"
        "def summarize(text: str, words: int = 50) -> str:\n"
        '    """Summarize a support ticket in few words."""\n'
        "    messages = [{'role': 'user', 'content': text}]\n"
        "    return client.messages.create(model='m', messages=messages)\n"
    )
    _, scan = repo(tmp_path, **{"jobs/batch.py": code})
    provider, _ = fake(reply("Summarize: my card was charged twice"))
    out = tmp_path / "d.jsonl"
    run_draft(options(scan, out, target="m", gaps=["uncovered_code"]))
    prompt = provider.calls[0][1].content
    assert "jobs/batch.py" in prompt
    assert "summarize(text: str, words: int=50) -> str" in prompt
    assert "Summarize a support ticket in few words." in prompt
    [row] = rows(out)
    assert row["draft_for"] == {"gap": "uncovered_code", "path": "jobs/batch.py", "line": 6}


def test_a_thin_suite_gets_inputs_and_the_prompt_shows_what_it_has(
    tmp_path: Path, fake: Install
) -> None:
    _, scan = repo(tmp_path)
    provider, _ = fake(reply("new one", "question 1", "Question 2.", "another"))
    out = tmp_path / "d.jsonl"
    result = run_draft(options(scan, out, target="m", gaps=["few_cases"], per_gap=4))
    prompt = provider.calls[0][1].content
    assert "- question 0" in prompt and "needs at least 20" in prompt
    assert [r["input"] for r in rows(out)] == ["new one", "another"]  # the other two exist
    assert rows(out)[0]["draft_for"] == {"gap": "few_cases", "suite": "evals/support.yaml"}
    assert result.by_gap == {"few_cases evals/support.yaml": 2}


def test_duplicates_empties_and_repeats_across_gaps_are_dropped(
    tmp_path: Path, fake: Install
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    fake(reply("Refund me", "refund ME.", "  ", "Another one", "Another one"))
    out = tmp_path / "d.jsonl"
    run_draft(options(scan, out, target="m", gaps=["tool_unchecked"], per_gap=5))
    assert [r["input"] for r in rows(out)] == ["Refund me", "Another one"]


def test_inputs_are_trimmed_to_per_gap_and_to_max_cases(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    fake(reply(*[f"input {i}" for i in range(10)]))
    out = tmp_path / "d.jsonl"
    result = run_draft(
        options(scan, out, target="m", gaps=["tool_unchecked", "few_cases"], per_gap=4, max_cases=6)
    )
    assert len(result.cases) == 6
    assert result.by_gap == {"tool_unchecked refund": 4, "few_cases evals/support.yaml": 2}
    again = run_draft(
        options(
            scan,
            tmp_path / "e.jsonl",
            target="m",
            gaps=["tool_unchecked", "few_cases"],
            per_gap=5,
            max_cases=5,
        )
    )
    assert len(again.cases) == 5
    assert any("stopped at --max-cases 5" in n for n in again.notes)


def test_a_malformed_reply_is_retried_then_the_gap_is_skipped(
    tmp_path: Path, fake: Install
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    provider, _ = fake(["Sure! Here are some ideas.", "still not JSON"])
    out = tmp_path / "d.jsonl"
    result = run_draft(options(scan, out, target="m", gaps=["tool_unchecked"]))
    assert len(provider.calls) == 2
    assert result.cases == [] and result.wrote is None and not out.exists()
    assert any("tool_unchecked refund: skipped" in w for w in result.warnings)


def test_a_malformed_reply_then_a_good_one_is_used(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    provider, _ = fake(["oops", reply("refund order 1")])
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", gaps=["tool_unchecked"]))
    assert len(provider.calls) == 2 and len(result.cases) == 1 and result.warnings == []


def test_a_reply_that_is_all_duplicates_warns(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path)
    fake(reply("question 0", "question 1"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", gaps=["few_cases"]))
    assert result.cases == []
    assert any("duplicate or empty" in w for w in result.warnings)


def test_one_gap_failing_does_not_stop_the_others(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)

    def script(messages: list[Message]) -> str:
        return "garbage" if "`refund`" in messages[1].content else reply("a new question")

    fake(script)
    result = run_draft(
        options(scan, tmp_path / "d.jsonl", target="m", gaps=["tool_unchecked", "few_cases"])
    )
    assert [c.tags[1] for c in result.cases] == ["gap:few_cases"]
    assert any("refund" in w for w in result.warnings)


def test_a_provider_error_is_not_swallowed(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)

    def boom(messages: list[Message]) -> str:
        raise ProviderError("the endpoint refused the key")

    fake(boom)
    with pytest.raises(ProviderError):
        run_draft(options(scan, tmp_path / "d.jsonl", target="m", gaps=["tool_unchecked"]))


def test_the_response_cache_serves_a_repeat_run_and_no_cache_skips_it(
    tmp_path: Path, fake: Install
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    provider, _ = fake(reply("refund order 1"))
    base = {"target": "m", "gaps": ["tool_unchecked"], "force": True}
    run_draft(options(scan, tmp_path / "a.jsonl", **base))
    run_draft(options(scan, tmp_path / "b.jsonl", **base))
    assert len(provider.calls) == 1
    run_draft(options(scan, tmp_path / "c.jsonl", no_cache=True, **base))
    assert len(provider.calls) == 2
    assert rows(tmp_path / "a.jsonl") == rows(tmp_path / "b.jsonl")


def test_the_cache_is_the_suites_when_there_is_one_and_the_working_directory_otherwise(
    tmp_path: Path, fake: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    fake(reply("refund order 1"))
    run_draft(options(scan, tmp_path / "a.jsonl", target="m", gaps=["tool_unchecked"]))
    assert (tmp_path / "repo" / "evals" / ".evalproof" / "cache").is_dir()
    bare = tmp_path / "bare"
    bare.mkdir()
    write(bare, {"app/chat.py": MULTI})
    bare_scan = scan_file(bare, tmp_path / "bare-out")
    monkeypatch.chdir(tmp_path)
    run_draft(options(bare_scan, tmp_path / "b.jsonl", target="m"))
    assert (tmp_path / ".evalproof" / "cache").is_dir()


# --- choosing the model ------------------------------------------------------------------------


def test_without_a_target_the_suites_judge_is_used(tmp_path: Path, fake: Install) -> None:
    judge = "judge:\n  target: ollama:judge-model\n"
    files = {
        "evals/support.yaml": suite("python:app.tools:x", CHECKS, judge),
        "app/tools.py": TOOLS,
    }
    _, scan = repo(tmp_path, **files)
    _, targets = fake(reply("refund order 2"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", gaps=["tool_unchecked"]))
    assert [t.target for t in targets] == ["ollama:judge-model"]
    assert result.drafted_by == "ollama:judge-model"


def test_target_wins_over_the_judge(tmp_path: Path, fake: Install) -> None:
    judge = "judge:\n  target: ollama:judge-model\n"
    files = {
        "evals/support.yaml": suite("python:app.tools:x", CHECKS, judge),
        "app/tools.py": TOOLS,
    }
    _, scan = repo(tmp_path, **files)
    _, targets = fake(reply("refund order 2"))
    run_draft(options(scan, tmp_path / "d.jsonl", target="ollama:mine", gaps=["tool_unchecked"]))
    assert [t.target for t in targets] == ["ollama:mine"]


def test_no_target_and_no_judge_is_a_usage_error_that_says_what_to_do(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    out = tmp_path / "d.jsonl"
    assert main(["draft", "--scan", str(scan), "--out", str(out), "--gap", "tool_unchecked"]) == 2
    err = capsys.readouterr().err
    assert "evalproof: error:" in err and "--target MODEL" in err and "--from-tests-only" in err
    assert not out.exists()


def test_two_different_judges_ask_for_a_choice(tmp_path: Path) -> None:
    other = suite("python:app.tools:x", CHECKS, "judge:\n  target: ollama:b\n").replace(
        "name: support", "name: other"
    )
    files = {
        **TOOL_REPO,
        "evals/support.yaml": suite("python:app.tools:x", CHECKS, "judge:\n  target: ollama:a\n"),
        "evals/other.yaml": other,
        "evals/other.jsonl": cases(1),
    }
    _, scan = repo(tmp_path, **files)
    with pytest.raises(ConfigError, match="different judges"):
        run_draft(options(scan, tmp_path / "d.jsonl", gaps=["tool_unchecked"]))


def test_suite_picks_one_suite_when_there_are_several(tmp_path: Path, fake: Install) -> None:
    other = suite("python:app.tools:x", CHECKS, "judge:\n  target: ollama:b\n").replace(
        "name: support", "name: other"
    )
    files = {
        **TOOL_REPO,
        "evals/support.yaml": suite("python:app.tools:x", CHECKS, "judge:\n  target: ollama:a\n"),
        "evals/other.yaml": other,
        "evals/other.jsonl": cases(1),
    }
    root, scan = repo(tmp_path, **files)
    _, targets = fake(reply("refund order 3"))
    run_draft(
        options(
            scan,
            tmp_path / "d.jsonl",
            suite=root / "evals" / "other.yaml",
            gaps=["tool_unchecked"],
        )
    )
    assert [t.target for t in targets] == ["ollama:b"]


def test_a_suite_that_cannot_be_loaded_is_an_error_when_named(tmp_path: Path) -> None:
    _, scan = repo(tmp_path)
    with pytest.raises(ConfigError):
        run_draft(options(scan, tmp_path / "d.jsonl", suite=tmp_path / "nope.yaml"))


# --- filters -----------------------------------------------------------------------------------


def test_tool_narrows_to_that_tool_and_implies_the_gap_kind(tmp_path: Path, fake: Install) -> None:
    root, scan = repo(tmp_path, **TOOL_REPO)
    (root / "evals" / "support.yaml").write_text(
        suite("python:app.tools:x", "  - name: t\n    type: checks\n"), encoding="utf-8"
    )
    scan = scan_file(root, tmp_path / "out2")
    provider, _ = fake(lambda m: reply(f"input for {m[1].content.split('`')[1]}"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", tools=["lookup_order"]))
    assert list(result.by_gap) == ["tool_unchecked lookup_order"]
    assert len(provider.calls) == 1


def test_file_narrows_to_that_file_and_accepts_a_path_from_the_root(
    tmp_path: Path, fake: Install
) -> None:
    code = LLM.replace("answer", "second")
    root, scan = repo(tmp_path, **{"jobs/a.py": LLM, "jobs/b.py": code})
    fake(reply("a question for b"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", files=["jobs/b.py"]))
    assert list(result.by_gap) == ["uncovered_code b.py"]
    result = run_draft(
        options(scan, tmp_path / "e.jsonl", target="m", files=[str(root / "jobs" / "a.py")])
    )
    assert list(result.by_gap) == ["uncovered_code a.py"]


def test_suite_narrows_few_cases(tmp_path: Path, fake: Install) -> None:
    root, scan = repo(tmp_path)
    provider, _ = fake(reply("a brand new question"))
    result = run_draft(
        options(scan, tmp_path / "d.jsonl", target="m", suite=root / "evals" / "support.yaml")
    )
    assert list(result.by_gap) == ["few_cases evals/support.yaml"]
    assert len(provider.calls) == 1


def test_a_suite_given_from_the_scanned_root_is_found(
    tmp_path: Path, fake: Install, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, scan = repo(tmp_path)
    fake(reply("a brand new question"))
    monkeypatch.chdir(tmp_path)
    result = run_draft(
        options(scan, tmp_path / "d.jsonl", target="m", suite=Path("evals/support.yaml"))
    )
    assert list(result.by_gap) == ["few_cases evals/support.yaml"]


def test_gap_names_the_kinds_to_draft_for(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **{**TOOL_REPO, **PYTEST_REPO})
    fake(reply("something new"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", gaps=["few_cases"]))
    assert result.from_pytest == 0 and list(result.by_gap) == ["few_cases evals/support.yaml"]


def test_with_no_filter_every_kind_is_drafted(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **{**TOOL_REPO, **PYTEST_REPO, "app/chat.py": MULTI})

    def script(messages: list[Message]) -> str:
        text = messages[1].content
        if "conversations" in text:
            return json.dumps([["hello there", "and one more thing"]])
        return reply("drafted for " + text.split("\n")[0][:30])

    fake(script)
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", per_gap=1))
    kinds = {c.tags[1] for c in result.cases}
    assert kinds == {
        "gap:test_cases_not_imported",
        "gap:tool_unchecked",
        "gap:no_multi_turn_cases",
        "gap:uncovered_code",
        "gap:few_cases",
    }
    assert result.from_pytest == 2


def test_a_gap_the_scan_did_not_find_drafts_nothing(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    write(root, {"notes.txt": "nothing here\n"})
    scan = scan_file(root, tmp_path / "out")
    out = tmp_path / "d.jsonl"
    result = run_draft(options(scan, out, from_tests_only=True))
    assert result.cases == [] and not out.exists()


# --- the output file ---------------------------------------------------------------------------


def test_an_existing_out_is_refused_before_any_model_call(tmp_path: Path, fake: Install) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    provider, _ = fake(reply("refund order 1"))
    out = tmp_path / "d.jsonl"
    out.write_text("keep me\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="--force"):
        run_draft(options(scan, out, target="m", gaps=["tool_unchecked"]))
    assert provider.calls == [] and out.read_text(encoding="utf-8") == "keep me\n"
    run_draft(options(scan, out, target="m", gaps=["tool_unchecked"], force=True))
    assert "refund order 1" in out.read_text(encoding="utf-8")


def test_an_out_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    _, scan = repo(tmp_path)
    with pytest.raises(ConfigError, match="directory"):
        run_draft(options(scan, tmp_path, force=True))


@pytest.mark.parametrize("flag", ["per_gap", "max_cases"])
def test_counts_must_be_positive(tmp_path: Path, flag: str) -> None:
    _, scan = repo(tmp_path)
    with pytest.raises(ConfigError, match="at least 1"):
        run_draft(options(scan, tmp_path / "d.jsonl", **{flag: 0}))


def test_suite_inputs_that_are_objects_get_a_warning(tmp_path: Path, fake: Install) -> None:
    rows_ = [{"id": f"c{i}", "input": {"q": f"object {i}"}, "reference": "x"} for i in range(2)]
    root, _ = repo(tmp_path)
    (root / "evals" / "support.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_) + "\n", encoding="utf-8"
    )
    scan = scan_file(root, tmp_path / "out2")
    fake(reply("something new"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", gaps=["few_cases"]))
    assert any("inputs are objects" in w for w in result.warnings)


def test_an_unreadable_dataset_warns_and_drafting_goes_on(tmp_path: Path, fake: Install) -> None:
    root, _ = repo(tmp_path)
    scan = scan_file(root, tmp_path / "out2")
    (root / "evals" / "support.jsonl").write_text("not json\n", encoding="utf-8")
    fake(reply("something new"))
    result = run_draft(options(scan, tmp_path / "d.jsonl", target="m", per_gap=1))
    assert any("dataset could not be read" in w for w in result.warnings)


# --- the scan file -----------------------------------------------------------------------------


def test_load_scan_reads_what_scan_wrote(tmp_path: Path) -> None:
    root, scan = repo(tmp_path, **TOOL_REPO)
    loaded = load_scan(scan)
    assert loaded.root == root
    assert {f["kind"] for f in loaded.findings} >= {"model_call", "tool"}
    assert loaded.finding_at("tool", "app/tools.py") is not None
    assert loaded.finding_at("tool", "nowhere.py") is None
    assert loaded.suites[0]["path"] == "evals/support.yaml"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("{ nope", "not valid JSON"),
        ("[]", "not an evalproof scan file"),
        ('{"hello": 1}', "not an evalproof scan file"),
        ('{"format_version": 2, "root": "."}', "format_version"),
        ('{"format_version": 1}', "'root' is missing"),
        ('{"format_version": 1, "root": "no-such-dir-here"}', "does not exist from here"),
        ('{"format_version": 1, "root": ".", "findings": 3}', "'findings'"),
        (
            '{"format_version": 1, "root": ".", "findings": [], "suites": [1], "gaps": []}',
            "'suites'",
        ),
    ],
)
def test_a_scan_file_that_cannot_be_used_says_why(tmp_path: Path, text: str, message: str) -> None:
    bad = tmp_path / "scan.json"
    bad.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_scan(bad)


def test_a_missing_scan_file_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the scan file"):
        load_scan(tmp_path / "missing.json")


# --- the command -------------------------------------------------------------------------------


def test_draft_prints_what_it_wrote_and_what_a_person_must_do(
    tmp_path: Path, fake: Install, capsys: pytest.CaptureFixture[str]
) -> None:
    _, scan = repo(tmp_path, **{**TOOL_REPO, **PYTEST_REPO})
    fake(reply("refund order 1", "refund order 2"))
    out = tmp_path / "drafts" / "d.jsonl"
    code = main(
        [
            *["draft", "--scan", str(scan), "--out", str(out), "--target", "ollama:qwen3"],
            *["--gap", "tool_unchecked", "--gap", "test_cases_not_imported"],
        ]
    )
    assert code == 0
    text = capsys.readouterr().out
    assert f"wrote 4 draft case(s) to {out}" in text
    assert "from pytest files: 2" in text
    assert "from a model (ollama:qwen3): 2 (tool_unchecked refund: 2)" in text
    assert "a person must add the expected answer or checks" in text
    assert out.is_file()


def test_draft_warnings_go_to_stderr(
    tmp_path: Path, fake: Install, capsys: pytest.CaptureFixture[str]
) -> None:
    _, scan = repo(tmp_path, **TOOL_REPO)
    fake(["no json", "no json"])
    code = main(
        [
            *["draft", "--scan", str(scan), "--out", str(tmp_path / "d.jsonl")],
            *["--target", "m", "--gap", "tool_unchecked"],
        ]
    )
    captured = capsys.readouterr()
    assert code == 0
    assert "evalproof: warning: tool_unchecked refund: skipped" in captured.err
    assert "nothing to draft, so no file was written" in captured.out


def test_draft_with_nothing_to_write_does_not_create_the_file_or_its_directory_gaps(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"
    write(root, {"notes.txt": "nothing here\n"})
    scan = scan_file(root, tmp_path / "out")
    out = tmp_path / "d.jsonl"
    assert main(["draft", "--scan", str(scan), "--out", str(out), "--from-tests-only"]) == 0
    assert not out.exists()
    assert "nothing to draft" in capsys.readouterr().out


@pytest.mark.parametrize(
    "extra",
    [["--per-gap", "0"], ["--max-cases", "0"], ["--scan", "missing.json"]],
)
def test_draft_usage_errors_exit_two(
    extra: list[str], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, scan = repo(tmp_path)
    argv = ["draft", "--scan", str(scan), "--out", str(tmp_path / "d.jsonl"), "--from-tests-only"]
    assert main(argv + extra) == 2
    assert "evalproof: error:" in capsys.readouterr().err


def test_draft_refuses_an_existing_out_and_force_overwrites(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _, scan = repo(tmp_path, **PYTEST_REPO)
    out = tmp_path / "d.jsonl"
    argv = ["draft", "--scan", str(scan), "--out", str(out), "--from-tests-only"]
    assert main(argv) == 0
    capsys.readouterr()
    assert main(argv) == 2
    assert "exists; pass --force" in capsys.readouterr().err
    assert main([*argv, "--force"]) == 0


def test_draft_needs_scan_and_out(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["draft", "--out", "x.jsonl"])
    assert stop.value.code == 2
    assert "--scan" in capsys.readouterr().err


def test_draft_rejects_an_unknown_gap_kind(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as stop:
        main(["draft", "--scan", "s.json", "--out", "o.jsonl", "--gap", "no_suite"])
    assert stop.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_draft_says_once_that_it_is_experimental(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(experimental, "_noticed", set())
    _, scan = repo(tmp_path, **PYTEST_REPO)
    argv = ["draft", "--scan", str(scan), "--out", str(tmp_path / "d.jsonl"), "--from-tests-only"]
    main([*argv, "--force"])
    first = capsys.readouterr().err
    main([*argv, "--force"])
    second = capsys.readouterr().err
    assert first.startswith("evalproof: draft is experimental: ")
    assert first.count("experimental") == 1
    assert second == ""


def test_the_scan_suggestions_run_as_written(
    tmp_path: Path,
    fake: Install,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The command a gap suggests is a command ``draft`` accepts."""
    import shlex

    root = tmp_path / "repo"
    shop(root, **TOOL_REPO)
    monkeypatch.chdir(root)
    assert main(["scan", ".", "--out", "scan.json"]) == 0
    gaps = json.loads(Path("scan.json").read_text(encoding="utf-8"))["gaps"]
    capsys.readouterr()
    fake(reply("refund order 9", "refund order 10"))
    ran = 0
    for gap in gaps:
        if gap["kind"] not in ("tool_unchecked", "few_cases", "uncovered_code"):
            continue
        argv = shlex.split(gap["suggest"])
        assert argv[:2] == ["evalproof", "draft"]
        argv = [*argv[1:], "--target", "m", "--force"]
        assert main(argv) == 0, gap["suggest"]
        ran += 1
    assert ran >= 2
