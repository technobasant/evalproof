"""The ``command:`` system target: any program that reads a case on stdin and prints the answer.

The program is the running interpreter, so the tests need no other language and run on every
platform; the scripts they run are written next to the suite.
"""

from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, cast

import pytest

from evalproof import systems
from evalproof.cli import main
from evalproof.config import ConfigError, TargetConfig, command_argv, load_config
from evalproof.doctor import run_doctor
from evalproof.models import Case, SystemOutput
from evalproof.runner import load_run, run_suite, save_run
from evalproof.systems import (
    CommandSystem,
    HttpSystem,
    check_turns_supported,
    check_variant_vars,
    make_system,
    resolve_program,
)

SCRIPT = """\
import json
import os
import subprocess
import sys
import time

mode = sys.argv[1] if len(sys.argv) > 1 else "json"
raw = sys.stdin.buffer.read()
body = json.loads(raw.decode("utf-8"))


def say(text):
    sys.stdout.buffer.write(text.encode("utf-8"))


if mode == "json":
    say(
        json.dumps(
            {
                "output": str(body["input"]).upper(),
                "retrieved": ["doc-1"],
                "tokens_in": 12,
                "tokens_out": 3,
                "cost_usd": 0.002,
            }
        )
    )
elif mode == "text":
    say("  a plain answer \\n\\n")
elif mode == "echo":
    say(json.dumps({"output": json.dumps(body, sort_keys=True)}))
elif mode == "utf8":
    say(json.dumps({"output": body["input"] + " \\u2192 caf\\u00e9"}, ensure_ascii=False))
elif mode == "raw":
    sys.stdout.buffer.write(b"caf\\xe9")
elif mode == "cwd":
    say(os.getcwd())
elif mode == "env":
    say(os.environ.get("EVALPROOF_COMMAND_TEST", "unset"))
elif mode == "fail":
    sys.stderr.write("x" * 400 + "boom\\n")
    sys.exit(3)
elif mode == "fail-quiet":
    sys.exit(4)
elif mode == "fail-bytes":
    sys.stderr.buffer.write(b"bad \\xff\\xfe bytes")
    sys.exit(1)
elif mode == "sleep":
    time.sleep(60)
elif mode == "tree":
    # A child that keeps stdout and stderr open, as ``npx`` leaves ``node`` behind.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    sys.stdout.write("started\\n")
    sys.stdout.flush()
    child.wait()
elif mode == "object":
    say(json.dumps({"answer": 1}))
elif mode == "number":
    say("42\\n")
elif mode == "bad-tokens":
    say(json.dumps({"output": "x", "tokens_in": "many"}))
"""


def command(tmp_path: Path, mode: str = "", *, script: str = "sys_cmd.py") -> str:
    """A ``command:`` target that runs the test script under the running interpreter."""
    (tmp_path / "sys_cmd.py").write_text(SCRIPT, encoding="utf-8")
    exe = shlex.quote(Path(sys.executable).as_posix())
    return f"command:{exe} {script} {mode}".rstrip()


def system(
    tmp_path: Path,
    mode: str = "",
    *,
    timeout_s: float = 60.0,
    **kwargs: Any,
) -> CommandSystem:
    cfg = TargetConfig(target=command(tmp_path, mode), timeout_s=timeout_s)
    return CommandSystem(cfg, base_dir=str(tmp_path), **kwargs)


def run_case(sut: CommandSystem, case_input: str = "hello", **kwargs: Any) -> SystemOutput:
    return sut.run(Case(id="c1", input=case_input), **kwargs)


# -- the target string ----------------------------------------------------------------------------


def test_kind_and_argv_split_like_posix() -> None:
    cfg = TargetConfig(target="command:node eval/system.js --mode fast")
    assert cfg.kind == "command"
    assert command_argv(cfg.target) == ["node", "eval/system.js", "--mode", "fast"]
    assert command_argv("command:'my prog' \"two words\" a\\ b") == ["my prog", "two words", "a b"]


@pytest.mark.parametrize("text", ["", "   ", "\t"])
def test_an_empty_command_fails_when_the_suite_loads(tmp_path: Path, text: str) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ndataset: d.jsonl\n"
        f'system: {{target: "command:{text}"}}\n'
        "scorers: [{name: m, type: exact_match}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="command target is empty"):
        load_config(suite)


def test_a_command_with_an_unclosed_quote_fails_when_the_suite_loads(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ndataset: d.jsonl\n"
        "system: {target: 'command:node \"x.js'}\n"
        "scorers: [{name: m, type: exact_match}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="cannot be split into words"):
        load_config(suite)


def test_a_variant_cannot_swap_in_an_empty_command(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ndataset: d.jsonl\n"
        f'system: {{target: "{command(tmp_path)}"}}\n'
        "scorers: [{name: m, type: exact_match}]\n"
        "variants:\n  - {name: v, system: {target: 'command:'}}\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="command target is empty"):
        load_config(suite)


# -- resolving the program ------------------------------------------------------------------------


def test_an_unresolvable_program_is_a_config_error_naming_it(tmp_path: Path) -> None:
    cfg = TargetConfig(target="command:no-such-program-evalproof --flag")
    with pytest.raises(ConfigError, match="no-such-program-evalproof"):
        make_system(cfg, search_path=str(tmp_path))
    with pytest.raises(ConfigError, match="no-such-program-evalproof"):
        CommandSystem(cfg)


def test_a_missing_path_is_a_config_error_naming_it(tmp_path: Path) -> None:
    cfg = TargetConfig(target="command:tools/missing")
    with pytest.raises(ConfigError, match="tools/missing"):
        make_system(cfg, search_path=str(tmp_path))


def test_a_bare_name_is_found_on_path() -> None:
    exe = Path(sys.executable)
    resolved = resolve_program(exe.name)
    assert Path(resolved).is_absolute() and Path(resolved).name.lower().startswith("python")


def test_a_relative_path_is_resolved_against_the_suite_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tools = tmp_path / "suite" / "tools"
    tools.mkdir(parents=True)
    if sys.platform == "win32":
        program = tools / "answer.cmd"
        program.write_text("@echo off\r\necho from-the-script\r\n", encoding="utf-8")
        target = "command:tools/answer.cmd"
    else:
        program = tools / "answer"
        program.write_text("#!/bin/sh\ncat > /dev/null\necho from-the-script\n", encoding="utf-8")
        program.chmod(0o755)
        target = "command:tools/answer"
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # not the suite directory
    sut = make_system(TargetConfig(target=target), search_path=str(tmp_path / "suite"))
    out = sut.run(Case(id="c", input="q"))
    assert out.error is None and out.output == "from-the-script"
    assert Path(cast(CommandSystem, sut).argv[0]).samefile(program)


@pytest.mark.skipif(sys.platform == "win32", reason="needs an executable bit")
def test_a_file_that_is_not_executable_is_refused_at_construction(tmp_path: Path) -> None:
    program = tmp_path / "plain.txt"
    program.write_text("not a program\n", encoding="utf-8")
    program.chmod(0o644)
    with pytest.raises(ConfigError, match=r"plain\.txt"):
        make_system(TargetConfig(target="command:./plain.txt"), search_path=str(tmp_path))


def test_the_program_runs_in_the_suite_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite_dir = tmp_path / "suite"
    suite_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    out = run_case(system(suite_dir, "cwd"))
    assert out.error is None
    assert Path(out.output).resolve() == suite_dir.resolve()


# -- what the program receives and returns --------------------------------------------------------


def test_json_output_carries_tokens_cost_and_retrieved(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "json"), "hello")
    assert out.error is None
    assert out.output == "HELLO"
    assert out.retrieved == ["doc-1"]
    assert (out.tokens_in, out.tokens_out, out.cost_usd) == (12, 3, 0.002)
    assert out.latency_ms > 0


def test_pricing_prices_a_result_that_reports_no_cost(tmp_path: Path) -> None:
    cfg = TargetConfig(
        target=command(tmp_path, "json"),
        pricing={"input_per_mtok": 1.0, "output_per_mtok": 1.0},  # type: ignore[arg-type]
    )
    out = run_case(CommandSystem(cfg, base_dir=str(tmp_path)))
    assert out.cost_usd == 0.002  # the program's own figure wins
    cfg = cfg.model_copy(update={"target": command(tmp_path, "text")})
    assert run_case(CommandSystem(cfg, base_dir=str(tmp_path))).cost_usd == 0.0


def test_plain_text_output_is_stripped(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "text"))
    assert out.error is None and out.output == "a plain answer"
    assert out.tokens_in == 0 and out.tokens_out == 0


def test_text_that_is_json_but_not_an_object_stays_text(tmp_path: Path) -> None:
    assert run_case(system(tmp_path, "number")).output == "42"


def test_a_json_object_without_output_is_an_error_on_the_case(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "object"))
    assert out.output == "" and out.error is not None and "'output' key" in out.error


def test_a_bad_figure_is_an_error_on_the_case(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "bad-tokens"))
    assert (
        out.output == "" and out.error is not None and "'tokens_in' must be a number" in out.error
    )


def test_stdin_holds_id_and_input_and_nothing_else_for_a_plain_case(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "echo"), "what is 2+2?")
    assert json.loads(out.output) == {"id": "c1", "input": "what is 2+2?"}


def test_stdin_holds_turns_state_variant_vars_and_trial(tmp_path: Path) -> None:
    sut = system(tmp_path, "echo", variant="terse", vars={"tone": "dry", "n": 2}, trials=3)
    case = Case(
        id="chat-1",
        input="hi",
        turns=["and then?", {"role": "user", "content": "thanks"}],
        state={"cart": ["a"]},
    )
    out = sut.run(case, trial=2)
    assert json.loads(out.output) == {
        "id": "chat-1",
        "input": "hi",
        "turns": ["and then?", {"role": "user", "content": "thanks"}],
        "state": {"cart": ["a"]},
        "variant": "terse",
        "vars": {"n": 2, "tone": "dry"},
        "trial": 2,
    }


def test_trial_is_sent_only_when_cases_run_more_than_once(tmp_path: Path) -> None:
    once = json.loads(run_case(system(tmp_path, "echo", trials=1), trial=0).output)
    assert "trial" not in once
    many = json.loads(run_case(system(tmp_path, "echo", trials=2), trial=0).output)
    assert many["trial"] == 0


def test_stdin_is_the_same_body_an_http_system_posts(tmp_path: Path) -> None:
    case = Case(id="c9", input={"q": "x"}, turns=["t"], state={"s": 1})
    kwargs: dict[str, Any] = {"variant": "v", "vars": {"a": 1}, "trials": 2}
    sent = json.loads(system(tmp_path, "echo", **kwargs).run(case, trial=1).output)
    http = HttpSystem(TargetConfig(target="http://localhost:1/x"), **kwargs)
    try:
        assert sent == http._body(case, 1)
    finally:
        http.close()


def test_non_ascii_text_survives_the_round_trip(tmp_path: Path) -> None:
    text = "héllo 日本語 ✓ \U0001f600"
    out = run_case(system(tmp_path, "utf8"), text)
    assert out.error is None
    assert out.output == text + " → café"
    echoed = json.loads(run_case(system(tmp_path, "echo"), text).output)
    assert echoed["input"] == text


def test_stdout_that_is_not_utf8_is_decoded_with_replacement(tmp_path: Path) -> None:
    assert run_case(system(tmp_path, "raw")).output == "caf�"


def test_the_program_inherits_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EVALPROOF_COMMAND_TEST", "from-the-parent")
    assert run_case(system(tmp_path, "env")).output == "from-the-parent"


# -- errors and timeouts --------------------------------------------------------------------------


def test_a_non_zero_exit_records_the_stderr_tail(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "fail"))
    assert out.output == ""
    assert out.error is not None and out.error.startswith("command exited with code 3: ")
    tail = out.error.removeprefix("command exited with code 3: ")
    assert len(tail) == 300 and tail.endswith("boom") and set(tail[:-4]) == {"x"}


def test_a_non_zero_exit_with_no_stderr_names_only_the_code(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "fail-quiet"))
    assert out.error == "command exited with code 4"


def test_stderr_that_is_not_utf8_does_not_break_the_error(tmp_path: Path) -> None:
    out = run_case(system(tmp_path, "fail-bytes"))
    assert out.error is not None and "bad �� bytes" in out.error


def test_a_timeout_kills_the_process_and_records_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    procs: list[subprocess.Popen[bytes]] = []
    real = systems.kill_tree

    def spy(proc: subprocess.Popen[bytes]) -> None:
        procs.append(proc)
        real(proc)

    monkeypatch.setattr(systems, "kill_tree", spy)
    sut = system(tmp_path, "sleep", timeout_s=0.5)
    start = time.perf_counter()
    out = run_case(sut)
    assert time.perf_counter() - start < 30
    assert out.output == "" and out.error == "command timed out after 0.5s"
    assert len(procs) == 1 and procs[0].returncode is not None  # reaped, not a zombie


def test_a_timeout_kills_the_children_that_hold_the_pipes(tmp_path: Path) -> None:
    """Without a tree kill the child outlives its parent and reading the pipes never ends; the
    reap step gives up after five seconds, so a quick return means the child was killed."""
    sut = system(tmp_path, "tree", timeout_s=1.5)
    start = time.perf_counter()
    out = run_case(sut)
    elapsed = time.perf_counter() - start
    assert out.error == "command timed out after 1.5s"
    assert elapsed < 4.5, f"took {elapsed:.1f}s: the child was left running"


def test_an_error_does_not_stop_the_run(tmp_path: Path) -> None:
    sut = system(tmp_path, "fail")
    outs = [sut.run(Case(id=f"c{i}", input="q")) for i in range(3)]
    assert [o.error is not None for o in outs] == [True, True, True]


def test_a_program_that_cannot_start_is_an_error_on_the_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sut = system(tmp_path, "json")

    def refuse(*args: Any, **kwargs: Any) -> subprocess.Popen[bytes]:
        raise PermissionError("[Errno 13] Permission denied")

    monkeypatch.setattr(systems.subprocess, "Popen", refuse)
    out = run_case(sut)
    assert out.output == ""
    assert out.error is not None and "Permission denied" in out.error
    assert Path(sut.argv[0]).name in out.error


def test_an_interrupt_kills_the_process_and_goes_on_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    started: list[subprocess.Popen[bytes]] = []

    def interrupt(self: subprocess.Popen[bytes], *args: Any, **kwargs: Any) -> Any:
        started.append(self)
        raise KeyboardInterrupt

    monkeypatch.setattr(subprocess.Popen, "communicate", interrupt)
    with pytest.raises(KeyboardInterrupt):
        run_case(system(tmp_path, "sleep"))
    assert started[0].returncode is not None  # killed and reaped


# -- where a command target is and is not special -------------------------------------------------


def write_suite(tmp_path: Path, target: str, *, extra: str = "") -> Path:
    rows = [
        {"id": "a", "input": "alpha", "reference": "ALPHA"},
        {"id": "b", "input": "beta", "reference": "BETA"},
        {"id": "c", "input": "gamma", "reference": "NOT GAMMA"},
    ]
    (tmp_path / "cases.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: command-suite\n"
        "dataset: cases.jsonl\n"
        f"system: {{target: {json.dumps(target)}, timeout_s: 30}}\n"
        "scorers: [{name: exact, type: exact_match}]\n"
        f"{extra}",
        encoding="utf-8",
    )
    return suite


def test_a_suite_runs_end_to_end_and_writes_a_run_file(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, command(tmp_path, "json"), extra="concurrency: 2\n")
    run = run_suite(load_config(suite), use_cache=False)
    assert [r.case_id for r in run.results] == ["a", "b", "c"]
    assert [r.scores["exact"].passed for r in run.results] == [True, True, False]
    assert all(r.output.error is None for r in run.results)
    assert sum(r.output.tokens_in for r in run.results) == 36
    assert run.info.system == command(tmp_path, "json")
    saved = load_run(save_run(run, tmp_path / "runs" / "r.json"))
    assert [r.output.output for r in saved.results] == ["ALPHA", "BETA", "GAMMA"]


def test_a_suite_with_repeated_trials_sends_the_trial(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, command(tmp_path, "echo"), extra="trials: 2\n")
    run = run_suite(load_config(suite), use_cache=False)
    assert len(run.results) == 6
    sent = [json.loads(r.output.output) for r in run.results]
    assert sorted((s["id"], s["trial"]) for s in sent) == [
        (case, trial) for case in "abc" for trial in (0, 1)
    ]


def test_a_variant_reaches_the_command_in_its_vars(tmp_path: Path) -> None:
    suite = write_suite(
        tmp_path,
        command(tmp_path, "echo"),
        extra="variants:\n  - {name: terse, vars: {tone: dry, input: clash}}\n",
    )
    run = run_suite(load_config(suite), use_cache=False, variant="terse")
    sent = json.loads(run.results[0].output.output)
    # ``input`` is both a var and a case field: fine here, they travel apart.
    assert sent["variant"] == "terse" and sent["vars"] == {"input": "clash", "tone": "dry"}


def test_a_command_variant_system_override(tmp_path: Path) -> None:
    loud = command(tmp_path, "json")
    suite = write_suite(
        tmp_path,
        command(tmp_path, "text"),
        extra=f"variants:\n  - {{name: loud, system: {{target: {json.dumps(loud)}}}}}\n",
    )
    run = run_suite(load_config(suite), use_cache=False, variant="loud", limit=1)
    assert run.results[0].output.output == "ALPHA"


def test_cache_only_refuses_a_command_system(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, command(tmp_path, "json"))
    with pytest.raises(ConfigError, match="--cache-only needs a model system: a command system"):
        run_suite(load_config(suite), cache_only=True)
    out = tmp_path / "out.json"
    assert main(["run", str(suite), "--cache-only", "--out", str(out), "--quiet"]) == 2
    assert not out.exists()


def test_the_cli_runs_a_command_suite(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, command(tmp_path, "json"))
    out = tmp_path / "run.json"
    code = main(["run", str(suite), "--out", str(out), "--no-cache", "--quiet"])
    assert code == 0 and out.exists()
    assert len(load_run(out).results) == 3


def test_a_multi_turn_case_and_variant_vars_are_accepted() -> None:
    cfg = TargetConfig(target="command:prog")
    case = Case(id="c", input="hi", turns=["more"])
    check_turns_supported(cfg, [case])
    check_variant_vars(cfg, "v", {"input": "clash"}, [case])


# -- evalproof doctor -----------------------------------------------------------------------------


def doctor(suite: Path) -> tuple[int, str]:
    out = io.StringIO()
    return run_doctor(str(suite), out), out.getvalue()


def test_doctor_shows_the_program_and_that_no_credential_is_used(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, command(tmp_path, "json"))
    code, text = doctor(suite)
    assert code == 0
    assert f"{command(tmp_path, 'json')} (system)" in text
    (program_line,) = [ln for ln in text.splitlines() if ln.startswith("  program: ")]
    assert Path(program_line.removeprefix("  program: ")).samefile(sys.executable)
    assert "credential: none" in text
    assert "working directory: " in text and "ok" in text.splitlines()[-1]


def test_doctor_reports_a_program_that_does_not_resolve(tmp_path: Path) -> None:
    suite = write_suite(tmp_path, "command:no-such-program-evalproof run")
    code, text = doctor(suite)
    assert code == 1
    assert "problem:" in text and "no-such-program-evalproof" in text
    assert "not ready" in text


def test_doctor_resolves_a_relative_program_against_the_suite_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = write_suite(tmp_path, "command:tools/missing")
    monkeypatch.chdir(tmp_path.parent)
    code, text = doctor(suite)
    assert code == 1 and "tools/missing" in text
