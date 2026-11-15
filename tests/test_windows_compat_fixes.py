"""Windows compatibility, tested by simulation (no Windows runner here).

Where a fix depends on the platform, the test sets ``sys.platform`` or injects a fake in place of
``os.replace`` / ``shutil.rmtree`` / ``time.sleep``, so the Windows branch runs on any OS. Where
it is plain portable behaviour (LF output, a BOM, a file name), the test checks the bytes."""

from __future__ import annotations

import io
import json
import ntpath
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import SimpleNamespace
from typing import Any

import pytest

import evalproof.providers.agent_cli_experimental as experimental
from evalproof import _fsutil
from evalproof._fsutil import replace_retry, rmtree_retry, safe_filename
from evalproof.cache import ResponseCache
from evalproof.cli import _emit, main
from evalproof.config import GenerationParams, ScorerConfig, SuiteConfig, TargetConfig, load_config
from evalproof.dataset import load_dataset, write_dataset
from evalproof.doctor import format_flags
from evalproof.models import Case
from evalproof.providers import (
    ChatResponse,
    CursorProvider,
    Message,
    OpenCodeProvider,
    ProviderError,
)
from evalproof.providers.agent_cli import CodexProvider, config_path
from evalproof.providers.agent_cli_experimental import (
    argv_limit,
    argv_size,
    cursor_sandbox_profile,
    display_workdir,
    env_limit,
    env_size,
    escapes_workspace,
)
from evalproof.runner import RunFileError, load_run, save_run
from evalproof.scorers import make_scorer
from tests.conftest import ScriptedProvider, make_run

BOM = b"\xef\xbb\xbf"
PARAMS = GenerationParams()


def no_sleep(_: float) -> None:
    return None


def resp(text: str = "hello") -> ChatResponse:
    return ChatResponse(text=text, model="m", tokens_in=1, tokens_out=2, latency_ms=3.0)


# -- 1. the response cache: unique temp files, retried replace, best-effort ------------------------

KEY = "ab" + "c" * 62


def leftovers(root: Path) -> list[Path]:
    return sorted(root.glob("*/*.tmp"))


def test_threads_storing_the_same_key_never_raise_and_leave_one_valid_entry(
    tmp_path: Path,
) -> None:
    cache = ResponseCache(tmp_path)
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []

    def work(i: int) -> None:
        try:
            barrier.wait()
            for _ in range(25):
                cache.put(KEY, resp(f"reply {i}"))
                got = cache.get(KEY)
                assert got is None or got.text.startswith("reply ")  # a miss or a whole entry
        except BaseException as exc:  # the test is that nothing raises
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(work, range(16)))
    assert errors == []
    final = cache.get(KEY)
    assert final is not None and final.cached and final.text.startswith("reply ")
    assert leftovers(tmp_path) == []
    assert cache.stats()["entries"] == 1


def test_a_put_goes_through_a_unique_temp_file_in_the_same_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[tuple[str, str]] = []
    real = os.replace

    def spy(src: str, dst: str) -> None:
        seen.append((str(src), str(dst)))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    cache = ResponseCache(tmp_path)
    cache.put(KEY, resp())
    cache.put(KEY, resp("again"))
    (src1, dst1), (src2, dst2) = seen
    assert dst1 == dst2 == str(tmp_path / "ab" / f"{KEY}.json")
    assert src1 != src2  # not one fixed ".tmp" name
    assert Path(src1).parent == Path(dst1).parent and src1.endswith(".tmp")


def test_a_replace_that_windows_refuses_is_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = os.replace
    refusals = [PermissionError(13, "in use"), PermissionError(13, "in use")]

    def flaky(src: str, dst: str) -> None:
        if refusals:
            raise refusals.pop()
        real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    sleeps: list[float] = []
    cache = ResponseCache(tmp_path, sleep=sleeps.append)
    cache.put(KEY, resp("kept"))
    got = cache.get(KEY)
    assert got is not None and got.text == "kept"
    assert len(sleeps) == 2 and sleeps == sorted(sleeps)  # waited between tries, backing off
    assert leftovers(tmp_path) == []


def test_a_replace_refused_every_time_is_swallowed_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[int] = []

    def refuse(src: str, dst: str) -> None:
        calls.append(1)
        raise PermissionError(13, "in use")

    monkeypatch.setattr(os, "replace", refuse)
    sleeps: list[float] = []
    cache = ResponseCache(tmp_path, sleep=sleeps.append)
    cache.put(KEY, resp())  # no exception: a failed put never fails the case
    assert len(calls) == _fsutil.RETRY_ATTEMPTS and len(sleeps) == _fsutil.RETRY_ATTEMPTS - 1
    assert not (tmp_path / "ab" / f"{KEY}.json").exists()
    assert leftovers(tmp_path) == []
    assert cache.get(KEY) is None and cache.stats() == {"entries": 0, "bytes": 0}


def test_any_other_os_error_while_storing_is_swallowed_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def full_disk(*_: Any, **__: Any) -> Any:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)
    ResponseCache(tmp_path, sleep=no_sleep).put(KEY, resp())
    assert leftovers(tmp_path) == []


def test_a_lookup_that_hits_a_permission_error_is_a_miss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = ResponseCache(tmp_path)
    cache.put(KEY, resp())
    assert cache.get(KEY) is not None
    real_read = Path.read_text

    def locked(self: Path, *args: Any, **kwargs: Any) -> str:
        if self.suffix == ".json":
            raise PermissionError(13, "being replaced")
        return real_read(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", locked)
    assert cache.get(KEY) is None
    monkeypatch.undo()
    (tmp_path / "ab" / f"{KEY}.json").write_text("{torn", "utf-8")
    assert cache.get(KEY) is None  # a torn write is still a miss


def test_stray_temp_files_are_not_cache_entries(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    cache.put(KEY, resp())
    before = cache.stats()
    (tmp_path / "ab" / f"{KEY}.k3j2h1.tmp").write_text("half a write", "utf-8")
    assert cache.stats() == before and before["entries"] == 1


# -- the retry helpers -----------------------------------------------------------------------------


def test_replace_retry_succeeds_once_the_lock_clears() -> None:
    attempts: list[tuple[str, str]] = []

    def replace(src: str, dst: str) -> None:
        attempts.append((src, dst))
        if len(attempts) < 3:
            raise PermissionError(13, "locked")

    sleeps: list[float] = []
    assert replace_retry("a", "b", sleep=sleeps.append, replace=replace) is True
    assert len(attempts) == 3 and len(sleeps) == 2


def test_replace_retry_gives_up_after_its_attempts_and_only_retries_permission_errors() -> None:
    def always(src: str, dst: str) -> None:
        raise PermissionError(13, "locked")

    sleeps: list[float] = []
    assert replace_retry("a", "b", attempts=4, sleep=sleeps.append, replace=always) is False
    assert len(sleeps) == 3  # no sleep after the last refusal

    def missing(src: str, dst: str) -> None:
        raise FileNotFoundError(2, "gone")

    with pytest.raises(FileNotFoundError):
        replace_retry("a", "b", sleep=no_sleep, replace=missing)


# -- 6. the temp dir is removed even when a just-killed child still holds it -----------------------


def test_rmtree_retry_tries_again_after_a_failure() -> None:
    calls: list[str] = []

    def rmtree(path: str) -> None:
        calls.append(path)
        if len(calls) <= 2:
            raise PermissionError(13, "in use by another process")

    sleeps: list[float] = []
    assert rmtree_retry("/some/dir", sleep=sleeps.append, rmtree=rmtree) is True
    assert calls == ["/some/dir"] * 3 and len(sleeps) == 2


def test_rmtree_retry_gives_up_silently_and_treats_a_missing_dir_as_done() -> None:
    def stuck(path: str) -> None:
        raise PermissionError(13, "in use")

    assert rmtree_retry("/d", attempts=3, sleep=no_sleep, rmtree=stuck) is False

    def gone(path: str) -> None:
        raise FileNotFoundError(2, "gone")

    assert rmtree_retry("/d", sleep=no_sleep, rmtree=gone) is True


def test_rmtree_retry_really_removes_a_tree(tmp_path: Path) -> None:
    root = tmp_path / "t"
    (root / "a" / "b").mkdir(parents=True)
    (root / "a" / "b" / "f.txt").write_text("x", "utf-8")
    assert rmtree_retry(root) is True and not root.exists()


def test_a_cursor_call_removes_its_temp_root_through_the_retrying_helper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    removed: list[str] = []
    monkeypatch.setattr(experimental, "rmtree_retry", lambda p: removed.append(str(p)) or True)
    monkeypatch.setattr(CursorProvider, "binary_path", lambda self: "cursor-agent")
    ran: list[str] = []

    def fake_run(cmd: list[str], **kwargs: Any) -> tuple[int, str, str]:
        ran.append(kwargs["cwd"])
        return 1, "", "boom"  # a failed call still has to clean up

    monkeypatch.setattr(experimental, "run_process", fake_run)
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    with pytest.raises(ProviderError):
        CursorProvider("auto", sleep=no_sleep).chat([Message(role="user", content="hi")], PARAMS)
    assert len(removed) == 1 and Path(ran[0]).parent == Path(removed[0])


# -- 2. argv and environment limits ----------------------------------------------------------------


def as_platform(monkeypatch: pytest.MonkeyPatch, platform: str) -> None:
    """Make ``experimental`` (and ``doctor``) see ``platform`` without changing ``sys.platform``
    for the rest of the process: stdlib calls like ``shutil.which`` keep the real one."""
    fake_sys = SimpleNamespace(platform=platform, stderr=sys.stderr, stdout=sys.stdout)
    monkeypatch.setattr(experimental, "sys", fake_sys)


def test_the_posix_limit_is_the_old_one_in_bytes(monkeypatch: pytest.MonkeyPatch) -> None:
    as_platform(monkeypatch, "linux")
    assert argv_limit("/usr/bin/cursor-agent") == (100_000, "bytes", argv_limit().reason)
    assert "128 KiB" in argv_limit().reason and env_limit().limit == 100_000
    assert argv_size("é" * 10) == 20  # bytes, not characters


def test_the_windows_limit_is_in_characters_and_below_the_command_line_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_platform(monkeypatch, "win32")
    exe = argv_limit(r"C:\Tools\cursor-agent.exe")
    assert exe.unit == "characters" and 0 < exe.limit < 32_767 and "32767" in exe.reason
    for shim in (r"C:\npm\opencode.cmd", r"C:\npm\copilot.CMD", r"C:\x\y.bat"):
        limit = argv_limit(shim)
        assert 0 < limit.limit < 8_191 and "8191" in limit.reason and "cmd.exe" in limit.reason
    assert argv_limit(r"C:\Tools\a.cmd").limit < argv_limit(r"C:\Tools\a.exe").limit
    env = env_limit()
    assert env.unit == "characters" and 30_000 < env.limit < 32_767 and "32767" in env.reason


def test_windows_counts_utf16_characters_and_the_quoting_of_the_argument(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_platform(monkeypatch, "win32")
    assert argv_size("abc") == 3  # no space or quote: sent as it is
    assert argv_size("a b") == 5  # list2cmdline wraps it in quotes
    assert argv_size('a"b') == 4  # the quote becomes \"
    assert argv_size("😀") == 2  # one code point, two UTF-16 units
    assert env_size("😀é") == 3
    as_platform(monkeypatch, "linux")
    assert argv_size("😀") == 4 and env_size("😀é") == 6


def test_a_prompt_over_the_windows_limit_is_refused_and_the_error_names_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_platform(monkeypatch, "win32")
    monkeypatch.setattr(CursorProvider, "binary_path", lambda self: r"C:\Tools\cursor-agent.exe")
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    limit = argv_limit(r"C:\Tools\cursor-agent.exe")
    big = [Message(role="user", content="x" * (limit.limit + 1))]
    with pytest.raises(ProviderError) as err:
        CursorProvider("auto", sleep=no_sleep).chat(big, PARAMS)
    text = str(err.value)
    assert f"over {limit.limit} characters" in text
    assert "Windows allows a command line of 32767 characters" in text


def test_a_cmd_shim_gets_the_tighter_limit_in_the_error(monkeypatch: pytest.MonkeyPatch) -> None:
    as_platform(monkeypatch, "win32")
    monkeypatch.setattr(CursorProvider, "binary_path", lambda self: r"C:\npm\cursor-agent.cmd")
    monkeypatch.setenv("EVALPROOF_CURSOR_SANDBOX", "off")
    # Under the 32,767 cap but over cmd.exe's 8,191.
    prompt = [Message(role="user", content="x" * 12_000)]
    with pytest.raises(ProviderError, match=r"cmd\.exe, which runs a \.cmd shim, allows .* 8191"):
        CursorProvider("auto", sleep=no_sleep).chat(prompt, PARAMS)


def test_an_opencode_config_over_the_windows_variable_limit_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    as_platform(monkeypatch, "win32")
    provider = OpenCodeProvider("ollama/qwen3:8b", sleep=no_sleep)
    limit = env_limit().limit
    with pytest.raises(ProviderError) as err:
        provider.call_env("s" * (limit + 1), "aux")
    assert f"over {limit} characters" in str(err.value)
    assert "32767 characters in one environment variable" in str(err.value)
    env = provider.call_env("s" * 20_000, "aux")  # fits: no error
    assert "OPENCODE_CONFIG_CONTENT" in env


# -- 3. the sandbox profile is POSIX whatever the host -----------------------------------------


def test_the_cursor_sandbox_profile_is_joined_with_slashes_even_where_os_path_is_ntpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    posix_profile = cursor_sandbox_profile("/H")
    monkeypatch.setattr(experimental, "os", SimpleNamespace(path=ntpath, environ=os.environ))
    assert cursor_sandbox_profile("/H") == posix_profile
    assert '(subpath "/H/.cursor/skills-cursor")' in posix_profile and "\\" not in posix_profile


def test_the_doctor_view_of_the_working_directory_is_always_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert display_workdir("<tmp>/system.txt") == "<tmp>/cwd"
    monkeypatch.setattr(experimental, "os", SimpleNamespace(path=ntpath, environ=os.environ))
    assert display_workdir("<tmp>/system.txt") == "<tmp>/cwd"
    command = CursorProvider("auto", sleep=no_sleep).command(
        "cursor-agent", "<tmp>/system.txt", PARAMS
    )
    assert command[-2:] == ["--workspace", "<tmp>/cwd"]


# -- 4. the workspace-escape check, read as POSIX and as Windows ----------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/etc/x",
        "C:\\x",
        "C:/x",
        "c:x",
        "\\x",
        "\\\\server\\share\\x",
        "..",
        "a/../b",
        "a\\..\\b",
        "../b",
        "..\\b",
        "a:stream",
    ],
)
def test_paths_that_leave_the_workspace_are_refused_on_every_os(bad: str) -> None:
    assert escapes_workspace(bad) is True
    # Which view caught it differs by case, but neither view alone is enough for all of them:
    assert PurePosixPath(bad).is_absolute() or PureWindowsPath(bad).anchor or ".." in str(bad)


@pytest.mark.parametrize(
    "good", ["a.txt", ".claude/skills/x/SKILL.md", "dir/sub/f.md", "a..b/c", "name with space.md"]
)
def test_plain_relative_paths_stay_allowed(good: str) -> None:
    assert escapes_workspace(good) is False


def test_the_canary_files_cannot_be_planted_outside_the_workspace(tmp_path: Path) -> None:
    cwd, aux = tmp_path / "cwd", tmp_path / "aux"
    cwd.mkdir()
    aux.mkdir()
    for bad in ("/etc/x", "C:\\x"):
        provider = CursorProvider("auto", sleep=no_sleep, extra_workspace_files={bad: "t"})
        with pytest.raises(ProviderError, match="must be inside the workspace"):
            provider.prepare_workdir(cwd)
    ok = CursorProvider("auto", sleep=no_sleep, extra_workspace_files={".claude/s/SKILL.md": "t"})
    ok.prepare_workdir(cwd)
    assert (cwd / ".claude" / "s" / "SKILL.md").read_text("utf-8") == "t"


# -- 5. codex gets a TOML-safe instructions path ---------------------------------------------------


def test_codex_instructions_path_has_forward_slashes_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    windows = r"C:\Users\Ünal\AppData\Local\Temp\evalproof-ab12\system.txt"
    monkeypatch.setattr(sys, "platform", "win32")
    command = CodexProvider("gpt-5", sleep=no_sleep).command("codex", windows, PARAMS)
    flag = next(a for a in command if a.startswith("model_instructions_file="))
    value = flag.split("=", 1)[1]
    assert value == "C:/Users/Ünal/AppData/Local/Temp/evalproof-ab12/system.txt"
    assert "\\" not in value  # no \U, \A, \L, \T, \e, \s left to read as TOML escapes
    assert PureWindowsPath(value) == PureWindowsPath(windows)  # the same file
    assert command[command.index(flag) - 1] == "-c"


def test_a_posix_instructions_path_is_passed_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert config_path("/tmp/evalproof-x/system.txt") == "/tmp/evalproof-x/system.txt"
    assert config_path("/tmp/a\\b/system.txt") == "/tmp/a\\b/system.txt"  # a legal POSIX name


# -- 7. LF on every OS -----------------------------------------------------------------------------


@pytest.fixture
def open_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Record the ``newline`` every ``io.open`` (what ``Path.open`` and ``write_text`` call) got."""
    calls: list[dict[str, Any]] = []
    real = io.open

    def spy(file: Any, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
        positional_newline = args[3] if len(args) > 3 else None
        calls.append(
            {"file": str(file), "mode": mode, "newline": kwargs.get("newline", positional_newline)}
        )
        return real(file, mode, *args, **kwargs)

    monkeypatch.setattr(io, "open", spy)
    return calls


def test_emit_appends_markdown_with_lf_endings(
    tmp_path: Path, open_calls: list[dict[str, Any]]
) -> None:
    target = tmp_path / "summary.md"
    _emit("# Report\n| a | b |\n", str(target), quiet=True)
    _emit("second\n", str(target), quiet=True)
    data = target.read_bytes()
    assert b"\r" not in data and data == b"# Report\n| a | b |\n\nsecond\n\n"
    writes = [c for c in open_calls if c["file"] == str(target) and c["mode"] != "rb"]
    assert len(writes) == 2 and all(c["mode"] == "a" and c["newline"] == "\n" for c in writes)


def test_a_json_output_is_written_with_lf_endings(
    tmp_path: Path, open_calls: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    a = make_run({"c1": {"m": True}, "c2": {"m": False}})
    b = make_run({"c1": {"m": True}, "c2": {"m": True}})
    save_run(a, tmp_path / "a.json")
    save_run(b, tmp_path / "b.json")
    out, md = tmp_path / "out.json", tmp_path / "out.md"
    argv = ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json"), "--metric", "m"]
    code = main([*argv, "--json", str(out), "--markdown", str(md)])
    assert code in (0, 1, 3)  # whichever verdict; what is checked is how the files are written
    for path in (out, md, tmp_path / "a.json"):
        assert path.exists() and b"\r" not in path.read_bytes()
    json.loads(out.read_text("utf-8"))
    mine = {str(out), str(md), str(tmp_path / "a.json")}
    writes = [c for c in open_calls if c["file"] in mine and c["mode"] in ("w", "a")]
    assert len(writes) >= 3 and all(c["newline"] == "\n" for c in writes)


def test_a_dataset_is_written_with_lf_endings(
    tmp_path: Path, open_calls: list[dict[str, Any]]
) -> None:
    path = write_dataset([Case(id="a", input="x"), Case(id="b", input="y")], tmp_path / "d.jsonl")
    assert b"\r" not in path.read_bytes() and path.read_bytes().count(b"\n") == 2
    writes = [c for c in open_calls if c["file"] == str(path) and c["mode"] == "w"]
    assert [c["newline"] for c in writes] == ["\n"]


# -- 8. a leading BOM is tolerated -------------------------------------------------------------

SUITE_TEXT = (
    "name: demo\ndataset: cases.jsonl\n"
    "system: {target: 'ollama:m', system_prompt_file: prompt.txt}\n"
    "scorers: [{name: m, type: contains, params: {any_of: [ok]}}]\n"
)


def with_bom(path: Path, text: str) -> Path:
    path.write_bytes(BOM + text.encode("utf-8"))
    return path


def test_a_suite_saved_with_a_bom_loads_and_so_do_its_prompt_files(tmp_path: Path) -> None:
    with_bom(tmp_path / "suite.yaml", SUITE_TEXT)
    with_bom(tmp_path / "prompt.txt", "Be brief.")
    cfg = load_config(tmp_path / "suite.yaml")
    assert cfg.name == "demo" and cfg.system.system_prompt == "Be brief."  # no U+FEFF in it


def test_a_dataset_saved_with_a_bom_loads(tmp_path: Path) -> None:
    with_bom(tmp_path / "cases.jsonl", '{"id": "a", "input": "x"}\n{"id": "b", "input": "y"}\n')
    assert [c.id for c in load_dataset(tmp_path / "cases.jsonl")] == ["a", "b"]


def test_a_rubric_file_saved_with_a_bom_loads_clean(tmp_path: Path) -> None:
    with_bom(tmp_path / "rubric.md", "custom rubric")
    suite = SuiteConfig(
        name="s",
        dataset="d.jsonl",
        system=TargetConfig(target="ollama:m"),
        judge=TargetConfig(target="ollama:j"),
        scorers=[ScorerConfig(name="j", type="llm_judge", params={"rubric_file": "rubric.md"})],
        base_dir=str(tmp_path),
    )
    judge = make_scorer(suite.scorers[0], suite, ScriptedProvider([]))
    assert judge.rubric == "custom rubric"  # type: ignore[attr-defined]


def test_a_run_file_saved_with_a_bom_loads_and_its_version_is_still_checked(
    tmp_path: Path,
) -> None:
    run = make_run({"c1": {"m": True}})
    path = save_run(run, tmp_path / "run.json")
    path.write_bytes(BOM + path.read_bytes())
    assert [r.case_id for r in load_run(path).results] == ["c1"]
    newer = json.loads(path.read_bytes().decode("utf-8-sig"))
    newer["format_version"] = 999
    with_bom(tmp_path / "newer.json", json.dumps(newer))
    with pytest.raises(RunFileError, match="format version 999"):
        load_run(tmp_path / "newer.json")


# -- 9. the default run-file name is valid on NTFS -------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("demo", "demo"),
        ("my suite", "my suite"),
        ('a<b>:c"d/e\\f|g?h*i', "a-b--c-d-e-f-g-h-i"),
        ("tab\tnew\nline\x00nul", "tab-new-line-nul"),
        ("ends with dots...", "ends with dots"),
        ("trailing space ", "trailing space"),
        (". .", "run"),
        ("", "run"),
        ("   ", "run"),
        ("ünïcode-Δ", "ünïcode-Δ"),
    ],
)
def test_safe_filename(name: str, expected: str) -> None:
    assert safe_filename(name) == expected


def test_the_default_run_path_uses_a_cleaned_suite_name_and_out_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    modname = f"winmod_{tmp_path.name}"
    (tmp_path / modname).with_suffix(".py").write_text("def ask(x):\n    return 'ok'\n", "utf-8")
    (tmp_path / "cases.jsonl").write_text('{"id": "a", "input": "q"}\n', "utf-8")
    (tmp_path / "suite.yaml").write_text(
        "name: 'a<b>:c/d. '\ndataset: cases.jsonl\n"
        f"system: {{target: 'python:{modname}:ask'}}\n"
        "scorers: [{name: m, type: contains, params: {any_of: [ok]}}]\n",
        "utf-8",
    )
    assert main(["run", "suite.yaml", "--no-cache", "--quiet"]) == 0
    written = list((tmp_path / "runs").iterdir())
    assert len(written) == 1
    assert written[0].name.startswith("a-b--c-d-") and written[0].suffix == ".json"
    assert not any(ch in written[0].name for ch in '<>:"/\\|?*')
    # A path the user gave is theirs: used exactly as given.
    mine = tmp_path / "my results" / "run (1).json"
    assert main(["run", "suite.yaml", "--no-cache", "--quiet", "--out", str(mine)]) == 0
    assert mine.exists()


# -- 10. the flags doctor prints -------------------------------------------------------------------


def test_doctor_flags_are_quoted_for_the_os_and_tmp_paths_are_always_posix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = ["--workspace", "<tmp>\\cwd", "--usage-output-file", "<tmp>/aux/usage.json", "-m", "a b"]
    monkeypatch.setattr(sys, "platform", "linux")
    assert format_flags(args) == (
        "--workspace '<tmp>/cwd' --usage-output-file '<tmp>/aux/usage.json' -m 'a b'"
    )
    monkeypatch.setattr(sys, "platform", "win32")
    assert format_flags(args) == (
        '--workspace <tmp>/cwd --usage-output-file <tmp>/aux/usage.json -m "a b"'
    )
    assert format_flags(["--cwd", "C:\\Users\\x"]) == "--cwd C:\\Users\\x"  # real paths untouched
