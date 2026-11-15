"""``evalproof import-dataset`` (formats, ``--suite-out``, field options) and ``init --from``.

Everything runs against the hand-written fixtures in ``tests/fixtures/datasets``. The suites
the commands write are loaded with the real config loader, and one is run against a ``python:``
target, so no test calls a model: the cases used for that run need no judge.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evalproof.cli import build_parser, main
from evalproof.config import load_config
from evalproof.dataset import load_dataset
from evalproof.importers import DATASET_FORMATS
from evalproof.models import Run

DATASETS = Path(__file__).parent / "fixtures" / "datasets"
PROMPTFOO_JSONL = DATASETS / "promptfoo" / "tests.jsonl"  # equals + icontains-any: no judge
PROMPTFOO_CSV = DATASETS / "promptfoo" / "tests.csv"
PROMPTFOO_CONFIG = DATASETS / "promptfoo" / "promptfooconfig.yaml"
RAGAS = DATASETS / "ragas" / "samples.jsonl"  # free text answers: a judge is needed


@pytest.fixture(autouse=True)
def no_agent_clis(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``choose_judge`` looks for claude, codex and ollama on PATH: make it find none."""
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))


def say(capsys: pytest.CaptureFixture[str]) -> tuple[str, str]:
    captured = capsys.readouterr()
    return captured.out, captured.err


# --- import-dataset ---------------------------------------------------------------------------


def test_auto_prints_the_detected_format_and_the_unmapped_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    code = main(["import-dataset", "auto", str(PROMPTFOO_CONFIG), "--out", str(out)])
    stdout, stderr = say(capsys)
    assert code == 0
    assert stdout == ""
    assert "evalproof: detected format: promptfoo\n" in stderr
    assert (
        "evalproof: note: 2 assertions not imported: javascript x 1, similar x 1 "
        "(kept on each case under 'unmapped')\n"
    ) in stderr
    assert stderr.endswith(f"wrote 4 case(s) to {out}\n")
    cases = load_dataset(out)
    assert [c.id for c in cases][:2] == ["free-plan-notebooks", "refund-window"]
    unmapped = {c.id: c.model_extra.get("unmapped") for c in cases if c.model_extra}
    assert unmapped["refund-window"] == [{"type": "javascript", "value": "output.length < 300"}]


def test_naming_the_format_does_not_say_it_was_detected(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    assert main(["import-dataset", "ragas", str(RAGAS), "--out", str(out)]) == 0
    _, stderr = say(capsys)
    assert "detected format" not in stderr
    assert "ragas importer is experimental" in stderr or "wrote 3 case(s)" in stderr
    assert len(load_dataset(out)) == 3


def test_the_command_takes_every_dataset_format() -> None:
    parser = build_parser()
    for fmt in DATASET_FORMATS:
        args = parser.parse_args(["import-dataset", fmt, "f", "--out", "o"])
        assert args.format == fmt
    action = next(a for a in _subparser(parser, "import-dataset")._actions if a.dest == "format")
    assert list(action.choices or ()) == list(DATASET_FORMATS)


def _subparser(parser, name):  # type: ignore[no-untyped-def]
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and name in choices:
            return choices[name]
    raise AssertionError(name)


def test_input_field_repeats_or_takes_a_comma_list(tmp_path: Path) -> None:
    source = DATASETS / "jsonl" / "qa.jsonl"
    repeated = tmp_path / "repeated.jsonl"
    listed = tmp_path / "listed.jsonl"
    common = ["--id-field", "qid", "--reference-field", "answer", "--tags-field", "labels"]
    assert (
        main(
            [
                "import-dataset",
                "jsonl",
                str(source),
                "--out",
                str(repeated),
                "--input-field",
                "question",
                "--input-field",
                "context",
                *common,
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "import-dataset",
                "jsonl",
                str(source),
                "--out",
                str(listed),
                "--input-field",
                "question,context",
                *common,
            ]
        )
        == 0
    )
    assert load_dataset(repeated) == load_dataset(listed)
    case = load_dataset(repeated)[0]
    assert case.id == "q-1"
    assert case.input == {"question": "Can I get a refund?", "context": "Refunds within 30 days."}
    assert case.reference == "Yes, within 30 days."
    assert case.tags == ["billing", "support", "imported", "from:jsonl"]


def test_the_csv_format_reads_the_named_columns(tmp_path: Path) -> None:
    out = tmp_path / "cases.jsonl"
    args = ["import-dataset", "csv", str(DATASETS / "csv" / "qa.csv"), "--out", str(out)]
    assert main([*args, "--input-field", "question", "--reference-field", "answer"]) == 0
    cases = load_dataset(out)
    assert [c.input for c in cases] == ["Can I get a refund?", "What is the Pro price?"]
    assert [c.reference for c in cases] == ["Yes, within 30 days.", "Pro costs $8 per month."]


def test_an_ambiguous_or_unknown_file_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    mixed = tmp_path / "mixed.jsonl"
    mixed.write_text('{"input": "q", "target": "t", "expected": "e", "ideal": "i"}\n', "utf-8")
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    assert main(["import-dataset", "auto", str(mixed), "--out", str(out)]) == 2
    _, err = say(capsys)
    assert "evalproof: error: " in err
    assert "fits more than one format" in err
    odd = tmp_path / "odd.jsonl"
    odd.write_text('{"unrelated": 1}\n', encoding="utf-8")
    assert (
        main(["import-dataset", "auto", str(odd), "--out", str(out), "--suite-out", str(suite)])
        == 2
    )
    assert "cannot tell which tool wrote this file" in say(capsys)[1]
    assert not out.exists()
    assert not suite.exists()


def test_a_missing_file_exits_2(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "cases.jsonl"
    assert main(["import-dataset", "auto", str(tmp_path / "nope.jsonl"), "--out", str(out)]) == 2
    assert "evalproof: error: " in say(capsys)[1]
    assert not out.exists()


def test_field_options_on_another_format_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    code = main(["import-dataset", "ragas", str(RAGAS), "--out", str(out), "--id-field", "x"])
    assert code == 2
    assert "apply to the jsonl and csv formats" in say(capsys)[1]
    assert not out.exists()


# --- --suite-out ------------------------------------------------------------------------------


def test_suite_out_writes_exactly_the_scorers_the_cases_need_and_no_judge(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "promptfoo", str(PROMPTFOO_JSONL), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 0
    assert say(capsys)[1].endswith(f"wrote suite to {suite}\n")
    config = load_config(suite)
    assert [(s.name, s.type) for s in config.scorers] == [
        ("exact", "exact_match"),
        ("contains", "contains"),
    ]
    assert config.scorers[1].params == {"any_of_field": "accept"}
    assert config.judge is None  # nothing in these cases needs one
    assert config.gate.metric == "exact"
    assert config.system.prompt_template == "{question}"
    assert config.dataset == "cases.jsonl"
    assert len(load_dataset(config.resolve(config.dataset))) == 2


def test_suite_out_gates_the_scorer_that_checks_the_most_cases(tmp_path: Path) -> None:
    # In this config only one of four cases has an `equals` (exact), while the defaultTest
    # not-contains reaches every case: gating `exact` would leave the gate one paired case.
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "promptfoo", str(PROMPTFOO_CONFIG), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 0
    config = load_config(suite)
    assert config.scorers[0].name == "exact"
    assert config.gate.metric == "omits"
    text = suite.read_text(encoding="utf-8")
    assert "  # checks 4 of 4 case(s)\n  - name: omits" in text
    assert "  # checks 1 of 4 case(s)\n  - name: exact" in text


def test_suite_out_adds_a_judge_block_only_when_a_scorer_needs_one(
    tmp_path: Path,
) -> None:
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "ragas", str(RAGAS), "--out", str(out), "--suite-out", str(suite)]
    assert main(args) == 0
    config = load_config(suite)
    assert [(s.name, s.type) for s in config.scorers] == [("judge", "llm_judge")]
    assert config.judge is not None
    assert config.gate.metric == "judge"
    assert "judge:" in suite.read_text(encoding="utf-8")


def test_suite_out_uses_the_judge_choose_judge_picks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    name = "claude.exe" if __import__("sys").platform == "win32" else "claude"
    (bin_dir / name).write_text("", encoding="utf-8")
    (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir))
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "ragas", str(RAGAS), "--out", str(tmp_path / "c.jsonl")]
    assert main([*args, "--suite-out", str(suite)]) == 0
    config = load_config(suite)
    assert config.judge is not None
    assert config.judge.target == "claude-code:haiku"


def test_suite_out_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    suite.write_text("keep: me\n", encoding="utf-8")
    args = ["import-dataset", "promptfoo", str(PROMPTFOO_JSONL), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 2
    assert "pass --force to overwrite it" in say(capsys)[1]
    assert suite.read_text(encoding="utf-8") == "keep: me\n"
    assert not out.exists()  # nothing was written
    assert main([*args, "--suite-out", str(suite), "--force"]) == 0
    assert load_config(suite).scorers
    assert out.exists()


def test_suite_out_with_nothing_checkable_exits_2_and_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = tmp_path / "only-code.json"
    tests = [{"vars": {"q": "a"}, "assert": [{"type": "javascript", "value": "true"}]}]
    source.write_text(json.dumps({"tests": tests}), encoding="utf-8")
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "promptfoo", str(source), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 2
    assert "evalproof: error: " in say(capsys)[1]
    assert not out.exists()
    assert not suite.exists()
    # the cases alone can still be imported, and keep the assertion
    assert main(args) == 0
    case = load_dataset(out)[0]
    assert case.model_extra is not None
    assert case.model_extra["unmapped"] == [{"type": "javascript", "value": "true"}]


def test_the_suite_refers_to_the_cases_file_relative_to_itself(tmp_path: Path) -> None:
    (tmp_path / "evals").mkdir()
    out = tmp_path / "evals" / "cases.jsonl"
    suite = tmp_path / "evals" / "suite.yaml"
    args = ["import-dataset", "promptfoo", str(PROMPTFOO_JSONL), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 0
    assert 'dataset: "cases.jsonl"' in suite.read_text(encoding="utf-8")
    assert [c.id for c in load_dataset(load_config(suite).resolve(load_config(suite).dataset))] == [
        c.id for c in load_dataset(out)
    ]


def test_a_suite_from_suite_out_runs_against_a_python_target(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "cases.jsonl"
    suite = tmp_path / "suite.yaml"
    args = ["import-dataset", "promptfoo", str(PROMPTFOO_JSONL), "--out", str(out)]
    assert main([*args, "--suite-out", str(suite)]) == 0
    (tmp_path / "echo.py").write_text(
        "def ask(text):\n    return 'The answer is 4, or red.'\n", encoding="utf-8"
    )
    say(capsys)
    run_file = tmp_path / "runs" / "latest.json"
    code = main(
        [
            "run",
            str(suite),
            "--system",
            "python:echo:ask",
            "--limit",
            "2",
            "--out",
            str(run_file),
        ]
    )
    assert code == 0
    run = Run.model_validate_json(run_file.read_text(encoding="utf-8"))
    assert len(run.results) == 2
    assert {name for r in run.results for name in r.scores} == {"exact", "contains"}
    by_case = {r.case_id: r for r in run.results}
    assert by_case[next(iter(by_case))].scores["exact"].value is not None


# --- init --from ------------------------------------------------------------------------------


def init(tmp_path: Path, *extra: str) -> int:
    return main(["init", "--dir", str(tmp_path / "repo"), "--judge", "openai:gpt-test", *extra])


def test_init_from_writes_the_cases_and_a_suite_with_the_scorers_they_need(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert init(tmp_path, "--from", str(PROMPTFOO_CSV)) == 0
    stdout, stderr = say(capsys)
    text = stdout + stderr
    assert "imported 3 case(s) from tests.csv (detected: promptfoo)" in text
    assert "1 assertion not imported: javascript x 1" in text
    repo = tmp_path / "repo"
    cases = load_dataset(repo / "evals" / "cases.jsonl")
    assert [c.id for c in cases] == ["free-plan", "refund-window", "pro-price"]
    config = load_config(repo / "evals" / "suite.yaml")
    assert [s.name for s in config.scorers] == [
        "exact",
        "contains",
        "contains_cs",
        "regex",
        "omits",
        "rubrics",
    ]
    assert config.judge is not None
    assert config.judge.target == "openai:gpt-test"  # the --judge init was given
    assert config.system.target == "openai:gpt-test"
    assert [c.id for c in load_dataset(config.resolve(config.dataset))] == [c.id for c in cases]
    # everything else init writes is still written
    for rel in (
        "evals/README.md",
        "evals/.gitignore",
        ".github/workflows/evalproof.yml",
        "AGENTS.md",
    ):
        assert (repo / rel).is_file()


def test_init_from_a_judge_free_import_has_no_judge_block(tmp_path: Path) -> None:
    assert init(tmp_path, "--from", str(PROMPTFOO_JSONL)) == 0
    config = load_config(tmp_path / "repo" / "evals" / "suite.yaml")
    assert [s.name for s in config.scorers] == ["exact", "contains"]
    assert config.judge is None


def test_init_from_with_a_named_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert init(tmp_path, "--from", str(RAGAS), "--from-format", "ragas") == 0
    stdout, stderr = say(capsys)
    assert "imported 3 case(s) from samples.jsonl (format: ragas)" in stdout + stderr
    cases = load_dataset(tmp_path / "repo" / "evals" / "cases.jsonl")
    assert all("from:ragas" in c.tags for c in cases)


def test_init_from_keeps_files_that_exist_and_force_replaces_them(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    (repo / "evals").mkdir(parents=True)
    mine = repo / "evals" / "cases.jsonl"
    mine.write_text('{"id": "mine", "input": "hand written"}\n', encoding="utf-8")
    assert init(tmp_path, "--from", str(PROMPTFOO_JSONL)) == 0
    assert "kept (exists)  evals/cases.jsonl" in say(capsys)[0]
    assert [c.id for c in load_dataset(mine)] == ["mine"]
    assert (repo / "evals" / "suite.yaml").is_file()  # the rest was still written
    assert init(tmp_path, "--from", str(PROMPTFOO_JSONL), "--force") == 0
    assert "mine" not in [c.id for c in load_dataset(mine)]
    assert len(load_dataset(mine)) == 2


def test_init_from_a_file_that_cannot_be_imported_writes_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    odd = tmp_path / "odd.jsonl"
    odd.write_text('{"unrelated": 1}\n', encoding="utf-8")
    assert init(tmp_path, "--from", str(odd)) == 2
    assert "cannot tell which tool wrote this file" in say(capsys)[1]
    assert not (tmp_path / "repo").exists()
    only_code = tmp_path / "only-code.json"
    tests = [{"vars": {"q": "a"}, "assert": [{"type": "javascript", "value": "true"}]}]
    only_code.write_text(json.dumps({"tests": tests}), encoding="utf-8")
    assert init(tmp_path, "--from", str(only_code)) == 2
    assert not (tmp_path / "repo").exists()
    assert init(tmp_path, "--from", str(tmp_path / "nope.jsonl")) == 2
    assert not (tmp_path / "repo").exists()


def test_init_from_with_agent_files_only_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert init(tmp_path, "--from", str(PROMPTFOO_JSONL), "--agent-files-only") == 2
    assert "--agent-files-only" in say(capsys)[1]
    assert not (tmp_path / "repo").exists()


def test_init_without_from_still_writes_the_sample_cases(tmp_path: Path) -> None:
    assert init(tmp_path) == 0
    config = load_config(tmp_path / "repo" / "evals" / "suite.yaml")
    assert config.scorers
    assert len(load_dataset(tmp_path / "repo" / "evals" / "cases.jsonl")) == 3


def test_init_parser_knows_from_and_from_format() -> None:
    args = build_parser().parse_args(["init"])
    assert args.from_file is None
    assert args.from_format == "auto"
    args = build_parser().parse_args(["init", "--from", "f.jsonl", "--from-format", "inspect"])
    assert (args.from_file, args.from_format) == ("f.jsonl", "inspect")
    with pytest.raises(SystemExit):
        build_parser().parse_args(["init", "--from-format", "lm-eval"])
