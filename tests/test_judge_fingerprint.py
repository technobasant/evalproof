"""Run facts for a CLI target, and the judge check in compare and the gate."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evalproof.cli import main
from evalproof.compare import compare_runs
from evalproof.config import GateConfig, ScorerConfig, SuiteConfig, TargetConfig
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, EXIT_REGRESSION, decide
from evalproof.models import Run
from evalproof.report import comparison_markdown
from evalproof.runner import environment, run_suite, save_run
from tests.conftest import make_run, mock_client
from tests.fakecli import FakeCLI, claude_ok, install

VERDICT = '{"reason": "fine", "pass": true, "score": 1}'
CLAUDE_VERSION = "2.1.322 (Claude Code)"


def suite_for(tmp_path: Path, *, system: str, judge: str) -> SuiteConfig:
    dataset = tmp_path / "cases.jsonl"
    rows = [{"id": f"c{i}", "input": f"question {i}", "reference": "answer"} for i in range(3)]
    dataset.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return SuiteConfig(
        name="s",
        dataset=str(dataset),
        system=TargetConfig(target=system),
        judge=TargetConfig(target=judge),
        scorers=[ScorerConfig(name="judge", type="llm_judge")],
        cache_dir=str(tmp_path / "cache"),
        base_dir=str(tmp_path),
    )


@pytest.fixture
def claude(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeCLI:
    return install(
        tmp_path,
        monkeypatch,
        "claude",
        version=CLAUDE_VERSION,
        responses=[claude_ok(VERDICT)],
    )


# -- what a run records --------------------------------------------------------------------


def test_environment_records_the_cli_facts_and_resolved_models(
    tmp_path: Path, claude: FakeCLI
) -> None:
    cfg = suite_for(tmp_path, system="claude-code:haiku", judge="claude-code:haiku")
    env = run_suite(cfg).info.environment
    assert env["judge_target"] == env["system_target"] == "claude-code:haiku"
    assert env["judge_cli"] == "claude-code"
    assert env["judge_cli_version"] == CLAUDE_VERSION
    assert env["judge_isolation"] == "claude-code/1"
    assert env["system_cli_version"] == CLAUDE_VERSION
    assert env["judge_model_resolved"] == "claude-haiku-5-5"
    assert env["system_model_resolved"] == "claude-haiku-5-5"
    # The version lookups are not model calls: 3 cases, one system call and one judge call each.
    assert len(claude.calls()) == 6


def test_resolved_models_include_cached_replies(tmp_path: Path, claude: FakeCLI) -> None:
    cfg = suite_for(tmp_path, system="claude-code:haiku", judge="claude-code:haiku")
    run_suite(cfg)
    calls = len(claude.calls())
    again = run_suite(cfg)
    assert len(claude.calls()) == calls  # all served from the cache
    assert again.info.environment["judge_model_resolved"] == "claude-haiku-5-5"
    assert again.info.environment["system_model_resolved"] == "claude-haiku-5-5"


def test_every_resolved_model_is_listed_sorted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(
        tmp_path,
        monkeypatch,
        "claude",
        version=CLAUDE_VERSION,
        responses=[
            claude_ok(VERDICT, model="claude-opus-5-5"),
            claude_ok(VERDICT, model="claude-haiku-5-5"),
        ],
    )
    cfg = suite_for(tmp_path, system="claude-code:haiku", judge="claude-code:haiku")
    env = run_suite(cfg).info.environment
    # The first system call came back from opus, the rest from haiku.
    assert env["system_model_resolved"] == "claude-haiku-5-5,claude-opus-5-5"
    assert env["judge_model_resolved"] == "claude-haiku-5-5"


def test_a_target_that_is_not_a_cli_records_only_its_target(tmp_path: Path) -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/api/chat":
            return httpx.Response(
                200,
                json={
                    "model": "judge-m",
                    "message": {"content": VERDICT},
                    "prompt_eval_count": 5,
                    "eval_count": 5,
                },
            )
        return httpx.Response(200, json={"version": "0.0.test", "models": []})

    cfg = suite_for(tmp_path, system="ollama:judge-m", judge="ollama:judge-m")
    env = run_suite(cfg, client=mock_client(handler)).info.environment
    assert env["judge_target"] == "ollama:judge-m" and env["system_target"] == "ollama:judge-m"
    assert not any(key.startswith(("judge_cli", "system_cli")) for key in env)
    assert env["judge_model_resolved"] == "judge-m"


def test_the_system_target_is_recorded_only_for_a_model_target(tmp_path: Path) -> None:
    cfg = suite_for(tmp_path, system="python:nope:fn", judge="ollama:m")
    env = environment(cfg)
    assert "system_target" not in env and env["judge_target"] == "ollama:m"


# -- compare -------------------------------------------------------------------------------


def run_with(env: dict[str, str], **kwargs: object) -> Run:
    outcomes = {f"c{i}": {"m": i % 2 == 0} for i in range(30)}
    run = make_run(outcomes, **kwargs)  # type: ignore[arg-type]
    run.info.environment = env
    return run


JUDGE_ENV = {
    "judge_target": "claude-code:haiku",
    "judge_cli": "claude-code",
    "judge_cli_version": CLAUDE_VERSION,
    "judge_isolation": "claude-code/1",
    "judge_model_resolved": "claude-haiku-5-5",
}


def test_same_judge_when_nothing_changed() -> None:
    cmp = compare_runs(run_with(JUDGE_ENV), run_with(dict(JUDGE_ENV)), n_boot=200)
    assert cmp.same_judge and cmp.judge_changes == [] and cmp.system_changes == []


def test_a_changed_judge_is_listed() -> None:
    newer = {**JUDGE_ENV, "judge_cli_version": "2.1.400 (Claude Code)"}
    cmp = compare_runs(run_with(JUDGE_ENV), run_with(newer), n_boot=200)
    assert not cmp.same_judge
    assert cmp.judge_changes == [
        "judge_cli_version: 2.1.322 (Claude Code) -> 2.1.400 (Claude Code)"
    ]
    moved = {**JUDGE_ENV, "judge_model_resolved": "claude-haiku-5-6", "judge_target": "codex:m"}
    cmp = compare_runs(run_with(JUDGE_ENV), run_with(moved), n_boot=200)
    assert [c.split(":")[0] for c in cmp.judge_changes] == ["judge_target", "judge_model_resolved"]


def test_only_keys_present_in_both_runs_count() -> None:
    old_run = run_with({"python": "3.12"})  # written before the keys existed
    cmp = compare_runs(old_run, run_with(JUDGE_ENV), n_boot=200)
    assert cmp.same_judge
    cmp = compare_runs(run_with({"judge_target": "ollama:m"}), run_with(JUDGE_ENV), n_boot=200)
    assert cmp.judge_changes == ["judge_target: ollama:m -> claude-code:haiku"]


def test_a_changed_system_is_not_a_judge_change() -> None:
    a = run_with({**JUDGE_ENV, "system_target": "claude-code:haiku", "system_cli_version": "1"})
    b = run_with({**JUDGE_ENV, "system_target": "claude-code:sonnet", "system_cli_version": "2"})
    cmp = compare_runs(a, b, n_boot=200)
    assert cmp.same_judge and len(cmp.system_changes) == 2


# -- gate ----------------------------------------------------------------------------------

GATE = GateConfig(metric="m", min_cases=5)


def changed_pair() -> tuple[Run, Run]:
    newer = {**JUDGE_ENV, "judge_cli_version": "2.1.400 (Claude Code)"}
    return run_with(JUDGE_ENV), run_with(newer)


def test_the_gate_refuses_a_changed_judge() -> None:
    cmp = compare_runs(*changed_pair(), n_boot=200)
    decision = decide(cmp, GATE)
    assert decision.exit_code == EXIT_CANNOT_GATE and decision.status == "fail"
    assert "judge changed" in decision.reasons[0]
    assert "judge_cli_version: 2.1.322" in decision.reasons[0]
    assert "--allow-judge-change" in decision.reasons[0]


def test_the_gate_can_allow_a_changed_judge() -> None:
    cmp = compare_runs(*changed_pair(), n_boot=200)
    decision = decide(cmp, GATE.model_copy(update={"allow_judge_change": True}))
    assert decision.exit_code == EXIT_PASS
    assert any("judge changed" in w and "allowed" in w for w in decision.warnings)


def test_a_changed_judge_and_dataset_are_both_reported() -> None:
    a, b = changed_pair()
    b.info.dataset_hash = "other"
    decision = decide(compare_runs(a, b, n_boot=200), GATE)
    assert decision.exit_code == EXIT_CANNOT_GATE
    assert len(decision.reasons) == 2 and "different datasets" in decision.reasons[0]


def test_a_changed_system_only_warns() -> None:
    a = run_with({**JUDGE_ENV, "system_cli_version": "1"})
    b = run_with({**JUDGE_ENV, "system_cli_version": "2"})
    decision = decide(compare_runs(a, b, n_boot=200), GATE)
    assert decision.exit_code == EXIT_PASS and decision.status == "warn"
    assert "system under test changed" in decision.warnings[0]


def test_runs_without_the_keys_gate_as_before() -> None:
    a = run_with({"python": "3.12"})
    b = run_with({})
    decision = decide(compare_runs(a, b, n_boot=200), GATE)
    assert decision.exit_code == EXIT_PASS and decision.status == "pass"
    worse = make_run({f"c{i}": {"m": False} for i in range(30)})
    decision = decide(compare_runs(a, worse, n_boot=200), GATE)
    assert decision.exit_code == EXIT_REGRESSION


# -- report and CLI ------------------------------------------------------------------------


def test_the_report_shows_judge_changes_without_a_gate() -> None:
    cmp = compare_runs(*changed_pair(), n_boot=200)
    text = comparison_markdown(cmp)
    assert "The judge changed" in text and "2.1.400" in text
    gated = comparison_markdown(cmp, decide(cmp, GATE), ["m"])
    assert gated.count("judge_cli_version") == 1  # once, among the gate's reasons


def test_compare_exit_codes_and_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = changed_pair()
    save_run(a, tmp_path / "a.json")
    save_run(b, tmp_path / "b.json")
    argv = ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    argv += ["--metric", "m", "--min-cases", "5", "--gate"]
    assert main(argv) == EXIT_CANNOT_GATE
    assert "judge changed" in capsys.readouterr().out
    payload = tmp_path / "out.json"
    assert main([*argv, "--allow-judge-change", "--json", str(payload)]) == EXIT_PASS
    data = json.loads(payload.read_text("utf-8"))
    assert data["comparison"]["same_judge"] is False
    assert data["comparison"]["judge_changes"][0].startswith("judge_cli_version")
    # Without --gate the comparison reports the change but exits 0.
    assert main(argv[:-1]) == EXIT_PASS


def test_a_suite_can_allow_the_change_and_the_flag_does_not_undo_it(tmp_path: Path) -> None:
    suite = tmp_path / "suite.yaml"
    suite.write_text(
        "name: t\ndataset: d.jsonl\nsystem: {target: 'ollama:m'}\n"
        "scorers: [{name: m, type: exact_match}]\n"
        "gate: {metric: m, min_cases: 5, allow_judge_change: true}\n",
        "utf-8",
    )
    a, b = changed_pair()
    save_run(a, tmp_path / "a.json")
    save_run(b, tmp_path / "b.json")
    argv = ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")]
    assert main([*argv, "--suite", str(suite), "--gate"]) == EXIT_PASS
