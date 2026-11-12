"""Importers: promptfoo, Inspect, DeepEval, lm-eval and the generic jsonl/csv tables."""

from __future__ import annotations

import copy
import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from evalproof import import_run
from evalproof.cli import main
from evalproof.gate import EXIT_CANNOT_GATE, EXIT_PASS, EXIT_REGRESSION, EXIT_USAGE
from evalproof.importers import FORMATS, ImportFormatError
from evalproof.models import Run
from evalproof.runner import load_run

FX = Path(__file__).parent / "fixtures"
PF = FX / "promptfoo" / "sample-export.json"
INSPECT = FX / "inspect" / "log_formats.json"
DEEPEVAL = FX / "deepeval" / "test_run.json"
HELLASWAG = FX / "lm-eval" / "gpt2" / "samples_hellaswag_2024-05-01T10-20-30.123456.jsonl"
GSM8K = FX / "lm-eval" / "llama3" / "samples_gsm8k_2024-05-01T10-20-30.123456.jsonl"
LONG_CSV, LONG_JSONL, WIDE_CSV = (
    FX / "table" / "long.csv",
    FX / "table" / "long.jsonl",
    FX / "table" / "wide.csv",
)


def _flags(run: Run) -> dict[str, dict[str, tuple[float | None, bool | None]]]:
    return {r.case_id: {k: (s.value, s.passed) for k, s in r.scores.items()} for r in run.results}


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", "utf-8")
    return path


# --- promptfoo -----------------------------------------------------------------------------


def _pf_row(idx: int, question: str, success: bool, **kw: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "testIdx": idx,
        "vars": {"q": question},
        "success": success,
        "score": 1.0 if success else 0.0,
        "provider": {"id": kw.get("provider", "prov"), "label": kw.get("label", "")},
        "prompt": {"raw": f"rendered {question}", "label": kw.get("prompt", "tmpl")},
        "latencyMs": 100.0,
        "cost": 0.001,
        "namedScores": kw.get("named", {}),
        "response": {"output": "out", "tokenUsage": {"prompt": 10, "completion": 5}},
        "testCase": {"vars": {"q": question}, "description": kw.get("description")},
        "gradingResult": {"pass": success, "reason": "r", "componentResults": kw.get("comps", [])},
        "failureReason": kw.get("failure", 0 if success else 1),
    }
    for key in ("error", "repeatGroupId", "repeatIndex"):
        if key in kw:
            row[key] = kw[key]
    return row


def _pf_file(path: Path, successes: list[bool], prefix: str = "q") -> Path:
    return _jsonl(path, [_pf_row(i, f"{prefix}{i}", s) for i, s in enumerate(successes)])


def test_promptfoo_fixture_maps_scores_and_run_info() -> None:
    run = import_run("promptfoo", PF, system="openai:gpt-5")
    info = run.info
    assert info.system == "openai:gpt-5" and info.suite.startswith("Comparing OpenAI")
    assert info.environment == {
        "imported_from": "promptfoo",
        "source_schema": "3",
        "source_tool_version": "0.118.10",
    }
    assert info.dataset_path == str(PF) and info.seed == 0 and info.wall_time_s == 0
    assert info.started_at == "2025-10-01T20:08:11.168Z" and info.finished_at == ""
    first = run.results[0]
    assert first.output.latency_ms == 6197 and first.output.tokens_in == 37
    assert first.output.tokens_out == 204 and first.output.cost_usd == pytest.approx(0.00208625)
    assert {k: s.passed for k, s in first.scores.items()} == {
        "assert:cost": False,
        "assert:latency": False,
        "assert:contains": True,
        "assert:llm-rubric": True,
        "pass": False,
    }


def test_promptfoo_ids_ignore_order_and_index() -> None:
    a = import_run("promptfoo", PF, system="openai:gpt-5")
    assert a.results[0].case_id != a.results[1].case_id
    assert import_run("promptfoo", PF, system="openai:gpt-5") == a  # two imports agree


def test_promptfoo_ids_survive_reordering(tmp_path: Path) -> None:
    doc = json.loads(PF.read_text("utf-8"))
    doc["results"]["results"].reverse()
    for row in doc["results"]["results"]:
        row["testIdx"] += 10
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(doc), "utf-8")
    a = import_run("promptfoo", PF, system="openai:gpt-5")
    b = import_run("promptfoo", moved, system="openai:gpt-5")
    assert {r.case_id for r in a.results} == {r.case_id for r in b.results}
    assert a.info.dataset_hash == b.info.dataset_hash


def test_promptfoo_system_selection(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match=r"2 systems.*--system"):
        import_run("promptfoo", PF)
    with pytest.raises(ImportFormatError, match="no system 'nope'"):
        import_run("promptfoo", PF, system="nope")
    rows = [_pf_row(0, "a", True, label="A"), _pf_row(0, "a", False, label="B")]
    path = _jsonl(tmp_path / "labelled.jsonl", rows)
    assert import_run("promptfoo", path, system="A").results[0].scores["pass"].passed is True
    assert import_run("promptfoo", path, system="B").results[0].scores["pass"].passed is False
    # Several prompts: the system is the provider and the prompt.
    two = [_pf_row(0, "a", True, prompt="v1"), _pf_row(0, "a", True, prompt="v2")]
    with pytest.raises(ImportFormatError, match=r"prov · v1.*prov · v2"):
        import_run("promptfoo", _jsonl(tmp_path / "prompts.jsonl", two))


def test_promptfoo_json_and_jsonl_agree(tmp_path: Path) -> None:
    rows = json.loads(PF.read_text("utf-8"))["results"]["results"]
    path = _jsonl(tmp_path / "rows.jsonl", rows)
    as_json = import_run("promptfoo", PF, system="openai:gpt-5")
    as_jsonl = import_run("promptfoo", path, system="openai:gpt-5")
    assert (
        as_json.results == as_jsonl.results
        and as_json.info.dataset_hash == as_jsonl.info.dataset_hash
    )
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps({"version": 3, "results": rows}), "utf-8")
    assert import_run("promptfoo", bare, system="openai:gpt-5").results == as_json.results


def test_promptfoo_identical_tests_get_numbered_ids(tmp_path: Path) -> None:
    rows = [_pf_row(0, "same", True), _pf_row(1, "same", False), _pf_row(2, "other", True)]
    run = import_run("promptfoo", _jsonl(tmp_path / "dup.jsonl", rows))
    ids = [r.case_id for r in run.results]
    assert ids[1] == f"{ids[0]}#2" and len(set(ids)) == 3
    # The description is part of the test's identity.
    rows = [_pf_row(0, "same", True, description="d1"), _pf_row(1, "same", True, description="d2")]
    ids = [r.case_id for r in import_run("promptfoo", _jsonl(tmp_path / "d.jsonl", rows)).results]
    assert "#" not in ids[0] + ids[1]


@pytest.mark.parametrize(
    ("epochs", "value", "passed"),
    [("mean", 2 / 3, None), ("first", 1.0, True), ("all", 0.0, False), ("any", 1.0, True)],
)
def test_promptfoo_repeats_collapse_by_epochs(
    tmp_path: Path, epochs: str, value: float, passed: bool | None
) -> None:
    rows = [
        _pf_row(i, "q", ok, repeatGroupId="g", repeatIndex=i)
        for i, ok in enumerate([True, True, False])
    ]
    run = import_run("promptfoo", _jsonl(tmp_path / "r.jsonl", rows), epochs=epochs)
    assert len(run.results) == 1
    score = run.results[0].scores["pass"]
    assert score.value == pytest.approx(value) and score.passed is passed
    assert run.info.environment["epochs"] == epochs


def test_promptfoo_repeat_index_without_group_id(tmp_path: Path) -> None:
    rows = [_pf_row(i, "q", True, repeatIndex=i % 2) for i in range(4)]
    run = import_run("promptfoo", _jsonl(tmp_path / "ri.jsonl", rows), epochs="all")
    ids = [r.case_id for r in run.results]
    assert len(ids) == 2 and ids[1] == f"{ids[0]}#2"  # two identical tests, two repeats each


def test_promptfoo_error_rows_fail_every_metric(tmp_path: Path) -> None:
    comp = [{"pass": True, "assertion": {"type": "contains"}}]
    ok = _pf_row(0, "a", True, named={"rel": 0.7}, comps=comp)
    bad = _pf_row(1, "b", False, failure=2, error="429 Too Many Requests")
    assertion_failure = _pf_row(2, "c", False, failure=1, comps=comp, named={"rel": 0.2})
    run = import_run("promptfoo", _jsonl(tmp_path / "e.jsonl", [ok, bad, assertion_failure]))
    by_id = {r.output.error: r for r in run.results}
    errored = by_id["429 Too Many Requests"]
    assert set(errored.scores) == {"rel", "assert:contains", "pass"}
    assert all(s.value == 0.0 and s.passed is False for s in errored.scores.values())
    failed_only = [r for r in run.results if r.output.error is None]
    assert len(failed_only) == 2  # failureReason 1 is a failure, not a system error


def test_promptfoo_repeat_with_one_error_merges_outputs(tmp_path: Path) -> None:
    rows = [
        _pf_row(0, "q", True, repeatGroupId="g", repeatIndex=0),
        _pf_row(1, "q", False, repeatGroupId="g", repeatIndex=1, failure=2, error="boom"),
    ]
    run = import_run("promptfoo", _jsonl(tmp_path / "m.jsonl", rows))
    result = run.results[0]
    assert result.output.error == "1 of 2 repeats errored: boom"
    assert result.scores["pass"].value == 0.5  # the errored repeat counts as a fail


def test_promptfoo_named_scores_and_non_finite(tmp_path: Path) -> None:
    # JSON has no NaN: promptfoo's own writer turns it into null, which is undecided.
    row = _pf_row(0, "q", True, named={"rel": None, "tone": 0.4})
    scores = import_run("promptfoo", _jsonl(tmp_path / "n.jsonl", [row])).results[0].scores
    assert scores["rel"].value is None and scores["rel"].passed is None
    assert scores["tone"].value == 0.4 and scores["tone"].passed is None
    # A literal NaN or Infinity is refused, and the message says where.
    for bad in (float("nan"), float("inf")):
        row = _pf_row(0, "q", True, named={"rel": bad})
        with pytest.raises(ImportFormatError, match=r"case '[0-9a-f]{16}', metric 'rel'"):
            import_run("promptfoo", _jsonl(tmp_path / "bad.jsonl", [row]))


def test_promptfoo_dataset_hash(tmp_path: Path) -> None:
    a = import_run("promptfoo", _pf_file(tmp_path / "a.jsonl", [True, False]))
    b = import_run("promptfoo", _pf_file(tmp_path / "b.jsonl", [False, True]))
    c = import_run("promptfoo", _pf_file(tmp_path / "c.jsonl", [True, False], prefix="z"))
    assert a.info.dataset_hash == b.info.dataset_hash  # results differ, the dataset does not
    assert a.info.dataset_hash != c.info.dataset_hash


def test_promptfoo_rejects_other_files(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope", "utf-8")
    with pytest.raises(ImportFormatError, match="not valid JSON"):
        import_run("promptfoo", bad)
    bad.write_text(json.dumps({"hello": 1}), "utf-8")
    with pytest.raises(ImportFormatError, match="not a promptfoo output"):
        import_run("promptfoo", bad)
    bad.write_text(json.dumps([{"testIdx": 0}]), "utf-8")
    with pytest.raises(ImportFormatError, match="success"):
        import_run("promptfoo", bad)
    with pytest.raises(ImportFormatError, match="jsonl and csv"):
        import_run("promptfoo", PF, wide=True)
    with pytest.raises(ImportFormatError, match="no results"):
        import_run("promptfoo", _jsonl(tmp_path / "empty.jsonl", []))


# --- Inspect -------------------------------------------------------------------------------


def _sample(sid: Any, epoch: int, scores: dict[str, Any], **kw: Any) -> dict[str, Any]:
    out = {
        "model": "m",
        "choices": [{"message": {"content": "ans"}}],
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }
    return {
        "id": sid,
        "epoch": epoch,
        "input": f"question {sid}",
        "target": "x",
        "output": out,
        "scores": scores,
        **kw,
    }


def _log(samples: list[dict[str, Any]], **over: Any) -> dict[str, Any]:
    log = json.loads(INSPECT.read_text("utf-8"))
    log["samples"] = samples
    log.update(over)
    return log


def _write_log(tmp_path: Path, log: dict[str, Any], name: str = "log.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(log), "utf-8")
    return path


def _write_eval(path: Path, log: dict[str, Any], compression: int = zipfile.ZIP_DEFLATED) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as zf:
        zf.writestr("header.json", json.dumps({k: v for k, v in log.items() if k != "samples"}))
        zf.writestr("_journal/start.json", "{}")
        zf.writestr("summaries.json", "[]")
        for s in log["samples"]:
            zf.writestr(f"samples/{s['id']}_epoch_{s['epoch']}.json", json.dumps(s))
    return path


def test_inspect_fixture() -> None:
    run = import_run("inspect", INSPECT)
    assert run.info.suite == "popularity" and run.info.system == "ollama/llama3.1"
    assert run.info.started_at == "2024-10-15T12:53:13-04:00"
    assert run.info.environment["source_schema"] == "2"
    assert run.info.environment["source_tool_version"].startswith("0.3.41")
    assert [r.case_id for r in run.results] == ["1"]
    assert _flags(run)["1"] == {"match": (0.0, False)}
    assert run.results[0].output.output == "No" and run.results[0].output.tokens_in == 66


def test_inspect_ids_hash_and_system_override(tmp_path: Path) -> None:
    log = _log([_sample(1, 1, {"m": {"value": "C"}}), _sample("b", 1, {"m": {"value": "I"}})])
    path = _write_log(tmp_path, log)
    a, b = import_run("inspect", path), import_run("inspect", path, system="mine")
    assert [r.case_id for r in a.results] == ["1", "b"] and b.info.system == "mine"
    assert a.info.dataset_hash == b.info.dataset_hash
    log["samples"][0]["input"] = "changed"
    changed = import_run("inspect", _write_log(tmp_path, log, "changed.json"))
    assert changed.info.dataset_hash != a.info.dataset_hash


def test_inspect_score_mapping(tmp_path: Path) -> None:
    values = {
        "a": "C", "b": "I", "c": "N", "d": "P", "e": True, "f": 0.7,
        "g": {"x": 1, "y": "C"}, "h": ["l"], "i": "weird",
    }  # fmt: skip
    scores: dict[str, Any] = {k: {"value": v, "explanation": "why"} for k, v in values.items()}
    scores["k"] = 0.25  # a bare value instead of a Score object
    notes: list[str] = []
    path = _write_log(tmp_path, _log([_sample(1, 1, scores)]))
    run = import_run("inspect", path, warn=notes.append)
    got = _flags(run)["1"]
    assert got == {
        "a": (1.0, True), "b": (0.0, False), "c": (0.0, False), "d": (0.5, None),
        "e": (1.0, True), "f": (0.7, None), "g.x": (1.0, None), "g.y": (1.0, True),
        "k": (0.25, None),
    }  # fmt: skip
    assert run.results[0].scores["a"].reason == "why"
    assert len(notes) == 2 and any("'h'" in n for n in notes) and any("'i'" in n for n in notes)


def test_inspect_rejects_non_finite_scores(tmp_path: Path) -> None:
    for name, value in (("f", float("nan")), ("g", {"x": float("-inf")})):
        log = _log([_sample(7, 1, {name: {"value": value}})])
        match = r"case '7', metric 'f'" if name == "f" else r"case '7', metric 'g\.x'"
        with pytest.raises(ImportFormatError, match=match):
            import_run("inspect", _write_log(tmp_path, log))


@pytest.mark.parametrize(
    ("epochs", "value", "passed"),
    [("mean", 0.5, None), ("first", 1.0, True), ("all", 0.0, False), ("any", 1.0, True)],
)
def test_inspect_epochs(tmp_path: Path, epochs: str, value: float, passed: bool | None) -> None:
    samples = [
        _sample(1, 1, {"m": {"value": "C"}, "f": {"value": 0.2}}),
        _sample(1, 2, {"m": {"value": "I"}, "f": {"value": 0.8}}),
    ]
    run = import_run("inspect", _write_log(tmp_path, _log(samples)), epochs=epochs)
    assert len(run.results) == 1
    m, f = run.results[0].scores["m"], run.results[0].scores["f"]
    assert m.value == value and m.passed is passed
    expected_f = {"mean": 0.5, "first": 0.2, "all": 0.2, "any": 0.8}[epochs]
    assert f.value == pytest.approx(expected_f)


def test_inspect_error_rows_status_and_latency(tmp_path: Path) -> None:
    bad = _sample(2, 1, {}, error={"message": "boom", "traceback": "tb"})
    timed = _sample(1, 1, {"m": {"value": "C"}}, working_time=1.5, total_time=9.0)
    total_only = _sample(3, 1, {"m": {"value": "C"}}, total_time=2.0)
    log = _log([timed, bad, total_only], status="cancelled")
    notes: list[str] = []
    run = import_run("inspect", _write_log(tmp_path, log), warn=notes.append)
    by_id = run.by_id()
    assert by_id["2"].output.error == "boom"
    assert by_id["2"].scores["m"].value == 0.0 and by_id["2"].scores["m"].passed is False
    assert by_id["1"].output.latency_ms == 1500 and by_id["3"].output.latency_ms == 2000
    assert run.info.environment["source_status"] == "cancelled"
    assert any("cancelled" in n for n in notes)


def test_inspect_eval_zip_matches_json(tmp_path: Path) -> None:
    samples = [_sample(i, 1, {"m": {"value": "C" if i % 2 else "I"}}) for i in (10, 2, 1)]
    samples.append(_sample("abc", 1, {"m": {"value": "C"}}))
    log = _log(samples)
    zipped = import_run("inspect", _write_eval(tmp_path / "log.eval", log))
    plain = import_run("inspect", _write_log(tmp_path, log))
    assert [r.case_id for r in zipped.results] == ["1", "2", "10", "abc"]  # numeric, then text
    assert {r.case_id: r for r in zipped.results} == {r.case_id: r for r in plain.results}
    assert zipped.info.dataset_hash == plain.info.dataset_hash


def test_inspect_zstd_needs_a_newer_python(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _write_eval(tmp_path / "log.eval", _log([_sample(1, 1, {})]))

    def unsupported(self: zipfile.ZipFile, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("That compression method is not supported")

    monkeypatch.setattr(zipfile.ZipFile, "open", unsupported)
    with pytest.raises(ImportFormatError, match=r"zstd-compressed.*3\.14.*inspect log convert"):
        import_run("inspect", path)


def test_inspect_rejects_other_files(tmp_path: Path) -> None:
    text = tmp_path / "notzip.eval"
    text.write_text("hello", "utf-8")
    with pytest.raises(ImportFormatError, match="not a zip"):
        import_run("inspect", text)
    empty = tmp_path / "empty.eval"
    with zipfile.ZipFile(empty, "w") as zf:
        zf.writestr("other.json", "{}")
    with pytest.raises(ImportFormatError, match=r"header.json"):
        import_run("inspect", empty)
    with pytest.raises(ImportFormatError, match="no 'eval' section"):
        import_run("inspect", _write_log(tmp_path, {"hello": 1}, "x.json"))
    nosamples = copy.deepcopy(_log([]))
    del nosamples["samples"]
    with pytest.raises(ImportFormatError, match="no samples"):
        import_run("inspect", _write_log(tmp_path, nosamples, "y.json"))
    with pytest.raises(ImportFormatError, match="jsonl and csv"):
        import_run("inspect", INSPECT, wide=True)


# --- DeepEval ------------------------------------------------------------------------------


def _de_doc(**over: Any) -> dict[str, Any]:
    doc = json.loads(DEEPEVAL.read_text("utf-8"))
    doc["testRunData"].update(over)
    return doc


def _de_write(tmp_path: Path, doc: dict[str, Any], name: str = "run.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(doc), "utf-8")
    return path


def test_deepeval_fixture() -> None:
    notes: list[str] = []
    run = import_run("deepeval", DEEPEVAL, warn=notes.append)
    assert run.info.system == "support-bot-v2" and run.info.suite == "test_support_bot"
    assert run.info.wall_time_s == 6.4
    assert any("conversational" in n for n in notes)
    unnamed = hashlib.sha256(b"Do you ship abroad?").hexdigest()[:16]
    assert [r.case_id for r in run.results] == [
        "refund-policy",
        "shipping-time",
        "cancel-order",
        unnamed,
    ]
    flags = _flags(run)
    assert flags["refund-policy"] == {
        "Answer Relevancy": (1.0, True),
        "Answer Relevancy:score": (0.92, None),
        "Faithfulness": (0.0, False),
        "Faithfulness:score": (0.55, None),
    }
    assert flags["cancel-order"]["Faithfulness"] == (None, None)  # the judge broke: undecided
    assert flags["cancel-order"]["Faithfulness:score"] == (None, None)
    refund = run.results[0]
    assert refund.output.latency_ms == 1800 and refund.output.retrieved == [
        "Refunds are available within 30 days."
    ]
    assert refund.scores["Answer Relevancy"].judge_cost_usd == pytest.approx(0.0004)
    assert all(r.output.error is None for r in run.results)


@pytest.mark.filterwarnings("ignore:skipped 1 conversational")
def test_deepeval_system_precedence(tmp_path: Path) -> None:
    def system(**over: Any) -> str:
        label = over.pop("label", None)
        return import_run(
            "deepeval", _de_write(tmp_path, _de_doc(**over)), system=label
        ).info.system

    assert system(label="mine") == "mine"
    assert system() == "support-bot-v2"
    assert system(identifier=None) == "gpt-4o-mini"
    assert system(identifier=None, hyperparameters=None) == "deepeval"


def test_deepeval_ids_and_hash(tmp_path: Path) -> None:
    def case(name: str, text: str) -> dict[str, Any]:
        return {"name": name, "input": text, "actualOutput": "o", "metricsData": []}

    cases = [case("dup", "a"), case("dup", "a"), case("", "b"), case("solo", "c")]
    path = _de_write(tmp_path, {"testCases": cases})  # a bare test run, no wrapper
    run = import_run("deepeval", path)
    h = lambda s: hashlib.sha256(s.encode()).hexdigest()[:16]  # noqa: E731
    assert [r.case_id for r in run.results] == [h("a"), f"{h('a')}#2", h("b"), "solo"]
    assert import_run("deepeval", path).info.dataset_hash == run.info.dataset_hash
    cases[3]["input"] = "changed"
    assert import_run(
        "deepeval", _de_write(tmp_path, {"testCases": cases}, "b.json")
    ).info.dataset_hash != (run.info.dataset_hash)


def test_deepeval_metric_edge_cases(tmp_path: Path) -> None:
    metrics = [
        {"name": "M", "success": True, "score": 0.9},
        {"name": "M", "success": False, "score": 0.1},
        {"name": "N", "score": 0.5},  # no verdict, no error
    ]
    case = {"name": "x", "input": "i", "actualOutput": "o", "metricsData": metrics}
    run = import_run("deepeval", _de_write(tmp_path, {"testCases": [case]}))
    assert _flags(run)["x"] == {
        "M": (1.0, True), "M:score": (0.9, None),
        "M#2": (0.0, False), "M#2:score": (0.1, None),
        "N": (None, None), "N:score": (0.5, None),
    }  # fmt: skip


def test_deepeval_rejects_other_files(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match="not a DeepEval test run"):
        import_run("deepeval", _de_write(tmp_path, {"hello": 1}))
    with pytest.raises(ImportFormatError, match="no cases"):
        import_run("deepeval", _de_write(tmp_path, {"testCases": []}, "e.json"))
    with pytest.raises(ImportFormatError, match="jsonl and csv"):
        import_run("deepeval", DEEPEVAL, wide=True)


def test_python_api_warns_by_default() -> None:
    with pytest.warns(UserWarning, match="conversational"):
        import_run("deepeval", DEEPEVAL)


# --- lm-eval -------------------------------------------------------------------------------


def _lm_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()]


def test_lm_eval_single_filter() -> None:
    run = import_run("lm-eval", HELLASWAG)
    assert run.info.system == "gpt2" and run.info.suite == "hellaswag"
    assert run.info.finished_at == "2024-05-01T10:20:30.123456"
    assert [r.case_id for r in run.results] == [f"hellaswag:{i}" for i in range(4)]
    assert _flags(run)["hellaswag:1"] == {"acc": (0.0, False), "acc_norm": (1.0, True)}
    assert import_run("lm-eval", HELLASWAG, system="renamed").info.system == "renamed"


def test_lm_eval_several_filters_become_metric_suffixes() -> None:
    run = import_run("lm-eval", GSM8K)
    assert len(run.results) == 3
    assert _flags(run)["gsm8k:1"] == {
        "exact_match,strict-match": (0.0, False),
        "exact_match,flexible-extract": (1.0, True),
    }


def test_lm_eval_ids_stable_and_hash_follows_doc_hash(tmp_path: Path) -> None:
    a, b = import_run("lm-eval", HELLASWAG), import_run("lm-eval", HELLASWAG)
    assert a == b
    rows = _lm_rows(HELLASWAG)
    rows[0]["doc_hash"] = "different"
    path = _jsonl(tmp_path / "samples_hellaswag_2024-05-01T10-20-30.123456.jsonl", rows)
    changed = import_run("lm-eval", path)
    assert changed.info.dataset_hash != a.info.dataset_hash
    assert [r.case_id for r in changed.results] == [r.case_id for r in a.results]


def test_lm_eval_metric_kinds(tmp_path: Path) -> None:
    row = _lm_rows(HELLASWAG)[0] | {
        "metrics": ["acc", "acc_norm", "f1", "exact_match", "bleu", "ppl"],
        "acc": 0.5, "acc_norm": True, "f1": 1.0, "exact_match": 0.0, "bleu": "text", "ppl": None,
    }  # fmt: skip
    notes: list[str] = []
    run = import_run("lm-eval", _jsonl(tmp_path / "s.jsonl", [row]), warn=notes.append)
    assert run.info.suite == "s" and run.info.finished_at == ""  # a name the harness did not write
    assert _flags(run)[f"s:{row['doc_id']}"] == {
        "acc": (0.5, None), "acc_norm": (1.0, True), "f1": (1.0, None),
        "exact_match": (0.0, False),
    }  # fmt: skip
    assert len(notes) == 2 and "'bleu'" in notes[0] and "'ppl'" in notes[1]


@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), float("nan")])
def test_lm_eval_rejects_non_finite_scores(tmp_path: Path, bad: float) -> None:
    row = _lm_rows(HELLASWAG)[0] | {"metrics": ["acc", "perplexity"], "acc": 1, "perplexity": bad}
    with pytest.raises(ImportFormatError, match=r"case 's:0', metric 'perplexity'.*(inf|nan)"):
        import_run("lm-eval", _jsonl(tmp_path / "s.jsonl", [row]))


def test_lm_eval_rejects_bad_files(tmp_path: Path) -> None:
    row = _lm_rows(HELLASWAG)[0]
    with pytest.raises(ImportFormatError, match="appears twice"):
        import_run("lm-eval", _jsonl(tmp_path / "a.jsonl", [row, row]))
    with pytest.raises(ImportFormatError, match="log_samples"):
        import_run("lm-eval", _jsonl(tmp_path / "b.jsonl", [{"hello": 1}]))
    with pytest.raises(ImportFormatError, match="no samples"):
        import_run("lm-eval", _jsonl(tmp_path / "c.jsonl", []))
    with pytest.raises(ImportFormatError, match="each line"):
        (tmp_path / "d.jsonl").write_text("[1]\n", "utf-8")
        import_run("lm-eval", tmp_path / "d.jsonl")
    with pytest.raises(ImportFormatError, match="jsonl and csv"):
        import_run("lm-eval", HELLASWAG, wide=True)


# --- jsonl and csv -------------------------------------------------------------------------


def test_long_csv_systems_and_error_rows() -> None:
    with pytest.raises(ImportFormatError, match=r"2 systems.*'a', 'b'"):
        import_run("csv", LONG_CSV)
    with pytest.raises(ImportFormatError, match="no system 'z'"):
        import_run("csv", LONG_CSV, system="z")
    run = import_run("csv", LONG_CSV, system="a")
    assert run.info.system == "a" and run.info.suite == "long"
    flags = _flags(run)
    assert flags["t1"] == {"correct": (1.0, True), "tone": (0.8, None)}
    assert flags["t2"] == {"correct": (0.0, False), "tone": (0.4, None)}
    assert flags["t3"] == {"correct": (0.0, False), "tone": (0.0, False)}  # error: fails both
    t3 = run.by_id()["t3"]
    assert t3.output.error == "timeout" and run.by_id()["t1"].output.latency_ms == 120
    assert run.by_id()["t1"].output.output == "x"


def test_csv_and_jsonl_agree_and_hash_is_stable() -> None:
    from_csv = import_run("csv", LONG_CSV, system="b")
    from_jsonl = import_run("jsonl", LONG_JSONL, system="b")
    assert from_csv.results == from_jsonl.results
    assert from_csv.info.dataset_hash == from_jsonl.info.dataset_hash
    assert from_csv.info.environment["imported_from"] == "csv"
    assert from_jsonl.info.environment["imported_from"] == "jsonl"
    # The same cases under the other system share the dataset hash: they compare.
    assert import_run("csv", LONG_CSV, system="a").info.dataset_hash == from_csv.info.dataset_hash


def _csv(tmp_path: Path, text: str, name: str = "t.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, "utf-8")
    return path


@pytest.mark.parametrize(
    ("epochs", "t1", "t2", "t3"),
    [
        ("mean", (0.5, None), (0.0, None), (None, None)),
        ("first", (1.0, True), (0.0, False), (None, None)),
        ("all", (0.0, False), (0.0, False), (None, None)),
        ("any", (1.0, True), (None, None), (None, None)),
    ],
)
def test_table_repeats_and_undecided(
    tmp_path: Path, epochs: str, t1: Any, t2: Any, t3: Any
) -> None:
    text = (
        "id,metric,passed,repeat\n"
        "t1,m,true,1\nt1,m,false,2\n"
        "t2,m,false,1\nt2,m,,2\n"
        "t3,m,,1\nt3,m,,2\n"
    )
    run = import_run("csv", _csv(tmp_path, text), epochs=epochs)
    assert [_flags(run)[c]["m"] for c in ("t1", "t2", "t3")] == [t1, t2, t3]


def test_table_continuous_repeats_all_is_min_any_is_max(tmp_path: Path) -> None:
    text = "id,metric,score,repeat\nt,m,0.2,1\nt,m,0.8,2\n"
    path = _csv(tmp_path, text)
    assert import_run("csv", path, epochs="all").results[0].scores["m"].value == 0.2
    assert import_run("csv", path, epochs="any").results[0].scores["m"].value == 0.8


def test_table_rejects_duplicates_and_bad_cells(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match=r"twice.*repeat column"):
        import_run("csv", _csv(tmp_path, "id,metric,passed\nt,m,true\nt,m,false\n"))
    with pytest.raises(ImportFormatError, match="true/false/1/0/yes/no"):
        import_run("csv", _csv(tmp_path, "id,metric,passed\nt,m,maybe\n", "b.csv"))
    with pytest.raises(ImportFormatError, match="not a number"):
        import_run("csv", _csv(tmp_path, "id,metric,score\nt,m,high\n", "c.csv"))
    with pytest.raises(ImportFormatError, match="empty 'id'"):
        import_run("csv", _csv(tmp_path, "id,metric,passed\n,m,true\n", "d.csv"))
    with pytest.raises(ImportFormatError, match="no rows"):
        import_run("csv", _csv(tmp_path, "id,metric,passed\n", "e.csv"))


@pytest.mark.parametrize("bad", ["nan", "NaN", "inf", "-inf", "1e999"])
def test_table_rejects_non_finite_scores(tmp_path: Path, bad: str) -> None:
    with pytest.raises(ImportFormatError, match=r"case 'b', metric 'm'"):
        import_run("csv", _csv(tmp_path, f"id,metric,score\na,m,0.5\nb,m,{bad}\n"))
    # The same cell in a wide table, and in a JSON row.
    with pytest.raises(ImportFormatError, match=r"case 'b', metric 'quality'"):
        import_run("csv", _csv(tmp_path, f"id,quality\na,0.5\nb,{bad}\n", "w.csv"), wide=True)


def test_table_jsonl_rejects_nan_and_an_empty_score_stays_undecided(tmp_path: Path) -> None:
    rows = [{"id": "a", "metric": "m", "score": float("nan")}]
    with pytest.raises(ImportFormatError, match=r"case 'a', metric 'm'.*nan"):
        import_run("jsonl", _jsonl(tmp_path / "n.jsonl", rows))
    run = import_run("csv", _csv(tmp_path, "id,metric,score\na,m,0.5\nb,m,\n", "e.csv"))
    assert [_flags(run)[c]["m"] for c in "ab"] == [(0.5, None), (None, None)]


def test_deepeval_rejects_a_non_finite_metric_score(tmp_path: Path) -> None:
    metric = {"name": "Faithfulness", "success": True, "score": float("nan")}
    case = {"name": "x", "input": "i", "actualOutput": "o", "metricsData": [metric]}
    with pytest.raises(ImportFormatError, match=r"case 'x', metric 'Faithfulness:score'"):
        import_run("deepeval", _de_write(tmp_path, {"testCases": [case]}))


def test_table_missing_columns_name_the_flag(tmp_path: Path) -> None:
    with pytest.raises(ImportFormatError, match="--wide"):
        import_run("csv", _csv(tmp_path, "id,passed\nt,true\n"))
    with pytest.raises(ImportFormatError, match="--id-field"):
        import_run("csv", _csv(tmp_path, "case,metric,passed\nt,m,true\n", "b.csv"))
    with pytest.raises(ImportFormatError, match="--passed-field"):
        import_run("csv", _csv(tmp_path, "id,metric,verdict\nt,m,true\n", "c.csv"))
    with pytest.raises(ImportFormatError, match="unknown field"):
        import_run("csv", LONG_CSV, system="a", fields={"bogus": "x"})
    with pytest.raises(ImportFormatError, match="unknown format"):
        import_run("xml", LONG_CSV)
    with pytest.raises(ImportFormatError, match="epochs must be"):
        import_run("csv", LONG_CSV, system="a", epochs="median")


def test_table_field_remapping_and_system_label(tmp_path: Path) -> None:
    text = "case,name,ok,val,who\nt1,m,yes,0.9,A\nt1,n,0,,A\nt1,m,no,0.1,B\n"
    fields = {"id": "case", "metric": "name", "passed": "ok", "score": "val", "system": "who"}
    run = import_run("csv", _csv(tmp_path, text), system="A", fields=fields)
    assert _flags(run)["t1"] == {"m": (0.9, True), "n": (0.0, False)}
    # Without a system column, --system is only the label; the default is the file name.
    plain = _csv(tmp_path, "id,metric,passed\nt,m,1\n", "plain.csv")
    assert import_run("csv", plain).info.system == "plain"
    assert import_run("csv", plain, system="mine").info.system == "mine"


def test_wide_csv_and_jsonl(tmp_path: Path) -> None:
    run = import_run("csv", WIDE_CSV, wide=True, fields={"id": "case"})
    assert _flags(run) == {
        "t1": {"correct": (1.0, True), "tone": (0.8, None)},
        "t2": {"correct": (0.0, False), "tone": (0.4, None)},
        "t3": {"correct": (1.0, True), "tone": (None, None)},  # an empty cell is undecided
    }
    rows = [
        {"id": "a", "system": "s", "ok": True, "val": 0.5, "output": {"k": 1}},
        {"id": "b", "system": "s", "ok": 0, "val": 0.25, "error": "boom"},
    ]
    wide = import_run("jsonl", _jsonl(tmp_path / "w.jsonl", rows), wide=True)
    assert _flags(wide)["a"] == {"ok": (1.0, True), "val": (0.5, None)}
    assert wide.by_id()["b"].output.error == "boom" and wide.by_id()["a"].output.output == '{"k":1}'
    with pytest.raises(ImportFormatError, match="do not apply to --wide"):
        import_run("csv", WIDE_CSV, wide=True, fields={"id": "case", "score": "tone"})
    with pytest.raises(ImportFormatError, match="at least one metric"):
        import_run("csv", _csv(tmp_path, "id,output\na,x\n"), wide=True)


def test_every_format_is_registered() -> None:
    assert FORMATS == ("promptfoo", "inspect", "deepeval", "lm-eval", "jsonl", "csv")


# --- CLI and end to end --------------------------------------------------------------------


def test_cli_import_writes_a_loadable_run(tmp_path: Path, capsys) -> None:
    out, md = tmp_path / "run.json", tmp_path / "report.md"
    argv = ["import", "promptfoo", str(PF), "--system", "openai:gpt-5", "--out", str(out)]
    assert main([*argv, "--markdown", str(md)]) == EXIT_PASS
    captured = capsys.readouterr()
    assert f"wrote {out}" in captured.err and "pass" in captured.out
    assert load_run(out).info.system == "openai:gpt-5" and "pass" in md.read_text("utf-8")


def test_cli_import_errors_exit_2(tmp_path: Path, capsys) -> None:
    out = str(tmp_path / "run.json")
    assert main(["import", "promptfoo", str(PF), "--out", out]) == EXIT_USAGE
    assert "choose one with --system" in capsys.readouterr().err
    assert main(["import", "csv", str(tmp_path / "missing.csv"), "--out", out]) == EXIT_USAGE
    with pytest.raises(SystemExit) as exc:
        main(["import", "nonsense", str(PF), "--out", out])
    assert exc.value.code == 2
    assert not (tmp_path / "run.json").exists()


def test_cli_import_warnings_go_to_stderr_and_table_options_work(tmp_path: Path, capsys) -> None:
    out = tmp_path / "d.json"
    assert main(["import", "deepeval", str(DEEPEVAL), "--out", str(out), "--epochs", "all"]) == 0
    assert "warning: skipped 1 conversational" in capsys.readouterr().err
    wide = tmp_path / "w.json"
    argv = ["import", "csv", str(WIDE_CSV), "--wide", "--id-field", "case", "--out", str(wide)]
    assert main(argv) == EXIT_PASS
    assert [r.case_id for r in load_run(wide).results] == ["t1", "t2", "t3"]


def test_import_then_compare_then_gate(tmp_path: Path) -> None:
    """Two promptfoo files, the candidate with `success` flipped on 12 of 40 tests."""
    runs = {}
    for name, successes in {
        "base": [True] * 40,
        "worse": [False] * 12 + [True] * 28,
        "same": [False] + [True] * 39,
    }.items():
        src = _pf_file(tmp_path / f"{name}-in.jsonl", successes)
        runs[name] = tmp_path / f"{name}.json"
        assert main(["import", "promptfoo", str(src), "--out", str(runs[name])]) == EXIT_PASS
    gate = ["--gate", "--metric", "pass", "--min-cases", "1"]
    assert main(["compare", str(runs["base"]), str(runs["worse"]), *gate]) == EXIT_REGRESSION
    # One flipped test is a drop, but not one that can be told from noise.
    assert main(["compare", str(runs["base"]), str(runs["same"]), *gate]) == EXIT_PASS
    # A different dataset is not comparable at all.
    other = _pf_file(tmp_path / "other-in.jsonl", [True] * 40, prefix="z")
    assert main(["import", "promptfoo", str(other), "--out", str(tmp_path / "other.json")]) == 0
    assert (
        main(["compare", str(runs["base"]), str(tmp_path / "other.json"), *gate])
        == EXIT_CANNOT_GATE
    )
