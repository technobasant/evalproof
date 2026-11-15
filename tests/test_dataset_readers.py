"""Test-case importers: ``import_cases`` on every format, detection, ids, tags, assertions.

The fixtures under ``tests/fixtures/datasets`` are hand-written from each tool's documented
shape, not copied from real projects, so every format but promptfoo, jsonl and csv is marked
experimental. Nothing here calls a model or the network.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from evalproof.dataset import dataset_text, load_dataset
from evalproof.importers import (
    DATASET_FORMATS,
    EXPERIMENTAL_DATASET_FORMATS,
    ImportedDataset,
    ImportFormatError,
    detect_format,
    import_cases,
    import_dataset,
)
from evalproof.models import Case

DATASETS = Path(__file__).parent / "fixtures" / "datasets"
ADK = Path(__file__).parent / "fixtures" / "adk"


def load(path: Path, fmt: str = "auto", **options: Any) -> ImportedDataset:
    return import_cases(fmt, path, warn=lambda message: None, **options)


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    text = "".join(json.dumps(r) + "\n" for r in records)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def dump(case: Case) -> dict[str, Any]:
    return case.model_dump(mode="json", exclude_defaults=True)


def by_id(dataset: ImportedDataset) -> dict[str, Case]:
    return {c.id: c for c in dataset.cases}


# --- detection --------------------------------------------------------------------------------

DETECTED = [
    ("promptfoo/promptfooconfig.yaml", "promptfoo", 4),
    ("promptfoo/tests.csv", "promptfoo", 3),
    ("promptfoo/tests.jsonl", "promptfoo", 2),
    ("deepeval/goldens.json", "deepeval", 3),
    ("deepeval/goldens.csv", "deepeval", 2),
    ("deepeval/multi_turn.json", "deepeval", 1),
    ("ragas/samples.jsonl", "ragas", 3),
    ("ragas/legacy.jsonl", "ragas", 2),
    ("langsmith/examples.jsonl", "langsmith", 3),
    ("openai-evals/samples.jsonl", "openai-evals", 3),
    ("openai-evals/platform.jsonl", "openai-evals", 2),
    ("inspect/samples.jsonl", "inspect", 3),
    ("inspect/samples.csv", "inspect", 2),
    ("braintrust/records.jsonl", "braintrust", 3),
    ("phoenix/examples.jsonl", "phoenix", 3),
]


@pytest.mark.parametrize(("name", "fmt", "count"), DETECTED)
def test_auto_detects_every_fixture(name: str, fmt: str, count: int) -> None:
    result = load(DATASETS / name)
    assert result.format == fmt
    assert result.detected is True
    assert len(result.cases) == count
    assert all(f"from:{fmt}" in c.tags and "imported" in c.tags for c in result.cases)


@pytest.mark.parametrize(("name", "fmt", "count"), DETECTED)
def test_naming_the_format_gives_the_same_cases_as_auto(name: str, fmt: str, count: int) -> None:
    named = load(DATASETS / name, fmt)
    assert named.detected is False
    assert named.cases == load(DATASETS / name).cases
    assert len(named.cases) == count


def test_a_langfuse_export_is_detected_as_deepeval_and_the_note_names_langfuse() -> None:
    auto = load(DATASETS / "langfuse" / "items.json")
    assert auto.format == "deepeval"
    assert any("langfuse" in note for note in auto.notes)
    named = load(DATASETS / "langfuse" / "items.json", "langfuse")
    assert named.format == "langfuse"
    assert [c.id for c in named.cases] == ["lf-1", "lf-2"]  # the item id is the case id
    assert all("from:langfuse" in c.tags for c in named.cases)


def test_an_adk_evalset_is_detected_as_adk() -> None:
    path = ADK / "basic_criteria.home_automation.evalset.json"
    result = load(path)
    assert result.format == "adk"
    assert result.detected is True
    assert [c.id for c in result.cases] == [c.id for c in import_dataset("adk", path)]


def test_detect_format_names_a_single_candidate() -> None:
    assert detect_format(DATASETS / "ragas" / "samples.jsonl", DATASET_FORMATS) == "ragas"


def test_an_ambiguous_file_lists_the_formats_and_asks_for_one(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "mixed.jsonl", [{"input": "q", "target": "t", "expected": "e", "ideal": "i"}]
    )
    with pytest.raises(ImportFormatError) as raised:
        load(path)
    message = str(raised.value)
    assert "fits more than one format" in message
    assert "Name the format" in message
    for fmt in ("promptfoo", "ragas", "jsonl", "csv"):
        assert fmt in message


def test_an_unknown_file_asks_for_a_format(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "odd.jsonl", [{"unrelated": 1}])
    with pytest.raises(ImportFormatError, match="cannot tell which tool wrote this file"):
        load(path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ImportFormatError, match="Name the format"):
        load(empty)


def test_a_missing_file_or_format_is_an_import_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):  # the command turns this into exit 2
        load(tmp_path / "nope.jsonl", "ragas")
    with pytest.raises(ImportFormatError, match="unknown"):
        load(DATASETS / "ragas" / "samples.jsonl", "lm-eval")


def test_the_experimental_set_is_every_new_format_but_promptfoo_jsonl_and_csv() -> None:
    assert set(EXPERIMENTAL_DATASET_FORMATS) == {
        "deepeval",
        "ragas",
        "langsmith",
        "openai-evals",
        "inspect",
        "braintrust",
        "langfuse",
        "phoenix",
    }
    assert not {"promptfoo", "jsonl", "csv", "adk", "auto"} & set(EXPERIMENTAL_DATASET_FORMATS)
    assert DATASET_FORMATS[0] == "auto"
    assert {"promptfoo", "jsonl", "csv", "adk", *EXPERIMENTAL_DATASET_FORMATS} <= set(
        DATASET_FORMATS
    )


# --- ids and tags -----------------------------------------------------------------------------


def test_source_ids_are_kept_and_the_rest_are_a_content_hash() -> None:
    cases = by_id(load(DATASETS / "deepeval" / "goldens.json"))
    assert "refund-window" in cases
    assert "export-formats" in cases
    hashed = [i for i in cases if i.startswith("deepeval-")]
    assert len(hashed) == 1
    assert len(hashed[0].removeprefix("deepeval-")) == 10
    int(hashed[0].removeprefix("deepeval-"), 16)  # ten hex digits


def test_hashed_ids_do_not_depend_on_line_order(tmp_path: Path) -> None:
    records = [
        {"input": "one", "expected_output": "1"},
        {"input": "two", "expected_output": "2"},
        {"input": "three", "expected_output": "3"},
    ]
    forward = load(write_jsonl(tmp_path / "a.jsonl", records), "deepeval")
    backward = load(write_jsonl(tmp_path / "b.jsonl", records[::-1]), "deepeval")
    assert {c.id: c.input for c in forward.cases} == {c.id: c.input for c in backward.cases}
    assert len({c.id for c in forward.cases}) == 3


def test_a_hashed_id_follows_the_input_and_the_expected_answer_only(tmp_path: Path) -> None:
    a = write_jsonl(tmp_path / "a.jsonl", [{"input": "q", "expected_output": "a"}])
    b = write_jsonl(
        tmp_path / "b.jsonl",
        [{"input": "q", "expected_output": "a", "tags": ["extra"], "context": ["c"]}],
    )
    c = write_jsonl(tmp_path / "c.jsonl", [{"input": "q", "expected_output": "other"}])
    assert load(a, "deepeval").cases[0].id == load(b, "deepeval").cases[0].id
    assert load(a, "deepeval").cases[0].id != load(c, "deepeval").cases[0].id


def test_duplicate_ids_are_numbered_and_the_note_says_so(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "dup.jsonl",
        [
            {"qid": "a", "input": "one", "answer": "1"},
            {"qid": "a", "input": "two", "answer": "2"},
            {"qid": "a", "input": "three", "answer": "3"},
            {"qid": "b", "input": "four", "answer": "4"},
        ],
    )
    result = load(path, "jsonl", id_field="qid", reference_field="answer")
    assert [c.id for c in result.cases] == ["a", "a-2", "a-3", "b"]
    assert any("repeated case id" in n and "'a'" in n for n in result.notes)


def test_identical_records_without_ids_are_numbered_too(tmp_path: Path) -> None:
    record = {"input": "one", "expected_output": "1"}
    result = load(write_jsonl(tmp_path / "same.jsonl", [record, record]), "deepeval")
    first, second = (c.id for c in result.cases)
    assert second == f"{first}-2"
    assert any("repeated case id" in n for n in result.notes)


def test_tags_are_the_sources_tags_plus_imported_and_the_format() -> None:
    cases = by_id(load(DATASETS / "promptfoo" / "promptfooconfig.yaml"))
    assert cases["refund-window"].tags == ["billing", "imported", "from:promptfoo"]
    assert cases["free-plan-notebooks"].tags == ["support", "imported", "from:promptfoo"]
    csv_cases = by_id(load(DATASETS / "promptfoo" / "tests.csv"))
    assert csv_cases["free-plan"].tags == ["plans", "support", "imported", "from:promptfoo"]
    inspect_cases = by_id(load(DATASETS / "inspect" / "samples.jsonl"))
    assert inspect_cases["1"].tags == ["geography", "imported", "from:inspect"]
    braintrust = by_id(load(DATASETS / "braintrust" / "records.jsonl"))
    assert braintrust["bt-1"].tags == ["billing", "support", "imported", "from:braintrust"]
    phoenix = by_id(load(DATASETS / "phoenix" / "examples.jsonl"))
    assert phoenix["px-1"].tags == ["train", "imported", "from:phoenix"]
    langsmith = by_id(load(DATASETS / "langsmith" / "examples.jsonl"))
    assert langsmith["a1b2c3d4-0001"].tags == ["base", "imported", "from:langsmith"]


def test_the_tags_field_reads_a_list_or_a_comma_separated_string(tmp_path: Path) -> None:
    path = write_jsonl(
        tmp_path / "t.jsonl",
        [
            {"input": "a", "labels": ["x", "y"]},
            {"input": "b", "labels": "p, q"},
            {"input": "c", "labels": "solo"},
            {"input": "d"},
        ],
    )
    result = load(path, "jsonl", tags_field="labels")
    assert [c.tags for c in result.cases] == [
        ["x", "y", "imported", "from:jsonl"],
        ["p", "q", "imported", "from:jsonl"],
        ["solo", "imported", "from:jsonl"],
        ["imported", "from:jsonl"],
    ]
    plain = load(path, "jsonl")
    assert all(c.tags == ["imported", "from:jsonl"] for c in plain.cases)


# --- generic jsonl and csv --------------------------------------------------------------------


def test_the_generic_jsonl_reader_maps_the_named_columns() -> None:
    result = load(
        DATASETS / "jsonl" / "qa.jsonl",
        "jsonl",
        id_field="qid",
        input_fields=("question", "context"),
        reference_field="answer",
        tags_field="labels",
    )
    assert [c.id for c in result.cases] == ["q-1", "q-2"]
    assert result.cases[0].input == {
        "question": "Can I get a refund?",
        "context": "Refunds within 30 days.",
    }
    assert result.cases[0].reference == "Yes, within 30 days."
    assert result.cases[0].tags == ["billing", "support", "imported", "from:jsonl"]
    assert result.prompt_template == "question: {question}\ncontext: {context}"
    assert [(s.name, s.type) for s in result.scorers] == [("judge", "llm_judge")]


def test_the_generic_csv_reader_maps_the_named_columns() -> None:
    result = load(
        DATASETS / "csv" / "qa.csv",
        "csv",
        id_field="qid",
        input_fields=("question",),
        reference_field="answer",
        tags_field="labels",
    )
    assert [c.id for c in result.cases] == ["q-1", "q-2"]
    assert result.cases[0].input == "Can I get a refund?"  # one column stays text
    assert result.cases[1].reference == "Pro costs $8 per month."
    assert result.cases[1].tags == ["plans", "imported", "from:jsonl".replace("jsonl", "csv")]
    assert result.prompt_template is None


def test_the_generic_readers_default_to_an_input_column(tmp_path: Path) -> None:
    path = tmp_path / "plain.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows([["input", "reference"], ["hello", "world"]])
    result = load(path, "csv", reference_field="reference")
    assert [(c.input, c.reference) for c in result.cases] == [("hello", "world")]


def test_the_generic_readers_name_the_columns_they_found(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "q.jsonl", [{"q": "a", "answer": "z"}])
    with pytest.raises(ImportFormatError, match=r"no column 'zzz' \(columns found: answer, q\)"):
        load(path, "jsonl", input_fields=("zzz",))
    with pytest.raises(ImportFormatError, match="--input-field"):
        load(path, "jsonl")  # no `input` column and no --input-field
    with pytest.raises(ImportFormatError, match="no column 'nope'"):
        load(path, "jsonl", input_fields=("q",), reference_field="nope")


def test_the_field_options_belong_to_jsonl_and_csv_only() -> None:
    with pytest.raises(ImportFormatError, match="apply to the jsonl and csv formats, not ragas"):
        load(DATASETS / "ragas" / "samples.jsonl", "ragas", id_field="x")
    with pytest.raises(ImportFormatError, match="not ragas"):
        load(DATASETS / "ragas" / "samples.jsonl", "auto", reference_field="x")


# --- the other readers ------------------------------------------------------------------------


def test_free_text_answers_become_a_reference_graded_by_one_judge() -> None:
    for name, fmt in (
        ("deepeval/goldens.json", "deepeval"),
        ("ragas/samples.jsonl", "ragas"),
        ("langsmith/examples.jsonl", "langsmith"),
        ("braintrust/records.jsonl", "braintrust"),
        ("phoenix/examples.jsonl", "phoenix"),
    ):
        result = load(DATASETS / name, fmt)
        assert [(s.name, s.type, s.params) for s in result.scorers] == [("judge", "llm_judge", {})]
        assert result.needs_judge is True
        assert result.cases[0].reference


def test_openai_evals_reads_one_ideal_as_exact_and_a_list_as_contains() -> None:
    result = load(DATASETS / "openai-evals" / "samples.jsonl")
    assert [(s.name, s.type, s.params) for s in result.scorers] == [
        ("exact", "exact_match", {}),
        ("contains", "contains", {"any_of_field": "accept"}),
    ]
    assert result.needs_judge is False
    colours = next(c for c in result.cases if c.input == "Name a primary colour.")
    assert dump(colours)["accept"] == ["red", "blue", "yellow"]
    assert any("does not record how the eval compares answers" in n for n in result.notes)
    # the Platform export wraps each row in `item`
    platform = load(DATASETS / "openai-evals" / "platform.jsonl")
    assert [c.reference for c in platform.cases] == ["Paris", "Tokyo"]


def test_inspect_choices_are_appended_and_the_target_letter_is_the_reference() -> None:
    cases = by_id(load(DATASETS / "inspect" / "samples.jsonl"))
    planet = cases["q2"]
    assert isinstance(planet.input, str)
    assert "A) Mars" in planet.input
    assert "B) Jupiter" in planet.input
    assert "C) Venus" in planet.input
    assert "letter" in planet.input
    assert planet.reference == "B"
    colours = next(c for c in cases.values() if c.input == "Name a primary colour.")
    assert dump(colours)["accept"] == ["red", "blue", "yellow"]
    # one string target is exact; free text without choices in a CSV goes to the judge
    csv_result = load(DATASETS / "inspect" / "samples.csv")
    assert [s.type for s in csv_result.scorers] == ["llm_judge"]


def test_ragas_old_and_new_column_names_both_import() -> None:
    new = load(DATASETS / "ragas" / "samples.jsonl")
    old = load(DATASETS / "ragas" / "legacy.jsonl")
    assert [c.input for c in old.cases] == ["Can I get a refund?", "What is the Pro price?"]
    assert old.cases[0].reference == "Yes, within 30 days of purchase."
    assert new.cases[0].reference == "Yes, within 30 days of purchase."
    assert any("several ground truths" in n for n in old.notes)


def test_a_deepeval_scenario_without_turns_is_skipped_with_a_note() -> None:
    result = load(DATASETS / "deepeval" / "multi_turn.json")
    assert len(result.cases) == 1
    assert any("user simulation" in n for n in result.notes)


# --- chat handling ----------------------------------------------------------------------------


def test_chat_system_messages_are_dropped_and_user_only_chats_become_turns(
    tmp_path: Path,
) -> None:
    path = write_jsonl(
        tmp_path / "chat.jsonl",
        [
            {"input": [{"role": "system", "content": "S"}, {"role": "user", "content": "a"}],
             "ideal": "x"},
            {"input": [{"role": "user", "content": "a"}, {"role": "user", "content": "b"}],
             "ideal": "y"},
            {"input": [
                {"role": "user", "content": "a"},
                {"role": "assistant", "content": "m"},
                {"role": "user", "content": "b"},
            ], "ideal": "z"},
        ],
    )  # fmt: skip
    result = load(path, "openai-evals")
    first, second, third = result.cases
    assert first.input == "a"
    assert dump(first).get("turns") is None
    assert second.input == "a"
    assert dump(second)["turns"] == ["b"]
    assert third.input == {
        "messages": [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "m"},
            {"role": "user", "content": "b"},
        ]
    }
    assert any("system messages were dropped from 1 case" in n for n in result.notes)
    assert any('{"messages": [...]}' in n for n in result.notes)


# --- promptfoo assertions ---------------------------------------------------------------------


def promptfoo(
    tmp_path: Path, assertions: list[dict[str, Any]], variables: dict[str, Any] | None = None
) -> ImportedDataset:
    test = {"vars": variables or {"q": "hi"}, "assert": assertions}
    path = tmp_path / "suite.json"
    path.write_text(json.dumps({"tests": [test]}), encoding="utf-8")
    return load(path, "promptfoo")


def fields(result: ImportedDataset) -> dict[str, Any]:
    out = dump(result.cases[0])
    for key in ("id", "input", "tags"):
        out.pop(key, None)
    return out


def scorers(result: ImportedDataset) -> list[tuple[str, str, dict[str, Any]]]:
    return [(s.name, s.type, s.params) for s in result.scorers]


CONTAINS_CS = ("contains", "contains", {"any_of_field": "accept", "case_sensitive": True})
CONTAINS_CI = ("contains", "contains", {"any_of_field": "accept"})
REQUIRE_CS = ("contains", "contains", {"all_of_field": "require", "case_sensitive": True})
REQUIRE_CI = ("contains", "contains", {"all_of_field": "require"})
REGEX = ("regex", "regex", {"any_of_field": "pattern", "flags": ""})
OMITS = ("omits", "checks", {"field": "expect", "on_missing": "skip"})
RUBRICS = ("rubrics", "llm_judge", {"case_rubrics": True})
SCHEMA = ("json_schema", "json_schema", {"schema_field": "schema"})

MAPPED: list[tuple[dict[str, Any], dict[str, Any], tuple[str, str, dict[str, Any]]]] = [
    ({"type": "equals", "value": "Paris"}, {"reference": "Paris"}, ("exact", "exact_match", {})),
    ({"type": "contains", "value": "Par"}, {"accept": ["Par"]}, CONTAINS_CS),
    ({"type": "icontains", "value": "par"}, {"accept": ["par"]}, CONTAINS_CI),
    ({"type": "contains-any", "value": ["a", "b"]}, {"accept": ["a", "b"]}, CONTAINS_CS),
    ({"type": "icontains-any", "value": ["a", "b"]}, {"accept": ["a", "b"]}, CONTAINS_CI),
    ({"type": "contains-all", "value": ["a", "b"]}, {"require": ["a", "b"]}, REQUIRE_CS),
    ({"type": "icontains-all", "value": ["a", "b"]}, {"require": ["a", "b"]}, REQUIRE_CI),
    ({"type": "starts-with", "value": "Hel.lo"}, {"pattern": ["^Hel\\.lo"]}, REGEX),
    ({"type": "regex", "value": "^\\d+$"}, {"pattern": ["^\\d+$"]}, REGEX),
    (
        {"type": "not-contains", "value": "bad"},
        {"expect": [{"type": "output_omits", "text": "bad", "case_sensitive": True}]},
        OMITS,
    ),
    (
        {"type": "not-icontains", "value": "bad"},
        {"expect": [{"type": "output_omits", "text": "bad"}]},
        OMITS,
    ),
    (
        {"type": "not-contains-any", "value": ["x", "y"]},
        {
            "expect": [
                {"type": "output_omits", "text": "x", "case_sensitive": True},
                {"type": "output_omits", "text": "y", "case_sensitive": True},
            ]
        },
        OMITS,
    ),
    (
        {"type": "not-regex", "value": "\\d"},
        {"expect": [{"type": "not", "check": {"type": "output_regex", "pattern": "\\d"}}]},
        OMITS,
    ),
    (
        {"type": "not-starts-with", "value": "No"},
        {"expect": [{"type": "not", "check": {"type": "output_regex", "pattern": "^No"}}]},
        OMITS,
    ),
    (
        {"type": "not-equals", "value": "no"},
        {"expect": [{"type": "not", "check": {"type": "output_regex", "pattern": "\\Ano\\Z"}}]},
        OMITS,
    ),
    (
        {"type": "is-json", "value": {"type": "object", "required": ["a"]}},
        {"schema": {"required": ["a"], "type": "object"}},
        SCHEMA,
    ),
    (
        {"type": "contains-json", "value": {"type": "object"}},
        {"schema": {"type": "object"}},
        SCHEMA,
    ),
    ({"type": "llm-rubric", "value": "Is polite"}, {"rubrics": ["Is polite"]}, RUBRICS),
    ({"type": "g-eval", "value": "fluent"}, {"rubrics": ["fluent"]}, RUBRICS),
    ({"type": "model-graded-closedqa", "value": "on topic"}, {"rubrics": ["on topic"]}, RUBRICS),
    (
        {"type": "factuality", "value": "Paris"},
        {"rubrics": ["The response is factually consistent with this reference: Paris"]},
        RUBRICS,
    ),
    ({"type": "contains", "value": "{{q}}"}, {"accept": ["hi"]}, CONTAINS_CS),  # filled from vars
    ({"type": "contains", "value": "x", "metric": "m"}, {"accept": ["x"]}, CONTAINS_CS),
]


@pytest.mark.parametrize(("assertion", "expected", "scorer"), MAPPED)
def test_each_promptfoo_assertion_type_maps_to_the_fields_and_scorer_it_needs(
    tmp_path: Path,
    assertion: dict[str, Any],
    expected: dict[str, Any],
    scorer: tuple[str, str, dict[str, Any]],
) -> None:
    result = promptfoo(tmp_path, [assertion])
    assert fields(result) == expected
    assert scorers(result) == [scorer]
    assert not any("not imported" in n for n in result.notes)


UNMAPPED = [
    ({"type": "javascript", "value": "output.length<3"}, "javascript"),
    ({"type": "python", "value": "True"}, "python"),
    ({"type": "similar", "value": "x", "threshold": 0.8}, "similar"),
    ({"type": "latency", "threshold": 100}, "latency"),
    ({"type": "cost", "threshold": 0.1}, "cost"),
    ({"type": "ends-with", "value": "!"}, "ends-with"),
    ({"type": "not-contains-all", "value": ["a", "b"]}, "not-contains-all"),
    ({"type": "not-llm-rubric", "value": "rude"}, "not-llm-rubric"),
    ({"type": "not-is-json"}, "not-is-json"),
    ({"type": "is-json"}, "is-json"),  # no schema: nothing to check against
    ({"type": "equals", "value": {"a": 1}}, "equals"),  # only a string can be compared exactly
    (
        {"type": "contains", "value": "x", "transform": "output.trim()"},
        "contains (with transform)",
    ),
    ({"type": "contains", "value": "x", "weight": 0}, "contains (weight 0, informational)"),
]


@pytest.mark.parametrize(("assertion", "label"), UNMAPPED)
def test_assertions_with_no_equivalent_are_kept_on_the_case_and_counted(
    tmp_path: Path, assertion: dict[str, Any], label: str
) -> None:
    result = promptfoo(tmp_path, [assertion])
    assert fields(result) == {"unmapped": [assertion]}
    assert result.scorers == []
    assert result.notes == [
        f"1 assertion not imported: {label} x 1 (kept on each case under 'unmapped')"
    ]


def test_the_unmapped_summary_counts_by_type_largest_first(tmp_path: Path) -> None:
    tests = [
        {
            "vars": {"q": str(i)},
            "assert": [{"type": "javascript", "value": "true"}, {"type": "contains", "value": "a"}]
            + ([{"type": "similar", "value": "a"}] if i < 4 else [{"type": "javascript"}]),
        }
        for i in range(6)
    ]
    path = tmp_path / "many.json"
    path.write_text(json.dumps({"tests": tests}), encoding="utf-8")
    result = load(path, "promptfoo")
    summaries = [n for n in result.notes if "not imported" in n]
    assert summaries == [
        "12 assertions not imported: javascript x 8, similar x 4"
        " (kept on each case under 'unmapped')"
    ]
    for case in result.cases:
        raw = dump(case)["unmapped"]
        assert {a["type"] for a in raw} <= {"javascript", "similar"}
        assert len(raw) == 2


def test_mixed_assertions_keep_what_maps_and_the_rest(tmp_path: Path) -> None:
    result = promptfoo(
        tmp_path,
        [
            {"type": "equals", "value": "4"},
            {"type": "equals", "value": "four"},  # one reference per case
            {"type": "javascript", "value": "true"},
            {"type": "not-contains", "value": "sorry"},
        ],
    )
    out = fields(result)
    assert out["reference"] == "4"
    assert out["unmapped"] == [
        {"type": "equals", "value": "four"},
        {"type": "javascript", "value": "true"},
    ]
    assert out["expect"] == [{"type": "output_omits", "text": "sorry", "case_sensitive": True}]
    assert [s.name for s in result.scorers] == ["exact", "omits"]


def test_several_contains_assertions_must_all_hold(tmp_path: Path) -> None:
    result = promptfoo(
        tmp_path,
        [{"type": "contains", "value": "a"}, {"type": "contains", "value": "b"}],
    )
    assert fields(result) == {"require": ["a", "b"]}
    assert scorers(result) == [REQUIRE_CS]
    grouped = promptfoo(
        tmp_path,
        [
            {
                "type": "assert-set",
                "assert": [{"type": "contains", "value": "a"}, {"type": "contains", "value": "b"}],
            }
        ],
    )
    assert fields(grouped) == {"require": ["a", "b"]}


def test_case_sensitive_and_insensitive_lists_get_separate_scorers(tmp_path: Path) -> None:
    config = DATASETS / "promptfoo" / "promptfooconfig.yaml"
    result = load(config)
    names = [s.name for s in result.scorers]
    assert "contains" in names
    assert "contains_cs" in names
    cs = next(s for s in result.scorers if s.name == "contains_cs")
    assert cs.params["case_sensitive"] is True
    assert cs.params["any_of_field"] == "accept_cs"
    refund = by_id(result)["refund-window"]
    assert dump(refund)["accept_cs"] == ["30 days", "thirty days"]
    assert "accept" not in dump(refund)


def test_the_promptfoo_config_default_test_and_variables_carry_over() -> None:
    result = load(DATASETS / "promptfoo" / "promptfooconfig.yaml")
    assert result.prompt_template == "Answer the customer: {question}"
    cases = by_id(result)
    free = cases["free-plan-notebooks"]
    assert free.input == {
        "product": "Notebird",
        "question": "How many notebooks are on the Free plan?",
    }
    for case in cases.values():  # defaultTest's assertion applies to every test
        assert any(c["text"] == "as an ai language model" for c in dump(case)["expect"])
    assert any("javascript x 1, similar x 1" in n for n in result.notes)
    assert dump(cases["refund-window"])["unmapped"] == [
        {"type": "javascript", "value": "output.length < 300"}
    ]


def test_csv_expected_columns_cover_the_plain_and_the_typed_forms() -> None:
    result = load(DATASETS / "promptfoo" / "tests.csv")
    cases = by_id(result)
    free = dump(cases["free-plan"])
    assert free["accept"] == ["3"]
    assert free["expect"] == [{"type": "output_omits", "text": "unlimited", "case_sensitive": True}]
    assert free["rubrics"] == ["Says three notebooks"]
    refund = dump(cases["refund-window"])
    assert refund["accept_cs"] == ["30 days"]
    assert refund["pattern"] == ["\\b30\\b"]
    assert refund["unmapped"] == [{"type": "javascript", "value": "output.length < 300"}]
    assert cases["pro-price"].reference == "$8 per month"  # a bare value means equals


def test_csv_not_forms_and_metadata_tags(tmp_path: Path) -> None:
    path = tmp_path / "tests.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(
            [
                ["q", "__expected", "__expected1", "__expected2", "__metadata:tags[]"],
                ["one", "not-icontains:sorry", "not-regex:^no", "starts-with:Hi", "a,b"],
            ]
        )
    result = load(path, "promptfoo")
    out = fields(result)
    assert out["expect"] == [
        {"type": "output_omits", "text": "sorry"},
        {"type": "not", "check": {"type": "output_regex", "pattern": "^no"}},
    ]
    assert out["pattern"] == ["^Hi"]
    assert result.cases[0].tags == ["a", "b", "imported", "from:promptfoo"]


# --- the cases survive the dataset file -------------------------------------------------------


@pytest.mark.parametrize(("name", "fmt", "count"), DETECTED)
def test_imported_cases_round_trip_through_the_dataset_file(
    tmp_path: Path, name: str, fmt: str, count: int
) -> None:
    result = load(DATASETS / name)
    path = tmp_path / "cases.jsonl"
    path.write_text(dataset_text(result.cases), encoding="utf-8", newline="\n")
    assert load_dataset(path) == result.cases


# --- backward compatibility -------------------------------------------------------------------


def test_import_dataset_still_returns_plain_cases_for_adk() -> None:
    path = ADK / "basic_criteria.home_automation.evalset.json"
    cases = import_dataset("adk", path, warn=lambda message: None)
    assert cases
    assert all(isinstance(c, Case) for c in cases)
    imported = import_cases("adk", path, warn=lambda message: None)
    assert imported.cases == cases
    assert imported.format == "adk"
    assert imported.detected is False


def test_import_dataset_takes_the_new_formats_too() -> None:
    cases = import_dataset("ragas", DATASETS / "ragas" / "samples.jsonl", warn=lambda m: None)
    assert len(cases) == 3
