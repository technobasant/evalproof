"""Tell which tool wrote a test-case file (``evalproof import-dataset auto``).

Detection reads the file name and the keys of the first records, never a guess from the values.
A file that fits no format, or more than one, is an error that lists the formats and asks for an
explicit one: importing it as the wrong format would produce plausible but wrong cases.

* promptfoo: a YAML file, a name that holds ``promptfooconfig``, a ``vars`` key, or a
  ``__expected`` column;
* ragas: ``user_input``, or ``question`` with ``ground_truth``;
* langsmith: ``inputs`` (with ``outputs``);
* openai-evals: ``input`` with ``ideal``, or an ``item`` wrapper;
* inspect: ``input`` with ``target``;
* deepeval: ``input`` with ``expected_output`` (or ``expectedOutput``), or a multi-turn golden
  (``scenario``, ``expected_outcome``). Langfuse items have the same keys and map the same way, so
  they are read as deepeval; ask for ``langfuse`` to use the item ``id`` as the case id;
* braintrust: ``input`` with ``expected``;
* phoenix: ``input`` with an ``output`` object;
* adk: an ``eval_cases`` eval set, or the flat ``query`` / ``expected_tool_use`` list.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from evalproof.importers._common import ImportFormatError, parse_json, read_text
from evalproof.importers._dataset import read_records

_PROMPTFOO_CONFIG_KEYS = frozenset({"tests", "providers", "prompts", "defaultTest"})


def detect_format(path: Path, formats: Sequence[str]) -> str:
    """The one format ``path`` is in. Raises ``ImportFormatError`` when it is none or several;
    ``formats`` is the list the error offers."""
    found = candidates(path)
    if len(found) == 1:
        return found[0]
    offered = ", ".join(f for f in formats if f != "auto")
    if not found:
        raise ImportFormatError(
            f"{path}: cannot tell which tool wrote this file. Name the format: one of {offered} "
            "(a plain table of questions and answers is jsonl or csv, with --input-field)"
        )
    raise ImportFormatError(
        f"{path}: this file fits more than one format ({', '.join(found)}). Name the format: "
        f"one of {offered}"
    )


def candidates(path: Path) -> list[str]:
    """Every format whose shape ``path`` has, in a fixed order."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml") or "promptfooconfig" in path.name.lower():
        return ["promptfoo"]
    if suffix == ".json":
        top = _top_level(path)
        if isinstance(top, dict):
            if "eval_cases" in top:
                return ["adk"]
            if top.keys() & _PROMPTFOO_CONFIG_KEYS and not _has_records(top):
                return ["promptfoo"]
    keys: set[str] = set()
    for record in read_records(path)[:50]:
        keys.update(str(k) for k in record)
    found: list[str] = []
    if "vars" in keys or any(k.startswith("__expected") for k in keys):
        found.append("promptfoo")
    if "user_input" in keys or ("question" in keys and keys & {"ground_truth", "ground_truths"}):
        found.append("ragas")
    if "inputs" in keys:
        found.append("langsmith")
    if "item" in keys or {"input", "ideal"} <= keys:
        found.append("openai-evals")
    if {"input", "target"} <= keys:
        found.append("inspect")
    if ("input" in keys and keys & {"expected_output", "expectedOutput"}) or (
        "scenario" in keys and "expected_outcome" in keys
    ):
        found.append("deepeval")
    if {"input", "expected"} <= keys:
        found.append("braintrust")
    if "input" in keys and _output_objects(path):
        found.append("phoenix")
    if "query" in keys and keys & {"expected_tool_use", "reference"}:
        found.append("adk")
    return found


def _top_level(path: Path) -> Any:
    try:
        return parse_json(read_text(path), path)
    except ImportFormatError:
        return None  # a JSONL file named .json: the records are read below


def _has_records(top: dict[str, Any]) -> bool:
    """A dict that is a list of records under a usual key rather than a promptfoo config."""
    tests = top.get("tests")
    return (
        isinstance(tests, list)
        and bool(tests)
        and not any(isinstance(t, dict) and "vars" in t for t in tests)
        and not (top.keys() & {"providers", "prompts", "defaultTest"})
    )


def _output_objects(path: Path) -> bool:
    """Every ``output`` the first records hold is an object (Phoenix), and one of them is."""
    outputs = [r.get("output") for r in read_records(path)[:50] if r.get("output") is not None]
    return bool(outputs) and all(isinstance(o, dict) for o in outputs)
