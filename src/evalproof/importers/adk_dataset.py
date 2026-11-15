"""Google ADK eval sets as a dataset: ``*.evalset.json`` and the flat legacy ``*.test.json`` list.

Unlike the importers of results, this writes cases (``evalproof import-dataset adk``), not a run.
Stable: the two shapes below are read from real files in google/adk-python v2.14.1 (the samples in
``contributing/samples/evaluation/`` and ``tests/integration/fixture/home_automation_agent/``).
Field references are ``src/google/adk/evaluation/`` in that release.

Two shapes are read, and nothing else:

* an ``EvalSet`` (``eval_set.py:24-41``): ``eval_set_id`` and ``eval_cases``, snake_case at the top
  level (``EvalSet`` has no alias generator). Each ``EvalCase`` (``eval_case.py:190-236``) has
  ``eval_id``, a ``conversation`` of ``Invocation`` objects (``eval_case.py:97-154``) or a
  ``conversation_scenario``, ``session_input.state`` and ``rubrics``. Inside a case both
  snake_case and camelCase keys are read (``EvalBaseModel`` accepts both, ``common.py``);
* the legacy flat list (``local_eval_sets_manager.py:89-173``): ``[{"query", "expected_tool_use":
  [{"tool_name", "tool_input"}], "reference"}]``. ADK reads the whole list as one eval case whose
  items are its invocations, so it becomes one case: the first query is the ``input``, the later
  ones the ``turns``, the last ``reference`` the ``reference``, and each item's
  ``expected_tool_use`` a ``tool_order`` check with that item's turn. The other legacy shape,
  ``{name, data, initial_session}``, is refused: no real file of it was found.

What a case becomes (``Case`` fields, see ``dataset.py``):

* ``eval_id`` is the ``id`` (the legacy list has none: the file name without ``.test.json``);
* the text of the first invocation's ``user_content`` is the ``input``, and the later invocations'
  are ``turns``. A ``thought`` part is not text;
* the text of the last invocation's ``final_response`` is the ``reference``, when it has one;
* ``session_input.state`` is ``state``, when it is not empty (``app_name`` and ``user_id`` are the
  agent's, not the case's, and are not read);
* every invocation that expects tool calls (``intermediate_data.tool_uses``, or the
  ``function_call`` parts of ``invocation_events``) becomes one ``tool_order`` check with
  ``match: exact``, that invocation's ``turn`` and its calls as ``{name, args}``, in ``expect``;
  ``tool_responses`` and the ``intermediate_responses`` text are not read;
* ``EvalCase.rubrics`` and every ``Invocation.rubrics`` are the case's ``rubrics``, as
  ``{id, text, type}`` (``rubric_id``, ``rubric_content.text_property``, ``type``; no ``type``
  when it is null): the ``llm_judge`` rubric-set mode reads them with ``case_rubrics``. A rubric
  that appears twice with one id and one text is kept once; two texts for one id are refused.
  Rubrics in an ``eval_config.json`` are in another file and are not read;
* ``final_session_state`` is not read.

An ``EvalCase`` with a ``conversation_scenario`` (user simulation) has no turns to replay, so it
is skipped, all of them under one warning that names them; a file with no case left is refused.
An ``EvalSetResult`` (a result, with ``eval_case_results``) is refused with a pointer to
``evalproof import adk``.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from evalproof.config import ScorerConfig
from evalproof.dataset import validate_case_fields
from evalproof.importers._common import ImportFormatError, read_json
from evalproof.importers._dataset import DatasetOptions, ImportedDataset, reject_fields
from evalproof.importers.adk import _content_text, _field, _list
from evalproof.models import Case

FORMAT = "adk"


def read(path: Path, warn: Callable[[str], None]) -> list[Case]:
    """The cases of the ADK file at ``path``; ``warn`` hears about the cases it skipped."""
    data = read_json(path)
    if isinstance(data, list):
        return _legacy_list(path, data)
    if isinstance(data, dict):
        if isinstance(data.get("eval_cases"), list):
            return _eval_set(path, data["eval_cases"], warn)
        if isinstance(_field(data, "eval_case_results"), list):
            raise ImportFormatError(
                f"{path}: this is an ADK eval result, not an eval set; import it as a run with "
                "`evalproof import adk`. `import-dataset` reads the *.evalset.json (eval_cases) "
                "or the flat *.test.json list"
            )
        if isinstance(data.get("evalCases"), list):
            raise ImportFormatError(
                f"{path}: found evalCases; an ADK eval set has snake_case keys at the top level "
                "(eval_set_id, eval_cases)"
            )
    raise ImportFormatError(
        f"{path}: not an ADK eval set or test file; expected an *.evalset.json (an object with "
        "eval_set_id and eval_cases) or a *.test.json list of {query, expected_tool_use, "
        "reference}"
    )


def _eval_set(path: Path, raw_cases: list[Any], warn: Callable[[str], None]) -> list[Case]:
    cases: list[Case] = []
    skipped: list[str] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases):
        where = f"{path}: eval_cases[{position}]"
        if not isinstance(raw, dict):
            raise ImportFormatError(f"{where} is not an object")
        eval_id = _field(raw, "eval_id")
        if not isinstance(eval_id, str) or not eval_id.strip():
            raise ImportFormatError(f"{where} has no eval_id")
        if eval_id in seen:
            raise ImportFormatError(f"{path}: two eval cases have the eval_id {eval_id!r}")
        seen.add(eval_id)
        conversation = _field(raw, "conversation")
        scenario = _field(raw, "conversation_scenario")
        if conversation is None and scenario is not None:
            skipped.append(eval_id)
            continue
        if conversation is not None and scenario is not None:
            raise ImportFormatError(
                f"{where} ({eval_id!r}) has both conversation and conversation_scenario; ADK "
                "allows exactly one"
            )
        if not isinstance(conversation, list) or not conversation:
            raise ImportFormatError(
                f"{where} ({eval_id!r}) has no conversation (a non-empty list of invocations) "
                "and no conversation_scenario"
            )
        cases.append(_eval_case(path, eval_id, raw, conversation))
    if skipped:
        warn(
            f"{path}: skipped {len(skipped)} case(s) that use conversation_scenario (user "
            f"simulation, which has no fixed turns to replay): {', '.join(skipped)}"
        )
    if not cases:
        raise ImportFormatError(
            f"{path}: no case to import: every eval case uses conversation_scenario"
            if skipped
            else f"{path}: the eval set has no eval_cases"
        )
    return cases


def _eval_case(path: Path, eval_id: str, raw: dict[str, Any], conversation: list[Any]) -> Case:
    texts: list[str] = []
    expect: list[dict[str, Any]] = []
    rubrics: list[dict[str, str]] = []
    rubrics.extend(_rubrics(path, eval_id, _field(raw, "rubrics")))
    for turn, invocation in enumerate(conversation):
        if not isinstance(invocation, dict):
            raise ImportFormatError(f"{path}: case {eval_id!r}, invocation {turn} is not an object")
        text = _content_text(_field(invocation, "user_content"))
        if not text:
            raise ImportFormatError(
                f"{path}: case {eval_id!r}, invocation {turn} has no user_content text"
            )
        texts.append(text)
        calls = [_call(path, eval_id, turn, use) for use in _tool_uses(invocation)]
        if calls:
            expect.append({"type": "tool_order", "match": "exact", "turn": turn, "calls": calls})
        rubrics.extend(_rubrics(path, eval_id, _field(invocation, "rubrics")))
    final = _content_text(_field(conversation[-1], "final_response"))
    state = _field(_field(raw, "session_input"), "state")
    if state is not None and not isinstance(state, dict):
        raise ImportFormatError(f"{path}: case {eval_id!r}: session_input.state is not an object")
    return _case(
        path,
        eval_id,
        texts,
        reference=final,
        state=state,
        expect=expect,
        rubrics=_merged(path, eval_id, rubrics),
    )


def _legacy_id(path: Path) -> str:
    """The case id of a flat list: the file name without ``.test.json`` (or its last extension)."""
    name = path.name
    return name[: -len(".test.json")] if name.endswith(".test.json") else path.stem


def _legacy_list(path: Path, items: list[Any]) -> list[Case]:
    """The flat list is one conversation, so one case: ADK reads the whole file as the ``data``
    of one eval case, each item one invocation (``agent_evaluator.py``
    ``_get_eval_set_from_old_format``, ``local_eval_sets_manager.py``
    ``_convert_invocation_to_pydantic_schema``)."""
    if not items:
        raise ImportFormatError(f"{path}: the list has no invocations")
    case_id = _legacy_id(path)
    texts: list[str] = []
    expect: list[dict[str, Any]] = []
    reference: Any = ""
    for turn, item in enumerate(items):
        where = f"{path}: item {turn}"
        if not isinstance(item, dict):
            raise ImportFormatError(f"{where} is not an object")
        if "query" not in item:
            hint = (
                " (the {name, data, initial_session} legacy shape is not supported)"
                if "data" in item
                else ""
            )
            raise ImportFormatError(f"{where} has no query{hint}")
        query = item["query"]
        if not isinstance(query, str) or not query.strip():
            raise ImportFormatError(f"{where}: query must be a non-empty string")
        texts.append(query)
        uses = item.get("expected_tool_use")
        if uses is not None and not isinstance(uses, list):
            raise ImportFormatError(f"{where}: expected_tool_use must be a list")
        calls = [_legacy_call(path, case_id, use) for use in uses or []]
        if calls:
            expect.append({"type": "tool_order", "match": "exact", "turn": turn, "calls": calls})
        reference = item.get("reference")
    return [
        _case(
            path,
            case_id,
            texts,
            reference=reference if isinstance(reference, str) else "",
            state=None,
            expect=expect,
            rubrics=[],
        )
    ]


def _case(
    path: Path,
    case_id: str,
    texts: list[str],
    *,
    reference: str,
    state: dict[str, Any] | None,
    expect: list[dict[str, Any]],
    rubrics: list[dict[str, str]],
) -> Case:
    """One ``Case`` in the order the writer keeps: input, reference, then the agent fields."""
    record: dict[str, Any] = {"id": case_id, "input": texts[0]}
    if reference:
        record["reference"] = reference
    if len(texts) > 1:
        record["turns"] = texts[1:]
    if state:
        record["state"] = state
    if expect:
        record["expect"] = expect
    if rubrics:
        record["rubrics"] = rubrics
    case = Case.model_validate(record)
    try:
        validate_case_fields(case)
    except ValueError as exc:
        raise ImportFormatError(f"{path}: {exc}") from exc
    return case


def _tool_uses(invocation: dict[str, Any]) -> list[Any]:
    """The calls an invocation expects: its ``tool_uses``, or, in the event form, the
    ``function_call`` parts of its ``invocation_events``."""
    data = _field(invocation, "intermediate_data")
    uses = _list(_field(data, "tool_uses"))
    if uses:
        return uses
    return [
        call
        for event in _list(_field(data, "invocation_events"))
        for part in _list(_field(_field(event, "content"), "parts"))
        if (call := _field(part, "function_call")) is not None
    ]


def _call(path: Path, case_id: str, turn: int, use: Any) -> dict[str, Any]:
    name = _field(use, "name")
    if not isinstance(name, str) or not name:
        raise ImportFormatError(
            f"{path}: case {case_id!r}, invocation {turn}: a tool use has no name"
        )
    return _spec(path, case_id, name, _field(use, "args"))


def _legacy_call(path: Path, case_id: str, use: Any) -> dict[str, Any]:
    name = use.get("tool_name") if isinstance(use, dict) else None
    if not isinstance(name, str) or not name:
        raise ImportFormatError(f"{path}: case {case_id!r}: an expected_tool_use has no tool_name")
    return _spec(path, case_id, name, use.get("tool_input"))


def _spec(path: Path, case_id: str, name: str, args: Any) -> dict[str, Any]:
    """A call as ``tool_order`` writes it; no ``args`` key when it expects none."""
    if args is not None and not isinstance(args, dict):
        raise ImportFormatError(
            f"{path}: case {case_id!r}: the arguments of {name!r} are not an object"
        )
    return {"name": name, "args": args} if args else {"name": name}


def _rubrics(path: Path, case_id: str, raw: Any) -> list[dict[str, str]]:
    rubrics = []
    for position, rubric in enumerate(_list(raw)):
        rid = _field(rubric, "rubric_id")
        text = _field(_field(rubric, "rubric_content"), "text_property")
        where = f"{path}: case {case_id!r}, rubrics[{position}]"
        if not isinstance(rid, str) or not rid.strip():
            raise ImportFormatError(f"{where} has no rubric_id")
        if not isinstance(text, str) or not text.strip():
            raise ImportFormatError(f"{where} ({rid!r}) has no rubric_content.text_property")
        item = {"id": rid.strip(), "text": text}
        kind = _field(rubric, "type")
        if isinstance(kind, str) and kind:
            item["type"] = kind
        rubrics.append(item)
    return rubrics


def _merged(path: Path, case_id: str, rubrics: list[dict[str, str]]) -> list[dict[str, str]]:
    """The case's and its invocations' rubrics as one list with one entry per id."""
    by_id: dict[str, dict[str, str]] = {}
    for rubric in rubrics:
        first = by_id.setdefault(rubric["id"], rubric)
        if first["text"] != rubric["text"]:
            raise ImportFormatError(
                f"{path}: case {case_id!r}: the rubric {rubric['id']!r} has two different texts "
                "(on the case and an invocation, or on two invocations); give them distinct ids"
            )
    return list(by_id.values())


def read_dataset(path: Path, opts: DatasetOptions) -> ImportedDataset:
    """``read`` with the scorers its cases need: a ``checks`` scorer for the tool-order checks,
    the rubric-set judge for the rubrics, and the judge against the reference."""
    reject_fields(opts, FORMAT)
    cases = read(path, opts.warn)
    extras = [c.model_extra or {} for c in cases]
    scorers: list[ScorerConfig] = []
    if any(e.get("expect") for e in extras):
        scorers.append(
            ScorerConfig(
                name="checks", type="checks", params={"field": "expect", "on_missing": "skip"}
            )
        )
    if any(c.reference for c in cases):
        scorers.append(ScorerConfig(name="judge", type="llm_judge"))
    if any(e.get("rubrics") for e in extras):
        scorers.append(
            ScorerConfig(name="rubrics", type="llm_judge", params={"case_rubrics": True})
        )
    return ImportedDataset(cases=cases, scorers=scorers, format=FORMAT)
