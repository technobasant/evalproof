"""``evalproof draft``: inputs for what a scan found untested, never expected answers.

Two sources, kept apart in the output:

* pytest files: a call with a literal input and an assertion on its result is a case whose
  expected answer a person already wrote in a test. No model is called.
* a model, for the gaps that are about missing inputs (a tool no case exercises, no multi-turn
  case, code no suite runs, too few cases). It writes inputs only, tagged ``draft``, with no
  ``reference``, ``accept``, ``pattern``, ``schema`` or ``expect``: a person adds those.

The result is a separate JSONL file. It is never a suite's dataset.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import sys
import warnings
from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import ConfigError, GenerationParams, SuiteConfig, TargetConfig, load_config
from evalproof.dataset import load_dataset, write_dataset
from evalproof.draft.ask import (
    Request,
    ask,
    code_request,
    few_cases_request,
    key_of,
    multi_turn_request,
    tool_request,
)
from evalproof.draft.pytests import harvest
from evalproof.draft.scanfile import ScanFile, load_scan
from evalproof.models import Case
from evalproof.providers import make_provider
from evalproof.providers.base import close_resource
from evalproof.scan.model import MODEL_GAP_KINDS

PYTEST_GAP = "test_cases_not_imported"
DEFAULT_CACHE_DIR = ".evalproof/cache"
GAP_CHOICES = (*MODEL_GAP_KINDS, PYTEST_GAP)


@dataclass
class DraftOptions:
    scan: Path
    out: Path
    target: str | None = None
    suite: Path | None = None
    per_gap: int = 3
    max_cases: int = 30
    from_tests_only: bool = False
    no_cache: bool = False
    force: bool = False
    gaps: Sequence[str] = ()
    tools: Sequence[str] = ()
    files: Sequence[str] = ()


@dataclass
class DraftResult:
    cases: list[Case] = field(default_factory=list)
    from_pytest: int = 0
    by_gap: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    drafted_by: str | None = None
    wrote: Path | None = None


def draft_id(messages: Sequence[str]) -> str:
    """``draft-`` and ten hex digits of the hash of the input and its later turns."""
    digest = hashlib.sha256(json.dumps(list(messages), ensure_ascii=False).encode()).hexdigest()
    return f"draft-{digest[:10]}"


def _rel(root: Path, text: str) -> str:
    """``text`` (a path the user gave, from the current directory or from the scanned root) as
    the root-relative POSIX path the scan uses."""
    path = Path(text)
    for candidate in (path, root / path):
        try:
            if candidate.exists():
                return candidate.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return posixpath.normpath(text.replace("\\", "/")).removeprefix("./")


def _load_suite(path: Path) -> SuiteConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return load_config(path)


def _suite_text(case: Case) -> str:
    return case.input if isinstance(case.input, str) else json.dumps(case.input, sort_keys=True)


def _select_gaps(scan: ScanFile, options: DraftOptions) -> list[dict[str, Any]]:
    wanted = set(options.gaps)
    if options.from_tests_only:
        wanted = {PYTEST_GAP}
    elif not wanted:
        # A narrowing flag names the kind it narrows, so `--tool x` is not also a few_cases run.
        wanted = (
            ({"tool_unchecked"} if options.tools else set())
            | ({"uncovered_code"} if options.files else set())
            | ({"few_cases"} if options.suite and not (options.tools or options.files) else set())
        ) or set(GAP_CHOICES)
    tools = set(options.tools)
    files = {_rel(scan.root, f) for f in options.files}
    suite = _rel(scan.root, str(options.suite)) if options.suite else None
    keep: list[dict[str, Any]] = []
    for gap in scan.gaps:
        kind = gap.get("kind")
        if kind not in wanted:
            continue
        if kind == "tool_unchecked" and tools and gap.get("tool") not in tools:
            continue
        if kind in ("uncovered_code", PYTEST_GAP) and files and gap.get("path") not in files:
            continue
        if kind == "few_cases" and suite and gap.get("suite") != suite:
            continue
        keep.append(gap)
    return keep


def _suites(scan: ScanFile, options: DraftOptions, result: DraftResult) -> list[SuiteConfig]:
    """The suites whose datasets and judge the draft reads: the one named, else every suite in
    the scan."""
    if options.suite:
        named = options.suite if options.suite.exists() else scan.root / options.suite
        paths = [named]
    else:
        paths = [scan.root / str(s["path"]) for s in scan.suites if s.get("path")]
    loaded: list[SuiteConfig] = []
    for path in paths:
        try:
            loaded.append(_load_suite(path))
        except Exception as exc:
            if options.suite:
                raise ConfigError(f"{path}: {exc}") from exc
            result.warnings.append(f"{path}: could not be loaded ({exc}); its inputs are not used")
    return loaded


def _existing_inputs(suites: Sequence[SuiteConfig], result: DraftResult) -> list[str]:
    inputs: list[str] = []
    objects = False
    for cfg in suites:
        try:
            cases = load_dataset(cfg.resolve(cfg.dataset))
        except (OSError, ValueError) as exc:
            result.warnings.append(f"{cfg.name}: its dataset could not be read ({exc})")
            continue
        for case in cases:
            objects = objects or not isinstance(case.input, str)
            inputs.append(_suite_text(case))
    if objects:
        result.warnings.append(
            "some suite inputs are objects, and drafted inputs are plain text: check they fit "
            "the system's input before using them"
        )
    return inputs


def _pick_target(
    options: DraftOptions, suites: Sequence[SuiteConfig]
) -> tuple[TargetConfig, SuiteConfig | None]:
    """The model to ask: ``--target``, else the one judge the suites share."""
    if options.target:
        return TargetConfig(target=options.target), None
    judges = {c.judge.target: c for c in suites if c.judge is not None}
    if len(judges) == 1:
        cfg = next(iter(judges.values()))
        assert cfg.judge is not None
        return cfg.judge, cfg
    if judges:
        raise ConfigError(
            "the suites use different judges ("
            + ", ".join(sorted(judges))
            + "): pass --suite to pick one, or --target MODEL"
        )
    raise ConfigError(
        "drafting inputs needs a model and none is given: the suite has no judge, so pass "
        "--target MODEL (for example --target ollama:qwen3:8b), or use --from-tests-only to "
        "draft from pytest files without a model"
    )


def _cache_dir(
    options: DraftOptions, source: SuiteConfig | None, suites: Sequence[SuiteConfig]
) -> Path:
    cfg = source or (suites[0] if suites else None)
    return cfg.resolve(cfg.cache_dir) if cfg else Path(DEFAULT_CACHE_DIR)


def _requests(
    scan: ScanFile, gaps: list[dict[str, Any]], options: DraftOptions
) -> list[tuple[Request, dict[str, Any]]]:
    """One request for each model gap, with the ``draft_for`` the cases it yields will carry."""
    out: list[tuple[Request, dict[str, Any]]] = []
    for gap in gaps:
        kind = gap["kind"]
        count = options.per_gap
        if kind == "tool_unchecked":
            tool = next(
                (f for f in scan.findings_of("tool") if f.get("name") == gap.get("tool")), None
            )
            if tool is None:
                continue
            out.append(
                (tool_request(tool, count, scan.findings), {"gap": kind, "tool": gap["tool"]})
            )
        elif kind == "no_multi_turn_cases":
            evidence = scan.findings_of("multi_turn")
            out.append(
                (
                    multi_turn_request(evidence, count, scan.findings),
                    {"gap": kind, "path": gap.get("path"), "line": gap.get("line")},
                )
            )
        elif kind == "uncovered_code":
            path = str(gap.get("path"))
            out.append(
                (
                    code_request(scan.root, path, int(gap.get("line") or 1), count, scan.findings),
                    {"gap": kind, "path": path, "line": gap.get("line")},
                )
            )
        elif kind == "few_cases":
            suite = next((s for s in scan.suites if s.get("path") == gap.get("suite")), None)
            if suite is None:
                continue
            out.append(
                (
                    few_cases_request(suite, count, scan.findings),
                    {"gap": kind, "suite": suite["path"]},
                )
            )
    return out


def run_draft(options: DraftOptions) -> DraftResult:
    """Draft from a scan. Raises ``ConfigError`` for a usage problem (no model for a gap that
    needs one, an ``out`` that exists); a model that returns nothing usable skips its gap with a
    warning in the result."""
    if options.per_gap < 1:
        raise ConfigError("--per-gap must be at least 1")
    if options.max_cases < 1:
        raise ConfigError("--max-cases must be at least 1")
    if options.out.exists() and not options.force:
        raise ConfigError(f"{options.out} exists; pass --force to overwrite it")
    if options.out.is_dir():
        raise ConfigError(f"cannot write {options.out}: it is a directory")
    scan = load_scan(options.scan)
    result = DraftResult()
    gaps = _select_gaps(scan, options)
    suites = _suites(scan, options, result)
    existing = _existing_inputs(suites, result)
    seen = {key_of([e]) for e in existing}
    drafted: list[Case] = []

    # -- from pytest files: no model, and the expected answer is the test's own
    for gap in gaps:
        if gap["kind"] != PYTEST_GAP:
            continue
        finding = scan.finding_at("test_cases", str(gap.get("path")))
        if not finding or finding.get("format") != "pytest":
            continue
        path = str(finding["path"])
        found = harvest(scan.root, path, dict(finding.get("imports") or {}))
        result.warnings.extend(found.warnings)
        for item in found.cases:
            if key_of([item.input]) in seen:
                result.notes.append(f"{item.source}: input already in a suite; skipped")
                continue
            seen.add(key_of([item.input]))
            drafted.append(
                Case(
                    id=draft_id([item.input]),
                    input=item.input,
                    tags=["draft", f"gap:{PYTEST_GAP}", "from:pytest"],
                    draft_for={"gap": PYTEST_GAP, "path": path},
                    source=item.source,
                    **item.fields(),
                )
            )
    if len(drafted) > options.max_cases:
        result.notes.append(
            f"kept {options.max_cases} of {len(drafted)} pytest cases (--max-cases)"
        )
        drafted = drafted[: options.max_cases]
    result.from_pytest = len(drafted)

    # -- from a model: inputs only
    requests = [] if options.from_tests_only else _requests(scan, gaps, options)
    if requests:
        target, source = _pick_target(options, suites)
        params = (
            target.params.model_copy()
            if isinstance(target.params, GenerationParams)
            else GenerationParams()
        )
        provider = make_provider(target)
        with ExitStack() as owned:
            owned.callback(close_resource, provider)
            if not options.no_cache:
                provider = CachedProvider(
                    provider,
                    ResponseCache(_cache_dir(options, source, suites)),
                    target.base_url,
                    role="draft",
                )
            result.drafted_by = target.target
            for request, draft_for in requests:
                room = options.max_cases - len(drafted)
                if room <= 0:
                    result.notes.append(
                        f"stopped at --max-cases {options.max_cases}: {request.label} and later "
                        "gaps were not drafted"
                    )
                    break
                request.count = min(request.count, room)
                inputs, problem = ask(provider, params, request, existing)
                if inputs is None:
                    result.warnings.append(f"{request.label}: skipped: {problem}")
                    continue
                added = 0
                for messages in inputs:
                    if added >= request.count:
                        break
                    key = key_of(messages)
                    if key in seen:
                        continue
                    seen.add(key)
                    extra: dict[str, Any] = {"turns": messages[1:]} if len(messages) > 1 else {}
                    drafted.append(
                        Case(
                            id=draft_id(messages),
                            input=messages[0],
                            tags=["draft", f"gap:{request.gap}", "from:model"],
                            draft_for=draft_for,
                            drafted_by=target.target,
                            **extra,
                        )
                    )
                    added += 1
                result.by_gap[request.label] = added
                if added == 0:
                    result.warnings.append(
                        f"{request.label}: every input the model wrote was a duplicate or empty"
                    )

    result.cases = drafted
    if drafted:
        _write(options.out, drafted)
        result.wrote = options.out
    return result


def _write(path: Path, cases: list[Case]) -> None:
    write_dataset(cases, path)


def render_result(result: DraftResult, out: Path) -> str:
    """What the command prints: what was written, from where, and what a person must do next."""
    lines: list[str] = []
    if not result.cases:
        lines.append("evalproof draft: nothing to draft, so no file was written")
        for note in result.notes:
            lines.append(f"  note: {note}")
        return "\n".join(lines) + "\n"
    lines.append(f"evalproof draft: wrote {len(result.cases)} draft case(s) to {out}")
    if result.from_pytest:
        lines.append(
            f"  from pytest files: {result.from_pytest} (the test's own assertion is kept)"
        )
    model = len(result.cases) - result.from_pytest
    if model:
        detail = ", ".join(f"{k}: {n}" for k, n in result.by_gap.items() if n)
        lines.append(f"  from a model ({result.drafted_by}): {model} ({detail})")
    for note in result.notes:
        lines.append(f"  note: {note}")
    lines += [
        "",
        "These are inputs only. A model never wrote an expected answer. Before a draft goes",
        "into a dataset, a person must add the expected answer or checks to it (reference,",
        "accept, pattern, schema or expect) and check that the input is one you want tested.",
        "Drafts from pytest carry the assertion their test made; read it before you trust it.",
    ]
    return "\n".join(lines) + "\n"


def emit_warnings(result: DraftResult) -> None:
    for message in result.warnings:
        sys.stderr.write(f"evalproof: warning: {message}\n")
