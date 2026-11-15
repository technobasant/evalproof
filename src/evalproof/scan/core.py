"""``evalproof scan``: read a repository without running it, say what its LLM code is, and say what
no suite tests yet.

Static only: no model call, and no user code is imported or executed (Python is read with
``ast``, TypeScript and JavaScript with regular expressions). Suites are loaded with
``load_config``, which does not import anything either. What a suite's target runs is known for a
``python:`` target and for a ``command:`` that starts a script with node, tsx, bun or deno.
"""

from __future__ import annotations

import json
import posixpath
import re
import shlex
import subprocess
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from evalproof.config import ConfigError, SuiteConfig, command_argv, load_config
from evalproof.dataset import load_dataset
from evalproof.importers._detect import candidates
from evalproof.models import Case
from evalproof.scan.graph import ModuleIndex, reachable
from evalproof.scan.model import (
    DEFAULT_MIN_CASES,
    FINDING_KINDS,
    GAP_KINDS,
    Finding,
    Gap,
    ScanReport,
    SuiteInfo,
)
from evalproof.scan.pyscan import EXCERPT_CHARS as EXCERPT_LIMIT
from evalproof.scan.pyscan import PyInfo, analyze_python
from evalproof.scan.tsgraph import NodeCommand, TsGraph, node_command
from evalproof.scan.tsscan import TsInfo, is_import_only, read_typescript
from evalproof.scan.walk import (
    DATA_SUFFIXES,
    PROMPT_SUFFIXES,
    PY_SUFFIXES,
    TS_SUFFIXES,
    YAML_SUFFIXES,
    walk,
)


def _arg(value: str) -> str:
    """``value`` quoted for the shell a suggested command is pasted into: the Windows way
    (``list2cmdline``) on Windows, as ``evalproof doctor`` does, and POSIX quoting elsewhere."""
    return subprocess.list2cmdline([value]) if sys.platform == "win32" else shlex.quote(value)


DRAFTS_OUT = "evals/drafts.jsonl"
DEFAULT_MAX_DRAFTS = 30  # what ``evalproof draft --max-cases`` defaults to


def _read(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError):
        return None


def _stem(rel: str) -> str:
    stem = posixpath.basename(rel).split(".", 1)[0]
    return re.sub(r"[^\w-]+", "_", stem).strip("_") or "cases"


# --- suites -------------------------------------------------------------------------------------


def _load_suite(path: Path) -> SuiteConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # load_config may warn about the suite's own settings
        return load_config(path)


def _looks_like_suite(text: str) -> bool:
    if "scorers" not in text or "system" not in text:
        return False
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError:
        return False
    return isinstance(raw, dict) and "scorers" in raw and "system" in raw


def _check_names(check: Any) -> set[str]:
    """The tool names a check, or the checks inside an ``any_of`` or a ``not``, mention."""
    names: set[str] = set()
    if not isinstance(check, dict):
        return names
    name = check.get("name")
    if isinstance(name, str):
        names.add(name)
    calls = check.get("calls")
    if isinstance(calls, list):
        for entry in calls:
            if isinstance(entry, str):
                names.add(entry)
            elif isinstance(entry, dict) and isinstance(entry.get("name"), str):
                names.add(entry["name"])
    children = check.get("checks")
    if isinstance(children, list):
        for child in children:
            names |= _check_names(child)
    names |= _check_names(check.get("check"))
    return names


def _suite_check_names(cfg: SuiteConfig, cases: list[Case]) -> set[str]:
    names: set[str] = set()
    for scorer in cfg.scorers:
        if scorer.type != "checks":
            continue
        suite_wide = scorer.params.get("checks")
        for check in suite_wide if isinstance(suite_wide, list) else []:
            names |= _check_names(check)
        field = str(scorer.params.get("field", "expect"))
        for case in cases:
            own = case.field(field)
            for check in own if isinstance(own, list) else []:
                names |= _check_names(check)
    return names


# --- the scan -----------------------------------------------------------------------------------


def scan_path(
    root: Path,
    *,
    excludes: Sequence[str] = (),
    suite_files: Sequence[Path] = (),
    scan_ref: str = "scan.json",
) -> ScanReport:
    """Scan the directory ``root``. ``suite_files`` replaces discovery of suites; ``scan_ref`` is
    the file name the suggested ``draft`` commands give for the scan's JSON."""
    if not root.exists():
        raise ConfigError(f"{root}: no such directory")
    if not root.is_dir():
        raise ConfigError(f"{root}: scan takes a directory, not a file")
    real_root = root.resolve()

    def shown(path: Path) -> str:
        try:
            return path.resolve().relative_to(real_root).as_posix()
        except ValueError:
            return path.as_posix()

    walked = walk(root, excludes)
    py: dict[str, PyInfo] = {}
    ts: dict[str, TsInfo] = {}
    declarations: list[str] = []
    findings: list[Finding] = []
    yaml_files: list[str] = []
    data_files: list[str] = []
    prompt_files: list[str] = []
    unparsable = 0
    for rel in walked.files:
        suffix = posixpath.splitext(rel)[1].lower()
        if rel.endswith(".d.ts"):
            declarations.append(rel)  # types only: no finding, but an import can name it
        elif suffix in PY_SUFFIXES or suffix in TS_SUFFIXES:
            text = _read(root / rel)
            if text is None:
                unparsable += 1
                continue
            if suffix in PY_SUFFIXES:
                try:
                    info = analyze_python(rel, text)
                except (SyntaxError, ValueError, RecursionError, MemoryError):
                    unparsable += 1
                    continue
                py[rel] = info
                findings.extend(info.findings)
            else:
                ts[rel] = read_typescript(rel, text)
                findings.extend(ts[rel].findings)
        elif suffix in YAML_SUFFIXES:
            yaml_files.append(rel)
        elif suffix in DATA_SUFFIXES:
            data_files.append(rel)
        if suffix in PROMPT_SUFFIXES and "prompts" in rel.split("/")[:-1]:
            prompt_files.append(rel)

    findings = _merge_tools(findings)
    notes: list[str] = []

    # -- suites
    loaded: list[tuple[str, SuiteConfig]] = []
    invalid = 0
    suite_rels: set[str] = set()
    if suite_files:
        for path in suite_files:
            loaded.append((shown(path), _load_suite(path)))
    else:
        for rel in yaml_files:
            if posixpath.basename(rel).lower().startswith("promptfooconfig"):
                continue
            text = _read(root / rel)
            if text is None or not _looks_like_suite(text):
                continue
            try:
                loaded.append((rel, _load_suite(root / rel)))
            except Exception as exc:  # a file that looks like a suite and is not one: say why
                invalid += 1
                first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
                notes.append(f"not a suite evalproof accepts: {rel}: {first}")
    suite_rels = {rel for rel, _ in loaded}

    index = ModuleIndex(
        py, [posixpath.dirname(rel) for rel in suite_rels if not rel.startswith("/")]
    )
    edges = {path: index.edges(info) for path, info in py.items()}
    workspace, alias_keys = _ts_aliases(root, data_files) if ts else (set(), [])
    graph = TsGraph(ts, declarations, workspace, alias_keys)

    suites: list[SuiteInfo] = []
    suite_cases: dict[str, list[Case] | None] = {}
    suite_datasets: set[str] = set()
    checked: set[str] = set()
    for rel, cfg in loaded:
        dataset = cfg.resolve(cfg.dataset)
        suite_datasets.add(shown(dataset))
        try:
            cases: list[Case] | None = load_dataset(dataset)
        except (OSError, ValueError):
            cases = None
            notes.append(f"{rel}: its dataset {shown(dataset)} could not be read")
        suite_cases[rel] = cases
        checked |= _suite_check_names(cfg, cases or [])
        reachability, reaches, reason = _reach(cfg, rel, index, edges, graph)
        suites.append(
            SuiteInfo(
                path=rel,
                name=cfg.name,
                dataset=shown(dataset),
                cases=len(cases) if cases is not None else None,
                target=cfg.system.target,
                judge=cfg.judge.target if cfg.judge else None,
                min_cases=cfg.gate.min_cases if cfg.gate else DEFAULT_MIN_CASES,
                reachability=reachability,
                reaches=reaches,
                reason=reason,
            )
        )
        count = f"{len(cases)} cases" if cases is not None else "dataset unreadable"
        findings.append(
            Finding(
                "suite",
                rel,
                1,
                None,
                f"{cfg.name}: {count}, target {cfg.system.target}",
                None,
                {"name": cfg.name, "dataset": shown(dataset)},
            )
        )

    # -- prompt files and test cases
    for rel in prompt_files:
        if rel in suite_rels:
            continue
        text = _read(root / rel)
        if text is None or not text.strip():
            continue
        findings.append(
            Finding(
                "prompt",
                rel,
                1,
                None,
                f"prompt file ({len(text.strip())} characters)",
                None,
                {"name": posixpath.basename(rel), "excerpt": text.strip()[:EXCERPT_LIMIT]},
            )
        )
    findings.extend(_dataset_findings(root, yaml_files, data_files, suite_datasets))
    model_files = {
        f.path for f in findings if f.kind in ("model_call", "agent") and f.language == "python"
    }
    known_reach = {p for s in suites if s.reachability == "known" for p in s.reaches}
    findings.extend(_pytest_findings(py, index, edges, model_files, known_reach, scan_ref))

    findings.sort(key=lambda f: (FINDING_KINDS.index(f.kind), f.path, f.line))
    gaps, not_computed = _gaps(
        findings,
        suites,
        suite_cases,
        checked,
        py,
        scan_ref,
        {p for p, i in ts.items() if i.is_test},
    )
    return ScanReport(
        root=root.as_posix(),
        files=len(walked.files),
        findings=findings,
        suites=suites,
        gaps=gaps,
        not_computed=not_computed,
        notes=notes,
        unparsable=unparsable,
        too_large=walked.too_large,
        invalid_suites=invalid,
    )


def _merge_tools(findings: list[Finding]) -> list[Finding]:
    """A tool listed in an agent's ``tools=`` that another file defines is that tool, not a second
    one; a listed tool with no definition in the scan stays, once per name."""
    defined = {f.extra["name"] for f in findings if f.kind == "tool" and "listed" not in f.extra}
    seen: set[str] = set()
    out: list[Finding] = []
    for f in findings:
        if f.kind == "tool" and "listed" in f.extra:
            name = str(f.extra["name"])
            if name in defined or name in seen:
                continue
            seen.add(name)
            f.extra.pop("listed")
        out.append(f)
    return out


def _reach(
    cfg: SuiteConfig,
    suite_rel: str,
    index: ModuleIndex,
    edges: dict[str, set[str]],
    graph: TsGraph,
) -> tuple[str, list[str], str | None]:
    """What the suite's target runs: ``known`` with the files it reaches for a ``python:`` target
    found in the repository, or for a ``command:`` that starts a script with node, tsx, bun or deno
    and whose imports can all be followed; otherwise ``unknown`` with the reason."""
    target = cfg.system.target
    if cfg.system.kind == "command":
        try:
            command = node_command(command_argv(target))
        except ValueError:
            command = None
        if command is not None:
            return _reach_script(target, command, posixpath.dirname(suite_rel), graph)
    if cfg.system.kind != "python":
        return (
            "unknown",
            [],
            f"the target {target!r} is not a python target, so the code it runs is not known",
        )
    parts = target.split(":", 2)
    module = parts[1] if len(parts) > 1 else ""
    start = index.resolve(module, None, first=[posixpath.dirname(suite_rel)])
    if start is None:
        return "unknown", [], f"the python module {module!r} is not among the scanned files"
    return "known", reachable(start, edges), None


def _reach_script(
    target: str, command: NodeCommand, suite_dir: str, graph: TsGraph
) -> tuple[str, list[str], str | None]:
    """The files a ``command:`` target reaches by following the relative imports from its script,
    which the command reads from the suite's directory. A script or an import the scan cannot
    resolve makes the answer ``unknown``: it may lead anywhere."""
    if command.problem:
        return (
            "unknown",
            [],
            f"the target {target!r} {command.problem}, so the code it runs is not known",
        )
    entries: list[str] = []
    for script in command.scripts:
        found = graph.resolve_path(posixpath.join(suite_dir, script))
        if found is None or found not in graph.infos:
            return (
                "unknown",
                [],
                f"the script {script!r} that {target!r} runs is not among the scanned files",
            )
        entries.append(found)
    files, problems = graph.reach(entries)
    if problems:
        shown = "; ".join(problems[:3]) + (
            f"; and {len(problems) - 3} more" if len(problems) > 3 else ""
        )
        return "unknown", [], f"the code {target!r} runs cannot be followed: {shown}"
    return "known", files, None


def _ts_aliases(root: Path, data_files: list[str]) -> tuple[set[str], list[str]]:
    """The names of the packages this repository defines (``name`` in each ``package.json``) and the
    path aliases its ``tsconfig.json`` and ``jsconfig.json`` files declare: an import of either is
    this repository's code, which the scan does not follow."""
    names: set[str] = set()
    keys: list[str] = []
    for rel in data_files:
        base = posixpath.basename(rel).lower()
        is_config = base in ("tsconfig.json", "jsconfig.json") or (
            base.startswith("tsconfig.") and base.endswith(".json")
        )
        if base != "package.json" and not is_config:
            continue
        text = _read(root / rel)
        if text is None:
            continue
        if base == "package.json":
            try:
                data = json.loads(text)
            except ValueError:
                continue
            name = data.get("name") if isinstance(data, dict) else None
            if isinstance(name, str) and name:
                names.add(name)
            continue
        for block in re.finditer(r'"paths"\s*:\s*\{(.*?)\}', text, re.DOTALL):
            for key in re.findall(r'"([^"]+)"\s*:', block.group(1)):
                if key.split("*")[0]:
                    keys.append(key)
    return names, list(dict.fromkeys(keys))


def _dataset_findings(
    root: Path, yaml_files: list[str], data_files: list[str], suite_datasets: set[str]
) -> list[Finding]:
    out: list[Finding] = []

    def add(rel: str, fmt: str | None, options: list[str]) -> None:
        if fmt is None:
            detail = f"test cases that fit more than one format ({', '.join(options)})"
            extra: dict[str, Any] = {"format": None, "candidates": options}
        else:
            command = f"evalproof import-dataset {fmt} {_arg(rel)} --out evals/{_stem(rel)}.jsonl"
            detail = f"{fmt} test cases, not in an evalproof suite"
            extra = {"format": fmt, "candidates": options, "command": command}
        out.append(Finding("test_cases", rel, 1, fmt, detail, None, extra))

    for rel in yaml_files:
        if posixpath.basename(rel).lower().startswith("promptfooconfig"):
            add(rel, "promptfoo", ["promptfoo"])
    for rel in data_files:
        if rel in suite_datasets:
            continue
        try:
            found = candidates(root / rel)
        except Exception:  # not a test-case file, or one this reader cannot take: not a finding
            continue
        if found:
            add(rel, found[0] if len(found) == 1 else None, found)
    return out


def _reaching(targets: set[str], edges: dict[str, set[str]]) -> set[str]:
    """``targets`` and every file that imports one of them, directly or through other files."""
    reverse: dict[str, set[str]] = {}
    for src, dsts in edges.items():
        for dst in dsts:
            reverse.setdefault(dst, set()).add(src)
    seen = set(targets)
    stack = list(targets)
    while stack:
        for nxt in reverse.get(stack.pop(), ()):
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _pytest_findings(
    py: dict[str, PyInfo],
    index: ModuleIndex,
    edges: dict[str, set[str]],
    model_files: set[str],
    known_reach: set[str],
    scan_ref: str,
) -> list[Finding]:
    """A pytest file that imports code which calls a model or builds an agent (itself or through
    the files it imports) has test cases written as assertions. ``imports`` maps each name the test
    imports from such a file to that file (``file::name`` for a name inside it), and includes the
    files a suite's target reaches."""
    tests = {p for p, i in py.items() if i.is_test}
    eligible = (_reaching(model_files, edges) | known_reach) - tests
    out: list[Finding] = []
    for path in sorted(tests):
        info = py[path]
        imports: dict[str, str] = {}
        direct: set[str] = set()
        for ref in info.imports:
            for binding, name, file in index.targets(ref, path):
                if file is None or file == path or file not in eligible:
                    continue
                imports[binding] = f"{file}::{name}" if name else file
                direct.add(file)
        if direct:
            out.append(
                Finding(
                    "test_cases",
                    path,
                    1,
                    "pytest",
                    f"pytest file; imports {', '.join(sorted(direct))}",
                    "python",
                    {
                        "format": "pytest",
                        "imports": imports,
                        "command": f"evalproof draft --scan {_arg(scan_ref)} "
                        f"--out {DRAFTS_OUT} --from-tests-only",
                    },
                )
            )
    return out


def _gaps(
    findings: list[Finding],
    suites: list[SuiteInfo],
    suite_cases: dict[str, list[Case] | None],
    checked: set[str],
    py: dict[str, PyInfo],
    scan_ref: str,
    ts_tests: set[str],
) -> tuple[list[Gap], dict[str, str]]:
    """What the code has that no suite tests. A gap that cannot be told (a dataset that will not
    read, a target whose code is unknown) is listed in the second result with the reason."""
    gaps: list[Gap] = []
    not_computed: dict[str, str] = {}
    scan_arg = _arg(scan_ref)
    hint = "" if any(s.judge for s in suites) else " --target MODEL"
    base = f"evalproof draft --scan {scan_arg} --out {DRAFTS_OUT}{hint}"
    unreadable = [s.path for s in suites if suite_cases.get(s.path) is None]
    all_cases = [c for cases in suite_cases.values() for c in (cases or [])]

    if not suites and findings:  # a repository with nothing to test has no gap
        gaps.append(
            Gap(
                "no_suite",
                "no evalproof suite found: nothing in this code is tested by evalproof yet",
                "evalproof init",
            )
        )

    # -- tools that no check names
    tools: dict[str, Finding] = {}
    for f in findings:
        if f.kind == "tool":
            tools.setdefault(str(f.extra["name"]), f)
    if tools and unreadable:
        not_computed["tool_unchecked"] = (
            f"the dataset of {', '.join(unreadable)} could not be read, so which tools its "
            "checks name is not known"
        )
    elif tools:
        for name, f in sorted(tools.items(), key=lambda kv: (kv[1].path, kv[1].line)):
            if name not in checked:
                gaps.append(
                    Gap(
                        "tool_unchecked",
                        f"no check in any suite names the tool `{name}`",
                        f"{base} --gap tool_unchecked --tool {_arg(name)}",
                        f.path,
                        f.line,
                        {"tool": name},
                    )
                )

    # -- code no suite's target reaches
    test_files = {p for p, i in py.items() if i.is_test} | ts_tests
    code: dict[str, int] = {}
    for f in findings:
        # a TypeScript file that only imports an SDK, with no call found, is not code to test
        is_code = f.kind in ("model_call", "agent", "tool") and not is_import_only(f)
        if is_code and f.path not in test_files:
            code[f.path] = min(code.get(f.path, f.line), f.line)
    unknown = [s for s in suites if s.reachability != "known"]
    if code and unknown:
        reasons = "; ".join(f"{s.path}: {s.reason}" for s in unknown)
        not_computed["uncovered_code"] = (
            f"which code the suites run is not known ({reasons}), so no file is reported as "
            "untested"
        )
    elif code:
        reached = {p for s in suites for p in s.reaches}
        for path, line in sorted(code.items()):
            if path not in reached:
                gaps.append(
                    Gap(
                        "uncovered_code",
                        "no suite's target runs this file's model call, agent or tool"
                        if suites
                        else "this file's model call, agent or tool is not in any suite",
                        f"{base} --gap uncovered_code --file {_arg(path)}",
                        path,
                        line,
                    )
                )

    # -- multi-turn use with no multi-turn case
    evidence = [f for f in findings if f.kind == "multi_turn"]
    if evidence and unreadable:
        not_computed["no_multi_turn_cases"] = (
            f"the dataset of {', '.join(unreadable)} could not be read, so whether it has "
            "multi-turn cases is not known"
        )
    elif evidence and not any(c.turns for c in all_cases):
        first = evidence[0]
        gaps.append(
            Gap(
                "no_multi_turn_cases",
                f"the code keeps a conversation ({len(evidence)} place(s)) but no case has "
                "a later turn",
                f"{base} --gap no_multi_turn_cases",
                first.path,
                first.line,
            )
        )

    # -- test cases that are not evalproof cases yet
    for f in findings:
        if f.kind != "test_cases":
            continue
        if f.extra.get("format") == "pytest":
            message = "pytest assertions over this code are test cases evalproof does not have"
        elif f.extra.get("format"):
            message = f"{f.extra['format']} test cases that no evalproof suite uses"
        else:
            message = "test cases that no evalproof suite uses (the format is ambiguous)"
        suggest = str(f.extra.get("command") or f"evalproof import-dataset <format> {_arg(f.path)}")
        gaps.append(
            Gap(
                "test_cases_not_imported",
                message,
                suggest,
                f.path,
                f.line,
                {"format": f.extra.get("format")},
            )
        )

    # -- a suite too small for its gate
    for s in suites:
        if s.cases is not None and s.cases < s.min_cases:
            need = s.min_cases - s.cases
            room = f" --max-cases {need}" if need > DEFAULT_MAX_DRAFTS else ""
            gaps.append(
                Gap(
                    "few_cases",
                    f"{s.path} has {s.cases} case(s); its gate wants at least {s.min_cases}",
                    f"{base} --gap few_cases --suite {_arg(s.path)} --per-gap {need}{room}",
                    s.path,
                    1,
                    {"suite": s.path, "cases": s.cases, "min_cases": s.min_cases},
                )
            )

    gaps.sort(key=lambda g: (GAP_KINDS.index(g.kind), g.path or "", g.line or 0))
    return gaps, not_computed
