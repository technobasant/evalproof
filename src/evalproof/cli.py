"""Command line.

``evalproof init | run | import | import-dataset | compare | verify | rehash | calibrate | report |
doctor | pairwise | readiness | hook | scan | draft``
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError

from evalproof._fsutil import safe_filename
from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache, case_scope
from evalproof.calibration import calibrate
from evalproof.compare import AGGREGATES, Comparison, ComparisonRefused, compare_runs, run_label
from evalproof.config import (
    MAX_TRIALS,
    ConfigError,
    GateConfig,
    SuiteConfig,
    format_validation_error,
    load_config,
)
from evalproof.dataset import (
    DatasetError,
    load_dataset,
    make_tag_filter,
    select_cases,
    write_dataset,
)
from evalproof.doctor import CANARY_MODEL, run_cursor_canary, run_doctor
from evalproof.draft import GAP_CHOICES, DraftOptions, emit_warnings, render_result, run_draft
from evalproof.experimental import experimental_notice
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    GateDecision,
    absolute_checks,
    continuous_floors,
    decide,
    refusal,
)
from evalproof.hook import DEFAULT_RUN, DEFAULT_SUITE, install_hook, uninstall_hook
from evalproof.importers import DATASET_FORMATS, EPOCHS, FORMATS, import_cases, import_run
from evalproof.junit import gate_testsuite, run_testsuite, testsuites, write_junit
from evalproof.models import CaseResult, Run, TagFilter
from evalproof.providers import ProviderError, make_provider
from evalproof.providers.base import close_resource
from evalproof.readiness import (
    assess,
    exit_code,
    parse_required,
    readiness_json,
    readiness_markdown,
)
from evalproof.rehash import RehashError, rehash_run
from evalproof.report import (
    MatrixEntry,
    calibration_markdown,
    comparison_markdown,
    matrix_markdown,
    run_markdown,
    to_json,
)
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scaffold import (
    CI_MODES,
    CLI_AUTHS,
    PLATFORMS,
    choose_judge,
    init_project,
    render_import_suite,
)
from evalproof.scan import render_text, report_json, scan_path
from evalproof.scorers import pairwise
from evalproof.stats import holm
from evalproof.verify import verify_markdown, verify_run

EXIT_INTERRUPTED = 130  # the shell's code for a process stopped by Ctrl-C
HOOK_ACTIONS = ("install", "uninstall")


def _emit(markdown: str, path: str | None, *, quiet: bool = False) -> None:
    """Print the report to stdout (unless ``quiet``) and append it to ``path`` if given."""
    if not quiet:
        sys.stdout.write(markdown)
    if path:
        # Append, so it can target $GITHUB_STEP_SUMMARY alongside other steps.
        # ``newline="\n"``: the file is LF on Windows too, like the one a Linux runner writes.
        with Path(path).open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(markdown + "\n")


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _trials(text: str) -> int:
    """argparse type of ``--trials``: a whole number from 1 to ``MAX_TRIALS``."""
    try:
        value = int(text)
    except ValueError:
        value = 0
    if not 1 <= value <= MAX_TRIALS:
        raise argparse.ArgumentTypeError(f"must be a whole number from 1 to {MAX_TRIALS}")
    return value


def _add_tag_options(parser: argparse.ArgumentParser, verb: str) -> None:
    """``--tags`` and ``--exclude-tags``: repeatable, and each value may list several tags."""
    parser.add_argument(
        "--tags",
        action="append",
        metavar="TAG,...",
        help=f"{verb} only the cases that carry at least one of these tags "
        "(comma-separated; may repeat)",
    )
    parser.add_argument(
        "--exclude-tags",
        action="append",
        metavar="TAG,...",
        help=f"{verb} no case that carries any of these tags (comma-separated; may repeat); "
        "wins over --tags",
    )


def _check_writable(path: Path, what: str = "the run file") -> None:
    """Fail before any case runs if the run file (or ``what`` else) cannot be written: create
    the parent, then create and delete a file in it."""
    if path.is_dir():
        raise ConfigError(f"cannot write {what} {path}: it is a directory")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evalproof-write-check-"):
            pass
    except OSError as exc:
        raise ConfigError(f"cannot write {what} {path}: {exc}") from exc


def _selected_variants(cfg: SuiteConfig, args: argparse.Namespace) -> list[str | None]:
    """The variants to run: ``[None]`` (the suite as written) unless ``--variant`` or
    ``--all-variants`` pick some."""
    if args.variant and args.all_variants:
        raise ConfigError("use --variant or --all-variants, not both")
    if args.all_variants:
        if not cfg.variants:
            raise ConfigError("--all-variants: the suite defines no variants")
        return [v.name for v in cfg.variants]
    if args.variant:
        names = list(dict.fromkeys(args.variant))
        for name in names:
            cfg.variant(name)  # an unknown name is a ConfigError listing the real ones
        return list(names)
    return [None]


def _run_paths(args: argparse.Namespace, variants: list[str | None]) -> list[str | None]:
    """Where each run is written; None is the timestamped default of a run without variants.
    A variant's default is ``runs/<variant>.json``; ``--out`` names the file only when exactly
    one variant runs (with several it would make them overwrite each other)."""
    if variants == [None]:
        return [args.out]
    if args.out and len(variants) > 1:
        raise ConfigError(
            "--out names one file but several variants run; leave it out and each variant "
            "writes runs/<variant>.json"
        )
    return [args.out or f"runs/{name}.json" for name in variants]


def _worst(codes: list[int]) -> int:
    """The exit code of several runs: a usage error, then a gate that could not decide, then a
    regression, then a budget breach, then a pass."""
    for code in (EXIT_USAGE, EXIT_CANNOT_GATE, EXIT_REGRESSION, EXIT_BUDGET):
        if code in codes:
            return code
    return EXIT_PASS


def _run_variant(
    args: argparse.Namespace,
    cfg: SuiteConfig,
    variant: str | None,
    out_path: str | None,
    junit_runs: list[Run] | None = None,
    tag_filter: TagFilter | None = None,
) -> int:
    override = None
    if args.system or args.system_prompt_file or args.label:
        override = cfg.variant_system(variant).model_copy(deep=True)
        if args.system:
            override.target = args.system
        if args.label:
            override.label = args.label
        if args.system_prompt_file:
            override.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8-sig")
            if not args.label:  # so two runs with different prompt files are told apart
                base = override.label or override.target
                override.label = f"{base} + {Path(args.system_prompt_file).name}"

    def progress(i: int, n: int, res: CaseResult) -> None:
        if not args.quiet:
            flag = " (cached)" if res.output.cached else ""
            err = f" ERROR {res.output.error[:60]}" if res.output.error else ""
            sys.stderr.write(f"\r[{i}/{n}] {res.case_id}{flag}{err}      ")
            if i == n:
                sys.stderr.write("\n")

    def plan(cases: int, trials: int) -> None:
        if trials > 1 and not args.quiet:
            sys.stderr.write(f"{cases} cases × {trials} trials = {cases * trials} runs\n")

    if variant is not None and not args.quiet:
        sys.stderr.write(f"variant {variant}\n")
    run = run_suite(
        cfg,
        limit=args.limit,
        tag_filter=tag_filter,
        use_cache=not args.no_cache,
        cache_only=args.cache_only,
        progress=progress,
        system_override=override,
        variant=variant,
        trials=args.trials,
        on_plan=plan,
    )
    # The suite's name goes into a file name, so it is cleaned for every OS; a ``--out`` is not.
    stamp = run.info.started_at.replace(":", "")
    out = out_path or f"runs/{safe_filename(cfg.name)}-{stamp}.json"
    save_run(run, out)
    sys.stderr.write(f"wrote {out}\n")
    if args.junit and junit_runs is not None:
        # The file holds every run finished so far, so a later variant that stops the command
        # still leaves the earlier ones reported.
        junit_runs.append(run)
        write_junit(args.junit, testsuites(run_testsuite(r) for r in junit_runs))
        sys.stderr.write(f"wrote {args.junit}\n")
    markdown = run_markdown(run, args.show)
    failures: list[str] = []
    refused: list[str] = []
    if args.gate and cfg.gate is not None:
        refused = continuous_floors(run, cfg.gate)
        failures, notes = absolute_checks(run, cfg.gate)
        failures = refused + failures
        verdict = "⚠️ cannot gate" if refused else ("❌ fail" if failures else "✅ pass")
        markdown += f"\n**Gate (absolute checks only): {verdict}**\n\n"
        markdown += "".join(f"- {r}\n" for r in failures)
        markdown += "".join(f"- ⚠️ {w}\n" for w in notes)
    _emit(markdown, args.markdown)
    if run.results and all(r.output.error for r in run.results):
        first = _clip(next(r.output.error or "" for r in run.results))
        sys.stderr.write(f"evalproof: error: every case errored; the first error: {first}\n")
        return EXIT_USAGE
    if refused:
        return EXIT_CANNOT_GATE
    return EXIT_REGRESSION if failures else EXIT_PASS


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.suite)
    if args.gate and (cfg.gate is None or not cfg.gate.has_absolute_checks):
        raise ConfigError(
            "run --gate needs absolute checks: set gate.min_pass_rate or gate.must_pass in the "
            "suite (a comparison with a baseline is `evalproof compare --gate`)"
        )
    variants = _selected_variants(cfg, args)
    paths = _run_paths(args, variants)
    tag_filter = make_tag_filter(args.tags, args.exclude_tags)
    for path in paths:
        _check_writable(Path(path) if path else Path("runs") / "run.json")
    if args.junit:
        _check_writable(Path(args.junit), "the JUnit file")
    junit_runs: list[Run] = []
    return _worst(
        [
            _run_variant(args, cfg, v, path, junit_runs, tag_filter)
            for v, path in zip(variants, paths, strict=True)
        ]
    )


def _gate_config(args: argparse.Namespace) -> GateConfig | None:
    base: GateConfig | None = None
    if args.suite:
        base = load_config(args.suite).gate
    overrides = {
        k: v
        for k, v in {
            # --metric may repeat: the first fills ``metric``, the rest ``metrics`` (Holm).
            "metric": args.metric[0] if args.metric else None,
            "metrics": args.metric[1:] if args.metric else None,
            "max_drop": args.max_drop,
            "alpha": args.alpha,
            "min_cases": args.min_cases,
            "max_p95_latency_ms": args.max_p95_ms,
            "max_cost_usd": args.max_cost,
            "allow_judge_change": args.allow_judge_change,
            "min_pass_rate": args.min_pass_rate,
            "must_pass": args.must_pass,
            "max_new_failures": args.max_new_failures,
            "aggregate": args.aggregate,
        }.items()
        if v is not None
    }
    if base is None and "metric" not in overrides:
        return None
    # Validate the merged config: model_copy would skip the check that a metric is named,
    # and an empty --metric (an unset variable in CI) would then gate nothing and pass.
    merged = overrides if base is None else {**base.model_dump(), **overrides}
    try:
        return GateConfig.model_validate(merged)
    except ValidationError as exc:
        raise ConfigError(format_validation_error(exc, "gate", GateConfig)) from exc


def _write_gate_junit(
    args: argparse.Namespace,
    baseline: Run,
    candidates: list[tuple[str, Run, GateDecision]],
    gate: GateConfig,
    exit_code: int,
) -> None:
    """Write ``--junit``: the gate's checks, whatever the verdict. Does nothing without it."""
    if args.junit:
        suite = gate_testsuite(baseline, candidates, gate, exit_code)
        write_junit(args.junit, testsuites([suite]))


def _refused(
    args: argparse.Namespace, message: str, baseline: Run, candidate: Run, gate: GateConfig | None
) -> int:
    """Two runs that cannot be compared at all. Under ``--gate`` that is a gate that cannot
    decide (exit 4), reported like any other; without it, a usage error."""
    if not args.gate:
        raise ConfigError(message)
    decision = refusal(message)
    _emit(
        f"## evalproof\n\n**Gate: ⚠️ cannot gate** (exit {EXIT_CANNOT_GATE})\n\n- {message}\n",
        args.markdown,
    )
    if args.json:
        payload = {"gate": json.loads(to_json(decision))}
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    if gate is not None:
        _write_gate_junit(
            args, baseline, [(run_label(candidate), candidate, decision)], gate, EXIT_CANNOT_GATE
        )
    return EXIT_CANNOT_GATE


def _compare_matrix(
    args: argparse.Namespace,
    baseline: Run,
    gate: GateConfig | None,
    alpha: float,
    aggregate: str,
) -> int:
    """One baseline against several candidates. Each gets its own comparison and decision, and
    Holm's correction runs once over every candidate and every gated metric, so adding a
    candidate cannot make a false failure likelier."""
    entries: list[MatrixEntry] = []
    for path in args.candidate:
        run = load_run(path)
        label = run_label(run)
        try:
            cmp = compare_runs(baseline, run, alpha=alpha, seed=args.seed, aggregate=aggregate)
        except ComparisonRefused as exc:
            if not args.gate:
                raise ConfigError(f"{path}: {exc}") from exc
            message = f"{label}: {exc}"
            entries.append(MatrixEntry(label, path, None, refusal(message), run, message))
            continue
        entries.append(MatrixEntry(label, path, cmp, None, run))
    # Candidates that share a label (two runs of one system) are told apart by their file.
    labels = [e.label for e in entries]
    for entry in entries:
        if labels.count(entry.label) > 1:
            entry.label = f"{entry.label} [{Path(entry.path).name}]"

    if gate is not None:
        family: list[tuple[int, str, float]] = []
        for i, entry in enumerate(entries):
            if entry.comparison is None:
                continue
            for name in gate.gated:
                if name in entry.comparison.metrics:
                    family.append((i, name, entry.comparison.metrics[name].p_value))
        adjusted = holm([p for _, _, p in family])
        for i, entry in enumerate(entries):
            if entry.comparison is None:
                continue
            mine = {n: adjusted[j] for j, (c, n, _) in enumerate(family) if c == i}
            entry.decision = decide(
                entry.comparison, gate, entry.run, adjusted=mine, candidates=len(entries)
            )
    _emit(
        matrix_markdown(run_label(baseline), entries, gate, show=args.show, baseline=baseline),
        args.markdown,
    )
    if args.json:
        payload = {
            "baseline": run_label(baseline),
            "candidates": [
                {
                    "label": e.label,
                    "path": e.path,
                    "comparison": json.loads(to_json(e.comparison)) if e.comparison else None,
                    "gate": json.loads(to_json(e.decision)) if e.decision else None,
                }
                for e in entries
            ],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    code = _worst([e.decision.exit_code for e in entries if e.decision is not None])
    if gate is not None and args.gate:
        _write_gate_junit(
            args,
            baseline,
            [(e.label, e.run or baseline, e.decision) for e in entries if e.decision is not None],
            gate,
            code,
        )
    return code if args.gate else EXIT_PASS


def cmd_compare(args: argparse.Namespace) -> int:
    if args.junit:
        if not args.gate:
            raise ConfigError(
                "--junit needs --gate: a comparison without a gate has no checks to report"
            )
        _check_writable(Path(args.junit), "the JUnit file")
    baseline = load_run(args.baseline)
    candidates = [load_run(path) for path in args.candidate]
    known = list(
        dict.fromkeys(
            [*baseline.scorer_names(), *(n for run in candidates for n in run.scorer_names())]
        )
    )
    for name in args.metric or []:
        if name not in known:
            raise ConfigError(
                f"--metric {name!r} is not a scorer in either run; "
                f"available: {', '.join(known) or 'none'}"
            )
    gate = _gate_config(args)
    if args.gate and gate is None:
        raise ConfigError("--gate needs a gate: pass --suite with a gate block, or --metric")
    alpha = gate.alpha if gate else (args.alpha or 0.05)
    aggregate = args.aggregate or (gate.aggregate if gate else "mean")
    if len(candidates) > 1:
        return _compare_matrix(args, baseline, gate, alpha, aggregate)
    candidate = candidates[0]
    try:
        cmp: Comparison = compare_runs(
            baseline, candidate, alpha=alpha, seed=args.seed, aggregate=aggregate
        )
    except ComparisonRefused as exc:
        return _refused(args, str(exc), baseline, candidate, gate)
    decision = decide(cmp, gate, candidate) if gate else None
    _emit(
        comparison_markdown(
            cmp,
            decision,
            gate.gated if gate else None,
            runs=(baseline, candidate),
            show=args.show,
        ),
        args.markdown,
    )
    if args.json:
        payload = {"comparison": json.loads(to_json(cmp))}
        if decision:
            payload["gate"] = json.loads(to_json(decision))
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    if args.gate and decision is not None:
        if gate is not None:
            _write_gate_junit(
                args,
                baseline,
                [(run_label(candidate), candidate, decision)],
                gate,
                decision.exit_code,
            )
        return decision.exit_code
    return EXIT_PASS


def cmd_verify(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    result = verify_run(run, load_config(args.suite))
    _emit(verify_markdown(result, args.run, args.suite), args.markdown)
    return EXIT_PASS if result.ok else EXIT_CANNOT_GATE


def cmd_rehash(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    try:
        rehashed = rehash_run(run, load_config(args.suite))
    except RehashError as exc:
        sys.stderr.write(f"evalproof: error: {exc}\n")
        return EXIT_CANNOT_GATE
    out = Path(args.out or args.run)
    _check_writable(out)
    save_run(rehashed, out)
    n_cases = len({r.case_id for r in rehashed.results})
    trials = f" × {rehashed.trials} trials" if rehashed.trials > 1 else ""
    sys.stderr.write(f"rehashed {n_cases} cases{trials} in {out}\n")
    return EXIT_PASS


def cmd_readiness(args: argparse.Namespace) -> int:
    required = parse_required(args.require)  # a bad name is a usage error before any file is read
    cfg = load_config(args.suite)
    cases = load_dataset(cfg.resolve(cfg.dataset))
    run = load_run(args.run) if args.run else None
    report = assess(cfg, cases, run, args.run)
    sys.stdout.write(
        readiness_json(report, required) if args.json else readiness_markdown(report, required)
    )
    return exit_code(report, required)


def cmd_scan(args: argparse.Namespace) -> int:
    experimental_notice("scan", "its findings, gaps and JSON shape may change in a minor release")
    out = Path(args.out) if args.out else None
    if out is not None:
        _check_writable(out, "the scan file")
    report = scan_path(
        Path(args.path),
        excludes=args.exclude or [],
        suite_files=[Path(s) for s in args.suite or []],
        scan_ref=args.out or "scan.json",
    )
    if out is not None:
        out.write_text(report_json(report), encoding="utf-8", newline="\n")
        sys.stderr.write(f"wrote {out}\n")
    sys.stdout.write(report_json(report) if args.json else render_text(report))
    return EXIT_PASS


def cmd_draft(args: argparse.Namespace) -> int:
    experimental_notice("draft", "its drafted cases and flags may change in a minor release")
    options = DraftOptions(
        scan=Path(args.scan),
        out=Path(args.out),
        target=args.target,
        suite=Path(args.suite) if args.suite else None,
        per_gap=args.per_gap,
        max_cases=args.max_cases,
        from_tests_only=args.from_tests_only,
        no_cache=args.no_cache,
        force=args.force,
        gaps=args.gap or (),
        tools=args.tool or (),
        files=args.file or (),
    )
    if not options.out.exists() or options.force:
        _check_writable(options.out, "the drafts file")
    result = run_draft(options)
    emit_warnings(result)
    sys.stdout.write(render_result(result, options.out))
    return EXIT_PASS


def cmd_doctor(args: argparse.Namespace) -> int:
    if args.cursor_canary:
        if args.suite:
            sys.stderr.write("evalproof: error: --cursor-canary does not take a SUITE\n")
            return EXIT_USAGE
        return run_cursor_canary(args.model)
    if args.model:
        sys.stderr.write("evalproof: error: --model only applies with --cursor-canary\n")
        return EXIT_USAGE
    return run_doctor(args.suite)


def _say(line: str) -> None:
    sys.stdout.write(line + "\n")


def cmd_init(args: argparse.Namespace) -> int:
    return init_project(
        Path(args.dir),
        judge=args.judge,
        ci=args.ci,
        skill=not args.no_skill,
        force=args.force,
        cli_auth=args.cli_auth,
        platform=args.platform,
        agents_md=not args.no_agents_md,
        agent_files_only=args.agent_files_only,
        from_file=Path(args.from_file) if args.from_file else None,
        from_format=args.from_format,
        node=args.node,
        emit=_say,
    )


def cmd_hook(args: argparse.Namespace) -> int:
    directory = Path.cwd()
    if args.action == "uninstall":
        return uninstall_hook(directory, emit=_say)
    return install_hook(directory, suite=args.suite, run=args.run, force=args.force, emit=_say)


def cmd_import(args: argparse.Namespace) -> int:
    fields = {
        name: value
        for name in ("id", "metric", "score", "passed", "system")
        if (value := getattr(args, f"{name}_field")) is not None
    }

    def warn(message: str) -> None:
        sys.stderr.write(f"evalproof: warning: {message}\n")

    run = import_run(
        args.format,
        args.file,
        system=args.system,
        epochs=args.epochs,
        warn=warn,
        fields=fields,
        wide=args.wide,
        threshold=args.threshold,
    )
    save_run(run, args.out)
    sys.stderr.write(f"wrote {args.out}\n")
    _emit(run_markdown(run), args.markdown, quiet=args.quiet)
    return EXIT_PASS


def _dataset_ref(dataset: Path, suite: Path) -> str:
    """How a suite written at ``suite`` names the dataset at ``dataset`` (relative to the suite)."""
    try:
        return Path(os.path.relpath(dataset.resolve(), suite.resolve().parent)).as_posix()
    except ValueError:  # another drive on Windows: there is no relative path
        return dataset.resolve().as_posix()


def cmd_import_dataset(args: argparse.Namespace) -> int:
    def warn(message: str) -> None:
        sys.stderr.write(f"evalproof: warning: {message}\n")

    suite_out = Path(args.suite_out) if args.suite_out else None
    if suite_out is not None:
        if suite_out.exists() and not args.force:
            raise ConfigError(f"{suite_out} exists; pass --force to overwrite it")
        _check_writable(suite_out, "the suite")
    input_fields = tuple(
        name.strip()
        for value in args.input_field or []
        for name in value.split(",")
        if name.strip()
    )
    imported = import_cases(
        args.format,
        args.file,
        warn=warn,
        id_field=args.id_field,
        input_fields=input_fields,
        reference_field=args.reference_field,
        tags_field=args.tags_field,
    )
    suite_text = None
    if suite_out is not None:  # rendered before anything is written: it can refuse
        suite_text = render_import_suite(
            imported,
            choose_judge(None),
            source=Path(args.file).name,
            dataset_path=_dataset_ref(Path(args.out), suite_out),
        )
    write_dataset(imported.cases, args.out)
    if imported.detected:
        sys.stderr.write(f"evalproof: detected format: {imported.format}\n")
    for note in imported.notes:
        sys.stderr.write(f"evalproof: note: {note}\n")
    sys.stderr.write(f"wrote {len(imported.cases)} case(s) to {args.out}\n")
    if suite_out is not None and suite_text is not None:
        suite_out.write_text(suite_text, encoding="utf-8", newline="\n")
        sys.stderr.write(f"wrote suite to {suite_out}\n")
    return EXIT_PASS


def cmd_calibrate(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    labels = None
    if args.labels:
        labels = {}
        for lineno, line in enumerate(
            Path(args.labels).read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            if not line.strip():
                continue
            where = f"{args.labels}:{lineno}"
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(f"{where}: not valid JSON ({exc.msg})") from exc
            if not isinstance(rec, dict) or "id" not in rec or "label" not in rec:
                raise ConfigError(f'{where}: each line needs an "id" and a "label"')
            labels[str(rec["id"])] = bool(rec["label"])
    rep = calibrate(
        run,
        args.judge_scorer,
        labels=labels,
        gold_scorer=args.gold_scorer,
        gold_sample=args.gold_sample,
        seed=args.seed,
        alpha=args.alpha,
    )
    _emit(calibration_markdown(rep), args.markdown)
    if args.json:
        Path(args.json).write_text(to_json(rep), encoding="utf-8", newline="\n")
    return EXIT_PASS


def cmd_report(args: argparse.Namespace) -> int:
    _emit(run_markdown(load_run(args.run), args.show), args.markdown)
    return EXIT_PASS


def cmd_pairwise(args: argparse.Namespace) -> int:
    cfg = load_config(args.suite)
    if cfg.judge is None:
        raise ConfigError("pairwise needs a 'judge' in the suite")
    tag_filter = make_tag_filter(args.tags, args.exclude_tags)
    judge = make_provider(cfg.judge)
    with ExitStack() as owned:  # closes the judge's own HTTP client, never a passed one
        owned.callback(close_resource, judge)
        if not args.no_cache:
            judge = CachedProvider(
                judge,
                ResponseCache(cfg.resolve(cfg.cache_dir)),
                cfg.judge.base_url,
                cache_only=args.cache_only,
                role="judge",
            )
        selected = select_cases(load_dataset(cfg.resolve(cfg.dataset)), tag_filter)
        cases = {c.id: c for c in selected}
        a_by, b_by = load_run(args.baseline).by_id(), load_run(args.candidate).by_id()
        ids = [cid for cid in sorted(set(a_by) & set(b_by)) if cid in cases][: args.limit]
        tally = {"a": 0, "b": 0, "tie": 0}
        inconsistent = 0
        rows = []
        for cid in ids:
            with case_scope(cid):
                res = pairwise(
                    judge,
                    cfg.judge.params,
                    cases[cid],
                    a_by[cid].output.output,
                    b_by[cid].output.output,
                    swap=not args.no_swap,
                )
            tally[res.winner] += 1
            inconsistent += 0 if res.consistent else 1
            rows.append({"id": cid, **res.__dict__})
        n = len(ids)
        md = [
            "## Pairwise preference",
            "",
            f"{n} cases · baseline wins {tally['a']} · candidate wins {tally['b']} · "
            f"ties {tally['tie']}",
        ]
        if not args.no_swap:
            md.append(
                f"Position-inconsistent verdicts (flipped when the order was swapped): "
                f"{inconsistent}/{n} ({100 * inconsistent / n if n else 0:.1f}%) — reported as ties"
            )
        _emit("\n".join(md) + "\n", args.markdown)
        if args.json:
            Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8", newline="\n")
        return EXIT_PASS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="evalproof", description=__doc__)
    p.add_argument("--version", action="version", version=f"evalproof {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    ini = sub.add_parser(
        "init", help="write a starter suite, a CI workflow and an agent skill into a repository"
    )
    ini.add_argument("--dir", default=".", help="where to write (default: the current directory)")
    ini.add_argument("--judge", help="model for the judge and the system, e.g. claude-code:haiku")
    ini.add_argument(
        "--ci",
        choices=CI_MODES,
        default="keyless",
        help="keyless: CI compares committed run files, no model calls (default); "
        "api-key: CI runs the suite with a secret; none: no workflow",
    )
    ini.add_argument(
        "--platform",
        choices=PLATFORMS,
        default="github",
        help="where CI runs: github writes .github/workflows/evalproof.yml (default); gitlab "
        "writes .gitlab-ci.evalproof.yml, which you include from .gitlab-ci.yml (never edited); "
        "bitbucket writes bitbucket-pipelines.yml, or evals/bitbucket-pipelines.snippet.yml "
        "when that file exists",
    )
    ini.add_argument(
        "--cli-auth",
        choices=CLI_AUTHS,
        default="api-key",
        help="with --ci api-key and a coding-agent CLI judge, what the workflow passes the CLI: "
        "its API key (default), or oauth-token: CLAUDE_CODE_OAUTH_TOKEN from `claude "
        "setup-token`, for claude-code in your own repository and CI only",
    )
    ini.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="start from test cases you already have (a promptfoo config, a DeepEval, Ragas, "
        "LangSmith or Inspect dataset, ...): evals/cases.jsonl is those cases and the suite has "
        "the scorers they need, instead of the sample cases; see `import-dataset`",
    )
    ini.add_argument(
        "--from-format",
        choices=DATASET_FORMATS,
        default="auto",
        help="the tool the --from file is from (default: auto, read off the file)",
    )
    ini.add_argument(
        "--node",
        action="store_true",
        help="the system under test is a Node.js program: write evals/system.mjs (it imports "
        "runCommand from the evalproof npm package) and a suite whose target is "
        "command:node system.mjs, instead of a model with a system prompt; the judge is still a "
        "model (experimental)",
    )
    ini.add_argument("--no-skill", action="store_true", help="do not write the agent skill")
    ini.add_argument(
        "--no-agents-md",
        action="store_true",
        help="do not add the evalproof section to AGENTS.md",
    )
    ini.add_argument(
        "--agent-files-only",
        action="store_true",
        help="write only the agent files (the evalproof section of AGENTS.md and the skill): no "
        "suite and no CI workflow, so --judge, --ci, --platform and --cli-auth do not apply "
        "(experimental)",
    )
    ini.add_argument(
        "--force",
        action="store_true",
        help="overwrite files that already exist (AGENTS.md is never overwritten: only the text "
        "between its evalproof markers is replaced)",
    )
    ini.set_defaults(func=cmd_init)

    r = sub.add_parser("run", help="run a suite and write a run artifact")
    r.add_argument("suite")
    r.add_argument(
        "--out",
        help="where to write the run (with --variant, only when exactly one variant runs)",
    )
    r.add_argument("--limit", type=int, help="run only the first N cases (after --tags)")
    _add_tag_options(r, "run")
    cache_mode = r.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--no-cache", action="store_true", help="do not read or write the response cache"
    )
    cache_mode.add_argument(
        "--cache-only",
        action="store_true",
        help="answer every model call (the judge's too) from the cache; a miss is an error and "
        "no model is called",
    )
    r.add_argument(
        "--trials",
        type=_trials,
        metavar="K",
        help=f"run each case K times (1 to {MAX_TRIALS}), overriding the suite's trials",
    )
    r.add_argument("--system", help="override the system target")
    r.add_argument("--system-prompt-file", help="override the system prompt")
    r.add_argument("--label", help="name this system in reports")
    r.add_argument(
        "--variant",
        action="append",
        metavar="NAME",
        help="run one of the suite's variants and write runs/<variant>.json; repeat to run "
        "several, one run file each",
    )
    r.add_argument(
        "--all-variants",
        action="store_true",
        help="run every variant the suite defines, one run file each",
    )
    r.add_argument("--markdown", help="also append the report to this file")
    r.add_argument(
        "--junit",
        metavar="PATH",
        help="write a JUnit XML report to PATH: one testsuite per run written, one testcase per "
        "case (per trial when the run repeats its cases)",
    )
    r.add_argument("--quiet", action="store_true")
    r.add_argument(
        "--show",
        type=int,
        default=10,
        metavar="N",
        help="list up to N failed cases in the report (default 10; 0 hides the list)",
    )
    r.add_argument(
        "--gate",
        action="store_true",
        help="apply the suite's absolute checks (gate.min_pass_rate, gate.must_pass) to this "
        "run, with no baseline; exit 1 when one fails",
    )
    r.set_defaults(func=cmd_run)

    im = sub.add_parser(
        "import", help="turn another tool's results into a run artifact you can compare and gate"
    )
    im.add_argument("format", choices=FORMATS, help="the tool (or table layout) the file is from")
    im.add_argument("file")
    im.add_argument("--out", required=True, help="where to write the run artifact")
    im.add_argument(
        "--system",
        help="pick one system out of a file that holds several; otherwise the run's label",
    )
    im.add_argument(
        "--epochs",
        choices=EPOCHS,
        default="mean",
        help="how repeats of one case collapse: mean (default), first, all (every repeat must "
        "pass) or any; or trials, which keeps each repeat as its own trial of the case",
    )
    im.add_argument("--markdown", help="also append the report to this file")
    im.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the report to stdout (--markdown still writes it)",
    )
    im.add_argument(
        "--threshold",
        type=float,
        help="garak only: the detector score at which an output counts as a hit (default: the "
        "run's eval_threshold, else 0.5)",
    )
    tab = im.add_argument_group("jsonl and csv only")
    tab.add_argument("--id-field", help="column holding the case id (default: id)")
    tab.add_argument("--metric-field", help="column holding the metric name (default: metric)")
    tab.add_argument("--score-field", help="column holding the continuous score (default: score)")
    tab.add_argument("--passed-field", help="column holding pass/fail (default: passed)")
    tab.add_argument("--system-field", help="column holding the system (default: system)")
    tab.add_argument("--wide", action="store_true", help="one row per case, one column per metric")
    im.set_defaults(func=cmd_import)

    imd = sub.add_parser(
        "import-dataset",
        help="turn another tool's test cases into a dataset (JSONL) for a suite",
        description="Read another tool's test cases and write them as an evalproof dataset, one "
        "case per line. Writes cases, not a run: use `import` for results.",
    )
    imd.add_argument(
        "format",
        choices=DATASET_FORMATS,
        help="the tool the cases are from; auto reads it off the file and says which it chose "
        "(adk: an *.evalset.json or a flat *.test.json list; jsonl and csv: any table, with "
        "--input-field)",
    )
    imd.add_argument("file")
    imd.add_argument(
        "--out", required=True, help="where to write the dataset (an existing file is replaced)"
    )
    imd.add_argument(
        "--suite-out",
        help="also write a suite with the scorers these cases need (a judge block only when one "
        "of them is a judge); an existing file is kept unless --force",
    )
    imd.add_argument(
        "--force", action="store_true", help="overwrite --suite-out when it already exists"
    )
    imd.add_argument(
        "--id-field", help="jsonl and csv only: the column that holds the case id (default: a hash)"
    )
    imd.add_argument(
        "--input-field",
        action="append",
        metavar="FIELD",
        help="jsonl and csv only: the column that holds the input (default: input). Repeat it or "
        "separate names with commas to make the input an object of those columns",
    )
    imd.add_argument(
        "--reference-field",
        help="jsonl and csv only: the column that holds the expected answer (checked by a judge)",
    )
    imd.add_argument(
        "--tags-field",
        help="jsonl and csv only: the column that holds the tags (a list, or comma-separated)",
    )
    imd.set_defaults(func=cmd_import_dataset)

    c = sub.add_parser(
        "compare",
        help="compare two runs, or several candidates with one baseline; --gate turns the "
        "result into an exit code",
        description="Compare CANDIDATE with BASELINE on the cases both runs share. Give several "
        "CANDIDATEs to compare each with the baseline in one table; --gate then corrects the "
        "p-values for all candidates and gated metrics at once and exits with the worst code "
        "(2, then 4, 1, 3, 0).",
    )
    c.add_argument("baseline")
    c.add_argument("candidate", nargs="+", help="one run, or several for a matrix comparison")
    c.add_argument("--suite", help="take gate settings from this suite")
    c.add_argument(
        "--metric",
        action="append",
        help="scorer to gate on; repeat to gate several (Holm-corrected)",
    )
    c.add_argument("--max-drop", type=float)
    c.add_argument("--alpha", type=float)
    c.add_argument(
        "--aggregate",
        choices=AGGREGATES,
        help="when the runs repeat each case (trials > 1), how a case's trials become one value "
        "on a pass/fail metric: mean (the pass rate, default), all (passed every trial) or any "
        "(passed at least once); overrides gate.aggregate",
    )
    c.add_argument("--min-cases", type=int)
    c.add_argument("--max-p95-ms", type=float)
    c.add_argument("--max-cost", type=float)
    c.add_argument(
        "--min-pass-rate",
        type=float,
        help="lowest candidate pass rate on each gated metric; errored cases count as failures",
    )
    c.add_argument(
        "--must-pass",
        action="append",
        metavar="TAG",
        help="a case tag whose cases must all pass every gated metric; repeat for several",
    )
    c.add_argument(
        "--max-new-failures",
        type=int,
        help="most cases that may go from pass to fail on a binary gated metric, whatever the "
        "p-value; 0 suits deterministic scorers",
    )
    c.add_argument(
        "--allow-judge-change",
        action="store_const",
        const=True,
        default=None,
        help="gate even if the judge (target, CLI, isolation or resolved model) differs "
        "between the runs; it is refused otherwise. A new CLI version alone only warns",
    )
    c.add_argument("--gate", action="store_true", help="exit non-zero when the gate fails")
    c.add_argument(
        "--show",
        type=int,
        default=5,
        metavar="N",
        help="show both outputs for up to N regressed cases of the gated metrics (default 5; "
        "0 hides them)",
    )
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--markdown")
    c.add_argument("--json")
    c.add_argument(
        "--junit",
        metavar="PATH",
        help="write a JUnit XML report of the gate's checks to PATH, whatever the verdict; "
        "needs --gate",
    )
    c.set_defaults(func=cmd_compare)

    k = sub.add_parser("calibrate", help="measure a judge against gold labels and correct its rate")
    k.add_argument("run")
    k.add_argument("--judge-scorer", required=True)
    src = k.add_mutually_exclusive_group(required=True)
    src.add_argument("--labels", help="JSONL of {id, label} human labels")
    src.add_argument("--gold-scorer", help="a trusted scorer in the same run")
    k.add_argument(
        "--gold-sample", type=int, help="use a random sample of this size as the gold set"
    )
    k.add_argument("--seed", type=int, default=0)
    k.add_argument("--alpha", type=float, default=0.05)
    k.add_argument("--markdown")
    k.add_argument("--json")
    k.set_defaults(func=cmd_calibrate)

    rep = sub.add_parser("report", help="summarise one run")
    rep.add_argument("run")
    rep.add_argument("--markdown")
    rep.add_argument(
        "--show",
        type=int,
        default=10,
        metavar="N",
        help="list up to N failed cases (default 10; 0 hides the list)",
    )
    rep.set_defaults(func=cmd_report)

    ver = sub.add_parser(
        "verify",
        help="check that a run file matches the suite and dataset, without calling a model",
        description="Check that RUN was made with SUITE's system, judge, scorers and seed, and "
        "scored exactly the dataset's cases as they are now. Makes no model call. Exit 0 when "
        "it matches, 4 when it does not, 2 for a usage error.",
    )
    ver.add_argument("run")
    ver.add_argument("suite")
    ver.add_argument("--markdown", help="also append the verdict to this file")
    ver.set_defaults(func=cmd_verify)

    rh = sub.add_parser(
        "rehash",
        help="add per-case hashes to a run file written before evalproof 0.5, without calling "
        "a model",
        description="Add per-case hashes to RUN, a run file written before evalproof 0.5, so it "
        "can be gated against a changed dataset. Run files do not store case inputs, so this "
        "works only when the cases RUN scored are still in SUITE's dataset, unchanged and in "
        "the same place (cases appended after them are fine), which the dataset_hash stored in "
        "RUN proves. Otherwise check out the dataset as it was, for example on the base branch, "
        "and rehash there. Scores, outputs and the suite hash are not "
        "changed, so `verify` still needs a run made with evalproof 0.5 or newer. Safe to run "
        "again on a file that already has hashes. Makes no model call. Exit 0 on success, 4 "
        "when the dataset does not match the run, 2 for a usage error.",
    )
    rh.add_argument("run")
    rh.add_argument("suite")
    rh.add_argument("--out", help="write the result here instead of overwriting RUN")
    rh.set_defaults(func=cmd_rehash)

    dr = sub.add_parser(
        "doctor",
        help="check the coding-agent CLI targets, without calling a model",
        description="Check each coding-agent CLI target (claude-code, codex, and the "
        "experimental opencode, cursor and copilot): binary, version, login and the "
        "isolation flags. Makes no model call, except with --cursor-canary, which makes one "
        "real cursor-agent call (it uses your Cursor quota) to see what reaches the prompt.",
    )
    dr.add_argument(
        "suite", nargs="?", help="check the targets this suite uses (default: every CLI found)"
    )
    dr.add_argument(
        "--cursor-canary",
        action="store_true",
        help="make ONE real cursor-agent call (uses your Cursor quota; the only doctor check "
        "that does) with a canary skill in its workspace, and report whether the canary and "
        "any other skills or user rules reached the prompt; the answer is model-reported, so "
        "best effort. Exit 0 when only the canary is seen, 1 otherwise, 4 when the call fails",
    )
    dr.add_argument(
        "--model",
        metavar="MODEL",
        help=f"the Cursor model for --cursor-canary (default: {CANARY_MODEL})",
    )
    dr.set_defaults(func=cmd_doctor)

    rd = sub.add_parser(
        "readiness",
        help="report which parts of quality a suite covers and which it leaves untested",
        description="Read SUITE and its dataset (and RUN, if given) and report, for each "
        "dimension of AI quality, whether the suite covers it, covers it in part, leaves it "
        "untested or leaves it to another kind of tool, with the evidence. Static: makes no "
        "model call. Exit 0, or 1 when a --require'd dimension is not covered, 2 for a usage "
        "error.",
    )
    rd.add_argument("suite")
    rd.add_argument("--run", metavar="FILE", help="add what this run file shows (traces, errors)")
    rd.add_argument("--json", action="store_true", help="print JSON instead of a table")
    rd.add_argument(
        "--require",
        action="append",
        metavar="DIM,...",
        help="exit 1 unless these dimensions are covered (comma-separated; may repeat)",
    )
    rd.set_defaults(func=cmd_readiness)

    hk = sub.add_parser(
        "hook",
        help="install or remove a git pre-push hook that runs `evalproof verify`",
        description="install writes a git pre-push hook that runs `evalproof verify RUN SUITE` "
        "before every push, so a run file that no longer matches the suite and dataset stops "
        "the push. Makes no model call. uninstall removes the hook, only when evalproof wrote "
        "it. Exit 0 on success, 2 for a usage error (not in a git repository, or a hook that "
        "evalproof did not write).",
    )
    hk.add_argument("action", metavar="ACTION", choices=HOOK_ACTIONS, help="install or uninstall")
    hk.add_argument(
        "--suite",
        default=DEFAULT_SUITE,
        help="the suite the hook verifies against (default: %(default)s)",
    )
    hk.add_argument(
        "--run",
        default=DEFAULT_RUN,
        help="the committed run file the hook verifies (default: %(default)s)",
    )
    hk.add_argument(
        "--force",
        action="store_true",
        help="install over a pre-push hook that evalproof did not write",
    )
    hk.set_defaults(func=cmd_hook)

    sc = sub.add_parser(
        "scan",
        help="find the model calls, agents and tools in a repository and what no suite tests",
        description="Read the code under PATH without running it and report what it does with "
        "an LLM (model calls, agents, tools, prompts, multi-turn use), the test cases and "
        "suites it already has, and the gaps: code and tools no suite tests, and test cases "
        "that are not evalproof cases yet. Static: makes no model call and imports no code. "
        "Exit 0 with or without gaps, 2 for a usage error. Experimental.",
    )
    sc.add_argument("path", nargs="?", default=".", metavar="PATH")
    sc.add_argument("--json", action="store_true", help="print JSON instead of text")
    sc.add_argument("--out", metavar="FILE", help="also write the JSON to FILE")
    sc.add_argument(
        "--exclude",
        action="append",
        metavar="GLOB",
        help="skip files and directories matching GLOB (may repeat)",
    )
    sc.add_argument(
        "--suite",
        action="append",
        metavar="SUITE",
        help="use this suite file instead of looking for suites (may repeat)",
    )
    sc.set_defaults(func=cmd_scan)

    dr = sub.add_parser(
        "draft",
        help="draft test inputs for what a scan found untested (inputs only, never answers)",
        description="Turn the gaps in a scan file (from `evalproof scan --out`) into a separate "
        "JSONL of draft cases. Cases harvested from pytest files keep the assertion their test "
        "made and call no model. For tools, multi-turn use, uncovered code and thin suites a "
        "model writes inputs only, tagged `draft`, with no expected answer: a person adds the "
        "reference or checks before a draft goes into a dataset. The output is never a suite's "
        "dataset. Experimental.",
    )
    dr.add_argument(
        "--scan", required=True, metavar="FILE", help="the JSON from `evalproof scan --out`"
    )
    dr.add_argument(
        "--out", required=True, metavar="FILE", help="the JSONL to write (refused if it exists)"
    )
    dr.add_argument(
        "--target",
        metavar="MODEL",
        help="the model that writes inputs (default: the suite's judge)",
    )
    dr.add_argument("--suite", metavar="SUITE", help="draft for this suite only")
    dr.add_argument(
        "--per-gap",
        type=int,
        default=3,
        metavar="N",
        help="inputs to ask a model for, for each gap (default 3)",
    )
    dr.add_argument(
        "--max-cases",
        type=int,
        default=30,
        metavar="N",
        help="stop after this many drafts in all (default 30)",
    )
    dr.add_argument(
        "--from-tests-only",
        action="store_true",
        help="draft only from pytest files: no model is called",
    )
    dr.add_argument(
        "--no-cache", action="store_true", help="do not read or write the response cache"
    )
    dr.add_argument("--force", action="store_true", help="overwrite --out if it exists")
    dr.add_argument(
        "--gap",
        action="append",
        choices=GAP_CHOICES,
        metavar="KIND",
        help="draft only for this kind of gap (may repeat; choices: "
        + ", ".join(GAP_CHOICES)
        + ")",
    )
    dr.add_argument(
        "--tool", action="append", metavar="NAME", help="draft only for this tool (may repeat)"
    )
    dr.add_argument(
        "--file",
        action="append",
        metavar="PATH",
        help="draft only for code in this file (may repeat)",
    )
    dr.set_defaults(func=cmd_draft)

    pw = sub.add_parser("pairwise", help="judge baseline vs candidate outputs head to head")
    pw.add_argument("baseline")
    pw.add_argument("candidate")
    pw.add_argument("--suite", required=True)
    pw.add_argument("--limit", type=int, help="judge only the first N cases (after --tags)")
    _add_tag_options(pw, "judge")
    pw.add_argument("--no-swap", action="store_true")
    pw_cache = pw.add_mutually_exclusive_group()
    pw_cache.add_argument(
        "--no-cache", action="store_true", help="do not read or write the response cache"
    )
    pw_cache.add_argument(
        "--cache-only",
        action="store_true",
        help="answer every judge call from the cache; a miss is an error and no model is called",
    )
    pw.add_argument("--markdown")
    pw.add_argument("--json")
    pw.set_defaults(func=cmd_pairwise)
    return p


def _utf8_streams() -> None:
    """Write UTF-8 to stdout and stderr. Redirected on Windows they use the ANSI code page, which
    has no check mark or arrow, and a report would stop with ``UnicodeEncodeError``."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    if sys.platform == "win32":  # pragma: no cover - the test calls _utf8_streams itself
        _utf8_streams()
    args = build_parser().parse_args(argv)
    try:
        code: int = args.func(args)
    except (ConfigError, DatasetError, ProviderError, FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"evalproof: error: {exc}\n")
        return EXIT_USAGE
    except KeyboardInterrupt:
        sys.stderr.write("\nevalproof: interrupted\n")
        return EXIT_INTERRUPTED
    except Exception as exc:
        if os.environ.get("EVALPROOF_DEBUG"):
            raise
        sys.stderr.write(
            f"evalproof: error: {type(exc).__name__}: {exc}\n"
            "set EVALPROOF_DEBUG=1 for the traceback\n"
        )
        return EXIT_USAGE
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
