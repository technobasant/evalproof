"""Command line.

``evalproof init | run | import | compare | verify | calibrate | report | doctor | pairwise``
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.calibration import calibrate
from evalproof.compare import compare_runs
from evalproof.config import ConfigError, GateConfig, format_validation_error, load_config
from evalproof.dataset import DatasetError, load_dataset
from evalproof.doctor import run_doctor
from evalproof.gate import (
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    absolute_checks,
    decide,
)
from evalproof.importers import EPOCHS, FORMATS, import_run
from evalproof.models import CaseResult
from evalproof.providers import ProviderError, make_provider
from evalproof.report import (
    calibration_markdown,
    comparison_markdown,
    run_markdown,
    to_json,
)
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scaffold import CI_MODES, init_project
from evalproof.scorers import pairwise
from evalproof.verify import verify_markdown, verify_run

EXIT_INTERRUPTED = 130  # the shell's code for a process stopped by Ctrl-C


def _emit(markdown: str, path: str | None, *, quiet: bool = False) -> None:
    """Print the report to stdout (unless ``quiet``) and append it to ``path`` if given."""
    if not quiet:
        sys.stdout.write(markdown)
    if path:
        # Append, so it can target $GITHUB_STEP_SUMMARY alongside other steps.
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")


def _clip(text: str, limit: int = 200) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _check_writable(path: Path) -> None:
    """Fail before any case runs if the run file cannot be written: create the parent, then
    create and delete a file in it."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".evalproof-write-check-"):
            pass
    except OSError as exc:
        raise ConfigError(f"cannot write the run file {path}: {exc}") from exc


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.suite)
    if args.gate and (cfg.gate is None or not cfg.gate.has_absolute_checks):
        raise ConfigError(
            "run --gate needs absolute checks: set gate.min_pass_rate or gate.must_pass in the "
            "suite (a comparison with a baseline is `evalproof compare --gate`)"
        )
    _check_writable(Path(args.out) if args.out else Path("runs") / "run.json")
    override = None
    if args.system or args.system_prompt_file or args.label:
        override = cfg.system.model_copy(deep=True)
        if args.system:
            override.target = args.system
        if args.label:
            override.label = args.label
        if args.system_prompt_file:
            override.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
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

    run = run_suite(
        cfg,
        limit=args.limit,
        use_cache=not args.no_cache,
        progress=progress,
        system_override=override,
    )
    out = args.out or f"runs/{cfg.name}-{run.info.started_at.replace(':', '')}.json"
    save_run(run, out)
    sys.stderr.write(f"wrote {out}\n")
    markdown = run_markdown(run, args.show)
    failures: list[str] = []
    if args.gate and cfg.gate is not None:
        failures, notes = absolute_checks(run, cfg.gate)
        verdict = "❌ fail" if failures else "✅ pass"
        markdown += f"\n**Gate (absolute checks only): {verdict}**\n\n"
        markdown += "".join(f"- {r}\n" for r in failures)
        markdown += "".join(f"- ⚠️ {w}\n" for w in notes)
    _emit(markdown, args.markdown)
    if run.results and all(r.output.error for r in run.results):
        first = _clip(next(r.output.error or "" for r in run.results))
        sys.stderr.write(f"evalproof: error: every case errored; the first error: {first}\n")
        return EXIT_USAGE
    return EXIT_REGRESSION if failures else EXIT_PASS


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


def cmd_compare(args: argparse.Namespace) -> int:
    baseline, candidate = load_run(args.baseline), load_run(args.candidate)
    known = list(dict.fromkeys([*baseline.scorer_names(), *candidate.scorer_names()]))
    for name in args.metric or []:
        if name not in known:
            raise ConfigError(
                f"--metric {name!r} is not a scorer in either run; "
                f"available: {', '.join(known) or 'none'}"
            )
    gate = _gate_config(args)
    alpha = gate.alpha if gate else (args.alpha or 0.05)
    cmp = compare_runs(baseline, candidate, alpha=alpha, seed=args.seed)
    decision = decide(cmp, gate, candidate) if gate else None
    if args.gate and decision is None:
        raise ConfigError("--gate needs a gate: pass --suite with a gate block, or --metric")
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
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.gate and decision is not None:
        return decision.exit_code
    return EXIT_PASS


def cmd_verify(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    result = verify_run(run, load_config(args.suite))
    _emit(verify_markdown(result, args.run, args.suite), args.markdown)
    return EXIT_PASS if result.ok else EXIT_CANNOT_GATE


def cmd_doctor(args: argparse.Namespace) -> int:
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
        emit=_say,
    )


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
    )
    save_run(run, args.out)
    sys.stderr.write(f"wrote {args.out}\n")
    _emit(run_markdown(run), args.markdown, quiet=args.quiet)
    return EXIT_PASS


def cmd_calibrate(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    labels = None
    if args.labels:
        labels = {}
        for lineno, line in enumerate(
            Path(args.labels).read_text(encoding="utf-8").splitlines(), start=1
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
        Path(args.json).write_text(to_json(rep), encoding="utf-8")
    return EXIT_PASS


def cmd_report(args: argparse.Namespace) -> int:
    _emit(run_markdown(load_run(args.run), args.show), args.markdown)
    return EXIT_PASS


def cmd_pairwise(args: argparse.Namespace) -> int:
    cfg = load_config(args.suite)
    if cfg.judge is None:
        raise ConfigError("pairwise needs a 'judge' in the suite")
    judge = make_provider(cfg.judge)
    if not args.no_cache:
        judge = CachedProvider(judge, ResponseCache(cfg.resolve(cfg.cache_dir)), cfg.judge.base_url)
    cases = {c.id: c for c in load_dataset(cfg.resolve(cfg.dataset))}
    a_by, b_by = load_run(args.baseline).by_id(), load_run(args.candidate).by_id()
    ids = [cid for cid in sorted(set(a_by) & set(b_by)) if cid in cases][: args.limit]
    tally = {"a": 0, "b": 0, "tie": 0}
    inconsistent = 0
    rows = []
    for cid in ids:
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
        Path(args.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
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
    ini.add_argument("--no-skill", action="store_true", help="do not write the agent skill")
    ini.add_argument("--force", action="store_true", help="overwrite files that already exist")
    ini.set_defaults(func=cmd_init)

    r = sub.add_parser("run", help="run a suite and write a run artifact")
    r.add_argument("suite")
    r.add_argument("--out")
    r.add_argument("--limit", type=int)
    r.add_argument("--no-cache", action="store_true")
    r.add_argument("--system", help="override the system target")
    r.add_argument("--system-prompt-file", help="override the system prompt")
    r.add_argument("--label", help="name this system in reports")
    r.add_argument("--markdown", help="also append the report to this file")
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
        "pass) or any",
    )
    im.add_argument("--markdown", help="also append the report to this file")
    im.add_argument(
        "--quiet",
        action="store_true",
        help="do not print the report to stdout (--markdown still writes it)",
    )
    tab = im.add_argument_group("jsonl and csv only")
    tab.add_argument("--id-field", help="column holding the case id (default: id)")
    tab.add_argument("--metric-field", help="column holding the metric name (default: metric)")
    tab.add_argument("--score-field", help="column holding the continuous score (default: score)")
    tab.add_argument("--passed-field", help="column holding pass/fail (default: passed)")
    tab.add_argument("--system-field", help="column holding the system (default: system)")
    tab.add_argument("--wide", action="store_true", help="one row per case, one column per metric")
    im.set_defaults(func=cmd_import)

    c = sub.add_parser(
        "compare", help="compare two runs; --gate turns the result into an exit code"
    )
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--suite", help="take gate settings from this suite")
    c.add_argument(
        "--metric",
        action="append",
        help="scorer to gate on; repeat to gate several (Holm-corrected)",
    )
    c.add_argument("--max-drop", type=float)
    c.add_argument("--alpha", type=float)
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

    dr = sub.add_parser(
        "doctor",
        help="check the coding-agent CLI targets, without calling a model",
        description="Check each coding-agent CLI target (claude-code, codex, and the "
        "experimental opencode, cursor and copilot): binary, version, login and the "
        "isolation flags. Makes no model call.",
    )
    dr.add_argument(
        "suite", nargs="?", help="check the targets this suite uses (default: every CLI found)"
    )
    dr.set_defaults(func=cmd_doctor)

    pw = sub.add_parser("pairwise", help="judge baseline vs candidate outputs head to head")
    pw.add_argument("baseline")
    pw.add_argument("candidate")
    pw.add_argument("--suite", required=True)
    pw.add_argument("--limit", type=int)
    pw.add_argument("--no-swap", action="store_true")
    pw.add_argument("--no-cache", action="store_true")
    pw.add_argument("--markdown")
    pw.add_argument("--json")
    pw.set_defaults(func=cmd_pairwise)
    return p


def main(argv: Sequence[str] | None = None) -> int:
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
