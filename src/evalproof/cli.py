"""Command line: ``evalproof run | compare | calibrate | report | pairwise``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.calibration import calibrate
from evalproof.compare import compare_runs
from evalproof.config import ConfigError, GateConfig, load_config
from evalproof.dataset import DatasetError, load_dataset
from evalproof.gate import EXIT_PASS, EXIT_USAGE, decide
from evalproof.models import CaseResult
from evalproof.providers import ProviderError, make_provider
from evalproof.report import (
    calibration_markdown,
    comparison_markdown,
    run_markdown,
    to_json,
)
from evalproof.runner import load_run, run_suite, save_run
from evalproof.scorers import pairwise


def _emit(markdown: str, path: str | None) -> None:
    sys.stdout.write(markdown)
    if path:
        # Append, so it can target $GITHUB_STEP_SUMMARY alongside other steps.
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(markdown + "\n")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.suite)
    override = None
    if args.system or args.system_prompt_file or args.label:
        override = cfg.system.model_copy(deep=True)
        if args.system:
            override.target = args.system
        if args.label:
            override.label = args.label
        if args.system_prompt_file:
            override.system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")

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
    _emit(run_markdown(run), args.markdown)
    return EXIT_PASS


def _gate_config(args: argparse.Namespace) -> GateConfig | None:
    base: GateConfig | None = None
    if args.suite:
        base = load_config(args.suite).gate
    overrides = {
        k: v
        for k, v in {
            "metric": args.metric,
            "max_drop": args.max_drop,
            "alpha": args.alpha,
            "min_cases": args.min_cases,
            "max_p95_latency_ms": args.max_p95_ms,
            "max_cost_usd": args.max_cost,
        }.items()
        if v is not None
    }
    if base is None and "metric" not in overrides:
        return None
    if base is None:
        return GateConfig(**overrides)
    return base.model_copy(update=overrides)


def cmd_compare(args: argparse.Namespace) -> int:
    baseline, candidate = load_run(args.baseline), load_run(args.candidate)
    gate = _gate_config(args)
    alpha = gate.alpha if gate else (args.alpha or 0.05)
    cmp = compare_runs(baseline, candidate, alpha=alpha, seed=args.seed)
    decision = decide(cmp, gate) if gate else None
    if args.gate and decision is None:
        raise ConfigError("--gate needs a gate: pass --suite with a gate block, or --metric")
    _emit(comparison_markdown(cmp, decision, gate.metric if gate else None), args.markdown)
    if args.json:
        payload = {"comparison": json.loads(to_json(cmp))}
        if decision:
            payload["gate"] = json.loads(to_json(decision))
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.gate and decision is not None:
        return decision.exit_code
    return EXIT_PASS


def cmd_calibrate(args: argparse.Namespace) -> int:
    run = load_run(args.run)
    labels = None
    if args.labels:
        labels = {}
        for line in Path(args.labels).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
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
    _emit(run_markdown(load_run(args.run)), args.markdown)
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
    r.set_defaults(func=cmd_run)

    c = sub.add_parser(
        "compare", help="compare two runs; --gate turns the result into an exit code"
    )
    c.add_argument("baseline")
    c.add_argument("candidate")
    c.add_argument("--suite", help="take gate settings from this suite")
    c.add_argument("--metric")
    c.add_argument("--max-drop", type=float)
    c.add_argument("--alpha", type=float)
    c.add_argument("--min-cases", type=int)
    c.add_argument("--max-p95-ms", type=float)
    c.add_argument("--max-cost", type=float)
    c.add_argument("--gate", action="store_true", help="exit non-zero when the gate fails")
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
    rep.set_defaults(func=cmd_report)

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
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
