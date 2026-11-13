"""Reports: Markdown for a PR comment or GitHub step summary, JSON for machines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from evalproof.calibration import CalibrationReport, Estimate
from evalproof.compare import Comparison, MetricComparison, dataset_change_note
from evalproof.gate import GateDecision
from evalproof.models import Run
from evalproof.stats import percentile, wilson

STATUS_ICON = {"pass": "✅ pass", "warn": "⚠️ pass with warnings", "fail": "❌ fail"}


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _rate_or_mean(mc: MetricComparison, value: float) -> str:
    return _pct(value) if mc.kind == "binary" else f"{value:.3f}"


def _delta(mc: MetricComparison) -> str:
    if mc.kind == "binary":
        return f"{100 * mc.delta:+.1f} pts [{100 * mc.ci_low:+.1f}, {100 * mc.ci_high:+.1f}]"
    return f"{mc.delta:+.3f} [{mc.ci_low:+.3f}, {mc.ci_high:+.3f}]"


def comparison_markdown(
    cmp: Comparison,
    decision: GateDecision | None = None,
    gate_metric: str | Sequence[str] | None = None,
) -> str:
    gated = [gate_metric] if isinstance(gate_metric, str) else list(gate_metric or [])
    # With several gated metrics the decision carries Holm-adjusted p-values; show them.
    adjusted = decision.adjusted_p if decision is not None and len(gated) > 1 else {}
    lines = ["## evalproof", ""]
    if decision is not None:
        lines += [f"**Gate: {STATUS_ICON[decision.status]}** (exit {decision.exit_code})", ""]
        lines += [f"- {r}" for r in decision.reasons]
        lines += [f"- ⚠️ {w}" for w in decision.warnings]
        if decision.reasons or decision.warnings:
            lines.append("")
    else:
        # With a gate these are already among its reasons and warnings.
        if cmp.judge_changes:
            lines += [
                f"- ⚠️ The judge changed, so the scores may not be comparable: {c}"
                for c in cmp.judge_changes
            ]
        lines += [f"- ⚠️ {w}" for w in cmp.judge_warnings]
        if cmp.system_changes:
            lines += [f"- ⚠️ The system under test changed: {c}" for c in cmp.system_changes]
        note = dataset_change_note(cmp)
        if note:
            lines.append(f"- ⚠️ {note}")
        if cmp.judge_changes or cmp.judge_warnings or cmp.system_changes or note:
            lines.append("")
    lines += [
        f"Baseline `{cmp.baseline_label}` vs candidate `{cmp.candidate_label}` on "
        f"{len(cmp.paired_ids)} paired cases.",
        "",
        "| Metric | Baseline | Candidate | Δ (95% CI) | Test | p |"
        + (" Holm-adjusted p |" if adjusted else ""),
        "|---|---:|---:|---:|---|---:|" + ("---:|" if adjusted else ""),
    ]
    for name, mc in cmp.metrics.items():
        label = f"**{name}**" if name in gated else name
        row = (
            f"| {label} | {_rate_or_mean(mc, mc.baseline)} | {_rate_or_mean(mc, mc.candidate)} | "
            f"{_delta(mc)} | {mc.test} | {mc.p_value:.4f} |"
        )
        if adjusted:
            row += f" {adjusted[name]:.4f} |" if name in adjusted else " — |"
        lines.append(row)
    lines += [
        "",
        "| Run | Cases | Errors | p50 latency | p95 latency | Cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, st in (("baseline", cmp.baseline), ("candidate", cmp.candidate)):
        lines.append(
            f"| {label} | {st.cases} | {st.errors} | {st.p50_latency_ms:.0f} ms | "
            f"{st.p95_latency_ms:.0f} ms | ${st.cost_usd:.4f} |"
        )
    for name in gated:
        focus = cmp.metrics.get(name)
        if focus and focus.regressed:
            shown = ", ".join(f"`{c}`" for c in focus.regressed[:20])
            more = f" and {len(focus.regressed) - 20} more" if len(focus.regressed) > 20 else ""
            lines += ["", f"Regressed on **{focus.name}** ({len(focus.regressed)}): {shown}{more}"]
        if focus and focus.improved:
            lines.append(f"Improved on **{focus.name}**: {len(focus.improved)} case(s)")
    return "\n".join(lines) + "\n"


def run_markdown(run: Run) -> str:
    lines = [
        f"## {run.info.suite} — `{run.info.system}`",
        "",
        f"{len(run.results)} cases · wall {run.info.wall_time_s:.1f} s · "
        f"dataset `{run.info.dataset_hash}`",
        "",
        "| Scorer | Decided | Pass rate (95% CI) | Mean |",
        "|---|---:|---:|---:|",
    ]
    for name in run.scorer_names():
        scores = [r.scores[name] for r in run.results if name in r.scores]
        decided = [s for s in scores if s.value is not None]
        binary = [s.passed for s in decided if s.passed is not None]
        mean = (
            sum(s.value for s in decided if s.value is not None) / len(decided) if decided else None
        )
        if binary:
            k = sum(binary)
            lo, hi = wilson(k, len(binary))
            rate = f"{_pct(k / len(binary))} [{_pct(lo)}, {_pct(hi)}]"
        else:
            rate = "—"
        lines.append(
            f"| {name} | {len(decided)}/{len(scores)} | {rate} | "
            f"{'—' if mean is None else f'{mean:.3f}'} |"
        )
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    lines += [
        "",
        f"Latency p50 {percentile(lat, 50):.0f} ms · p95 {percentile(lat, 95):.0f} ms · "
        f"errors {sum(1 for r in run.results if r.output.error)} · "
        f"cached {sum(1 for r in run.results if r.output.cached)}",
    ]
    return "\n".join(lines) + "\n"


def _est_row(e: Estimate) -> str:
    ci = "—" if e.low is None or e.high is None else f"[{_pct(e.low)}, {_pct(e.high)}]"
    note = f" ({e.note})" if e.note else ""
    return f"| {e.method}{note} | {_pct(e.point)} | {ci} | {e.n} |"


def calibration_markdown(rep: CalibrationReport) -> str:
    cm = rep.confusion
    lines = [
        f"## Judge calibration — `{rep.judge_scorer}`",
        "",
        f"Gold source: {rep.gold_source} · {rep.n_gold} labelled of {rep.n_judged} judged "
        f"· {rep.undecided_judge} undecided verdict(s) excluded",
        "",
        "| | Gold pass | Gold fail |",
        "|---|---:|---:|",
        f"| **Judge pass** | {cm.tp} | {cm.fp} |",
        f"| **Judge fail** | {cm.fn} | {cm.tn} |",
        "",
        f"TPR {_pct(cm.tpr)} · FPR {_pct(cm.fpr)} · accuracy {_pct(cm.accuracy)} · "
        f"Cohen's κ {'—' if cm.kappa is None else f'{cm.kappa:.3f}'}",
        "",
        "| Pass-rate estimate | Point | 95% CI | n |",
        "|---|---:|---:|---:|",
    ]
    lines += [_est_row(e) for e in rep.estimates]
    if rep.truth is not None:
        lines.append(_est_row(rep.truth))
    if rep.disagreements:
        lines += [
            "",
            f"Judge disagreed with gold on: {', '.join(f'`{c}`' for c in rep.disagreements)}",
        ]
    return "\n".join(lines) + "\n"


def to_json(obj: Any) -> str:
    return json.dumps(asdict(obj), indent=2, default=str)
