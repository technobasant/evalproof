"""Reports: Markdown for a PR comment or GitHub step summary, JSON for machines."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from evalproof.calibration import CalibrationReport, Estimate
from evalproof.compare import Comparison, MetricComparison, dataset_change_note
from evalproof.gate import EXIT_CANNOT_GATE, GateDecision
from evalproof.models import CaseResult, Run
from evalproof.stats import percentile, wilson

STATUS_ICON = {"pass": "✅ pass", "warn": "⚠️ pass with warnings", "fail": "❌ fail"}
CANNOT_GATE = "⚠️ cannot gate"  # exit 4: nothing was decided, which is not a quality failure

REGRESSED_SHOWN = 10  # ids listed per metric
REASON_CHARS = 120  # a failed case's reason, in the run report
EXAMPLE_CHARS = 400  # a regressed case's output, in a <details> block


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _ids(ids: list[str], limit: int = REGRESSED_SHOWN) -> str:
    shown = ", ".join(f"`{c}`" for c in ids[:limit])
    return shown + (f" and {len(ids) - limit} more" if len(ids) > limit else "")


def _fence(text: str) -> str:
    """``text`` in a code fence long enough that nothing inside can close it."""
    ticks = "```"
    while ticks in text:
        ticks += "`"
    return f"{ticks}\n{text}\n{ticks}"


def _pct(x: float | None) -> str:
    return "—" if x is None else f"{100 * x:.1f}%"


def _rate_or_mean(mc: MetricComparison, value: float) -> str:
    return _pct(value) if mc.kind == "binary" else f"{value:.3f}"


def _delta(mc: MetricComparison) -> str:
    if mc.kind == "binary":
        return f"{100 * mc.delta:+.1f} pts [{100 * mc.ci_low:+.1f}, {100 * mc.ci_high:+.1f}]"
    return f"{mc.delta:+.3f} [{mc.ci_low:+.3f}, {mc.ci_high:+.3f}]"


def _examples(
    cmp: Comparison, gated: list[str], baseline: Run, candidate: Run, show: int
) -> list[str]:
    """A ``<details>`` block for up to ``show`` cases that regressed on a gated metric: what
    each run answered and why the candidate's scorer failed it."""
    if show <= 0:
        return []
    b_by, c_by = baseline.by_id(), candidate.by_id()
    picked: list[tuple[str, str]] = []
    for name in gated:
        mc = cmp.metrics.get(name)
        for cid in mc.regressed if mc else []:
            if all(cid != seen for seen, _ in picked):
                picked.append((cid, name))
    lines: list[str] = []
    for cid, name in picked[:show]:
        base, cand = b_by[cid], c_by[cid]
        score = cand.scores.get(name)
        reason = cand.output.error and f"system error: {cand.output.error}"
        reason = reason or (score.reason if score and score.reason else "none recorded")
        lines += [
            "",
            f"<details><summary><code>{cid}</code> regressed on {name}</summary>",
            "",
            "**Baseline output**",
            "",
            _fence(_clip(base.output.output, EXAMPLE_CHARS) or "(empty)"),
            "",
            "**Candidate output**",
            "",
            _fence(_clip(cand.output.output, EXAMPLE_CHARS) or "(empty)"),
            "",
            f"**Candidate's {name} reason:** {_clip(reason, EXAMPLE_CHARS)}",
            "",
            "</details>",
        ]
    return lines


def comparison_markdown(
    cmp: Comparison,
    decision: GateDecision | None = None,
    gate_metric: str | Sequence[str] | None = None,
    *,
    runs: tuple[Run, Run] | None = None,
    show: int = 0,
) -> str:
    """``runs`` is ``(baseline, candidate)``; with it, ``show`` regressed cases of the gated
    metrics get a ``<details>`` block holding both outputs."""
    gated = [gate_metric] if isinstance(gate_metric, str) else list(gate_metric or [])
    # With several gated metrics the decision carries Holm-adjusted p-values; show them.
    adjusted = decision.adjusted_p if decision is not None and len(gated) > 1 else {}
    lines = ["## evalproof", ""]
    if decision is not None:
        status = (
            CANNOT_GATE if decision.exit_code == EXIT_CANNOT_GATE else STATUS_ICON[decision.status]
        )
        lines += [f"**Gate: {status}** (exit {decision.exit_code})", ""]
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
    for name, focus in cmp.metrics.items():
        label = f"**{name}**" if name in gated else name
        if focus.regressed:
            lines += ["", f"Regressed on {label} ({len(focus.regressed)}): {_ids(focus.regressed)}"]
        if name in gated and focus.improved:
            lines.append(f"Improved on {label}: {len(focus.improved)} case(s)")
    if runs is not None:
        lines += _examples(cmp, gated, runs[0], runs[1], show)
    return "\n".join(lines) + "\n"


def _failed_cases(run: Run, max_cases: int) -> list[str]:
    """The ``### Failed cases`` section: cases that errored, failed a scorer or were undecided."""

    def problem(r: CaseResult) -> tuple[str, list[str]] | None:
        failed = [n for n, s in r.scores.items() if s.passed is False]
        undecided = [n for n, s in r.scores.items() if s.value is None]
        if r.output.error:
            return "errored", failed + undecided
        if failed:
            return "failed", failed
        if undecided:
            return "undecided", undecided
        return None

    found = [(r, p) for r in run.results if (p := problem(r)) is not None]
    if not found or max_cases <= 0:
        return []
    lines = ["", f"### Failed cases ({len(found)})", ""]
    for r, (kind, names) in found[:max_cases]:
        what = f"{kind} on {', '.join(names)}" if names else kind
        first = r.scores[names[0]] if names else None
        reason = r.output.error or (first.reason if first and first.reason else "")
        lines.append(
            f"- `{r.case_id}` · {what}" + (f" · {_clip(reason, REASON_CHARS)}" if reason else "")
        )
    if len(found) > max_cases:
        lines.append(f"- … and {len(found) - max_cases} more")
    return lines


def run_markdown(run: Run, max_cases: int = 10) -> str:
    """The run's summary table. ``max_cases`` caps the failed-cases list; 0 leaves it out."""
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
    lines += _failed_cases(run, max_cases)
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
