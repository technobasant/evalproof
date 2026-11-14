"""Reports: Markdown for a PR comment or GitHub step summary, JSON for machines."""

from __future__ import annotations

import html
import json
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from evalproof.calibration import CalibrationReport, Estimate
from evalproof.compare import Comparison, MetricComparison, dataset_change_note
from evalproof.config import GateConfig
from evalproof.gate import EXIT_CANNOT_GATE, GateDecision
from evalproof.models import CaseResult, Run
from evalproof.stats import percentile, wilson
from evalproof.trials import (
    all_trials_identical,
    case_counts,
    is_flaky,
    pooled_rate_ci,
    trial_outcome,
)

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
    return _pct(value) if mc.kind in ("binary", "rate") else f"{value:.3f}"


def _delta(mc: MetricComparison) -> str:
    if mc.kind in ("binary", "rate"):
        return f"{100 * mc.delta:+.1f} pts [{100 * mc.ci_low:+.1f}, {100 * mc.ci_high:+.1f}]"
    return f"{mc.delta:+.3f} [{mc.ci_low:+.3f}, {mc.ci_high:+.3f}]"


def _examples(
    cmp: Comparison, gated: list[str], baseline: Run, candidate: Run, show: int
) -> list[str]:
    """A ``<details>`` block for up to ``show`` cases that regressed on a gated metric: what
    each run answered and why the candidate's scorer failed it."""
    if show <= 0:
        return []
    b_by, c_by = baseline.by_case(), candidate.by_case()
    picked: list[tuple[str, str]] = []
    for name in gated:
        mc = cmp.metrics.get(name)
        for cid in mc.regressed if mc else []:
            if all(cid != seen for seen, _ in picked):
                picked.append((cid, name))
    lines: list[str] = []
    for cid, name in picked[:show]:
        # Of a case's trials, show the baseline's first pass and the candidate's first failure:
        # the pair that shows what changed. One trial per case is just that trial.
        base = next((r for r in b_by[cid] if trial_outcome(r, name) is not False), b_by[cid][0])
        cand = next((r for r in c_by[cid] if trial_outcome(r, name) is not True), c_by[cid][0])
        score = cand.scores.get(name)
        reason = cand.output.error and f"system error: {cand.output.error}"
        reason = reason or (score.reason if score and score.reason else "none recorded")
        which = f" (trial {cand.trial})" if candidate.trials > 1 else ""
        lines += [
            "",
            f"<details><summary><code>{html.escape(cid)}</code> regressed on "
            f"{html.escape(name)}{which}</summary>",
            "",
            "**Baseline output**",
            "",
            _fence(_clip(base.output.output, EXAMPLE_CHARS) or "(empty)"),
            "",
            "**Candidate output**",
            "",
            _fence(_clip(cand.output.output, EXAMPLE_CHARS) or "(empty)"),
            "",
            f"**Candidate's {html.escape(name)} reason:** "
            f"{html.escape(_clip(reason, EXAMPLE_CHARS), quote=False)}",
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
    repeated = f" (trials: {cmp.trials}, aggregate: {cmp.aggregate})" if cmp.trials > 1 else ""
    lines += [
        f"Baseline `{cmp.baseline_label}` vs candidate `{cmp.candidate_label}` on "
        f"{len(cmp.paired_ids)} paired cases{repeated}.",
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
    if decision is not None:
        for name, power in decision.power.items():
            lines += ["", f"Power on **{name}**: {power.note}"]
    for name in cmp.aggregate_ignored:
        if name not in gated:  # a gated one is refused, and the gate says so
            lines += [
                "",
                f"- ⚠️ aggregate {cmp.aggregate} needs a pass/fail metric: {name!r} is "
                "continuous and was compared on the mean",
            ]
    lines += [
        "",
        "| Run | Cases | Errors | p50 latency | p95 latency | Cost |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, st in (("baseline", cmp.baseline), ("candidate", cmp.candidate)):
        size = f"{st.cases // st.trials} × {st.trials}" if st.trials > 1 else f"{st.cases}"
        lines.append(
            f"| {label} | {size} | {st.errors} | {st.p50_latency_ms:.0f} ms | "
            f"{st.p95_latency_ms:.0f} ms | ${st.cost_usd:.4f} |"
        )
    for name, focus in cmp.metrics.items():
        label = f"**{name}**" if name in gated else name
        majority = " by majority" if focus.kind == "rate" else ""
        if focus.regressed:
            lines += [
                "",
                f"Regressed on {label}{majority} ({len(focus.regressed)}): {_ids(focus.regressed)}",
            ]
        if name in gated and focus.improved:
            lines.append(f"Improved on {label}{majority}: {len(focus.improved)} case(s)")
        if name in gated and focus.flaky_baseline:
            lines += [
                "",
                f"Flaky in baseline ({len(focus.flaky_baseline)}): {_ids(focus.flaky_baseline)}",
            ]
        if name in gated and focus.flaky_candidate:
            lines += [
                "",
                f"Flaky in candidate ({len(focus.flaky_candidate)}): {_ids(focus.flaky_candidate)}",
            ]
    if runs is not None:
        lines += _examples(cmp, gated, runs[0], runs[1], show)
    return "\n".join(lines) + "\n"


@dataclass
class MatrixEntry:
    """One candidate of a matrix comparison: its comparison and gate decision, or the reason it
    could not be compared at all (``refusal``)."""

    label: str
    path: str
    comparison: Comparison | None
    decision: GateDecision | None
    run: Run | None = None
    refusal: str | None = None


def _verdict(mc: MetricComparison, p_adj: float, gate: GateConfig) -> str:
    """The gate's quality rule for one metric of one candidate, on the matrix-adjusted p."""
    significant = p_adj < gate.alpha
    if mc.delta <= -gate.max_drop and significant:
        return "❌ regression"
    if mc.delta <= -gate.max_drop:
        return "⚠️ drop, not significant"
    if mc.delta > 0 and significant:
        return "✅ improved"
    return "✅ no regression"


def _entry_status(entry: MatrixEntry) -> str:
    if entry.decision is None:
        return "—"
    d = entry.decision
    status = CANNOT_GATE if d.exit_code == EXIT_CANNOT_GATE else STATUS_ICON[d.status]
    return f"{status} (exit {d.exit_code})"


def matrix_markdown(
    baseline_label: str,
    entries: Sequence[MatrixEntry],
    gate: GateConfig | None = None,
    *,
    show: int = 0,
    baseline: Run | None = None,
) -> str:
    """Several candidates against one baseline: one table, a row per candidate and, for each
    gated metric, ``Δ (CI) · adjusted p · verdict``; then the gate status of each candidate and,
    in a ``<details>`` block, its own comparison. The p-values are Holm-adjusted across every
    candidate and every gated metric. Without a gate the columns are every compared metric and
    the p-values are raw."""
    if gate is not None:
        gated = gate.gated
    else:
        gated = list(
            dict.fromkeys(n for e in entries if e.comparison for n in e.comparison.metrics)
        )
    p_name = "adj p" if gate is not None else "p"
    lines = [
        "## evalproof",
        "",
        f"Baseline `{baseline_label}` against {len(entries)} candidates"
        + (
            f"; p-values are Holm-adjusted across all {len(entries)} candidates and "
            f"{len(gated)} gated metric(s)."
            if gate is not None
            else "."
        ),
        "",
        "| Candidate | "
        + " | ".join(f"{n}: Δ (95% CI) · {p_name}" + (" · verdict" if gate else "") for n in gated)
        + (" | Gate |" if gate is not None else " |"),
        "|---|" + "---|" * len(gated) + ("---|" if gate is not None else ""),
    ]
    for entry in entries:
        cells = []
        for name in gated:
            mc = entry.comparison.metrics.get(name) if entry.comparison else None
            if mc is None:
                cells.append("—")
                continue
            p_shown = mc.p_value
            if (
                gate is not None
                and entry.decision is not None
                and name in entry.decision.adjusted_p
            ):
                p_shown = entry.decision.adjusted_p[name]
            cell = f"{_delta(mc)} · {p_shown:.4f}"
            if gate is not None:
                cell += f" · {_verdict(mc, p_shown, gate)}"
            cells.append(cell)
        row = f"| `{entry.label}` | " + " | ".join(cells)
        row += f" | {_entry_status(entry)} |" if gate is not None else " |"
        lines.append(row)
    for entry in entries:
        lines += ["", f"<details><summary><code>{html.escape(entry.label)}</code>"]
        lines[-1] += f" — {html.escape(_entry_status(entry))}</summary>" if gate else "</summary>"
        lines.append("")
        if entry.comparison is None or entry.run is None:
            lines += [f"- {entry.refusal}"] if entry.refusal else []
        else:
            body = comparison_markdown(
                entry.comparison,
                entry.decision,
                gate.gated if gate else None,
                runs=(baseline, entry.run) if baseline is not None else None,
                show=show,
            )
            lines.append(body.removeprefix("## evalproof\n\n").rstrip("\n"))
        lines += ["", "</details>"]
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
    repeated = run.trials > 1
    lines = ["", f"### Failed {'trials' if repeated else 'cases'} ({len(found)})", ""]
    for r, (kind, names) in found[:max_cases]:
        what = f"{kind} on {', '.join(names)}" if names else kind
        first = r.scores[names[0]] if names else None
        reason = r.output.error or (first.reason if first and first.reason else "")
        label = f"`{r.case_id}` trial {r.trial}" if repeated else f"`{r.case_id}`"
        lines.append(
            f"- {label} · {what}" + (f" · {_clip(reason, REASON_CHARS)}" if reason else "")
        )
    if len(found) > max_cases:
        lines.append(f"- … and {len(found) - max_cases} more")
    return lines


def _single_table(run: Run) -> list[str]:
    """The summary table of a run with one trial per case."""
    lines = [
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
    return lines


def _repeated_table(run: Run) -> list[str]:
    """The summary table of a run with several trials per case. Trials of one case are not
    independent, so the pass rate is pooled over trials and its interval resamples cases."""
    k = run.trials
    lines = [
        f"| Scorer | Decided | Pass rate (95% CI) | pass@{k} | pass^{k} | Mean | Within-case SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_case = run.by_case()
    for name in run.scorer_names():
        scores = [r.scores[name] for r in run.results if name in r.scores]
        decided = [s for s in scores if s.value is not None]
        counts = list(case_counts(run, name).values())
        rate = at_least_one = every = "—"
        if counts:
            passes = sum(p for p, _ in counts)
            trials = sum(d for _, d in counts)
            lo, hi = pooled_rate_ci(counts)
            rate = f"{_pct(passes / trials)} [{_pct(lo)}, {_pct(hi)}]"
            at_least_one = _pct(sum(1 for p, _ in counts if p > 0) / len(counts))
            every = _pct(sum(1 for p, d in counts if p == d) / len(counts))
        per_case = [
            [s.value for r in results if (s := r.scores.get(name)) and s.value is not None]
            for results in by_case.values()
        ]
        means = [statistics.fmean(v) for v in per_case if v]
        sds = [statistics.stdev(v) for v in per_case if len(v) > 1]
        continuous = not counts and bool(decided)
        mean = f"{statistics.fmean(means):.3f}" if means else "—"
        spread = f"{statistics.fmean(sds):.3f}" if continuous and sds else "—"
        lines.append(
            f"| {name} | {len(decided)}/{len(scores)} | {rate} | {at_least_one} | {every} | "
            f"{mean} | {spread} |"
        )
    return lines


def _flaky_cases(run: Run, max_cases: int) -> list[str]:
    """Cases with a pass and a fail among their decided trials, on any binary scorer."""
    flaky: dict[str, list[str]] = {}
    for name in run.scorer_names():
        for case_id, (passes, decided) in case_counts(run, name).items():
            if is_flaky(passes, decided):
                flaky.setdefault(case_id, []).append(f"{name} {passes}/{decided}")
    if max_cases <= 0:
        return []
    if not flaky:
        return ["", "No flaky cases: no case both passed and failed across its trials."]
    lines = ["", f"### Flaky cases ({len(flaky)})", ""]
    lines += [f"- `{cid}` · {' · '.join(found)}" for cid, found in list(flaky.items())[:max_cases]]
    if len(flaky) > max_cases:
        lines.append(f"- … and {len(flaky) - max_cases} more")
    return lines


def run_markdown(run: Run, max_cases: int = 10) -> str:
    """The run's summary table. ``max_cases`` caps the failed-cases and flaky-cases lists; 0
    leaves them out. A run with several trials per case reports pass@k, pass^k and the flaky
    cases as well."""
    variant = f" · variant `{run.info.variant}`" if run.info.variant else ""
    repeated = run.trials > 1
    counted = (
        f"{len(run.by_case())} cases × {run.trials} trials = {len(run.results)} runs"
        if repeated
        else f"{len(run.results)} cases"
    )
    lines = [
        f"## {run.info.suite} — `{run.info.system}`{variant}",
        "",
        f"{counted} · wall {run.info.wall_time_s:.1f} s · dataset `{run.info.dataset_hash}`",
        "",
    ]
    lines += _repeated_table(run) if repeated else _single_table(run)
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    lines += [
        "",
        f"Latency p50 {percentile(lat, 50):.0f} ms · p95 {percentile(lat, 95):.0f} ms · "
        f"errors {sum(1 for r in run.results if r.output.error)} · "
        f"cached {sum(1 for r in run.results if r.output.cached)}",
    ]
    if repeated:
        if all_trials_identical(run):
            lines += [
                "",
                "All trials identical: the system may be deterministic at these settings, so "
                "repeating it shows no flakiness.",
            ]
        lines += _flaky_cases(run, max_cases)
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
        *(
            [
                "",
                f"Trial 0 of {rep.trials}: the run repeats each case, and calibration reads the "
                "first trial of each.",
            ]
            if rep.trials > 1
            else []
        ),
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
