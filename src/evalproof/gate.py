"""The CI gate: fail on a regression that is both large enough to matter and unlikely to be noise.

Exit codes are distinct so a pipeline can tell a quality regression from a budget breach
from a setup problem:

* ``0``  pass (possibly with warnings)
* ``1``  quality regression: drop >= ``max_drop`` and p < ``alpha`` (Holm-adjusted p when
  several metrics are gated); or more than ``max_new_failures`` cases newly failing; or an
  absolute check failed: a pass rate below ``min_pass_rate``, a ``must_pass`` case that did
  not pass (in every trial, when the runs repeat each case)
* ``3``  budget breach: candidate p95 latency or total cost over the limit
* ``4``  cannot gate: too few paired cases, metric missing, the dataset changed and a run predates
  per-case hashes, ``min_pass_rate`` on a metric with only continuous scores, ``aggregate`` all
  or any on a continuous metric, the runs repeat each case a different number of times (refused
  before this module), or the judge's target, CLI, isolation or resolved model changed (unless
  the gate sets ``allow_judge_change``). A new judge CLI version only warns: it is advisory.
* ``2``  usage or configuration error (raised by the CLI, not here)

Cases added, removed or edited between the runs are left out of the comparison, with a warning;
the gate decides on the cases both runs share.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalproof.compare import Comparison, dataset_change_note
from evalproof.config import GateConfig
from evalproof.models import Run
from evalproof.power import Power, estimate_power
from evalproof.stats import holm

EXIT_PASS = 0
EXIT_REGRESSION = 1
EXIT_USAGE = 2
EXIT_BUDGET = 3
EXIT_CANNOT_GATE = 4


@dataclass
class GateDecision:
    status: str  # "pass" | "warn" | "fail"
    exit_code: int
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Holm-adjusted p-value per gated metric; empty when the gate checks one metric.
    adjusted_p: dict[str, float] = field(default_factory=dict)
    # How big a drop the suite can detect, per gated metric. Informational: it never changes the
    # status, the warnings or the exit code.
    power: dict[str, Power] = field(default_factory=dict)


SHOW_IDS = 10


def _ids(ids: list[str]) -> str:
    shown = ", ".join(f"`{i}`" for i in ids[:SHOW_IDS])
    return shown + (f" and {len(ids) - SHOW_IDS} more" if len(ids) > SHOW_IDS else "")


def pass_counts(run: Run, name: str) -> tuple[int, int]:
    """``(passed, counted)`` for one scorer: every case with a decided binary score, and every
    errored case, which counts as a failure whatever the scorer made of its empty output."""
    passed = counted = 0
    for r in run.results:
        score = r.scores.get(name)
        if r.output.error:
            counted += 1
        elif score is not None and score.passed is not None:
            counted += 1
            passed += bool(score.passed)
    return passed, counted


def _only_continuous(run: Run, name: str) -> bool:
    scores = [r.scores[name] for r in run.results if name in r.scores]
    return any(s.value is not None for s in scores) and all(s.passed is None for s in scores)


def continuous_floors(run: Run, gate: GateConfig) -> list[str]:
    """``min_pass_rate`` metrics that cannot have a pass rate: the run has scores for them, but
    only continuous ones (no pass/fail). A setup problem, not a regression."""
    bad: list[str] = []
    for name in gate.pass_rate_floors:
        if _only_continuous(run, name):
            bad.append(
                f"min_pass_rate needs a pass/fail metric, and {name!r} has only continuous "
                "scores: gate a pass/fail scorer (a retrieval scorer takes params.pass_threshold), "
                "or drop it from min_pass_rate"
            )
    return bad


def absolute_checks(run: Run, gate: GateConfig) -> tuple[list[str], list[str]]:
    """The checks that need no baseline, as ``(failures, warnings)``.

    ``min_pass_rate``: the run's pass rate on each metric it names. ``must_pass``: a case with
    one of the tags must pass every gated metric; erroring or being undecided is failing.
    Used by ``evalproof run --gate`` on its own run and by ``decide`` on the candidate. A
    metric ``continuous_floors`` names is skipped here; the callers refuse to gate on it."""
    failures: list[str] = []
    warnings: list[str] = []
    skip = {name for name in gate.pass_rate_floors if _only_continuous(run, name)}
    for name, floor in gate.pass_rate_floors.items():
        if name in skip:
            continue
        passed, counted = pass_counts(run, name)
        if not counted:
            failures.append(
                f"{name}: no case has a decided pass/fail score, so the pass rate is unknown"
            )
            continue
        rate = passed / counted
        if rate < floor:
            failures.append(
                f"{name}: pass rate {rate:.3f} below the minimum {floor:.3f} "
                f"({passed}/{counted} {'trials' if run.trials > 1 else 'cases'})"
            )
    if gate.must_pass:
        wanted = set(gate.must_pass)
        carried = set().union(*(set(r.tags) for r in run.results))
        for tag in gate.must_pass:
            if tag not in carried:
                warnings.append(f"no case carries the must_pass tag {tag!r}")
        bad: list[str] = []
        binary_seen = False
        for r in run.results:
            if not wanted & set(r.tags):
                continue
            verdicts = [r.scores.get(name) for name in gate.gated]
            binary_seen = binary_seen or any(
                s is not None and s.passed is not None for s in verdicts
            )
            # A failed pass/fail verdict, an undecided score or a missing one fails the case. A
            # continuous score with a value has no pass/fail of its own, so it neither passes nor
            # fails a must-pass case.
            # Every trial of a case is checked, so a case that fails one of its trials fails.
            if (
                r.output.error
                or any(s is None or s.value is None or s.passed is False for s in verdicts)
            ) and r.case_id not in bad:
                bad.append(r.case_id)
        if carried & wanted and not binary_seen:
            warnings.append(
                "must_pass needs a pass/fail gated metric; the gated metrics are continuous, "
                "so must_pass checked only for errors and undecided scores"
            )
        if bad:
            where = "in at least one trial " if run.trials > 1 else ""
            failures.append(
                f"{len(bad)} must-pass case(s) failed, errored or were undecided {where}on a "
                f"gated metric: {_ids(bad)}"
            )
    return failures, warnings


def decide(
    cmp: Comparison,
    gate: GateConfig,
    candidate: Run | None = None,
    *,
    adjusted: dict[str, float] | None = None,
    candidates: int = 1,
) -> GateDecision:
    """Decide the gate. ``candidate`` is the candidate run; the absolute checks
    (``min_pass_rate``, ``must_pass``) need its scores, so a gate that sets one cannot be
    decided without it.

    A matrix comparison decides each candidate on its own but corrects the p-values for the whole
    set at once: it passes the Holm-adjusted p per gated metric as ``adjusted`` and the number of
    ``candidates``, which the power estimate divides alpha by as well."""
    reasons: list[str] = []
    warnings: list[str] = []

    refused: list[str] = []
    if not cmp.comparable:
        refused.append(
            "the dataset changed between the runs and one of them predates per-case hashes, so "
            "the cases that stayed the same cannot be told from the ones that changed: run "
            "`evalproof rehash BASELINE SUITE` (no model calls; it needs the cases the baseline "
            "scored unchanged), or re-run the baseline with evalproof 0.5 or newer"
        )
    if not cmp.same_judge:
        changes = "; ".join(cmp.judge_changes) or "unrecorded"
        if gate.allow_judge_change:
            warnings.append(f"the judge changed between the runs (allowed): {changes}")
        else:
            refused.append(
                f"the judge changed between the runs, so the scores are not comparable: {changes}"
                " (pass --allow-judge-change or set gate.allow_judge_change to accept it)"
            )
    if gate.has_absolute_checks and candidate is None:
        refused.append("min_pass_rate and must_pass need the candidate run, which was not given")
    elif candidate is not None:
        refused.extend(continuous_floors(candidate, gate))
    refused.extend(
        f"aggregate {cmp.aggregate} needs a pass/fail metric, and {name!r} has only continuous "
        "scores: use aggregate mean for it (or gate a pass/fail scorer)"
        for name in gate.gated
        if name in cmp.aggregate_ignored
    )
    if refused:
        return GateDecision("fail", EXIT_CANNOT_GATE, refused)
    warnings.extend(cmp.judge_warnings)
    if cmp.system_changes:
        warnings.append(f"the system under test changed: {'; '.join(cmp.system_changes)}")
    names = gate.gated
    if not names:  # a config built without validation must not pass by gating nothing
        return GateDecision("fail", EXIT_CANNOT_GATE, ["no metric to gate"])
    unusable: list[str] = []
    for name in names:
        found = cmp.metrics.get(name)
        if found is None:
            unusable.append(f"metric {name!r} has no paired scores")
        elif found.n_paired < gate.min_cases:
            unusable.append(
                f"only {found.n_paired} paired cases for {name!r}; the gate needs "
                f"{gate.min_cases} (add cases to the dataset, or lower gate.min_cases)"
            )
    if unusable:
        return GateDecision("fail", EXIT_CANNOT_GATE, unusable)

    # One metric is gated on its own p-value, exactly as before. Several are corrected for
    # testing the whole set, so adding a metric to the gate cannot make a false failure likelier.
    many = len(names) > 1 or adjusted is not None
    if adjusted is not None:
        adjusted_ps = [adjusted[name] for name in names]
    else:
        raw = [cmp.metrics[name].p_value for name in names]
        adjusted_ps = holm(raw) if many else raw

    exit_code = EXIT_PASS
    for name, p_adj in zip(names, adjusted_ps, strict=True):
        mc = cmp.metrics[name]
        if mc.undecided:
            warnings.append(f"{mc.undecided} paired case(s) had an undecided {name!r} verdict")

        drop_big = mc.delta <= -gate.max_drop
        significant = p_adj < gate.alpha
        if drop_big and significant:
            p_text = f"{mc.test} p={mc.p_value:.4f}"
            if many:
                p_text += f", Holm-adjusted p={p_adj:.4f}"
            reasons.append(
                f"{name}: {mc.baseline:.3f} → {mc.candidate:.3f} (Δ {mc.delta:+.3f}, "
                f"95% CI [{mc.ci_low:+.3f}, {mc.ci_high:+.3f}], {p_text}) — "
                f"a drop beyond {gate.max_drop:.3f} at α={gate.alpha}"
            )
            exit_code = EXIT_REGRESSION
        elif drop_big:
            p_text = f"p={mc.p_value:.3f}"
            if many:
                p_text = f"Holm-adjusted p={p_adj:.3f} (raw p={mc.p_value:.3f})"
            warnings.append(
                f"{name} dropped {mc.delta:+.3f}, beyond the {gate.max_drop:.3f} threshold, but "
                f"{p_text} ≥ α={gate.alpha}: not distinguishable from noise "
                f"at n={mc.n_paired}. "
                "Add cases before trusting either direction."
            )
        elif mc.delta < 0 and significant:
            p_text = f"p={mc.p_value:.4f}"
            if many:
                p_text = f"Holm-adjusted p={p_adj:.4f}, raw p={mc.p_value:.4f}"
            warnings.append(
                f"{name} dropped {mc.delta:+.3f} (significant, {p_text}) but within "
                f"the {gate.max_drop:.3f} tolerance"
            )

    if gate.max_new_failures is not None:
        for name in names:
            mc = cmp.metrics[name]
            if mc.kind == "continuous":
                warnings.append(
                    f"max_new_failures counts pass/fail cases, and {name!r} is continuous, so "
                    "it was not checked on it"
                )
            elif len(mc.regressed) > gate.max_new_failures:
                # A metric averaged over trials has no single failure per case: it counts the
                # cases that passed by majority in the baseline and no longer do.
                how = " by majority of their trials" if mc.kind == "rate" else ""
                reasons.append(
                    f"{name}: {len(mc.regressed)} case(s) newly failing{how}, more than the "
                    f"{gate.max_new_failures} allowed: {_ids(mc.regressed)}"
                )
                exit_code = EXIT_REGRESSION
    if candidate is not None and gate.has_absolute_checks:
        failures, notes = absolute_checks(candidate, gate)
        reasons.extend(failures)
        warnings.extend(notes)
        if failures:
            exit_code = EXIT_REGRESSION

    if (
        gate.max_p95_latency_ms is not None
        and cmp.candidate.p95_latency_ms > gate.max_p95_latency_ms
    ):
        reasons.append(
            f"p95 latency {cmp.candidate.p95_latency_ms:.0f} ms exceeds "
            f"{gate.max_p95_latency_ms:.0f} ms"
        )
        exit_code = exit_code or EXIT_BUDGET
    if gate.max_cost_usd is not None and cmp.candidate.cost_usd > gate.max_cost_usd:
        reasons.append(f"cost ${cmp.candidate.cost_usd:.4f} exceeds ${gate.max_cost_usd:.4f}")
        exit_code = exit_code or EXIT_BUDGET

    if cmp.candidate.errors > cmp.baseline.errors:
        warnings.append(f"system errors rose from {cmp.baseline.errors} to {cmp.candidate.errors}")
    note = dataset_change_note(cmp)
    if note:
        warnings.append(note)

    # The estimate shares alpha among the gated metrics, and among the candidates of a matrix.
    alpha_used = gate.alpha / (candidates * len(names))
    power = {name: estimate_power(cmp.metrics[name], gate.max_drop, alpha_used) for name in names}
    status = "fail" if exit_code else ("warn" if warnings else "pass")
    return GateDecision(
        status,
        exit_code,
        reasons,
        warnings,
        dict(zip(names, adjusted_ps, strict=True)) if many else {},
        power,
    )
