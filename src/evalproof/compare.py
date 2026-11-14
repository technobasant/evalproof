"""Paired comparison of two runs on the cases they share."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evalproof.models import CaseResult, Run
from evalproof.stats import (
    bootstrap_ci,
    mcnemar_exact,
    paired_bootstrap,
    percentile,
    sign_flip_test,
    wilson,
)
from evalproof.trials import is_flaky, trial_outcome

AGGREGATES = ("mean", "all", "any")


class ComparisonRefused(ValueError):
    """The two runs cannot be compared at all (they repeat each case a different number of
    times). ``compare`` exits 4 under ``--gate`` and 2 otherwise."""


# Recorded in ``RunInfo.environment``. A judge key counts only when both runs have it, so runs
# written before these keys existed still compare as they did.
JUDGE_KEYS = (
    "judge_target",
    "judge_cli",
    "judge_isolation",
    "judge_model_resolved",
)
# The judge's CLI version is advisory: a CLI that updates itself still answers with the model
# the run resolved, so a new version warns and does not stop the gate.
JUDGE_WARN_KEYS = ("judge_cli_version",)
# The system under test is usually the thing being changed, so these only warn.
SYSTEM_KEYS = ("system_target", "system_cli", "system_cli_version", "system_isolation")


@dataclass
class MetricComparison:
    name: str
    # "binary" (pass/fail per case), "rate" (the mean pass rate of a binary metric over a case's
    # trials) or "continuous"
    kind: str
    n_paired: int
    baseline: float
    candidate: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    test: str
    regressed: list[str] = field(default_factory=list)  # passed in baseline, failed in candidate
    improved: list[str] = field(default_factory=list)
    undecided: int = 0
    baseline_ci: tuple[float, float] | None = None
    candidate_ci: tuple[float, float] | None = None
    trials: int = 1  # trials per case in both runs
    aggregate: str | None = None  # how a case's trials became one value; None at one trial
    flaky_baseline: list[str] = field(default_factory=list)  # a pass and a fail among its trials
    flaky_candidate: list[str] = field(default_factory=list)
    # What the power estimate needs: the sample standard deviation of the paired per-case
    # differences and the sample variance of the baseline's per-case values (None below 2 cases).
    diff_sd: float | None = None
    baseline_var: float | None = None


@dataclass
class RunStats:
    cases: int  # results in the run: cases times trials
    errors: int
    p50_latency_ms: float
    p95_latency_ms: float
    cost_usd: float
    cached: int
    trials: int = 1


@dataclass
class Comparison:
    paired_ids: list[str]
    only_baseline: list[str]
    only_candidate: list[str]
    same_dataset: bool  # informational: the two dataset hashes are equal
    metrics: dict[str, MetricComparison]
    baseline: RunStats
    candidate: RunStats
    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    same_judge: bool = True
    judge_changes: list[str] = field(default_factory=list)  # "key: before -> after"
    system_changes: list[str] = field(default_factory=list)
    # False only when the datasets differ and a run predates per-case hashes, so the cases
    # that stayed the same cannot be told from the ones that changed.
    comparable: bool = True
    edited: list[str] = field(default_factory=list)  # same id, different content: not compared
    judge_warnings: list[str] = field(default_factory=list)  # advisory judge changes
    trials: int = 1  # trials per case in both runs
    aggregate: str | None = None  # None at one trial
    # Continuous metrics that ``aggregate`` all or any cannot apply to: compared on the mean, and
    # a gate on one of them refuses.
    aggregate_ignored: list[str] = field(default_factory=list)


def environment_changes(baseline: Run, candidate: Run, keys: tuple[str, ...]) -> list[str]:
    """``"key: before -> after"`` for each key both runs recorded with different values."""
    before, after = baseline.info.environment, candidate.info.environment
    return [
        f"{k}: {before[k]} -> {after[k]}"
        for k in keys
        if k in before and k in after and before[k] != after[k]
    ]


def judge_warnings(baseline: Run, candidate: Run) -> list[str]:
    """Advisory judge changes: a new CLI version. The wording says whether the resolved model
    is known to be the same, because that is what makes the change harmless."""
    before, after = baseline.info.environment, candidate.info.environment
    key = "judge_model_resolved"
    same_model = key in before and key in after and before[key] == after[key]
    out = []
    for k in JUDGE_WARN_KEYS:
        if k in before and k in after and before[k] != after[k]:
            note = "; the resolved model is unchanged" if same_model else ""
            out.append(f"the judge's CLI version changed: {before[k]} -> {after[k]}{note}")
    return out


def dataset_change_note(cmp: Comparison) -> str | None:
    """One line on the cases left out of the comparison, or None when every case was shared."""
    if not (cmp.only_baseline or cmp.only_candidate or cmp.edited):
        return None
    return (
        f"{len(cmp.only_candidate)} case(s) added, {len(cmp.only_baseline)} removed, "
        f"{len(cmp.edited)} edited: excluded from the comparison; gated on the "
        f"{len(cmp.paired_ids)} cases both runs share"
    )


def run_label(run: Run) -> str:
    """The run's system, with its variant when it has one: two variants of a suite often run the
    same target, and only the variant tells them apart."""
    return f"{run.info.system} ({run.info.variant})" if run.info.variant else run.info.system


def records_case_hashes(run: Run) -> bool:
    """True when every result carries a ``case_hash`` (written by evalproof 0.5 or newer)."""
    return bool(run.results) and all(r.case_hash is not None for r in run.results)


def run_stats(run: Run) -> RunStats:
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    cost = sum(r.output.cost_usd for r in run.results) + sum(
        s.judge_cost_usd for r in run.results for s in r.scores.values()
    )
    return RunStats(
        cases=len(run.results),
        errors=sum(1 for r in run.results if r.output.error),
        p50_latency_ms=percentile(lat, 50),
        p95_latency_ms=percentile(lat, 95),
        cost_usd=cost,
        cached=sum(1 for r in run.results if r.output.cached),
        trials=run.trials,
    )


def _spread(a: list[float], c: list[float]) -> tuple[float | None, float | None]:
    """``(sd of the paired differences, variance of the baseline values)``, both with one degree
    of freedom lost; None below two cases, where neither exists. What ``power`` needs."""
    if len(a) < 2:
        return None, None
    base = np.asarray(a, dtype=np.float64)
    diff = np.asarray(c, dtype=np.float64) - base
    return float(np.std(diff, ddof=1)), float(np.var(base, ddof=1))


def _compare_single(
    name: str,
    baseline: Run,
    candidate: Run,
    ids: list[str],
    *,
    alpha: float,
    n_boot: int,
    seed: int,
) -> MetricComparison | None:
    """Two runs with one trial per case: every number is what evalproof 0.6 gave."""
    b_by, c_by = baseline.by_id(), candidate.by_id()
    pairs = []
    undecided = 0
    for cid in ids:
        bs, cs = b_by[cid].scores.get(name), c_by[cid].scores.get(name)
        if bs is None or cs is None:
            continue
        if bs.value is None or cs.value is None:
            undecided += 1
            continue
        pairs.append((cid, bs, cs))
    if not pairs:
        return None

    binary = all(bs.passed is not None and cs.passed is not None for _, bs, cs in pairs)
    if binary:
        a = [1.0 if bs.passed else 0.0 for _, bs, _ in pairs]
        c = [1.0 if cs.passed else 0.0 for _, _, cs in pairs]
        regressed = [cid for cid, bs, cs in pairs if bs.passed and not cs.passed]
        improved = [cid for cid, bs, cs in pairs if not bs.passed and cs.passed]
        boot = paired_bootstrap(a, c, n_boot=n_boot, alpha=alpha, seed=seed)
        n = len(pairs)
        diff_sd, base_var = _spread(a, c)
        return MetricComparison(
            name=name,
            kind="binary",
            n_paired=n,
            baseline=sum(a) / n,
            candidate=sum(c) / n,
            delta=boot.delta,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            p_value=mcnemar_exact(len(regressed), len(improved)),
            test="exact McNemar",
            regressed=regressed,
            improved=improved,
            undecided=undecided,
            baseline_ci=wilson(int(sum(a)), n, alpha),
            candidate_ci=wilson(int(sum(c)), n, alpha),
            diff_sd=diff_sd,
            baseline_var=base_var,
        )

    a = [float(bs.value) for _, bs, _ in pairs if bs.value is not None]
    c = [float(cs.value) for _, _, cs in pairs if cs.value is not None]
    boot = paired_bootstrap(a, c, n_boot=n_boot, alpha=alpha, seed=seed)
    regressed = [cid for cid, bs, cs in pairs if (cs.value or 0) < (bs.value or 0)]
    improved = [cid for cid, bs, cs in pairs if (cs.value or 0) > (bs.value or 0)]
    diff_sd, base_var = _spread(a, c)
    return MetricComparison(
        name=name,
        kind="continuous",
        n_paired=len(pairs),
        baseline=sum(a) / len(a),
        candidate=sum(c) / len(c),
        delta=boot.delta,
        ci_low=boot.ci_low,
        ci_high=boot.ci_high,
        p_value=boot.p_value,
        test="paired bootstrap",
        regressed=regressed,
        improved=improved,
        undecided=undecided,
        diff_sd=diff_sd,
        baseline_var=base_var,
    )


@dataclass
class _CaseValue:
    """One case in one run, reduced over its trials for one metric."""

    value: float  # the aggregated value: a pass rate, 0 or 1, or the mean score
    passes: int = 0  # decided trials that passed (binary metrics)
    decided: int = 0  # decided trials


def _case_value(
    results: list[CaseResult], name: str, *, binary: bool, aggregate: str
) -> _CaseValue | None:
    """The case's value on ``name`` over its trials, or None when no trial was decided. A binary
    metric counts an errored trial as a failure and drops an undecided one; ``aggregate`` then
    makes the pass rate (mean), 1 only if every decided trial passed (all) or if one did (any). A
    continuous metric is the mean of the decided scores."""
    if binary:
        outcomes = [o for r in results if (o := trial_outcome(r, name)) is not None]
        if not outcomes:
            return None
        passes = sum(outcomes)
        decided = len(outcomes)
        if aggregate == "all":
            value = 1.0 if passes == decided else 0.0
        elif aggregate == "any":
            value = 1.0 if passes else 0.0
        else:
            value = passes / decided
        return _CaseValue(value, passes, decided)
    values = [s.value for r in results if (s := r.scores.get(name)) and s.value is not None]
    return _CaseValue(sum(values) / len(values), decided=len(values)) if values else None


def _compare_repeated(
    name: str,
    baseline: Run,
    candidate: Run,
    ids: list[str],
    *,
    aggregate: str,
    alpha: float,
    n_boot: int,
    seed: int,
) -> MetricComparison | None:
    """Two runs that repeat each case: the trials of a case become one value per case, and the
    cases, never the trials, are what is resampled and tested."""
    b_cases, c_cases = baseline.by_case(), candidate.by_case()
    k = baseline.trials
    in_scope = [
        cid
        for cid in ids
        if any(name in r.scores for r in b_cases[cid])
        and any(name in r.scores for r in c_cases[cid])
    ]
    scored = [
        s
        for cid in in_scope
        for results in (b_cases[cid], c_cases[cid])
        for r in results
        if (s := r.scores.get(name)) is not None and s.value is not None
    ]
    binary = bool(scored) and all(s.passed is not None for s in scored)
    used = aggregate if binary else "mean"
    pairs: list[tuple[str, _CaseValue, _CaseValue]] = []
    undecided = 0
    for cid in in_scope:
        b = _case_value(b_cases[cid], name, binary=binary, aggregate=used)
        c = _case_value(c_cases[cid], name, binary=binary, aggregate=used)
        if b is None or c is None:
            undecided += 1
        else:
            pairs.append((cid, b, c))
    if not pairs:
        return None

    n = len(pairs)
    a = [b.value for _, b, _ in pairs]
    c_vals = [c.value for _, _, c in pairs]
    boot = paired_bootstrap(a, c_vals, n_boot=n_boot, alpha=alpha, seed=seed)
    diff_sd, base_var = _spread(a, c_vals)
    common: dict[str, Any] = {
        "name": name,
        "n_paired": n,
        "baseline": sum(a) / n,
        "candidate": sum(c_vals) / n,
        "delta": boot.delta,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
        "undecided": undecided,
        "trials": k,
        "aggregate": used,
        "diff_sd": diff_sd,
        "baseline_var": base_var,
    }
    if binary and used != "mean":
        # all or any: each case passed or failed, so the test is McNemar's
        regressed = [cid for cid, b, c in pairs if b.value and not c.value]
        improved = [cid for cid, b, c in pairs if not b.value and c.value]
        return MetricComparison(
            kind="binary",
            p_value=mcnemar_exact(len(regressed), len(improved)),
            test="exact McNemar",
            regressed=regressed,
            improved=improved,
            baseline_ci=wilson(int(sum(a)), n, alpha),
            candidate_ci=wilson(int(sum(c_vals)), n, alpha),
            flaky_baseline=[cid for cid, b, _ in pairs if is_flaky(b.passes, b.decided)],
            flaky_candidate=[cid for cid, _, c in pairs if is_flaky(c.passes, c.decided)],
            **common,
        )

    # Pass rates over exactly k trials sit on the grid of 1/k, so the test can be exact for any
    # number of cases; an undecided trial moves a case off the grid.
    on_grid = binary and all(b.decided == k and c.decided == k for _, b, c in pairs)
    flip = sign_flip_test(
        np.asarray(c_vals) - np.asarray(a), grid=k if on_grid else None, seed=seed
    )
    test = "sign-flip permutation (exact)"
    if flip.method != "exact":
        test = f"sign-flip permutation (Monte Carlo, {flip.draws})"
    if binary:
        return MetricComparison(
            kind="rate",
            p_value=flip.p_value,
            test=test,
            # A case passes by majority when its rate is above one half.
            regressed=[cid for cid, b, c in pairs if b.value > 0.5 and not c.value > 0.5],
            improved=[cid for cid, b, c in pairs if c.value > 0.5 and not b.value > 0.5],
            baseline_ci=bootstrap_ci(a, n_boot=n_boot, alpha=alpha, seed=seed),
            candidate_ci=bootstrap_ci(c_vals, n_boot=n_boot, alpha=alpha, seed=seed),
            flaky_baseline=[cid for cid, b, _ in pairs if is_flaky(b.passes, b.decided)],
            flaky_candidate=[cid for cid, _, c in pairs if is_flaky(c.passes, c.decided)],
            **common,
        )
    return MetricComparison(
        kind="continuous",
        p_value=flip.p_value,
        test=test,
        regressed=[cid for cid, b, c in pairs if c.value < b.value],
        improved=[cid for cid, b, c in pairs if c.value > b.value],
        **common,
    )


def compare_metric(
    name: str,
    baseline: Run,
    candidate: Run,
    ids: list[str],
    *,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
    aggregate: str = "mean",
) -> MetricComparison | None:
    """Compare one metric on the paired cases. With one trial per case this is the paired test of
    evalproof 0.6 (and ``aggregate`` does not apply); with several, each case's trials are
    reduced to one value first (see ``_case_value``) and the cases are tested."""
    if baseline.trials == 1 and candidate.trials == 1:
        return _compare_single(
            name, baseline, candidate, ids, alpha=alpha, n_boot=n_boot, seed=seed
        )
    if baseline.trials != candidate.trials:
        raise ComparisonRefused(_trials_message(baseline, candidate))
    if aggregate not in AGGREGATES:
        raise ValueError(f"aggregate must be one of {', '.join(AGGREGATES)}, got {aggregate!r}")
    return _compare_repeated(
        name,
        baseline,
        candidate,
        ids,
        aggregate=aggregate,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )


def _trials_message(baseline: Run, candidate: Run) -> str:
    return (
        f"baseline has k={baseline.trials}, candidate has k={candidate.trials}; "
        "rerun with the same --trials"
    )


def compare_runs(
    baseline: Run,
    candidate: Run,
    *,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
    metrics: list[str] | None = None,
    aggregate: str = "mean",
) -> Comparison:
    """Compare two runs on the cases they share. ``aggregate`` (``mean``, ``all`` or ``any``) says
    how a case's trials become one value when the runs repeat each case; it is ignored at one
    trial. Raises ``ComparisonRefused`` when the runs have a different number of trials."""
    if baseline.trials != candidate.trials:
        raise ComparisonRefused(_trials_message(baseline, candidate))
    repeated = baseline.trials > 1
    b_by, c_by = baseline.by_id(), candidate.by_id()
    b_ids, c_ids = set(b_by), set(c_by)
    same_dataset = baseline.info.dataset_hash == candidate.info.dataset_hash
    comparable = same_dataset or (records_case_hashes(baseline) and records_case_hashes(candidate))
    # A case is paired when both runs scored the same content. The content hash leaves out the
    # tags, so a tag change is not an edit; run files from evalproof 0.5.0 carry only the case
    # hash, which does include them, so pairing falls back to it. Without per-case hashes the
    # dataset hash is all there is: equal means every shared case is the same case, and when it
    # differs the gate refuses (``comparable`` is False), so the old pairing is only for display.
    paired: list[str] = []
    edited: list[str] = []
    for cid in sorted(b_ids & c_ids):
        b, c = b_by[cid], c_by[cid]
        b_hash: str | None
        c_hash: str | None
        if b.content_hash is not None and c.content_hash is not None:
            b_hash, c_hash = b.content_hash, c.content_hash
        else:
            b_hash, c_hash = b.case_hash, c.case_hash
        if b_hash is not None and c_hash is not None and b_hash != c_hash:
            edited.append(cid)
        else:
            paired.append(cid)
    names = metrics or [n for n in baseline.scorer_names() if n in candidate.scorer_names()]
    out: dict[str, MetricComparison] = {}
    ignored: list[str] = []
    for name in names:
        mc = compare_metric(
            name,
            baseline,
            candidate,
            paired,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            aggregate=aggregate,
        )
        if mc is not None:
            out[name] = mc
            if repeated and mc.kind == "continuous" and aggregate != "mean":
                ignored.append(name)
    judge_changes = environment_changes(baseline, candidate, JUDGE_KEYS)
    return Comparison(
        paired_ids=paired,
        only_baseline=sorted(b_ids - c_ids),
        only_candidate=sorted(c_ids - b_ids),
        same_dataset=same_dataset,
        metrics=out,
        baseline=run_stats(baseline),
        candidate=run_stats(candidate),
        baseline_label=run_label(baseline),
        candidate_label=run_label(candidate),
        same_judge=not judge_changes,
        judge_changes=judge_changes,
        system_changes=environment_changes(baseline, candidate, SYSTEM_KEYS),
        comparable=comparable,
        edited=edited,
        judge_warnings=judge_warnings(baseline, candidate),
        trials=baseline.trials,
        aggregate=aggregate if repeated else None,
        aggregate_ignored=ignored,
    )
