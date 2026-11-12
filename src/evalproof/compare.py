"""Paired comparison of two runs on the cases they share."""

from __future__ import annotations

from dataclasses import dataclass, field

from evalproof.models import Run
from evalproof.stats import mcnemar_exact, paired_bootstrap, percentile, wilson

# Recorded in ``RunInfo.environment``. A judge key counts only when both runs have it, so runs
# written before these keys existed still compare as they did.
JUDGE_KEYS = (
    "judge_target",
    "judge_cli",
    "judge_cli_version",
    "judge_isolation",
    "judge_model_resolved",
)
# The system under test is usually the thing being changed, so these only warn.
SYSTEM_KEYS = ("system_target", "system_cli", "system_cli_version", "system_isolation")


@dataclass
class MetricComparison:
    name: str
    kind: str  # "binary" | "continuous"
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


@dataclass
class RunStats:
    cases: int
    errors: int
    p50_latency_ms: float
    p95_latency_ms: float
    cost_usd: float
    cached: int


@dataclass
class Comparison:
    paired_ids: list[str]
    only_baseline: list[str]
    only_candidate: list[str]
    same_dataset: bool
    metrics: dict[str, MetricComparison]
    baseline: RunStats
    candidate: RunStats
    baseline_label: str = "baseline"
    candidate_label: str = "candidate"
    same_judge: bool = True
    judge_changes: list[str] = field(default_factory=list)  # "key: before -> after"
    system_changes: list[str] = field(default_factory=list)


def environment_changes(baseline: Run, candidate: Run, keys: tuple[str, ...]) -> list[str]:
    """``"key: before -> after"`` for each key both runs recorded with different values."""
    before, after = baseline.info.environment, candidate.info.environment
    return [
        f"{k}: {before[k]} -> {after[k]}"
        for k in keys
        if k in before and k in after and before[k] != after[k]
    ]


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
) -> MetricComparison | None:
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
        )

    a = [float(bs.value) for _, bs, _ in pairs if bs.value is not None]
    c = [float(cs.value) for _, _, cs in pairs if cs.value is not None]
    boot = paired_bootstrap(a, c, n_boot=n_boot, alpha=alpha, seed=seed)
    regressed = [cid for cid, bs, cs in pairs if (cs.value or 0) < (bs.value or 0)]
    improved = [cid for cid, bs, cs in pairs if (cs.value or 0) > (bs.value or 0)]
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
    )


def compare_runs(
    baseline: Run,
    candidate: Run,
    *,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
    metrics: list[str] | None = None,
) -> Comparison:
    b_ids, c_ids = set(baseline.by_id()), set(candidate.by_id())
    paired = sorted(b_ids & c_ids)
    names = metrics or [n for n in baseline.scorer_names() if n in candidate.scorer_names()]
    out: dict[str, MetricComparison] = {}
    for name in names:
        mc = compare_metric(
            name, baseline, candidate, paired, alpha=alpha, n_boot=n_boot, seed=seed
        )
        if mc is not None:
            out[name] = mc
    judge_changes = environment_changes(baseline, candidate, JUDGE_KEYS)
    return Comparison(
        paired_ids=paired,
        only_baseline=sorted(b_ids - c_ids),
        only_candidate=sorted(c_ids - b_ids),
        same_dataset=baseline.info.dataset_hash == candidate.info.dataset_hash,
        metrics=out,
        baseline=run_stats(baseline),
        candidate=run_stats(candidate),
        baseline_label=baseline.info.system,
        candidate_label=candidate.info.system,
        same_judge=not judge_changes,
        judge_changes=judge_changes,
        system_changes=environment_changes(baseline, candidate, SYSTEM_KEYS),
    )
