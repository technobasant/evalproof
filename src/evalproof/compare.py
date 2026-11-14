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


def _label(run: Run) -> str:
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
        same_dataset=same_dataset,
        metrics=out,
        baseline=run_stats(baseline),
        candidate=run_stats(candidate),
        baseline_label=_label(baseline),
        candidate_label=_label(candidate),
        same_judge=not judge_changes,
        judge_changes=judge_changes,
        system_changes=environment_changes(baseline, candidate, SYSTEM_KEYS),
        comparable=comparable,
        edited=edited,
        judge_warnings=judge_warnings(baseline, candidate),
    )
