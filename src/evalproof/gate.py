"""The CI gate: fail on a regression that is both large enough to matter and unlikely to be noise.

Exit codes are distinct so a pipeline can tell a quality regression from a budget breach
from a setup problem:

* ``0``  pass (possibly with warnings)
* ``1``  quality regression: drop >= ``max_drop`` and p < ``alpha`` (Holm-adjusted p when
  several metrics are gated)
* ``3``  budget breach: candidate p95 latency or total cost over the limit
* ``4``  cannot gate: too few paired cases, metric missing, or the runs used different data
* ``2``  usage or configuration error (raised by the CLI, not here)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalproof.compare import Comparison
from evalproof.config import GateConfig
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


def decide(cmp: Comparison, gate: GateConfig) -> GateDecision:
    reasons: list[str] = []
    warnings: list[str] = []

    if not cmp.same_dataset:
        return GateDecision(
            "fail",
            EXIT_CANNOT_GATE,
            ["baseline and candidate were run on different datasets (dataset hash differs)"],
        )
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
                f"only {found.n_paired} paired cases for {name!r}; the gate needs {gate.min_cases}"
            )
    if unusable:
        return GateDecision("fail", EXIT_CANNOT_GATE, unusable)

    # One metric is gated on its own p-value, exactly as before. Several are corrected for
    # testing the whole set, so adding a metric to the gate cannot make a false failure likelier.
    many = len(names) > 1
    raw = [cmp.metrics[name].p_value for name in names]
    adjusted = holm(raw) if many else raw

    exit_code = EXIT_PASS
    for name, p_adj in zip(names, adjusted, strict=True):
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
    if cmp.only_baseline or cmp.only_candidate:
        warnings.append(
            f"{len(cmp.only_baseline)} case(s) only in baseline, "
            f"{len(cmp.only_candidate)} only in candidate — excluded from the comparison"
        )

    status = "fail" if exit_code else ("warn" if warnings else "pass")
    return GateDecision(
        status,
        exit_code,
        reasons,
        warnings,
        dict(zip(names, adjusted, strict=True)) if many else {},
    )
