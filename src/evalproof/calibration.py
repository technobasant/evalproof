"""Judge calibration: measure the judge against gold labels, then correct its pass rate.

A judge's raw pass rate is classify-and-count. If the judge passes bad answers at
rate FPR and passes good answers at rate TPR, the raw rate converges to
``q = FPR + (TPR - FPR) * p``, not to the true pass rate ``p``. Two corrections:

* **ACC** (adjusted classify-and-count, Rogan–Gladen): ``p = (q - FPR) / (TPR - FPR)``,
  with TPR and FPR estimated on the gold sample. Bootstrap interval.
* **PPI** (prediction-powered inference, Angelopoulos et al., 2023): the judge's mean on
  the unlabelled cases plus the mean label-minus-judge residual on the gold sample.
  Needs no assumption about the judge; unbiased whenever the gold sample is random.

Background: https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

from evalproof.models import Run
from evalproof.stats import FloatArray, wilson, z_for


@dataclass(frozen=True)
class Confusion:
    tp: int  # judge pass, gold pass
    fp: int  # judge pass, gold fail
    fn: int  # judge fail, gold pass
    tn: int  # judge fail, gold fail

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def tpr(self) -> float | None:
        pos = self.tp + self.fn
        return self.tp / pos if pos else None

    @property
    def fpr(self) -> float | None:
        neg = self.fp + self.tn
        return self.fp / neg if neg else None

    @property
    def accuracy(self) -> float | None:
        return (self.tp + self.tn) / self.n if self.n else None

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa: agreement beyond what the two marginals give by chance."""
        n = self.n
        if not n:
            return None
        po = (self.tp + self.tn) / n
        pe = (
            (self.tp + self.fp) * (self.tp + self.fn) + (self.fn + self.tn) * (self.fp + self.tn)
        ) / (n * n)
        return None if pe == 1 else (po - pe) / (1 - pe)


def confusion(judge: list[bool], gold: list[bool]) -> Confusion:
    if len(judge) != len(gold):
        raise ValueError("judge and gold must be paired")
    tp = sum(j and g for j, g in zip(judge, gold, strict=True))
    fp = sum(j and not g for j, g in zip(judge, gold, strict=True))
    fn = sum((not j) and g for j, g in zip(judge, gold, strict=True))
    tn = sum((not j) and (not g) for j, g in zip(judge, gold, strict=True))
    return Confusion(tp, fp, fn, tn)


def acc_point(q: float, tpr: float, fpr: float) -> float | None:
    """Rogan–Gladen. None when the judge is no better than chance (TPR <= FPR)."""
    if tpr - fpr <= 1e-12:
        return None
    return min(1.0, max(0.0, (q - fpr) / (tpr - fpr)))


@dataclass(frozen=True)
class Estimate:
    method: str
    point: float | None
    low: float | None
    high: float | None
    n: int
    note: str = ""


def raw_estimate(judge_all: list[bool], alpha: float = 0.05) -> Estimate:
    k, n = sum(judge_all), len(judge_all)
    lo, hi = wilson(k, n, alpha)
    return Estimate("raw judge pass rate", k / n if n else None, lo, hi, n)


def gold_only_estimate(gold: list[bool], alpha: float = 0.05) -> Estimate:
    k, n = sum(gold), len(gold)
    lo, hi = wilson(k, n, alpha)
    return Estimate("gold labels only", k / n if n else None, lo, hi, n)


def acc_estimate(
    judge_all: list[bool],
    judge_gold: list[bool],
    gold: list[bool],
    *,
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: int = 0,
) -> Estimate:
    """ACC point estimate with a bootstrap interval that resamples both the full judged set
    (uncertainty in q) and the gold sample (uncertainty in TPR and FPR)."""
    cm = confusion(judge_gold, gold)
    q = sum(judge_all) / len(judge_all)
    if cm.tpr is None or cm.fpr is None:
        return Estimate("ACC", None, None, None, len(judge_all), "gold sample has only one class")
    point = acc_point(q, cm.tpr, cm.fpr)
    if point is None:
        return Estimate("ACC", None, None, None, len(judge_all), "judge no better than chance")
    rng = np.random.default_rng(seed)
    ja = np.asarray(judge_all, dtype=np.float64)
    jg = np.asarray(judge_gold, dtype=bool)
    gg = np.asarray(gold, dtype=bool)
    reps: list[float] = []
    undefined = 0
    for _ in range(n_boot):
        qa = float(ja[rng.integers(0, ja.size, ja.size)].mean())
        idx = rng.integers(0, gg.size, gg.size)
        j, g = jg[idx], gg[idx]
        pos, neg = int(g.sum()), int((~g).sum())
        if pos == 0 or neg == 0:
            undefined += 1
            continue
        est = acc_point(qa, float((j & g).sum() / pos), float((j & ~g).sum() / neg))
        if est is None:
            undefined += 1
            continue
        reps.append(est)
    if not reps:
        return Estimate("ACC", point, None, None, len(judge_all), "no defined bootstrap replicates")
    lo, hi = np.quantile(np.asarray(reps), [alpha / 2, 1 - alpha / 2])
    note = f"{undefined}/{n_boot} replicates undefined" if undefined else ""
    return Estimate("ACC", point, float(lo), float(hi), len(judge_all), note)


def ppi_estimate(
    judge_unlabelled: list[bool],
    judge_labelled: list[bool],
    gold: list[bool],
    *,
    alpha: float = 0.05,
) -> Estimate:
    """Prediction-powered mean: ``mean(f_unlabelled) + mean(y - f_labelled)``, normal CI."""
    if not judge_labelled or not judge_unlabelled:
        return Estimate("PPI", None, None, None, 0, "needs labelled and unlabelled cases")
    fu: FloatArray = np.asarray(judge_unlabelled, dtype=np.float64)
    fl: FloatArray = np.asarray(judge_labelled, dtype=np.float64)
    y: FloatArray = np.asarray(gold, dtype=np.float64)
    resid = y - fl
    point = float(fu.mean() + resid.mean())
    var_u = float(fu.var(ddof=1)) if fu.size > 1 else 0.0
    var_r = float(resid.var(ddof=1)) if resid.size > 1 else 0.0
    se = (var_u / fu.size + var_r / resid.size) ** 0.5
    z = z_for(alpha)
    n = int(fu.size + fl.size)
    return Estimate(
        "PPI", min(1.0, max(0.0, point)), max(0.0, point - z * se), min(1.0, point + z * se), n
    )


@dataclass
class CalibrationReport:
    judge_scorer: str
    gold_source: str
    n_judged: int
    n_gold: int
    undecided_judge: int
    confusion: Confusion
    estimates: list[Estimate]
    truth: Estimate | None = None
    disagreements: list[str] = field(default_factory=list)
    trials: int = 1  # trials per case in the run; only trial 0 of each case is calibrated


def calibrate(
    run: Run,
    judge_scorer: str,
    *,
    labels: dict[str, bool] | None = None,
    gold_scorer: str | None = None,
    gold_sample: int | None = None,
    seed: int = 0,
    alpha: float = 0.05,
    n_boot: int = 5000,
) -> CalibrationReport:
    """Calibrate ``judge_scorer`` against human ``labels`` or a trusted ``gold_scorer``.

    With ``gold_scorer`` and ``gold_sample``, a random sample of that size stands in for a
    human-reviewed set; the remaining gold labels are used only to report the truth the
    estimates are trying to recover.

    A run with several trials per case is calibrated on trial 0 of each case: a verdict per case,
    as the labels are, and not the same case counted ``k`` times.
    """
    if (labels is None) == (gold_scorer is None):
        raise ValueError("pass exactly one of labels or gold_scorer")
    judged: dict[str, bool] = {}
    undecided = 0
    first_trial = [r for r in run.results if r.trial == 0]
    for r in first_trial:
        s = r.scores.get(judge_scorer)
        if s is None:
            continue
        if s.passed is None:
            undecided += 1
        else:
            judged[r.case_id] = s.passed
    if not judged:
        raise ValueError(f"no decided verdicts from scorer {judge_scorer!r}")

    truth_all: dict[str, bool] = {}
    if labels is not None:
        gold_ids = [cid for cid in labels if cid in judged]
        gold_map = {cid: labels[cid] for cid in gold_ids}
        source = "labels file"
    else:
        assert gold_scorer is not None
        for r in first_trial:
            s = r.scores.get(gold_scorer)
            if s is not None and s.passed is not None and r.case_id in judged:
                truth_all[r.case_id] = s.passed
        ids = sorted(truth_all)
        if gold_sample is not None and gold_sample < len(ids):
            ids = sorted(random.Random(seed).sample(ids, gold_sample))
            source = f"{gold_scorer} (random sample of {gold_sample})"
        else:
            source = gold_scorer
        gold_map = {cid: truth_all[cid] for cid in ids}

    if not gold_map:
        raise ValueError("no overlap between gold labels and judged cases")
    lab_ids = sorted(gold_map)
    unl_ids = sorted(set(judged) - set(lab_ids))
    jl = [judged[c] for c in lab_ids]
    gl = [gold_map[c] for c in lab_ids]
    ja = [judged[c] for c in sorted(judged)]
    estimates = [
        raw_estimate(ja, alpha),
        gold_only_estimate(gl, alpha),
        acc_estimate(ja, jl, gl, alpha=alpha, n_boot=n_boot, seed=seed),
        ppi_estimate([judged[c] for c in unl_ids], jl, gl, alpha=alpha),
    ]
    truth = None
    if truth_all and len(truth_all) > len(gold_map):
        k = sum(truth_all.values())
        lo, hi = wilson(k, len(truth_all), alpha)
        truth = Estimate("truth (all gold labels)", k / len(truth_all), lo, hi, len(truth_all))
    disagreements = [c for c in lab_ids if judged[c] != gold_map[c]]
    return CalibrationReport(
        judge_scorer=judge_scorer,
        gold_source=source,
        n_judged=len(judged),
        n_gold=len(gold_map),
        undecided_judge=undecided,
        confusion=confusion(jl, gl),
        estimates=estimates,
        truth=truth,
        disagreements=disagreements,
        trials=run.trials,
    )
