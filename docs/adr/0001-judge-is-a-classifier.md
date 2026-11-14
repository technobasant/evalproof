---
description: Records the decision to treat an LLM judge as a classifier with error rates and calibrate it against gold labels.
---

# 1. An LLM judge is a classifier, and its pass rate is classify-and-count

Status: accepted · 2026-10

## Context

Most LLM evaluation reports a single number: the fraction of outputs a judge model marked
as passing. That number is classify-and-count. If the judge passes bad answers at a false
positive rate FPR and passes good answers at a true positive rate TPR, then on a set whose
true pass rate is *p* the judge's rate converges to

    q = FPR + (TPR − FPR) · p

which equals *p* only when FPR = 0 and TPR = 1. Every real judge has error. The bias also
moves with *p*: when a candidate system gets worse, a judge with a non-zero FPR reports a
smaller drop than really happened. The same algebra is worked through with a seeded
simulation in [prevalence-shift](https://github.com/technobasant/prevalence-shift).

## Decision

evalproof treats the judge as an instrument to be calibrated, not an oracle:

1. A calibration sample of gold labels — human review, or a trusted deterministic check —
   measures the judge's confusion matrix, TPR, FPR and Cohen's κ.
2. The judge's pass rate is reported next to two corrected estimates:
    * **ACC** (Rogan–Gladen): `p = (q − FPR) / (TPR − FPR)`, with a bootstrap interval that
      resamples both the judged set and the gold sample.
    * **PPI** (prediction-powered inference): the judge's mean on unlabelled cases plus the
      mean residual (label − judge) on the gold sample, with a normal interval.
3. Undecided verdicts (an unparseable reply after one retry) are counted and excluded,
   never silently scored as passes.

## Consequences

* Someone has to label a sample. The default example uses 30 cases — what a team could
  review by hand per release — and the measurements show how wide the intervals are at
  that size.
* ACC can be undefined (a judge no better than chance) and needs both classes in the gold
  sample; the report says so instead of printing a number.
* PPI needs no assumption about the judge but does need the gold sample to be random.
  evalproof draws it with a seeded RNG and records the seed.
