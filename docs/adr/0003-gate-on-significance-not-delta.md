# 3. Gate on a significant regression, not on a delta

Status: accepted · 2026-10

## Context

The common CI gate is "fail if the candidate's score is more than X below the baseline".
On a golden set of a few dozen cases that gate is mostly noise: two flipped cases out of
sixty move a pass rate by 3.3 points. A gate that fails on noise gets disabled; a gate
that is loosened until it stops failing on noise also stops catching regressions.

## Decision

* Compare runs **on paired cases** — the same case ids in both runs — because each case
  serves as its own control and only discordant pairs carry information.
* For pass/fail metrics use the **exact McNemar test** on the discordant pairs; for
  continuous scores use a **paired bootstrap** of the mean difference. Report the delta
  with a 95% interval either way.
* **Fail only when both hold:** the drop is at least `max_drop` *and* `p < alpha`.
  * A large drop that is not significant passes with a warning that names the sample size.
  * A significant drop smaller than `max_drop` passes with a warning.
* Refuse to gate (exit 4) when there are fewer than `min_cases` paired cases or the two runs
  used different datasets (the dataset hash differs).
* Budgets — p95 latency and cost — are separate checks with their own exit code (3).

## Consequences

* A real but small regression on a small golden set can pass with a warning. That is the
  honest outcome: the data cannot distinguish it from noise. The fix is more cases, and
  the warning says so.
* Exit codes are distinct (0 pass, 1 regression, 3 budget, 4 cannot gate, 2 usage) so a
  pipeline can treat them differently.
* Testing several metrics at α = 0.05 each inflates the family-wise error rate. The gate
  checks one metric by default; the others are reported, not gated. Gating several is
  possible with a Holm correction (see the addendum below).

## Addendum, 2026-11-12: several gated metrics, with Holm's correction

The gate can now check more than one metric (`gate.metrics` in the suite, or `--metric`
repeated). A regression fails the gate if any gated metric has a drop of at least `max_drop`
and a **Holm-adjusted** p-value below `alpha`. With one metric nothing changes: the adjusted
p-value is the raw one, and the messages are as before.

Holm's step-down procedure rather than Bonferroni, because:

* it is uniformly more powerful: every metric Bonferroni rejects, Holm rejects too, and
  sometimes more;
* it gives the same guarantee, strong control of the family-wise error rate at `alpha`;
* it needs no assumption about how the metrics depend on each other, and they usually do
  (a judge score and an answer-key score on the same cases are correlated).

Each gated metric must be usable on its own (enough paired cases, comparable runs), or the
gate cannot decide (exit 4). Metrics that are not gated are still reported, with their raw
p-values.
