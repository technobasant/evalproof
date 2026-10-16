# Measurements

Every figure here comes from real runs of [`examples/support-qa`](examples/support-qa) on
one machine. The run files are committed in `examples/support-qa/runs/`, and
`uv run python examples/support-qa/measure.py` recomputes the calibration, comparison and
latency tables from them without a model.

## Setup

| | |
|---|---|
| Machine | Apple M5, 24 GB RAM, macOS 27.0.1 (arm64) |
| Ollama | 0.23.2 |
| System under test | `qwen3:1.7b`, digest `8f68893c685c` — temperature 0, seed 0, `think: false`, max 200 tokens |
| Judge | `qwen3:8b`, digest `500a1f067a9f` — temperature 0, seed 0, `think: false`, max 256 tokens |
| Reasoning judge (section 8) | the same model with `think: true` and max 4,096 tokens, in [`suite.reasoning-judge.yaml`](examples/support-qa/suite.reasoning-judge.yaml) |
| evalproof / Python | 0.1.0 / 3.13.15 |
| Golden set | 83 questions about a fictional product, dataset hash `c6f08ac201023363`; 9 are not answerable from the facts |
| "Gold" | a per-question regex answer key, written before any model output was seen and not changed after. It stands in for human review; it is not human labelling. |
| Change under test | grounded system prompt → "friendly, always give a complete answer" prompt; same model |

## 1. The two runs

| | Baseline (grounded prompt) | Candidate (friendly prompt) |
|---|---:|---:|
| Answer-key pass rate | 81.9% [72.3, 88.7] | 75.9% [65.7, 83.8] |
| Judge pass rate | 77.1% [67.0, 84.8] | 72.3% [61.8, 80.8] |
| System latency p50 / p95 | 624 / 1,469 ms | 2,243 / 4,422 ms |
| Mean output length | 19 tokens | 77 tokens |
| Run wall time (system + judge, sequential) | 363 s | 588 s |
| Re-run with a warm cache | 0.05 s, results identical to the cold run | — |

Intervals are 95% Wilson. Wall times are for 83 cases with one system call and one judge
call each, no concurrency.

## 2. How good is the judge? Measured against the full answer key

| | Baseline | Candidate |
|---|---:|---:|
| Judge pass · key pass (TP) | 63 | 58 |
| Judge pass · key fail (FP) | 1 | 2 |
| Judge fail · key pass (FN) | 5 | 5 |
| Judge fail · key fail (TN) | 14 | 18 |
| TPR / FPR | 92.6% / 6.7% | 92.1% / 10.0% |
| Cohen's κ | 0.779 | 0.781 |

## 3. Correcting the judge's rate from a 30-case gold sample

A team will not hand-label all 83 cases every release. The suite calibrates against a
random 30. Over 200 different random samples of 30 (seeds 0–199), against the answer key on
all 83 as the truth:

| Baseline (truth 81.9%) | Mean point | Mean \|error\| | 95% CI covers truth | Mean CI width |
|---|---:|---:|---:|---:|
| Raw judge rate (all 83) | 77.1% | 4.8 pts | 100% | 17.8 pts |
| Gold sample alone (30) | 82.8% | 4.1 pts | 100% | 26.2 pts |
| ACC (judge on 83, error rates from 30) | 81.5% | 3.8 pts | 100% | 34.1 pts |
| PPI (judge on 53, residuals from 30) | 81.2% | 4.7 pts | 98.5% | 28.6 pts |

| Candidate (truth 75.9%) | Mean point | Mean \|error\| | 95% CI covers truth | Mean CI width |
|---|---:|---:|---:|---:|
| Raw judge rate | 72.3% | 3.6 pts | 100% | 18.9 pts |
| Gold sample alone | 76.1% | 4.3 pts | 99.0% | 29.1 pts |
| ACC | 75.3% | 4.0 pts | 100% | 41.0 pts |
| PPI | 75.4% | 4.7 pts | 98.5% | 31.6 pts |

**What this shows.** The raw judge rate sits 3.6–4.8 points below the key, and both
corrections remove that offset on average. At this size it buys little: the offset is
smaller than the raw rate's own interval, which covered the truth every time, while the
corrected intervals are about twice as wide. Calibration earns its keep when the suite is
large enough that the raw interval is narrow and the bias dominates — the regime the
[prevalence-shift](https://github.com/technobasant/prevalence-shift) simulation shows,
where the raw interval covered the truth 0% of the time at 5,000 reports per period — or
when the judge is worse than this one.

## 4. The gold was wrong more often than the judge

The judge and the answer key disagreed on 13 cases across the two runs. Each was read by
hand:

| Case | Run | Key | Judge | Who was right | Why |
|---|---|---|---|---|---|
| q76 lifetime licence | both | pass | fail | **judge** | the answer invents "no lifetime licence"; the key's abstention regex matched "not … provide" |
| q78 HIPAA | both | pass | fail | **judge** | the answer asserts "not HIPAA compliant"; the key matched "does not mention" |
| q77 uptime SLA | baseline | pass | fail | **judge** | the answer invents 99.9% and 99.99% SLAs; the key matched "no SLA … specified" |
| q28 which plans have trials | candidate | pass | fail | **judge** | the answer adds a trial to the Free plan; the key only checked that Plus and Team appear |
| q31 cancelling monthly | candidate | pass | fail | **judge** | the answer says service stops immediately; the key matched "until the end" in a later bullet |
| q56 two-factor methods | baseline | pass | fail | **key** | correct answer plus a true extra fact (no SMS); the judge penalised the extra fact |
| q65 integrations | baseline | pass | fail | **key** | correct answer plus a true extra fact; same failure |
| q32 payment methods | candidate | pass | fail | **key** | correct answer plus a true extra fact; same failure |
| q05 Enterprise pricing | candidate | fail | pass | **key** | the answer quotes Team's prices for Enterprise; the judge saw "custom pricing" in a bullet and passed it |
| q66 Microsoft Teams | both | fail | pass | ambiguous | "Yes, Quillfox offers an integration…" then "Microsoft Teams is not supported" |

**The key was wrong 7 times, the judge 4 times, 2 are ambiguous.** The key's errors are
structural: a regex can tell that the right fact is present but not that a wrong one is
absent, and it cannot tell an abstention from an invention phrased as one. Correcting the
key on these cases puts the baseline at 78.3% and the candidate at 71.1% — within about a
point of the raw judge. Most of the "bias" that section 3 corrected was the gold's, not the
judge's. Calibration is only as good as its labels, and the labels need the same
scrutiny as the judge. The judge's own failure is consistent too: three of its four
errors penalise correct extra detail the rubric explicitly allows.

## 5. Did the prompt change regress?

`evalproof compare … --gate` on the 83 paired cases:

| Metric | Baseline | Candidate | Δ (95% CI) | Test | p |
|---|---:|---:|---:|---|---:|
| answer_key | 81.9% | 75.9% | −6.0 pts [−14.5, +1.2] | exact McNemar | 0.227 |
| **judge** (gated) | 77.1% | 72.3% | −4.8 pts [−13.3, +3.6] | exact McNemar | 0.388 |

The judge saw 8 cases get worse and 4 get better. **Gate: pass with a warning (exit 0)** —
the drop is beyond the 3-point tolerance but p = 0.39 at n = 83, so the data cannot tell
it from noise. A delta-only gate would have failed the change on the same numbers.

The regression that *is* measurable is latency: p95 went from 1,469 ms to 4,422 ms. With
`--max-p95-ms 2500` the same command fails with **exit 3** (budget breach) and still
prints the quality warning.

## 6. Pairwise judging: position and length bias

`evalproof pairwise` asked the judge to pick the better answer for each of the 83 pairs,
in both orders (166 calls, 776 s):

* **35 of 83 verdicts (42.2%) flipped when the order was swapped.** In 32 of those the
  answer shown first won both times. evalproof reports them as ties.
* Of the 48 consistent verdicts, the judge preferred the candidate 42 times and the
  baseline 6 — although the pointwise judge and the answer key both rate the candidate
  lower. In 31 of the 42, both answers were correct by the key; on those cases the
  candidate's answers averaged 383 characters against the baseline's 78. In 9, the
  candidate's answer was wrong by the key.

An 8B pairwise judge without the order swap would have reported a decisive win for the
worse prompt. Position bias is cancelled by the swap; length bias is not, and is why
evalproof gates on pointwise verdicts and reports pairwise preference as a diagnostic.

## 7. Determinism and the cache

* 20 cases re-run with the cache disabled: **20/20 identical system outputs, 20/20
  identical verdicts, 20/20 identical judge reasons** (temperature 0, seed 0, Ollama 0.23.2).
* Re-running the full baseline with a warm cache took 0.05 s instead of 363 s and produced
  results identical to the cold run, apart from the per-case `cached` flag.
* Three days later, the full baseline judged again with the judge's cache entries removed
  (system answers still cached) gave **83/83 identical verdicts** in 201 s
  ([`baseline.think-off-rerun.json`](examples/support-qa/runs/baseline.think-off-rerun.json)).

## 8. A reasoning judge

The same `qwen3:8b` judge with thinking switched on (issue #3), on the same cached answers,
so the only change is the judge. `measure.py` prints every table below from the committed
runs.

| | Thinking off | Thinking on |
|---|---:|---:|
| Judge time, 83 baseline verdicts (answers cached) | 201 s (2.4 s each) | 1,338 s (16.1 s each) |
| Judge time, 83 candidate verdicts | — | 1,188 s |
| Judge tokens per verdict, baseline / candidate | 306 / 369 | 590 / 607 |
| Undecided verdicts | 0 | 0 |
| Verdicts that changed, baseline / candidate | — | 7 / 6 |

**Against the answer key, thinking looks worse**: κ falls from 0.779 to 0.676 on the
baseline and from 0.781 to 0.727 on the candidate. That comparison is unfair, because
section 4 showed the key is often the one that is wrong. So every case where the key and the
two judges did not all agree (22, the 13 from section 4 and 9 that only the thinking judge
disputes) was read case by case against the rubric and the product facts. The 9 new ones:

| Case | Run | Key | Off | On | Right | Why |
|---|---|---|---|---|---|---|
| q28 which plans have trials | baseline | pass | pass | fail | **on** | the answer gives the Free plan a trial, the same invention section 4 found in the candidate |
| q07 annual saving on Plus | candidate | pass | pass | fail | **on** | the answer states "$60 per year" before working its way to $12 |
| q42 Linux desktop app | baseline | fail | fail | pass | **on** | "Yes, there is no Linux desktop app; the web app works on Linux": the facts are right; the key rejects any answer that starts with "Yes" |
| q42 Linux desktop app | candidate | fail | fail | pass | **on** | the same facts; the off judge's reason contradicts itself |
| q83 Plus in euros | baseline | pass | pass | fail | **off** | the answer gives the dollar price and says the euro price is not covered; correct extra detail, not an invented answer |
| q21 SSO, q22 admin console | both | pass | pass | fail | **off** | the answer names Team. The reference adds Enterprise, which the product facts do not say; the thinking judge followed the reference |

Against those audited labels (the key with every reading applied, q66 left out as ambiguous):

| | Thinking off | Thinking on |
|---|---:|---:|
| Baseline TPR / FPR | 95.4% / 5.9% | 95.4% / **0.0%** |
| Baseline Cohen's κ | 0.858 | **0.895** |
| Candidate TPR / FPR | 96.6% / 8.7% | 96.6% / **0.0%** |
| Candidate Cohen's κ | 0.879 | **0.941** |
| Right on the 20 decided contested cases | 12 | **15** (the answer key: 9) |

**What this shows.** Thinking made the judge better where it matters most for a gate: it
passed no wrong answer in either run, against three for the judge without it, and it fixed
all three cases where that judge penalised correct extra detail. It costs 6.7 times the judge
time and about twice the tokens. It also reads the reference more literally, which cuts both
ways: it caught an answer that states a wrong number before the right one, and it failed two
correct answers because the golden set's reference claims more than the product facts do.
That is a defect in two reference answers, now visible because a stricter judge tripped on it.

The gate's decision did not change. With thinking on, the judge saw 6 cases get worse and 1
get better (−6.0 points, 95% CI [−12.0, +0.0], exact McNemar p = 0.125) instead of 8 and 4
(p = 0.388): cleaner evidence of the same small regression, still not distinguishable from
noise at n = 83, so the gate still passes with a warning.

The readings in sections 4 and 8 are careful case-by-case reads, not an independent human
review; issue #4 tracks calibrating against human-reviewed labels.

## Limits

* One machine, one pair of small local models, one fictional domain, 83 cases. The numbers
  describe this setup; the method is what transfers.
* The answer key is a stand-in for human review, and section 4 shows it is wrong in
  predictable ways. A real deployment should calibrate against human labels.
* Ollama's determinism with a fixed seed was checked on 20 cases on this backend only.
* Section 8 measures a reasoning judge on one model and one domain. Its literal reading
  helped here and hurt where the reference itself was wrong; that balance will differ
  elsewhere.
* Two reference answers (q21, q22) name Enterprise, which the product facts do not; they
  are left as recorded so the committed runs stay comparable.
