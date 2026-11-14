---
description: Suites, runs, scorers, calibration, the gate and how cases are paired: how evalproof reaches a verdict.
---

# Concepts

The ideas the rest of the documentation uses: suites, runs, scorers, calibration, the gate and how cases are paired. It is for anyone who wants to know how evalproof reaches a verdict.

## Suite

one YAML file: a JSONL golden set, the system under test, scorers, an optional
judge, calibration settings and a gate.

```yaml title="suite.yaml"
name: support-qa
dataset: cases.jsonl              # {"id", "input", "reference", ...extra fields}
system:
  target: ollama:qwen3:1.7b       # or groq:…, anthropic:…, python:module:func, https://…
  system_prompt_file: prompt.md
  prompt_template: "{question}"
  params: {temperature: 0, seed: 0, think: false}
judge:
  target: ollama:qwen3:8b
scorers:
  - {name: answer_key, type: regex, params: {any_of_field: accept, none_of_field: reject}}
  - {name: judge, type: llm_judge, params: {rubric_file: rubric.md}}
calibration: {judge_scorer: judge, gold_scorer: answer_key, gold_sample: 30}
gate: {metric: judge, max_drop: 0.03, alpha: 0.05, min_cases: 40, max_p95_latency_ms: 2500}
```

## Run

`evalproof run suite.yaml` writes one self-describing JSON file: every case's
output, latency, tokens, cost and scores, plus the dataset hash, a hash of each case, a
fingerprint of the suite (system, judge, scorers, seed), the Ollama version and model digests,
and for a CLI target the CLI, its version, the isolation scheme and the model that answered.
`verify` and `compare` read these, so a run file can be committed and checked later without
running anything.

## Scorers

`exact_match`, `contains`, `regex` (accept/reject answer keys),
`json_schema`, `numeric_tolerance`, `recall_at_k`, `mrr`, `ndcg`, `checks` (what an agent did:
[Testing agents](../guides/testing-agents.md#testing-agents)), and `llm_judge`: a strict
JSON verdict, one retry on an unparseable reply, and an explicit *undecided* state that is
counted, never scored as a pass. `evalproof pairwise` judges two runs head to head in both
orders and reports order-dependent verdicts as ties.

## Calibrate

`evalproof calibrate run.json --judge-scorer judge --labels gold.jsonl`
(human labels) or `--gold-scorer answer_key --gold-sample 30`:

| Estimate | What it is |
|---|---|
| raw judge pass rate | classify-and-count, Wilson interval |
| gold labels only | the labelled sample alone |
| ACC | `(q − FPR) / (TPR − FPR)`, bootstrap interval over both the judged set and the gold sample |
| PPI | judge mean on unlabelled cases + mean (label − judge) on the gold sample, normal interval |

## Compare and gate

`evalproof compare baseline.json candidate.json --suite suite.yaml --gate`

| Exit | Meaning |
|---|---|
| 0 | pass — possibly with warnings (a big drop that is not significant; a significant drop within tolerance) |
| 1 | quality regression: drop ≥ `max_drop` **and** p < `alpha`; or an absolute check failed (`min_pass_rate`, `must_pass`, `max_new_failures`) |
| 3 | budget breach: candidate p95 latency or cost over the limit |
| 4 | cannot gate, which is not a regression: fewer than `min_cases` paired cases, a gated metric with no paired scores, the dataset changed and a run predates per-case hashes, the judge changed (unless `--allow-judge-change`), `verify` found the candidate does not match the suite, or the two runs have a different number of trials per case |
| 2 | usage or configuration error (an unknown key in the suite, an unknown `--metric`); also `run` when every case errored |
| 130 | interrupted (Ctrl-C) |

The report says "cannot gate" for exit 4 and "fail" only for a regression, and it lists the
ids that regressed on every compared metric (ten per metric, then "and N more"). `--show N`
(default 5) adds, for that many regressed cases of the gated metrics, both answers and the
candidate scorer's reason in a collapsed block. `run` and `report` list failed cases the same
way (`--show N`, default 10; 0 hides the list).

## When the cases change

Every case in a run carries a hash of its content, and `compare`
pairs two runs by id and hash. A case added since the baseline has no baseline score and is not
compared. A case edited since the baseline (same id, new content) is excluded and listed in the
report, so adding or editing cases never blocks the gate; the baseline catches up when the
change is merged and its run file replaces the old one. Tags label a case and do not change what
the system is asked, so adding or changing a tag is not an edit. A `--limit` run pairs with a
full baseline on the cases it ran. Only when the dataset differs and one run predates per-case
hashes (written before 0.5) can the cases that stayed the same not be told from the ones that
changed, and the gate exits 4. Fix it without a model call: run
`evalproof rehash BASELINE SUITE`. That works when the cases the baseline scored are still the
first ones in the dataset, unchanged (cases appended after them are fine), which the
`dataset_hash` stored in the run file proves. When a case was edited or inserted, rehash on the
dataset the baseline was made from (the base branch, before your change), or re-run the
baseline with 0.5 or newer. Run files from 0.5.0 keep working: they have
no tag-free hash, so pairing falls back to the case hash, which includes tags. A dataset change
is a warning.

## Checks without a baseline

The significance test needs a baseline. These do not:

```yaml title="suite.yaml"
gate:
  metric: judge
  min_pass_rate: 0.8           # a floor for the candidate; or per scorer: {judge: 0.8, answer_key: 0.9}
  must_pass: [critical]        # every case with this tag must pass every pass/fail gated metric
  max_new_failures: 0          # most cases that may go from pass to fail on a pass/fail metric
```

`min_pass_rate`, `must_pass` and `max_new_failures` need a pass/fail metric. On a continuous one
`min_pass_rate` cannot gate (exit 4), `max_new_failures` is skipped with a warning, and
`must_pass` fails only an errored or undecided case.

`evalproof run suite.yaml --gate` applies `min_pass_rate` and `must_pass` to the run alone and
exits 1 when one fails (a usage error if the suite sets neither), which is the gate for a first
run. `compare --gate` applies all three, and the same flags exist there:
`--min-pass-rate`, `--must-pass` (repeatable) and `--max-new-failures`. `max_new_failures: 0`
suits deterministic scorers, where one case flipping is a real change that the statistics, with
few cases, would call noise. Errored cases count as failures in every one of these checks.

| `gate:` key | Meaning | Default |
|---|---|---|
| `metric` / `metrics` | the scorer(s) to gate; several get a Holm correction | required |
| `max_drop` | the largest tolerated drop, as a fraction | 0.02 |
| `alpha` | significance level | 0.05 |
| `min_cases` | fewest paired cases the gate will decide on | 20 |
| `aggregate` | with several [trials](../guides/repeated-trials.md#repeated-trials) per case, how a case's trials become one value on a pass/fail metric: `mean`, `all` or `any` | `mean` |
| `max_p95_latency_ms`, `max_cost_usd` | budgets on the candidate | none |
| `allow_judge_change` | accept a changed judge | false |
| `min_pass_rate`, `must_pass`, `max_new_failures` | the checks above | none |

Unknown keys are errors everywhere in a suite, not silently ignored: a typo such as
`max_drops` fails with `suite.yaml: gate.max_drops: unknown key (did you mean 'max_drop'?)`.

The gate checks one metric by default. To gate several, repeat `--metric` or list them in
the suite; each metric's p-value is then Holm-adjusted for the number of metrics gated, so
the chance of failing a build on noise stays at `alpha` for the set as a whole, and the
report shows the raw and the adjusted p-value side by side. Metrics that are not gated are
still reported.

```yaml title="suite.yaml"
gate:
  metrics: [judge, answer_key]   # one metric: `metric: judge`
  max_drop: 0.03
  alpha: 0.05
  min_cases: 40
```

```bash title="Terminal"
evalproof compare baseline.json candidate.json --gate --metric judge --metric answer_key
```

## Next steps

* Run the gate in GitHub Actions: [In CI](../guides/ci.md#in-ci).
* Every command and flag: [Commands](commands.md).
