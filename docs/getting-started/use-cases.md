# Use cases

Each of these is a suite file and one or two commands. The [quickstart](quickstart.md#five-minute-quickstart)
writes a starting suite.

**Gate a prompt or model change in a pull request.** Run the suite on the base branch and on the
change, then gate. The build fails only when the drop is larger than `max_drop` *and*
significant, so a two-case wobble on 60 cases does not block a merge:

```bash
evalproof run evals/suite.yaml --out base.json                     # on main
evalproof run evals/suite.yaml --out cand.json                     # on the branch
evalproof compare base.json cand.json --suite evals/suite.yaml --gate --show 5
```

**Choose between two models or providers.** Run the same suite with `--system` and compare the
two runs. The report gives the paired difference with its interval and p-value, the cases that
got worse, p95 latency and cost, so "cheaper and not measurably worse" is something you can show:

```bash
evalproof run evals/suite.yaml --system anthropic:claude-sonnet-4-5 --label sonnet --out a.json
evalproof run evals/suite.yaml --system groq:llama-3.3-70b-versatile --label llama --out b.json
evalproof compare a.json b.json --suite evals/suite.yaml
```

**Find out whether your LLM judge can be trusted.** Label 30 to 50 cases by hand
(`{"id": ..., "label": true}` per line) and calibrate. You get the judge's TPR, FPR and
Cohen's κ, and its pass rate corrected for those errors, each with an interval:

```bash
evalproof calibrate runs/latest.json --judge-scorer judge --labels gold.jsonl
```

**Use your Claude Code or Codex subscription as the judge.** Set `judge.target:
claude-code:haiku` (or `codex:<model>`), run `evalproof doctor evals/suite.yaml`, and run as
usual. No API key and no per-call bill; the CLI runs isolated, with no tools and none of your
project's instructions ([details](../guides/providers.md#use-your-claude-code-or-codex-login)).

**Gate results you already have from another eval tool.** Import each side, then compare as
usual. Nothing is re-run:

```bash
evalproof import promptfoo main/output.json --out base.json
evalproof import promptfoo pr/output.json --out cand.json
evalproof compare base.json cand.json --gate --metric pass
```

**Check a retrieval or RAG change.** Score the ranked ids with `recall_at_k`, `mrr` and `ndcg`
and gate on the one that matters to you. Recall@5 does not move when the right document drops
from rank 1 to rank 5; MRR and nDCG do ([`examples/retrieval`](https://github.com/technobasant/evalproof/blob/main/examples/retrieval)).

**Cases that must never fail, and a floor on the first run.** Tag the critical cases and set
`gate.must_pass: [critical]` and `gate.min_pass_rate`. They need no baseline, so they work
from the first run: `evalproof run evals/suite.yaml --gate` exits 1 when one fails
([Checks without a baseline](../reference/concepts.md#concepts)).

**Hold latency and cost while you swap a model.** `gate.max_p95_latency_ms` and
`gate.max_cost_usd` fail a change with exit 3, separate from a quality regression. On the
[measured run](../index.md#what-it-found-on-a-real-run), a prompt change that was not a significant quality drop still failed on
latency.

**Gate in CI with no model calls and no secrets.** `evalproof init --ci keyless` writes a
workflow that compares the run file developers commit against the base branch's.
The action runs `verify` first, so a stale run, a `--limit` run or a hand-edited
deterministic score fails the job ([In CI](../guides/ci.md#in-ci)).

**Test a non-LLM system the same way.** Any Python function (`python:module:func`) or HTTP
endpoint can be the system, scored by deterministic scorers: a classifier, a router, a search
service. The [offline example](quickstart.md#five-minute-quickstart) is a keyword router.
