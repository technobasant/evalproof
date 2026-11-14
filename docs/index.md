# evalproof

[![PyPI](https://img.shields.io/pypi/v/evalproof?label=PyPI)](https://pypi.org/project/evalproof/)
[![Python](https://img.shields.io/pypi/pyversions/evalproof?label=Python)](https://pypi.org/project/evalproof/)
[![ci](https://github.com/technobasant/evalproof/actions/workflows/ci.yml/badge.svg)](https://github.com/technobasant/evalproof/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](https://github.com/technobasant/evalproof/blob/main/LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evalproof.streamlit.app)

**Treat LLM evaluation as measurement.** Calibrate the judge, correct its pass rate, and
gate CI on regressions that are real rather than noise.

**Try it in the browser: [evalproof.streamlit.app](https://evalproof.streamlit.app).** Check whether
a drop between two runs is real, or replay the run below and read the twelve answers that
changed. It runs this repository's code; after a quiet spell it takes about half a minute to wake.

An LLM judge is a classifier with error rates. If it passes bad answers at rate FPR and
good answers at rate TPR, its pass rate converges to `FPR + (TPR − FPR) · p`, not to the
true rate `p`. That is classify-and-count, and it is biased the way any classifier's count
is — the effect is worked through with a seeded simulation in
[*The classifier was 95% accurate. The trend it counted was a third too flat*](https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/)
([lab](https://github.com/technobasant/prevalence-shift)).

evalproof:

1. **measures the judge** against a small gold-labelled sample — confusion matrix, TPR,
   FPR, Cohen's κ;
2. **corrects its pass rate** with adjusted classify-and-count and prediction-powered
   inference, each with a confidence interval;
3. **compares runs on paired cases** — exact McNemar for pass/fail, paired bootstrap for
   scores, and a paired permutation test when each case runs several times;
4. **gates CI** only when a drop is both larger than your tolerance *and* statistically
   significant, plus latency and cost budgets, with distinct exit codes;
5. **checks the run files it gates** — `verify` proves, with no model call, that a committed
   run was made with this suite and covers the dataset as it is now;
6. **tests agents** — asserts on the tools an agent called and what it said, across several
   turns, several builds of the same system and repeated trials, with adapters for ADK,
   LangChain/LangGraph, the Claude Agent SDK and OpenAI Agents
   ([Testing agents](guides/testing-agents.md)).

It is local-first: Ollama is the default, every major hosted provider is one `target:` line
away ([Providers](guides/providers.md#providers)), a logged-in Claude Code or Codex can be the judge with no API
key, and every model call is cached by content so re-runs are free and reproducible. Results
from promptfoo, Inspect, DeepEval and lm-eval can be [imported](guides/import.md#import-results-from-other-tools)
and gated without re-running them.

## What it found on a real run

A local `qwen3:1.7b` product support assistant, judged by `qwen3:8b`, 83
questions, one Apple M5. Full method and every table in [MEASUREMENTS.md](about/measurements.md).

| Question | Measured |
|---|---|
| How good is the 8B judge? | TPR 92.6%, FPR 6.7%, Cohen's κ 0.78 against the answer key |
| Did swapping the grounded prompt for a "friendly" one regress quality? | −4.8 pts on the judge, 8 cases worse vs 4 better, McNemar p = 0.39 — **gate warns, does not fail**; a delta-only gate would have failed it |
| What did regress? | p95 latency, 1,469 → 4,422 ms; mean answer 19 → 77 tokens. A 2,500 ms budget fails the change with exit 3 |
| Does correcting the judge's rate help? | On average yes (offset 4.8 → ~3.8 pts), but with 30 labels the interval doubles; at 83 cases the judge's bias is smaller than its own noise |
| Was the gold trustworthy? | No: of 13 judge–key disagreements the key was wrong 7 times, the judge 4. Most of the "bias" corrected was the gold's |
| What changes with reviewed labels? | On a random 30, the key was wrong once. That one case moved the corrected rate 3 points, the wrong way for this sample, and showed the judge's FPR at 25%, not 14%. A reasoning judge reached κ 0.91 against the reviewed labels where the key scored both judges 0.81 |
| Is pairwise judging safer? | No: 42% of verdicts flipped with the order of the two answers, and the judge preferred the candidate 42 to 6 — mostly where both answers were right and the candidate's was five times longer |
| Does a reasoning judge help? | Against audited labels, yes: thinking on passed no wrong answer (three without it) and κ rose from 0.86–0.88 to 0.90–0.94, at 6.7× the judge time. Against the raw answer key it looked worse, because the key is wrong on exactly those cases. The gate's decision did not change |
| Is it reproducible? | 20/20 identical outputs and verdicts with the cache off; a warm-cache re-run took 0.05 s instead of 363 s; a re-judge three days later matched 83/83 |

## Where to go next

* **Getting started:** [Install](getting-started/install.md),
  [what it needs to run](getting-started/credentials.md), the
  [five-minute quickstart](getting-started/quickstart.md) and [use cases](getting-started/use-cases.md).
* **Guides:** [Providers](guides/providers.md), [Testing agents](guides/testing-agents.md),
  [Repeated trials](guides/repeated-trials.md), [Readiness](guides/readiness.md),
  [Agent SDK adapters](guides/adapters.md), [Import results from other tools](guides/import.md),
  [From Python](guides/python.md) and [In CI](guides/ci.md).
* **Reference:** [Commands](reference/commands.md), [Concepts](reference/concepts.md) and the
  [API reference](reference/api.md).
* **About:** [Architecture](about/architecture.md), [Decisions](about/decisions.md),
  [How it relates to other tools](about/related-tools.md), [Limits](about/limits.md),
  [Roadmap](about/roadmap.md), [Measurements](about/measurements.md) and the
  [Changelog](about/changelog.md).
