---
description: evalproof calibrates the LLM judge, corrects its pass rate and gates CI on regressions that are real rather than noise.
---

<div class="hero" markdown>

# evalproof

Treat LLM evaluation as measurement. Calibrate the judge, correct its pass rate, and gate CI on regressions that are real rather than noise.
{ .lead }

[Get started](getting-started/index.md){ .md-button .md-button--primary }
[Test an agent](guides/testing-agents.md#testing-agents){ .md-button }

</div>

[![PyPI](https://img.shields.io/pypi/v/evalproof?label=PyPI)](https://pypi.org/project/evalproof/)
[![Python](https://img.shields.io/pypi/pyversions/evalproof?label=Python)](https://pypi.org/project/evalproof/)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](https://github.com/technobasant/evalproof/blob/main/LICENSE)

**Try it in the browser: [evalproof.streamlit.app](https://evalproof.streamlit.app).** Check whether
a drop between two runs is real, or replay the run below and read the twelve answers that
changed. It runs this repository's code; after a quiet spell it takes about half a minute to wake.

An LLM judge is a classifier with error rates. If it passes bad answers at rate FPR and
good answers at rate TPR, its pass rate converges to `FPR + (TPR − FPR) · p`, not to the
true rate `p`. That is classify-and-count, and it is biased the way any classifier's count
is — the effect is worked through with a seeded simulation in
[*The classifier was 95% accurate. The trend it counted was a third too flat*](https://basantbhattarai.com.np/writing/count-prevalence-not-classifications/)
([lab](https://github.com/technobasant/prevalence-shift)).

<div class="grid cards" markdown>

-   :material-scale-balance: **Measure the judge**

    ---

    Confusion matrix, TPR, FPR and Cohen's κ against a small gold-labelled sample, then a pass rate corrected for the judge's errors.

    [:octicons-arrow-right-24: Calibrate](reference/concepts.md#calibrate)

-   :material-compare: **Compare runs**

    ---

    Paired cases, exact McNemar for pass/fail and a paired bootstrap for scores, with an interval and a p-value.

    [:octicons-arrow-right-24: Compare and gate](reference/concepts.md#compare-and-gate)

-   :material-shield-check: **Gate CI**

    ---

    Fail only when a drop is larger than your tolerance and significant, plus latency and cost budgets, with distinct exit codes.

    [:octicons-arrow-right-24: In CI](guides/ci.md#in-ci)

-   :material-robot-outline: **Test agents**

    ---

    Assert on the tools an agent called and what it said, across several turns and several builds of the same system.

    [:octicons-arrow-right-24: Testing agents](guides/testing-agents.md#testing-agents)

-   :material-repeat: **Repeated trials**

    ---

    Run each case several times and read pass@k, pass^k and the flaky cases, or compare several candidates in one table.

    [:octicons-arrow-right-24: Repeated trials](guides/repeated-trials.md#repeated-trials)

-   :material-import: **Import results**

    ---

    Gate results from promptfoo, Inspect, DeepEval and lm-eval without re-running them.

    [:octicons-arrow-right-24: Import results](guides/import.md#import-results-from-other-tools)

</div>

## What evalproof does

1. **Measures the judge** against a small gold-labelled sample: confusion matrix, TPR, FPR, Cohen's κ.
2. **Corrects its pass rate** with adjusted classify-and-count and prediction-powered inference, each with a confidence interval.
3. **Compares runs on paired cases**: exact McNemar for pass/fail, paired bootstrap for scores, and a paired permutation test when each case runs several times.
4. **Gates CI** only when a drop is both larger than your tolerance *and* statistically significant, plus latency and cost budgets, with distinct exit codes.
5. **Checks the run files it gates**: `verify` proves, with no model call, that a committed run was made with this suite and covers the dataset as it is now.
6. **Tests agents**: asserts on the tools an agent called and what it said, across several turns, several builds of the same system and repeated trials, with adapters for ADK, LangChain/LangGraph, the Claude Agent SDK and OpenAI Agents ([Testing agents](guides/testing-agents.md#testing-agents)).

It is local-first: Ollama is the default, every major hosted provider is one `target:` line
away ([Providers](guides/providers.md#providers)), a logged-in Claude Code or Codex can be the judge with no API
key, and every model call is cached by content so re-runs are free and reproducible. Results
from promptfoo, Inspect, DeepEval and lm-eval can be [imported](guides/import.md#import-results-from-other-tools)
and gated without re-running them.

## Install in one line

=== "uvx"

    ```bash title="Run without installing"
    uvx evalproof --version
    ```

=== "uv tool"

    ```bash title="Install the CLI"
    uv tool install evalproof
    ```

=== "pipx"

    ```bash title="Install the CLI"
    pipx install evalproof
    ```

=== "pip"

    ```bash title="Add it to an environment"
    pip install evalproof
    ```

Then follow the [five-minute quickstart](getting-started/quickstart.md#five-minute-quickstart).

## What it found on a real run

A local `qwen3:1.7b` product support assistant, judged by `qwen3:8b`, 83
questions, one Apple M5. Full method and every table in [Measurements](about/measurements.md).

| Question | Measured |
|---|---|
| How good is the 8B judge? | TPR 92.6%, FPR 6.7%, Cohen's κ 0.78 against the answer key |
| Did swapping the grounded prompt for a "friendly" one regress quality? | −4.8 pts on the judge, 8 cases worse vs 4 better, McNemar p = 0.39 — **gate warns, does not fail**; a delta-only gate would have failed it |
| What did regress? | p95 latency, 1,469 → 4,422 ms; mean answer 19 → 77 tokens. A 2,500 ms budget fails the change with exit 3 |
| Was the gold trustworthy? | No: of 13 judge–key disagreements the key was wrong 7 times, the judge 4. Most of the "bias" corrected was the gold's |
| Is pairwise judging safer? | No: 42% of verdicts flipped with the order of the two answers, and the judge preferred the candidate 42 to 6 — mostly where both answers were right and the candidate's was five times longer |
| Is it reproducible? | 20/20 identical outputs and verdicts with the cache off; a warm-cache re-run took 0.05 s instead of 363 s; a re-judge three days later matched 83/83 |

The other findings (correcting the judge's rate, reviewed labels, a reasoning judge) are in [Measurements](about/measurements.md).

Documented version: **0.7.x**. This site tracks the latest release.
