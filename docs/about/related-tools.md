---
description: How evalproof relates to promptfoo, DeepEval, Ragas and Inspect: a narrow measurement layer that can sit beside them.
---

# How it relates to other tools

[promptfoo](https://www.promptfoo.dev/docs/intro/) (open-source, local evals of prompts,
models and RAG pipelines with deterministic and model-graded assertions, a GitHub Action,
and red teaming), [DeepEval](https://deepeval.com/docs/getting-started) (pytest-style
LLM tests with research-backed metrics such as G-Eval), [Ragas](https://docs.ragas.io/en/stable/)
(LLM-driven metrics and experiments for LLM and RAG applications) and
[Inspect](https://inspect.aisi.org.uk/) (the UK AI Security Institute's framework for
model and agent evaluations, with sandboxed tools and a log viewer) are broad evaluation
frameworks.

evalproof is deliberately narrow: it is the measurement layer — how much a judge's verdict
is worth, and whether a difference between two runs is real. Its run file is plain,
documented JSON, so it can sit beside an existing harness: produce the per-case verdicts
however you like, then calibrate and gate here. For promptfoo, Inspect, DeepEval and
lm-eval you do not even need to do that: evalproof [imports their results](../guides/import.md#import-results-from-other-tools).
