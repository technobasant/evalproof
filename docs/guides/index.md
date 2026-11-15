---
description: Task-oriented guides for providers, agents, repeated trials, importing results and CI.
---

# Guides

Each guide takes one task from start to finish. They are for people who have run the [quickstart](../getting-started/quickstart.md) and want to do something specific.

| Guide | What it covers |
|---|---|
| [Providers](providers.md) | every model target, the key it reads, and logged-in Claude Code and Codex as judge |
| [Coding agents](coding-agents.md) | the `AGENTS.md` section and the skill `init` writes, so a coding agent can use evalproof in your repository |
| [Testing agents](testing-agents.md) | traces, the `checks` scorer, several turns and variants |
| [Testing a system in any language](command-target.md) | run a Node.js, Go or shell program as the system, with the case on stdin |
| [Repeated trials](repeated-trials.md) | run each case several times, and compare several candidates in one table |
| [Readiness](readiness.md) | which parts of quality a suite covers and which it leaves untested |
| [Agent SDK adapters](adapters.md) | build a trace from ADK, LangChain/LangGraph, the Claude Agent SDK or OpenAI Agents |
| [Import results from other tools](import.md) | gate promptfoo, Inspect, DeepEval, lm-eval, garak, ADK, Claude Code plugin eval, CSV and JSONL results, or turn an ADK eval set into a dataset |
| [Import test cases from other tools](import-test-cases.md) | turn the test cases you already have in promptfoo, DeepEval, Ragas, LangSmith, OpenAI Evals, Inspect, Braintrust, Langfuse, Phoenix, JSONL or CSV into a dataset and a suite, or start a repository with `init --from` |
| [From Python](python.md) | the same functions the CLI uses |
| [In CI](ci.md) | the GitHub Action and the workflows `init` writes |
| [Troubleshooting](troubleshooting.md) | what an error message means and how to fix it |
