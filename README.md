# evalproof

[![PyPI](https://img.shields.io/pypi/v/evalproof?label=PyPI)](https://pypi.org/project/evalproof/)
[![Python](https://img.shields.io/pypi/pyversions/evalproof?label=Python)](https://pypi.org/project/evalproof/)
[![ci](https://github.com/technobasant/evalproof/actions/workflows/ci.yml/badge.svg)](https://github.com/technobasant/evalproof/actions/workflows/ci.yml)
[![licence](https://img.shields.io/badge/licence-Apache--2.0-blue)](https://github.com/technobasant/evalproof/blob/main/LICENSE)
[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://evalproof.streamlit.app)

**Documentation: [basantbhattarai.com.np/evalproof](https://basantbhattarai.com.np/evalproof/)**

**Treat LLM evaluation as measurement.** Calibrate the judge, correct its pass rate, and
gate CI on regressions that are real rather than noise.

**Try it in the browser: [evalproof.streamlit.app](https://evalproof.streamlit.app).** Check whether
a drop between two runs is real, or replay the
[measured run](https://basantbhattarai.com.np/evalproof/about/measurements/) and read the twelve answers that
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
   ([Testing agents](https://basantbhattarai.com.np/evalproof/guides/testing-agents/)).

It is local-first: Ollama is the default, every major hosted provider is one `target:` line
away ([Providers](https://basantbhattarai.com.np/evalproof/guides/providers/)), a logged-in Claude Code or Codex can be the judge with no API
key, and every model call is cached by content so re-runs are free and reproducible. Results
from promptfoo, Inspect, DeepEval and lm-eval can be [imported](https://basantbhattarai.com.np/evalproof/guides/import/)
and gated without re-running them.

## Install

evalproof is a command-line tool and a Python library, published on
[PyPI](https://pypi.org/project/evalproof/). It needs Python 3.11 to 3.14 and has four runtime
dependencies (httpx, numpy, pydantic, PyYAML); no vendor SDK, no optional extras.

**Run it without installing** (needs [uv](https://docs.astral.sh/uv/)):

```bash
uvx evalproof --version
uvx evalproof@0.6.0 init          # pin a version
```

**Install the CLI** on your `PATH`, isolated from your projects:

```bash
uv tool install evalproof         # or: pipx install evalproof
evalproof --version               # evalproof 0.6.0
```

**Add it to a project**, to call it from Python or pin it with the project's other dev tools:

```bash
uv add --dev evalproof            # or: pip install evalproof
uv run evalproof --version
```

**Upgrade** with `uv tool upgrade evalproof`, `pipx upgrade evalproof` or
`pip install -U evalproof`. Run files from older versions keep working; see the
[changelog](CHANGELOG.md) for what changed.

**In GitHub Actions**, no install step is needed: the composite action
`technobasant/evalproof@v0.6.0` runs the evalproof of the tag you pin
([In CI](https://basantbhattarai.com.np/evalproof/guides/ci/)).

## What it needs to run

Which credential a run needs depends on what the system and the judge are. **Keyless mode needs no
secrets:** CI compares run files you committed and makes no model call. Fork pull requests never
receive secrets, so no workflow runs a model for them.

| Target | Credential variable | Local use | CI use | Where allowed |
|---|---|---|---|---|
| No model: a Python function or an HTTP endpoint with no LLM behind it, or committed run files | none | nothing | nothing | anywhere, including fork pull requests |
| Local models (Ollama, vLLM, LM Studio, llama.cpp) | none | the server running, with the model pulled | a server on the runner, started in a workflow step | anywhere |
| Hosted APIs (Anthropic, OpenAI and the rest of [Providers](https://basantbhattarai.com.np/evalproof/guides/providers/)) | the provider's key, such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | the variable in your environment | the key as a repository secret | your own repository; fork pull requests never receive it |
| Claude Code CLI | `ANTHROPIC_API_KEY` (api-key mode) or `CLAUDE_CODE_OAUTH_TOKEN` | your Claude Code login, no variable | `ANTHROPIC_API_KEY` as a secret with `EVALPROOF_CLI_AUTH=api-key`, or `CLAUDE_CODE_OAUTH_TOKEN` as a secret (made by `claude setup-token`) | the token: your own repository and your own CI only; use an API key for a secret shared across a team or an organisation |
| Codex CLI | `OPENAI_API_KEY` | your Codex login, no variable | `OPENAI_API_KEY` as a secret with `EVALPROOF_CLI_AUTH=api-key` | a personal Codex login only on trusted private infrastructure, never on public repos or for open source work |

`evalproof init --ci api-key` writes the workflow with the secret for the judge it chose, and
prints this requirement for that judge when it finishes. For Claude Code,
`evalproof init --ci api-key --cli-auth oauth-token` passes `CLAUDE_CODE_OAUTH_TOKEN` instead of
the API key and warns that it is for your own repository and CI only; Codex has no token option.
[The full page](https://basantbhattarai.com.np/evalproof/getting-started/credentials/) adds what `evalproof doctor` reports.

## Use cases

Each is a suite file and one or two commands ([use cases](https://basantbhattarai.com.np/evalproof/getting-started/use-cases/)):
gate a prompt or model change in a pull request; choose between two models or providers; find out
whether your LLM judge can be trusted; use your Claude Code or Codex subscription as the judge; gate
results you already have from another eval tool; check a retrieval or RAG change; hold latency and
cost while you swap a model; gate in CI with no model calls and no secrets; test a non-LLM system
the same way.

## Five-minute quickstart

In your own repository, one command writes a starter suite, a CI workflow and an agent skill,
and a second runs it:

```bash
uvx evalproof init
uvx evalproof run evals/suite.yaml --out evals/runs/latest.json
```

`init` picks the judge from what is on your `PATH`: a logged-in `claude`, then `codex`, then
Ollama, then a hosted Anthropic model with `ANTHROPIC_API_KEY`. It writes `evals/` (the
suite, three sample cases, a README), `.github/workflows/evalproof.yml` (`--ci keyless`, the
default, makes no model calls in CI; `--ci api-key` runs the suite there; `--ci none` writes
no workflow) and a skill for coding agents at `.agents/skills/evalproof/SKILL.md` and
`.claude/skills/evalproof/SKILL.md` (`--no-skill` skips it). It prints one line per file,
never overwrites one without `--force`, and makes no model call. Replace the sample cases
with your own: the gate needs 20 before it will decide.

The offline example, the retrieval example and the LLM example are on the
[quickstart page](https://basantbhattarai.com.np/evalproof/getting-started/quickstart/).

## Documentation

The site at **[basantbhattarai.com.np/evalproof](https://basantbhattarai.com.np/evalproof/)** has:

* [Getting started](https://basantbhattarai.com.np/evalproof/getting-started/install/): install, what it needs to run, the quickstart, use cases.
* [Guides](https://basantbhattarai.com.np/evalproof/guides/providers/): providers, testing agents, repeated trials, readiness, Agent SDK adapters, importing other tools' results, calling it from Python, and CI.
* [Reference](https://basantbhattarai.com.np/evalproof/reference/commands/): every command, the concepts, and the [API](https://basantbhattarai.com.np/evalproof/reference/api/).
* [About](https://basantbhattarai.com.np/evalproof/about/architecture/): architecture, [decisions](https://basantbhattarai.com.np/evalproof/about/decisions/), how it relates to other tools, limits, the roadmap, the [measured run](https://basantbhattarai.com.np/evalproof/about/measurements/) and the [changelog](https://basantbhattarai.com.np/evalproof/about/changelog/).

[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) ·
[Security policy](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md) ·
[How to cite](CITATION.cff). Releases are published to PyPI from GitHub Actions with
[trusted publishing](https://docs.pypi.org/trusted-publishers/), and each file carries a
signed provenance attestation.

## Licence

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
