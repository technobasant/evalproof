---
description: Which credentials a run needs, locally and in CI, and which modes need none.
---

# What it needs to run

Which credential a run needs depends on what the system and the judge are. This page is for anyone setting up a run on a laptop or in CI.

!!! note "CI and secrets"

    A CI run that calls a model needs a model key, or a CLI token, as a repository secret; a login on your machine does not carry over to CI. **Keyless mode needs no secrets:** CI compares run files you committed and makes no model call. Fork pull requests never receive secrets, so no workflow runs a model for them.

## Requirements by target

| Target | Credential variable | Local use | CI use | Where allowed |
|---|---|---|---|---|
| No model: a Python function or an HTTP endpoint with no LLM behind it, or committed run files | none | nothing | nothing | anywhere, including fork pull requests |
| Local models (Ollama, vLLM, LM Studio, llama.cpp) | none | the server running, with the model pulled | a server on the runner, started in a workflow step | anywhere |
| Hosted APIs (Anthropic, OpenAI and the rest of [Providers](../guides/providers.md#providers)) | the provider's key, such as `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` | the variable in your environment | the key as a repository secret | your own repository; fork pull requests never receive it |
| Claude Code CLI | `ANTHROPIC_API_KEY` (api-key mode) or `CLAUDE_CODE_OAUTH_TOKEN` | your Claude Code login, no variable | `ANTHROPIC_API_KEY` as a secret with `EVALPROOF_CLI_AUTH=api-key`, or `CLAUDE_CODE_OAUTH_TOKEN` as a secret (made by `claude setup-token`) | the token: your own repository and your own CI only; use an API key for a secret shared across a team or an organisation |
| Codex CLI | `OPENAI_API_KEY` | your Codex login, no variable | `OPENAI_API_KEY` as a secret with `EVALPROOF_CLI_AUTH=api-key` | a personal Codex login only on trusted private infrastructure, never on public repos or for open source work |

## Notes per target

!!! warning "Claude Code token in CI"

    `CLAUDE_CODE_OAUTH_TOKEN` (made by `claude setup-token`) is for your own repository and your own CI only. Use an API key for a secret shared across a team or an organisation.

!!! warning "Codex login in CI"

    Codex has no token option. A personal Codex login belongs only on trusted private infrastructure, never on public repos or for open source work; use `OPENAI_API_KEY` as a secret with `EVALPROOF_CLI_AUTH=api-key` otherwise.

!!! tip "Let `init` write the workflow"

    `evalproof init --ci api-key` writes the workflow with the secret for the judge it chose, and
    prints this requirement for that judge when it finishes. For Claude Code,
    `evalproof init --ci api-key --cli-auth oauth-token` passes `CLAUDE_CODE_OAUTH_TOKEN` instead of
    the API key and warns that it is for your own repository and CI only; Codex has no token option.

!!! tip "Check with `doctor`"

    `evalproof doctor` prints, for each CLI target, which credential it would use here and whether
    that works in CI ("login only, not available in CI", or usable from a secret). It looks at the
    names of environment variables, never at their values.

## Next steps

* Wire the gate into GitHub Actions: [In CI](../guides/ci.md#in-ci).
* Every provider and the key it reads: [Providers](../guides/providers.md#providers).
