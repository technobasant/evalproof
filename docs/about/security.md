---
description: What evalproof sends, stores and trusts, how it handles credentials and coding-agent logins, and how to report a vulnerability.
---

# Security and data handling

What leaves your machine when you run evalproof, what it writes to disk, how it treats credentials, and what it trusts. It is for anyone deciding whether the tool is safe to run on their data and in their CI.

## What leaves the machine

evalproof sends data only to the targets you configure.

| Target | What is sent, and where |
|---|---|
| A hosted model (`anthropic:`, `openai:` and the other providers) | each case's prompt, to that provider's API or to the `base_url` you set |
| A local model (`ollama:`) | the prompt, to the Ollama server you run; evalproof also asks it for its version and the model's digest so the run file can record them |
| An `http(s)://` endpoint | every case, to that endpoint |
| A `python:module:function` target | nothing: it calls your function, which does what you wrote it to do |
| `claude-code:` or `codex:` | the prompt, to your own `claude` or `codex` binary, which talks to its own service |
| The judge | the text its prompt is built from (the system's output and the rubric), to the judge's target, which can be any of the above |

The network clients in the package are the provider clients, the `http` system and the Ollama check above. Nothing else in evalproof opens a connection. `compare`, `calibrate`, `report`, `verify`, `rehash`, `readiness` and `doctor` make no model call.

## What is stored

| Where | What it holds |
|---|---|
| The response cache, `.evalproof/cache` unless the suite sets `cache_dir` | the reply each model call returned, with its token counts and latency, keyed on a hash of the provider, model, endpoint, messages and parameters |
| A run file | each case's output, its scores with their reasons, any error, the trace of tool calls when the system returns one, the suite and dataset hashes, and the environment |
| The environment in a run file | the Python version, the platform, the machine and processor, the model ids that answered, and the Ollama facts above |

A run file does not store a case's input. It does store outputs and tool results, so a run file can hold whatever your system produced. Treat the cache and run files like the data they were computed from. If you commit run files, which keyless CI relies on, check that the outputs are safe to commit.

## Credentials

!!! warning "Keys do not belong in the suite file"
    A suite file is usually committed. `headers` and any key placed in it would be committed too. Keep keys in environment variables.

* **API keys come from the environment.** A provider reads its key from its usual variable, or from the one a suite names with `api_key_env`, and never from a suite file. The key is sent only to the provider's API.
* **Keys are not written down.** Run files, the cache and reports do not hold them, and the suite's `headers` never reach a run file or the cache and are not part of the cache key. A path where a key is written would be a vulnerability.
* **evalproof never reads a CLI's token.** For `claude-code:` and `codex:` it runs your own binary as a subprocess and reads its JSON output. It does not open a token file or a keychain entry.
* **`doctor` reads names, not values.** `evalproof doctor` checks which credential variables are set by name only, so nothing secret can be printed or logged.
* **In `login` mode, API-key variables are removed** from the environment the CLI is started with, so the call uses the login you chose.

A login is a file on one machine, so it never works in CI. There, a coding-agent CLI needs a key or a token as a repository secret. A Claude Code token is for your own repository and CI only, and a personal Codex login belongs on trusted private infrastructure, never on public repositories. The [credentials table](../getting-started/credentials.md) lists each target.

### Coding agents run isolated

A coding agent, left alone, loads the project's instructions, your personal memory, plugins and tools into the prompt, and it can run commands. A judge that read those would not be a judge, so every call to a coding-agent CLI runs in a fresh empty directory, with the agent's own system prompt replaced and its tools, settings, MCP servers and skills switched off. No permission-bypass flag is ever passed: tools are removed instead of being allowed. A Codex call that used a tool is refused.

## CI and pull requests from forks

Keyless CI compares run files you committed and makes no model call, so it needs no secrets and runs anywhere, including on pull requests from forks. A workflow that runs a model needs a secret, and pull requests from forks never receive secrets, so for them the workflow runs nothing and says so. See [In CI](../guides/ci.md).

## What evalproof trusts

* **A suite file is code.** A `python:module:function` target imports and runs that function, and an `https://` target sends every case to that endpoint. Run suites you trust, as you would a test file.
* **Run files are data.** `compare`, `calibrate` and `report` parse them with Pydantic and never execute anything from them.
* **The cache is data too,** but it is trusted to hold the replies it was given: anyone who can write to `cache_dir` can change what a re-run reports. Keep it out of shared or untrusted locations.

## This documentation site

The site is built with MkDocs Material and its privacy plugin, which downloads fonts and other external assets at build time and serves them from the site itself. A page makes no request to a third-party host.

## Releases

Releases reach PyPI from GitHub Actions through trusted publishing, so no long-lived PyPI token exists, and each file carries a signed provenance attestation. [Versioning and support](versioning.md#how-a-release-is-built-and-published) describes the steps.

## Reporting a vulnerability

Please do not open a public issue. Report it privately through [GitHub's private vulnerability reporting](https://github.com/technobasant/evalproof/security/advisories/new), or by email to contact@basantbhattarai.com.np. Include the version, a minimal suite or run file that reproduces it, and what an attacker gains. You will get an acknowledgement within seven days, and a confirmed issue gets a fix in a patch release and a published advisory crediting you, unless you prefer otherwise.

Only the latest minor release is supported. See [Versioning and support](versioning.md#supported-versions).

## Next steps

* Which credential each target uses: [Credentials](../getting-started/credentials.md).
* Running evalproof in CI: [In CI](../guides/ci.md).
* The policy itself: [SECURITY.md](https://github.com/technobasant/evalproof/blob/main/SECURITY.md).
