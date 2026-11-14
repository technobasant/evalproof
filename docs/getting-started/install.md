---
description: Install evalproof with uvx, uv tool, pipx or pip; supported Python versions and how to upgrade.
---

# Install

evalproof is a command-line tool and a Python library, published on
[PyPI](https://pypi.org/project/evalproof/). It has four runtime dependencies (httpx, numpy, pydantic, PyYAML); no vendor SDK, no optional extras. This page is for anyone installing it for the first time.

## Choose an installer

=== "uvx (no install)"

    Run it without installing (needs [uv](https://docs.astral.sh/uv/)):

    ```bash title="Terminal"
    uvx evalproof --version
    uvx evalproof@0.7.1 init          # pin a version
    ```

=== "uv tool"

    Install the CLI on your `PATH`, isolated from your projects:

    ```bash title="Terminal"
    uv tool install evalproof
    evalproof --version               # evalproof 0.7.1
    ```

=== "pipx"

    Install the CLI on your `PATH`, isolated from your projects:

    ```bash title="Terminal"
    pipx install evalproof
    evalproof --version               # evalproof 0.7.1
    ```

=== "pip or a project"

    Add it to a project, to call it from Python or pin it with the project's other dev tools:

    ```bash title="Terminal"
    uv add --dev evalproof            # or: pip install evalproof
    uv run evalproof --version
    ```

## Supported Python

Python 3.11 to 3.14.

## Upgrade

| Installer | Command |
|---|---|
| uv tool | `uv tool upgrade evalproof` |
| pipx | `pipx upgrade evalproof` |
| pip | `pip install -U evalproof` |

Run files from older versions keep working; see the [changelog](../about/changelog.md) for what changed.

## In GitHub Actions

No install step is needed: the composite action
`technobasant/evalproof@v0.7.1` runs the evalproof of the tag you pin ([In CI](../guides/ci.md#in-ci)).

## What else you need

What else you need depends on what you run:

| To | You need |
|---|---|
| `compare`, `calibrate`, `verify`, `rehash`, `report`, `import`, `init`, `doctor`, `readiness` | nothing: they read run files and make no model call |
| `run` against a Python function or an HTTP endpoint | that function or endpoint |
| `run` on a local model | [Ollama](https://ollama.com) (the default), vLLM, LM Studio or llama.cpp, with the model pulled |
| `run` on a hosted model | the provider's key in an environment variable ([Providers](../guides/providers.md#providers)) |
| `run` with your Claude Code or Codex login | the `claude` or `codex` CLI on your `PATH`, logged in; check with `evalproof doctor` |

## Next steps

* Run a first suite: the [five-minute quickstart](quickstart.md#five-minute-quickstart).
* Check which credentials a run needs: [What it needs to run](credentials.md).
* Moving from an older version: [Upgrading](../about/upgrading.md).
