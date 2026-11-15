---
description: Install evalproof with uvx, uv tool, pipx, pip or Docker; supported Python versions and how to upgrade.
---

# Install

evalproof is a command-line tool and a Python library, published on
[PyPI](https://pypi.org/project/evalproof/). It has four runtime dependencies (httpx, numpy, pydantic, PyYAML); no vendor SDK, no optional extras. This page is for anyone installing it for the first time.

## Choose an installer

=== "uvx (no install)"

    Run it without installing (needs [uv](https://docs.astral.sh/uv/)):

    ```bash title="Terminal"
    uvx evalproof --version
    uvx evalproof@1.0.0 init          # pin a version
    ```

=== "uv tool"

    Install the CLI on your `PATH`, isolated from your projects:

    ```bash title="Terminal"
    uv tool install evalproof
    evalproof --version               # evalproof 1.0.0
    ```

=== "pipx"

    Install the CLI on your `PATH`, isolated from your projects:

    ```bash title="Terminal"
    pipx install evalproof
    evalproof --version               # evalproof 1.0.0
    ```

=== "pip or a project"

    Add it to a project, to call it from Python or pin it with the project's other dev tools:

    ```bash title="Terminal"
    uv add --dev evalproof            # or: pip install evalproof
    uv run evalproof --version
    ```

### npm

If your project already uses Node, run it with `npx`. The `evalproof` package on npm is a small
launcher: it finds [uv](https://docs.astral.sh/uv/) or [pipx](https://pipx.pypa.io/) and runs the
PyPI release of the same version.

```bash title="Terminal"
npx evalproof --version
npx evalproof init
```

You need uv or pipx on your `PATH`. The launcher does not install Python or evalproof itself.

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
`technobasant/evalproof@v1.0.0` runs the evalproof of the tag you pin ([In CI](../guides/ci.md#in-ci)).

## Docker

A container image holds evalproof and its four dependencies on Python 3.13, for `linux/amd64` and `linux/arm64`. It is published to the GitHub container registry with each release from 1.0.0 on, and it installs the same wheel that went to PyPI. Use it where you would rather not install Python packages, or in a CI system that runs containers.

```bash title="Terminal"
docker pull ghcr.io/technobasant/evalproof:latest
docker run --rm ghcr.io/technobasant/evalproof:latest --version
```

The entry point is `evalproof`, so what follows the image name is the command. The working directory is `/work`: mount your suite directory there and give paths relative to it.

```bash title="Terminal"
docker run --rm -v "$PWD:/work" ghcr.io/technobasant/evalproof:latest \
  run evals/suite.yaml --out evals/runs/latest.json
docker run --rm -v "$PWD:/work" ghcr.io/technobasant/evalproof:latest \
  verify evals/runs/latest.json evals/suite.yaml
```

Pass a provider key with `-e` and no value. Docker then copies it from your environment, so the key is not in the command line or your shell history:

```bash title="Terminal"
docker run --rm -e ANTHROPIC_API_KEY -v "$PWD:/work" \
  ghcr.io/technobasant/evalproof:latest run evals/suite.yaml --out evals/runs/latest.json
```

**Pin a version** in CI, as you would with any other installer. Each release has three tags: `1.0.0` (this release only), `1.0` (the newest patch of that minor release) and `latest` (the newest stable release). A pre-release gets only its exact tag.

What the image does not have:

* **No coding-agent CLI.** `claude`, `codex` and the other agent CLIs are not in the image, so a suite whose system or judge is `claude-code:`, `codex:` or another CLI provider fails: every case errors with "`claude` was not found on PATH", and the command exits 2. It makes no model call. Run those suites with an installed evalproof, on a machine where the CLI is logged in.
* **No packages beyond evalproof's own.** A `python:` system that imports anything else needs an image of its own, built on this one:

    ```dockerfile title="Dockerfile"
    FROM ghcr.io/technobasant/evalproof:1.0.0
    RUN pip install --no-cache-dir --user your-package
    ```

* **No model server.** Inside the container, `127.0.0.1` is the container. To reach Ollama on your own machine, set `base_url: http://host.docker.internal:11434` in the suite. On Linux, add `--add-host=host.docker.internal:host-gateway` to `docker run`.

The process runs as an unprivileged user (uid 10001). Docker Desktop on macOS and Windows lets it write to the mounted directory. On Linux, run it as yourself so the run files and the `.evalproof/` cache belong to you: add `--user "$(id -u):$(id -g)"`.

## What else you need

What else you need depends on what you run:

| To | You need |
|---|---|
| `compare`, `calibrate`, `verify`, `rehash`, `report`, `import`, `init`, `doctor`, `readiness` | nothing: they read run files and make no model call |
| `run` against a Python function, an HTTP endpoint or a command | that function, endpoint or program |
| `run` on a local model | [Ollama](https://ollama.com) (the default), vLLM, LM Studio or llama.cpp, with the model pulled |
| `run` on a hosted model | the provider's key in an environment variable ([Providers](../guides/providers.md#providers)) |
| `run` with your Claude Code or Codex login | the `claude` or `codex` CLI on your `PATH`, logged in; check with `evalproof doctor` |

## Next steps

* Run a first suite: the [five-minute quickstart](quickstart.md#five-minute-quickstart).
* Check which credentials a run needs: [What it needs to run](credentials.md).
* Moving from an older version: [Upgrading](../about/upgrading.md).
