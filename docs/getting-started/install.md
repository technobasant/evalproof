# Install

evalproof is a command-line tool and a Python library, published on
[PyPI](https://pypi.org/project/evalproof/). It needs Python 3.11 to 3.14 and has four runtime
dependencies (httpx, numpy, pydantic, PyYAML); no vendor SDK, no optional extras.

**Run it without installing** (needs [uv](https://docs.astral.sh/uv/)):

```bash
uvx evalproof --version
uvx evalproof@0.7.0 init          # pin a version
```

**Install the CLI** on your `PATH`, isolated from your projects:

```bash
uv tool install evalproof         # or: pipx install evalproof
evalproof --version               # evalproof 0.7.0
```

**Add it to a project**, to call it from Python or pin it with the project's other dev tools:

```bash
uv add --dev evalproof            # or: pip install evalproof
uv run evalproof --version
```

**Upgrade** with `uv tool upgrade evalproof`, `pipx upgrade evalproof` or
`pip install -U evalproof`. Run files from older versions keep working; see the
[changelog](../about/changelog.md) for what changed.

**In GitHub Actions**, no install step is needed: the composite action
`technobasant/evalproof@v0.7.0` runs the evalproof of the tag you pin ([In CI](../guides/ci.md#in-ci)).

What else you need depends on what you run:

| To | You need |
|---|---|
| `compare`, `calibrate`, `verify`, `rehash`, `report`, `import`, `init`, `doctor`, `readiness` | nothing: they read run files and make no model call |
| `run` against a Python function or an HTTP endpoint | that function or endpoint |
| `run` on a local model | [Ollama](https://ollama.com) (the default), vLLM, LM Studio or llama.cpp, with the model pulled |
| `run` on a hosted model | the provider's key in an environment variable ([Providers](../guides/providers.md#providers)) |
| `run` with your Claude Code or Codex login | the `claude` or `codex` CLI on your `PATH`, logged in; check with `evalproof doctor` |
