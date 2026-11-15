# Agent instructions for working on evalproof

This file is for coding agents that change evalproof's own code, tests and docs. It is not the
file `evalproof init` writes into a user's repository: that text is
`src/evalproof/templates/AGENTS.md`, between its `<!-- evalproof:begin -->` and
`<!-- evalproof:end -->` markers. Read `CONTRIBUTING.md` too. This file repeats its commands so
you can find them in one place; where the two differ, `CONTRIBUTING.md` and
`.github/workflows/ci.yml` are the source.

## Setup

```bash
uv sync                       # creates .venv with the dev group
uvx pre-commit install        # ruff format and check, YAML and TOML checks, on every commit
```

evalproof supports Python 3.11 to 3.14. CI runs all four on Linux, and Python 3.13 on macOS and
Windows.

## Run what CI runs

Run all of these before you say a change is done.

```bash
uv run ruff format --check . && uv run ruff check .      # use `uv run ruff format .` to fix
uv run mypy src                                          # strict
uv run pytest --cov --cov-report=term-missing --cov-fail-under=85
uv run --locked --group docs mkdocs build --strict       # the docs, with no warnings allowed
uvx zizmor@1.30.1 --offline --persona=regular .github/workflows action.yml   # workflow audit
```

* Targeted runs while you work: `uv run pytest tests/test_init.py -q`, or
  `uv run pytest -k name`.
* The Streamlit app in `app/` imports `src/` directly and has its own job. CI installs
  `app/requirements.txt` into a separate environment and runs `python -m pytest app`.
* `scripts/test-in-docker.sh` runs the whole test suite in Linux containers (Python 3.11 to
  3.14 on Debian, 3.13 on Alpine and on `linux/amd64`). It needs Docker and a network. Use it
  from macOS before CI. Run `git add` on a new file first: untracked files are not copied in.
* The `gate-contract` job in `.github/workflows/ci.yml` runs the examples in
  `examples/deterministic`, `examples/retrieval` and `examples/agent-tools` through
  `evalproof run` and `evalproof compare --gate` and asserts the exit codes. Run those
  commands by hand when you change gate behaviour.
* The docs build takes the "Last updated" dates from git history. A page that git has not seen
  yet can warn, and `--strict` fails on a warning: commit the page, then build.

## Layout

| Path | What is there |
|---|---|
| `src/evalproof/` | the package: `cli.py` (parser and commands), `gate.py` (exit codes), `config.py`, `runner.py`, `scorers/`, `providers/`, `adapters/`, `importers/`, `scaffold.py` (`init`) |
| `src/evalproof/templates/` | the files `init` writes: the starter suite and cases, the CI workflows, `SKILL.md` and `AGENTS.md` |
| `tests/` | pytest |
| `docs/` | the MkDocs site (`mkdocs.yml` holds the nav); `docs/adr/` holds the decision records and `docs/specs/` the specs |
| `examples/` | runnable examples; CI gates on three of them |
| `app/` | the Streamlit demo |
| `npm/` | the `npx evalproof` launcher; its version must equal `src/evalproof/_version.py` |
| `action.yml` | the GitHub Action |

## Conventions

* Ruff with line length 100 (rules `E F W I B UP SIM RUF`) and `mypy --strict` on `src`. The
  version is written only in `src/evalproof/_version.py`.
* Tests never call a real model. Provider code is tested against `httpx.MockTransport`. The
  live tests in `tests/test_live_agent_cli.py` are skipped unless `EVALPROOF_LIVE=1`; do not set
  it.
* Statistical code needs a known-value test: work the number out by hand and put the
  arithmetic in the test's comment (see `tests/test_stats.py`).
* A change to the gate's behaviour needs an ADR in `docs/adr/` and a case in the offline example
  that shows it.
* The public surface follows `docs/about/versioning.md`: names in `evalproof.__all__`, commands,
  flags, exit codes, `--json` fields, suite keys, the run file format and dataset fields.
  Additions go in a minor release. An incompatible change needs a deprecation first and a major
  release. Note every user-visible change in `CHANGELOG.md` under `## [Unreleased]`.
* A new command or flag needs a row in `docs/reference/commands.md`: `tests/test_docs_reference.py`
  compares that page with the argument parser, and it does the same for config keys, run-file
  fields and exit codes. Update the docs in the same change.
* Files must work on Windows: pass `encoding="utf-8"` and an explicit `newline` when you read or
  write text, and do not create symlinks. `src/evalproof/_fsutil.py` and
  `src/evalproof/_process.py` hold the helpers. `tests/test_init.py` is skipped on Windows
  because it fakes CLIs with shell scripts, so put init tests that must run there in a file
  that is not skipped, as `tests/test_agents_md.py` does.
* Run files under `examples/*/runs/` are recorded output and are kept byte for byte as
  evalproof wrote them; the pre-commit hooks leave them alone.
* Numbers in `MEASUREMENTS.md` come from real runs only, with the command, the machine and the
  model digests recorded next to them.

## The `init` templates

`evalproof init` writes `src/evalproof/templates/AGENTS.md` into a user's repository. It sits
between two markers, and `init` replaces only the text between them. Keep these true when you
edit it:

* It starts with `<!-- evalproof:begin -->` and ends with `<!-- evalproof:end -->`, each once.
* Every command and flag in it exists. `tests/test_agents_md.py` checks that against the
  parser, so a rename fails the test.
* It states only what the docs or the code confirm. The skill in `templates/SKILL.md` keeps to
  checking a change and points to this section for everything else.
* If you change a command, a flag or an exit code, update the template in the same change.

## Releasing

Releases are the maintainer's. Do not tag or publish. The steps are in `CONTRIBUTING.md`.
