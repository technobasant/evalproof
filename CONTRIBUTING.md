# Contributing

Thanks for helping. Bugs and ideas go in [issues](https://github.com/technobasant/evalproof/issues);
security problems go to [private reporting](SECURITY.md), never a public issue. Everyone
taking part agrees to the [code of conduct](CODE_OF_CONDUCT.md).

## Setup

```bash
uv sync
uvx pre-commit install        # ruff format and check, YAML/TOML checks, on every commit
```

Before a pull request, run what CI runs:

```bash
uv run ruff format . && uv run ruff check . && uv run mypy src
uv run pytest --cov
```

evalproof supports Python 3.11 to 3.14, and CI tests all four.

## Testing on Linux in Docker

If you work on macOS, `scripts/test-in-docker.sh` runs the whole test suite in Linux containers
before CI does. It needs Docker and a network connection.

```bash
scripts/test-in-docker.sh                       # all six targets, one after another
scripts/test-in-docker.sh --only py313,alpine313
scripts/test-in-docker.sh --list
```

The targets are Python 3.11, 3.12, 3.13 and 3.14 on Debian, Python 3.13 on Alpine (musl libc
instead of glibc), and Python 3.13 on `linux/amd64`, which runs under emulation on an Apple
silicon Mac and is slow. Each container gets a read-only snapshot of your working tree: the
files git tracks, with your uncommitted edits. Run `git add` on a new file for it to be tested;
untracked files, `.git` and `.venv` are left out. In the container, root installs git and uv, then
drops to an unprivileged user that runs `uv sync --frozen` and the tests, because root skips
permission checks.

The script prints the directory with each target's full log, then a table of the Python
version, the result, the passed, failed and skipped counts and the time per target. It exits 1 if
any target failed. Delete the log directory when you are done. `--target 'NAME IMAGE [PLATFORM]'`
adds your own target in place of the defaults, for example another Python image.

To build the release image from your working tree, run `docker build -t evalproof .`. The
comments at the top of the `Dockerfile` explain how it takes its wheel.

## What a change needs

* Statistical code needs a known-value test: work the expected number out by hand and put
  the arithmetic in the test's comment (see `tests/test_stats.py`).
* Provider code is tested against `httpx.MockTransport`; tests never call a real model.
* A change to the gate's behaviour needs an ADR in `docs/adr/` and a case in the offline
  example that shows it.
* A change to the public surface (names in `evalproof.__all__`, commands, flags, exit codes,
  `--json` fields, suite keys, the run file format) follows the
  [versioning policy](docs/about/versioning.md): additions go in a minor release, and an
  incompatible change needs a deprecation first and a major release. Note it in `CHANGELOG.md`.
* Measurements in `MEASUREMENTS.md` come from real runs only, with the command, the
  machine and the model digests recorded next to them.

## Releasing (maintainer)

1. Set the version in `src/evalproof/_version.py`, the only place it is written, and move
   the `CHANGELOG.md` entries under it with the date. Update `CITATION.cff` and set the same
   version in `npm/package.json`, the launcher that `npx evalproof` runs: a test fails when the
   two differ.
2. Commit, push to `main`, and wait for CI.
3. Tag `vX.Y.Z` on that commit and publish a GitHub release whose notes are the changelog
   section. The `release` workflow checks the tag against the version, runs the tests,
   builds once, publishes to PyPI with trusted publishing, attaches the files and their
   build-provenance attestation to the release, and publishes the container image to
   `ghcr.io/technobasant/evalproof`. After the first image is pushed, check in the package
   settings on GitHub that the package is public, or `docker pull` asks for a login. When the
   repository variable `NPM_PUBLISH` is `true`, it also publishes `npm/` to npm with trusted
   publishing.
