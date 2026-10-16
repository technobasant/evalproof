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

## What a change needs

* Statistical code needs a known-value test: work the expected number out by hand and put
  the arithmetic in the test's comment (see `tests/test_stats.py`).
* Provider code is tested against `httpx.MockTransport`; tests never call a real model.
* A change to the gate's behaviour needs an ADR in `docs/adr/` and a case in the offline
  example that shows it.
* A change to a name in `evalproof.__all__` is a change to the public API: note it in
  `CHANGELOG.md`, and before 1.0 expect it to need a minor version.
* Measurements in `MEASUREMENTS.md` come from real runs only, with the command, the
  machine and the model digests recorded next to them.

## Releasing (maintainer)

1. Set the version in `src/evalproof/_version.py`, the only place it is written, and move
   the `CHANGELOG.md` entries under it with the date. Update `CITATION.cff`.
2. Commit, push to `main`, and wait for CI.
3. Tag `vX.Y.Z` on that commit and publish a GitHub release whose notes are the changelog
   section. The `release` workflow checks the tag against the version, runs the tests,
   builds once, publishes to PyPI with trusted publishing, and attaches the files and their
   build-provenance attestation to the release.
