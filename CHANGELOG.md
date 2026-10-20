# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added

- `examples/retrieval`: an offline retrieval example for recall@k, MRR and nDCG, with a
  regression that recall@5 cannot see and MRR and nDCG@5 can. Contributed by
  @HarshRajSinghania in #5. CI asserts its gate exits 1 on the regression and 0 on no change.
- MEASUREMENTS §8: the support-qa judge with thinking on (#3), with its runs, suite and
  analysis in `measure.py`. Against labels audited case by case it passes no wrong answer
  and κ rises, at 6.7× the judge time; the gate's decision does not change.

### Fixed

- `examples/support-qa/cases.jsonl`: the q21 and q22 references named Enterprise, which the
  product facts do not give SSO or an admin console (#7). The original set is kept as
  `cases.v1.jsonl` so the recorded runs stay reproducible; the runs judged on the revised set
  are in `runs/v2/`, and MEASUREMENTS §8 reports both.

## [0.2.1] — 2026-10-16

### Fixed

- The PyPI and Python badges on the 0.2.0 PyPI page read "package or version not found".
  PyPI's image proxy fetched them about a minute after the first upload, before the badge
  service could see the package, and keeps that answer for 12 to 24 hours. The PyPI page
  now leaves those two badges out, since it shows the version and supported Pythons
  itself, and the README's copies use new URLs so no cache serves the old images. CI fails
  if a PyPI-data badge reaches the PyPI description again.

## [0.2.0] — 2026-10-16

The first release on PyPI: `pip install evalproof`.

### Added

- A supported Python API: `from evalproof import compare_runs, decide, calibrate, ...`.
  The names in `evalproof.__all__` are the public surface; other modules are internal.
- Python 3.14 support, tested in CI alongside 3.11 to 3.13.
- Releases publish to PyPI from GitHub Actions with trusted publishing, so no token is
  stored anywhere. Each file on PyPI carries a PEP 740 attestation, and the GitHub release
  carries the same files with a build-provenance attestation.
- `SECURITY.md` with private vulnerability reporting and the trust boundaries, backed by a
  test that runs both hosted providers with a placeholder key and finds it in no file written.
- `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), issue and pull-request templates,
  `CITATION.cff`, `CODEOWNERS`, Dependabot for actions and Python dependencies with a
  three-day cooldown, and a pre-commit configuration.

### Changed

- The version is written once, in `src/evalproof/_version.py`; the build reads it.
- The README published to PyPI rewrites relative links to GitHub and replaces the Mermaid
  diagram with a link, so the PyPI page has no broken links. CI checks this on every
  commit, along with `twine check --strict` and a clean install of the built wheel.
- CI pins every action to a commit SHA, checks out without persisted credentials, sets
  timeouts and cancels superseded pull-request runs, and audits the workflows with zizmor.
  The composite action pins `setup-uv` the same way.
- Package metadata: documentation, source, changelog and live-demo links; classifiers for
  typing, Python 3.14 and beta status.

## [0.1.2] — 2026-10-15

### Added

- `app/`: an interactive Streamlit app, live at https://evalproof.streamlit.app.
  *Check a drop* starts from four scenarios (the recorded run, the same rates on 4× the
  cases, a small but certain drop, a twenty-case suite) or from sliders, and shows the
  gate's verdict, every paired case as a grid, the coin-flip distribution behind the exact
  McNemar p-value, and the suite size at which the same drop would become significant.
  *The recorded run* replays the committed `examples/support-qa` runs: select any of the
  twelve changed answers to read both versions side by side, add a latency budget, and
  calibrate the judge on a gold sample of chosen size. It runs the library from `src/`
  through the same `compare_runs` and `decide` calls as the CLI; CI installs the app's own
  requirements and drives every scenario.

### Fixed

- `evalproof compare` raised OverflowError when a metric had more than about 1,030
  discordant pairs: the exact McNemar test summed `comb(n, i) * 0.5**n` in floating
  point, and `comb(n, i)` outgrows the largest float. The binomial tail is now summed
  as integers and divided once, so it stays exact at any size.

## [0.1.1] — 2026-10-13

### Fixed

- The Wilson interval now returns exactly 0 at k=0 and exactly 1 at k=n. Computing
  the boundaries by subtraction left platform-dependent rounding (2.8e-17 on x86_64
  Linux, 0.0 on arm64 macOS).

### Changed

- The three-day `exclude-newer` resolution guard is declared in `[tool.uv]`, so
  `uv.lock` means the same thing on every machine and in CI.
- CI pins uv to the version that wrote `uv.lock`.

## [0.1.0] — 2026-10-13

First release.

### Added

- Suites in YAML: a JSONL golden set, a system under test, scorers, an optional judge,
  calibration settings and a gate.
- Systems under test: Ollama, Anthropic and OpenAI models with a prompt template; Python
  functions; HTTP endpoints. System failures are recorded per case, never fatal.
- Scorers: exact match, contains, regex with accept/reject answer keys, a JSON Schema
  subset, numeric tolerance, recall@k, MRR, nDCG, and an LLM judge with a strict JSON
  verdict, one parse retry and an explicit undecided state.
- Pairwise judging with order swap; position-inconsistent verdicts become ties.
- Judge calibration: confusion matrix, TPR, FPR, Cohen's κ, and prevalence-corrected pass
  rates by adjusted classify-and-count (bootstrap interval) and prediction-powered
  inference (normal interval).
- Paired comparison of two runs: exact McNemar for pass/fail, paired bootstrap for scores.
- A CI gate that fails only on a drop that is both beyond `max_drop` and significant at
  `alpha`, plus p95-latency and cost budgets, with distinct exit codes.
- Content-addressed response cache; runs record the Ollama version and model digests.
- Markdown reports for GitHub step summaries, JSON reports, and a composite GitHub Action.
- Two examples: an offline ticket router (CI's gate contract) and a product support
  assistant judged by a local model, with measurements.

[Unreleased]: https://github.com/technobasant/evalproof/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/technobasant/evalproof/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/technobasant/evalproof/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/technobasant/evalproof/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/technobasant/evalproof/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/technobasant/evalproof/releases/tag/v0.1.0
