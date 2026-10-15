# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

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
- Two examples: an offline ticket router (CI's gate contract) and a support assistant for
  a fictional product judged by a local model, with measurements.
