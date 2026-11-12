# Changelog

All notable changes are listed here. The project follows [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-11-12

### Added

- `evalproof import` and `evalproof.import_run`: per-case results from promptfoo (JSON and
  JSONL), Inspect (`.json` and `.eval` logs), DeepEval, lm-evaluation-harness
  (`samples_*.jsonl`) and generic long or wide `jsonl` and `csv` files become a run that
  `compare` and the gate read. Case ids are stable across runs, system errors score as
  failures, repeats collapse by `--epochs`, and NaN or infinite scores are refused with the
  case and metric named. Raises `evalproof.ImportFormatError` (a `ValueError`; exit 2 on the
  command line).
- Gating several metrics: `gate.metrics` in the suite or `--metric` repeated on
  `compare`, with Holm-adjusted p-values (`evalproof.holm`). The decision carries
  `adjusted_p`, the report gains a Holm-adjusted p column, and a single metric behaves as
  before. ADR 0003 has the reasoning.
- The composite action's `metric` input takes several names, separated by newlines or commas.
- `examples/retrieval`: an offline retrieval example for recall@k, MRR and nDCG, with a
  regression that recall@5 cannot see and MRR and nDCG@5 can. Contributed by
  @HarshRajSinghania in #5. CI asserts its gate exits 1 on the regression and 0 on no change.
- MEASUREMENTS §8: the support-qa judge with thinking on (#3), with its runs, suite and
  analysis in `measure.py`. Against labels audited case by case it passes no wrong answer
  and κ rises, at 6.7× the judge time; the gate's decision does not change.
- A provider profile table (`providers/profiles.py`) behind one OpenAI-compatible adapter:
  `openai`, `azure`, `gemini`, `mistral`, `groq`, `together`, `fireworks`, `deepseek`, `xai`,
  `openrouter`, `cerebras`, the local servers `vllm`, `lmstudio` and `llamacpp`, and
  `openai-compatible` for any other server (set `base_url`). Each profile carries its base
  URL, key variable, auth header, token-limit field and a link to the provider's docs.
  `OpenAIProvider` stays importable.
- `TargetConfig.api_key_env` names the environment variable that holds the key, and
  `TargetConfig.headers` adds request headers to every HTTP provider; suite headers win over
  a provider's own, and neither is part of the cache key or written to a run. A named key
  variable that is not set is an error rather than a 401.
- Anthropic: a `refusal` stop reason is recorded as a case error naming its category, and so
  is `max_tokens` reached before any text, with a hint to raise `params.max_tokens`. The
  OpenAI-compatible adapter does the same for `finish_reason` `length` with no text, and
  for `content_filter`, and reads replies that arrive as a list of typed content parts.
- A test pins the cache key of an existing Ollama call, so a change to it cannot slip in.

### Changed

- Hosted providers (everything except Ollama and the local servers) send `temperature` and
  `seed` only when the suite sets them. Before, `openai` sent `temperature: 0` and
  `seed: 0` on every call, which reasoning models reject; now an unset value is left to the
  provider's default. A hosted run that relied on the implicit `temperature: 0` should set
  it in `params`. Ollama and the local servers still send both, so local runs repeat as
  before and keep their cache keys; `GenerationParams` defaults are unchanged. A hosted call
  that now leaves a value out gets a new cache key, so a reply cached at the old implicit
  `temperature: 0` is not reused for it. To point `openai` at an older local server through
  `base_url`, use `openai-compatible` or a local profile instead, which send both values.
- `openai` sends the token limit as `max_completion_tokens`, which OpenAI's reasoning models
  require (they reject `max_tokens`), and `mistral` sends its seed as `random_seed`. The
  default limit is still 512.
- The llama.cpp profile defaults to port 9931, the one its server documents now; for an older
  build on 8080, set `base_url`.

### Fixed

- Anthropic calls to current Claude models failed with HTTP 400, because `temperature` was
  sent on every request. It goes out only when the suite sets it, and no seed is sent, since
  the Messages API has none.
- A gate built without validation and given no metric passed by gating nothing; it now
  cannot gate (exit 4), and `compare --metric ""` is a usage error (exit 2).
- Found in review before release, in the new importers: two promptfoo prompts whose first 60
  characters matched merged into one system (the label is no longer truncated, and two
  prompts that share a label are an error), promptfoo output with repeated tests but no
  `repeatIndex` now warns that it reads as separate cases, Inspect message ids no longer
  change the dataset hash, a `null` promptfoo `testIdx` no longer crashes, and under
  `--epochs any` or `mean` a case with one good repeat is no longer marked as an error.
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

[Unreleased]: https://github.com/technobasant/evalproof/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/technobasant/evalproof/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/technobasant/evalproof/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/technobasant/evalproof/compare/v0.1.2...v0.2.0
[0.1.2]: https://github.com/technobasant/evalproof/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/technobasant/evalproof/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/technobasant/evalproof/releases/tag/v0.1.0
