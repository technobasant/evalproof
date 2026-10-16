# Security policy

## Supported versions

Security fixes go into the latest minor release. While evalproof is below 1.0, that is the
only supported line.

| Version | Supported |
|---|---|
| 0.2.x | yes |
| < 0.2 | no |

## Reporting a vulnerability

Please do not open a public issue. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/technobasant/evalproof/security/advisories/new),
or by email to contact@basantbhattarai.com.np. Include the version, a minimal suite or run
file that reproduces it, and what an attacker gains.

You will get an acknowledgement within seven days. A confirmed issue gets a fix in a patch
release and a published advisory crediting you, unless you prefer otherwise.

## What evalproof trusts

Knowing the trust boundaries helps tell a vulnerability from intended behaviour:

* **A suite file is code.** A `python:module:function` target imports and runs that
  function, and an `https://` target sends every case to that endpoint. Run suites you
  trust, as you would a test file.
* **API keys come from the environment** (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and are
  sent only to the provider's API. They are not written to run files, caches or reports;
  a path where one is would be a vulnerability.
* **The response cache stores model outputs** under the suite's `cache_dir`. Treat it like
  the data it was computed from.
* **Run files are data.** `compare`, `calibrate` and `report` parse them with Pydantic and
  never execute anything from them.

## Releases

Releases are built and published to PyPI by GitHub Actions with trusted publishing, so no
long-lived token exists. Each file on PyPI carries a signed provenance attestation, and the
GitHub release carries a build-provenance attestation for the same files.
