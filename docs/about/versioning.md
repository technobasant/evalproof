---
description: How evalproof numbers its releases, which versions get security fixes, which Python versions it supports and how a release is built and published.
---

# Versioning and support

How evalproof is versioned, which versions are supported, and how a release reaches PyPI. It is for people deciding whether to pin a version, and for teams that need to know what they can rely on between releases.

## Version numbers

The project follows [Semantic Versioning](https://semver.org/), as the [changelog](changelog.md) states. It is below 1.0, so the public surface is not frozen.

| Release | What to expect | Example |
|---|---|---|
| Minor (0.x.0) | new features, and sometimes breaking changes, listed under "Breaking" in the changelog | 0.5.0 and 0.6.0 each list breaking changes |
| Patch (0.x.y) | fixes and documentation, and small additions that keep earlier run files working | 0.5.1 added `rehash`; 0.5.2 changed no behaviour |

Pin the version in CI, and read [Upgrading](upgrading.md) before moving to a new minor release. `evalproof init` pins the current version in the workflows it writes.

## Supported versions

Security fixes go into the latest minor release. While evalproof is below 1.0, that is the only supported line.

| Version | Supported |
|---|---|
| 0.7.x | yes |
| < 0.7 | no |

Report a vulnerability privately, as the [security policy](https://github.com/technobasant/evalproof/blob/main/SECURITY.md) describes. [Security and data handling](security.md) covers what evalproof stores and sends.

## Supported Python versions

evalproof requires Python 3.11 or newer. The package declares support for 3.11, 3.12, 3.13 and 3.14, and its development status is Beta.

It has four runtime dependencies, each with a minimum version:

| Package | Minimum |
|---|---|
| `httpx` | 0.27 |
| `numpy` | 1.26 |
| `pydantic` | 2.7 |
| `pyyaml` | 6.0 |

## What stays compatible

* **Run files.** Every field added after 0.1 has a default, so an older run file still loads. A newer evalproof reads older files; what an old file lacks (per-case hashes, a suite hash, trials) limits what you can do with it, as [Upgrading](upgrading.md#run-files-from-older-versions) lists. The [run file format](../reference/run-file.md) page gives every field.
* **The response cache.** Cache keys come from the provider, model, endpoint, messages and generation parameters. When a release changes how a key is built, the changelog says so: 0.5.0 changed the keys of agent CLI calls once, so cached replies from 0.4.x are not reused. Trial 0 of a repeated run keeps the key it always had.
* **Suites.** Since 0.5, an unknown key in a suite is an error, so a misspelt or removed key fails loudly rather than being ignored.

## How a release is built and published

A release starts when a GitHub release is published. The `release` workflow then runs three jobs in order:

1. **Build and check.** It confirms the release tag matches the package version, runs the test suite, builds the distributions once, runs `twine check --strict` on them, and installs the wheel into a clean environment to confirm that `evalproof --version` prints the tagged version.
2. **Publish to PyPI.** It uploads the files that were built, using [trusted publishing](https://docs.pypi.org/trusted-publishers/): the job holds a short-lived identity token and no stored PyPI token exists. PyPI attaches [PEP 740](https://peps.python.org/pep-0740/) attestations to each file.
3. **Attach to the GitHub release.** It adds a build-provenance attestation for the same files and uploads them to the release.

The documentation site is rebuilt and published to GitHub Pages on each published release, so it matches the latest release.

## Next steps

* What changed between releases: [Upgrading](upgrading.md) and the [changelog](changelog.md).
* What evalproof stores and sends: [Security and data handling](security.md).
* What is planned next: [Roadmap](roadmap.md).
