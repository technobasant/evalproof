---
description: How evalproof numbers its releases, which versions get security fixes, which Python versions it supports and how a release is built and published.
---

# Versioning and support

How evalproof is versioned, which versions are supported, and how a release reaches PyPI. It is for people deciding whether to pin a version, and for teams that need to know what they can rely on between releases.

## Version numbers

The project follows [Semantic Versioning](https://semver.org/), as the [changelog](changelog.md) states. Since 1.0 the public surface below is stable: it changes incompatibly only in a major release.

| Release | What to expect |
|---|---|
| Major (x.0.0) | removal of something deprecated, or another incompatible change to the public surface, listed under "Breaking" in the changelog |
| Minor (1.x.0) | new features, and deprecations; anything that worked keeps working |
| Patch (1.x.y) | fixes and documentation |

Pin the version in CI, and read [Upgrading](upgrading.md) before moving to a new major release. `evalproof init` pins the current version in the workflows it writes.

## The public surface

These are covered by the guarantee:

* **Python names** in `evalproof.__all__` and `evalproof.adapters.__all__`, as the [API reference](../reference/api.md) documents them. Any other module or name is internal and can change in any release.
* **Commands**: every `evalproof` command, its flags and its [exit codes](../reference/exit-codes.md), and the fields of every `--json` output.
* **Suite files**: every [configuration](../reference/configuration.md) key and what it means.
* **Run files**: the [run file format](../reference/run-file.md). Each run file records its `format_version`; a release that writes a new format still reads the older ones.
* **Datasets**: the case fields evalproof reads.

These are not:

* **Anything marked experimental**: the `opencode:`, `cursor:` and `copilot:` providers, the Cursor sandbox, the `adk` and `claude-plugin-eval` importers, the OpenTelemetry adapter, the `fromVercelAI` and `fromLangChain` converters of the npm package, the LlamaIndex, CrewAI, Pydantic AI, DSPy, Haystack, AutoGen and Semantic Kernel adapters, the `scan` and `draft` commands (what they find, the JSON shape, the fields of a drafted case and their flags), and the `import-dataset` formats `deepeval`, `ragas`, `langsmith`, `openai-evals`, `inspect`, `braintrust`, `langfuse` and `phoenix` (written from each tool's documented fields, not yet checked against files exported by each tool). They can change in a minor release until they are declared stable. The adapters and importers warn when used, and so do the providers, `scan`, `draft` and the converters (one line on stderr the first time they are called); the `import-dataset` formats are marked in [Import test cases](../guides/import-test-cases.md).
* **What 1.0.0 added on top of that** is experimental too, without a warning, until it has been used in more projects: the `command:` target's input and output, the [JUnit reports](../reference/junit.md) (`--junit`) and the names of their test cases, the `checks` list in the gate's `--json` output, `init --platform gitlab` and `init --platform bitbucket`, the action's `pr-comment` input, the npm launcher, the npm package's helper library (`runCommand` and `serve`, which [Testing a TypeScript or JavaScript system](../guides/typescript.md) describes), `init --node` and the container image's contents. They can change in a minor release, and the changelog says when.
* **The coding-agent files** `init` writes: the text of the evalproof section of `AGENTS.md`, the skill's text and `init --agent-files-only`. The markers around the section and the rule that `init` never changes anything outside them are stable.
* **Human-readable output**: the terminal and Markdown reports can be reworded or rearranged in any release. Read `--json` output or run files from scripts.
* **The response cache**: a release may change how a cache key is built, which makes cached replies miss once; the changelog says when.
* **Numbers that come from fixing a statistical bug**: a fix that changes a verdict on the same run files goes into a minor release, listed under "Changed", never a patch.

## Deprecation

Before something on the public surface is removed or changed incompatibly, it is deprecated: it keeps working for at least one minor release, it warns when used (a `DeprecationWarning` from Python, a line on stderr from a command or a suite key), and the changelog lists it under "Deprecated" with what replaces it. Removal happens only in the next major release.

## Supported versions

Fixes go into the latest minor release. Security fixes also go into the previous minor release for 90 days after a new minor is published.

| Version | Supported |
|---|---|
| 1.4.x | yes |
| 1.3.x, 1.2.x, 1.1.x, 1.0.x | security fixes until 2027-02-13 |
| < 1.0 | no |

Report a vulnerability privately, as the [security policy](https://github.com/technobasant/evalproof/blob/main/SECURITY.md) describes. [Security and data handling](security.md) covers what evalproof stores and sends.

## Supported Python versions

evalproof requires Python 3.11 or newer. The package declares support for 3.11, 3.12, 3.13 and 3.14 on Linux, macOS and Windows; CI runs the test suite on all three. Its development status is Production/Stable. Support for a Python version is dropped only in a minor release, after the version reaches its end of life, and the changelog says so.

It has four runtime dependencies, each with a minimum version:

| Package | Minimum |
|---|---|
| `httpx` | 0.27 |
| `numpy` | 1.26 |
| `pydantic` | 2.7 |
| `pyyaml` | 6.0 |

## What stays compatible

* **Run files.** Every field added after 0.1 has a default, so an older run file still loads. Since 1.0 a run file records its `format_version`. From 1.0 on, an evalproof that meets a file with a newer format than it reads refuses it with an error asking for an upgrade, rather than dropping the fields it does not know; releases before 1.0 do not check. A newer evalproof reads older files; what an old file lacks (per-case hashes, a suite hash, trials) limits what you can do with it, as [Upgrading](upgrading.md#run-files-from-older-versions) lists. The [run file format](../reference/run-file.md) page gives every field.
* **The response cache.** Cache keys come from the provider, model, endpoint, messages and generation parameters. When a release changes how a key is built, the changelog says so: 0.5.0 changed the keys of agent CLI calls once, so cached replies from 0.4.x are not reused. Trial 0 of a repeated run keeps the key it always had.
* **Suites.** Since 0.5, an unknown key in a suite is an error, so a misspelt or removed key fails loudly rather than being ignored.

## How a release is built and published

A release starts when a GitHub release is published. The `release` workflow then runs five jobs:

1. **Build and check.** It confirms the release tag matches the package version, runs the test suite, builds the distributions once, runs `twine check --strict` on them, and installs the wheel into a clean environment to confirm that `evalproof --version` prints the tagged version.
2. **Publish to PyPI.** It uploads the files that were built, using [trusted publishing](https://docs.pypi.org/trusted-publishers/): the job holds a short-lived identity token and no stored PyPI token exists. PyPI attaches [PEP 740](https://peps.python.org/pep-0740/) attestations to each file.
3. **Attach to the GitHub release.** It adds a build-provenance attestation for the same files and uploads them to the release.
4. **Publish the container image.** It starts only after the PyPI upload succeeded, builds an image for `linux/amd64` and `linux/arm64` that installs the wheel from the first job (the file that is on PyPI, not a second build), and pushes it to `ghcr.io/technobasant/evalproof` with the tags `<version>`, `<major>.<minor>` and `latest`. A pre-release gets only its exact version tag. The job then attaches a build-provenance attestation to the pushed image, stored with it in the registry. See [Docker](../getting-started/install.md#docker).

5. **Publish the npm launcher.** Also after the PyPI upload, it publishes the `npm/` package to npm with [trusted publishing](https://docs.npmjs.com/trusted-publishers): no stored npm token exists, and npm attaches a provenance statement. The job first confirms that `npm/package.json` carries the tagged version.

Jobs 3, 4 and 5 run side by side once the files are on PyPI.

The documentation site is rebuilt and published to GitHub Pages on each published release, so it matches the latest release.

## Next steps

* What changed between releases: [Upgrading](upgrading.md) and the [changelog](changelog.md).
* What evalproof stores and sends: [Security and data handling](security.md).
* What is planned next: [Roadmap](roadmap.md).
