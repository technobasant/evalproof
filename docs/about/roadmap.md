---
description: What evalproof plans to build next, release by release, and what is left for later. The plan is subject to change.
---

# Roadmap

What is planned for the next releases of evalproof. It is for people deciding whether a feature they need is coming. The plan is subject to change: an item can move, shrink or be dropped, and nothing here is a commitment to a date.

What has already shipped is in the [changelog](changelog.md). Version 1.0 declared the public surface stable ([Versioning and support](versioning.md)); new features now arrive in minor releases without breaking what works.

## Next: a coding-agent add-on

Suites for coding agents: tasks that run against a repository, and checks that read whether the tests pass.

## Next: systems written in other languages

A `command:` target that runs any program as the system under test, so an agent written in TypeScript or another language can be evaluated without a Python wrapper.

## Later

* **Stratified gold sampling and per-tag calibration.** Draw the gold sample so that each tag is represented, and calibrate the judge per tag rather than over the whole run.
* **A published study** comparing models on an agent suite.

## Next steps

* Try what exists today: [Quickstart](../getting-started/quickstart.md).
* Ask for a feature or report a problem: [open an issue](https://github.com/technobasant/evalproof/issues).
