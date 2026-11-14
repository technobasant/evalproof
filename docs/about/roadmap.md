---
description: What evalproof plans to build next, release by release, and what is left for later. The plan is subject to change.
---

# Roadmap

What is planned for the next releases of evalproof. It is for people deciding whether a feature they need is coming. The plan is subject to change: an item can move, shrink or be dropped, and nothing here is a commitment to a date.

What has already shipped is in the [changelog](changelog.md). Version 0.8 added importers for garak, Google ADK, `claude plugin eval` and promptfoo red-team runs, the Inspect tool trace, an OpenTelemetry trace adapter, rubric sets and the `facts` scorer.

## 0.9: a coding-agent add-on

Suites for coding agents: tasks that run against a repository, and checks that read whether the tests pass.

## Later

* **Stratified gold sampling and per-tag calibration.** Draw the gold sample so that each tag is represented, and calibrate the judge per tag rather than over the whole run.
* **A published study** comparing models on an agent suite.

## Next steps

* Try what exists today: [Quickstart](../getting-started/quickstart.md).
* Ask for a feature or report a problem: [open an issue](https://github.com/technobasant/evalproof/issues).
