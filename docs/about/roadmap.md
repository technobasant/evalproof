---
description: What evalproof plans to build next, release by release, and what is left for later. The plan is subject to change.
---

# Roadmap

What is planned for the next releases of evalproof. It is for people deciding whether a feature they need is coming. The plan is subject to change: an item can move, shrink or be dropped, and nothing here is a commitment to a date.

What has already shipped is in the [changelog](changelog.md). Version 1.0 declared the public surface stable ([Versioning and support](versioning.md)); new features now arrive in minor releases without breaking what works.

## Next: find what is not tested (1.3)

* **`evalproof scan`.** Read a repository, with no model call, for the places that call a model, the prompts, the tools an agent can use, and the test cases and fixtures that already exist.
* **A gap report.** Model call sites with no cases, tools no case checks, and multi-turn code with no multi-turn case.
* **Draft cases.** A model may write new inputs for those gaps, tagged `draft`. Expected answers come only from your code, docs, tests or a person, never from a model.
* **More places to run.** `init --platform bitbucket`, a pre-push hook, and `run --tags` to run a subset.

## Next: more frameworks (1.4)

* **Adapters** for Pydantic AI, LlamaIndex, CrewAI, DSPy, Haystack, AutoGen and Semantic Kernel, next to the ones for LangChain, ADK, the Claude Agent SDK and the OpenAI Agents SDK.
* **TypeScript.** Helpers for testing a TypeScript agent through a `command:` or HTTP target, and `scan` support for LangChain.js and the Vercel AI SDK.

## Later

* **A coding-agent add-on.** Suites for coding agents: tasks that run against a repository, and checks that read whether the tests pass.
* **Stratified gold sampling and per-tag calibration.** Draw the gold sample so that each tag is represented, and calibrate the judge per tag rather than over the whole run.
* **A published study** comparing models on an agent suite.

## Next steps

* Try what exists today: [Quickstart](../getting-started/quickstart.md).
* Ask for a feature or report a problem: [open an issue](https://github.com/technobasant/evalproof/issues).
