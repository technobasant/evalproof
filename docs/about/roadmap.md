---
description: What evalproof plans to build next, release by release, and what is left for later. The plan is subject to change.
---

# Roadmap

What is planned for the next releases of evalproof. It is for people deciding whether a feature they need is coming. The plan is subject to change: an item can move, shrink or be dropped, and nothing here is a commitment to a date.

What has already shipped is in the [changelog](changelog.md). Version 1.0 declared the public surface stable ([Versioning and support](versioning.md)); new features now arrive in minor releases without breaking what works.

## Next: more frameworks (1.4)

* **Adapters** for Pydantic AI, LlamaIndex, CrewAI, DSPy, Haystack, AutoGen and Semantic Kernel, next to the ones for LangChain, ADK, the Claude Agent SDK and the OpenAI Agents SDK.
* **TypeScript.** A helper library in the `evalproof` npm package for testing a TypeScript agent through a `command:` or HTTP target (`runCommand`, `serve`), converters that turn a Vercel AI SDK or LangChain.js result into a trace, `init --node`, and `scan` support for the tools and agents of LangChain.js and the Vercel AI SDK. It is experimental when it ships. See [Testing a TypeScript or JavaScript system](../guides/typescript.md).

## Later

* **A coding-agent add-on.** Suites for coding agents: tasks that run against a repository, and checks that read whether the tests pass.
* **Stratified gold sampling and per-tag calibration.** Draw the gold sample so that each tag is represented, and calibrate the judge per tag rather than over the whole run.
* **A published study** comparing models on an agent suite.

## Next steps

* Try what exists today: [Quickstart](../getting-started/quickstart.md).
* Ask for a feature or report a problem: [open an issue](https://github.com/technobasant/evalproof/issues).
