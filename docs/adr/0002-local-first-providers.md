---
description: Records the decision to default to local Ollama, plain HTTP providers and a content-addressed cache.
---

# 2. Local-first providers over plain HTTP

Status: accepted · 2026-10

## Context

An evaluation that only runs with an API key and a budget gets run less often, and its
results cannot be reproduced by anyone without the same account. Hosted judges also change
under you: a model alias can point at new weights between two runs.

## Decision

* **Ollama is the default provider.** A local model with `temperature: 0` and a fixed
  `seed` gives repeatable verdicts at zero marginal cost, and the run records the Ollama
  version and the model digest so the result can be tied to exact weights.
* **Anthropic and OpenAI are optional**, reached over plain HTTP with `httpx` and a key
  from the environment. No vendor SDK is a dependency.
* **Every call goes through a content-addressed cache** keyed on provider, model,
  endpoint, messages and generation parameters. An unchanged suite re-runs from cache;
  changing one prompt re-runs only the calls it affects.
* Reasoning models run with thinking off by default (`think: false`) for speed and
  repeatability; a suite can turn it on.

## Consequences

* A local 8B judge is weaker than a frontier model. That is acceptable precisely because
  ADR 1 measures and corrects the judge rather than trusting it — a weaker, calibrated
  judge beats a stronger, uncalibrated one for gating.
* Determinism is only as good as the backend's seeding. The measurements re-run a sample
  without the cache to check it.
* Cost reporting needs a price table in the suite for hosted models; local runs report $0.

## Addendum, 2026-11-12: a profile table for the other providers

The decision above stands: plain HTTP, a key from the environment, no vendor SDK. What
changed is how many providers sit behind it. Ollama and Anthropic keep their native
adapters; everything that speaks the OpenAI Chat Completions request (OpenAI, Azure, Gemini,
Mistral, Groq, Together, Fireworks, DeepSeek, xAI, OpenRouter, Cerebras, and the local
vLLM, LM Studio and llama.cpp servers, plus any other server through `openai-compatible`) is
one adapter driven by a table of profiles in `providers/profiles.py`.

Why a table, not LiteLLM or each vendor's SDK:

* **No new dependency surface.** An evaluation tool that gates CI runs with the team's
  provider keys in its environment, which is exactly what a compromised dependency goes
  after. LiteLLM's PyPI package shipped malicious versions 1.82.7 and 1.82.8 on 2026-03-24.
  evalproof's only runtime dependencies stay `httpx`, `pydantic`, `numpy` and `pyyaml`,
  which a reader can audit.
* **One HTTP shape covers most providers.** The differences are a base URL, a key variable,
  an auth header, the name of the token-limit field and whether the server is local. That is
  data, so a profile is a few lines rather than a module.
* **Profiles are easy to review.** Each row cites the provider's documentation for its base
  URL and key variable, and a test writes the whole table out by hand, so a change to a URL
  or a field name shows up in a diff and fails a test until someone confirms it.

Two rules follow from reproducibility, which is still the point of the local default. Local
servers get `temperature` and `seed` on every call, so a local run repeats; hosted providers
get them only when the suite sets them, because hosted reasoning models reject a
non-default temperature. The cache key and `GenerationParams` defaults are unchanged, so
cached local runs still hit. AWS Bedrock and Google Vertex AI authenticate with signed cloud
credentials and are not implemented; a gateway's OpenAI-compatible endpoint reaches them.
