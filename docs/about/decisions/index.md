---
description: The design decisions behind evalproof, the alternatives they were chosen over, and the ADRs that record them.
---

# Decisions

Each row is a design decision, the alternative it was chosen over and why; the ADR column links the full record.

| Decision | Alternative | Why | ADR |
|---|---|---|---|
| Treat the judge as a classifier to be calibrated | Trust the judge's pass rate | Its count is biased by its own error rates, and the bias moves with the true rate | [0001](../../adr/0001-judge-is-a-classifier.md) |
| Local Ollama by default, HTTP-only providers, content-addressed cache | Hosted APIs and vendor SDKs | Reproducible, free to re-run, tied to a model digest; no SDK dependency | [0002](../../adr/0002-local-first-providers.md) |
| Gate on "large AND significant", on paired cases | "Fail if the score dropped by X" | On a few dozen cases, X-point drops are mostly noise, and a noisy gate gets switched off | [0003](../../adr/0003-gate-on-significance-not-delta.md) |
| One metric by default; several with Holm correction | Gate every metric at α each | Testing many metrics at α = 0.05 each inflates false failures | 0003 |
| Run a coding-agent CLI as a provider only in isolation, and gate on its fingerprint | Pass the CLI's defaults through, or ignore its version | A CLI that reads your project's instructions is not a judge, and a silent upgrade changes the instrument | [0004](../../adr/0004-agent-clis-as-providers.md) |
| Pointwise verdicts for gating; pairwise as a diagnostic | Pairwise preference for gating | Measured: 42% of pairwise verdicts flipped with answer order, with a strong length preference | [MEASUREMENTS §6](../measurements.md) |
| numpy and the standard library for statistics | SciPy, statsmodels | Small enough to read and to test against hand-worked values | — |
