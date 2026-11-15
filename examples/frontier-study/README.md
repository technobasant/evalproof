# Frontier-model study (2026-11)

The support Q&A and ticket-router cases from [`../support-qa`](../support-qa) and
[`../deterministic`](../deterministic), run on three hosted models from three providers:

| Provider | Model | Settings |
|---|---|---|
| Anthropic | `claude-sonnet-5-5` | provider default |
| OpenAI | `gpt-6-astra` | reasoning effort medium |
| Google | `gemini-4-argon-medium` | provider default |

No provider exposes temperature or seed here, so every call samples at the provider's default.
The results are in [MEASUREMENTS.md, section 10](../../MEASUREMENTS.md#10-frontier-models-on-a-sample).

## What was run

A sample, to keep the number of calls small: every second case of each dataset (`lines[::2]`), so
the sample keeps the original mix of topics, unanswerable questions and queues.

| Suite | Cases | Per model | Calls per model |
|---|---:|---|---:|
| [`support-qa.yaml`](support-qa.yaml), grounded prompt | 42 | 2 runs | 84 |
| [`support-qa.yaml`](support-qa.yaml), friendly prompt | 42 | 1 run | 42 |
| [`ticket-router.yaml`](ticket-router.yaml), [`prompt.router.md`](prompt.router.md) | 30 | 2 runs | 60 |

558 calls in all, no judge. The support suite is scored by its regex answer key; the router by an
exact label match.

## Files

* `support-qa.yaml`, `ticket-router.yaml`: the suites. Each model is a variant; its `target`
  says how the model was reached, which is what you need to repeat the run.
* `support-qa.sample.jsonl`, `ticket-router.sample.jsonl`: the sampled cases, with both answer
  keys.
* `runs/`: one run file per model and suite. `support-qa-*-baseline-t0.json` is the first run of
  the grounded prompt on its own, rebuilt from the cache, so the one-run friendly prompt can be
  gated against it.
* `logs/`: the output of the last rebuild of `runs/`, made from the cache with no model
  reachable, so every row in it reads `(cached)`.
* `measure.py`: recomputes every table in section 10 from `runs/`, without a model, and
  re-scores every stored answer against the current dataset to check the verdicts in the run
  files: `uv run python examples/frontier-study/measure.py`.
* `run-study.sh`: makes the runs. Finished calls are answered from the cache, so a rerun is free
  and an interrupted study resumes.

## Answer key review

The answer key was written for small local models. On these models it failed 12 answers that
were correct, so the suite carries two scorers: `answer_key` (the key as written) and
`answer_key_reviewed` (the gate metric). Every run file keeps both. The reviewed key makes four
changes:

| Change | Why | Answers it flipped |
|---|---|---|
| Accept `’` wherever the key accepts `'` | refusals written "I don’t have information…" matched no accept pattern | 5, all `gpt-6-astra` (q75, q79, q81) |
| q37 rejects 200 MB only when it is said of Plus | "Plus is 25 MB; Team and Enterprise allow 200 MB" was rejected for naming Plus and 200 MB | 3, two models |
| q63 rejects Plus only when it is listed among the plans with API access | "the Free and Plus plans don't include API access" was rejected for naming Plus | 2, two models |
| q05 rejects any dollar figure that is not a list price ($6, $60, $11, $110) | answers that gave the Team price after "Enterprise pricing is custom" were rejected | 2, two models |

The review was made after reading the outputs, so it is not blind, and it cannot show that these
models never fail: a key fitted to their answers will pass them. What it can show is that each
of the 12 was a correct answer, and that the narrower patterns still catch the wrong claim they
were meant for. All four were checked on wrong answers written for the purpose, in the style of
the real ones, and each of these fails the reviewed key:

* q05: "Enterprise pricing is custom. It typically works out to about $25 per user per month."
* q37: "The limit is 25 MB on Free and 200 MB on Plus."
* q63: "The REST API is available on Plus, Team and Enterprise, though not on Free." and "The REST
  API is only on paid plans."

The q63 pattern is strict: a correct answer that happens to write "Plus, Team" in a list, such as
"not on Free or Plus, Team and Enterprise only", would fail it. None of the 378 answers here does.
`measure.py` lists every answer whose verdict differs between the two keys.

The five questions the product facts do not answer (q75, q77, q79, q81, q83) have accept patterns for a refusal
and no reject patterns, so an answer that declines and then guesses still passes. Under the
friendly prompt, `gpt-6-astra` and `gemini-4-argon-medium` both declined to give a euro price for
q83 and then said the amount would depend on the payment provider's exchange rate, which the
product facts do not say.

The router's only misses are t53, "Do you have a student discount?", labelled `other`. All three
models answered `billing` on both runs, and the router prompt puts prices under billing, so the
label is arguable. It is left as it is and reported.
