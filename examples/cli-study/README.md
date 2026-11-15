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
* `support-qa.sample.jsonl`, `ticket-router.sample.jsonl`: the sampled cases.
* `runs/`: one run file per model and suite. `support-qa-*-baseline-t0.json` is the first run of
  the grounded prompt on its own, rebuilt from the cache, so the one-run friendly prompt can be
  gated against it.
* `logs/`: the summaries printed while the runs were made.
* `measure_cli.py`: recomputes every table in section 10 from `runs/`, without a model:
  `uv run python examples/cli-study/measure_cli.py`.
* `run-study.sh`: makes the runs. Finished calls are answered from the cache, so a rerun is free
  and an interrupted study resumes.

## Answer key review

The answer key was written for small local models. On these models it failed 12 answers that
were correct, so the suite carries two scorers: `answer_key` (the key as written) and
`answer_key_reviewed` (the gate metric). Every run file keeps both. The reviewed key changes
three things and nothing else:

| Change | Why | Cases it flipped |
|---|---|---|
| Accept `’` wherever the key accepts `'` | refusals written "I don’t have information…" matched no accept pattern | q75, q79, q81 (`gpt-6-astra`, 5 answers) |
| q37 rejects 200 MB only when it is said of Plus | correct answers ("Plus is 25 MB; Team and Enterprise allow 200 MB") were rejected | 3 answers, two models |
| q63 rejects a sentence naming Plus only if it has no negation | "the Free and Plus plans don't include API access" was rejected | 2 answers, two models |
| q05 rejects a dollar figure only when it is attached to Enterprise | answers that added the Team price after "Enterprise pricing is custom" were rejected | 2 answers, two models |

The review was made after reading the outputs, so it is not blind. To keep it honest: it was
checked against all 378 support answers and flipped exactly the 12 failures, all to pass, and no
pass to fail; and the reviewed patterns still fail invented answers such as "Enterprise costs $25
per user per month" or "Plus includes REST API access". `measure_cli.py` lists every answer
whose verdict differs between the two keys.

The router's only misses are t53, "Do you have a student discount?", labelled `other`. All three
models answered `billing` on both runs, and the router prompt puts prices under billing, so the
label is arguable. It is left as it is and reported.
