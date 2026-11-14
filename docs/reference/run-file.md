---
description: The JSON run file evalproof writes: every field of the run, case result, output, score and trace, with the version each appeared in.
---

# Run file format

A run file is the JSON record of one run of a suite: what the system answered for every case and how every scorer scored it. This page is for anyone reading run files from a script, importing results from another tool, or committing runs for CI, and a test checks it against the models that write them.

`evalproof run` and `evalproof import` write run files; `compare`, `verify`, `rehash`, `report`, `calibrate` and `pairwise` read them. They are pretty-printed JSON with two-space indent, written by `Run.model_dump_json` and read back by `load_run`, so a file you write by hand must pass the same validation.

## An example

One case of a run of the ticket-router example, as evalproof 0.7.0 wrote it. A real file has one entry in `results` per case and trial.

```json title="runs/baseline.json"
{
  "info": {
    "suite": "ticket-router",
    "system": "python:router:baseline_route",
    "dataset_path": "cases.jsonl",
    "dataset_hash": "bc97ea936a7d2a2a",
    "started_at": "2026-11-14T14:34:46+00:00",
    "finished_at": "2026-11-14T14:34:46+00:00",
    "wall_time_s": 0.001,
    "evalproof_version": "0.7.0",
    "seed": 0,
    "environment": {
      "python": "3.13.15",
      "platform": "macOS-27.0.1-arm64-arm-64bit-Mach-O",
      "machine": "arm64",
      "processor": "arm"
    },
    "suite_hash": "2600c532d6653efe",
    "variant": null,
    "trials": 1
  },
  "results": [
    {
      "case_id": "t01",
      "tags": [
        "billing"
      ],
      "output": {
        "output": "billing",
        "retrieved": [],
        "latency_ms": 0.0037080608308315277,
        "tokens_in": 0,
        "tokens_out": 0,
        "cost_usd": 0.0,
        "cached": false,
        "error": null,
        "trace": null
      },
      "scores": {
        "queue": {
          "value": 1.0,
          "passed": true,
          "reason": null,
          "judge_tokens": 0,
          "judge_cost_usd": 0.0
        }
      },
      "case_hash": "58122e3549d8b9e0",
      "content_hash": "413a3f1b96fc5896",
      "trial": 0
    }
  ]
}
```

## Run

| Field | Type | What it holds |
|---|---|---|
| `info` | object | facts about the run; see [Run info](#run-info) |
| `results` | list | one [case result](#case-result) per case and trial |

## Run info

| Field | Type | What it holds | Since |
|---|---|---|---|
| `suite` | string | the suite's `name` | 0.1 |
| `system` | string | the system's `label`, or its target when it has none | 0.1 |
| `dataset_path` | string | the dataset path as written in the suite | 0.1 |
| `dataset_hash` | string | a short hash of the dataset's cases | 0.1 |
| `started_at` | string | start time, ISO 8601 in UTC | 0.1 |
| `finished_at` | string | end time, ISO 8601 in UTC | 0.1 |
| `wall_time_s` | number | seconds from start to end | 0.1 |
| `evalproof_version` | string | the version that wrote the file | 0.1 |
| `seed` | integer | the suite's `seed` | 0.1 |
| `environment` | object | `python`, `platform`, `machine` and `processor`, plus facts about the model targets, such as `judge_model_resolved`, the model id the provider reported | 0.1 |
| `suite_hash` | string or null | the suite fingerprint; null before 0.5 and in imported runs | 0.5 |
| `variant` | string or null | the variant the run was made with; null for the suite as written and before 0.6 | 0.6 |
| `trials` | integer | repeats per case; 1 before 0.7 and in imported runs that kept no repeats | 0.7 |

## Case result

| Field | Type | What it holds | Since |
|---|---|---|---|
| `case_id` | string | the case's `id` | 0.1 |
| `tags` | list of strings | the case's `tags` | 0.1 |
| `output` | object | what the system returned; see [System output](#system-output) | 0.1 |
| `scores` | object | one [score](#score) per scorer, by scorer name | 0.1 |
| `case_hash` | string or null | a short hash of the case as scored, so `compare` can tell an edited case from an unchanged one; null before 0.5 | 0.5 |
| `content_hash` | string or null | the same without the tags, so a tag change is not an edit; null in files written by 0.5.0 | 0.5.1 |
| `trial` | integer | which repeat of the case this is, counting from 0; 0 before 0.7 | 0.7 |

## System output

| Field | Type | What it holds |
|---|---|---|
| `output` | string | what the system answered; for a multi-turn case, the last reply |
| `retrieved` | list of strings | the ranked ids a retrieval system returned |
| `latency_ms` | number | milliseconds taken |
| `tokens_in` | integer | input tokens |
| `tokens_out` | integer | output tokens |
| `cost_usd` | number | cost in US dollars |
| `cached` | boolean | true when the answer came from the response cache |
| `error` | string or null | the error when the system failed; null otherwise |
| `trace` | object or null | what the agent did; see [Trace](#trace); null for a system that reports none and in files written before the trace existed |

## Score

| Field | Type | What it holds |
|---|---|---|
| `value` | number or null | the continuous score, used by bootstrap comparisons; null when the scorer could not decide |
| `passed` | boolean or null | the pass/fail outcome, used by pass rates and McNemar; null when undecided or when the scorer has no threshold |
| `reason` | string or null | why, when the scorer says |
| `judge_tokens` | integer | tokens the judge used for this case |
| `judge_cost_usd` | number | what the judge cost for this case |

An undecided case, with a null `value`, is counted and reported. It is never scored as a pass.

## Trace

| Field | Type | What it holds |
|---|---|---|
| `responses` | list of strings | the reply per user turn; the last is the output |
| `tool_calls` | list | every [tool call](#tool-call) the agent made |
| `rounds` | integer or null | model calls or generation rounds, when known |
| `usage` | list | [model usage](#model-usage) by role |
| `meta` | object | anything else, such as the agent's name |

## Tool call

| Field | Type | What it holds |
|---|---|---|
| `name` | string | the tool's name |
| `args` | object | the arguments |
| `result` | any | what the tool returned |
| `ok` | boolean | false when the call failed |
| `error` | string or null | the error text |
| `id` | string or null | the call id the SDK gave it |
| `turn` | integer | the user turn the call belongs to, counting from 0 |
| `latency_ms` | number or null | milliseconds the call took |

## Model usage

| Field | Type | What it holds |
|---|---|---|
| `role` | string | free text: `agent`, `judge`, `worker` and so on |
| `model` | string or null | the model id |
| `tokens_in` | integer | input tokens |
| `tokens_out` | integer | output tokens |
| `cached_tokens` | integer | tokens served from a provider cache |
| `cost_usd` | number or null | cost in US dollars |

## Repeated trials

With `trials` above 1, `results` holds one entry per case and trial, each with its own `trial` number. `Run.by_id()` returns the first trial of each case, and `Run.by_case()` returns every trial of each, in trial order. Run files written before 0.7 hold one result per case, all with `trial` 0.

## Reading older files

Every field added after 0.1 has a default, so any older file still loads. `compare` and the gate say what an older file lacks: without `case_hash` they cannot tell an edited case from an unchanged one, and `verify` needs a run made with evalproof 0.5 or newer. `evalproof rehash` adds the case hashes to a file from before 0.5.

## Next steps

* Produce a run file: [Commands](commands.md#evalproof-run).
* Build one from another tool's results: [Import results from other tools](../guides/import.md#import-results-from-other-tools).
* Read one from Python: [From Python](../guides/python.md).
