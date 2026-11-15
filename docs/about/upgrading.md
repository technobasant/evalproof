---
description: What to change when you move an evalproof setup from 0.5 to 1.0, release by release, and how run files from older versions behave.
---

# Upgrading

What each release from 0.5 onwards changes for an existing setup, and what to do about it. It is for anyone who already has a suite and committed run files and is moving to a newer version. The [changelog](changelog.md) has the complete list; this page keeps the parts that can break something.

Since 1.0 the public surface changes incompatibly only in a major release, after a deprecation ([Versioning and support](versioning.md)). Before 1.0 a minor release could change behaviour, so moving from an older version, read every section between your version and 1.0.

## The short path

1. Upgrade the package and the version pinned in your CI workflow.
2. Run the suite on the base branch with the new version and commit the run file (by default `evals/runs/latest.json`).
3. Run `evalproof verify RUN SUITE` on it. It makes no model call, and exit 0 means the file matches the suite and its cases.
4. Open a pull request. Comparisons now pair on the new run file's hashes.

Steps 2 and 3 matter most when you move from 0.4 or earlier, because older run files lack the hashes that `compare` and `verify` read.

## 1.4

Nothing is removed or renamed, and a suite, dataset or run file that worked under 1.3 works under 1.4.

**Added, with nothing to change:** adapters for LlamaIndex, CrewAI, Pydantic AI, DSPy, Haystack, AutoGen and Semantic Kernel ([Agent SDK adapters](../guides/adapters.md)); a helper library in the `evalproof` npm package, `init --node` and a scan of TypeScript tools and agents ([Testing a TypeScript or JavaScript system](../guides/typescript.md)).

**One change in what `scan` reports:** TypeScript and JavaScript code now takes part in the `uncovered_code` and `tool_unchecked` gaps, so a repository with a TypeScript agent can show gaps it did not show under 1.3.

## 1.3

Nothing is removed or renamed, and a suite, dataset or run file that worked under 1.2 works under 1.3. A run file gains an optional `tag_filter` field; `format_version` is unchanged.

**Added, with nothing to change:** `evalproof scan` and `evalproof draft` find the LLM code no suite tests and draft inputs for it ([Find what is not tested](../guides/find-gaps.md)); `init --platform bitbucket`; `evalproof hook install` for a pre-push `verify`; `run --tags` and `--exclude-tags` ([CI](../guides/ci.md)).

**One new refusal:** `compare --gate` exits 4 when the two runs were made with different tag filters. Runs made without `--tags` are not affected.

## 1.2

Nothing is removed or renamed, and a suite, dataset or run file that worked under 1.1 works under 1.2.

**Added, with nothing to change:** `evalproof import-dataset` reads test cases from promptfoo, DeepEval, Ragas, LangSmith, OpenAI Evals, Inspect, Braintrust, Langfuse, Phoenix and any JSONL or CSV table, with `auto` to detect the format; `--suite-out` writes a suite with the scorers those cases need; `evalproof init --from FILE` starts a repository from them. See [Import test cases](../guides/import-test-cases.md).

`evalproof import-dataset adk` writes the same dataset as before, and also accepts `--suite-out`.

## 1.1

Nothing is removed or renamed. `evalproof init` also writes an evalproof section into the repository's `AGENTS.md` (between markers, leaving the rest of the file as it was); pass `--no-agents-md` to skip it. See [Coding agents](../guides/coding-agents.md).

## 1.0

Nothing from 0.8 is removed or renamed, and a suite or run file that worked under 0.8 works under 1.0.

**Changed:** for a pass/fail metric compared with the exact McNemar test (one trial per case, or repeated trials combined with `all` or `any`), the power line and the `power` fields of `--json` output never report a detectable drop below what the exact McNemar test can call significant at all: at α 0.05 that is 6 cases worse and none better, so 14.3 points on 42 cases. On small suites the figure is larger than 0.8 printed, and a suite with fewer than 6 cases is told that no drop can be significant. The gate's verdict is unchanged; only the advice beside it is.

**Changed, for run files:** a run file now records `format_version`. Files without it (written before 1.0) load as version 1. From 1.0 on, a file whose version is newer than the installed evalproof understands is refused with an error, instead of loading with the unknown fields dropped; 0.8 and earlier do not check, and read a 1.0 file as before.

**Changed, for agent CLI targets:** when the CLI binary is not on `PATH` while a run is made (a run served from the cache), the run's environment still records the CLI, its isolation tag and its sampling, with `cli_version: unavailable`. Under 0.8 those keys were missing.

**Added, with nothing to change:** `--cache-only` on `run` and `pairwise`, which rebuilds a run from cached replies and exits 2 on the first case it would have to call a model for; `fold_quotes` on the `regex` scorer; the Cursor sandbox on macOS and `evalproof doctor --cursor-canary`. See [Commands](../reference/commands.md) and [Configuration](../reference/configuration.md).

## 0.8

**Changed:** the parameters of an `llm_judge` scorer are checked when the suite loads. A misspelled key, a wrong type, or `rubric` combined with a rubric set used to pass loading and fail only when the scorer was built; now `evalproof run` and `evalproof verify` stop at load with the key named. If a suite that loaded under 0.7 fails now, fix the key the message names.

**Changed, for imported runs:** an imported result now carries its case's tags (garak and promptfoo red-team tags, DeepEval's own), and repeats are ordered by the index the source gave them. Tags are not part of a case's hash, so a 0.8 import still pairs with a 0.7 import of the same file. If you gate with `--epochs trials`, re-import the baseline with 0.8 so trial `n` means the same run on both sides.

**Added, with nothing to change:** the `garak`, `adk` and `claude-plugin-eval` import formats, `evalproof import-dataset adk`, the Inspect tool trace, the OpenTelemetry adapter, rubric sets, `per_rubric_metrics` and the `facts` scorer. See [Importing results](../guides/import.md) and [Configuration](../reference/configuration.md).

## 0.7

**Changed:** `Run.by_id()` returns each case's first trial (trial 0), where it used to return the last result with that id. `Run.by_case()` returns every trial of each case.

If your code reads a run file from Python and the run repeats cases, check which one you want. For a run with one trial per case, nothing changes. Run files from before 0.7 load as one trial per case, and a run with one trial per case is written, read and compared as before.

**Added, with nothing to change:** repeated trials (`trials` or `--trials`), `--aggregate` and `gate.aggregate`, the power line beside the gate decision, `compare` with several candidates, and `readiness` reliability. See [Repeated trials](../guides/repeated-trials.md).

Two things to know when you start using trials: runs with different trial counts cannot be compared (exit 4 under `--gate`, exit 2 without), and `all` or `any` on a gated continuous metric is refused.

## 0.6

**Breaking:** the case fields `turns`, `state` and `expect` now have a meaning and are validated when the dataset loads.

| Field | Must be | Read as |
|---|---|---|
| `turns` | a list of strings or objects | the user turns that follow `input` |
| `state` | an object | the initial session state |
| `expect` | a list of valid checks | the inline checks of the `checks` scorer |

A dataset that used one of these names for something else fails to load, and the message says to rename the field. Rename your own field in every case.

**Added, with nothing to change:** the `checks` and `python` scorers, adapters, suite `variants`, multi-turn cases, `evalproof readiness`, and `evalproof doctor` printing the credential each target would use. See [Testing agents](../guides/testing-agents.md).

A suite whose system is a coding-agent CLI is refused before any case runs if a case has `turns`, since a CLI takes one prompt per call.

## 0.5.2 and 0.5.1

0.5.2 changes no behaviour. It rewrites the README and the documentation, and `evalproof init` pins 0.5.2 in the workflows it writes.

0.5.1 changes how cases are paired:

* Run files carry a new per-case `content_hash`, the case hash without its tags. `compare` pairs on it when both runs have one, and otherwise on `case_hash` as in 0.5.0. Adding or changing a tag no longer counts as an edit once both runs come from 0.5.1 or newer.
* `verify` still counts a tag change, because `must_pass` reads tags from the run file.
* `verify` now scores the stored outputs again with the scorers that need no model (`exact_match`, `contains`, `regex`, `json_schema`, `numeric_tolerance`, `recall_at_k`, `mrr`, `ndcg`) and fails when a stored score differs.
* `evalproof rehash RUN SUITE` adds per-case hashes to a file written before 0.5, with no model call.

Run files written by 0.5.0 keep working. They have no `content_hash`, so they pair on the per-case hash, tags included.

## 0.5

Version 0.5 has the most breaking changes. In the order you are likely to meet them:

1. **Unknown suite keys are errors.** Every config model forbids extra keys. A typo such as `gate.max_drops` used to be ignored and now fails, naming the key and the closest valid one. Fix the key, or remove it.
2. **`evalproof run` exits 2 when every case errored.** It still writes the run file and prints the report first. A pipeline that relied on exit 0 here must now handle the failure.
3. **Cache keys of agent CLI calls change once.** The CLI version is no longer part of the key, so a CLI that updates itself keeps its cached replies. Cached replies from 0.4.x are not reused. Use `--no-cache` to ask again at any time.
4. **`compare --metric NAME` exits 2** when neither run has a scorer called `NAME`. It used to exit 4.
5. **`compare` no longer exits 4 because the dataset changed**, when both runs carry per-case hashes. Edited cases are excluded and listed instead. A new case is not compared.
6. **The GitHub Action runs `verify` first.** Given a `suite`, it runs `evalproof verify` on the candidate and fails with exit 4 when the committed run does not match the suite and its cases. A run file from 0.4.x has no suite fingerprint and fails it. Re-run the suite with 0.5 and commit the file, or set `verify-candidate: "false"`.
7. **A cost budget can trip where it did not.** A `claude-code` system or judge with no `pricing` now records the CLI's reported `total_cost_usd`, and the judge's cost counts toward `max_cost_usd`, so a suite with `gate.max_cost_usd` can newly exit 3.

A 0.4.x baseline has no per-case hashes, so a pull request that also changes the dataset cannot be gated (exit 4) until the base branch has a run file from 0.5 or newer. Run the suite with 0.5 on the base branch and commit the file first. Tags are part of a case's hash in 0.5.0, so adding a `must_pass` tag counts as an edit until the baseline is refreshed.

**Added in 0.5, with nothing to change:** per-case `case_hash`, `evalproof verify`, the gate checks that need no baseline (`min_pass_rate`, `must_pass`, `max_new_failures`), and the "Failed cases" list in `run` and `report`.

## Run files from older versions

Every field added after 0.1 has a default, so an older run file still loads. What an older file lacks limits what you can do with it:

| The file was written before | What it cannot do | What to do |
|---|---|---|
| 0.5 | pair edited cases from unchanged ones; be checked by `verify` | `evalproof rehash RUN SUITE` adds the case hashes; `verify` still needs a run made with 0.5 or newer, so re-run the suite |
| 0.5.1 | ignore a tag change when pairing | nothing; it pairs on the per-case hash, tags included, until you re-run |
| 0.7 | hold repeated trials | nothing; it loads as one trial per case |

`rehash` works only when the cases the run scored are still the first ones in the dataset, unchanged, which the `dataset_hash` stored in the file proves. Otherwise it exits 4 and writes nothing. The [run file format](../reference/run-file.md) page lists every field and the version that added it.

## Next steps

* Every release in full: [Changelog](changelog.md).
* What a version number promises: [Versioning and support](versioning.md).
* An error after upgrading: [Troubleshooting](../guides/troubleshooting.md).
