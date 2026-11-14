---
description: Symptom, cause and fix for the errors evalproof prints, from a missing API key to a gate that cannot reach a verdict.
---

# Troubleshooting

The messages evalproof prints when a run, a comparison or a check cannot go ahead, with what causes each and how to clear it. It is for anyone who has hit an error and wants to find the line in this page that matches it.

Most errors print as `evalproof: error: <message>` on stderr and exit with code 2 (a setup problem) or 4 (the runs cannot be gated). The [exit codes](../reference/exit-codes.md) page says what each code means. Set `EVALPROOF_DEBUG=1` to get the full traceback for an error that is not one of these.

## Providers and credentials

### `anthropic: set ANTHROPIC_API_KEY to use this provider`

**Cause:** the target needs an API key and the environment variable is empty or unset. The message names the provider and the variable. A suite can name a different variable with `api_key_env`.

**Fix:** export the variable in the shell that runs evalproof, or add it as a repository secret in CI. The key is read from the environment only, never from the suite file. See [Credentials](../getting-started/credentials.md) for each target and the variable it reads.

### `HTTP 401: ...` or another `HTTP <status>: ...`

**Cause:** the provider answered with a status evalproof does not retry, and the text after the status is the first 300 characters of the provider's own message. Read that message first: it says whether the key, the model name or the request was refused.

**Fix:** correct what the provider names. Check the key, the model id after the colon in `provider:model`, and `base_url` when the suite sets one.

### `giving up after 4 attempts: ...`

**Cause:** the call failed four times in a row. evalproof retries a transport error (the host could not be reached, the connection dropped) and the statuses 408, 409, 425, 429, 500, 502, 503, 504 and 529, with a jittered exponential backoff that starts at 0.5 seconds. The error after the colon is the last one it saw. A rate limit or an overloaded provider is the usual cause.

**Fix:** run the suite again. Only calls that succeeded are cached, so the re-run repeats just the ones that failed. If it keeps failing, lower the load on the provider or wait out the limit.

### Ollama is not running

**Symptom:** `giving up after 4 attempts:` followed by a connection error, on an `ollama:<model>` target.

**Cause:** nothing is listening where evalproof looks. The default endpoint is `http://127.0.0.1:11434`.

**Fix:** start Ollama, or set `base_url` on the target to where it runs. In CI a local model cannot run on the runner; use a hosted judge there.

### `<provider>: set base_url in the suite to use this provider`

**Cause:** the provider is an OpenAI-compatible profile that has no default endpoint, so evalproof does not know where to send the call.

**Fix:** set `base_url` on the `system` or `judge` block that uses it. The key table is in [Providers](providers.md#providers).

### `target 'x' must look like '<provider>:<model>'`

**Cause:** a `system` or `judge` target is missing the colon, or has nothing after it (`target 'x' has no model`). `unknown provider 'x'; expected one of [...]` is the same mistake with a provider name evalproof does not have, and it lists the ones it does.

**Fix:** write the target as `provider:model`, for example `anthropic:claude-sonnet-4-5`. Everything after the first colon is the model id, so `openrouter:anthropic/claude-sonnet-4.5` is valid.

### `anthropic: refused (...)`, or `response blocked by the content filter`

**Cause:** the model declined to answer, or the provider's content filter blocked the reply. The text in brackets is the refusal category when the provider gave one. The case is recorded as errored rather than failed, because no answer was scored.

**Fix:** if the prompt is legitimate, reword the case. If the refusal is the behaviour under test, score it with a scorer that reads the output, and run the case against a target that returns the refusal as text.

### `max_tokens reached before any text; raise params.max_tokens`

**Cause:** the model spent its whole output budget before writing any text, which a small `max_tokens` allows. The message appears as `anthropic: max_tokens reached ...` and as `<name>: max tokens reached ...` for the OpenAI-compatible profiles.

**Fix:** raise `params.max_tokens` on the target. See [Generation parameters](../reference/configuration.md#generation-parameters).

### `no choices in response`, or `ollama: unexpected response shape`

**Cause:** the endpoint answered with JSON evalproof cannot read as a chat reply. The first 300 characters of the reply follow the message.

**Fix:** call the endpoint outside evalproof with the same body and check what it returns. Confirm that `base_url` points at the chat API of the provider and not at a web page.

## Coding-agent CLIs

### `` claude-code: `claude` was not found on PATH; install it with `...` ``

**Cause:** a `claude-code:` or `codex:` target runs your own `claude` or `codex` binary, and it is not on the `PATH` of the process that runs evalproof. This is common in CI, where nothing is installed by default.

**Fix:** install the CLI with the command the message gives, or run `evalproof doctor SUITE` to check every CLI target the suite uses without calling a model. See [`evalproof doctor`](../reference/commands.md#evalproof-doctor).

### `evalproof doctor` says `not logged in; run `claude auth login``

**Cause:** the `claude` CLI is installed but has no login, and the target is in `login` mode. `doctor` checks this without calling a model, and exits 1 when a target is not ready.

**Fix:** log in on this machine. A login does not carry over to CI: there, set the key as a secret, or use a Claude Code token for your own repository only. The [credentials table](../getting-started/credentials.md) lists what each target needs in each place.

### `EVALPROOF_CLI_AUTH must be one of login, api-key, got 'x'`

**Cause:** the variable `EVALPROOF_CLI_AUTH` is set to something else. Unset or empty means `login`.

**Fix:** set it to `login` or `api-key`, or unset it.

### `timed out after 120s`, or `exited with status 1`

**Cause:** the CLI did not answer within the target's timeout, or it exited with an error. When the CLI wrote anything to stderr or stdout, that text replaces `exited with status`.

**Fix:** raise `timeout_s` on the target for a slow model. For an exit status, run the same CLI by hand in an empty directory and read its own message.

### `codex used a tool (...); isolation broken`

**Cause:** a Codex call ran a tool. evalproof runs a coding agent with its tools, settings and skills switched off so that a judge never reads your project's instructions; a call that used a tool is refused rather than scored.

**Fix:** use a hosted model, or the `claude-code` target, for that role. Run `evalproof doctor SUITE` to see what each CLI target would use.

### `multi-turn cases need a python, http or direct model target`

**Cause:** a case in the dataset has `turns`, and the system is a coding-agent CLI, which takes one prompt per call. The check runs before any case does.

**Fix:** point the system at a python, http or direct model target, or move the multi-turn cases to a separate suite.

## The suite and the dataset

### `<path>: <key>: unknown key (did you mean 'x'?)`

**Cause:** the suite has a key evalproof does not know, since version 0.5. A typo such as `max_drops` is the usual cause. When no key is close, the message lists the valid keys.

**Fix:** use the key it suggests. Every key is on the [Configuration](../reference/configuration.md) page.

### `suite file not found`, `dataset not found`, `dataset is empty`

**Cause:** the path is wrong, or the dataset has no cases. Relative paths in a suite resolve against the suite file's directory, not the working directory.

**Fix:** check the path in the message, and that the dataset has at least one JSONL line.

### `duplicate case id 'x'`

**Cause:** two lines of the dataset share an id. The message gives the file and line number of the second.

**Fix:** make every `id` unique. Runs are paired by id, so a duplicate would also make a comparison wrong.

### `an llm_judge scorer needs a 'judge' target`

**Cause:** the suite has an `llm_judge` scorer and no `judge` block.

**Fix:** add a `judge` target, or use a scorer that needs no model.

### `python target must be 'python:<module>:<function>'`

**Cause:** the `python:` target has the wrong shape, or the name it points at is not a function (`... is not callable`).

**Fix:** write `python:package.module:function`, and check that the module imports from the directory evalproof runs in.

### `case 'x': turns must be a list of strings or objects`

**Cause:** since 0.6 the case fields `turns`, `state` and `expect` have a meaning and are checked when the dataset loads. A dataset that used one of the names for something else now fails.

**Fix:** rename your own field. The message says what evalproof reads the field as. See [Upgrading](../about/upgrading.md).

## Running

### `evalproof: error: every case errored; the first error: ...`

**Cause:** every case failed to produce an output, so there is nothing to score. The run file is still written and the report printed first. The first error is usually one of the provider messages above.

**Fix:** fix that error, then run again.

### A re-run returns the same answers after you changed something

**Cause:** the response cache is keyed on the provider, model, endpoint, messages and generation parameters, so a change outside those, such as the model's own behaviour behind an alias, does not miss the cache.

**Fix:** run with `--no-cache`, or delete the suite's `cache_dir` (`.evalproof/cache` by default).

## Comparing and gating

### `only 12 paired cases for 'accuracy'; the gate needs 30`

**Cause:** fewer cases appear in both runs than `gate.min_cases` asks for, so a difference could be chance. The command exits 4.

**Fix:** add cases to the dataset, or lower `gate.min_cases` if a smaller suite is acceptable. The run report's power line shows the smallest drop the suite can detect.

### `metric 'x' has no paired scores`

**Cause:** the metric named by `--metric` or `gate.gated` has no score that appears in both runs, usually because the scorer is named differently in one of them.

**Fix:** check the scorer names in each run file against the one you gate on. A `--metric` that neither run has exits 2.

### `the judge changed between the runs, so the scores are not comparable`

**Cause:** the judge's target, CLI, isolation or resolved model differs between the baseline and the candidate. Scores from two judges are not on the same scale. A new CLI version alone only warns.

**Fix:** re-run the baseline with the current judge. To accept the change anyway, pass `--allow-judge-change` or set `gate.allow_judge_change`.

### `the dataset changed between the runs and one of them predates per-case hashes`

**Cause:** the dataset differs, and a run was written before evalproof 0.5, so the cases that stayed the same cannot be told from the ones that changed.

**Fix:** run `evalproof rehash BASELINE SUITE`, which needs no model call, or re-run the baseline with 0.5 or newer. `rehash` works only when the cases the run scored are still the first ones in the dataset, unchanged.

### `baseline has k=5, candidate has k=3; rerun with the same --trials`

**Cause:** the two runs repeat each case a different number of times. They exit 4 under `--gate` and 2 without it.

**Fix:** run both with the same `--trials`. See [Repeated trials](repeated-trials.md).

### `aggregate all needs a pass/fail metric`

**Cause:** `all` or `any` asks whether a case passed every trial or at least once, which a continuous score cannot say. `min_pass_rate` has the same limit.

**Fix:** use `--aggregate mean` for that metric, or gate a pass/fail scorer.

### `the run has no suite hash`, or `the suite's system, judge, scorers or seed differ`

**Cause:** `evalproof verify` found that the committed run file was not made with the committed suite. The first message means the file was written before 0.5 or was imported.

**Fix:** run the suite again and commit the new file. In CI, the action runs `verify` first, and `verify-candidate: "false"` switches that off.

### `--ci api-key needs a hosted judge with an API key`

**Cause:** `evalproof init --ci api-key` was run with a local judge, which cannot run on a CI runner.

**Fix:** pass `--judge` with a hosted target, or use `--ci keyless`, which compares committed run files and makes no model call.

### `--require: unknown dimension 'x'`

**Cause:** `evalproof readiness --require` was given a name that is not one of the dimensions. The message lists the valid names.

**Fix:** use one of the listed names, comma-separated.

## Next steps

* What each exit code means: [Exit codes](../reference/exit-codes.md).
* Every flag and its default: [Commands](../reference/commands.md).
* Changes that can break an existing setup: [Upgrading](../about/upgrading.md).
