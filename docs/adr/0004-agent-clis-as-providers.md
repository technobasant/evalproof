# 4. Coding-agent CLIs as providers, isolated and fingerprinted

Status: accepted · 2026-11

## Context

Many people who would run evalproof already have Claude Code or Codex logged in on their
machine, on a subscription, and no API key. Local-first (ADR 0002) means a run should need
nothing beyond what is already there, so those two CLIs are worth supporting as a judge or a
system under test.

A coding agent is not a model endpoint, though. Started with its defaults it loads the
project's `CLAUDE.md` or `AGENTS.md`, the user's memory, plugins, MCP servers and skills into
the prompt, it has a shell and file tools, and it chooses its own model, sampling and
reasoning effort. As a judge that is a different instrument on every machine and in every
directory, and it can answer by running commands instead of reading the case. It also
changes with each CLI release, with nothing in the run to say so.

## Decision

* Add two providers, `claude-code:<model>` and `codex:<model>`. Each runs the user's own
  binary as a subprocess, prompt on stdin, JSON on stdout. No model call goes through
  anything but that binary.
* **Isolation is a requirement.** Every call runs in a fresh empty directory that is removed
  afterwards, with the agent's system prompt replaced by the case's own (or one neutral line)
  and its tools, settings, MCP servers, skills and memory switched off. No permission-bypass
  flag is ever passed. Codex runs with its shell disabled in a read-only sandbox, and a
  command in its output fails the call. The flags are listed by `evalproof doctor`. The CLI
  has no switch that removes everything (Codex still lists its bundled skills and some tool
  descriptions in the prompt), so the isolation is named, versioned (`claude-code/1`,
  `codex/1`) and checked by canary tests, not claimed to be total.
* **A CLI target has a fingerprint:** the CLI, its version, the isolation scheme and the
  sampling note (`cli-default`, since neither CLI takes a temperature, seed or token limit).
  The run records it, with the model id that actually answered, in `runner.environment`. The
  response cache keys on it, so a CLI upgrade is a cache miss, never a stale hit.
* **The gate compares the judge.** `compare` lists a change in any judge fact between two runs
  (`same_judge`, `judge_changes`), and the gate refuses to decide (exit 4) unless
  `--allow-judge-change` or `gate.allow_judge_change` says the change is intended. A changed
  system target only warns, because that is usually the change under test.
* Authentication stays with the CLI. evalproof does not read token files or the keychain. API
  key variables that would switch the CLI from the login to a key are removed from the child's
  environment; `api_key_env` names the variable that is passed on instead, which is the route
  for CI.
* The feature is opt-in: nothing runs a CLI unless a suite names one of the two targets.
  The README states the vendors' terms as they stand and says to use an API key for CI or
  shared use.

## Consequences

* A judge on a CLI is noisier than a seeded local one. The run does not hide that; it records
  what produced each score, and the existing tools for noise (the noise-aware gate, judge
  calibration, `--no-cache` repeats) apply unchanged.
* Calls cost about 1.4 s (`claude-code:haiku`) to 4 s (Codex) each, mostly start-up, and draw
  on the subscription's limits instead of an API bill. `concurrency` is the lever.
* A 0.3.0 pipeline that compared two runs after an upgrade of the judge's CLI would have
  passed silently. With the check it exits 4 until someone decides. Runs written before this
  change carry none of the new keys, and only keys present in both runs are compared, so they
  compare as before.
* Isolation can drift as the CLIs change. The live test module (`EVALPROOF_LIVE=1`) is the
  check to run before a release, and a change to the flags bumps the isolation version, which
  changes the fingerprint and so the cache and the judge check.

## Rejected

* **The vendors' SDKs and agent frameworks.** The SDK for Claude Code needs an API key, which
  is the case this feature exists to avoid, and each SDK is a dependency that evalproof
  otherwise does not have (plain HTTP throughout).
* **`claude --bare`.** It skips the project and user context, as wanted, but it requires an
  API key as well.
* **Reading the CLIs' stored tokens** and calling the API directly. It would sidestep the
  isolation question and the process start-up, and it would be using subscription credentials
  outside the product they were issued for. evalproof never touches credentials.
* **Passing the CLI's defaults through** and documenting the risk. A judge that has read the
  project's instructions is not a judge, and a result that depends on the working directory
  cannot be gated on.
