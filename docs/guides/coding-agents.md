---
description: Let a coding agent use evalproof in your repository: the AGENTS.md section and the skill that init writes, how they merge with your own file, and how to refresh them.
---

# Coding agents

`evalproof init` writes two things for a coding agent that works in your repository: a section of `AGENTS.md`, and a skill. This page says what each holds, how `init` merges the section into a file you already have, and how to refresh both.

!!! note "Experimental"
    The text of the `AGENTS.md` section, the skill's text and `init --agent-files-only` are [experimental](../about/versioning.md) and can change in a minor release; the changelog says when. The markers around the section, and the rule that `init` never changes anything outside them, are stable.

## What gets written

| File | What it is for |
|---|---|
| `AGENTS.md` | the evalproof section: which command to run for each task, how to read the exit codes, what never to weaken, and how each kind of provider and target is set up. Any agent that reads `AGENTS.md` sees it |
| `.agents/skills/evalproof/SKILL.md` and `.claude/skills/evalproof/SKILL.md` | one narrow skill, checking a change against the committed baseline. Two identical copies, not a symlink, so a Windows checkout works. It points to the `AGENTS.md` section for every other task |

`init` makes no model call.

## What the section tells an agent

It is plain instructions, about 270 lines, written to be followed rather than read:

* where the suite, the cases and the committed baseline live, and how to run evalproof (`evalproof`, else `uvx evalproof`);
* a table of tasks and the command for each: check a change, run for the first time with `--gate`, add a case, build or extend the suite, compare models, calibrate a judge, test an agent, score retrieval, hold latency or cost, handle flaky output with repeated trials, import another tool's results, check readiness, set up CI, test a program in another language, run in a container;
* how to read the result and the exit codes: 0 pass, 1 regression, 2 usage, 3 budget, 4 cannot gate;
* what the agent must never do to make a gate pass: raise `max_drop` or `alpha`, lower `min_cases`, loosen the rubric, delete or edit cases, switch the judge, run a subset with `--limit`, re-run until it passes, or edit `evals/runs/latest.json` by hand. On exit 1, 3 or 4 it stops and reports to you;
* how to draft cases when it builds a suite: expected answers only from a source of truth, every drafted case tagged `draft` and listed for a human to review, and never a `draft` case in `gate.must_pass`;
* how providers and targets are set up: Ollama and other local servers, Anthropic, OpenAI-compatible profiles, `python:`, HTTP and `command:` targets, and the container image;
* a command reference for the flags an agent needs.

It states only what the [reference](../reference/commands.md) and the other guides state, and it names providers and models, not accounts or plans.

## How `init` merges it into `AGENTS.md`

The section sits between two markers:

```markdown
<!-- evalproof:begin -->
...
<!-- evalproof:end -->
```

`AGENTS.md` is the one file `init` merges instead of keeping or overwriting, with or without `--force`:

| Your `AGENTS.md` | What `init` does | Line it prints |
|---|---|---|
| does not exist | creates it: a one-line heading, a blank line, the section | `wrote          AGENTS.md` |
| exists, with no markers | appends the section after one blank line; your content is not touched | `appended       AGENTS.md` |
| exists, with the markers | replaces the text from the begin marker to the end marker, inclusive, with the current section | `updated        AGENTS.md` |
| exists, and the section is already current | writes nothing | `unchanged      AGENTS.md` |

Every byte outside the markers stays as it was. `init` also keeps your line endings: a file that uses CRLF gets a CRLF section, and one that uses LF gets LF. The file is read and written as UTF-8; a file that is not UTF-8 text is refused.

If the markers are damaged, `init` refuses, exits 2 and writes nothing, not even the other files. Damaged means a begin marker without an end marker (or the reverse), a marker that appears twice, or the end marker before the begin marker. Fix or remove them by hand, then run `init` again.

Write your own notes above or below the markers. Text you add between them is replaced the next time you refresh.

## Refresh the section

After you upgrade evalproof, run `init` again to bring the section up to date. In a repository that already has a suite, use `--agent-files-only` so that nothing else is touched:

```bash title="Terminal"
uvx evalproof init --agent-files-only
```

```text title="Output"
evalproof init: agent files only (no suite, no CI workflow)
  kept (exists)  .agents/skills/evalproof/SKILL.md
  kept (exists)  .claude/skills/evalproof/SKILL.md
  updated        AGENTS.md
```

`--agent-files-only` writes only the `AGENTS.md` section and the skill. It writes no suite and no CI workflow, so `--judge`, `--ci`, `--platform` and `--cli-auth` do not apply to it, and combining it with them is an error (exit 2). Existing skill copies are kept unless you pass `--force`; when a kept copy differs from the version that ships with this release, `init` prints a note saying so. `--force` replaces the skill copies and still never overwrites `AGENTS.md`.

Review the diff before you commit it, as you would any change to a file agents follow.

## Leave one out

| Flag | Effect |
|---|---|
| `--no-agents-md` | do not touch `AGENTS.md` |
| `--no-skill` | do not write the skill |

`--agent-files-only` with both of them has nothing to write, and is an error.

## What to commit

Commit `AGENTS.md` and the skill copies with your suite, so an agent that opens a fresh checkout finds them. They contain no secrets and no machine-specific paths.

## Related

* [Commands](../reference/commands.md#evalproof-init) lists every `init` flag.
* [In CI](ci.md) covers the workflows `init` writes.
* [Testing agents](testing-agents.md) is for testing an agent you build, which is a different task from an agent using evalproof.
