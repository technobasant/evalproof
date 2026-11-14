---
description: Every model target evalproof can call, the key it reads, and how to use a logged-in Claude Code or Codex as the judge.
---

# Providers

The target of a system or a judge is `<provider>:<model>`, and this page lists every provider evalproof can call and how to configure it. It is for anyone choosing a model, pointing evalproof at a gateway, or using a logged-in coding agent as the judge.

The model name is everything after the first colon, so
`openrouter:anthropic/claude-sonnet-4.5` and `ollama:qwen3:8b` both work. There are two native adapters
(`ollama`, `anthropic`) and one OpenAI-compatible adapter driven by a table of provider
profiles ([`providers/profiles.py`](https://github.com/technobasant/evalproof/blob/main/src/evalproof/providers/profiles.py), each row cites the
provider's documentation). Plain HTTP throughout, no vendor SDK. Two more targets run a coding
agent you already have logged in, `claude-code` and `codex`: see
[Use your Claude Code or Codex login](#use-your-claude-code-or-codex-login). Three experimental
ones run OpenCode, Cursor and GitHub Copilot the same way: see
[Experimental providers](#experimental-providers).

## Every target

| Provider | Example target | Key variable | Where |
|---|---|---|---|
| `ollama` | `ollama:qwen3:8b` | none | local |
| `vllm` | `vllm:meta-llama/Llama-3.1-8B-Instruct` | `VLLM_API_KEY`, optional | local |
| `lmstudio` | `lmstudio:qwen3-8b` | optional, via `api_key_env` | local |
| `llamacpp` | `llamacpp:qwen3-8b` | `LLAMA_API_KEY`, optional | local |
| `anthropic` | `anthropic:claude-sonnet-4-5` | `ANTHROPIC_API_KEY` | hosted |
| `openai` | `openai:gpt-5` | `OPENAI_API_KEY` | hosted |
| `azure` | `azure:<deployment>` | `AZURE_OPENAI_API_KEY` | hosted, needs `base_url` |
| `gemini` | `gemini:gemini-2.5-flash` | `GEMINI_API_KEY` | hosted |
| `mistral` | `mistral:mistral-large-latest` | `MISTRAL_API_KEY` | hosted |
| `groq` | `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` | hosted |
| `together` | `together:meta-llama/Llama-3.3-70B-Instruct-Turbo` | `TOGETHER_API_KEY` | hosted |
| `fireworks` | `fireworks:accounts/fireworks/models/llama-v3p3-70b-instruct` | `FIREWORKS_API_KEY` | hosted |
| `deepseek` | `deepseek:deepseek-chat` | `DEEPSEEK_API_KEY` | hosted |
| `xai` | `xai:grok-4` | `XAI_API_KEY` | hosted |
| `openrouter` | `openrouter:anthropic/claude-sonnet-4.5` | `OPENROUTER_API_KEY` | hosted |
| `cerebras` | `cerebras:gpt-oss-120b` | `CEREBRAS_API_KEY` | hosted |
| `openai-compatible` | `openai-compatible:<model>` | optional, via `api_key_env` | any server, needs `base_url` |
| `claude-code` | `claude-code:haiku` | none: your Claude Code login | local CLI, see below |
| `codex` | `codex:gpt-6-luna` | none: your ChatGPT login | local CLI, see below |
| `opencode` | `opencode:ollama/qwen3:8b` | none: OpenCode's own login, per model provider | local CLI, experimental |
| `cursor` | `cursor:auto` | none: your Cursor login | local CLI, experimental |
| `copilot` | `copilot:auto` | none: your GitHub Copilot login | local CLI, experimental |

## Pointing a target at your own endpoint

The `system` and `judge` blocks take these keys to reach a gateway or a regional endpoint:

```yaml title="suite.yaml"
system:
  target: openai-compatible:gpt-4o    # a gateway, LiteLLM proxy, TGI, SGLang, ...
  base_url: https://llm-gateway.internal/v1
  api_key_env: TEAM_GATEWAY_KEY       # the variable to read; the key itself never goes in the suite
  headers: {x-team: search}           # extra HTTP headers on every call, never secrets
  params: {max_tokens: 2048}
```

* `base_url` overrides any profile's default (a regional endpoint, a proxy, an older
  `llama-server`, which listened on port 8080 before October 2026 and now defaults to 9931).
* `api_key_env` replaces the profile's variable for any provider; a variable named there
  must be set.
* `headers` merge into every request and win over the adapter's own, for example
  `anthropic-version`. They are not part of the cache key, and a suite file is usually
  committed, so keep keys in the environment.
* Sampling: local servers (`ollama`, `vllm`, `lmstudio`, `llamacpp`) receive `temperature` and
  `seed` on every call, defaults included, so a local run repeats. Hosted providers receive
  them only when `params` sets them, because reasoning models reject a non-default
  temperature; Anthropic never receives a seed, and since its current models also reject
  `temperature`, it gets one only when the suite asks. `params.extra` adds any other body
  field, such as `{thinking: {type: adaptive}, output_config: {effort: low}}`.
* Thinking models spend output tokens on thinking inside `max_tokens`, which defaults to 512
  (changing it would change every cache key). Set `params.max_tokens` higher for them; a reply
  cut off before its first word, a refusal, or a content-filter block is recorded as that
  case's error rather than as an empty answer.
* AWS Bedrock and Google Vertex AI sign requests with cloud credentials, which evalproof does
  not implement. Use a gateway's OpenAI-compatible endpoint through `openai-compatible`.

## Use your Claude Code or Codex login

If Claude Code or Codex is already logged in on your machine, either can be the judge or the
system under test, with no API key:

=== "Claude Code"

    ```yaml title="suite.yaml"
    judge:
      target: claude-code:haiku        # an alias (haiku, sonnet, opus) or a full model id
    concurrency: 4
    ```

=== "Codex"

    ```yaml title="suite.yaml"
    system:
      target: codex:gpt-6-luna
    concurrency: 4
    ```

Run `evalproof doctor suite.yaml` first. It prints each CLI's path, version, login mode and
the isolation flags it will pass, and exits 1 if a target of the suite is not usable. Without a
suite it surveys every CLI it knows that is installed, and exits 1 only when none is usable. It
makes no model call.

evalproof runs your own `claude` or `codex` binary as a subprocess, sends the prompt on stdin
and reads the JSON it prints. It never reads a token file or the keychain. The environment
variable `EVALPROOF_CLI_AUTH` says which credentials the CLI uses:

* `login` (the default, also when it is unset or empty) removes the CLI's key variables from
  the call (`ANTHROPIC_API_KEY` and `ANTHROPIC_AUTH_TOKEN` for Claude Code, `OPENAI_API_KEY`
  and `CODEX_API_KEY` for Codex), so the login is what gets used.
* `api-key` leaves them in place, so the CLI uses its own key. This is the route for CI.
* Any other value is an error.

`api_key_env` in the suite works in either mode and wins: it passes that variable to the CLI as
its own key variable. The mode is not part of the run's fingerprint or the cache key, because
it is the same model either way. `evalproof doctor` prints the mode in effect and warns when
`api-key` is chosen but the CLI's key variable is not set.

* **Isolation is a requirement, not an option.** A coding agent is not a model endpoint. Left
  alone it loads the project's `CLAUDE.md` or `AGENTS.md`, your memory, plugins, MCP servers
  and skills into the prompt, and it can run commands. A judge that has read your project's
  instructions is not a judge. So every call runs in a fresh empty directory, with the
  agent's own system prompt replaced by yours (or one neutral line), its tools, settings, MCP
  servers and skills switched off, and no permission-bypass flag. `evalproof doctor` prints
  the exact flags. The two CLIs do not isolate equally well:
    * Claude Code is run with **no tools at all** (`--tools ''`).
    * Codex runs in a read-only sandbox with its shell, plugins, apps, hooks, multi-agent mode,
    sleep and image tools, bundled skills and your user config and rules switched off, but
    codex-cli 0.167.2 has no flag that removes every tool. Still offered are a JavaScript
    isolate (`exec`: no Node, no file system, no network, with a few built-in helpers such as
    the current time, MCP resource listing and `apply_patch`), sub-agents that get the same
    restricted tools, and a question to the user, which cannot be answered in exec mode. **Any
    tool use fails the call**: if the output holds an item that is not a message, reasoning,
    an error or a plan, evalproof raises `codex used a tool (<type>); isolation broken` and
    discards the reply. A codex too old to know a flag evalproof sets exits with `Unknown
    feature flag`, and evalproof says to upgrade it (it was tested with codex-cli 0.167.2).
* **No temperature, seed or token limit.** Neither CLI exposes one, so `temperature`, `seed`
  and `max_tokens` are ignored. `think` maps to Codex's reasoning effort (`false` is `low`,
  `true` is `medium`; `params.extra.reasoning_effort` sets it directly). A run is therefore
  not repeatable the way a seeded local run is, so the run records the CLI, its version, the
  isolation scheme and the model id that actually answered. The cache does not key on the CLI
  version, so a CLI that updates itself keeps its cached replies; `--no-cache` asks again. To
  see how much a judge wobbles, score the same outputs twice with `--no-cache`, or
  [calibrate](../reference/concepts.md#concepts) it against labels.
* **A changed judge is not comparable.** `compare` and the gate refuse (exit 4) when the judge's
  target, CLI, isolation or resolved model differs between the two runs, because the scores
  then come from a different instrument. Pass `--allow-judge-change` or set
  `gate.allow_judge_change: true` to accept it. A new judge CLI version alone only warns, since
  the resolved model is what decides whether the instrument changed; so does a changed system
  target or CLI version, since that is usually the change under test. Runs written before 0.4.0
  carry none of these keys and compare as before.
* **Latency and cost.** A call takes about 1.4 s on `claude-code:haiku` and about 4 s on
  Codex, mostly start-up, so set `concurrency`. There is no API bill; the calls count against
  your subscription's usage limits. `claude-code` also reports a cost for each call
  (`total_cost_usd`), and when the suite gives no `pricing` for that target the run records it
  as the call's cost, so `max_cost_usd` and the report's cost total work. Under a subscription
  login this is the CLI's API-equivalent estimate, not a bill: nothing is charged to you for it.
  A `pricing` block in the suite, when it has non-zero prices, takes precedence. Judge calls
  count too, retries included.
* **Terms.** OpenAI documents Codex CLI use on a ChatGPT plan, including automation.
  Anthropic's terms allow a user to run the unmodified Claude Code binary on their own
  subscription; its plan limits assume ordinary individual use, and it forbids third-party
  products from routing other users' requests through subscription credentials. evalproof
  never touches credentials: it runs your own `claude` binary, on your machine, for your own
  evals. For CI or team-wide use, use an API key (`EVALPROOF_CLI_AUTH=api-key`, or
  `api_key_env`). See Anthropic's
  [legal and compliance page](https://code.claude.com/docs/en/legal-and-compliance). Nothing
  uses a CLI unless a suite names a `claude-code:` or `codex:` target (or an experimental one,
  below).

### Experimental providers

!!! warning "Experimental"

    `opencode:`, `cursor:` and `copilot:` follow the same rules as the two above: your own logged-in
    binary, run as a subprocess in a fresh directory in the system temp directory (empty but for
    Cursor's permission file), its API-key variable removed from the child's environment unless
    `api_key_env` names one, a bounded runtime, and the CLI, its version and an isolation tag
    (`opencode/0`, `cursor/2`, `copilot/2`) in the run and in the cache key. They are experimental
    because their isolation is less proven than Claude Code's and Codex's. Each prints
    `evalproof: <name> provider is experimental` on stderr the first time it is used in a process.
    Like the other two they ignore `temperature`, `seed` and `max_tokens`, and `evalproof doctor`
    without a suite lists each of them whose binary is on `PATH`. One difference from the two
    above: Cursor and Copilot have no system-prompt option, so they keep their own agent system
    prompt and your system text goes in front of the prompt under a heading, as an instruction in
    the user turn.

What was checked, and what was not:

* **`opencode:<provider>/<model>`** runs `opencode run --standalone --archived --format json`
  with a custom agent that has every tool off and every permission denied, your system text as
  that agent's prompt, and project config, the global config directory and the Claude Code
  compatibility files switched off. The prompt is on stdin. *Verified* only against local
  ollama models (`qwen3:8b`, a handful of calls): no tool event, a canary `AGENTS.md` in the
  project not read, a canary `AGENTS.md` in the global config directory read by default and not
  read with these settings, the parser and the error path. *Not verified:* any hosted model;
  whether `~/.claude/CLAUDE.md` is read (that test was inconclusive); whether the empty
  `OPENCODE_CONFIG_DIR` hides a login stored there; the token counts (input adds cache reads
  and writes, output adds reasoning, summed per field name). OpenCode has no single login, so
  `doctor` prints `auth: not checked`. Only `OPENCODE_API_KEY` is removed from the environment:
  the key of the model's own provider (`ANTHROPIC_API_KEY` for `opencode:anthropic/...`) is
  left, because naming that provider in the model id is choosing its key, and removing it would
  break CI. `--archived` keeps a session out of OpenCode's lists but does not delete it, so
  sessions accumulate in OpenCode's data directory (each run is a private `--standalone`
  server in a temporary project); prune them yourself.
* **`cursor:<model>`** runs `cursor-agent -p --output-format stream-json --mode ask --trust`
  with a `--workspace` that holds only a `.cursor/cli.json` permission file (nothing allowed;
  reads, writes and shell commands denied), and with `CURSOR_CONFIG_DIR` and `CURSOR_DATA_DIR`
  pointed at empty directories in the call's temp root. *Verified live* on 2026-11-13 with
  cursor-agent 2026.11.11:

    * Ask mode has no write or shell tool, but it is **not** tool-free: it has a read tool that
    runs without approval and reads files inside the workspace and outside it by absolute path.
    The permission file made every read fail with "Permission denied", inside and outside, but
    the model still tries, so the stream carries a `tool_call` event; evalproof fails the call
    on any such event (`cursor used a tool (readToolCall); isolation broken`) and discards the
    reply. The write and shell entries in the deny list are a second guard for later versions;
    only reads were tested.
    * Cursor reads instruction files (`AGENTS.md`, `CLAUDE.md`, `.cursor/rules`) from the
    workspace **and from the directories above it**. The workspace is empty and sits in the
    system temp directory, so nothing is found there unless `TMPDIR` points inside a project.
    `~/.claude/CLAUDE.md` was not read.
    * **Your skills list is sent.** Cursor puts the names and descriptions of the skills under
    your home directory (`~/.claude/skills`, `~/.cursor/skills`, `~/.codex/skills`,
    `~/.agents/skills`, plugin skills) and its own subagents in every prompt. The paths are
    fixed and no setting turns them off; moving `HOME` logs the CLI out. They count as input
    tokens on every call, and `doctor` warns about it. Verified with a login; with
    `CURSOR_API_KEY` it was not tested.
    * The redirected config and data directories keep the login (it is in the system keychain),
    keep each call's chat out of `~/.cursor/chats` and `~/.cursor/projects`, and mean your own
    `~/.cursor/cli-config.json` (and its allow list) is not read. A fresh config directory gets
    a default allow list with `Shell(ls)`, so evalproof writes one that allows nothing.

    `stream-json` rather than `json` because the json result joins the model's preamble text
    ("I'll read the file now.") with the answer and carries no tool events. Token counts come from
    the result event (output includes reasoning). The model recorded is the one you asked for;
    the result carries no model id. The prompt is an argument (nothing says the CLI reads stdin),
    so it is visible in `ps` while the call runs and a prompt over 100,000 bytes is refused. The
    key variable is `CURSOR_API_KEY`, passed through the environment, never as `--api-key`.
    `doctor` reads `cursor-agent about` for the login state (`status` hung) and prints no account.

* **`copilot:<model>`** runs `copilot --no-auto-update --no-ask-user --no-custom-instructions
  --disable-builtin-mcps --available-tools=evalproof-no-tools --output-format json
  --usage-output-file <tmp>`, with the prompt on stdin and `COPILOT_HOME` pointed at an empty
  directory in the call's temp root. *Verified live* on 2026-11-13 with copilot 1.0.110:

    * An empty allow-list (`--available-tools=`) is **ignored**: the model called its `view` tool
    and read a file. An allow-list of one name that is no tool disables every tool
    (`session.info: Disabled tools: bash, create, edit, glob, grep, ...`), and that is what
    evalproof passes. Any `tool.execution*` event or tool request in the output also fails the
    call (`copilot used a tool (view); isolation broken`); with tools off it never fired live,
    so that guard is tested only against canned output. No `--allow-*` flag is ever passed.
    With no tools the model sometimes writes a made-up tool call and its "result" as plain
    text; that is reply text, not a tool run, and is scored as such.
    * `--no-custom-instructions` works: without it a canary `AGENTS.md`, `CLAUDE.md` and
    `.github/copilot-instructions.md` in the working directory were all read; with it none was,
    nor an `AGENTS.md` in the directory above. No skills or MCP servers were offered.
    * Copilot keeps its own coding-agent system prompt and adds the date and a note that it is
    not sandboxed; your system text is an instruction in the user turn.
    * `COPILOT_HOME` in the temp root keeps the login (it is in the system keychain) and keeps
    each call's session, logs and config out of `~/.copilot` (verified: no new file there after
    a call). Your own Copilot config, plugins and MCP servers live in that directory, so they
    are not used either; that follows from the redirect and was not tested separately.
    * Models and cost: on the test account named models (`gpt-6-luna`, `claude-haiku-5.5`) were
    rejected with "is not available", and only `copilot:auto` worked; which models you can name
    depends on your plan and your organisation's policy. `auto` routes each call to a model of
    Copilot's choosing (`gpt-6-luna` on one call, `mai-code-1.1-flash` on another), so the run
    artifact records every model that answered, not `auto`, and a suite run on `auto` can mix
    models. `auto` used one premium request per call there; the rate depends on the model and
    the plan.
    * **`copilot:auto` makes a poor judge.** The judge's resolved model is part of what
    [`compare` and the gate check](#use-your-claude-code-or-codex-login): two runs on `auto` often saw
    different models, so they are refused as a judge change (exit 4) unless you allow it.
    `doctor` warns about it. Name a model for the judge if your plan allows one.

    Token counts are read from the usage file (input already includes cache reads and writes,
    output includes reasoning); a missing file means 0, never a failure. Only
    `COPILOT_GITHUB_TOKEN` is removed; `GH_TOKEN` and `GITHUB_TOKEN` are left because `gh` and CI
    set them for other reasons. `doctor` prints `auth: not checked`.

## Next steps

* Check what each target needs in CI: [What it needs to run](../getting-started/credentials.md).
* Put a target to work: [In CI](ci.md#in-ci) and [Testing agents](testing-agents.md#testing-agents).
