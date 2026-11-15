---
description: Test a TypeScript or JavaScript system with evalproof. Run it as a command target with runCommand, serve it as an HTTP target, turn a Vercel AI SDK or LangChain.js result into a trace, and scan it for untested tools.
---

# Testing a TypeScript or JavaScript system

This page is for anyone whose system runs on Node.js, Bun or Deno: an agent built with the Vercel AI SDK or LangChain.js, a pipeline that calls a model SDK, or a service you can start locally. evalproof is a Python program, so it reaches your code through a [command target](command-target.md) or an HTTP endpoint. The `evalproof` npm package gives you the small amount of code that sits between the two.

!!! warning "Experimental"

    The helper library (`runCommand`, `serve`, `fromVercelAI`, `fromLangChain`), `init --node` and the scan of TypeScript tools and agents are new and experimental: they can change in a minor release. See [Versioning](../about/versioning.md#the-public-surface).

## Install the helper

```bash title="Terminal"
npm install --save-dev evalproof
```

The package needs Node 18 or newer and has no dependencies. It ships its own TypeScript types, and it works from ESM (`import`) and from CommonJS (`require`). It also holds the `npx evalproof` launcher, which runs the Python program of the same version. Running a suite still needs evalproof itself, from `pip install evalproof` or through `npx evalproof`; see [Install](../getting-started/install.md).

## Start with `init --node`

```bash title="Terminal"
evalproof init --node
```

This writes a suite and a program, `evals/system.mjs`, that the suite starts once for each case:

```yaml title="evals/suite.yaml"
system:
  target: "command:node system.mjs"
```

The target is `node system.mjs`, not `node evals/system.mjs`, because a command runs with the suite's directory as its working directory. The program holds a placeholder that throws until you replace it with a call into your own code, so every case errors until you do. `init --node` is described under [`evalproof init`](../reference/commands.md#evalproof-init). With `--ci api-key` on GitHub, the workflow it writes also sets up Node and runs `npm ci`.

## Run your code with `runCommand`

`runCommand` reads the case from stdin, calls your function, and prints the answer as JSON on stdout. The function can be `async`.

```js title="evals/system.mjs"
import { runCommand } from "evalproof";
import { anthropic } from "@ai-sdk/anthropic";
import { generateText } from "ai";

await runCommand(async ({ input }) => {
  const question = typeof input === "string" ? input : input.question;
  const { text } = await generateText({
    model: anthropic("claude-haiku-5-5"),
    prompt: question,
  });
  return text;
});
```

```yaml title="evals/suite.yaml"
name: support
dataset: cases.jsonl
system:
  target: "command:node system.mjs"
scorers:
  - {name: mentions-refund, type: contains, params: {any_of_field: accept}}
```

```json title="evals/cases.jsonl"
{"id": "refund", "input": {"question": "Can I get my money back?"}, "accept": ["refund", "30 days"]}
```

The function receives the object the [command target](command-target.md#what-the-program-receives) sends: `id`, `input`, and `turns`, `state`, `variant`, `vars` and `trial` when the case or run has them. It returns one of two things:

* A string, which is the answer.
* An object with `output` (a string) and any of `retrieved`, `trace`, `tokens_in`, `tokens_out` and `cost_usd`. This is the same result a command target may print, so a [trace](testing-agents.md#what-the-system-returns-the-trace) can carry the tool calls for a `checks` scorer.

If the function throws, or returns something that is not a string or a valid result, `runCommand` prints the message and the stack to stderr and exits with code 1, and the case is recorded as an error. `console.log`, `console.info` and `console.debug` write to stderr while the function runs, so a stray log line cannot corrupt the answer on stdout.

To run a TypeScript file without building it first, use `tsx`. The command is split into words with no shell, as [Testing a system in any language](command-target.md) describes, so each word is an argument:

```yaml title="evals/suite.yaml"
system:
  target: "command:npx tsx system.ts"
```

Bun and Deno work the same way: `command:bun run system.ts` and `command:deno run --allow-net --allow-env system.ts`.

You can try the program without evalproof, from the repository root:

```bash title="Terminal"
echo '{"id": "c1", "input": {"question": "How much does Pro cost?"}}' | node evals/system.mjs
```

## Return a trace from a Vercel AI SDK or LangChain.js result

A string answer is enough for a scorer that reads the text. A scorer that checks which tools the agent called needs a [trace](testing-agents.md#what-the-system-returns-the-trace). `fromVercelAI` and `fromLangChain` build one from what the framework already returns. Both are experimental, and each prints one line to stderr the first time it is called.

```js title="evals/system.mjs"
import { runCommand, fromVercelAI } from "evalproof";
import { anthropic } from "@ai-sdk/anthropic";
import { generateText, tool } from "ai";
import { z } from "zod";

const tools = {
  lookup_order: tool({
    description: "Find an order by its id",
    inputSchema: z.object({ id: z.string() }),
    execute: async ({ id }) => ({ id, status: "shipped" }),
  }),
};

await runCommand(async ({ input }) => {
  const result = await generateText({
    model: anthropic("claude-haiku-5-5"),
    tools,
    stopWhen: ({ steps }) => steps.length >= 5,
    prompt: input.question,
  });
  return fromVercelAI(result);
});
```

`fromVercelAI(result)` returns `{ output, trace, tokens_in, tokens_out }`. The tool calls come from each step's `toolCalls` and `toolResults`, a tool that failed becomes a call with `ok: false`, and the token counts are summed over the steps. For a case with later turns, call `generateText` once per turn and pass an array of the results, one per turn.

```js title="evals/system.mjs"
import { runCommand, fromLangChain } from "evalproof";
import { createAgent, tool } from "langchain";
import { z } from "zod";

const lookupOrder = tool(async ({ id }) => JSON.stringify({ id, status: "shipped" }), {
  name: "lookup_order",
  description: "Find an order by its id",
  schema: z.object({ id: z.string() }),
});

const agent = createAgent({ model: "anthropic:claude-haiku-5-5", tools: [lookupOrder] });

await runCommand(async ({ input }) => {
  const state = await agent.invoke({ messages: [{ role: "user", content: input.question }] });
  return fromLangChain(state);
});
```

`fromLangChain` takes the `{ messages }` an agent returns, or an array of messages. It reads the tool calls on each AI message, the matching tool messages (an error status marks the call as failed), and `usage_metadata` for tokens. A human message starts the next turn.

A trace changes what you can score. With `tool_called` and `tool_order` checks, a case passes or fails on what the agent did, not only on what it said. See [Testing agents](testing-agents.md).

## Serve your code over HTTP

If you would rather start your system once and have evalproof call it, use an [HTTP target](../reference/configuration.md) and `serve`:

```js title="server.mjs"
import { serve } from "evalproof";

serve(async ({ input }) => answer(input), { port: 8787 });
```

```yaml title="evals/suite.yaml"
system:
  target: "http://127.0.0.1:8787"
```

`serve` listens on `127.0.0.1` at port 8787 and path `/` unless you pass `port`, `host` or `path` (`port: 0` picks a free port). It returns the `node:http` server, already listening. It accepts a POST with the same JSON object a command target gets on stdin, and answers with the same result. The function is the same as for `runCommand`, so a function can serve both. This is experimental.

| Request | Response |
|---|---|
| a body that is not a JSON object | 400 with `{"error": ...}` |
| any other path | 404 |
| any method other than POST | 405 |
| a body over 10 MiB | 413 |
| the function throws | 500 with `{"error": ...}`, and the case records the error |

evalproof does not start the server. Start it before `evalproof run`, and stop it afterwards. In a CI job, start it in the background and wait until it answers before the run step. The server binds to localhost by default so that it is not reachable from other machines; pass `host` only when you need that.

An HTTP target costs more to set up than a command target and is no better at scoring. Prefer `runCommand` unless the system is already a service, or starting it for every case would be slow.

## Find what is not tested

`evalproof scan` reads TypeScript and JavaScript files as well as Python, and does not run them. It finds model calls from the common SDKs, Vercel AI SDK tools (`tool({ description })`, named by their key in `tools`), LangChain.js tools (`tool(fn, { name, description })`, `DynamicStructuredTool`) and agents (`createAgent`, `createReactAgent`, the Vercel AI SDK agent classes, `stopWhen` and `maxSteps`).

It also follows a suite whose target is a `node`, `tsx` (or `npx tsx`), `bun` or `deno` command. It starts from the script, relative to the suite's directory, and follows its static imports with a relative path, including `.js` imports that mean a `.ts` file and `index` files. Files it reaches count as tested; the rest are reported as `uncovered_code`.

The scan prefers saying it does not know to guessing. If the code a suite runs has an import it cannot resolve to a file in the repository, a path alias (`@/`, `~/`, `#` or a `paths` entry in `tsconfig.json`), a workspace package, a module loaded by a computed name, or if the command runs an inline script or a script that is not among the scanned files, the reach is unknown, and the scan reports `uncovered_code` as not computed and says why. A package from `node_modules` is not followed. Test files (`*.test.ts`, `*.spec.js`, `__tests__`) are not reported as untested code.

```bash title="Terminal"
evalproof scan . --out scan.json
```

See [Find what is not tested](find-gaps.md) for the gaps it reports and `evalproof draft`.

## What to know

* **The program runs for every case.** A command target is not cached, and a program with a slow start makes a slow run. Set `concurrency` in the suite to run several at once. See [What it is not](command-target.md#what-it-is-not).
* **The suite's command is trusted code.** Running a suite runs its command with your permissions. See [Trust](command-target.md#trust).
* **A converter reads documented fields, and a framework can change them.** If a converter returns a trace with no tool calls for an agent that used tools, check the result's shape against the framework version you run, or build the trace by hand as a `trace` object.

## Next steps

* The command target's input, output and errors in full: [Testing a system in any language](command-target.md).
* Check which tools an agent called: [Testing agents](testing-agents.md).
* Scan a repository for untested code: [Find what is not tested](find-gaps.md).
* The flags of `init --node` and `scan`: [Commands](../reference/commands.md).
