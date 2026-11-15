# evalproof for npm

This package does two things for a JavaScript or TypeScript project:

* It runs [evalproof](https://basantbhattarai.com.np/evalproof/), the LLM eval and CI regression
  gate, from `npx`. evalproof is a Python program published on
  [PyPI](https://pypi.org/project/evalproof/), and the launcher runs the PyPI release that has the
  same version number.
* It is a small helper library for testing a system written in JavaScript or TypeScript: it
  connects your code to a suite as a `command:` or `http://` target. The library is
  experimental.

```bash
npm install --save-dev evalproof
npx evalproof init --node          # write a suite and evals/system.mjs that uses the library
npx evalproof compare base.json candidate.json --gate
```

Node 18 or newer. The package has no dependencies and ships its TypeScript types. It works from
ESM and from CommonJS.

## Test your code

`runCommand` is the program a `command:` target starts once for each case. It reads the case as
JSON from stdin, calls your function and prints the answer as JSON.

```js
// evals/system.mjs
import { runCommand } from "evalproof";

await runCommand(async ({ input }) => {
  const question = typeof input === "string" ? input : input.question;
  return await askMyAgent(question); // a string, or {output, retrieved, trace, tokens_in, ...}
});
```

```yaml
# evals/suite.yaml
system:
  target: "command:node system.mjs"   # or "command:npx tsx system.ts"
```

The command runs with the suite's directory as its working directory. The function gets
`{id, input, turns?, state?, variant?, vars?, trial?}`. If it throws, the message and stack go to
stderr, the process exits with code 1, and the case is recorded as an error. `console.log` writes
to stderr while the function runs, so it cannot corrupt the answer.

`serve` is the same function behind an HTTP endpoint, for a system you start yourself:

```js
import { serve } from "evalproof";

serve(async ({ input }) => askMyAgent(input), { port: 8787 }); // 127.0.0.1, path "/"
// suite: system: { target: "http://127.0.0.1:8787" }
```

### Return a trace

A scorer that checks which tools an agent called needs a trace. These build one from what the
framework returns. Both are experimental and print one line to stderr the first time they are
called.

```js
import { runCommand, fromVercelAI, fromLangChain } from "evalproof";

// Vercel AI SDK: pass a generateText result, or an array of results, one per turn
await runCommand(async ({ input }) =>
  fromVercelAI(await generateText({ model, tools, prompt: input.question })),
);

// LangChain.js: pass an agent's { messages }, or an array of messages
await runCommand(async ({ input }) =>
  fromLangChain(await agent.invoke({ messages: [{ role: "user", content: input.question }] })),
);
```

Each returns `{ output, trace, tokens_in, tokens_out }`.

The guide, [Testing a TypeScript or JavaScript system](https://basantbhattarai.com.np/evalproof/guides/typescript/),
covers the options, the HTTP responses and what `evalproof scan` finds in TypeScript.

## Run evalproof from npx

The launcher needs one of these, tried in this order:

1. [uv](https://docs.astral.sh/uv/getting-started/installation/): runs `uvx --from evalproof==VERSION evalproof ...`.
2. [pipx](https://pipx.pypa.io/): runs `pipx run --spec evalproof==VERSION evalproof ...`.
3. An `evalproof` on your `PATH` that reports exactly the same version, from `pip install evalproof==VERSION`.

If none of them is found, the launcher prints how to install uv and exits with 127. Set
`EVALPROOF_LAUNCHER` to `uvx`, `pipx` or `path` to use only one of them.

The launcher passes its arguments and its exit code through, and forwards `SIGINT` and `SIGTERM`
to evalproof on Linux and macOS. The library does not need the launcher or Python: it only reads
stdin and writes stdout, or serves HTTP.

## More

The documentation is at <https://basantbhattarai.com.np/evalproof/>. The source is in the
[`npm` directory](https://github.com/technobasant/evalproof/tree/main/npm) of the repository.
Licensed under Apache-2.0.
