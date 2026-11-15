// The system under test, written by `evalproof init --node`. `evalproof run evals/suite.yaml`
// starts this program once for each case: the case arrives as JSON on stdin, and runCommand
// prints the answer as JSON on stdout. Install the helper it imports with
// `npm install --save-dev evalproof` in the repository root.
//
// Try it without evalproof, from the repository root:
//   echo '{"id": "c1", "input": {"question": "How much does Pro cost?"}}' | node evals/system.mjs
import { runCommand } from "evalproof";

// A placeholder: replace the body with a call into your own code, such as your model call, agent
// or pipeline. It gets the text of the case's input and returns the answer as a string. It throws
// until you edit it, so every case errors instead of scoring an answer nobody wrote.
async function askModel(question) {
  throw new Error("evals/system.mjs: replace askModel() with a call into your own code");
}

// To send more than text, return an object instead of a string:
//   {output, retrieved, trace, tokens_in, tokens_out, cost_usd}
// `fromVercelAI(result)` and `fromLangChain(messages)`, exported by "evalproof" too, build that
// object (experimental) from an AI SDK result or a LangChain.js agent's messages.
await runCommand(async ({ input }) => {
  const question = typeof input === "string" ? input : (input.question ?? JSON.stringify(input));
  return askModel(question);
});
