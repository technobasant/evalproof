---
description: Test a system written in any language, such as Node.js, Go or a shell script, by running it as a command target that reads a case on stdin and prints the answer.
---

# Testing a system in any language

This page is for anyone whose system is not Python and does not sit behind an HTTP endpoint. A `command:` target runs a program once for each case. The program reads the case as JSON on stdin and prints the answer. It can be written in Node.js, Go, Rust, Java or shell, or it can be a command-line tool you already have.

```yaml title="suite.yaml"
system:
  target: "command:node eval/system.js --mode fast"
```

Everything after `command:` is split into words the way a POSIX shell splits it, on every platform. Quotes group words and a backslash escapes the next character. No shell runs the command, so pipes, redirects and `$VARIABLES` in the line do nothing. Use `/` in paths, even on Windows. Put single quotes around an argument that has a space or a backslash.

An empty command is an error when the suite loads.

## What the program receives

The program gets one JSON object on stdin, encoded as UTF-8. It is the same object an HTTP target is sent.

| Field | When it is there |
|---|---|
| `id` | always: the case id |
| `input` | always: the case input, a string or an object |
| `turns` | the case has later turns |
| `state` | the case has an initial state |
| `variant` | the run is a [variant](testing-agents.md#variants-one-suite-several-builds) |
| `vars` | the variant sets `vars` |
| `trial` | the suite runs each case more than once; counts from 0 |

The program inherits your environment. It runs with the suite's directory as its working directory, so a relative path in the program or its arguments is relative to the suite file, whatever directory you ran evalproof from.

## What the program prints

Print the answer on stdout as UTF-8. There are two forms.

Plain text is the answer. Leading and trailing whitespace is removed. This includes text that happens to be a number or a JSON array, such as `42`.

A JSON object is a result with more in it:

```json
{"output": "billing", "tokens_in": 120, "tokens_out": 4, "cost_usd": 0.0003}
```

| Key | Required | What it is |
|---|---|---|
| `output` | yes | the answer, as a string |
| `retrieved` | no | the passages a retrieval system used, as a list of strings |
| `trace` | no | what an agent did; see [Testing agents](testing-agents.md#what-the-system-returns-the-trace) |
| `tokens_in`, `tokens_out` | no | token counts, as numbers |
| `cost_usd` | no | what the call cost, in USD |

A figure the program leaves out is filled from the trace's usage, then from the target's `pricing`. Latency is the wall time of the process.

To return JSON as the answer itself, put it in `output` as a string. A JSON object without `output` is an error on that case.

Write logs and warnings to stderr. Only stdout is read as the answer.

## Errors and timeouts

A failure is recorded on the case as an error and the run goes on. The program gave no answer, so the case is counted as errored, not as failed.

| What happens | What the case records |
|---|---|
| the program exits with a code other than 0 | `command exited with code N:` and the last 300 characters of stderr |
| it runs longer than `timeout_s` (120 seconds unless the target sets it) | `command timed out after Ns` |
| it cannot start | the error the operating system gave |
| stdout is a JSON object with no `output`, or a token count that is not a number | a message naming the problem |

On a timeout evalproof kills the program and every process it started, then waits for it to exit. A program that starts a child, such as `npx` starting `node`, is stopped as a whole. Nothing is left running.

Set the limit on the target:

```yaml title="suite.yaml"
system:
  target: "command:node eval/system.js"
  timeout_s: 30
```

## Finding the program

The first word is the program. evalproof resolves it before the first case runs and stops with an error that names it when it cannot.

* A word with a path separator, such as `eval/system` or `./bin/system`, is a path. A relative one is relative to the suite's directory.
* A bare word, such as `node` or `npx`, is looked up on `PATH`. On Windows the lookup tries the extensions Windows runs, so `npx` finds `npx.cmd` and `eval/system` finds `eval/system.exe`.

Run [`evalproof doctor SUITE`](../reference/commands.md#evalproof-doctor) to see which file a suite's command resolves to. It also shows that the target uses no credential.

## Example: Node.js

A router that sends a support message to `billing` or `general`:

```js title="eval/system.js"
let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => (raw += chunk));
process.stdin.on("end", () => {
  const c = JSON.parse(raw);
  const label = /refund|invoice/i.test(c.input) ? "billing" : "general";
  process.stdout.write(JSON.stringify({ output: label }));
});
```

```yaml title="suite.yaml"
name: router
dataset: cases.jsonl
system:
  target: "command:node eval/system.js"
scorers:
  - {name: label, type: exact_match}
```

The dataset has one case per line, with the label to expect in `reference`:

```json title="cases.jsonl"
{"id": "r1", "input": "Please refund my last invoice", "reference": "billing"}
{"id": "r2", "input": "How do I change my avatar?", "reference": "general"}
```

## Example: Go

Build the program once, then point the target at the binary. Starting `go run` for every case would spend most of the run compiling.

```go title="eval/system/main.go"
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"regexp"
)

func main() {
	var c struct {
		Input string `json:"input"`
	}
	if err := json.NewDecoder(os.Stdin).Decode(&c); err != nil {
		fmt.Fprintln(os.Stderr, "bad input:", err)
		os.Exit(1)
	}
	label := "general"
	if regexp.MustCompile(`(?i)refund|invoice`).MatchString(c.Input) {
		label = "billing"
	}
	json.NewEncoder(os.Stdout).Encode(map[string]string{"output": label})
}
```

```bash title="Terminal"
go build -o eval/system ./eval/system
```

```yaml title="suite.yaml"
system:
  target: "command:eval/system"
```

## What it is not

A command target is not cached. The program runs for every case on every run, as a `python:` or `http://` target does. `--cache-only` refuses a suite whose system is a command, because the program would run.

The program is started once per case and per trial, so a program with a slow start makes a slow run. Set `concurrency` in the suite to run several at once.

## Trust

The command comes from the suite file, so it has the same trust as a `python:` target. Running a suite runs its command with your permissions and your environment. Read a suite from someone else before you run it.

Case data reaches the program only on stdin, never in the command line, and no shell is involved. On Windows a `.cmd` or `.bat` file, such as `npx.cmd`, is started through `cmd.exe`. That is acceptable for the same reason: the command line is the suite author's, and the data is not part of it.
