---
description: Find the model calls, agents and tools in a repository that no suite tests, and draft inputs for them, without a model ever writing an expected answer.
---

# Find what is not tested: `evalproof scan` and `evalproof draft`

`evalproof scan` reads a repository and lists what its code does with a model and what no suite tests yet. `evalproof draft` turns those gaps into a file of draft cases. Both are experimental: their output and flags can change in a minor release, and each prints a one-line notice the first time it runs.

The rule behind both is that a model may suggest what to ask, never what the right answer is. A draft is an input. The expected answer comes from your code, your documentation, a test someone already wrote, or a person.

## Scan a repository

```bash title="Terminal"
evalproof scan . --out scan.json
```

The scan makes no model call and imports and runs none of your code. It parses Python with `ast` and reads TypeScript and JavaScript with patterns, so it finds the common SDKs and agent frameworks by how they are imported and called, not by running them. Hidden directories, `node_modules`, virtual environments and build output are skipped.

```text title="Output"
evalproof scan: . (6 files read)

suite (1)
  evals/support.yaml:1  support: 3 cases, target python:app.tools:x

model_call (1)
  app/llm.py:4  [anthropic]  client.messages.create(model='m')

tool (2)
  app/tools.py:3  [openai-agents]  lookup_order: Find an order.
  app/tools.py:6  [openai-agents]  refund: Refund it.

test_cases (1)
  tests/test_llm.py  [pytest]  pytest file; imports app/llm.py

gaps (4)
  uncovered_code app/llm.py:4: no suite's target runs this file's model call, agent or tool
    next: evalproof draft --scan scan.json --out evals/drafts.jsonl --target MODEL --gap uncovered_code --file app/llm.py
  tool_unchecked app/tools.py:6: no check in any suite names the tool `refund`
    next: evalproof draft --scan scan.json --out evals/drafts.jsonl --target MODEL --gap tool_unchecked --tool refund
  test_cases_not_imported tests/test_llm.py: pytest assertions over this code are test cases evalproof does not have
    next: evalproof draft --scan scan.json --out evals/drafts.jsonl --from-tests-only
  few_cases evals/support.yaml: evals/support.yaml has 3 case(s); its gate wants at least 20
    next: evalproof draft --scan scan.json --out evals/drafts.jsonl --target MODEL --gap few_cases --suite evals/support.yaml --per-gap 17
```

A finding is something the code is or has: a model call, an agent, a tool, a prompt, evidence that it keeps a conversation going, a file of test cases, or an evalproof suite. A gap is a finding no suite covers:

| Gap | What it means | What fills it |
|---|---|---|
| `no_suite` | the repository has LLM code and no evalproof suite | `evalproof init`, then write cases |
| `uncovered_code` | a file with a model call, an agent or a tool that no suite's target reaches | inputs from a model |
| `tool_unchecked` | a tool that no `tool_called`, `tool_order` or similar check in any suite or case names | inputs from a model, and a person adds the check |
| `no_multi_turn_cases` | the code keeps a conversation, and no case has `turns` | conversations from a model |
| `test_cases_not_imported` | test cases in another format: a pytest file, promptfoo, DeepEval, Ragas or the like | `evalproof draft` for pytest, `evalproof import-dataset` for the others |
| `few_cases` | a suite with fewer cases than its gate's `min_cases` | inputs from a model |

"Reaches" means the scan followed imports between files in the repository, starting from the suite's `python:module:function` target. If a suite's target is a command, an endpoint, a model, or Python code outside the repository, the scan cannot know what it runs, so it says `uncovered_code` was not computed instead of listing every file. It prints that reason; it does not guess.

The exit code is 0 whether or not there are gaps. A gap is advice, not a gate. Pass `--json` to print the JSON instead of text, `--exclude GLOB` to skip paths, and `--suite FILE` to name the suites yourself.

## Draft cases from tests you already have

```bash title="Terminal"
evalproof draft --scan scan.json --out evals/drafts.jsonl --from-tests-only
```

A pytest file that calls your code with a literal input and asserts on the result already holds a test case. `draft` reads the file without importing or running it, and turns these shapes into cases:

```python title="tests/test_llm.py"
def test_window():
    assert answer("What is the refund window?") == "30 days"  # reference


def test_contact():
    result = answer("How do I reach you?")
    assert "email" in result  # accept
    assert re.search(r"[a-z]+@", result)  # pattern
```

An exact comparison becomes a `reference`, `"text" in result` becomes an `accept` entry, and `re.search` becomes a `pattern` that keeps its flags. Calls through a module (`llm.answer(...)`), `await`, `.lower()` and `.strip()` on the result, `and` chains and a result held in a variable are followed. The function must be imported from a file in the repository that reaches a model call, and the input must be a string literal (the first argument, or a keyword such as `question=`). Anything else is skipped, not guessed. Each case records `source`, the test it came from, and is tagged `draft` and `from:pytest`. No model is called, so no `--target` is needed.

If two tests give the same input two different exact answers, the first is kept and a warning names the input. An input that is already in a suite is skipped.

## Draft inputs with a model

```bash title="Terminal"
evalproof draft --scan scan.json --out evals/drafts.jsonl --target ollama:qwen3:8b
```

For the gaps that are about missing inputs (`tool_unchecked`, `no_multi_turn_cases`, `uncovered_code` and `few_cases`) a model writes inputs. The model is `--target`, or the judge of the suite when there is exactly one, and the command exits 2 and says so when there is neither.

What the model is sent depends on the gap: a tool's name, signature and docstring; the docstring and signature of the function that holds the uncovered model call; a short excerpt of the application's prompt; and up to twelve inputs the suite already has, so it writes different ones. It is asked for a JSON array of strings, or an array of arrays for a conversation. A reply that is not that array is retried once with the bad reply shown, and then the gap is skipped with a warning. Duplicates, empty strings and inputs a suite already has are dropped. Replies go through the same response cache as a run, so repeating a draft costs nothing; `--no-cache` skips it. If the code is private, use a local model for `--target`.

A drafted case has an input and nothing that says what a right answer is:

```json title="evals/drafts.jsonl"
{"id": "draft-3f9a0c1d52", "input": "My order 4512 arrived broken, can I get my money back?", "tags": ["draft", "gap:tool_unchecked", "from:model"], "draft_for": {"gap": "tool_unchecked", "tool": "refund"}, "drafted_by": "ollama:qwen3:8b"}
```

It has no `reference`, `accept`, `require`, `pattern`, `schema` or `expect`, and the command never adds one.

## Turn drafts into cases

The drafts file is not a dataset. `evalproof draft` writes it to its own path and refuses to overwrite an existing file unless you pass `--force`. Before a draft goes into a suite's dataset:

1. Read it. Delete inputs you would never want tested, and fix any that do not look like what your users send.
2. Add the expected answer or checks yourself. For a tool, add an `expect` entry such as `{"type": "tool_called", "name": "refund"}`. For an answer, add a `reference`, an `accept` list or a `pattern`. Take them from your documentation or your code, not from the model that wrote the input.
3. Read a draft that came from pytest as carefully as one from a model. It carries the assertion its test made, which was only as good as that test.
4. Move the cases into the suite's dataset, drop the `draft` tag if you do not want it kept, and run `evalproof run`.

`--per-gap` sets how many inputs to ask for each gap (default 3) and `--max-cases` caps the file (default 30). `--gap`, `--tool` and `--file` narrow the run to one kind of gap, one tool or one file; the commands the scan prints already carry the right ones.

## Next steps

* The flags in full: [Commands](../reference/commands.md#evalproof-scan).
* See which parts of quality a suite covers: [Readiness](readiness.md).
* Bring test cases from another tool: [Import test cases](import-test-cases.md).
* Check tool calls in a case: [Testing agents](testing-agents.md).
