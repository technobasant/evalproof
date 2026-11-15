# Import test cases from other tools

If your test cases already live in another eval tool, you do not have to write them again.
`evalproof import-dataset` reads them and writes a dataset, and with `--suite-out` a suite that has
exactly the scorers those cases need. `evalproof init --from FILE` does the same while it sets up a
repository. This page is about test cases, which hold no results. To gate results a tool already
produced, see [Import results from other tools](import.md).

```bash title="Terminal"
evalproof import-dataset auto promptfooconfig.yaml --out evals/cases.jsonl --suite-out evals/suite.yaml
evalproof run evals/suite.yaml --out evals/runs/latest.json
```

Importing makes no model call. It prints which format it detected, and one summary of what it could
not import, so nothing is dropped without you hearing about it.

## Formats

`auto` reads the format off the file. Name a format to skip detection.

| Format | What it reads | Expected answer becomes | Status |
|---|---|---|---|
| `promptfoo` | `promptfooconfig.yaml` (its `tests:`, and the files it points at with `file://`), or a YAML, JSON, JSONL or CSV file of tests | the assertions, see [promptfoo assertions](#promptfoo-assertions) | stable |
| `deepeval` | goldens in JSON, JSONL or CSV: `input`, `expected_output`, `context`, or a multi-turn `turns` and `expected_outcome` | `reference`, graded by an `llm_judge` scorer | experimental |
| `ragas` | samples with `user_input` and `reference`, or the older `question` and `ground_truth` | `reference`, graded by `llm_judge` | experimental |
| `langsmith` | examples with `inputs` and `outputs`, one per JSONL line or in a JSON list | `reference`, graded by `llm_judge` | experimental |
| `openai-evals` | `samples.jsonl` lines with `input` and `ideal`, or a dataset line wrapped in `item` | one `ideal`: `reference` with `exact_match`; a list: `accept` with `contains` | experimental |
| `inspect` | samples with `input`, `target` and `choices` in JSON, JSONL or CSV | a letter target with choices: `exact_match`; a list of targets: `contains`; free text: `llm_judge` | experimental |
| `braintrust` | records with `input` and `expected`, `metadata` and `tags` | `reference`, graded by `llm_judge` | experimental |
| `langfuse` | dataset items with `input`, `expected_output` and `id` | `reference`, graded by `llm_judge` | experimental |
| `phoenix` | examples with `input` and `output` objects | `reference`, graded by `llm_judge` | experimental |
| `adk` | Google ADK eval sets, see [the ADK section](import.md#google-adk-eval-sets-as-a-dataset) | tool checks and rubrics | stable |
| `jsonl`, `csv` | any table; you name the columns | `reference`, graded by `llm_judge` | stable |

Experimental means the reader was written from the tool's documented fields and fixtures written
by hand, because no real exported file was available to check against. It may refuse or misread a
shape it has not seen. It prints a note once, on stderr. `promptfoo`, `jsonl` and `csv` are
stable: the promptfoo reader follows promptfoo's published config and assertion types, and the two
generic readers do only what you tell them.

Free-text answers become a `reference` that the `llm_judge` scorer compares each reply with, not an
exact match, because an exact match on a sentence fails correct answers. If your references are
short labels, change that scorer to `exact_match` in the suite.

### Detection

| The file has | Detected as |
|---|---|
| a YAML suffix, `promptfooconfig` in the name, a `vars` key or a `__expected` column | `promptfoo` |
| `user_input`, or `question` with `ground_truth` | `ragas` |
| `inputs` | `langsmith` |
| `input` with `ideal`, or an `item` wrapper | `openai-evals` |
| `input` with `target` | `inspect` |
| `input` with `expected_output` | `deepeval` |
| `input` with `expected` | `braintrust` |
| `input` with an `output` object | `phoenix` |
| an ADK `eval_cases` set | `adk` |

DeepEval goldens and Langfuse items have the same keys, so `auto` reads them as `deepeval` and a
note says so. Name `langfuse` to use each item's `id` as the case id. A file that fits more than
one format, or none, is an error that lists the formats and asks you to name one: importing it as
the wrong format would give cases that look right and are not. A plain table of questions and
answers is `jsonl` or `csv`, never detected.

### Any table: `jsonl` and `csv`

```bash title="Terminal"
evalproof import-dataset jsonl qa.jsonl --out evals/cases.jsonl \
  --id-field qid --input-field question --input-field context \
  --reference-field answer --tags-field labels
```

`--input-field` names the column that holds the input. Repeat it, or give a comma list, and the
input becomes an object of those columns, with a `prompt_template` in the suite that puts them in
the prompt. With no `--input-field` the file's `input` column is used. `--reference-field` becomes
the reference, `--id-field` the case id and `--tags-field` the tags (a list, or a comma-separated
string). Other columns are not carried over. A column that does not exist is an error that lists
the columns found. These four flags belong to `jsonl` and `csv`; with another format they are a
usage error.

## What every reader does

* **Ids.** The source's own id when it has one. Otherwise `<format>-` and the first ten hex digits
  of a hash of the input and the expected answer, so an id does not depend on the line order and
  the same test keeps its id when the file is reshuffled. A repeated id gets `-2`, `-3` and a note.
* **Tags.** The source's tags, plus `imported` and `from:<format>`.
* **Chats.** System messages are dropped, with a note: set the system prompt in the suite. A chat
  that is then all user messages becomes the first as `input` and the rest as `turns`. Any other
  chat is kept as `input: {"messages": [...]}`, with a note that a system which takes text needs a
  prompt for it.
* **Past output is not test data.** What a run produced (`actual_output`, `response`,
  `retrieved_contexts`) is not carried over; the note counts it. To gate results, use
  [`evalproof import`](import.md).

## promptfoo assertions

promptfoo checks are all required, so several assertions on a case must all hold here too.
`defaultTest` is inherited, and `{{var}}` in an assertion value is filled from the test's `vars`.
In a CSV, `__expected`, `__expected1`, ... hold `type:value` (a bare value means `equals`).

| promptfoo | Case field | Scorer |
|---|---|---|
| `equals` (a string) | `reference` | `exact_match` |
| `contains`, `contains-any` | `accept` | `contains`, case-sensitive, any of |
| `icontains`, `icontains-any` | `accept` | `contains`, any of |
| `contains-all`, several `contains`, an `assert-set` of them | `require` | `contains`, all of |
| `icontains-all` | `require` | `contains`, all of |
| `starts-with`, `regex` | `pattern` (`^prefix`, escaped) | `regex` |
| `not-contains`, `not-icontains`, `not-contains-any` | `expect`: `output_omits` | `checks` |
| `not-regex`, `not-starts-with`, `not-equals` | `expect`: `not` of `output_regex` | `checks` |
| `is-json`, `contains-json` with a schema object | `schema` | `json_schema` |
| `llm-rubric`, `g-eval`, `model-graded-closedqa` | `rubrics` | `llm_judge` with `case_rubrics` |
| `factuality` | a rubric that names the reference | `llm_judge` with `case_rubrics` |

When a file mixes case-sensitive and case-insensitive assertions, each kind gets its own field
and scorer (`accept` and `accept_cs`, `contains` and `contains_cs`). The `checks` scorer imported
here skips a case with no `expect` instead of passing it.

### What is not imported

An assertion no scorer can express stays on the case under `unmapped`, exactly as the file had it,
and is counted in one summary line on stderr, for example
`12 assertions not imported: javascript x 8, similar x 4`. These are not imported:

* `javascript` and `python` assertions: running someone else's code is not something an import
  should do;
* `similar` (embedding similarity), `latency` and `cost`, which have no scorer here;
* `ends-with`, `not-contains-all`, `not-llm-rubric`, `not-is-json`, an `is-json` with no schema,
  and `equals` with a value that is not a string;
* an assertion with a `transform` or `contextTransform`, a value that comes from a file or a
  template, or `weight: 0` (promptfoo's informational check, which should not fail a case);
* an assertion's `threshold` and `weight` elsewhere, which are counted under "fields not carried
  over".

Prompts and providers in a promptfoo config are not imported either, with one exception: a single
inline prompt that only inserts variables becomes the suite's `prompt_template`.

### Example

```yaml title="promptfooconfig.yaml (excerpt)"
prompts:
  - "Answer the customer: {{question}}"
defaultTest:
  assert:
    - type: not-icontains
      value: as an ai language model
tests:
  - description: Refund window
    vars:
      question: Can I get a refund?
    assert:
      - type: contains-any
        value: ["30 days", "thirty days"]
      - type: regex
        value: "\\b30\\b"
      - type: javascript
        value: output.length < 300
```

```bash title="Terminal"
evalproof import-dataset auto promptfooconfig.yaml --out evals/cases.jsonl --suite-out evals/suite.yaml
# evalproof: detected format: promptfoo
# evalproof: note: 1 assertion not imported: javascript x 1 (kept on each case under 'unmapped')
# wrote 1 case(s) to evals/cases.jsonl
# wrote suite to evals/suite.yaml
```

```json title="evals/cases.jsonl"
{"id": "refund-window", "input": {"question": "Can I get a refund?"}, "tags": ["imported", "from:promptfoo"], "accept": ["30 days", "thirty days"], "pattern": ["\\b30\\b"], "expect": [{"type": "output_omits", "text": "as an ai language model"}], "unmapped": [{"type": "javascript", "value": "output.length < 300"}]}
```

## The suite file

`--suite-out` writes a suite with only the scorers the imported cases use, a `prompt_template`
when the input is an object, and a placeholder system you replace with your own (a model target, a
`python:` function or an HTTP endpoint, see [Providers](providers.md)). A `judge:` block is written
only when a scorer needs one, picked the way [`init`](../reference/commands.md#evalproof-init)
picks it. It never overwrites an existing file unless you pass `--force`, and it writes nothing
when the cases hold nothing a scorer can check. The existing `--out` file is replaced, as with the
other import commands.

Each scorer has a comment saying how many cases it checks, since an assertion in one test reaches
only that case. The gate is set on the scorer that checks the most cases: a gated metric needs
`min_cases` paired cases of its own (20 by default), so add another scorer to `gate.metrics`
only once it checks that many.

## Start a repository from existing cases

```bash title="Terminal"
evalproof init --from promptfooconfig.yaml
```

`init --from FILE` (with `--from-format FORMAT`, default `auto`) writes `evals/cases.jsonl` from the
import and `evals/suite.yaml` from the scorers the cases need, in place of the three sample cases.
Everything else `init` writes is unchanged, and files that exist are kept unless you pass
`--force`. Nothing is written if the file cannot be imported. `--from` cannot be combined with
`--agent-files-only`.

Then run the suite once, commit `evals/runs/latest.json`, and change one thing at a time as in
[Get started](../getting-started/index.md).
