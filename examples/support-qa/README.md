# Support assistant for a fictional product, judged by a local model

**Quillfox** and the **Larkspur Software Cooperative** do not exist. Every name, price and
policy in [`knowledge_base.md`](knowledge_base.md) was invented for this example.

* **System under test:** `qwen3:1.7b` on Ollama, answering customer questions with the
  knowledge base in its system prompt.
* **Change under review:** the grounded prompt ([`prompt.baseline.md`](prompt.baseline.md):
  "answer using only the facts… if they don't answer it, say so") replaced by a
  friendlier one ([`prompt.candidate.md`](prompt.candidate.md): "always give them a
  complete answer"). The kind of edit that ships on a Friday.
* **Golden set:** 83 questions in [`cases.jsonl`](cases.jsonl), 9 of which the facts do not
  answer — the right reply is to say so.
* **Answer key:** a regex per question (`accept` must match, `reject` must not), written
  before any model output was seen and not changed afterwards. It **stands in for human
  review**; it is not human labelling, and [MEASUREMENTS.md](../../MEASUREMENTS.md) reports
  where it is wrong.
* **Judge:** `qwen3:8b` with the rubric in [`rubric.md`](rubric.md).

```bash
ollama pull qwen3:1.7b && ollama pull qwen3:8b
uv run evalproof run examples/support-qa/suite.yaml --out examples/support-qa/runs/baseline.json
uv run evalproof run examples/support-qa/suite.yaml \
  --system-prompt-file examples/support-qa/prompt.candidate.md \
  --label "qwen3:1.7b + friendly prompt" --out examples/support-qa/runs/candidate.json
uv run evalproof compare examples/support-qa/runs/baseline.json \
  examples/support-qa/runs/candidate.json --suite examples/support-qa/suite.yaml --gate
uv run evalproof calibrate examples/support-qa/runs/baseline.json \
  --judge-scorer judge --gold-scorer answer_key --gold-sample 30
# The runs in runs/ were recorded on cases.v1.jsonl; point the suite's `dataset` at it to
# reproduce them exactly. cases.jsonl is the revised set, whose runs are in runs/v2/.
# The reasoning judge (MEASUREMENTS §8): same cached answers, judge thinking on
uv run evalproof run examples/support-qa/suite.reasoning-judge.yaml \
  --out examples/support-qa/runs/baseline.reasoning-judge.json
uv run evalproof run examples/support-qa/suite.reasoning-judge.yaml \
  --system-prompt-file examples/support-qa/prompt.candidate.md \
  --label "qwen3:1.7b + friendly prompt" --out examples/support-qa/runs/candidate.reasoning-judge.json
uv run python examples/support-qa/measure.py   # every figure in MEASUREMENTS.md, from runs/
```

The committed `runs/` are the output of those commands on the machine and model digests
recorded in each run's `info.environment`.
