# Retrieval example

Offline example that exercises the retrieval scorers (`recall_at_k`, `mrr`, `ndcg`) end-to-end with a small labelled corpus and two deterministic retrievers.

No models or network required. Suitable for CI.

## Files

- `cases.jsonl` — 20 queries with graded relevance judgements.
- `retrievers.py` — two systems:
  - `good_retriever`: ranks by simple term overlap with the query.
  - `noisy_retriever`: same as good but randomly demotes the top relevant document on half the queries (simulating a regression).
- `suite.yaml` — configures the good retriever and gates on nDCG@5.

## Run

```bash
uv run evalproof run examples/retrieval/suite.yaml --out runs/retrieval-good.json --quiet
# For the noisy (regressed) version, temporarily change the system target in suite.yaml
# or run with a modified suite.
uv run evalproof compare runs/retrieval-good.json runs/retrieval-noisy.json --suite examples/retrieval/suite.yaml --gate
```

The gate fails on a real drop in nDCG and passes when the change is only noise.
