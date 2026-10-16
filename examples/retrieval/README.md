# Retrieval example

Offline example that exercises the retrieval scorers (`recall_at_k`, `mrr`, `ndcg`) end to
end on a small labelled corpus and two deterministic retrievers. No models or network.

## Files

- `cases.jsonl`: 20 queries with graded relevance judgements (0 to 3).
- `retrievers.py`: two systems.
  - `good_retriever` ranks the corpus by term overlap with the query.
  - `noisy_retriever` is the same ranking with a deliberate regression: on queries with an
    even number of characters, the top hit drops to rank 5.
- `suite.yaml`: recall@5, MRR@10 and nDCG@5, gating on nDCG@5.

## Run

```bash
for v in good noisy; do
  uv run evalproof run examples/retrieval/suite.yaml \
    --system "python:retrievers:${v}_retriever" --out "runs/retrieval-$v.json" --quiet
done
uv run evalproof compare runs/retrieval-good.json runs/retrieval-noisy.json \
  --suite examples/retrieval/suite.yaml --gate; echo "exit $?"   # 1: real regression
```

## What it shows

Moving the top hit to rank 5 leaves recall@5 unchanged at 0.775, because recall ignores
order inside the cut-off. MRR falls from 1.000 to 0.642 and the nDCG@5 pass rate from
100% to 55%: nine of twenty queries regress, none improve, and the exact McNemar test gives
p = 0.0039, so the gate fails the change with exit 1.
