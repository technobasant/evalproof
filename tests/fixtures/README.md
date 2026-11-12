# Fixtures for the importers

These files are data, never code. Tests read them and never execute them.

- `promptfoo/sample-export.json`: verbatim from promptfoo, `test/__fixtures__/sample-export.json` (MIT licence). Two providers, two tests, every result a failure.
- `inspect/log_formats.json`: verbatim from Inspect AI, `tests/log/test_eval_log/log_formats.json` (MIT licence). One model, one sample. The `.eval` zip form is built from it inside the tests.
- `deepeval/test_run.json`: synthetic, written to the schema in DeepEval's `test_run/test_run.py` and `api.py`; it has a metric error, a positional `test_case_N` name and a conversational case.
- `lm-eval/<model>/samples_<task>_<time>.jsonl`: synthetic, written to the record shape built in lm-evaluation-harness `evaluator.py` and `loggers/evaluation_tracker.py`; `gpt2` has one filter, `llama3` has two.
- `table/long.csv`, `table/long.jsonl`, `table/wide.csv`: synthetic, for the generic importers (two systems, an error row, an undecided cell).
