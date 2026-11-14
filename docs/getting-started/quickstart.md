# Five-minute quickstart

In your own repository, one command writes a starter suite, a CI workflow and an agent skill,
and a second runs it:

```bash
uvx evalproof init
uvx evalproof run evals/suite.yaml --out evals/runs/latest.json
```

`init` picks the judge from what is on your `PATH`: a logged-in `claude`, then `codex`, then
Ollama, then a hosted Anthropic model with `ANTHROPIC_API_KEY`. It writes `evals/` (the
suite, three sample cases, a README), `.github/workflows/evalproof.yml` (`--ci keyless`, the
default, makes no model calls in CI; `--ci api-key` runs the suite there; `--ci none` writes
no workflow) and a skill for coding agents at `.agents/skills/evalproof/SKILL.md` and
`.claude/skills/evalproof/SKILL.md` (`--no-skill` skips it). It prints one line per file,
never overwrites one without `--force`, and makes no model call. Replace the sample cases
with your own: the gate needs 20 before it will decide.

The offline example lives in the repository:

```bash
git clone https://github.com/technobasant/evalproof && cd evalproof
uv sync

# Offline: no model, no network. Four versions of a ticket router.
for v in baseline regressed noisy refactored; do
  uv run evalproof run examples/deterministic/suite.yaml \
    --system "python:router:${v}_route" --out "runs/$v.json" --quiet
done
uv run evalproof compare runs/baseline.json runs/regressed.json \
  --suite examples/deterministic/suite.yaml --gate; echo "exit $?"   # 1: real regression
uv run evalproof compare runs/baseline.json runs/noisy.json \
  --suite examples/deterministic/suite.yaml --gate; echo "exit $?"   # 0: 2 of 60 is noise
```

[`examples/retrieval`](https://github.com/technobasant/evalproof/blob/main/examples/retrieval) runs the same gate on a retrieval change: recall@5
does not move when the top hit drops to rank 5, while MRR and nDCG@5 do, and the gate fails
it. It was contributed by [@HarshRajSinghania](https://github.com/HarshRajSinghania), and CI
checks its exit codes too.

With Ollama installed, the LLM example is
[`examples/support-qa`](https://github.com/technobasant/evalproof/blob/main/examples/support-qa) (`ollama pull qwen3:1.7b` and
`ollama pull qwen3:8b`; about 6–10 minutes per run on an Apple M5).

The browser app in [`app/`](https://github.com/technobasant/evalproof/blob/main/app) runs locally too:
`uv run --with-requirements app/requirements.txt streamlit run app/streamlit_app.py`.
