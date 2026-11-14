---
description: Write a starter suite, run it, and watch a gate fail on a real regression, in five minutes.
---

# Five-minute quickstart

This page takes you from nothing to a suite, a run and a gate that fails on a real regression. It is for anyone trying evalproof for the first time.

## 1. Write a starter suite

In your own repository, one command writes a starter suite, a CI workflow and an agent skill:

```bash title="Terminal"
uvx evalproof init
```

Expected output, on a machine with `claude` on the `PATH`:

```text title="Output"
evalproof init: judge claude-code:haiku (claude is on PATH, so evals use your Claude Code login and need no API key)
  wrote          evals/suite.yaml
  wrote          evals/cases.jsonl
  wrote          evals/README.md
  wrote          evals/.gitignore
  wrote          .github/workflows/evalproof.yml
  wrote          .agents/skills/evalproof/SKILL.md
  wrote          .claude/skills/evalproof/SKILL.md

next:
  uvx evalproof run evals/suite.yaml --out evals/runs/latest.json
  commit evals/ (including evals/runs/latest.json) with your change
```

`init` picks the judge from what is on your `PATH`: a logged-in `claude`, then `codex`, then
Ollama, then a hosted Anthropic model with `ANTHROPIC_API_KEY`. It writes `evals/` (the
suite, three sample cases, a README), `.github/workflows/evalproof.yml` (`--ci keyless`, the
default, makes no model calls in CI; `--ci api-key` runs the suite there; `--ci none` writes
no workflow) and a skill for coding agents at `.agents/skills/evalproof/SKILL.md` and
`.claude/skills/evalproof/SKILL.md` (`--no-skill` skips it). It prints one line per file,
never overwrites one without `--force`, and makes no model call.

## 2. Run it

```bash title="Terminal"
uvx evalproof run evals/suite.yaml --out evals/runs/latest.json
```

The run file holds every case's output and scores, and the report is printed. Replace the sample cases
with your own: the gate needs 20 before it will decide.

## 3. Try a gate on an example

=== "Offline"

    The offline example needs no model and no network: four versions of a ticket router.

    ```bash title="Terminal"
    git clone https://github.com/technobasant/evalproof && cd evalproof
    uv sync

    for v in baseline regressed noisy refactored; do
      uv run evalproof run examples/deterministic/suite.yaml \
        --system "python:router:${v}_route" --out "runs/$v.json" --quiet
    done
    uv run evalproof compare runs/baseline.json runs/regressed.json \
      --suite examples/deterministic/suite.yaml --gate; echo "exit $?"   # 1: real regression
    uv run evalproof compare runs/baseline.json runs/noisy.json \
      --suite examples/deterministic/suite.yaml --gate; echo "exit $?"   # 0: 2 of 60 is noise
    ```

    The first comparison fails the gate:

    ```text title="Output (first lines)"
    ## evalproof

    **Gate: ❌ fail** (exit 1)

    - queue: 0.883 → 0.700 (Δ -0.183, 95% CI [-0.283, -0.083], exact McNemar p=0.0010) — a drop beyond 0.020 at α=0.05
    ```

    The second passes with a warning, because the drop is not distinguishable from noise:

    ```text title="Output (first lines)"
    ## evalproof

    **Gate: ⚠️ pass with warnings** (exit 0)

    - ⚠️ queue dropped -0.033, beyond the 0.020 threshold, but p=0.500 ≥ α=0.05: not distinguishable from noise at n=60. Add cases before trusting either direction.
    ```

=== "Retrieval"

    [`examples/retrieval`](https://github.com/technobasant/evalproof/blob/main/examples/retrieval) runs the same gate on a retrieval change: recall@5
    does not move when the top hit drops to rank 5, while MRR and nDCG@5 do, and the gate fails
    it. It was contributed by [@HarshRajSinghania](https://github.com/HarshRajSinghania), and CI
    checks its exit codes too.

=== "LLM (Ollama)"

    With Ollama installed, the LLM example is
    [`examples/support-qa`](https://github.com/technobasant/evalproof/blob/main/examples/support-qa) (`ollama pull qwen3:1.7b` and
    `ollama pull qwen3:8b`; about 6–10 minutes per run on an Apple M5).

!!! tip "The browser app"

    The browser app in [`app/`](https://github.com/technobasant/evalproof/blob/main/app) runs locally too:
    `uv run --with-requirements app/requirements.txt streamlit run app/streamlit_app.py`.

## Next steps

* See what else a suite and one or two commands can do: [Use cases](use-cases.md).
* Check what a run needs, locally and in CI: [What it needs to run](credentials.md).
* Gate in GitHub Actions: [In CI](../guides/ci.md#in-ci).
