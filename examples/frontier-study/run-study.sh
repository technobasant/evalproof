#!/usr/bin/env bash
# Runs every study call on the sampled cases: per model, support-qa baseline x2 trials (84),
# candidate x1 (42), ticket-router x2 (60): 186 calls per model, 558 in all. The three models run
# in parallel; each model's runs are sequential. The last step rebuilds the baseline's first trial
# as a one-trial run (support-qa-<model>-baseline-t0.json) from the cache, so the one-trial
# candidate can be gated against it; it makes no model call.
# Re-running answers finished calls from the cache, so an interrupted study resumes.
# Usage (from the repository root): examples/frontier-study/run-study.sh [anthropic|openai|google ...]
set -u
cd "$(dirname "$0")/../.." || exit 1
dir=examples/frontier-study
mkdir -p "$dir/runs" "$dir/logs"

study() {
  local m=$1
  {
    uv run evalproof run "$dir/support-qa.yaml" --variant "$m-baseline" --trials 2 \
      --out "$dir/runs/support-qa-$m-baseline.json" --show 0 &&
      uv run evalproof run "$dir/support-qa.yaml" --variant "$m-candidate" --trials 1 \
        --out "$dir/runs/support-qa-$m-candidate.json" --show 0 &&
      uv run evalproof run "$dir/ticket-router.yaml" --variant "$m" --trials 2 \
        --out "$dir/runs/ticket-router-$m.json" --show 0 &&
      uv run evalproof run "$dir/support-qa.yaml" --variant "$m-baseline" --trials 1 \
        --out "$dir/runs/support-qa-$m-baseline-t0.json" --show 0
  } >"$dir/logs/$m.log" 2>&1
  echo "$m exit=$?"
}

models=("$@")
[ ${#models[@]} -eq 0 ] && models=(anthropic openai google)
for m in "${models[@]}"; do study "$m" & done
wait
