# Commands

The `evalproof` commands and what each one does.

| Command | What it does |
|---|---|
| `evalproof init` | write a starter suite, a CI workflow and an agent skill; no model call |
| `evalproof run SUITE` | run the suite and write a run file; `--gate` applies the baseline-free checks (`min_pass_rate`, `must_pass`); `--show N` lists N failed cases; `--variant NAME` (repeatable) or `--all-variants` runs the suite's [variants](../guides/testing-agents.md#variants-one-suite-several-builds), one run file each; `--trials K` runs each case K times ([repeated trials](../guides/repeated-trials.md#repeated-trials)) |
| `evalproof compare BASE CAND [CAND ...]` | compare runs on paired cases; `--gate` turns the result into an exit code; `--show N` prints both answers for N regressed cases; `--aggregate mean\|all\|any` picks how a case's [trials](../guides/repeated-trials.md#repeated-trials) become one value; several candidates give [one table](../guides/repeated-trials.md#several-candidates) |
| `evalproof verify RUN SUITE` | check, with no model call, that a run file was made with this suite, covers exactly the dataset's cases as they are now, and has the scores its deterministic scorers give the stored outputs |
| `evalproof rehash RUN SUITE` | add per-case hashes to a run file written before 0.5, from the dataset it scored; no model call |
| `evalproof report RUN` | print a run file's report |
| `evalproof import FORMAT FILE` | turn another tool's results into a run file |
| `evalproof calibrate RUN` | measure the judge against gold labels and correct its pass rate |
| `evalproof pairwise A B` | judge two runs head to head, in both orders |
| `evalproof doctor [SUITE]` | check the agent CLIs a suite uses; no model call |
| `evalproof readiness SUITE [--run FILE] [--json] [--require DIM,...]` | report which parts of quality the suite covers (correctness, tool use, quality, gate, cost, safety, and so on) and which it leaves untested; exits 1 when a `--require`d one is not covered; no model call |
