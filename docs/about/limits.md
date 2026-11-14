# Limits

* The measurements cover one machine, two small local models, one product domain and
  83 cases. They show the method; they are not a benchmark of those models.
* Calibration is only as good as its gold labels — [MEASUREMENTS §4](measurements.md)
  shows a regex answer key being wrong more often than the judge. Use human labels when
  the decision matters.
* ACC needs both classes in the gold sample and a judge better than chance; PPI needs the
  gold sample to be random. evalproof reports when either does not hold rather than
  printing a number.
* The percentile-bootstrap p-value for continuous scores is an approximation; pass/fail
  metrics use the exact McNemar test. With several trials per case the test is a paired
  permutation test, exact for a whole suite of equally repeated cases and a seeded Monte Carlo
  estimate when it is not (see [repeated trials](../guides/repeated-trials.md#repeated-trials)).
* Repeated trials show variation only when the system varies. A model at a fixed seed, or a
  cached call, can give the same answer every time, and then five trials are one trial five
  times; the report says when every case did.
* Cost tracking needs a `pricing` table in the suite for hosted models; only `claude-code`
  reports a cost of its own (an API-equivalent estimate under a subscription).
* CLI targets (`claude-code`, `codex` and the experimental three) take no temperature or seed,
  so their runs do not repeat exactly; the cache makes a re-run identical, `--no-cache` does not.
* `verify` re-scores deterministic scorers only. It cannot check a model-judged score or that
  the stored outputs are what the system really said, so treat a run file from someone you do
  not trust as unverified, or run the suite in CI with a key.
