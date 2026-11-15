---
description: The JUnit XML that evalproof run --junit and evalproof compare --gate --junit write, how a case, a trial and a gate check map to a testcase, and how to show the report in GitLab, Jenkins and Azure DevOps.
---

# JUnit test reports

`--junit PATH` writes the result as JUnit XML, so GitLab, Jenkins, Azure DevOps and other CI systems show cases and gate checks as tests, with their own failure lists and history. This page is for anyone wiring evalproof into one of them. The report is an extra output: it changes no exit code ([Exit codes](exit-codes.md)).

Two commands write it.

| Command | The report holds |
|---|---|
| `evalproof run --junit PATH` | one testsuite per run written, one testcase per case result |
| `evalproof compare --gate --junit PATH` | one testsuite, `evalproof gate`, with one testcase per gate check |

The file is UTF-8 with an XML declaration and LF line endings. Characters that XML 1.0 cannot hold, such as control characters in a model's output, are removed from every value, so a parser never rejects the file. A file already at the path is replaced, and the folders above it are created.

## A run

`evalproof run --junit junit.xml` writes the report after each run. The root is `<testsuites>` with the totals of its suites.

* A `<testsuite>` is one run. Its `name` is `<suite>.<variant>`, or `<suite>.<system>` for a run without a variant. With `--all-variants` or several `--variant` flags the one file holds a testsuite per variant.
* A `<testcase>` is one case result. Its `classname` is the testsuite's name, and its `time` is the system's latency in seconds.
* With trials above 1 there is a testcase per trial, named `<case id> [trial N]`, with N counted from 0. With one trial the name is the case id.
* The case's output goes in `<system-out>`, cut at 2000 characters.
* The testsuite's `<properties>` hold the evalproof version, the dataset and suite hashes, the system, the variant, the trials, the seed, the start and finish times, the wall time, the cost and the run's environment facts.

A testcase is marked as the run report's "Failed cases" list marks it:

| Case | Testcase holds | Message |
|---|---|---|
| every scorer decided and passed | nothing: it passed | none |
| a scorer failed | `<failure type="scorer failed">` | `failed <scorer> (<score>)` for each failing scorer |
| the system raised an error | `<error type="system error">` | the error text |
| a scorer could not decide, and none failed | `<skipped>` | `undecided on <scorer>: no scorer reached a verdict` |

The body of a failure holds each failing scorer and its reason. A case with a failing scorer and an undecided one is a failure. An errored case has no `<system-out>`.

The report is written when every case errored too, so the errors show in CI. The command still exits 2 then, as it does without `--junit`.

## A gate

`evalproof compare baseline.json candidate.json --gate --junit gate.xml` writes one `<testsuite name="evalproof gate">`. It holds a testcase per check the gate evaluated, with `classname` `evalproof gate` and a `time` of 0. The testcases come from the checks of the gate's decision, so a check the gate adds later appears here with no change to the report.

| Testcase name | The check |
|---|---|
| `max_drop: <metric>` | the significance test on the metric's drop, one per gated metric |
| `max_new_failures: <metric>` | cases that went from pass to fail, when `max_new_failures` is set |
| `min_pass_rate: <metric>` | the candidate's pass rate, when `min_pass_rate` is set |
| `must_pass` | the cases tagged in `must_pass`, when it is set |
| `max_p95_latency_ms` | the candidate's p95 latency, when `max_p95_latency_ms` is set |
| `max_cost_usd` | the candidate's total cost, when `max_cost_usd` is set |

A check that does not apply is left out: a metric with only continuous scores has no `max_new_failures` testcase.

* A passing check is a testcase with its measurement in `<system-out>`.
* A failed check is a `<failure type="gate check failed">` whose `message` and body are the reason the terminal report prints.
* A check the gate could not decide is an `<error type="cannot gate">`. These are the causes of exit 4: too few paired cases (`min_cases: <metric>`), a judge that changed (`same_judge`), runs that cannot be compared (`comparable`), and the others [Exit codes](exit-codes.md#exit-4-cannot-gate) lists. When the runs cannot be compared at all the file holds one error testcase, `comparison`.
* The gate's warnings are lines in the testsuite's `<system-out>`.
* The testsuite's `<properties>` hold the exit code, the baseline and candidate, and the gate settings.

With several candidates each testcase name starts with the candidate's label, as in `steady / max_drop: m`, and the properties are `candidate.1`, `candidate.2` and so on. The file is one testsuite whatever the number of candidates.

### When the file is written

The file is written for every verdict, 0, 1, 3 and 4, so a failed gate always has its report. A usage error (exit 2) writes none. Two checks come before any work:

* `--junit` without `--gate` exits 2 with `--junit needs --gate`. A comparison with no gate has no checks to report.
* A path that cannot be written, such as a folder, exits 2 before the command runs.

### Example

A candidate that dropped on `m` and is under the pass rate, as the XML shows it, with the properties trimmed:

```xml title="gate.xml"
<?xml version="1.0" encoding="UTF-8"?>
<testsuites name="evalproof" tests="2" failures="2" errors="0" skipped="0" time="0.000">
  <testsuite name="evalproof gate" tests="2" failures="2" errors="0" skipped="0" time="0.000">
    <properties>
      <property name="exit_code" value="1" />
      <property name="baseline" value="main" />
      <property name="candidate" value="pr" />
    </properties>
    <testcase classname="evalproof gate" name="max_drop: m" time="0.000">
      <failure message="m: 1.000 → 0.667 (Δ -0.333, 95% CI [-0.500, -0.167], exact McNemar p=0.0020) — a drop beyond 0.020 at α=0.05" type="gate check failed">m: 1.000 → 0.667 (Δ -0.333, 95% CI [-0.500, -0.167], exact McNemar p=0.0020) — a drop beyond 0.020 at α=0.05</failure>
    </testcase>
    <testcase classname="evalproof gate" name="min_pass_rate: m" time="0.000">
      <failure message="m: pass rate 0.667 below the minimum 0.900 (20/30 cases)" type="gate check failed">m: pass rate 0.667 below the minimum 0.900 (20/30 cases)</failure>
    </testcase>
  </testsuite>
</testsuites>
```

## Show the report in CI

Write the file in the same job that gates, and upload it even when the gate fails. The gate's exit code still fails the job.

### GitLab

```yaml title=".gitlab-ci.yml"
evals:
  script:
    - evalproof compare runs/baseline.json runs/candidate.json --suite evals/suite.yaml --gate --junit gate.xml
  artifacts:
    when: always
    reports:
      junit: gate.xml
```

GitLab lists each failed check under the pipeline's Tests tab. `when: always` keeps the report from a job that exited non-zero.

### Jenkins

```groovy title="Jenkinsfile"
stage('Evals') {
  steps {
    sh 'evalproof compare runs/baseline.json runs/candidate.json --suite evals/suite.yaml --gate --junit gate.xml'
  }
  post {
    always {
      junit 'gate.xml'
    }
  }
}
```

The `junit` step sits in `post { always }`, so it reads the file after a failed gate. The `sh` step's non-zero exit code still fails the build.

### Azure DevOps

Use the `PublishTestResults@2` task with `testResultsFormat: JUnit`, `testResultsFiles: gate.xml` and `condition: always()`.

## Next steps

* Every flag of the two commands: [Commands](commands.md).
* What each exit code means: [Exit codes](exit-codes.md).
* A GitHub Actions workflow for the gate: [Continuous integration](../guides/ci.md).
* The gate's settings: [Configuration](configuration.md).
