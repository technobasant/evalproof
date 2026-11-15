"""Recompute every table of the three-model study from the committed run files. No model needed.

    uv run python examples/frontier-study/measure.py

It reads the 12 files in ``runs/``, checks each against the suite and dataset as they are now
(the dataset hash, and every stored verdict re-scored from the answer key), and prints Markdown to
stdout. Pass-rate intervals, the paired comparison and the gate verdict come from evalproof itself,
so they match what ``evalproof report`` and ``evalproof compare`` print for the same files.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from evalproof.compare import compare_runs
from evalproof.config import GateConfig, SuiteConfig, load_config
from evalproof.dataset import dataset_hash, load_dataset
from evalproof.gate import EXIT_CANNOT_GATE, decide
from evalproof.models import Case, CaseResult, Run
from evalproof.runner import load_run
from evalproof.scorers import is_deterministic, make_scorer
from evalproof.stats import binom_cdf, mcnemar_exact, percentile, wilson
from evalproof.trials import case_counts, pooled_rate_ci, trial_outcome

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODELS = ("anthropic", "openai", "google")
KEYS = ("answer_key", "answer_key_reviewed")
SNIPPET_CHARS = 120
ONE_SIDED_ALPHA = 0.05  # the flip-rate upper bound is a one-sided 95% bound
CEILING_MARK = "†"
CEILING_NOTE = (
    f"{CEILING_MARK} at 100% the bootstrap interval collapses; Wilson interval over cases shown"
)
VERDICT = {"pass": "pass", "warn": "pass with warnings", "fail": "fail"}

# Words that must never reach the output: the tools the models were reached through, and
# anything that names them, and the suites' own names. "cli" is matched as a whole word only, so
# "client" and "click" in a model's answer pass. The providers' names (anthropic, openai, google)
# are allowed: they are the output's column of providers, so only the hyphenated variant names
# ("anthropic-baseline") are banned, not the router's bare variants.
BANNED = (
    "claude-code",
    "claude code",
    "codex",
    "cursor",
    "support-qa-frontier",
    "ticket-router-frontier",
)
BANNED_WORD = re.compile(r"(?<![a-z])cli(?![a-z])")


@dataclass(frozen=True)
class ModelRuns:
    """The four run files of one model."""

    key: str
    baseline: Run  # grounded prompt, 2 trials
    t0: Run  # the same, trial 0 only
    candidate: Run  # friendly prompt, 1 trial
    router: Run  # ticket router, 2 trials

    @property
    def all(self) -> tuple[Run, ...]:
        return (self.baseline, self.t0, self.candidate, self.router)

    @property
    def model_id(self) -> str:
        """The target without its ``prefix:``."""
        target = self.baseline.info.environment["system_target"]
        return target.split(":", 1)[-1]

    @property
    def provider(self) -> str:
        """The first word of the run label."""
        return self.baseline.info.system.split()[0]


@dataclass(frozen=True)
class Study:
    """One suite as it stands now: its config and the cases it loads, the way the runner loads
    them, with the hash that every run of the suite stored."""

    config: SuiteConfig
    cases: dict[str, Case]
    data_hash: str


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


def load_study(name: str) -> Study:
    config = load_config(HERE / name)
    cases = load_dataset(config.resolve(config.dataset))
    return Study(config, {c.id: c for c in cases}, dataset_hash(cases))


def rescore(run: Run, study: Study) -> int:
    """Check ``run`` against ``study`` and return how many stored verdicts were re-scored.

    The run's dataset hash must equal the hash of the dataset file now. Every stored output is
    then scored again by the suite's own scorers, built the way the runner builds them, and each
    pass or fail must equal the stored one."""
    label = run.info.variant or run.info.suite
    require(run.info.suite == study.config.name, f"{label}: the run is of another suite")
    require(
        run.info.dataset_hash == study.data_hash,
        f"{label}: the dataset changed since the run "
        f"(run {run.info.dataset_hash}, now {study.data_hash})",
    )
    require(
        len(run.results) == len(study.cases) * run.trials
        and {r.case_id for r in run.results} == set(study.cases),
        f"{label}: the run does not hold every case once per trial",
    )
    require(
        all(is_deterministic(sc) for sc in study.config.scorers),
        f"{label}: a scorer needs a model, so its verdicts cannot be re-scored here",
    )
    scorers = [make_scorer(sc, study.config) for sc in study.config.scorers]
    checked = 0
    for r in run.results:
        for scorer in scorers:
            stored, fresh = r.scores[scorer.name], scorer.score(study.cases[r.case_id], r.output)
            require(
                stored.passed is not None and fresh.passed == stored.passed,
                f"{label}: {r.case_id} trial {r.trial} on {scorer.name}: stored "
                f"{stored.passed}, re-scored {fresh.passed}",
            )
            checked += 1
    return checked


def load_model(key: str) -> ModelRuns:
    m = ModelRuns(
        key,
        load_run(RUNS / f"support-qa-{key}-baseline.json"),
        load_run(RUNS / f"support-qa-{key}-baseline-t0.json"),
        load_run(RUNS / f"support-qa-{key}-candidate.json"),
        load_run(RUNS / f"ticket-router-{key}.json"),
    )
    for run in m.all:
        env = run.info.environment
        require(
            env.get("system_model_resolved") == m.model_id,
            f"{key}: the resolved model differs from the target",
        )
        require(
            all(r.output.error is None for r in run.results), f"{key}: a run holds an errored row"
        )
    # The trial-0 file is the baseline's first trial, row for row.
    first = {r.case_id: r for r in m.baseline.results if r.trial == 0}
    require(
        len(first) == len(m.t0.results)
        and all(
            r.case_id in first
            and r.output.output == first[r.case_id].output.output
            and all(s.passed == first[r.case_id].scores[n].passed for n, s in r.scores.items())
            for r in m.t0.results
        ),
        f"{key}: the trial-0 file does not match the baseline's first trial",
    )
    return m


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def cell(text: str) -> str:
    """``text`` as one code span on one line, safe inside a table cell."""
    one = " ".join(text.split()).replace("|", "\\|")
    longest = max((len(t) for t in re.findall("`+", one)), default=0)
    ticks = "`" * (longest + 1)
    pad = " " if one.startswith("`") or one.endswith("`") else ""
    return f"{ticks}{pad}{one}{pad}{ticks}"


def snippet(text: str) -> str:
    one = " ".join(text.split())
    return one if len(one) <= SNIPPET_CHARS else one[:SNIPPET_CHARS] + "…"


def ids(case_ids: list[str]) -> str:
    return ", ".join(f"`{c}`" for c in case_ids) if case_ids else "none"


def rate_cell(run: Run, scorer: str) -> str:
    """``passes/n (rate, [lo, hi])`` as ``evalproof report`` computes it: a run that repeats each
    case pools its trials and resamples cases for the interval; a single-trial run uses the Wilson
    interval. A pooled interval that collapses to a point is replaced by the Wilson interval over
    cases and marked."""
    mark = ""
    if run.trials > 1:
        counts = list(case_counts(run, scorer).values())
        passes = sum(p for p, _ in counts)
        n = sum(d for _, d in counts)
        lo, hi = pooled_rate_ci(counts)
        if lo == hi:
            # Every case passed every trial, so resampling cases gives the same rate each time.
            # Report the Wilson interval over cases (all trials passed) instead.
            require(
                passes == n, f"the interval collapsed below 100% ({passes}/{n}): extend the note"
            )
            lo, hi = wilson(sum(1 for p, d in counts if p == d), len(counts))
            mark = CEILING_MARK
    else:
        scores = [r.scores[scorer] for r in run.results if scorer in r.scores]
        binary = [s.passed for s in scores if s.value is not None and s.passed is not None]
        passes, n = sum(binary), len(binary)
        lo, hi = wilson(passes, n)
    return f"{passes}/{n} ({pct(passes / n)}, [{pct(lo)}, {pct(hi)}]){mark}"


def trial_rows(run: Run, trial: int) -> dict[str, CaseResult]:
    return {r.case_id: r for r in run.results if r.trial == trial}


def setup(models: list[ModelRuns], support: Study, verdicts: tuple[int, int]) -> list[str]:
    suite = support.config
    out = [
        "## Setup",
        "",
        "| Provider | Model | Settings | Run date | evalproof version |",
        "|---|---|---|---|---|",
    ]
    for m in models:
        extra = suite.variant_system(m.baseline.info.variant).params.extra
        effort = extra.get("reasoning_effort")
        settings = f"reasoning effort: {effort}" if effort else "provider default"
        dates = ", ".join(sorted({r.info.started_at[:10] for r in m.all}))
        versions = ", ".join(sorted({r.info.evalproof_version for r in m.all}))
        out.append(f"| {m.provider} | `{m.model_id}` | {settings} | {dates} | {versions} |")
    n_support, n_router = verdicts
    out += [
        "",
        "Sampling is the provider default; no temperature or seed control.",
        "",
        f"All {n_support + n_router} stored verdicts were re-scored from the current answer key "
        f"and matched ({n_support} support Q&A verdicts: two keys over every answer of the "
        f"grounded, trial-0 and friendly runs; {n_router} router verdicts). Every run's dataset "
        "hash equals the hash of the dataset file as the runner loads it.",
        "",
    ]
    return out


def pass_rates(models: list[ModelRuns]) -> list[str]:
    out = [
        "## Pass rates",
        "",
        "Pass rate over 84 grounded-prompt answers (42 cases, 2 trials), 42 friendly-prompt "
        "answers and 60 router answers (30 cases, 2 trials). Two-trial cells pool both trials and "
        "resample cases for the 95% interval; one-trial cells use the Wilson 95% interval.",
        "",
        "| Model | Grounded prompt, raw key | Grounded prompt, reviewed key "
        "| Friendly prompt, raw key | Friendly prompt, reviewed key | Ticket router |",
        "|---|---|---|---|---|---|",
    ]
    flagged = False
    for m in models:
        cells = [
            rate_cell(m.baseline, "answer_key"),
            rate_cell(m.baseline, "answer_key_reviewed"),
            rate_cell(m.candidate, "answer_key"),
            rate_cell(m.candidate, "answer_key_reviewed"),
            rate_cell(m.router, "queue"),
        ]
        out.append(f"| `{m.model_id}` | " + " | ".join(cells) + " |")
        flagged = flagged or any(CEILING_MARK in c for c in cells)
    return [*out, "", *([CEILING_NOTE, ""] if flagged else [])]


def clopper_pearson_upper(flips: int, n: int, alpha: float = ONE_SIDED_ALPHA) -> float:
    """The exact one-sided upper ``1 - alpha`` bound on a rate seen ``flips`` times in ``n``: the
    ``p`` at which ``P(X <= flips)`` falls to ``alpha`` for ``X ~ Binomial(n, p)``. At zero flips
    that is ``1 - alpha ** (1 / n)``."""
    if flips >= n:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if binom_cdf(flips, n, mid) > alpha else (lo, mid)
    return (lo + hi) / 2


def reworded(run: Run) -> tuple[int, int]:
    """``(cases, total)``: the cases whose answer text differs between trial 0 and trial 1."""
    t0, t1 = trial_rows(run, 0), trial_rows(run, 1)
    require(set(t0) == set(t1), "trials cover different cases")
    return sum(1 for c in t0 if t0[c].output.output != t1[c].output.output), len(t0)


def run_to_run(models: list[ModelRuns]) -> list[str]:
    out = [
        "## Run-to-run (A/A)",
        "",
        "The same prompt on the same cases twice: trial 0 against trial 1. The router's labels are "
        "exact and were written before any of these runs, so a router verdict that differs between "
        "the trials is sampling noise and nothing else. The reviewed support Q&A key was written "
        "after reading these outputs, so its verdicts agree between trials by construction; they "
        "are not run-to-run evidence and are not tabulated here. Answer text is: how many support "
        "Q&A answers were worded differently in the two trials.",
        "",
        "| Model | Router, trial 0 | Router, trial 1 | Router verdicts that differ | Case ids "
        "| Support Q&A answers reworded |",
        "|---|---|---|---|---|---|",
    ]
    flips = cases = 0
    for m in models:
        t0, t1 = trial_rows(m.router, 0), trial_rows(m.router, 1)
        require(set(t0) == set(t1), f"{m.key}: trials cover different cases")
        rates = []
        for rows in (t0, t1):
            outcomes = [trial_outcome(r, "queue") for r in rows.values()]
            require(None not in outcomes, f"{m.key}: a router verdict is undecided")
            passed = sum(1 for o in outcomes if o)
            rates.append(f"{passed}/{len(outcomes)} ({pct(passed / len(outcomes))})")
        differ = [c for c in t0 if trial_outcome(t0[c], "queue") != trial_outcome(t1[c], "queue")]
        flips, cases = flips + len(differ), cases + len(t0)
        words, total = reworded(m.baseline)
        out.append(
            f"| `{m.model_id}` | {rates[0]} | {rates[1]} | {len(differ)} of {len(t0)} "
            f"| {ids(differ)} | {words} of {total} |"
        )
    bound = clopper_pearson_upper(flips, cases)
    formula = f", 1 − {ONE_SIDED_ALPHA}^(1/{cases})" if flips == 0 else ""
    return [
        *out,
        "",
        f"Pooled over the three models, {flips} of {cases} router verdicts differed between the "
        "two trials. The exact one-sided 95% upper bound on the per-answer flip rate "
        f"(Clopper–Pearson{formula}) is {pct(bound)}.",
        "",
    ]


def identical_text(run: Run) -> str:
    """``n of N``: the cases whose output text is byte-identical in trial 0 and trial 1."""
    changed, total = reworded(run)
    return f"{total - changed} of {total}"


def median_words(run: Run, trial: int = 0) -> str:
    """The median answer length in words (whitespace split) over one trial's answers."""
    words = [len(r.output.output.split()) for r in trial_rows(run, trial).values()]
    return f"{statistics.median(words):g}"


def answer_text(models: list[ModelRuns]) -> list[str]:
    out = [
        "## Answer text",
        "",
        "Cases whose answer text is byte-identical in trial 0 and trial 1, and the median answer "
        "length in words (whitespace split) over the 42 answers of one trial.",
        "",
        "| Model | Identical answers, support Q&A | Identical answers, ticket router "
        "| Median words, grounded prompt (trial 0) | Median words, friendly prompt |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in models:
        out.append(
            f"| `{m.model_id}` | {identical_text(m.baseline)} | {identical_text(m.router)} | "
            f"{median_words(m.baseline)} | {median_words(m.candidate)} |"
        )
    return [*out, ""]


def exact_mde(n: int, alpha: float) -> tuple[int, float]:
    """``(k, p)``: the smallest number ``k`` of discordant pairs, all in one direction, whose exact
    two-sided McNemar p-value ``2 * 0.5**k`` (capped at 1) is below ``alpha``, among ``n`` paired
    cases. ``k / n`` is the smallest drop the test can call significant."""
    for k in range(1, n + 1):
        p = mcnemar_exact(k, 0)
        if p < alpha:
            return k, p
    raise AssertionError(f"{n} paired cases cannot reach p < {alpha}")


def prompt_change(models: list[ModelRuns], support: Study) -> list[str]:
    gate = support.config.gate
    if gate is None:
        raise AssertionError("the support Q&A suite has no gate")
    out = [
        "## Prompt change",
        "",
        "Grounded prompt (trial 0) against friendly prompt, paired on the same cases, gated as "
        f"the suite gates it: a drop of {100 * gate.max_drop:.0f} points or more at "
        f"alpha {gate.alpha}, at least {gate.min_cases} cases. The comparison is the exact "
        "McNemar test; the interval is a paired bootstrap.",
        "",
        "| Model | Metric | Grounded prompt | Friendly prompt | Δ (95% CI) | McNemar p "
        "| Gate verdict |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    notes = []
    paired = set()
    for m in models:
        for name in KEYS:
            one = GateConfig(
                metric=name,
                max_drop=gate.max_drop,
                alpha=gate.alpha,
                min_cases=gate.min_cases,
                n_boot=gate.n_boot,
            )
            cmp = compare_runs(m.t0, m.candidate, alpha=one.alpha, seed=0, metrics=[name])
            decision = decide(cmp, one, m.candidate)
            mc = cmp.metrics[name]
            verdict = (
                "cannot gate"
                if decision.exit_code == EXIT_CANNOT_GATE
                else VERDICT[decision.status]
            )
            notes += [f"- `{m.model_id}`, `{name}`: {w}" for w in decision.warnings]
            paired.add(mc.n_paired)
            out.append(
                f"| `{m.model_id}` | `{name}` | {pct(mc.baseline)} | {pct(mc.candidate)} | "
                f"{100 * mc.delta:+.1f} pts [{100 * mc.ci_low:+.1f}, {100 * mc.ci_high:+.1f}] | "
                f"{mc.p_value:.4f} | {verdict} |"
            )
    require(len(paired) == 1, f"the comparisons pair different numbers of cases: {sorted(paired)}")
    n = paired.pop()
    k, p = exact_mde(n, gate.alpha)
    out += [
        "",
        f"Smallest detectable drop: with {n} paired cases the exact two-sided McNemar test first "
        f"gives p < {gate.alpha} at {k} discordant pairs that all go the same way "
        f"(p = 2 × 0.5^{k} = {p:.4f}, against {mcnemar_exact(k - 1, 0):.4f} at {k - 1}), "
        f"so a drop of {k}/{n} = {100 * k / n:.1f} points. A smaller drop cannot reach "
        f"significance at this sample size.",
        "",
    ]
    return [*out, *notes, *([""] if notes else [])]


def key_review(models: list[ModelRuns]) -> list[str]:
    rows = []
    for m in models:
        for prompt, run in (("grounded", m.baseline), ("friendly", m.candidate)):
            for r in run.results:
                raw, reviewed = r.scores["answer_key"], r.scores["answer_key_reviewed"]
                if raw.passed != reviewed.passed:
                    verdict = f"{'pass' if raw.passed else 'fail'} → "
                    verdict += "pass" if reviewed.passed else "fail"
                    reason = cell(raw.reason or "none recorded")
                    rows.append(
                        f"| `{m.model_id}` | {prompt} | {r.trial} | `{r.case_id}` | {verdict} | "
                        f"{reason} | {cell(snippet(r.output.output))} |"
                    )
    return [
        "## Answer key review",
        "",
        f"Every answer where the raw key and the reviewed key disagree: {len(rows)}. The reason is "
        "the raw key's; the output is cut to the first "
        f"{SNIPPET_CHARS} characters.",
        "",
        "| Model | Prompt | Trial | Case | Raw → reviewed | Raw key's reason | Output |",
        "|---|---|---:|---|---|---|---|",
        *rows,
        "",
    ]


def router_misses(models: list[ModelRuns], router: Study) -> list[str]:
    expected = {c.id: c.reference for c in router.cases.values()}
    rows = []
    for m in models:
        for r in m.router.results:
            if trial_outcome(r, "queue") is not True:
                rows.append(
                    f"| `{m.model_id}` | {r.trial} | `{r.case_id}` | "
                    f"{cell(expected[r.case_id] or '')} | {cell(r.output.output)} |"
                )
    return [
        "## Router misses",
        "",
        f"Every router answer that failed: {len(rows)}.",
        "",
        "| Model | Trial | Case | Expected | Output |",
        "|---|---:|---|---|---|",
        *rows,
        "",
    ]


def latency_cell(run: Run) -> str:
    """p50 / p95 over the rows that did not error, as ``evalproof report`` prints them."""
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    return f"{percentile(lat, 50):.0f} / {percentile(lat, 95):.0f}"


def latency(models: list[ModelRuns]) -> list[str]:
    out = [
        "## Latency",
        "",
        "p50 / p95 in milliseconds per answer. The grounded prompt run holds both trials.",
        "",
        "| Model | Grounded prompt | Friendly prompt | Ticket router |",
        "|---|---:|---:|---:|",
    ]
    for m in models:
        cells = [latency_cell(m.baseline), latency_cell(m.candidate), latency_cell(m.router)]
        out.append(f"| `{m.model_id}` | " + " | ".join(cells) + " |")
    return [*out, ""]


def check_naming(text: str, models: list[ModelRuns], studies: tuple[Study, ...]) -> None:
    """The output names models and providers only: no tool, target prefix, variant, isolation tag,
    tool version or suite name. A match is a wording bug to fix, never a check to loosen."""
    low = text.lower()
    banned = list(BANNED)
    exact = []  # whole names, so a name may sit inside a longer word or a hyphenated name
    for study in studies:
        exact += [study.config.name.lower(), *(v.name.lower() for v in study.config.variants)]
    for m in models:
        for run in m.all:
            env = run.info.environment
            banned += [
                env[k].lower()
                for k in ("system_cli", "system_cli_version", "system_isolation")
                if k in env
            ]
            banned.append(env["system_target"].split(":", 1)[0].lower())
            exact += [run.info.suite.lower(), (run.info.variant or "").lower()]
    for word in dict.fromkeys(banned):
        at = low.find(word)
        require(at < 0, f"output names {word!r}: …{text[max(0, at - 30) : at + 40]!r}…")
    # The providers' names are printed on purpose; the router's variants are bare provider names.
    providers = {m.provider.lower() for m in models}
    for word in dict.fromkeys(e for e in exact if e and e not in providers):
        found = re.search(rf"(?<![\w-]){re.escape(word)}(?![\w-])", low)
        require(found is None, f"output names {word!r}")
    found = BANNED_WORD.search(low)
    require(
        found is None,
        f"output has the word 'cli': …{text[max(0, found.start() - 30) : found.end() + 30]!r}…"
        if found
        else "",
    )


def main() -> None:
    support, router = load_study("support-qa.yaml"), load_study("ticket-router.yaml")
    models = [load_model(k) for k in MODELS]
    verdicts = (
        sum(rescore(run, support) for m in models for run in (m.baseline, m.t0, m.candidate)),
        sum(rescore(m.router, router) for m in models),
    )
    lines = [
        *setup(models, support, verdicts),
        *pass_rates(models),
        *run_to_run(models),
        *answer_text(models),
        *prompt_change(models, support),
        *key_review(models),
        *router_misses(models, router),
        *latency(models),
    ]
    text = "\n".join(lines).rstrip("\n") + "\n"
    check_naming(text, models, (support, router))
    print(text, end="")


if __name__ == "__main__":
    main()
