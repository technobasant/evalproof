"""Recompute every table of the three-model study from the committed run files. No model needed.

    uv run python examples/cli-study/measure_cli.py

It reads the 12 files in ``runs/`` and prints Markdown to stdout. Pass-rate intervals, the paired
comparison, the gate verdict and the power estimate come from evalproof itself, so they match what
``evalproof report`` and ``evalproof compare`` print for the same files.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path

from evalproof.compare import compare_runs
from evalproof.config import GateConfig, load_config
from evalproof.dataset import dataset_hash, load_dataset
from evalproof.gate import EXIT_CANNOT_GATE, decide
from evalproof.models import CaseResult, Run
from evalproof.runner import load_run
from evalproof.stats import percentile, wilson
from evalproof.trials import case_counts, pooled_rate_ci, trial_outcome

HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs"
MODELS = ("claude", "codex", "gemini")
KEYS = ("answer_key", "answer_key_reviewed")
SNIPPET_CHARS = 120
VERDICT = {"pass": "pass", "warn": "pass with warnings", "fail": "fail"}

# Words that must never reach the output: the tools the models were reached through, and
# anything that names them. "cli" is matched as a whole word only, so "client" and "click" in a
# model's answer pass; the suites' own names ("...-cli") do not.
BANNED = ("claude-code", "claude code", "codex", "cursor")
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


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AssertionError(message)


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
            env.get("system_sampling") == "cli-default",
            f"{key}: sampling is not the provider default",
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
    interval."""
    if run.trials > 1:
        counts = list(case_counts(run, scorer).values())
        passes = sum(p for p, _ in counts)
        n = sum(d for _, d in counts)
        lo, hi = pooled_rate_ci(counts)
    else:
        scores = [r.scores[scorer] for r in run.results if scorer in r.scores]
        binary = [s.passed for s in scores if s.value is not None and s.passed is not None]
        passes, n = sum(binary), len(binary)
        lo, hi = wilson(passes, n)
    return f"{passes}/{n} ({pct(passes / n)}, [{pct(lo)}, {pct(hi)}])"


def trial_rows(run: Run, trial: int) -> dict[str, CaseResult]:
    return {r.case_id: r for r in run.results if r.trial == trial}


def setup(models: list[ModelRuns]) -> list[str]:
    suite = load_config(HERE / "support-qa.yaml")
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
    out += ["", "Sampling is the provider default; no temperature or seed control.", ""]
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
    for m in models:
        cells = [
            rate_cell(m.baseline, "answer_key"),
            rate_cell(m.baseline, "answer_key_reviewed"),
            rate_cell(m.candidate, "answer_key"),
            rate_cell(m.candidate, "answer_key_reviewed"),
            rate_cell(m.router, "queue"),
        ]
        out.append(f"| `{m.model_id}` | " + " | ".join(cells) + " |")
    return [*out, ""]


def run_to_run(models: list[ModelRuns]) -> list[str]:
    out = [
        "## Run-to-run (A/A)",
        "",
        "The same prompt on the same cases twice: trial 0 against trial 1. Any difference is "
        "sampling noise.",
        "",
        "| Model | Metric | Trial 0 | Trial 1 | Cases that differ | Case ids |",
        "|---|---|---|---|---|---|",
    ]
    for m in models:
        for run, scorer in (
            (m.baseline, "answer_key"),
            (m.baseline, "answer_key_reviewed"),
            (m.router, "queue"),
        ):
            t0, t1 = trial_rows(run, 0), trial_rows(run, 1)
            require(set(t0) == set(t1), f"{m.key}: trials cover different cases")
            rates = []
            for rows in (t0, t1):
                outcomes = [trial_outcome(r, scorer) for r in rows.values()]
                decided = [o for o in outcomes if o is not None]
                rates.append(f"{sum(decided)}/{len(decided)} ({pct(sum(decided) / len(decided))})")
            differ = [c for c in t0 if trial_outcome(t0[c], scorer) != trial_outcome(t1[c], scorer)]
            out.append(
                f"| `{m.model_id}` | `{scorer}` | {rates[0]} | {rates[1]} | {len(differ)} "
                f"of {len(t0)} | {ids(differ)} |"
            )
    return [*out, ""]


def identical_text(run: Run) -> str:
    """``n of N``: the cases whose output text is byte-identical in trial 0 and trial 1."""
    t0, t1 = trial_rows(run, 0), trial_rows(run, 1)
    require(set(t0) == set(t1), "trials cover different cases")
    same = sum(1 for c in t0 if t0[c].output.output == t1[c].output.output)
    return f"{same} of {len(t0)}"


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


def prompt_change(models: list[ModelRuns]) -> list[str]:
    gate = load_config(HERE / "support-qa.yaml").gate
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
        "| Gate verdict | MDE at 80% power |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    notes = []
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
            mde = decision.power[name].mde
            out.append(
                f"| `{m.model_id}` | `{name}` | {pct(mc.baseline)} | {pct(mc.candidate)} | "
                f"{100 * mc.delta:+.1f} pts [{100 * mc.ci_low:+.1f}, {100 * mc.ci_high:+.1f}] | "
                f"{mc.p_value:.4f} | {verdict} | "
                f"{'not estimable' if mde is None else f'{100 * mde:.1f} pts'} |"
            )
    return [*out, "", *notes, *([""] if notes else [])]


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


def router_misses(models: list[ModelRuns]) -> list[str]:
    suite = load_config(HERE / "ticket-router.yaml")
    cases = load_dataset(suite.resolve(suite.dataset))
    expected = {c.id: c.reference for c in cases}
    rows = []
    for m in models:
        require(
            dataset_hash(cases) == m.router.info.dataset_hash,
            f"{m.key}: the router dataset changed since the run",
        )
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


def check_naming(text: str, models: list[ModelRuns]) -> None:
    """The output names models and providers only: no tool, target prefix, variant, isolation tag,
    tool version or suite name. A match is a wording bug to fix, never a check to loosen."""
    low = text.lower()
    banned = list(BANNED)
    exact = []  # whole names: a variant "claude" must not match inside "claude-sonnet-5-5"
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
    for word in dict.fromkeys(e for e in exact if e):
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
    models = [load_model(k) for k in MODELS]
    lines = [
        *setup(models),
        *pass_rates(models),
        *run_to_run(models),
        *answer_text(models),
        *prompt_change(models),
        *key_review(models),
        *router_misses(models),
        *latency(models),
    ]
    text = "\n".join(lines).rstrip("\n") + "\n"
    check_naming(text, models)
    print(text, end="")


if __name__ == "__main__":
    main()
