"""Is this eval drop real? A front end for evalproof's paired comparison and CI gate.

Both views run the library in ``src/`` (imported, not copied), through the same
``compare_runs`` and ``decide`` calls that ``evalproof compare --gate`` makes:

* **Check a drop** builds two runs from the pass/fail counts you enter.
* **The recorded run** replays the committed ``examples/support-qa`` runs: the
  comparison, the answers that changed, latency against a budget, and the judge
  calibrated on a gold sample of the size you choose. Nothing calls a model.
"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from evalproof import __version__  # noqa: E402
from evalproof.calibration import CalibrationReport, calibrate  # noqa: E402
from evalproof.compare import Comparison, compare_runs  # noqa: E402
from evalproof.config import GateConfig  # noqa: E402
from evalproof.gate import GateDecision, decide  # noqa: E402
from evalproof.models import CaseResult, Run, RunInfo, Score, SystemOutput  # noqa: E402
from evalproof.stats import mcnemar_exact, z_for  # noqa: E402

REPO = "https://github.com/technobasant/evalproof"
POST = "https://basantbhattarai.com.np/writing/eval-score-drop-twelve-answers/"
EXAMPLE = ROOT / "examples" / "support-qa"
MAX_CASES = 2000  # the paired bootstrap holds n_boot x n resamples in memory
N_BOOT = 5000  # GateConfig's default, so intervals match the CLI's
SCORERS = {"judge": "LLM judge (gated)", "answer_key": "Regex answer key"}


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def show_decision(d: GateDecision) -> None:
    lines = "\n".join(f"- {line}" for line in d.reasons + d.warnings)
    text = f"**Gate: {d.status}** (exit {d.exit_code})" + (f"\n\n{lines}" if lines else "")
    {"fail": st.error, "warn": st.warning}.get(d.status, st.success)(text)


def synthetic_run(label: str, passed: list[bool]) -> Run:
    """A run whose only scorer is pass/fail, shaped like a file `evalproof run` writes."""
    info = RunInfo(
        suite="calculator",
        system=label,
        dataset_path="entered counts",
        dataset_hash="entered-counts",
        started_at="",
        finished_at="",
        wall_time_s=0.0,
        evalproof_version=__version__,
        seed=0,
    )
    results = [
        CaseResult(
            case_id=f"c{i:05d}",
            output=SystemOutput(output=""),
            scores={"pass rate": Score(value=float(ok), passed=ok)},
        )
        for i, ok in enumerate(passed)
    ]
    return Run(info=info, results=results)


@st.cache_data(show_spinner=False)
def check_drop(
    n: int, base_pass: int, worse: int, better: int, max_drop: float, alpha: float, min_cases: int
) -> tuple[Comparison, GateDecision]:
    base = [True] * base_pass + [False] * (n - base_pass)
    cand = (
        [False] * worse
        + [True] * (base_pass - worse)
        + [True] * better
        + [False] * (n - base_pass - better)
    )
    cmp = compare_runs(
        synthetic_run("baseline", base),
        synthetic_run("candidate", cand),
        alpha=alpha,
        n_boot=N_BOOT,
    )
    gate = GateConfig(metric="pass rate", max_drop=max_drop, alpha=alpha, min_cases=min_cases)
    return cmp, decide(cmp, gate)


@lru_cache(maxsize=4096)
def p_at(n: int, rate_worse: float, rate_better: float) -> float:
    return mcnemar_exact(round(rate_worse * n), round(rate_better * n))


def cases_needed(n: int, worse: int, better: int, alpha: float, cap: int = 200_000) -> int | None:
    """Smallest suite at which the same per-case rates of change reach p < alpha.

    The normal approximation to McNemar brackets the answer; the exact test then
    bisects inside the bracket, so the slow exact sums only run near the threshold.
    """
    rw, rb = worse / n, better / n
    approx = z_for(alpha) ** 2 * (rw + rb) / (rw - rb) ** 2
    if approx > cap:
        return None
    hi = max(n, int(approx * 2) + 10)
    while p_at(hi, rw, rb) >= alpha:
        hi *= 2
        if hi > cap:
            return None
    lo = n
    while lo < hi:
        mid = (lo + hi) // 2
        if p_at(mid, rw, rb) < alpha:
            hi = mid
        else:
            lo = mid + 1
    return lo


@st.cache_data
def load_recorded() -> tuple[Run, Run, dict[str, str]]:
    runs = EXAMPLE / "runs"
    base = Run.model_validate_json((runs / "baseline.json").read_text())
    cand = Run.model_validate_json((runs / "candidate.json").read_text())
    questions = {}
    for line in (EXAMPLE / "cases.jsonl").read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            questions[case["id"]] = case["input"]["question"]
    return base, cand, questions


@st.cache_data(show_spinner=False)
def recorded_comparison(alpha: float) -> Comparison:
    base, cand, _ = load_recorded()
    return compare_runs(base, cand, alpha=alpha, n_boot=N_BOOT)


@st.cache_data(show_spinner=False)
def recorded_calibration(gold_n: int, seed: int) -> CalibrationReport:
    base, _, _ = load_recorded()
    return calibrate(
        base, "judge", gold_scorer="answer_key", gold_sample=gold_n, seed=seed, n_boot=N_BOOT
    )


def verdict(r: CaseResult, scorer: str) -> str:
    s = r.scores.get(scorer)
    if s is None or s.passed is None:
        return "undecided"
    return "pass" if s.passed else "fail"


def mean_tokens(run: Run) -> float:
    return sum(r.output.tokens_out for r in run.results) / len(run.results)


st.set_page_config(page_title="Is this eval drop real?", layout="wide")
st.title("Is this eval drop real?")
st.markdown(
    "A score that falls between two evaluation runs is either a regression or noise. "
    "Compared on the same cases, only the cases whose verdict *changed* carry evidence, and "
    "the exact McNemar test asks whether they changed one way more than chance allows. "
    "evalproof's CI gate fails a change only when the drop is both larger than your tolerance "
    "and unlikely to be noise. "
    f"Code: [evalproof]({REPO}) · write-up: [The eval score fell 4.8 points. Twelve answers "
    f"decided whether it meant anything.]({POST})"
)

tab_check, tab_run = st.tabs(["Check a drop", "The recorded run"])

with tab_check:
    st.markdown(
        "Enter two runs of the same pass/fail suite as counts. The defaults are the "
        "recorded run in the other tab: 83 questions, judged before and after a prompt change."
    )
    c1, c2, c3, c4 = st.columns(4)
    n = c1.number_input("Paired cases", min_value=1, max_value=MAX_CASES, value=83)
    base_pass = c2.number_input(
        "Passed in the baseline", min_value=0, max_value=MAX_CASES, value=64
    )
    worse = c3.number_input(
        "Passed → failed",
        min_value=0,
        max_value=MAX_CASES,
        value=8,
        help="Cases the baseline passed and the candidate failed.",
    )
    better = c4.number_input(
        "Failed → passed",
        min_value=0,
        max_value=MAX_CASES,
        value=4,
        help="Cases the baseline failed and the candidate passed.",
    )
    g1, g2, g3 = st.columns(3)
    max_drop_pts = g1.slider(
        "Tolerated drop (points)",
        0.0,
        20.0,
        3.0,
        0.5,
        help="The gate fails only drops at least this large *and* significant.",
    )
    alpha = g2.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)
    min_cases = g3.number_input(
        "Fewest paired cases the gate accepts", min_value=1, max_value=MAX_CASES, value=40
    )
    st.caption(
        f"Gate defaults are the example suite's. The app stops at {MAX_CASES:,} cases; the CLI "
        "has no limit."
    )

    problems = []
    if base_pass > n:
        problems.append("The baseline cannot pass more cases than there are.")
    if worse > base_pass:
        problems.append("Passed → failed cannot exceed the cases the baseline passed.")
    if better > n - base_pass:
        problems.append("Failed → passed cannot exceed the cases the baseline failed.")

    if problems:
        st.error(" ".join(problems))
    else:
        cmp, decision = check_drop(
            n, base_pass, worse, better, max_drop_pts / 100, alpha, min_cases
        )
        mc = cmp.metrics["pass rate"]
        lo_b, hi_b = mc.baseline_ci or (0.0, 1.0)
        lo_c, hi_c = mc.candidate_ci or (0.0, 1.0)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Baseline pass rate", pct(mc.baseline), help=f"95% Wilson {pct(lo_b)}–{pct(hi_b)}"
        )
        m2.metric(
            "Candidate pass rate", pct(mc.candidate), help=f"95% Wilson {pct(lo_c)}–{pct(hi_c)}"
        )
        m3.metric("Paired difference", pts(mc.delta))
        m4.metric("Exact McNemar p", f"{mc.p_value:.3g}", help=f"on {worse + better} changed cases")
        st.caption(
            f"{1 - alpha:.0%} interval for the difference: {pts(mc.ci_low)} to {pts(mc.ci_high)} "
            f"({N_BOOT:,} paired bootstrap resamples). {n - worse - better} of {n} cases kept "
            "their verdict and carry no evidence either way."
        )
        show_decision(decision)

        st.subheader("How many cases would settle it?")
        if worse == better:
            st.markdown(
                "As many cases got better as got worse, so no suite size makes this significant."
            )
        elif better > worse:
            st.markdown("The candidate improved. The gate only fails drops.")
        elif mc.p_value < alpha:
            st.markdown(f"Already significant at {n:,} cases (p = {mc.p_value:.3g} < {alpha}).")
        else:
            need = cases_needed(n, worse, better, alpha)
            rw, rb = worse / n, better / n
            if need is None:
                st.markdown("Not within 200,000 cases at these rates.")
            else:
                inside = (
                    " The drop is inside your tolerance, so even then the gate would only warn."
                    if -mc.delta < max_drop_pts / 100
                    else ""
                )
                st.markdown(
                    f"If {pct(rw)} of cases keep getting worse and {pct(rb)} keep getting "
                    f"better, the same drop reaches p < {alpha} at about **{need:,}** paired "
                    f"cases, {need / n:.1f}× this suite. Until then the honest reading is *not "
                    f"measured yet*, not *no regression*.{inside}"
                )
                top = max(need * 2, n * 4)
                steps = 40
                sizes = sorted(
                    {round(n * (top / n) ** (i / steps)) for i in range(steps + 1)} | {need}
                )
                curve = pd.DataFrame(
                    {"Paired cases": sizes, "McNemar p": [p_at(s, rw, rb) for s in sizes]}
                )
                x = alt.X(
                    "Paired cases:Q",
                    scale=alt.Scale(type="log", domain=[n, top], nice=False),
                    title="Paired cases (log scale)",
                )
                line = (
                    alt.Chart(curve)
                    .mark_line(interpolate="step-after")
                    .encode(
                        x=x,
                        y=alt.Y(
                            "McNemar p:Q", scale=alt.Scale(domain=[0, 1]), title="Exact McNemar p"
                        ),
                        tooltip=["Paired cases", alt.Tooltip("McNemar p:Q", format=".3f")],
                    )
                )
                rule = (
                    alt.Chart(pd.DataFrame({"alpha": [alpha]}))
                    .mark_rule(strokeDash=[4, 4], color="#c0392b")
                    .encode(y=alt.Y("alpha:Q", title=None))
                )
                marker = (
                    alt.Chart(pd.DataFrame({"Paired cases": [need]}))
                    .mark_rule(color="#7f8c8d")
                    .encode(x=x)
                )
                st.altair_chart(line + rule + marker, width="stretch")
                st.caption(
                    "Each point rounds the two rates to whole cases, so the curve steps. "
                    f"The dashed line is α = {alpha}; the grey line is {need:,} cases."
                )

with tab_run:
    base, cand, questions = load_recorded()
    st.markdown(
        f"The run behind the write-up: **{base.info.system}** answered "
        f"{len(base.results)} support questions about a fictional product, then "
        f"**{cand.info.system}** answered them again after the system prompt changed. A local "
        "qwen3:8b judged both against a rubric; a regex answer key per question stands in for "
        f"human review. Recorded on {base.info.started_at[:10]} and committed to the "
        "repository; this page replays the JSON."
    )
    r1, r2, r3 = st.columns(3)
    run_drop = r1.slider("Tolerated drop (points)", 0.0, 20.0, 3.0, 0.5, key="run_drop")
    run_alpha = r2.select_slider(
        "Significance level α", options=[0.01, 0.05, 0.10], value=0.05, key="run_alpha"
    )
    budget_on = r3.toggle("Add a p95 latency budget", value=False)
    budget = r3.number_input(
        "Budget (ms)",
        min_value=100,
        max_value=20000,
        value=2500,
        step=100,
        disabled=not budget_on,
    )

    rcmp = recorded_comparison(run_alpha)
    gate = GateConfig(
        metric="judge",
        max_drop=run_drop / 100,
        alpha=run_alpha,
        min_cases=40,
        max_p95_latency_ms=float(budget) if budget_on else None,
    )
    show_decision(decide(rcmp, gate))

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Scorer": SCORERS.get(name, name),
                    "Baseline": pct(m.baseline),
                    "Candidate": pct(m.candidate),
                    "Difference": pts(m.delta),
                    f"{1 - run_alpha:.0%} interval": f"{pts(m.ci_low)} to {pts(m.ci_high)}",
                    "Worse / better": f"{len(m.regressed)} / {len(m.improved)}",
                    "Exact McNemar p": f"{m.p_value:.3f}",
                }
                for name, m in rcmp.metrics.items()
            ]
        ),
        hide_index=True,
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Run": label,
                    "p50 latency": f"{s.p50_latency_ms:,.0f} ms",
                    "p95 latency": f"{s.p95_latency_ms:,.0f} ms",
                    "Mean answer length": f"{mean_tokens(run):.0f} tokens",
                }
                for label, s, run in (
                    ("Baseline", rcmp.baseline, base),
                    ("Candidate", rcmp.candidate, cand),
                )
            ]
        ),
        hide_index=True,
    )

    judge = rcmp.metrics["judge"]
    changed = [(cid, "worse") for cid in judge.regressed] + [
        (cid, "better") for cid in judge.improved
    ]
    st.subheader(f"The {len(changed)} answers that decided it")
    st.markdown(
        f"Of {judge.n_paired} paired questions, {judge.n_paired - len(changed)} kept the judge's "
        "verdict. These are the rest: everything the McNemar test sees."
    )
    b_by, c_by = base.by_id(), cand.by_id()
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Case": cid,
                    "Judge": direction,
                    "Answer key": f"{verdict(b_by[cid], 'answer_key')} → "
                    f"{verdict(c_by[cid], 'answer_key')}",
                    "Question": questions.get(cid, ""),
                    "Baseline answer": b_by[cid].output.output,
                    "Candidate answer": c_by[cid].output.output,
                    "Judge on the candidate": c_by[cid].scores["judge"].reason or "",
                }
                for cid, direction in changed
            ]
        ),
        hide_index=True,
        height=36 * (len(changed) + 1) + 2,
        column_config={
            name: st.column_config.TextColumn(width="medium")
            for name in (
                "Question",
                "Baseline answer",
                "Candidate answer",
                "Judge on the candidate",
            )
        },
    )
    st.caption("Select a cell to read the whole answer.")

    st.subheader("Can the judge's pass rate be trusted?")
    st.markdown(
        "A judge is a classifier with error rates, so its raw pass rate is biased. Correcting "
        "it needs gold labels. Here a random sample of the answer key plays the hand-reviewed "
        "set, and the whole key shows the truth the estimates are trying to recover."
    )
    k1, k2 = st.columns(2)
    gold_n = k1.slider("Gold labels (hand-reviewed cases)", 10, len(base.results) - 1, 30)
    seed = k2.number_input("Sample seed", min_value=0, max_value=10_000, value=0)
    rep = recorded_calibration(gold_n, seed)
    cm = rep.confusion
    kappa = "undefined" if cm.kappa is None else f"{cm.kappa:.2f}"
    st.markdown(
        f"On the {rep.n_gold} sampled cases the judge's true-positive rate is **{pct(cm.tpr)}**, "
        f"its false-positive rate **{pct(cm.fpr)}**, and Cohen's κ {kappa}."
    )
    estimates = [*rep.estimates, *([rep.truth] if rep.truth else [])]
    raw = rep.estimates[0]
    raw_width = (raw.high - raw.low) if raw.low is not None and raw.high is not None else None
    rows = []
    for e in estimates:
        width = (e.high - e.low) if e.low is not None and e.high is not None else None
        rows.append(
            {
                "Estimate": e.method[:1].upper() + e.method[1:],
                "Pass rate": pct(e.point),
                "95% interval": "—" if width is None else f"{pct(e.low)}–{pct(e.high)}",
                "Width": "—" if width is None else f"{width * 100:.1f} pts",
                "Width vs raw": f"{width / raw_width:.1f}×" if width and raw_width else "—",
                "Note": e.note,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption(
        "ACC and PPI correct the raw rate with the measured error rates, and their intervals "
        "carry the uncertainty of those rates too. With few gold labels that uncertainty can "
        "outweigh the bias being removed. Move the slider to see where it stops doing so."
    )

st.divider()
st.caption(
    f"evalproof {__version__}, run from this repository's src/. "
    f"[Source]({REPO}/tree/main/app) · [Measurements]({REPO}/blob/main/MEASUREMENTS.md)"
)
