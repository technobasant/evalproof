"""Is this eval drop real? An interactive front end for evalproof's paired comparison and gate.

Both views run the library in ``src/`` (imported, not copied), through the same
``compare_runs`` and ``decide`` calls that ``evalproof compare --gate`` makes:

* **Check a drop**: pick a scenario or move the counts of two runs, and watch the
  verdict, the cases that carry evidence, and the coin-flip test behind the p-value.
* **The recorded run**: the committed ``examples/support-qa`` runs, replayed. Select a
  changed answer to read both versions, set a latency budget, and calibrate the judge
  on a gold sample of the size you choose. Nothing calls a model.
"""

from __future__ import annotations

import json
import math
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import altair as alt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

# Community Cloud pulls each push into the running process and re-reads this script on
# every run, but imported modules and st.cache_data results outlive the push: the app
# would keep serving the previous library while saying it runs this repository's code.
# Fingerprint the code and data it runs, and start afresh when they change.
_inputs = [*(ROOT / "src" / "evalproof").rglob("*.py"), *(ROOT / "examples").rglob("*.json*")]
_stamp = tuple(p.stat().st_mtime_ns for p in sorted(_inputs))
if getattr(sys.modules.get("evalproof"), "app_stamp", None) != _stamp:
    for _name in [m for m in sys.modules if m.split(".")[0] == "evalproof"]:
        del sys.modules[_name]
    st.cache_data.clear()

import evalproof  # noqa: E402
from evalproof import __version__  # noqa: E402
from evalproof.calibration import CalibrationReport, calibrate  # noqa: E402
from evalproof.compare import Comparison, MetricComparison, compare_runs  # noqa: E402
from evalproof.config import GateConfig  # noqa: E402
from evalproof.gate import (  # noqa: E402
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_REGRESSION,
    GateDecision,
    decide,
)
from evalproof.models import CaseResult, Run, RunInfo, Score, SystemOutput  # noqa: E402
from evalproof.stats import mcnemar_exact, z_for  # noqa: E402

evalproof.app_stamp = _stamp  # type: ignore[attr-defined]

REPO = "https://github.com/technobasant/evalproof"
POST = "https://basantbhattarai.com.np/writing/eval-score-drop-twelve-answers/"
EXAMPLE = ROOT / "examples" / "support-qa"
MAX_CASES = 2000  # the paired bootstrap holds n_boot x n resamples in memory
N_BOOT = 5000  # GateConfig's default, so intervals match the CLI's

# One palette for the case grid, the coin-flip chart and their legends.
CASE_COLOURS = {
    "Passed → failed": "#d1495b",
    "Failed → passed": "#2a9d8f",
    "Passed both times": "#c9ced6",
    "Failed both times": "#7d8590",
}
TAIL, BODY = "#e9a23b", "#c9ced6"
GATE_COLOURS = {"fail": "red", "warn": "orange", "pass": "green"}

PRESETS = {
    "The recorded run": {"n": 83, "base_pass": 64, "worse": 8, "better": 4},
    "Same rates, 4× the cases": {"n": 332, "base_pass": 256, "worse": 32, "better": 16},
    "Small but certain": {"n": 2000, "base_pass": 1540, "worse": 60, "better": 25},
    "Twenty cases": {"n": 20, "base_pass": 16, "worse": 4, "better": 0},
}
PRESET_NOTES = {
    "The recorded run": "The prompt change behind the write-up: 8 answers got worse and 4 "
    "got better, out of 83.",
    "Same rates, 4× the cases": "Exactly the same rates of change, on a suite four times "
    "larger. Only the amount of evidence moves.",
    "Small but certain": "A 1.75-point drop on 2,000 cases: almost certainly real, and still "
    "inside a 3-point tolerance.",
    "Twenty cases": "A 20-point drop on a 20-case suite. Big, and still not evidence.",
}


# ---------------------------------------------------------------- computation


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
def load_recorded() -> tuple[Run, Run, dict[str, dict[str, str]]]:
    runs = EXAMPLE / "runs"
    base = Run.model_validate_json((runs / "baseline.json").read_text())
    cand = Run.model_validate_json((runs / "candidate.json").read_text())
    cases = {}
    for line in (EXAMPLE / "cases.jsonl").read_text().splitlines():
        if line.strip():
            case = json.loads(line)
            cases[case["id"]] = {
                "question": case["input"]["question"],
                "reference": case.get("reference") or "",
            }
    return base, cand, cases


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


# ------------------------------------------------------------------- display


def pct(x: float | None) -> str:
    return "—" if x is None else f"{x:.1%}"


def pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def chance(p: float) -> str:
    if p >= 0.1:
        return f"{p:.0%}"
    if p >= 0.001:
        return f"{p:.1%}"
    return "less than 0.1%"


def headline(d: GateDecision, mc: MetricComparison, alpha: float) -> str:
    noise = mc.p_value >= alpha
    if d.exit_code == EXIT_CANNOT_GATE:
        return "Too few cases to gate"
    if d.exit_code == EXIT_REGRESSION:
        return "A real regression"
    if d.exit_code == EXIT_BUDGET:
        return "Over budget; the quality drop is noise" if noise else "Over budget"
    if mc.delta == 0:
        return "No net change"
    if noise:
        return "Not distinguishable from noise"
    if mc.delta > 0:
        return "A real improvement"
    return "A real drop, inside your tolerance"


def evidence(mc: MetricComparison) -> str:
    worse, better = len(mc.regressed), len(mc.improved)
    changed = worse + better
    if changed == 0:
        return f"None of the {mc.n_paired:,} cases changed its verdict: no evidence either way."
    head = (
        f"**{changed} of {mc.n_paired:,}** cases changed: {worse} got worse, {better} got "
        "better. If the change did nothing, each changed case is a coin flip"
    )
    if worse == better:
        return f"{head}, and an even split is exactly what chance predicts."
    return (
        f"{head}, and a split at least as lopsided as {max(worse, better)}–"
        f"{min(worse, better)} comes up **{chance(mc.p_value)}** of the time."
    )


def verdict_card(d: GateDecision, mc: MetricComparison, alpha: float) -> None:
    with st.container(border=True):
        colour = "gray" if d.exit_code == EXIT_CANNOT_GATE else GATE_COLOURS[d.status]
        st.badge(f"Gate: {d.status} · exit {d.exit_code}", color=colour)  # type: ignore[arg-type]
        st.markdown(f"## {headline(d, mc, alpha)}")
        st.markdown(evidence(mc))
        lines = "; ".join(d.reasons + d.warnings) or "no reasons or warnings"
        st.caption(f"`evalproof compare --gate` → exit {d.exit_code} ({d.status}): {lines}")


def case_grid(n: int, base_pass: int, worse: int, better: int) -> alt.Chart:
    """Every paired case as one cell, the changed ones first."""
    kinds = (
        ["Passed → failed"] * worse
        + ["Failed → passed"] * better
        + ["Passed both times"] * (base_pass - worse)
        + ["Failed both times"] * (n - base_pass - better)
    )
    # At most 340 px wide, so the grid fits a phone; the legend is drawn below it.
    cols = min(60, max(12, math.ceil(math.sqrt(n * 3))))
    cell = max(5, min(20, 340 // cols))
    grid = pd.DataFrame({"Case": kinds, "i": range(n)})
    grid["col"], grid["row"] = grid["i"] % cols, grid["i"] // cols
    return (
        alt.Chart(grid)
        .mark_rect(cornerRadius=2, stroke="white", strokeWidth=max(1, cell // 8))
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", axis=None),
            color=alt.Color(
                "Case:N",
                scale=alt.Scale(domain=list(CASE_COLOURS), range=list(CASE_COLOURS.values())),
                legend=None,
            ),
            tooltip=["Case"],
        )
        .properties(width=alt.Step(cell), height=alt.Step(cell))
        .configure_view(stroke=None)
    )


def grid_legend(counts: dict[str, int]) -> str:
    """The case grid's legend as HTML, so it wraps on a phone instead of being clipped."""
    items = "".join(
        f'<span style="display:inline-flex;align-items:center;gap:.4rem;margin:0 1.1rem .3rem 0">'
        f'<span style="width:.75rem;height:.75rem;border-radius:2px;background:{CASE_COLOURS[k]}">'
        f"</span>{k} · {v:,}</span>"
        for k, v in counts.items()
    )
    return f'<div style="font-size:.85rem;display:flex;flex-wrap:wrap">{items}</div>'


def coin_flips(worse: int, better: int) -> alt.Chart:
    """Binomial(changed, 1/2): how the changed cases would split if nothing had changed."""
    m = worse + better
    lo, hi = min(worse, better), max(worse, better)
    mean, sd = m / 2, math.sqrt(m) / 2
    k_min = max(0, min(lo, math.floor(mean - 4.5 * sd) - 1))
    k_max = min(m, max(hi, math.ceil(mean + 4.5 * sd) + 1))
    log_half = m * math.log(2)
    rows = []
    for k in range(k_min, k_max + 1):
        logp = math.lgamma(m + 1) - math.lgamma(k + 1) - math.lgamma(m - k + 1) - log_half
        tail = k <= lo or k >= hi
        rows.append(
            {
                "Got worse": k,
                "Chance": math.exp(logp),
                "Split": "At least as lopsided as yours" if tail else "Less lopsided",
            }
        )
    every = max(1, math.ceil((k_max - k_min + 1) / 14))
    ticks = [k for k in range(k_min, k_max + 1) if (k - k_min) % every == 0]
    return (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
        .encode(
            x=alt.X(
                "Got worse:O",
                title=f"How many of the {m} changed cases got worse",
                axis=alt.Axis(labelAngle=0, values=ticks),
            ),
            y=alt.Y("Chance:Q", axis=alt.Axis(format="%"), title="Chance"),
            color=alt.Color(
                "Split:N",
                scale=alt.Scale(
                    domain=["At least as lopsided as yours", "Less lopsided"], range=[TAIL, BODY]
                ),
                legend=alt.Legend(orient="top", title=None, labelLimit=0),
            ),
            tooltip=["Got worse", alt.Tooltip("Chance:Q", format=".2%"), "Split"],
        )
        .properties(height=300)
    )


def verdict(r: CaseResult, scorer: str) -> str:
    s = r.scores.get(scorer)
    if s is None or s.passed is None:
        return "undecided"
    return "pass" if s.passed else "fail"


def mean_tokens(run: Run) -> float:
    return sum(r.output.tokens_out for r in run.results) / len(run.results)


def apply_preset() -> None:
    name = st.session_state.get("preset")
    if name:
        st.session_state.update(PRESETS[name])


def clear_preset() -> None:
    st.session_state["preset"] = None


# ----------------------------------------------------------------------- page

st.set_page_config(
    page_title="Is this eval drop real? · evalproof",
    page_icon=":material/balance:",
    layout="wide",
)
st.title("Is this eval drop real?")
st.markdown(
    "Two runs of the same eval, before and after a change. Only the cases whose verdict "
    "**changed** carry evidence, and evalproof's CI gate fails a change only when the drop is "
    "both bigger than your tolerance *and* unlikely to be noise."
)
st.caption(
    f"[evalproof on GitHub]({REPO}) · [The write-up: twelve answers decided it]({POST}) · "
    f"evalproof {__version__}, run live from the repository"
)

tab_check, tab_run = st.tabs(["Check a drop", "The recorded run"])

with tab_check:
    for key, value in PRESETS["The recorded run"].items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("preset", "The recorded run")

    st.pills("Start from a scenario", list(PRESETS), key="preset", on_change=apply_preset)
    if st.session_state.get("preset"):
        st.caption(PRESET_NOTES[st.session_state["preset"]])

    controls, results = st.columns([1, 2], gap="large")
    with controls, st.container(border=True):
        st.markdown("**Two runs on the same cases**")
        n = st.slider("Paired cases", 10, MAX_CASES, key="n", on_change=clear_preset)
        st.session_state["base_pass"] = min(st.session_state["base_pass"], n)
        base_pass = st.slider(
            "Passed in the baseline", 0, n, key="base_pass", on_change=clear_preset
        )
        st.session_state["worse"] = min(st.session_state["worse"], base_pass)
        st.session_state["better"] = min(st.session_state["better"], n - base_pass)
        worse = st.slider(
            "Passed → failed",
            0,
            max(base_pass, 1),
            key="worse",
            on_change=clear_preset,
            help="Cases the baseline passed and the candidate failed.",
        )
        better = st.slider(
            "Failed → passed",
            0,
            max(n - base_pass, 1),
            key="better",
            on_change=clear_preset,
            help="Cases the baseline failed and the candidate passed.",
        )
        with st.expander("Gate settings"):
            max_drop_pts = st.slider(
                "Tolerated drop (points)",
                0.0,
                20.0,
                3.0,
                0.5,
                help="The gate fails only drops at least this large *and* significant.",
            )
            alpha = st.select_slider("Significance level α", options=[0.01, 0.05, 0.10], value=0.05)
            min_cases = st.number_input(
                "Fewest paired cases the gate accepts", 1, MAX_CASES, value=40
            )
            st.caption("Defaults are the example suite's. The CLI has no case limit.")

    with results:
        if worse > base_pass or better > n - base_pass:
            st.error("Those counts cannot happen together: lower the changed cases.")
            st.stop()
        cmp, decision = check_drop(
            n, base_pass, worse, better, max_drop_pts / 100, alpha, int(min_cases)
        )
        mc = cmp.metrics["pass rate"]
        verdict_card(decision, mc, alpha)
        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline pass rate", pct(mc.baseline), border=True)
        m2.metric("Candidate pass rate", pct(mc.candidate), delta=pts(mc.delta), border=True)
        m3.metric(
            "Exact McNemar p",
            f"{mc.p_value:.3g}",
            border=True,
            help=f"{1 - alpha:.0%} paired-bootstrap interval for the difference: "
            f"{pts(mc.ci_low)} to {pts(mc.ci_high)}",
        )
        st.markdown(f"**The {n:,} paired cases**")
        st.altair_chart(case_grid(n, base_pass, worse, better), width="content")
        counts = {
            "Passed → failed": worse,
            "Failed → passed": better,
            "Passed both times": base_pass - worse,
            "Failed both times": n - base_pass - better,
        }
        st.html(grid_legend(counts))
        if worse + better:
            st.markdown("**The coin-flip test behind the p-value**")
            st.caption(
                "If the change did nothing, each changed case is equally likely to go either "
                "way. The bars show how the changed cases would split by chance alone; the "
                "orange ones are splits at least as lopsided as yours, and together they are "
                "the p-value."
            )
            st.altair_chart(coin_flips(worse, better), width="stretch")

    st.divider()
    st.subheader("How many cases would settle it?")
    if worse == better:
        st.markdown("As many cases got better as got worse, so no suite size makes it significant.")
    elif better > worse:
        st.markdown("The candidate improved. The gate only fails drops.")
    elif mc.p_value < alpha:
        st.markdown(f"Already settled at {n:,} cases: p = {mc.p_value:.3g} < {alpha}.")
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
                f"If {pct(rw)} of cases keep getting worse and {pct(rb)} keep getting better, "
                f"the same drop reaches p < {alpha} at about **{need:,}** paired cases, "
                f"{need / n:.1f}× this suite. Until then the honest reading is *not measured "
                f"yet*, not *no regression*.{inside}"
            )
            top = max(need * 2, n * 4)
            steps = 40
            sizes = sorted({round(n * (top / n) ** (i / steps)) for i in range(steps + 1)} | {need})
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
                .mark_line(interpolate="step-after", color="#5a43a6")
                .encode(
                    x=x,
                    y=alt.Y("McNemar p:Q", scale=alt.Scale(domain=[0, 1]), title="Exact McNemar p"),
                    tooltip=["Paired cases", alt.Tooltip("McNemar p:Q", format=".3f")],
                )
            )
            rule = (
                alt.Chart(pd.DataFrame({"alpha": [alpha]}))
                .mark_rule(strokeDash=[4, 4], color="#d1495b")
                .encode(y=alt.Y("alpha:Q", title=None))
            )
            marker = (
                alt.Chart(pd.DataFrame({"Paired cases": [need]}))
                .mark_rule(color="#7d8590")
                .encode(x=x)
            )
            st.altair_chart((line + rule + marker).properties(height=260), width="stretch")
            st.caption(
                "Each point rounds the two rates to whole cases, so the curve steps. "
                f"Dashed: α = {alpha}. Grey: {need:,} cases."
            )

with tab_run:
    base, cand, cases = load_recorded()
    st.markdown(
        f"A local **{base.info.system}** answered {len(base.results)} support questions about "
        f"a product; **{cand.info.system}** answered them again after the system "
        "prompt changed. qwen3:8b judged both, and a regex answer key per question stands in "
        f"for human review. Recorded {base.info.started_at[:10]}; this page replays the JSON."
    )
    controls, results = st.columns([1, 2], gap="large")
    with controls, st.container(border=True):
        st.markdown("**Gate settings**")
        run_drop = st.slider("Tolerated drop (points)", 0.0, 20.0, 3.0, 0.5, key="run_drop")
        run_alpha = st.select_slider(
            "Significance level α", options=[0.01, 0.05, 0.10], value=0.05, key="run_alpha"
        )
        budget_on = st.toggle("Add a p95 latency budget", value=False)
        budget = st.slider(
            "p95 budget (ms)", 500, 6000, 2500, 100, disabled=not budget_on, key="budget"
        )

    rcmp = recorded_comparison(run_alpha)
    gate = GateConfig(
        metric="judge",
        max_drop=run_drop / 100,
        alpha=run_alpha,
        min_cases=40,
        max_p95_latency_ms=float(budget) if budget_on else None,
    )
    judge, key_mc = rcmp.metrics["judge"], rcmp.metrics["answer_key"]
    with results:
        verdict_card(decide(rcmp, gate), judge, run_alpha)
        m1, m2, m3 = st.columns(3)
        m1.metric(
            "LLM judge (gated)",
            pct(judge.candidate),
            delta=pts(judge.delta),
            border=True,
            help=f"Baseline {pct(judge.baseline)}; McNemar p = {judge.p_value:.3f}",
        )
        m2.metric(
            "Regex answer key",
            pct(key_mc.candidate),
            delta=pts(key_mc.delta),
            border=True,
            help=f"Baseline {pct(key_mc.baseline)}; McNemar p = {key_mc.p_value:.3f}",
        )
        m3.metric(
            "p95 latency",
            f"{rcmp.candidate.p95_latency_ms:,.0f} ms",
            delta=f"{rcmp.candidate.p95_latency_ms - rcmp.baseline.p95_latency_ms:+,.0f} ms",
            delta_color="inverse",
            border=True,
            help=f"Mean answer {mean_tokens(base):.0f} → {mean_tokens(cand):.0f} tokens",
        )

    changed = [(cid, "got worse") for cid in judge.regressed] + [
        (cid, "got better") for cid in judge.improved
    ]
    st.subheader(f"The {len(changed)} answers that decided it")
    st.caption(
        f"{judge.n_paired - len(changed)} of {judge.n_paired} questions kept the judge's "
        "verdict. These are the rest, everything the McNemar test sees. Select a row to read "
        "both answers."
    )
    b_by, c_by = base.by_id(), cand.by_id()
    table = pd.DataFrame(
        [
            {
                "Case": cid,
                "Judge": direction,
                "Answer key": f"{verdict(b_by[cid], 'answer_key')} → "
                f"{verdict(c_by[cid], 'answer_key')}",
                "Question": cases[cid]["question"],
            }
            for cid, direction in changed
        ]
    )
    picked = st.dataframe(
        table,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="changed_table",
        height=36 * (len(changed) + 1) + 2,
        column_config={"Question": st.column_config.TextColumn(width="large")},
    )
    rows = picked.selection.rows if picked is not None and picked.selection else []
    cid = changed[rows[0]][0] if rows else changed[0][0]
    st.markdown(f"**{cid} · {cases[cid]['question']}**")
    st.caption(f"Reference answer: {cases[cid]['reference']}")
    before, after = st.columns(2)
    for col, label, run_by in (
        (before, "Before: grounded prompt", b_by),
        (after, "After: friendly prompt", c_by),
    ):
        r = run_by[cid]
        judged, keyed = verdict(r, "judge"), verdict(r, "answer_key")
        with col, st.container(border=True):
            st.markdown(f"**{label}**")
            with st.container(horizontal=True):
                st.badge(f"judge: {judged}", color="green" if judged == "pass" else "red")
                st.badge(f"answer key: {keyed}", color="green" if keyed == "pass" else "red")
            st.markdown(r.output.output)
            st.caption(f"Judge's reason: {r.scores['judge'].reason or '—'}")

    st.divider()
    st.subheader("Can the judge's pass rate be trusted?")
    st.markdown(
        "A judge is a classifier with error rates, so its raw pass rate is biased. Correcting "
        "it needs gold labels. Here a random sample of the answer key plays the hand-reviewed "
        "set, and the whole key shows the truth the estimates are trying to recover."
    )
    k_controls, k_results = st.columns([1, 2], gap="large")
    with k_controls, st.container(border=True):
        gold_n = st.slider("Gold labels (hand-reviewed cases)", 10, len(base.results) - 1, 30)
        seed = st.number_input("Sample seed", min_value=0, max_value=10_000, value=0)
    rep = recorded_calibration(gold_n, int(seed))
    cm = rep.confusion
    kappa = "undefined" if cm.kappa is None else f"{cm.kappa:.2f}"
    raw = rep.estimates[0]
    raw_width = (raw.high - raw.low) if raw.low is not None and raw.high is not None else None
    est_rows = []
    for e in [*rep.estimates, *([rep.truth] if rep.truth else [])]:
        width = (e.high - e.low) if e.low is not None and e.high is not None else None
        est_rows.append(
            {
                "Estimate": e.method[:1].upper() + e.method[1:],
                "Pass rate": pct(e.point),
                "95% interval": "—" if width is None else f"{pct(e.low)}–{pct(e.high)}",
                "Width vs raw": f"{width / raw_width:.1f}×" if width and raw_width else "—",
                "Note": e.note,
            }
        )
    with k_results:
        st.markdown(
            f"On the {rep.n_gold} sampled cases the judge's true-positive rate is "
            f"**{pct(cm.tpr)}**, its false-positive rate **{pct(cm.fpr)}**, and Cohen's κ "
            f"{kappa}."
        )
        st.dataframe(pd.DataFrame(est_rows), hide_index=True)
        st.caption(
            "ACC and PPI correct the raw rate with the measured error rates, and their "
            "intervals carry the uncertainty of those rates too. With few gold labels that "
            "uncertainty can outweigh the bias being removed. Move the slider to see where it "
            "stops doing so."
        )

st.divider()
st.caption(
    f"[Source of this app]({REPO}/tree/main/app) · "
    f"[Measurements]({REPO}/blob/main/MEASUREMENTS.md) · "
    "Built on evalproof's own compare and gate code."
)
