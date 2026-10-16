"""Recompute every figure in MEASUREMENTS.md from the committed run files. No model needed.

    uv run python examples/support-qa/measure.py

The answer key (a regex per question, written before any model output was seen) stands
in for human review. "Truth" below means the answer key on all 83 cases — the
measurements also report where that key is itself wrong.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path

from evalproof.calibration import (
    acc_estimate,
    calibrate,
    confusion,
    ppi_estimate,
    raw_estimate,
)
from evalproof.compare import compare_runs
from evalproof.config import load_config
from evalproof.gate import decide
from evalproof.report import calibration_markdown, comparison_markdown
from evalproof.runner import load_run
from evalproof.stats import percentile

HERE = Path(__file__).parent
SEEDS = 200
GOLD_N = 30

# Every case where the answer key and the two judges (thinking off and on) did not all
# agree, read by hand against the rubric and the product facts: section 4 read the first
# 13, section 8 the 9 that only the thinking judge disputes. True is a pass, None is
# ambiguous (excluded from the audited comparison).
AUDITED = {
    # section 4
    ("q76", "baseline"): False,
    ("q76", "candidate"): False,
    ("q78", "baseline"): False,
    ("q78", "candidate"): False,
    ("q77", "baseline"): False,
    ("q28", "candidate"): False,
    ("q31", "candidate"): False,
    ("q56", "baseline"): True,
    ("q65", "baseline"): True,
    ("q32", "candidate"): True,
    ("q05", "candidate"): False,
    ("q66", "baseline"): None,
    ("q66", "candidate"): None,
    # section 8
    ("q28", "baseline"): False,
    ("q07", "candidate"): False,
    ("q42", "baseline"): True,
    ("q42", "candidate"): True,
    ("q83", "baseline"): True,
    ("q21", "baseline"): True,
    ("q22", "baseline"): True,
    ("q21", "candidate"): True,
    ("q22", "candidate"): True,
}


def verdicts(run, name):
    return {
        r.case_id: r.scores[name].passed for r in run.results if r.scores[name].passed is not None
    }


def sample_study(run, label: str) -> list[str]:
    """Draw a 30-case gold sample 200 times; how often does each estimate cover the truth?"""
    judge, key = verdicts(run, "judge"), verdicts(run, "answer_key")
    ids = sorted(set(judge) & set(key))
    truth = sum(key[c] for c in ids) / len(ids)
    rows = {"raw judge": [], "gold sample only": [], "ACC": [], "PPI": []}
    for seed in range(SEEDS):
        lab = sorted(random.Random(seed).sample(ids, GOLD_N))
        unl = [c for c in ids if c not in set(lab)]
        jl, gl = [judge[c] for c in lab], [key[c] for c in lab]
        ests = {
            "raw judge": raw_estimate([judge[c] for c in ids]),
            "gold sample only": raw_estimate(gl),  # same Wilson interval, on 30 labels
            "ACC": acc_estimate([judge[c] for c in ids], jl, gl, n_boot=2000, seed=seed),
            "PPI": ppi_estimate([judge[c] for c in unl], jl, gl),
        }
        for k, e in ests.items():
            rows[k].append(e)
    out = [
        f"#### {label}: {SEEDS} random gold samples of {GOLD_N} (truth = {100 * truth:.1f}%)",
        "",
        "| Estimate | Mean point | Mean \\|error\\| | 95% CI covers truth | Mean CI width "
        "| Undefined |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, es in rows.items():
        ok = [e for e in es if e.point is not None and e.low is not None and e.high is not None]
        undefined = len(es) - len(ok)
        mean_pt = statistics.mean(e.point for e in ok)
        mae = statistics.mean(abs(e.point - truth) for e in ok)
        cover = sum(e.low <= truth <= e.high for e in ok) / len(ok)
        width = statistics.mean(e.high - e.low for e in ok)
        out.append(
            f"| {k} | {100 * mean_pt:.1f}% | {100 * mae:.1f} pts | {100 * cover:.1f}% | "
            f"{100 * width:.1f} pts | {undefined} |"
        )
    return out


def latency(run) -> str:
    lat = [r.output.latency_ms for r in run.results if not r.output.error]
    toks = [r.output.tokens_out for r in run.results]
    return (
        f"system p50 {percentile(lat, 50):.0f} ms, p95 {percentile(lat, 95):.0f} ms, "
        f"mean output {statistics.mean(toks):.0f} tokens; "
        f"run wall time {run.info.wall_time_s:.0f} s"
    )


def audited(run, label: str) -> dict[str, bool]:
    """The answer key with every hand-read verdict applied; ambiguous cases left out."""
    out = {}
    for c, v in verdicts(run, "answer_key").items():
        truth = AUDITED.get((c, label), v)
        if truth is not None:
            out[c] = truth
    return out


def judge_row(run, truth: dict[str, bool]) -> str:
    """Pass rate, agreement with ``truth`` and cost of one run's judge, as table cells."""
    judge = verdicts(run, "judge")
    ids = sorted(set(judge) & set(truth))
    cm = confusion([judge[c] for c in ids], [truth[c] for c in ids])
    undecided = sum(1 for r in run.results if r.scores["judge"].passed is None)
    tokens = statistics.mean(r.scores["judge"].judge_tokens for r in run.results)
    kappa = f"{cm.kappa:.3f}" if cm.kappa is not None else "—"
    return (
        f"{100 * sum(judge.values()) / len(judge):.1f}% | {cm.tp} / {cm.fp} / {cm.fn} / {cm.tn} | "
        f"{100 * (cm.tpr or 0):.1f}% / {100 * (cm.fpr or 0):.1f}% | {kappa} | {undecided} | "
        f"{tokens:.0f}"
    )


def reasoning_judge(cfg) -> list[str]:
    """Section 8: the published judge against the same judge with thinking on (issue #3)."""
    runs = HERE / "runs"
    off = {
        "baseline": load_run(runs / "baseline.json"),
        "candidate": load_run(runs / "candidate.json"),
    }
    on = {
        "baseline": load_run(runs / "baseline.reasoning-judge.json"),
        "candidate": load_run(runs / "candidate.reasoning-judge.json"),
    }
    rerun = load_run(runs / "baseline.think-off-rerun.json")
    head = (
        "| Judge | Pass rate | TP / FP / FN / TN | TPR / FPR | κ | Undecided "
        "| Judge tokens per case |"
    )
    out = ["## Reasoning judge (issue #3)", ""]
    for truth_name in ("answer key", "audited labels"):
        out += [f"### Against the {truth_name}", ""]
        for label in ("baseline", "candidate"):
            truth = (
                verdicts(off[label], "answer_key")
                if truth_name == "answer key"
                else audited(off[label], label)
            )
            out += [f"#### {label}", "", head, "|---|---:|---:|---:|---:|---:|---:|"]
            out.append(f"| thinking off | {judge_row(off[label], truth)} |")
            out.append(f"| thinking on | {judge_row(on[label], truth)} |")
            out.append("")
    out += [
        "### Every contested case",
        "",
        "| Case | Run | Key | Off | On | Audited |",
        "|---|---|---|---|---|---|",
    ]
    fmt = {True: "pass", False: "fail", None: "undecided"}
    right = {"key": 0, "off": 0, "on": 0}
    decided = unaudited = 0
    for label in ("baseline", "candidate"):
        key = verdicts(off[label], "answer_key")
        j_off, j_on = verdicts(off[label], "judge"), verdicts(on[label], "judge")
        for c in sorted(key):
            seen = {key[c], j_off.get(c), j_on.get(c)}
            if len(seen) == 1:
                continue
            if (c, label) not in AUDITED:
                unaudited += 1
                truth_cell = "**not yet read**"
            elif AUDITED[(c, label)] is None:
                truth_cell = "ambiguous"
            else:
                t = AUDITED[(c, label)]
                truth_cell = fmt[t]
                decided += 1
                right["key"] += key[c] == t
                right["off"] += j_off.get(c) == t
                right["on"] += j_on.get(c) == t
            out.append(
                f"| {c} | {label} | {fmt[key[c]]} | {fmt[j_off.get(c)]} | {fmt[j_on.get(c)]} "
                f"| {truth_cell} |"
            )
    out += [
        "",
        f"Right on the {decided} decided contested cases: answer key {right['key']}, "
        f"judge thinking off {right['off']}, thinking on {right['on']}."
        + (f" {unaudited} contested cases are not yet read." if unaudited else ""),
        "",
    ]
    flips = {
        label: sum(
            1
            for c, v in verdicts(off[label], "judge").items()
            if verdicts(on[label], "judge").get(c) not in (None, v)
        )
        for label in off
    }
    same = sum(
        1
        for c, v in verdicts(off["baseline"], "judge").items()
        if verdicts(rerun, "judge").get(c) == v
    )
    out += [
        f"- Verdicts that changed with thinking: baseline {flips['baseline']}, "
        f"candidate {flips['candidate']}.",
        f"- Judge wall time on the 83 baseline answers, system replies cached: thinking off "
        f"{rerun.info.wall_time_s:.0f} s, thinking on {on['baseline'].info.wall_time_s:.0f} s; "
        f"candidate, thinking on {on['candidate'].info.wall_time_s:.0f} s.",
        f"- Determinism: the thinking-off re-run matched the published judge on {same} of 83 "
        "verdicts.",
        "",
        "### The gate with each judge",
        "",
    ]
    assert cfg.gate is not None
    for name, pair in (("thinking off", off), ("thinking on", on)):
        cmp = compare_runs(pair["baseline"], pair["candidate"])
        out += [
            f"#### {name}",
            "",
            comparison_markdown(cmp, decide(cmp, cfg.gate), cfg.gate.metric),
        ]
    return out


def main() -> None:
    cfg = load_config(HERE / "suite.yaml")
    base = load_run(HERE / "runs" / "baseline.json")
    cand = load_run(HERE / "runs" / "candidate.json")
    lines = ["## Calibration against the full answer key", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        rep = calibrate(run, "judge", gold_scorer="answer_key", seed=0)
        lines += [f"### {label}", "", calibration_markdown(rep)]
    lines += ["## Calibration with a 30-case gold sample (seed 0)", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        rep = calibrate(run, "judge", gold_scorer="answer_key", gold_sample=GOLD_N, seed=0)
        lines += [f"### {label}", "", calibration_markdown(rep)]
    lines += ["## Does the correction hold across samples?", ""]
    for label, run in (("baseline", base), ("candidate", cand)):
        lines += [*sample_study(run, label), ""]
    lines += ["## Baseline vs candidate", ""]
    cmp = compare_runs(base, cand)
    assert cfg.gate is not None
    lines += [comparison_markdown(cmp, decide(cmp, cfg.gate), cfg.gate.metric)]
    lines += ["## Latency", "", f"- baseline: {latency(base)}", f"- candidate: {latency(cand)}", ""]
    if (HERE / "runs" / "candidate.reasoning-judge.json").exists():
        lines += reasoning_judge(cfg)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
