"""``evalproof verify``: does a committed run file match the committed suite and dataset?

A keyless CI job compares a run file the pull request carries. Nothing it runs produced that
file, so this checks what can be checked without a model call: the run was made with the
suite's system, judge, scorers and seed (``suite_hash``), and it scored exactly the dataset's
cases as they are now (``case_hash`` per case, tags included, because ``must_pass`` reads them
from the run file). The deterministic scores are checked too: each stored output is scored
again with the suite's scorers that need no model, and a score that differs was edited by hand
or made by another scorer version. Model-judged scores and the stored outputs themselves still
cannot be checked; that needs the run to be made in CI (``--ci api-key``) or the file to be
reviewed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from evalproof.config import SuiteConfig, suite_fingerprint
from evalproof.dataset import case_hash, load_dataset
from evalproof.models import Case, Run, Score
from evalproof.scorers import Scorer, is_deterministic, make_scorer

SHOW = 10  # ids listed per problem


@dataclass
class Verification:
    ok: bool
    cases: int  # cases in the suite's dataset
    problems: list[str] = field(default_factory=list)
    rescored: list[str] = field(default_factory=list)  # scorers whose stored scores were re-checked
    not_rescored: list[str] = field(default_factory=list)  # scorers that need a model call


def _listed(items: list[str]) -> str:
    shown = ", ".join(items[:SHOW])
    return shown + (f" and {len(items) - SHOW} more" if len(items) > SHOW else "")


def _ids(ids: list[str]) -> str:
    return _listed([f"`{i}`" for i in ids])


def _same_score(stored: Score | None, fresh: Score) -> bool:
    """Equal outcome and value (to float noise); the reason is prose and is not compared."""
    if stored is None or stored.passed != fresh.passed:
        return False
    if stored.value is None or fresh.value is None:
        return stored.value is None and fresh.value is None
    return math.isclose(stored.value, fresh.value, rel_tol=1e-9, abs_tol=1e-12)


def _rescore(
    run: Run, cfg: SuiteConfig, cases: dict[str, Case], hashes: dict[str, str]
) -> tuple[list[str], list[str]]:
    """Score the stored outputs again with the suite's scorers that need no model. Returns the
    names of the scorers used and the problems found. Cases that are missing, extra or changed
    are skipped here: they are reported by the case checks."""
    problems: list[str] = []
    scorers: list[Scorer] = []
    for sc in cfg.scorers:
        if not is_deterministic(sc):
            continue
        try:
            scorers.append(make_scorer(sc, cfg))
        except Exception as exc:
            problems.append(
                f"scorer `{sc.name}` could not be built to re-score the stored outputs: {exc}"
            )
    differing: list[str] = []
    for r in run.results:
        case = cases.get(r.case_id)
        if case is None or r.case_hash != hashes[r.case_id]:
            continue
        for scorer in scorers:
            if not _same_score(r.scores.get(scorer.name), scorer.score(case, r.output)):
                differing.append(f"`{r.case_id}` on `{scorer.name}`")
    if differing:
        problems.append(
            f"{len(differing)} score(s) differ from re-scoring the stored outputs with the "
            f"suite's deterministic scorers (edited by hand, or made by a different scorer "
            f"version): {_listed(differing)}"
        )
    return [sc.name for sc in scorers], problems


def verify_run(run: Run, cfg: SuiteConfig) -> Verification:
    """Check ``run`` against ``cfg`` as loaded now (its ``system``, not an override). A run made
    with a variant is checked against that variant of the suite."""
    problems: list[str] = []
    variant = run.info.variant
    if variant is not None and all(v.name != variant for v in cfg.variants):
        return Verification(
            ok=False,
            cases=len(load_dataset(cfg.resolve(cfg.dataset))),
            problems=[
                f"the run was made with variant `{variant}`, which the suite does not define"
            ],
        )
    expected = suite_fingerprint(cfg, variant=variant)
    if run.info.suite_hash is None:
        problems.append(
            "the run has no suite hash (it was written before evalproof 0.5, or imported): "
            "re-run with evalproof 0.5 or newer"
        )
    elif run.info.suite_hash != expected:
        problems.append(
            f"the suite's system, judge, scorers or seed differ from the ones the run used "
            f"(run `{run.info.suite_hash}`, suite `{expected}`): run the suite again"
        )

    dataset = {c.id: c for c in load_dataset(cfg.resolve(cfg.dataset))}
    cases = {cid: case_hash(c) for cid, c in dataset.items()}
    if len(run.results) != len({r.case_id for r in run.results}):
        problems.append("the run has duplicate case ids")
    scored = {r.case_id: r.case_hash for r in run.results}
    missing = sorted(set(cases) - set(scored))
    extra = sorted(set(scored) - set(cases))
    changed = sorted(cid for cid in set(cases) & set(scored) if scored[cid] != cases[cid])
    if missing:
        problems.append(
            f"{len(missing)} case(s) in the dataset are missing from the run (a partial run, "
            f"such as one made with --limit?): {_ids(missing)}"
        )
    if extra:
        problems.append(
            f"{len(extra)} case(s) in the run are not in the dataset any more: {_ids(extra)}"
        )
    if changed:
        problems.append(
            f"{len(changed)} case(s) changed since the run was made (content or tags), or have "
            f"no case hash: {_ids(changed)}"
        )
    rescored, rescore_problems = _rescore(run, cfg, dataset, cases)
    problems += rescore_problems
    return Verification(
        ok=not problems,
        cases=len(cases),
        problems=problems,
        rescored=rescored,
        not_rescored=[sc.name for sc in cfg.scorers if not is_deterministic(sc)],
    )


def verify_markdown(result: Verification, run_path: str, suite_path: str) -> str:
    lines = ["## evalproof verify", ""]
    if result.ok:
        verdict = (
            f"✅ `{run_path}` matches `{suite_path}`: the same system, judge, scorers and seed, "
            f"and all {result.cases} cases as they are now"
        )
        if result.rescored:
            verdict += f"; re-scored the stored outputs with {', '.join(result.rescored)}."
        else:
            verdict += (
                ". No scorer could be re-scored without a model call, so the scores themselves "
                "are not checked."
            )
        if result.rescored and result.not_rescored:
            verdict += f" Not checked (they need a model call): {', '.join(result.not_rescored)}."
        lines.append(verdict)
    else:
        lines.append(f"❌ `{run_path}` does not match `{suite_path}`.")
        lines.append("")
        lines += [f"- {p}" for p in result.problems]
        lines += ["", f"Run `evalproof run {suite_path} --out {run_path}` and commit the result."]
    return "\n".join(lines) + "\n"
