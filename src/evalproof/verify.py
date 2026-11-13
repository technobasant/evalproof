"""``evalproof verify``: does a committed run file match the committed suite and dataset?

A keyless CI job compares a run file the pull request carries. Nothing it runs produced that
file, so this checks what can be checked without a model call: the run was made with the
suite's system, judge, scorers and seed (``suite_hash``), and it scored exactly the dataset's
cases as they are now (``case_hash`` per case). It cannot prove the scores were not edited by
hand; that needs the run to be made in CI (``--ci api-key``) or the file to be reviewed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from evalproof.config import SuiteConfig, suite_fingerprint
from evalproof.dataset import case_hash, load_dataset
from evalproof.models import Run

SHOW = 10  # ids listed per problem


@dataclass
class Verification:
    ok: bool
    cases: int  # cases in the suite's dataset
    problems: list[str] = field(default_factory=list)


def _ids(ids: list[str]) -> str:
    shown = ", ".join(f"`{i}`" for i in ids[:SHOW])
    return shown + (f" and {len(ids) - SHOW} more" if len(ids) > SHOW else "")


def verify_run(run: Run, cfg: SuiteConfig) -> Verification:
    """Check ``run`` against ``cfg`` as loaded now (its ``system``, not an override)."""
    problems: list[str] = []
    expected = suite_fingerprint(cfg)
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

    cases = {c.id: case_hash(c) for c in load_dataset(cfg.resolve(cfg.dataset))}
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
            f"{len(changed)} case(s) changed since the run was made (or have no case hash): "
            f"{_ids(changed)}"
        )
    return Verification(ok=not problems, cases=len(cases), problems=problems)


def verify_markdown(result: Verification, run_path: str, suite_path: str) -> str:
    lines = ["## evalproof verify", ""]
    if result.ok:
        lines.append(
            f"✅ `{run_path}` matches `{suite_path}`: the same system, judge, scorers and seed, "
            f"and all {result.cases} cases as they are now."
        )
    else:
        lines.append(f"❌ `{run_path}` does not match `{suite_path}`.")
        lines.append("")
        lines += [f"- {p}" for p in result.problems]
        lines += ["", f"Run `evalproof run {suite_path} --out {run_path}` and commit the result."]
    return "\n".join(lines) + "\n"
