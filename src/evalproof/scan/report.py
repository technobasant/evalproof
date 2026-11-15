"""The scan as text for a person, and as JSON for ``evalproof draft`` and for scripts."""

from __future__ import annotations

import json

from evalproof.scan.model import FINDING_KINDS, Finding, ScanReport

MAX_PER_KIND = 20


def report_json(report: ScanReport) -> str:
    return json.dumps(report.to_json(), indent=2, ensure_ascii=False) + "\n"


def _line(f: Finding) -> str:
    where = f"{f.path}:{f.line}" if f.line > 1 or f.kind not in ("prompt", "test_cases") else f.path
    framework = f"  [{f.framework}]" if f.framework else ""
    if f.language == "typescript":
        framework += " (typescript)"
    return f"  {where}{framework}  {f.detail}"


def render_text(report: ScanReport, limit: int = MAX_PER_KIND) -> str:
    """Findings grouped by kind (at most ``limit`` lines for each, then "and N more"), the suites,
    the gaps with the next command for each, and what was skipped."""
    out = [f"evalproof scan: {report.root} ({report.files} files read)", ""]
    if not report.findings:
        out += ["Nothing found: no model call, agent, tool, prompt, test case or suite.", ""]
    for kind in FINDING_KINDS:
        found = report.of_kind(kind)
        if not found:
            continue
        out.append(f"{kind} ({len(found)})")
        out.extend(_line(f) for f in found[:limit])
        if len(found) > limit:
            out.append(f"  ... and {len(found) - limit} more")
        out.append("")
    if report.gaps or report.not_computed:
        out.append(f"gaps ({len(report.gaps)})")
        for gap in report.gaps[:limit]:
            where = (
                f" {gap.path}:{gap.line}"
                if gap.path and (gap.line or 0) > 1
                else (f" {gap.path}" if gap.path else "")
            )
            out.append(f"  {gap.kind}{where}: {gap.message}")
            out.append(f"    next: {gap.suggest}")
        if len(report.gaps) > limit:
            out.append(f"  ... and {len(report.gaps) - limit} more")
        for kind, reason in report.not_computed.items():
            out.append(f"  {kind}: not computed: {reason}")
        out.append("")
    elif report.findings:
        out += ["gaps (0)", "  none found", ""]
    for note in report.notes:
        out.append(f"note: {note}")
    skipped = []
    if report.unparsable:
        skipped.append(f"{report.unparsable} file(s) that did not parse")
    if report.too_large:
        skipped.append(f"{report.too_large} file(s) over 1 MB")
    if skipped:
        out.append("skipped: " + ", ".join(skipped))
    return "\n".join(out).rstrip("\n") + "\n"
