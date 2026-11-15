"""Find what a repository's LLM code does and what no suite tests yet, without running it."""

from evalproof.scan.core import DRAFTS_OUT, scan_path
from evalproof.scan.model import (
    FORMAT_VERSION,
    GAP_KINDS,
    MODEL_GAP_KINDS,
    Finding,
    Gap,
    ScanReport,
    SuiteInfo,
)
from evalproof.scan.report import render_text, report_json

__all__ = [
    "DRAFTS_OUT",
    "FORMAT_VERSION",
    "GAP_KINDS",
    "MODEL_GAP_KINDS",
    "Finding",
    "Gap",
    "ScanReport",
    "SuiteInfo",
    "render_text",
    "report_json",
    "scan_path",
]
