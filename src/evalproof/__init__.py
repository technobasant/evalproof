"""evalproof: treat LLM evaluation as measurement.

An LLM judge is a classifier with error rates. evalproof calibrates it against a
small gold-labelled sample, reports prevalence-corrected pass rates with
confidence intervals, compares runs on paired cases, and gates CI on
statistically meaningful regressions rather than on noise.

The names below are the supported Python API; everything else is internal and may
change between minor versions.
"""

from evalproof._version import __version__
from evalproof.calibration import CalibrationReport, Confusion, Estimate, calibrate
from evalproof.compare import Comparison, MetricComparison, RunStats, compare_runs
from evalproof.config import ConfigError, GateConfig, SuiteConfig, load_config
from evalproof.dataset import DatasetError, load_dataset
from evalproof.gate import (
    EXIT_BUDGET,
    EXIT_CANNOT_GATE,
    EXIT_PASS,
    EXIT_REGRESSION,
    EXIT_USAGE,
    GateDecision,
    decide,
)
from evalproof.models import Case, CaseResult, Run, RunInfo, Score, SystemOutput
from evalproof.report import calibration_markdown, comparison_markdown, run_markdown
from evalproof.runner import load_run, run_suite, save_run
from evalproof.stats import mcnemar_exact, paired_bootstrap, wilson

__all__ = [
    "EXIT_BUDGET",
    "EXIT_CANNOT_GATE",
    "EXIT_PASS",
    "EXIT_REGRESSION",
    "EXIT_USAGE",
    "CalibrationReport",
    "Case",
    "CaseResult",
    "Comparison",
    "ConfigError",
    "Confusion",
    "DatasetError",
    "Estimate",
    "GateConfig",
    "GateDecision",
    "MetricComparison",
    "Run",
    "RunInfo",
    "RunStats",
    "Score",
    "SuiteConfig",
    "SystemOutput",
    "__version__",
    "calibrate",
    "calibration_markdown",
    "compare_runs",
    "comparison_markdown",
    "decide",
    "load_config",
    "load_dataset",
    "load_run",
    "mcnemar_exact",
    "paired_bootstrap",
    "run_markdown",
    "run_suite",
    "save_run",
    "wilson",
]
