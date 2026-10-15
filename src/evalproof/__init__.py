"""evalproof: treat LLM evaluation as measurement.

An LLM judge is a classifier with error rates. evalproof calibrates it against a
small gold-labelled sample, reports prevalence-corrected pass rates with
confidence intervals, compares runs on paired cases, and gates CI on
statistically meaningful regressions rather than on noise.
"""

__version__ = "0.1.2"
