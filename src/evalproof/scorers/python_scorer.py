"""The ``python`` scorer: a function of your own scores a case, and the helper that finds it."""

from __future__ import annotations

import importlib
import math
import re
import sys
from collections.abc import Callable
from typing import Any

from evalproof.config import ConfigError
from evalproof.models import Case, Score, SystemOutput

FN_SPEC = re.compile(r"[A-Za-z_][\w.]*:[A-Za-z_]\w*")


def resolve_fn(spec: str, search_path: str | None, what: str) -> Callable[..., Any]:
    """Import ``module:function`` and return the function. ``search_path`` (the suite's
    directory) is put on ``sys.path`` first, as ``PythonSystem`` does, so a suite can ship its
    scorers next to it. ``what`` names the caller in the error. Raises ``ConfigError``."""
    if not FN_SPEC.fullmatch(spec):
        raise ConfigError(f"{what}: fn must be 'module:function', got {spec!r}")
    if search_path and search_path not in sys.path:
        sys.path.insert(0, search_path)
    module_name, func_name = spec.split(":", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:  # a missing module, or one that fails while importing
        raise ConfigError(
            f"{what}: cannot import {module_name!r}: {type(exc).__name__}: {exc}"
        ) from exc
    func = getattr(module, func_name, None)
    if not callable(func):
        raise ConfigError(f"{what}: {module_name}.{func_name} is not callable")
    return func  # type: ignore[no-any-return]


class _BadReturn(Exception):
    """The scorer function returned something that is not a score."""


def _coerce(result: Any, threshold: float) -> Score:
    if isinstance(result, Score):
        return result
    if isinstance(result, bool):
        return Score(value=1.0 if result else 0.0, passed=result)
    if isinstance(result, int | float):
        value = float(result)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise _BadReturn(f"returned {result!r}; a number score must be between 0 and 1")
        return Score(value=value, passed=value >= threshold)
    if isinstance(result, tuple) and len(result) == 2:
        ok, reason = result
        if isinstance(ok, bool) and (reason is None or isinstance(reason, str)):
            return Score(value=1.0 if ok else 0.0, passed=ok, reason=reason)
    raise _BadReturn(
        f"returned {type(result).__name__}; expected a Score, a bool, a number from 0 to 1, "
        "or a (bool, reason) pair"
    )


class PythonScorer:
    """Scores a case with ``fn(case, output)``.

    ``fn`` returns a ``Score``, a bool, a float in [0, 1] (``passed`` when it reaches
    ``threshold``) or a ``(bool, reason)`` pair. An exception, or any other return value, leaves
    the case undecided with the reason ``scorer error: ...``: counted and reported, never a
    silent pass or fail. ``deterministic: true`` promises that the same case and output always
    score the same, which lets ``evalproof verify`` re-score the stored outputs with it.
    """

    def __init__(
        self,
        name: str,
        fn: str,
        deterministic: bool = False,
        threshold: float = 0.5,
        *,
        search_path: str | None = None,
    ) -> None:
        self.name = name
        self.deterministic = deterministic
        self.threshold = threshold
        self.fn = resolve_fn(fn, search_path, f"scorer {name!r}")

    def score(self, case: Case, out: SystemOutput) -> Score:
        try:
            return _coerce(self.fn(case, out), self.threshold)
        except _BadReturn as exc:
            return Score(value=None, passed=None, reason=f"scorer error: {exc}")
        except Exception as exc:
            return Score(
                value=None, passed=None, reason=f"scorer error: {type(exc).__name__}: {exc}"
            )


def validate_python_params(params: dict[str, Any], path: str = "params") -> None:
    """Check the ``python`` scorer's params without importing anything. Raises ``ValueError``."""
    known = ("fn", "deterministic", "threshold")
    for key in params:
        if key not in known:
            raise ValueError(f"{path} has unknown key {key!r} (valid keys: {', '.join(known)})")
    fn = params.get("fn")
    if not isinstance(fn, str) or not FN_SPEC.fullmatch(fn):
        raise ValueError(f"{path}.fn must be 'module:function', got {fn!r}")
    if not isinstance(params.get("deterministic", False), bool):
        raise ValueError(f"{path}.deterministic must be true or false")
    threshold = params.get("threshold", 0.5)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError(f"{path}.threshold must be a number between 0 and 1")
