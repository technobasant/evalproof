"""Run a suite: call the system on every case, score every output, write one run artifact."""

from __future__ import annotations

import platform
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache, case_scope
from evalproof.config import (
    MAX_TRIALS,
    ConfigError,
    GenerationParams,
    SuiteConfig,
    TargetConfig,
    suite_fingerprint,
)
from evalproof.dataset import case_hash, content_hash, dataset_hash, load_dataset
from evalproof.models import RUN_FORMAT_VERSION, Case, CaseResult, Run, RunInfo, Score
from evalproof.providers import (
    LLM_PROVIDERS,
    ChatResponse,
    Message,
    Provider,
    make_provider,
    parse_target,
)
from evalproof.providers.agent_cli import cli_facts
from evalproof.providers.base import close_resource
from evalproof.scorers import Scorer, make_scorer, score_metrics
from evalproof.systems import (
    LLMSystem,
    System,
    check_turns_supported,
    check_variant_vars,
    make_system,
)

Progress = Callable[[int, int, CaseResult], None]


def _ollama_facts(cfg: TargetConfig, client: httpx.Client | None) -> dict[str, str]:
    """Best effort: the server version and the model digest make a run reproducible."""
    if cfg.kind != "ollama":
        return {}
    _, model = parse_target(cfg.target)
    base = (cfg.base_url or "http://127.0.0.1:11434").rstrip("/")
    if client is not None:
        return _lookup_ollama(client, base, model)
    with httpx.Client(timeout=5) as http:  # one-shot: closed here, never left open
        return _lookup_ollama(http, base, model)


def _lookup_ollama(http: httpx.Client, base: str, model: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    try:
        facts["ollama_version"] = http.get(f"{base}/api/version").json().get("version", "")
        for m in http.get(f"{base}/api/tags").json().get("models", []):
            if m.get("name") == model or m.get("model") == model:
                facts[f"digest:{model}"] = str(m.get("digest", ""))[:12]
    except (httpx.HTTPError, ValueError):
        pass
    return facts


class ResolvedModels:
    """A thin wrapper that remembers which model ids a provider reported, cached replies
    included, so the run artifact can say which model actually answered."""

    def __init__(self, inner: Provider) -> None:
        self.inner = inner
        self.name = inner.name
        self.model = inner.model
        self.seen: set[str] = set()
        self._lock = threading.Lock()

    def close(self) -> None:
        close_resource(self.inner)

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        resp = self.inner.chat(messages, params)
        if resp.model:
            with self._lock:
                self.seen.add(resp.model)
        return resp


def _target_facts(role: str, cfg: TargetConfig) -> dict[str, str]:
    """What identifies a judge or system target: its target string and, for a coding-agent CLI,
    the CLI, its version and the isolation scheme (a version lookup, never a model call)."""
    facts = {f"{role}_target": cfg.target}
    facts.update({f"{role}_{key}": value for key, value in cli_facts(cfg).items()})
    return facts


def environment(
    cfg: SuiteConfig,
    client: httpx.Client | None = None,
    *,
    system: TargetConfig | None = None,
    resolved: dict[str, set[str]] | None = None,
    cache_only: bool = False,
) -> dict[str, str]:
    """The facts a run needs to be reproducible and comparable. ``system`` is the target that
    actually ran when it is not ``cfg.system``; ``resolved`` maps a role (``judge``,
    ``system``) to the model ids its provider reported. A ``cache_only`` run does not ask an
    Ollama server for its version and digests: it must not reach a model's server at all (a
    coding-agent CLI's ``--version`` is still read, which runs no model). It records
    ``cache_only: "true"``, so a run built from the cache can be told from one a model made."""
    system_cfg = system or cfg.system
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
    }
    if cache_only:
        env["cache_only"] = "true"  # the replies are the cache's: no model ran for this run
    else:
        env.update(_ollama_facts(system_cfg, client))
    if system_cfg.kind in LLM_PROVIDERS:
        env.update(_target_facts("system", system_cfg))
    if cfg.judge:
        if not cache_only:
            env.update(_ollama_facts(cfg.judge, client))
        env.update(_target_facts("judge", cfg.judge))
    for role, models in (resolved or {}).items():
        if models:
            env[f"{role}_model_resolved"] = ",".join(sorted(models))
    return env


def score_case(case: Case, system: System, scorers: list[Scorer], trial: int = 0) -> CaseResult:
    scores: dict[str, Score] = {}
    with case_scope(case.id, trial):  # names the case if a cache-only call misses
        out = system.run(case, trial=trial)
        for scorer in scorers:
            scores.update(score_metrics(scorer, case, out))
    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        output=out,
        scores=scores,
        case_hash=case_hash(case),
        content_hash=content_hash(case),
        trial=trial,
    )


def run_suite(
    cfg: SuiteConfig,
    *,
    limit: int | None = None,
    use_cache: bool = True,
    cache_only: bool = False,
    client: httpx.Client | None = None,
    progress: Progress | None = None,
    system_override: TargetConfig | None = None,
    variant: str | None = None,
    trials: int | None = None,
    on_plan: Callable[[int, int], None] | None = None,
) -> Run:
    """Run the suite and score it. ``variant`` names one of the suite's ``variants``: its partial
    ``system`` override is merged over the suite's system (unless ``system_override`` replaces the
    target outright), and its ``vars`` reach the system.

    Each case runs ``trials`` times (default: the suite's ``trials``), every run scored on its own;
    the results come back in case order, then trial order, however the runs were scheduled.
    ``on_plan(cases, trials)`` is called once before the first run.

    With ``cache_only`` every model call, the judge's included, is answered from the response
    cache; the first one that is not there raises ``CacheMiss`` and no run is returned. A python
    or http system is refused, because it does not go through the cache."""
    n_trials = cfg.trials if trials is None else trials
    if not 1 <= n_trials <= MAX_TRIALS:
        raise ConfigError(f"trials must be between 1 and {MAX_TRIALS}, got {n_trials}")
    if cache_only and not use_cache:
        raise ConfigError("cache_only needs the cache: it cannot be combined with use_cache=False")
    variant_vars = cfg.variant(variant).vars if variant is not None else {}
    system_cfg = system_override or cfg.variant_system(variant)
    if cache_only and system_cfg.kind in ("python", "http"):
        raise ConfigError(
            f"--cache-only needs a model system: a {system_cfg.kind} system is not cached, "
            "so its code would run"
        )
    cases = load_dataset(cfg.resolve(cfg.dataset))
    check_turns_supported(system_cfg, cases)
    if variant is not None:
        check_variant_vars(system_cfg, variant, variant_vars, cases)
    if limit is not None:
        cases = cases[:limit]
    cache = ResponseCache(cfg.resolve(cfg.cache_dir)) if use_cache else None
    with ExitStack() as owned:  # closes the HTTP clients that make_* opened, not a passed one
        system = make_system(
            system_cfg,
            cache=cache,
            client=client,
            search_path=str(Path(cfg.base_dir).resolve()),
            variant=variant,
            vars=variant_vars,
            trials=n_trials,
            cache_only=cache_only,
        )
        owned.callback(close_resource, system)

        judge: Provider | None = None
        judge_models: ResolvedModels | None = None
        if cfg.judge is not None:
            judge = make_provider(cfg.judge, client=client)
            owned.callback(close_resource, judge)
            if cache is not None:
                judge = CachedProvider(
                    judge, cache, base_url=cfg.judge.base_url, cache_only=cache_only, role="judge"
                )
            judge = judge_models = ResolvedModels(judge)
        system_models: ResolvedModels | None = None
        if isinstance(system, LLMSystem):
            system_models = ResolvedModels(system.provider)
            system.provider = system_models
        scorers = [make_scorer(s, cfg, judge) for s in cfg.scorers]

        if on_plan:
            on_plan(len(cases), n_trials)
        jobs = [(case, trial) for case in cases for trial in range(n_trials)]
        started = datetime.now(UTC)
        t0 = time.perf_counter()
        results: list[CaseResult] = []
        if cfg.concurrency == 1:
            for i, (case, trial) in enumerate(jobs, start=1):
                res = score_case(case, system, scorers, trial)
                results.append(res)
                if progress:
                    progress(i, len(jobs), res)
        else:
            with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
                futures = [pool.submit(score_case, c, system, scorers, t) for c, t in jobs]
                try:
                    for i, fut in enumerate(futures, start=1):
                        res = fut.result()
                        results.append(res)
                        if progress:
                            progress(i, len(jobs), res)
                except BaseException:
                    # A case that raised (a cache-only miss, Ctrl-C) ends the run: drop the
                    # cases not started yet rather than run them for a result that is discarded.
                    pool.shutdown(wait=True, cancel_futures=True)
                    raise
        wall = time.perf_counter() - t0

        info = RunInfo(
            suite=cfg.name,
            system=system.name,
            dataset_path=str(cfg.dataset),
            dataset_hash=dataset_hash(cases),
            started_at=started.isoformat(timespec="seconds"),
            finished_at=datetime.now(UTC).isoformat(timespec="seconds"),
            wall_time_s=round(wall, 3),
            evalproof_version=__version__,
            seed=cfg.seed,
            suite_hash=suite_fingerprint(cfg, system_cfg, variant),
            variant=variant,
            trials=n_trials,
            environment=environment(
                cfg,
                client,
                system=system_cfg,
                cache_only=cache_only,
                resolved={
                    role: rec.seen
                    for role, rec in (("judge", judge_models), ("system", system_models))
                    if rec is not None
                },
            ),
        )
        return Run(info=info, results=results)


def save_run(run: Run, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # LF on every platform: a run file committed from Windows has the bytes one from Linux has.
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8", newline="\n")
    return path


class RunFileError(ValueError):
    """A run file this evalproof cannot read: its format version is unknown to it."""


class _RunHeader(BaseModel):
    """Just the field ``load_run`` checks before it validates the rest of the file."""

    format_version: object = 1  # a file from before 1.0 has none, and is format 1


def _check_format_version(text: str, path: Path) -> None:
    try:
        version = _RunHeader.model_validate_json(text).format_version
    except ValidationError:
        return  # not a JSON object: validating it as a Run says what is wrong with it
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise RunFileError(
            f"{path}: `format_version` is {version!r}; it must be a whole number from 1 "
            f"(this evalproof reads run files up to format version {RUN_FORMAT_VERSION})"
        )
    if version > RUN_FORMAT_VERSION:
        raise RunFileError(
            f"{path}: this run file is format version {version}, written by a newer evalproof "
            f"than {__version__} (which reads up to format version {RUN_FORMAT_VERSION}): "
            "upgrade evalproof to read it"
        )


def load_run(path: str | Path) -> Run:
    """Read a run file. A file written by a newer evalproof (a higher ``format_version``) raises
    ``RunFileError`` rather than loading as a run it may misread."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    _check_format_version(text, path)
    return Run.model_validate_json(text)
