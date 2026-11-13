"""Run a suite: call the system on every case, score every output, write one run artifact."""

from __future__ import annotations

import platform
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import GenerationParams, SuiteConfig, TargetConfig, suite_fingerprint
from evalproof.dataset import case_hash, content_hash, dataset_hash, load_dataset
from evalproof.models import Case, CaseResult, Run, RunInfo
from evalproof.providers import (
    LLM_PROVIDERS,
    ChatResponse,
    Message,
    Provider,
    make_provider,
    parse_target,
)
from evalproof.providers.agent_cli import cli_facts
from evalproof.scorers import Scorer, make_scorer
from evalproof.systems import LLMSystem, System, make_system

Progress = Callable[[int, int, CaseResult], None]


def _ollama_facts(cfg: TargetConfig, client: httpx.Client | None) -> dict[str, str]:
    """Best effort: the server version and the model digest make a run reproducible."""
    if cfg.kind != "ollama":
        return {}
    _, model = parse_target(cfg.target)
    base = (cfg.base_url or "http://127.0.0.1:11434").rstrip("/")
    http = client or httpx.Client(timeout=5)
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
) -> dict[str, str]:
    """The facts a run needs to be reproducible and comparable. ``system`` is the target that
    actually ran when it is not ``cfg.system``; ``resolved`` maps a role (``judge``,
    ``system``) to the model ids its provider reported."""
    system_cfg = system or cfg.system
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
    }
    env.update(_ollama_facts(system_cfg, client))
    if system_cfg.kind in LLM_PROVIDERS:
        env.update(_target_facts("system", system_cfg))
    if cfg.judge:
        env.update(_ollama_facts(cfg.judge, client))
        env.update(_target_facts("judge", cfg.judge))
    for role, models in (resolved or {}).items():
        if models:
            env[f"{role}_model_resolved"] = ",".join(sorted(models))
    return env


def score_case(case: Case, system: System, scorers: list[Scorer]) -> CaseResult:
    out = system.run(case)
    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        output=out,
        scores={s.name: s.score(case, out) for s in scorers},
        case_hash=case_hash(case),
        content_hash=content_hash(case),
    )


def run_suite(
    cfg: SuiteConfig,
    *,
    limit: int | None = None,
    use_cache: bool = True,
    client: httpx.Client | None = None,
    progress: Progress | None = None,
    system_override: TargetConfig | None = None,
) -> Run:
    cases = load_dataset(cfg.resolve(cfg.dataset))
    if limit is not None:
        cases = cases[:limit]
    cache = ResponseCache(cfg.resolve(cfg.cache_dir)) if use_cache else None
    system_cfg = system_override or cfg.system
    system = make_system(
        system_cfg, cache=cache, client=client, search_path=str(Path(cfg.base_dir).resolve())
    )

    judge: Provider | None = None
    judge_models: ResolvedModels | None = None
    if cfg.judge is not None:
        judge = make_provider(cfg.judge, client=client)
        if cache is not None:
            judge = CachedProvider(judge, cache, base_url=cfg.judge.base_url)
        judge = judge_models = ResolvedModels(judge)
    system_models: ResolvedModels | None = None
    if isinstance(system, LLMSystem):
        system_models = ResolvedModels(system.provider)
        system.provider = system_models
    scorers = [make_scorer(s, cfg, judge) for s in cfg.scorers]

    started = datetime.now(UTC)
    t0 = time.perf_counter()
    results: list[CaseResult] = []
    if cfg.concurrency == 1:
        for i, case in enumerate(cases, start=1):
            res = score_case(case, system, scorers)
            results.append(res)
            if progress:
                progress(i, len(cases), res)
    else:
        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            futures = [pool.submit(score_case, c, system, scorers) for c in cases]
            for i, fut in enumerate(futures, start=1):
                res = fut.result()
                results.append(res)
                if progress:
                    progress(i, len(cases), res)
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
        suite_hash=suite_fingerprint(cfg, system_cfg),
        environment=environment(
            cfg,
            client,
            system=system_cfg,
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
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> Run:
    return Run.model_validate_json(Path(path).read_text(encoding="utf-8"))
