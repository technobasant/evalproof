"""Run a suite: call the system on every case, score every output, write one run artifact."""

from __future__ import annotations

import platform
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evalproof._version import __version__
from evalproof.cache import CachedProvider, ResponseCache
from evalproof.config import SuiteConfig, TargetConfig
from evalproof.dataset import dataset_hash, load_dataset
from evalproof.models import Case, CaseResult, Run, RunInfo
from evalproof.providers import Provider, make_provider, parse_target
from evalproof.scorers import Scorer, make_scorer
from evalproof.systems import System, make_system

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


def environment(cfg: SuiteConfig, client: httpx.Client | None = None) -> dict[str, str]:
    env = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "",
    }
    env.update(_ollama_facts(cfg.system, client))
    if cfg.judge:
        env.update(_ollama_facts(cfg.judge, client))
    return env


def score_case(case: Case, system: System, scorers: list[Scorer]) -> CaseResult:
    out = system.run(case)
    return CaseResult(
        case_id=case.id,
        tags=case.tags,
        output=out,
        scores={s.name: s.score(case, out) for s in scorers},
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
    if cfg.judge is not None:
        judge = make_provider(cfg.judge, client=client)
        if cache is not None:
            judge = CachedProvider(judge, cache, base_url=cfg.judge.base_url)
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
        environment=environment(cfg, client),
    )
    return Run(info=info, results=results)


def save_run(run: Run, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> Run:
    return Run.model_validate_json(Path(path).read_text(encoding="utf-8"))
