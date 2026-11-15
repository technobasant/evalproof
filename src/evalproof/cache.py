"""Content-addressed response cache.

The key is everything that determines a model's reply: provider, model, endpoint,
messages and generation parameters. Re-running an unchanged suite costs nothing and
returns byte-identical results; changing one prompt re-runs only the calls it touches.

A hit returns the *original* latency and token counts, flagged ``cached=True``, so a
re-run reproduces the run artifact; the run's own wall time shows the real speed-up.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from evalproof._fsutil import replace_retry
from evalproof.config import GenerationParams
from evalproof.providers.base import CacheMiss, ChatResponse, Message, Provider, close_resource


def cache_key(
    provider: str,
    model: str,
    base_url: str | None,
    messages: list[Message],
    params: GenerationParams,
    extra: dict[str, object] | None = None,
) -> str:
    """``extra`` is what a provider adds when its request differs from what the same
    params meant before (a hosted call that now omits ``temperature``). It joins the key only
    when present, so every key that existed before it was added is unchanged."""
    material: dict[str, object] = {
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "messages": [m.model_dump() for m in messages],
        "params": params.model_dump(mode="json"),
    }
    if extra:
        material["extra"] = extra
    blob = json.dumps(material, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# The trial a model call belongs to. ``LLMSystem.run`` sets it around its provider calls, in the
# thread that makes them, and ``CachedProvider`` reads it. A context variable keeps the provider
# interface as it was (no provider takes a ``trial`` argument), and the judge, which runs outside
# the system's call, always sees trial 0, so identical outputs get the same cached judgement.
_TRIAL: ContextVar[int] = ContextVar("evalproof_trial", default=0)


@contextmanager
def trial_scope(trial: int) -> Iterator[None]:
    """Inside the block, a ``CachedProvider`` call is keyed as that trial's. Trial 0 is the
    key as it always was; a later trial adds ``{"trial": n}`` to the key's ``extra``."""
    token = _TRIAL.set(trial)
    try:
        yield
    finally:
        _TRIAL.reset(token)


# The case (and trial) a call is made for, so a ``--cache-only`` miss can name it. ``score_case``
# sets it around the system's call and the scoring; unlike the trial it does not change a key.
_CASE: ContextVar[tuple[str, int] | None] = ContextVar("evalproof_case", default=None)


@contextmanager
def case_scope(case_id: str, trial: int = 0) -> Iterator[None]:
    """Inside the block, a ``CachedProvider`` miss in cache-only mode names this case and trial."""
    token = _CASE.set((case_id, trial))
    try:
        yield
    finally:
        _CASE.reset(token)


class ResponseCache:
    """One JSON file per key, written to a unique temp file and moved into place, so a reader
    never sees half a write and two threads storing the same key cannot trip over each other.

    The cache is best-effort: a lookup that cannot read an entry (a torn write, or Windows
    refusing a read while another handle replaces the file) is a miss, and a store that cannot
    complete is dropped. Neither ever fails the case that triggered it. ``sleep`` is how the
    retry of a refused replace waits; tests pass a fake."""

    def __init__(self, root: str | Path, *, sleep: Callable[[float], None] = time.sleep) -> None:
        self.root = Path(root)
        self._sleep = sleep

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ChatResponse | None:
        path = self._path(key)
        try:
            resp = ChatResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None  # absent, being replaced, or a torn write: a miss, not a crash
        return resp.model_copy(update={"cached": True})

    def put(self, key: str, response: ChatResponse) -> None:
        path = self._path(key)
        payload = response.model_copy(update={"cached": False}).model_dump_json().encode("utf-8")
        tmp: str | None = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f"{key}.", suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                fh.write(payload)
            if replace_retry(tmp, path, sleep=self._sleep):
                tmp = None  # moved into place; nothing to clean up
        except OSError:
            pass  # best-effort: a cache that cannot be written is a cache miss next time
        finally:
            if tmp is not None:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)

    def stats(self) -> dict[str, int]:
        # Only finished entries: a stray ``<key>.<random>.tmp`` from an interrupted put is not one.
        files = list(self.root.glob("*/*.json")) if self.root.exists() else []
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}


class CachedProvider:
    """Wraps any provider; transparent on a miss, free on a hit.

    With ``cache_only`` a miss raises ``CacheMiss`` instead of calling the inner provider, so
    the wrapper can answer from the cache alone and never reach a model. ``role`` (``system`` or
    ``judge``) is only used to word that error."""

    def __init__(
        self,
        inner: Provider,
        cache: ResponseCache,
        base_url: str | None = None,
        *,
        cache_only: bool = False,
        role: str = "model",
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.model = inner.model
        self.base_url = base_url
        self.cache_only = cache_only
        self.role = role
        self.hits = 0
        self.misses = 0

    def close(self) -> None:
        close_resource(self.inner)

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        salt = getattr(self.inner, "cache_salt", None)
        extra: dict[str, object] | None = salt(params) if callable(salt) else None
        trial = _TRIAL.get()
        if trial:  # trial 0 keeps the key it had before trials existed
            extra = {**(extra or {}), "trial": trial}
        key = cache_key(self.name, self.model, self.base_url, messages, params, extra)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        if self.cache_only:
            raise self._miss()
        self.misses += 1
        resp = self.inner.chat(messages, params)
        self.cache.put(key, resp)
        return resp

    def _miss(self) -> CacheMiss:
        case = _CASE.get()
        where = f" for case {case[0]!r} (trial {case[1]})" if case else ""
        return CacheMiss(
            f"--cache-only: no cached reply{where} from the {self.role} "
            f"({self.name}:{self.model}); no model was called. "
            "Run without --cache-only to make the call."
        )
