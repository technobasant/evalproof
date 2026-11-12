"""Content-addressed response cache.

The key is everything that determines a model's reply: provider, model, endpoint,
messages and generation parameters. Re-running an unchanged suite costs nothing and
returns byte-identical results; changing one prompt re-runs only the calls it touches.

A hit returns the *original* latency and token counts, flagged ``cached=True``, so a
re-run reproduces the run artifact; the run's own wall time shows the real speed-up.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from evalproof.config import GenerationParams
from evalproof.providers.base import ChatResponse, Message, Provider


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


class ResponseCache:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> ChatResponse | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            resp = ChatResponse.model_validate_json(path.read_text(encoding="utf-8"))
        except ValueError:
            return None  # a torn write is a miss, not a crash
        return resp.model_copy(update={"cached": True})

    def put(self, key: str, response: ChatResponse) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(response.model_copy(update={"cached": False}).model_dump_json(), "utf-8")
        tmp.replace(path)

    def stats(self) -> dict[str, int]:
        files = list(self.root.glob("*/*.json")) if self.root.exists() else []
        return {"entries": len(files), "bytes": sum(f.stat().st_size for f in files)}


class CachedProvider:
    """Wraps any provider; transparent on a miss, free on a hit."""

    def __init__(self, inner: Provider, cache: ResponseCache, base_url: str | None = None) -> None:
        self.inner = inner
        self.cache = cache
        self.name = inner.name
        self.model = inner.model
        self.base_url = base_url
        self.hits = 0
        self.misses = 0

    def chat(self, messages: list[Message], params: GenerationParams) -> ChatResponse:
        salt = getattr(self.inner, "cache_salt", None)
        extra: dict[str, object] | None = salt(params) if callable(salt) else None
        key = cache_key(self.name, self.model, self.base_url, messages, params, extra)
        hit = self.cache.get(key)
        if hit is not None:
            self.hits += 1
            return hit
        self.misses += 1
        resp = self.inner.chat(messages, params)
        self.cache.put(key, resp)
        return resp
