"""Embedding cache keyed by text hash, plus the cached/budgeted/logged wrapper used everywhere."""

from __future__ import annotations

import hashlib
import sqlite3
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from ..llm.budget import BudgetGuard
from ..llm.pricing import embedding_cost, estimate_tokens
from ..llm.usage import UsageSummary
from ..sim.events import EventSink, NullSink
from .base import Embedder, EmbedKind


def embedding_key(provider: str, model: str, kind: str, text: str) -> str:
    return hashlib.sha256(f"{provider}|{model}|{kind}|{text}".encode()).hexdigest()


class EmbeddingCache:
    """sqlite-backed ``key -> float32 vector`` store with an in-memory layer. ``path=None`` = memory only."""

    def __init__(self, path: str | Path | None) -> None:
        self._mem: dict[str, np.ndarray] = {}
        self._db: sqlite3.Connection | None = None
        if path is not None:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(p)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, dim INTEGER, vec BLOB)")
            self._db.commit()

    def get_many(self, keys: Sequence[str]) -> dict[str, np.ndarray]:
        found = {k: self._mem[k] for k in keys if k in self._mem}
        missing = [k for k in keys if k not in found]
        if missing and self._db is not None:
            for i in range(0, len(missing), 500):
                chunk = missing[i : i + 500]
                q = f"SELECT key, dim, vec FROM emb WHERE key IN ({','.join('?' * len(chunk))})"
                for key, dim, blob in self._db.execute(q, chunk):
                    vec = np.frombuffer(blob, dtype=np.float32).reshape(dim).copy()
                    self._mem[key] = vec
                    found[key] = vec
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        self._mem.update(items)
        if self._db is not None and items:
            self._db.executemany(
                "INSERT OR REPLACE INTO emb (key, dim, vec) VALUES (?, ?, ?)",
                [(k, int(v.shape[0]), np.asarray(v, np.float32).tobytes()) for k, v in items.items()],
            )
            self._db.commit()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None


class CachedEmbedder:
    """Cache lookup -> budget check -> provider call -> cost + ``embed_call`` event."""

    def __init__(
        self,
        inner: Embedder,
        cache: EmbeddingCache,
        *,
        price_per_mtok: float = 0.0,
        budget: BudgetGuard | None = None,
        sink: EventSink | None = None,
        usage: UsageSummary | None = None,
        chars_per_token: float = 3.0,
    ) -> None:
        self.inner = inner
        self.cache = cache
        self.price = price_per_mtok
        self.budget = budget
        self.sink = sink or NullSink()
        self.usage = usage
        self.chars_per_token = chars_per_token

    @property
    def provider(self) -> str:
        return self.inner.provider

    @property
    def model(self) -> str:
        return self.inner.model

    def key(self, text: str, kind: EmbedKind = "document") -> str:
        k = kind if self.inner.kind_sensitive else "*"
        return embedding_key(self.inner.provider, self.inner.model, k, text)

    def embed(self, texts: Sequence[str], kind: EmbedKind = "document") -> np.ndarray:
        keys = [self.key(t, kind) for t in texts]
        found = self.cache.get_many(keys)
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in found and k not in todo:
                todo[k] = t
        tokens, cost, latency_ms = 0, 0.0, 0.0
        if todo:
            if self.budget is not None and self.price > 0:
                est_tokens = estimate_tokens(sum(len(t) for t in todo.values()), self.chars_per_token)
                self.budget.check(embedding_cost(self.price, est_tokens), context=f"embed {len(todo)} texts")
            t0 = time.perf_counter()
            batch = self.inner.embed(list(todo.values()), kind)
            latency_ms = (time.perf_counter() - t0) * 1000
            tokens = batch.tokens
            cost = embedding_cost(self.price, tokens)
            if self.budget is not None and cost:
                self.budget.charge(cost)
            new = {k: v for k, v in zip(todo.keys(), batch.vectors)}
            self.cache.put_many(new)
            found.update(new)
        data = {
            "provider": self.inner.provider, "model": self.inner.model, "kind": kind,
            "n_texts": len(texts), "n_cached": len(texts) - len(todo),
            "tokens": tokens, "cost_usd": round(cost, 8), "latency_ms": round(latency_ms, 1),
        }
        if todo:  # pure cache hits are not worth an event each
            self.sink.emit("embed_call", data)
        if self.usage is not None:
            self.usage.add_embed_call(data)
        return np.stack([found[k] for k in keys]) if keys else np.zeros((0, 0), np.float32)

    def embed_one(self, text: str, kind: EmbedKind = "document") -> np.ndarray:
        return self.embed([text], kind)[0]
