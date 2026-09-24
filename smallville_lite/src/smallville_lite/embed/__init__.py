"""Embeddings behind one interface: local (default), Voyage, or stub; always cached by text hash."""

from __future__ import annotations

from ..config import Config
from ..llm.budget import BudgetGuard
from ..llm.usage import UsageSummary
from ..sim.events import EventSink
from .base import EmbedBatch, Embedder, EmbedKind
from .cache import CachedEmbedder, EmbeddingCache, embedding_key


def build_embedder(
    config: Config,
    *,
    budget: BudgetGuard | None = None,
    sink: EventSink | None = None,
    usage: UsageSummary | None = None,
    cache_path: str | None = "config",
) -> CachedEmbedder:
    """Build the configured embedder. ``cache_path="config"`` uses ``[embedding].cache_path``;
    ``None`` keeps the cache in memory (tests)."""
    settings = config.embedding
    spec = config.embedding_model()
    inner: Embedder
    if settings.provider == "stub":
        from .stub import StubEmbedder

        inner = StubEmbedder(model=spec.id, dim=spec.dim)
    elif settings.provider == "local":
        from .local import LocalEmbedder

        inner = LocalEmbedder(model=spec.id, batch_size=settings.batch_size)
    else:
        from .voyage import VoyageEmbedder

        inner = VoyageEmbedder(model=spec.id, batch_size=settings.batch_size)
    path = settings.cache_path if cache_path == "config" else cache_path
    return CachedEmbedder(
        inner, EmbeddingCache(path), price_per_mtok=spec.price, budget=budget, sink=sink, usage=usage,
        chars_per_token=config.llm.chars_per_token,
    )


__all__ = ["CachedEmbedder", "EmbedBatch", "Embedder", "EmbedKind", "EmbeddingCache", "build_embedder", "embedding_key"]
