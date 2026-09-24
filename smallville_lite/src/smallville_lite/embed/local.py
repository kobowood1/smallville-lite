"""Local sentence-transformers embedder: the zero-cost default for real runs."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import EmbedBatch, EmbedKind, l2_normalize

INSTALL_HINT = 'the local embedder needs sentence-transformers: pip install -e ".[local]"'


class LocalEmbedder:
    provider = "local"
    kind_sensitive = False

    def __init__(self, model: str = "all-MiniLM-L6-v2", batch_size: int = 64) -> None:
        self.model = model
        self.batch_size = batch_size
        self._st: Any = None

    def _load(self) -> Any:
        if self._st is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError(INSTALL_HINT) from exc
            self._st = SentenceTransformer(self.model, device="cpu")
        return self._st

    def embed(self, texts: Sequence[str], kind: EmbedKind = "document") -> EmbedBatch:
        if not texts:
            return EmbedBatch(vectors=np.zeros((0, 0), np.float32))
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        return EmbedBatch(vectors=l2_normalize(vectors))
