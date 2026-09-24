"""Voyage AI embedder. Reads VOYAGE_API_KEY from the environment."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .base import EmbedBatch, EmbedKind, l2_normalize

INSTALL_HINT = 'the Voyage embedder needs the voyageai package: pip install -e ".[voyage]"'


class VoyageEmbedder:
    provider = "voyage"
    kind_sensitive = True  # input_type="query" vs "document" produce different vectors

    def __init__(self, model: str = "voyage-4-lite", batch_size: int = 64, client: Any | None = None) -> None:
        self.model = model
        self.batch_size = batch_size
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import voyageai
            except ImportError as exc:
                raise RuntimeError(INSTALL_HINT) from exc
            self._client = voyageai.Client()
        return self._client

    def embed(self, texts: Sequence[str], kind: EmbedKind = "document") -> EmbedBatch:
        client = self._get_client()
        vectors: list[list[float]] = []
        tokens = 0
        for i in range(0, len(texts), self.batch_size):
            chunk = list(texts[i : i + self.batch_size])
            result = client.embed(chunk, model=self.model, input_type=kind)
            vectors.extend(result.embeddings)
            tokens += int(getattr(result, "total_tokens", 0) or 0)
        return EmbedBatch(vectors=l2_normalize(np.asarray(vectors, dtype=np.float32)), tokens=tokens)
