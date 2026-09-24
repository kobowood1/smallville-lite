"""Embedder interface (DESIGN §6.5)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

import numpy as np

EmbedKind = Literal["document", "query"]


@dataclass
class EmbedBatch:
    vectors: np.ndarray        # shape (n, dim), float32, L2-normalized
    tokens: int = 0            # billable tokens reported by the provider (0 for local/stub)


class Embedder(Protocol):
    provider: str
    model: str
    # True if the provider embeds queries and documents differently (Voyage input_type);
    # the cache key then includes the kind.
    kind_sensitive: bool

    def embed(self, texts: Sequence[str], kind: EmbedKind) -> EmbedBatch: ...


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms
