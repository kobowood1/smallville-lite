"""Deterministic stub embedder: signed feature hashing of words.

Texts that share words land close together, so retrieval tests stay meaningful without any
model download or network call.
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

from .base import EmbedBatch, EmbedKind, l2_normalize

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an the and or of to in on at for with is are was were be been it its this that he she they "
    "his her their i you we my your our as by from about".split()
)


class StubEmbedder:
    provider = "stub"
    kind_sensitive = False

    def __init__(self, model: str = "stub-hash-256", dim: int = 256) -> None:
        self.model = model
        self.dim = dim

    def embed(self, texts: Sequence[str], kind: EmbedKind = "document") -> EmbedBatch:
        return EmbedBatch(vectors=np.stack([self._one(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32))

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        words = [w for w in _WORD.findall(text.lower()) if w not in _STOP]
        for w in words:
            h = int.from_bytes(hashlib.blake2b(w.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 if (h >> 32) & 1 else -1.0
        if not vec.any():
            # No content words: a stable pseudo-random direction for this exact text.
            seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
            vec = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
        return l2_normalize(vec[None, :])[0]
