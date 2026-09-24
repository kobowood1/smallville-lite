"""Retrieval scoring (paper §4.1; DESIGN §5.2).

score = w_rec * recency + w_imp * importance + w_rel * relevance, where
  recency    = decay ** (game-hours since the memory was last accessed)
  importance = the 1-10 score assigned at creation
  relevance  = cosine(query embedding, memory embedding)
and each component is min-max normalized to [0, 1] over the candidate set (all-equal -> 0.5).
Retrieval updates ``last_accessed`` on the returned memories.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from .node import MemoryNode
from .stream import MemoryStream


@dataclass(frozen=True)
class Weights:
    recency: float = 1.0
    importance: float = 1.0
    relevance: float = 1.0
    decay: float = 0.995


@dataclass
class ScoredMemory:
    node: MemoryNode
    score: float
    recency: float
    importance: float
    relevance: float


def min_max(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-12:
        return np.full_like(x, 0.5, dtype=np.float64)
    return (x - lo) / (hi - lo)


def score_memories(
    nodes: Sequence[MemoryNode],
    vectors: np.ndarray,
    query_vec: np.ndarray,
    now: datetime,
    weights: Weights,
) -> list[ScoredMemory]:
    """Score and rank ``nodes`` (highest first). Pure function; does not touch last_accessed."""
    if not nodes:
        return []
    hours = np.array([max(0.0, (now - n.last_accessed).total_seconds() / 3600.0) for n in nodes])
    recency = min_max(np.power(weights.decay, hours))
    importance = min_max(np.array([float(n.importance) for n in nodes]))
    relevance = min_max(vectors.astype(np.float64) @ np.asarray(query_vec, dtype=np.float64))
    total = weights.recency * recency + weights.importance * importance + weights.relevance * relevance
    scored = [
        ScoredMemory(node=n, score=float(total[i]), recency=float(recency[i]),
                     importance=float(importance[i]), relevance=float(relevance[i]))
        for i, n in enumerate(nodes)
    ]
    # Deterministic ordering: score, then newer first, then insertion order.
    index = {n.id: i for i, n in enumerate(nodes)}
    scored.sort(key=lambda s: (-s.score, -s.node.created.timestamp(), -index[s.node.id]))
    return scored


def retrieve(
    stream: MemoryStream,
    query: str,
    now: datetime,
    weights: Weights,
    *,
    k: int,
    types: Iterable[str] | None = None,
    exclude_subtypes: Iterable[str] = (),
    purpose: str = "",
    update_access: bool = True,
) -> list[ScoredMemory]:
    type_set = set(types) if types is not None else None
    skip = set(exclude_subtypes)
    candidates = [
        n for n in stream.nodes
        if (type_set is None or n.type in type_set) and n.subtype not in skip and n.created <= now
    ]
    if not candidates:
        return []
    query_vec = stream.embedder.embed_one(query, "query")
    top = score_memories(candidates, stream.matrix(candidates), query_vec, now, weights)[:k]
    if update_access:
        for s in top:
            s.node.last_accessed = now
    stream.sink.emit(
        "memory_accessed",
        {
            "query": query,
            "purpose": purpose,
            "results": [
                {"node_id": s.node.id, "score": round(s.score, 4), "recency": round(s.recency, 4),
                 "importance": round(s.importance, 4), "relevance": round(s.relevance, 4)}
                for s in top
            ],
        },
        agent=stream.agent,
        game_time=now.isoformat(),
    )
    return top


def numbered(memories: Sequence[ScoredMemory] | Sequence[MemoryNode], start: int = 1) -> str:
    """Render memories as a numbered list for prompts ('1. [Feb 13 08:10] text')."""
    lines = []
    for i, m in enumerate(memories, start=start):
        node = m.node if isinstance(m, ScoredMemory) else m
        lines.append(f"{i}. [{node.created.strftime('%b %d %H:%M')}] {node.description}")
    return "\n".join(lines) if lines else "(nothing relevant)"
