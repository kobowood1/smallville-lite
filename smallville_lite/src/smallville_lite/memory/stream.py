"""Per-agent memory stream: append-only nodes, their embeddings, and the reflection accumulator."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np

from ..embed import CachedEmbedder
from ..sim.events import EventSink, NullSink
from .node import MemoryNode, NodeType


@dataclass
class NewMemory:
    type: NodeType
    subtype: str
    description: str
    importance: int
    evidence: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def agent_slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


class MemoryStream:
    def __init__(self, agent: str, embedder: CachedEmbedder, sink: EventSink | None = None) -> None:
        self.agent = agent
        self.embedder = embedder
        self.sink = sink or NullSink()
        self.nodes: list[MemoryNode] = []
        self.by_id: dict[str, MemoryNode] = {}
        self._vectors: dict[str, np.ndarray] = {}
        self._slug = agent_slug(agent)
        # Sum of importance of observations since the last reflection (paper §4.2).
        self.importance_since_reflection = 0.0

    def __len__(self) -> int:
        return len(self.nodes)

    # -- writing -------------------------------------------------------
    def add(self, created: datetime, memory: NewMemory) -> MemoryNode:
        return self.add_many(created, [memory])[0]

    def add_many(self, created: datetime, memories: Sequence[NewMemory]) -> list[MemoryNode]:
        if not memories:
            return []
        texts = [m.description for m in memories]
        vectors = self.embedder.embed(texts, "document")
        added = []
        for m, vec in zip(memories, vectors):
            for ev in m.evidence:
                if ev not in self.by_id:
                    raise ValueError(f"evidence {ev!r} is not in {self.agent}'s memory")
            depth = 1 + max((self.by_id[e].depth for e in m.evidence), default=0) if m.type == "reflection" else 0
            node = MemoryNode(
                id=f"{self._slug}:{len(self.nodes) + 1}", agent=self.agent, type=m.type, subtype=m.subtype,
                description=m.description, created=created, last_accessed=created,
                importance=max(1, min(10, int(m.importance))), embedding_key=self.embedder.key(m.description),
                evidence=list(m.evidence), depth=depth, meta=dict(m.meta),
            )
            self.nodes.append(node)
            self.by_id[node.id] = node
            self._vectors[node.id] = np.asarray(vec, dtype=np.float32)
            if node.type == "observation":
                self.importance_since_reflection += node.importance
            self.sink.emit("memory_added", {"node": node.to_dict()}, agent=self.agent,
                           game_time=created.isoformat())
            added.append(node)
        return added

    def reset_reflection_accumulator(self) -> None:
        self.importance_since_reflection = 0.0

    # -- reading -------------------------------------------------------
    def get(self, node_id: str) -> MemoryNode:
        return self.by_id[node_id]

    def recent(self, n: int, types: Iterable[str] | None = None) -> list[MemoryNode]:
        """The ``n`` most recently created nodes, oldest first."""
        pool = self.nodes if types is None else [x for x in self.nodes if x.type in set(types)]
        return pool[-n:] if n > 0 else []

    def of_type(self, type: str, subtype: str | None = None) -> list[MemoryNode]:
        return [x for x in self.nodes if x.type == type and (subtype is None or x.subtype == subtype)]

    def vector(self, node_id: str) -> np.ndarray:
        return self._vectors[node_id]

    def matrix(self, nodes: Sequence[MemoryNode]) -> np.ndarray:
        if not nodes:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([self._vectors[x.id] for x in nodes])

    # -- persistence (checkpoints) -------------------------------------
    def state(self) -> dict[str, Any]:
        return {"nodes": [x.to_dict() for x in self.nodes], "importance_since_reflection": self.importance_since_reflection}

    def load_state(self, state: dict[str, Any]) -> None:
        self.nodes = [MemoryNode.from_dict(d) for d in state["nodes"]]
        self.by_id = {x.id: x for x in self.nodes}
        # Vectors come back from the embedding cache (keyed by text), not from the checkpoint.
        vecs = self.embedder.embed([x.description for x in self.nodes], "document") if self.nodes else []
        self._vectors = {x.id: np.asarray(v, np.float32) for x, v in zip(self.nodes, vecs)}
        self.importance_since_reflection = float(state.get("importance_since_reflection", 0.0))
