"""Memory node (DESIGN §5.1; paper §4.1)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

NodeType = Literal["observation", "reflection", "plan"]


@dataclass
class MemoryNode:
    id: str
    agent: str
    type: NodeType
    # observation: seed | perception | dialogue | inner_voice
    # reflection:  insight | conversation_note
    # plan:        day_plan | hour_plan | commitment
    subtype: str
    description: str
    created: datetime
    last_accessed: datetime
    importance: int
    embedding_key: str
    evidence: list[str] = field(default_factory=list)
    depth: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "agent": self.agent, "type": self.type, "subtype": self.subtype,
            "description": self.description, "created": self.created.isoformat(),
            "last_accessed": self.last_accessed.isoformat(), "importance": self.importance,
            "embedding_key": self.embedding_key, "evidence": list(self.evidence), "depth": self.depth,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MemoryNode":
        return cls(
            id=d["id"], agent=d["agent"], type=d["type"], subtype=d["subtype"], description=d["description"],
            created=datetime.fromisoformat(d["created"]), last_accessed=datetime.fromisoformat(d["last_accessed"]),
            importance=int(d["importance"]), embedding_key=d["embedding_key"], evidence=list(d.get("evidence", [])),
            depth=int(d.get("depth", 0)), meta=dict(d.get("meta", {})),
        )
