"""Mutable world state: where agents are, what they are doing, object states."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .tree import World


@dataclass
class Transit:
    origin: str
    destination: str
    depart_tick: int
    arrive_tick: int


@dataclass
class Position:
    place: str
    area: str | None = None
    transit: Transit | None = None

    @property
    def in_transit(self) -> bool:
        return self.transit is not None


@dataclass
class ActionState:
    """The action an agent is visibly performing (what others observe)."""

    action_id: str
    description: str          # "Isabella Rodriguez is brewing coffee for customers"
    activity: str             # "brewing coffee for customers"
    emoji: str = "🙂"
    label: str = ""
    object: str | None = None  # object path
    object_state: str | None = None
    kind: str = "planned"      # planned | reaction | chat | sleep | travel


@dataclass
class WorldState:
    world: World
    positions: dict[str, Position] = field(default_factory=dict)
    actions: dict[str, ActionState] = field(default_factory=dict)
    object_states: dict[str, str] = field(default_factory=dict)
    object_users: dict[str, str] = field(default_factory=dict)   # object path -> agent using it

    @classmethod
    def initial(cls, world: World) -> "WorldState":
        return cls(world=world, object_states=world.default_states())

    def agents_at(self, place: str) -> list[str]:
        return sorted(a for a, p in self.positions.items() if p.place == place and not p.in_transit)

    def set_object_state(self, path: str, state: str | None, agent: str | None) -> bool:
        """Returns True if the visible state changed."""
        node = self.world.nodes.get(path)
        if node is None:
            return False
        new = state or node.default_state or "idle"
        old = self.object_states.get(path)
        self.object_states[path] = new
        if agent and state:
            self.object_users[path] = agent
        else:
            self.object_users.pop(path, None)
        return old != new

    def snapshot(self) -> dict[str, Any]:
        agents = {}
        for name, pos in sorted(self.positions.items()):
            act = self.actions.get(name)
            agents[name] = {
                "place": pos.place,
                "area": pos.area,
                "in_transit": None if pos.transit is None else {
                    "from": pos.transit.origin, "to": pos.transit.destination,
                    "depart_tick": pos.transit.depart_tick, "arrive_tick": pos.transit.arrive_tick,
                },
                "action_id": act.action_id if act else None,
                "description": act.description if act else None,
                "emoji": act.emoji if act else None,
                "label": act.label if act else None,
            }
        changed = {p: s for p, s in sorted(self.object_states.items())
                   if s != (self.world.nodes[p].default_state or "idle")}
        return {"agents": agents, "object_states": changed}
