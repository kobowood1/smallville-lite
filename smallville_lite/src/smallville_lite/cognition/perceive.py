"""Perception (paper §4.1 observations, §5): what an agent notices at its current place.

``observe`` reads the start-of-tick world snapshot (pure). ``perceive`` filters for new
observations (retention), caps them (attention bandwidth, other agents first), scores their
importance in one batched call, and writes them to the memory stream.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..memory import MemoryNode, NewMemory
from ..world import WorldState
from .importance import score_importance
from .mind import Mind

if TYPE_CHECKING:
    from ..agent import Agent


@dataclass(frozen=True)
class Observation:
    subject: str          # agent name or object path
    text: str
    kind: str             # self | agent | object
    salient: bool         # worth a react decision (another agent, or an object in a non-default state)


def observe(state: WorldState, name: str) -> list[Observation]:
    pos = state.positions.get(name)
    if pos is None or pos.in_transit:
        return []
    obs: list[Observation] = []
    own = state.actions.get(name)
    if own is not None:
        obs.append(Observation(subject=name, text=own.description, kind="self", salient=False))
    for other in state.agents_at(pos.place):
        if other == name:
            continue
        act = state.actions.get(other)
        text = act.description if act else f"{other} is at {pos.place}"
        obs.append(Observation(subject=other, text=text, kind="agent", salient=True))
    for node in state.world.objects_in(pos.place):
        current = state.object_states.get(node.path, node.default_state or "idle")
        if current != (node.default_state or "idle"):
            text = f"The {node.name} at {pos.place} is {current}"
            obs.append(Observation(subject=node.path, text=text, kind="object", salient=state.object_users.get(node.path) != name))
    return obs


def perceive(mind: Mind, agent: "Agent", observations: list[Observation], now: datetime) -> list[tuple[Observation, MemoryNode]]:
    fresh: list[Observation] = []
    seen = agent.scratch.recent_observations
    # Other agents first, then objects, then the agent's own action (paper: own behaviour is observed too).
    order = {"agent": 0, "object": 1, "self": 2}
    for ob in sorted(observations, key=lambda o: order[o.kind]):
        key = (ob.subject, ob.text)
        if key in seen:
            continue
        fresh.append(ob)
        if len(fresh) >= mind.sim.att_bandwidth:
            break
    if not fresh:
        return []
    for ob in fresh:
        seen.append((ob.subject, ob.text))
    scores = score_importance(mind, agent, [o.text for o in fresh])
    place = agent.scratch.action.place if agent.scratch.action else None
    nodes = agent.memory.add_many(now, [
        NewMemory(type="observation", subtype="perception", description=o.text, importance=s,
                  meta={"subject": o.subject, "kind": o.kind, "salient": o.salient, "place": place})
        for o, s in zip(fresh, scores)
    ])
    mind.sink.emit("perceived", {"observations": [
        {"node_id": n.id, "text": o.text, "salient": o.salient} for o, n in zip(fresh, nodes)
    ]}, agent=agent.name, game_time=now.isoformat())
    return list(zip(fresh, nodes))
