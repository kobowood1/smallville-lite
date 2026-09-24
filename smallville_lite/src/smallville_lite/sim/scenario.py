"""Load a scenario and assemble a ready-to-run Simulation (DESIGN §4)."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..agent import Agent, Identity
from ..clock import GameClock
from ..config import Config, load_config
from ..embed import build_embedder
from ..llm import build_llm
from ..llm.prompts import load_prompt
from ..memory import MemoryStream, NewMemory
from ..world import Position, World, WorldState, load_world
from .events import EventLog

SCENARIO_ROOT = Path(__file__).resolve().parents[3] / "scenarios"


@dataclass
class Scenario:
    name: str
    directory: Path
    description: str
    start: datetime
    days: int
    world: World
    identities: list[Identity]
    facts: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    config_overrides: dict[str, Any] = field(default_factory=dict)


def load_scenario(name_or_path: str | Path) -> Scenario:
    path = Path(name_or_path)
    directory = path if path.is_dir() else SCENARIO_ROOT / str(name_or_path)
    with (directory / "scenario.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    world = load_world(directory / raw.get("world", "world.toml"))
    with (directory / raw.get("agents", "agents.toml")).open("rb") as fh:
        agents_raw = tomllib.load(fh)["agent"]
    identities = [_identity(a, world) for a in agents_raw]
    names = [i.name for i in identities]
    if len(set(names)) != len(names):
        raise ValueError("agent names must be unique")
    events = []
    for ev in raw.get("events", []):
        ev = dict(ev)
        ev["start"], ev["end"] = str(ev["start"]), str(ev["end"])
        events.append(ev)
    return Scenario(
        name=raw.get("name", directory.name), directory=directory, description=raw.get("description", ""),
        start=datetime.fromisoformat(str(raw["start"])), days=int(raw.get("days", 2)), world=world,
        identities=identities, facts=list(raw.get("facts", [])), events=events,
        config_overrides=dict(raw.get("config", {})),
    )


def _identity(a: dict[str, Any], world: World) -> Identity:
    home = a["home"]
    world.place(home)
    start = a.get("start", home)
    if start not in world.nodes:
        raise ValueError(f"{a['name']}: start {start!r} is not in the world")
    for p in a.get("known_places", []):
        world.place(p)
    bio = [p.strip() for p in a["bio"].split(";") if p.strip()]
    return Identity(name=a["name"], age=int(a["age"]), traits=a.get("traits", ""), home=home,
                    home_area=a.get("home_area"), start=start, bio=bio, extra_places=list(a.get("known_places", [])),
                    wake_hint=a.get("wake_hint", ""))


def render_rules(world: World) -> str:
    return load_prompt("rules").render(
        town=world.name, places="\n".join(f"- {world.describe_place(p.name)}" for p in world.places)
    )


def scenario_config(scenario: Scenario, config_dir: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    from ..config import deep_merge

    merged = deep_merge(scenario.config_overrides, overrides or {})
    return load_config(config_dir, overrides=merged)


def wire_simulation(
    scenario: Scenario,
    config: Config,
    log: EventLog,
    *,
    embed_cache_path: str | None = "config",
    call_id_prefix: str = "c",
):
    """Model layer + world + empty agents, with no events and no memories. Shared by build and restore."""
    from ..cognition import Mind
    from .engine import Simulation

    llm = build_llm(config, sink=log, call_id_prefix=call_id_prefix)
    if config.stub:
        from ..llm.stub_brain import make_responders

        for task, fn in make_responders(scenario.facts, scenario.events).items():
            llm.backend.register(task, fn)  # type: ignore[attr-defined]
    embedder = build_embedder(config, budget=llm.budget, sink=log, usage=llm.usage, cache_path=embed_cache_path)
    clock = GameClock(scenario.start, config.sim.tick_minutes)
    mind = Mind(config=config, llm=llm, embedder=embedder, sink=log, world=scenario.world, clock=clock,
                rules=render_rules(scenario.world))
    mind.now = scenario.start
    state = WorldState.initial(scenario.world)
    agents: list[Agent] = []
    for ident in scenario.identities:
        agent = Agent(ident, MemoryStream(ident.name, embedder, sink=log), retention=config.sim.retention)
        agents.append(agent)
        mind.agents[ident.name] = agent
    return Simulation(mind, agents, state, log)


def build_simulation(scenario: Scenario, config: Config, log: EventLog, *, embed_cache_path: str | None = "config",
                     call_id_prefix: str = "c"):
    """Wire everything, emit world/agent init events, and seed memories (paper §3.1). Returns a Simulation at tick 0."""
    from ..cognition.importance import score_importance

    sim = wire_simulation(scenario, config, log, embed_cache_path=embed_cache_path, call_id_prefix=call_id_prefix)
    mind, state = sim.mind, sim.state
    log.set_time(None, scenario.start)
    log.emit("world_init", {
        "town": scenario.world.name,
        "places": [{"name": p.name, "public": p.public, "description": p.description,
                    "children": [_tree(c) for c in p.children]} for p in scenario.world.places],
        "travel_ticks_default": scenario.world.travel_default,
        "tick_minutes": config.sim.tick_minutes, "start": scenario.start.isoformat(),
    })
    for agent in sim.agents.values():
        ident = agent.identity
        place = scenario.world.place_of(ident.start)
        state.positions[ident.name] = Position(place=place, area=ident.start if ident.start != place else None)
        agent.scratch.visited.add(place)
        log.emit("agent_init", {"age": ident.age, "traits": ident.traits, "bio": ident.bio, "home": ident.home,
                                "start": ident.start, "known_places": agent.known_places(scenario.world.public_places())},
                 agent=ident.name)
        scores = score_importance(mind, agent, ident.bio)
        agent.memory.add_many(scenario.start, [
            NewMemory("observation", "seed", phrase, importance=s) for phrase, s in zip(ident.bio, scores)
        ])
        # Seed memories are the starting point, not a trigger for an immediate reflection.
        agent.memory.reset_reflection_accumulator()
    log.commit_tick()
    return sim


def _tree(node) -> dict[str, Any]:
    out: dict[str, Any] = {"name": node.name, "kind": node.kind, "path": node.path}
    if node.kind == "object":
        out["state"] = node.default_state
    if node.children:
        out["children"] = [_tree(c) for c in node.children]
    return out
