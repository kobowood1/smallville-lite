"""Save and restore a whole simulation at a tick boundary (DESIGN §8.4).

A checkpoint is taken after every committed tick, so it always describes a complete tick; a tick
aborted by the budget never reaches a checkpoint. Files are written atomically. Retention keeps the
most recent ``keep`` checkpoints plus every ``keep_every``-th tick, so interviews can be run against
the saved memory state at (almost) any point of a run.
"""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..agent import CurrentAction, DayPlan, HourBlock, Scratch, Step
from ..world import ActionState, Position, Transit

if TYPE_CHECKING:
    from .engine import Simulation

VERSION = 1
_NAME = re.compile(r"^tick_(\d{6})\.json$")


# ---------------------------------------------------------------------------
# save / load

def snapshot(sim: "Simulation") -> dict[str, Any]:
    backend = sim.mind.llm.backend
    return {
        "version": VERSION,
        "tick": sim.tick,
        "conv_counter": sim._conv_counter,
        "log": {"seq": sim.log.seq, "segment": sim.log.segment},
        "budget_spent": sim.mind.llm.budget.spent,
        "world": _world_to_dict(sim.state),
        "summaries": sim.mind.summaries.state(),
        "agents": {name: {"memory": a.memory.state(), "scratch": _scratch_to_dict(a.scratch)}
                   for name, a in sim.agents.items()},
        "stub_seen_prefixes": sorted(getattr(backend, "_seen_prefixes", [])),
    }


def save_checkpoint(sim: "Simulation", directory: str | Path, *, keep: int = 5, keep_every: int = 6) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"tick_{sim.tick:06d}.json"
    _atomic_write(path, json.dumps(snapshot(sim), ensure_ascii=False))
    _atomic_write(directory / "latest", path.name)
    _prune(directory, keep, keep_every)
    return path


def list_checkpoints(directory: str | Path) -> list[tuple[int, Path]]:
    directory = Path(directory)
    if not directory.exists():
        return []
    out = []
    for p in directory.iterdir():
        m = _NAME.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def latest_checkpoint(directory: str | Path) -> Path | None:
    pointer = Path(directory) / "latest"
    if pointer.exists():
        path = Path(directory) / pointer.read_text(encoding="utf-8").strip()
        if path.exists():
            return path
    items = list_checkpoints(directory)
    return items[-1][1] if items else None


def checkpoint_at_or_before(directory: str | Path, tick: int) -> Path | None:
    best = None
    for t, p in list_checkpoints(directory):
        if t <= tick:
            best = p
    return best


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("version") != VERSION:
        raise ValueError(f"unsupported checkpoint version {data.get('version')!r}")
    return data


def apply_checkpoint(sim: "Simulation", data: dict[str, Any]) -> None:
    """Load ``data`` into a freshly wired simulation (see ``sim/scenario.py::wire_simulation``)."""
    sim.tick = int(data["tick"])
    sim._conv_counter = int(data["conv_counter"])
    sim.mind.llm.budget.spent = float(data["budget_spent"])
    sim.mind.now = sim.clock.time_at(sim.tick)
    _world_from_dict(sim.state, data["world"])
    for name, a in data["agents"].items():
        agent = sim.agents[name]
        agent.memory.load_state(a["memory"])
        agent.scratch = _scratch_from_dict(a["scratch"], agent.scratch.recent_observations.maxlen or 10)
    sim.mind.summaries.load_state(data["summaries"])
    backend = sim.mind.llm.backend
    if hasattr(backend, "_seen_prefixes"):
        backend._seen_prefixes = set(data.get("stub_seen_prefixes", []))


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _prune(directory: Path, keep: int, keep_every: int) -> None:
    items = list_checkpoints(directory)
    recent = {t for t, _ in items[-keep:]}
    for t, p in items:
        if t not in recent and (keep_every <= 0 or t % keep_every != 0):
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# (de)serialization helpers

def _dt(x: datetime | None) -> str | None:
    return x.isoformat() if x is not None else None


def _pdt(s: str | None) -> datetime | None:
    return datetime.fromisoformat(s) if s else None


def _world_to_dict(state) -> dict[str, Any]:
    positions = {}
    for name, p in state.positions.items():
        positions[name] = {"place": p.place, "area": p.area,
                           "transit": asdict(p.transit) if p.transit else None}
    return {
        "positions": positions,
        "actions": {n: asdict(a) for n, a in state.actions.items()},
        "object_states": dict(state.object_states),
        "object_users": dict(state.object_users),
    }


def _world_from_dict(state, d: dict[str, Any]) -> None:
    state.positions = {
        n: Position(place=p["place"], area=p["area"], transit=Transit(**p["transit"]) if p["transit"] else None)
        for n, p in d["positions"].items()
    }
    state.actions = {n: ActionState(**a) for n, a in d["actions"].items()}
    state.object_states = dict(d["object_states"])
    state.object_users = dict(d["object_users"])


def _step_to_dict(s: Step) -> dict[str, Any]:
    return {**asdict(s), "start": _dt(s.start)}


def _block_to_dict(b: HourBlock) -> dict[str, Any]:
    return {
        "id": b.id, "start": _dt(b.start), "end": _dt(b.end), "activity": b.activity, "place": b.place,
        "source": b.source, "steps": [_step_to_dict(s) for s in b.steps],
        "decomposed_until": _dt(b.decomposed_until), "emoji": b.emoji, "label": b.label,
    }


def _block_from_dict(d: dict[str, Any]) -> HourBlock:
    return HourBlock(
        id=d["id"], start=_pdt(d["start"]), end=_pdt(d["end"]), activity=d["activity"], place=d["place"],
        source=d["source"], steps=[Step(**{**s, "start": _pdt(s["start"])}) for s in d["steps"]],
        decomposed_until=_pdt(d["decomposed_until"]), emoji=d["emoji"], label=d["label"],
    )


def _plan_to_dict(p: DayPlan | None) -> dict[str, Any] | None:
    if p is None:
        return None
    return {
        "day": p.day.isoformat(), "wake": _dt(p.wake), "sleep": _dt(p.sleep),
        "items": [[_dt(t), d] for t, d in p.items], "blocks": [_block_to_dict(b) for b in p.blocks],
        "revision": p.revision,
    }


def _plan_from_dict(d: dict[str, Any] | None) -> DayPlan | None:
    if d is None:
        return None
    return DayPlan(
        day=date.fromisoformat(d["day"]), wake=_pdt(d["wake"]), sleep=_pdt(d["sleep"]),
        items=[(_pdt(t), desc) for t, desc in d["items"]], blocks=[_block_from_dict(b) for b in d["blocks"]],
        revision=d["revision"],
    )


def _action_to_dict(a: CurrentAction | None) -> dict[str, Any] | None:
    return None if a is None else {**asdict(a), "start": _dt(a.start)}


def _action_from_dict(d: dict[str, Any] | None) -> CurrentAction | None:
    return None if d is None else CurrentAction(**{**d, "start": _pdt(d["start"])})


def _scratch_to_dict(s: Scratch) -> dict[str, Any]:
    return {
        "day_plan": _plan_to_dict(s.day_plan),
        "last_recap": s.last_recap,
        "action": _action_to_dict(s.action),
        "engaged_until": _dt(s.engaged_until),
        "engaged_label": s.engaged_label,
        "cooldowns": {k: _dt(v) for k, v in s.cooldowns.items()},
        "recent_observations": [list(x) for x in s.recent_observations],
        "emitted_until": _dt(s.emitted_until),
        "visited": sorted(s.visited),
        "counter": s.counter,
    }


def _scratch_from_dict(d: dict[str, Any], retention: int) -> Scratch:
    return Scratch(
        day_plan=_plan_from_dict(d["day_plan"]),
        last_recap=d["last_recap"],
        action=_action_from_dict(d["action"]),
        engaged_until=_pdt(d["engaged_until"]),
        engaged_label=d["engaged_label"],
        cooldowns={k: _pdt(v) for k, v in d["cooldowns"].items()},
        recent_observations=deque((tuple(x) for x in d["recent_observations"]), maxlen=retention),
        emitted_until=_pdt(d["emitted_until"]),
        visited=set(d["visited"]),
        counter=int(d["counter"]),
    )
