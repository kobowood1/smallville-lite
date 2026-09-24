"""Location tree (paper §5.1): town -> places -> optional areas -> objects.

Paths are '/'-joined names under the town, e.g. ``"Hobbs Cafe/cafe floor/espresso machine"``.
Only top-level places cost travel time; moving between areas or objects inside a place is instant.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Kind = Literal["town", "place", "area", "object"]


@dataclass
class LocationNode:
    name: str
    kind: Kind
    path: str
    children: list["LocationNode"] = field(default_factory=list)
    default_state: str | None = None     # objects only
    public: bool = False                 # places only
    description: str = ""

    def walk(self) -> Iterator["LocationNode"]:
        yield self
        for child in self.children:
            yield from child.walk()


class World:
    def __init__(self, town: LocationNode, travel_default: int = 1, travel: dict[frozenset[str], int] | None = None) -> None:
        self.town = town
        self.travel_default = travel_default
        self.travel = travel or {}
        self.nodes: dict[str, LocationNode] = {n.path: n for n in town.walk() if n.kind != "town"}

    # -- structure -----------------------------------------------------
    @property
    def name(self) -> str:
        return self.town.name

    @property
    def places(self) -> list[LocationNode]:
        return list(self.town.children)

    def place(self, name: str) -> LocationNode:
        node = self.nodes.get(name)
        if node is None or node.kind != "place":
            raise KeyError(f"unknown place {name!r}")
        return node

    def place_of(self, path: str) -> str:
        return path.split("/")[0]

    def objects_in(self, place: str) -> list[LocationNode]:
        return [n for n in self.place(place).walk() if n.kind == "object"]

    def areas_in(self, place: str) -> list[LocationNode]:
        return [n for n in self.place(place).walk() if n.kind == "area"]

    def public_places(self) -> list[str]:
        return [p.name for p in self.places if p.public]

    def travel_ticks(self, a: str, b: str) -> int:
        if a == b:
            return 0
        return self.travel.get(frozenset((a, b)), self.travel_default)

    def default_states(self) -> dict[str, str]:
        return {n.path: n.default_state or "idle" for n in self.nodes.values() if n.kind == "object"}

    # -- natural language (§5.1) -----------------------------------------
    def describe_place(self, place: str, states: dict[str, str] | None = None, with_states: bool = False) -> str:
        """'Hobbs Cafe: cafe floor (counter, espresso machine); Isabella's apartment (bed, desk)'."""
        node = self.place(place)
        parts: list[str] = []
        loose = [c for c in node.children if c.kind == "object"]
        if loose:
            parts.append(", ".join(self._obj(o, states, with_states) for o in loose))
        for area in (c for c in node.children if c.kind == "area"):
            objs = [c for c in area.walk() if c.kind == "object"]
            inner = ", ".join(self._obj(o, states, with_states) for o in objs)
            parts.append(f"{area.name} ({inner})" if inner else area.name)
        text = f"{node.name}: " + "; ".join(parts) if parts else node.name
        return text + (f". {node.description}" if node.description else "")

    def render_known(self, places: Iterable[str]) -> str:
        return "\n".join(f"- {self.describe_place(p)}" for p in places)

    @staticmethod
    def _obj(o: LocationNode, states: dict[str, str] | None, with_states: bool) -> str:
        if with_states and states is not None:
            state = states.get(o.path, o.default_state or "idle")
            if state != (o.default_state or "idle"):
                return f"{o.name} ({state})"
        return o.name


def load_world(path: str | Path) -> World:
    with Path(path).open("rb") as fh:
        raw = tomllib.load(fh)
    return world_from_dict(raw)


def world_from_dict(raw: dict[str, Any]) -> World:
    town_raw = raw.get("town", {})
    town = LocationNode(name=town_raw.get("name", "Town"), kind="town", path="")
    seen: set[str] = set()
    for p in raw.get("places", []):
        pname = p["name"]
        if "/" in pname or pname in seen:
            raise ValueError(f"invalid or duplicate place name {pname!r}")
        seen.add(pname)
        place = LocationNode(name=pname, kind="place", path=pname, public=bool(p.get("public", False)),
                             description=p.get("description", ""))
        place.children.extend(_objects(p.get("objects", []), pname))
        for a in p.get("areas", []):
            apath = f"{pname}/{a['name']}"
            area = LocationNode(name=a["name"], kind="area", path=apath, description=a.get("description", ""))
            area.children.extend(_objects(a.get("objects", []), apath))
            place.children.append(area)
        town.children.append(place)
    travel = {}
    for key, ticks in raw.get("travel", {}).items():
        a, _, b = key.partition("|")
        if a not in seen or b not in seen:
            raise ValueError(f"travel override {key!r} names an unknown place")
        travel[frozenset((a, b))] = int(ticks)
    return World(town, travel_default=int(town_raw.get("travel_ticks_default", 1)), travel=travel)


def _objects(items: list[Any], parent: str) -> list[LocationNode]:
    out = []
    for item in items:
        name, state = (item, "idle") if isinstance(item, str) else (item["name"], item.get("state", "idle"))
        out.append(LocationNode(name=name, kind="object", path=f"{parent}/{name}", default_state=state))
    return out
