"""Map layout for the visual town (Phase 5): where each place, area and object is drawn.

A scenario may ship ``layout.toml``; anything it leaves out is placed automatically, and a
scenario without one gets a full automatic layout. The layout is viewer-only data: the
simulation never reads it.

layout.toml::

    width = 1000
    height = 700
    [places."Hobbs Cafe"]
    rect = [40, 40, 320, 250]          # x, y, w, h
    door = [200, 290]                  # where walking paths start and end (default: bottom centre)
    [places."Hobbs Cafe".areas."cafe floor"]
    rect = [50, 80, 190, 200]
    [objects."Hobbs Cafe/cafe floor/counter"]
    at = [90, 120]
    [icons]                            # by object name
    bed = "🛏️"
"""

from __future__ import annotations

import math
import tomllib
from pathlib import Path
from typing import Any

from ..world import LocationNode, World

LABEL_BAND = 26          # px reserved at the top of a place / area for its label
PAD = 8

DEFAULT_ICONS = {
    "bed": "🛏️", "desk": "🗄️", "kitchenette": "🍳", "kitchen": "🍳", "stove": "🔥", "fridge": "🧊", "counter": "🧾",
    "espresso machine": "☕", "tables": "🍽️", "bulletin board": "📌", "garden beds": "🌱", "bench": "🪑",
    "pond path": "🦆", "shelves": "🥫", "checkout counter": "💳", "community notice board": "📋",
    "reading tables": "📖", "bookshelves": "📚", "bookshelf": "📚", "study carrels": "💡", "sofa": "🛋️", "TV": "📺",
    "armchair": "🪑",
}


class LayoutError(ValueError):
    pass


def build_layout(world: World, path: str | Path | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path is not None and Path(path).exists():
        with Path(path).open("rb") as fh:
            raw = tomllib.load(fh)
    places_raw = raw.get("places", {})
    unknown = set(places_raw) - {p.name for p in world.places}
    if unknown:
        raise LayoutError(f"layout names unknown places: {sorted(unknown)}")
    icons = {**DEFAULT_ICONS, **raw.get("icons", {})}
    objects_raw = raw.get("objects", {})
    bad = [p for p in objects_raw if p not in world.nodes or world.nodes[p].kind != "object"]
    if bad:
        raise LayoutError(f"layout names unknown objects: {bad}")

    width = int(raw.get("width", 1000))
    auto_rects = _auto_grid(len(world.places), width)
    height = int(raw.get("height", max(r[1] + r[3] for r in auto_rects) + 40 if auto_rects else 600))

    places = []
    for i, place in enumerate(world.places):
        praw = places_raw.get(place.name, {})
        rect = _rect(praw.get("rect"), auto_rects[i], f"places.{place.name}.rect")
        x, y, w, h = rect
        door = list(praw.get("door", [x + w / 2, y + h]))
        areas_raw = praw.get("areas", {})
        area_nodes = [c for c in place.children if c.kind == "area"]
        unknown_areas = set(areas_raw) - {a.name for a in area_nodes}
        if unknown_areas:
            raise LayoutError(f"layout names unknown areas of {place.name}: {sorted(unknown_areas)}")
        loose = [c for c in place.children if c.kind == "object"]
        inner = [x + PAD, y + LABEL_BAND, w - 2 * PAD, h - LABEL_BAND - PAD]
        loose_band = None
        if area_nodes and loose:
            band_h = min(60, inner[3] // 3)
            loose_band = [inner[0], inner[1] + inner[3] - band_h, inner[2], band_h]
            inner = [inner[0], inner[1], inner[2], inner[3] - band_h - PAD]
        area_auto = _split(inner, len(area_nodes))
        areas, objects = [], []
        for j, area in enumerate(area_nodes):
            arect = _rect(areas_raw.get(area.name, {}).get("rect"), area_auto[j], f"{area.path}.rect")
            areas.append({"name": area.name, "path": area.path, "rect": arect})
            a_inner = [arect[0] + PAD, arect[1] + LABEL_BAND - 6, arect[2] - 2 * PAD, arect[3] - LABEL_BAND]
            objects += _place_objects([c for c in area.walk() if c.kind == "object"], a_inner, objects_raw, icons)
        objects += _place_objects(loose, loose_band or inner, objects_raw, icons)
        places.append({"name": place.name, "public": place.public, "description": place.description,
                       "rect": rect, "door": door, "areas": areas, "objects": objects})
    return {"width": width, "height": height, "places": places, "source": "layout.toml" if raw else "auto"}


def _rect(value: Any, default: list[float], where: str) -> list[float]:
    if value is None:
        return [round(v, 1) for v in default]
    if not (isinstance(value, list) and len(value) == 4 and all(isinstance(v, (int, float)) for v in value)):
        raise LayoutError(f"{where} must be [x, y, w, h]")
    if value[2] <= 0 or value[3] <= 0:
        raise LayoutError(f"{where} needs a positive width and height")
    return [float(v) for v in value]


def _auto_grid(n: int, width: int) -> list[list[float]]:
    cols = max(1, math.ceil(math.sqrt(n * 1.5)))
    gap = 30
    w = (width - gap * (cols + 1)) / cols
    h = w * 0.72
    return [[gap + (i % cols) * (w + gap), gap + (i // cols) * (h + gap), w, h] for i in range(n)]


def _split(rect: list[float], n: int) -> list[list[float]]:
    """Split a rect into n side-by-side columns (or a 2-row grid when there are many)."""
    if n <= 0:
        return []
    x, y, w, h = rect
    rows = 1 if n <= 3 else 2
    cols = math.ceil(n / rows)
    cw, ch = (w - PAD * (cols - 1)) / cols, (h - PAD * (rows - 1)) / rows
    return [[x + (i % cols) * (cw + PAD), y + (i // cols) * (ch + PAD), cw, ch] for i in range(n)]


def _place_objects(nodes: list[LocationNode], rect: list[float], explicit: dict[str, Any], icons: dict[str, str]) -> list[dict]:
    out = []
    n = len(nodes)
    if not n:
        return out
    x, y, w, h = rect
    cols = max(1, min(n, math.ceil(math.sqrt(n * w / max(h, 1)))))
    rows = math.ceil(n / cols)
    for i, node in enumerate(nodes):
        auto = [x + w * ((i % cols) + 0.5) / cols, y + h * ((i // cols) + 0.5) / rows]
        at = explicit.get(node.path, {}).get("at", auto)
        out.append({"path": node.path, "name": node.name, "at": [round(float(at[0]), 1), round(float(at[1]), 1)],
                    "icon": icons.get(node.name, "▫️"), "default_state": node.default_state or "idle"})
    return out
