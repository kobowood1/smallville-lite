"""Town map layout (Phase 5)."""

from __future__ import annotations

import pytest

from smallville_lite.sim.scenario import load_scenario
from smallville_lite.web.layout import LayoutError, build_layout


def _inside(pt, rect):
    x, y, w, h = rect
    return x <= pt[0] <= x + w and y <= pt[1] <= y + h


def test_party_layout_covers_every_place_and_object():
    scenario = load_scenario("party")
    layout = build_layout(scenario.world, scenario.directory / "layout.toml")
    assert layout["source"] == "layout.toml"
    assert [p["name"] for p in layout["places"]] == [p.name for p in scenario.world.places]
    n_objects = sum(len(p["objects"]) for p in layout["places"])
    assert n_objects == sum(len(scenario.world.objects_in(p.name)) for p in scenario.world.places)
    for p in layout["places"]:
        assert _inside(p["door"], [p["rect"][0] - 1, p["rect"][1] - 1, p["rect"][2] + 2, p["rect"][3] + 2])
        for a in p["areas"]:
            assert _inside(a["rect"][:2], p["rect"])
        for o in p["objects"]:
            area = next((a for a in p["areas"] if o["path"].startswith(a["path"] + "/")), None)
            assert _inside(o["at"], area["rect"] if area else p["rect"]), o["path"]
            assert o["icon"]


def test_auto_layout_without_a_file(tmp_path):
    world = load_scenario("party").world
    layout = build_layout(world, tmp_path / "missing.toml")
    assert layout["source"] == "auto"
    rects = [p["rect"] for p in layout["places"]]
    for i, a in enumerate(rects):                      # places do not overlap
        for b in rects[i + 1:]:
            assert a[0] + a[2] <= b[0] or b[0] + b[2] <= a[0] or a[1] + a[3] <= b[1] or b[1] + b[3] <= a[1]


def test_bad_layouts_are_rejected(tmp_path):
    world = load_scenario("party").world
    bad = tmp_path / "layout.toml"
    bad.write_text('[places."Nowhere"]\nrect = [0, 0, 10, 10]\n', encoding="utf-8")
    with pytest.raises(LayoutError, match="unknown places"):
        build_layout(world, bad)
    bad.write_text('[places."Hobbs Cafe"]\nrect = [0, 0, 10]\n', encoding="utf-8")
    with pytest.raises(LayoutError, match=r"\[x, y, w, h\]"):
        build_layout(world, bad)
