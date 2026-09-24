from __future__ import annotations

from datetime import datetime

import pytest

from smallville_lite.clock import GameClock, parse_hhmm
from smallville_lite.sim.scenario import load_scenario, render_rules
from smallville_lite.world import WorldState


@pytest.fixture(scope="module")
def world():
    return load_scenario("party").world


def test_tree_structure(world):
    assert [p.name for p in world.places] == [
        "Hobbs Cafe", "Johnson Park", "The Willows Market", "Oak Hill College Library", "Oak Hill Dorm", "Moore House"]
    assert world.public_places() == ["Hobbs Cafe", "Johnson Park", "The Willows Market", "Oak Hill College Library"]
    beds = [o.path for o in world.objects_in("Oak Hill Dorm") if o.name == "bed"]
    assert beds == ["Oak Hill Dorm/Maria's room/bed", "Oak Hill Dorm/Klaus's room/bed"]
    assert world.travel_ticks("Hobbs Cafe", "Johnson Park") == 1
    assert world.travel_ticks("Hobbs Cafe", "Hobbs Cafe") == 0


def test_natural_language_rendering(world):
    # Paper §5.1: the tree is flattened into natural language for prompts.
    text = world.describe_place("Hobbs Cafe")
    assert text.startswith("Hobbs Cafe: cafe floor (counter, espresso machine, tables, bulletin board); "
                           "Isabella's apartment (bed, desk, kitchenette)")
    state = WorldState.initial(world)
    state.set_object_state("Hobbs Cafe/cafe floor/espresso machine", "brewing coffee", "Isabella Rodriguez")
    assert "espresso machine (brewing coffee)" in world.describe_place("Hobbs Cafe", state.object_states, with_states=True)
    assert "Moore House" in render_rules(world)


def test_object_state_resets_to_default(world):
    state = WorldState.initial(world)
    path = "Moore House/kitchen/stove"
    assert state.set_object_state(path, "burning", None) is True
    assert state.set_object_state(path, None, None) is True
    assert state.object_states[path] == "off"


def test_clock():
    clock = GameClock(datetime(2023, 2, 13, 6, 0), 10)
    assert clock.time_at(6) == datetime(2023, 2, 13, 7, 0)
    assert clock.tick_at(datetime(2023, 2, 13, 7, 9)) == 6
    assert clock.ticks_for(1) == 1 and clock.ticks_for(10) == 1 and clock.ticks_for(11) == 2
    d = datetime(2023, 2, 13).date()
    assert parse_hhmm("24:00", d) == datetime(2023, 2, 14, 0, 0)
    with pytest.raises(ValueError):
        parse_hhmm("7pm", d)
