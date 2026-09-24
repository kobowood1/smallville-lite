"""Recursive planning: day -> hour blocks -> 5-15 minute steps, just in time (paper §4.3, App. A)."""

from __future__ import annotations

from datetime import datetime, timedelta

from smallville_lite.cognition.plan import ensure_decomposed, plan_new_day, replan, resolve_action
from smallville_lite.memory import NewMemory

from helpers import START, FakeReply, Rig, script

DAY = START.date()


def _check_contiguous(plan):
    assert plan.blocks[0].start == plan.wake and plan.blocks[-1].end == plan.sleep
    for a, b in zip(plan.blocks, plan.blocks[1:]):
        assert a.end == b.start


def test_day_plan_to_hour_blocks():
    rig = Rig()
    isabella = rig.agents["Isabella"]
    plan = plan_new_day(rig.mind, isabella, DAY)
    assert plan.wake == datetime(2023, 2, 13, 6, 0) and plan.sleep == datetime(2023, 2, 13, 23, 0)
    assert 5 <= len(plan.items) <= 8
    _check_contiguous(plan)
    assert {b.place for b in plan.blocks} <= set(rig.mind.known_places(isabella))
    subtypes = [n.subtype for n in isabella.memory.nodes]
    assert subtypes.count("day_plan") == len(plan.items) and subtypes.count("hour_plan") == len(plan.blocks)
    assert all(n.importance >= 1 for n in isabella.memory.nodes)
    levels = [e["data"]["level"] for e in rig.events("plan_created")]
    assert levels == ["day", "hour"]
    # The day plan is conditioned on the cached summary prefix.
    (req,) = rig.requests("plan_day")
    assert req.system and "cache_control" in req.system[-1]


def test_places_are_an_enum_of_known_places():
    rig = Rig()
    plan_new_day(rig.mind, rig.agents["Sam"], DAY)
    (req,) = rig.requests("plan_hours")
    enum = req.output_schema["$defs"]["HourBlockOut"]["properties"]["place"]["enum"]
    assert enum[0] == "Moore House" and "Oak Hill Dorm" not in enum        # Sam doesn't know the dorm


def test_non_contiguous_blocks_get_one_repair():
    bad = {"blocks": [{"start": "07:00", "end": "12:00", "activity": "a", "place": "Oak Hill Dorm"},
                      {"start": "13:00", "end": "22:00", "activity": "b", "place": "Oak Hill Dorm"}]}
    good = {"blocks": [{"start": "07:00", "end": "22:00", "activity": "studying", "place": "Oak Hill Dorm"}]}
    day = {"wake_time": "07:00", "sleep_time": "22:00",
           "items": [{"time": f"{h:02d}:00", "description": f"item {h}"} for h in (7, 9, 12, 15, 19)]}
    rig = Rig(responders={"plan_day": script(day), "plan_hours": script(bad, good)})
    plan = plan_new_day(rig.mind, rig.agents["Maria"], DAY)
    assert len(plan.blocks) == 1
    reqs = rig.requests("plan_hours")
    assert len(reqs) == 2 and "contiguous" in reqs[1].messages[-1]["content"]


def test_persistent_failure_falls_back_and_is_logged():
    rig = Rig(responders={"plan_hours": script(FakeReply('{"blocks": []}'))})
    plan = plan_new_day(rig.mind, rig.agents["Maria"], DAY)
    assert len(plan.blocks) == 1 and plan.blocks[0].activity == "going about the day"
    assert rig.events("llm_fallback")[0]["data"]["task"] == "plan_hours"


def test_just_in_time_decomposition():
    rig = Rig(overrides={"planning": {"decompose_horizon_minutes": 60}})
    klaus = rig.agents["Klaus"]
    plan = plan_new_day(rig.mind, klaus, DAY)
    block = next(b for b in plan.blocks if (b.end - b.start) >= timedelta(minutes=120) and not b.is_sleep)
    assert block.steps == []                                  # nothing decomposed ahead of time
    rig.mind.now = block.start
    ensure_decomposed(rig.mind, klaus, block, block.start)
    assert block.decomposed_until == block.start + timedelta(minutes=60)
    assert sum(s.minutes for s in block.steps) == 60
    assert all(s.minutes in (5, 10, 15) for s in block.steps)
    assert all(s.emoji and s.label for s in block.steps)       # batched Haiku labels
    assert len(rig.requests("label_actions")) == 1
    ensure_decomposed(rig.mind, klaus, block, block.start + timedelta(minutes=10))
    assert len(rig.requests("plan_detail")) == 1              # not re-decomposed inside the horizon


def test_step_minutes_must_sum_to_the_chunk():
    wrong = {"steps": [{"minutes": 15, "activity": "x", "object": "none", "object_state": ""}]}
    rig = Rig(responders={"plan_detail": script(wrong)})
    klaus = rig.agents["Klaus"]
    plan = plan_new_day(rig.mind, klaus, DAY)
    block = next(b for b in plan.blocks if not b.is_sleep)
    rig.mind.now = block.start
    ensure_decomposed(rig.mind, klaus, block, block.start)
    assert sum(s.minutes for s in block.steps) == int((block.decomposed_until - block.start).total_seconds() // 60)
    assert any(e["data"]["task"] == "plan_detail" for e in rig.events("llm_fallback"))


def test_commitments_for_today_shape_the_plan():
    rig = Rig()
    sam = rig.agents["Sam"]
    sam.memory.add(START, NewMemory("plan", "commitment",
                                    "Sam Moore committed to: meet Klaus at Johnson Park on 2023-02-13 at 15:00 at Johnson Park",
                                    6, meta={"date": "2023-02-13", "time": "15:00", "place": "Johnson Park"}))
    plan = plan_new_day(rig.mind, sam, DAY)
    (req,) = rig.requests("plan_day")
    assert "meet Klaus" in req.messages[0]["content"]
    block = plan.block_at(datetime(2023, 2, 13, 15, 0))
    assert block.place == "Johnson Park"


def test_replan_keeps_the_past_and_revises_from_now():
    rig = Rig()
    maria = rig.agents["Maria"]
    plan = plan_new_day(rig.mind, maria, DAY)
    now = rig.at(13, 0)
    # Finished blocks stay; the block in progress is cut off at `now`.
    before = [(b.id, b.start, min(b.end, now)) for b in plan.blocks if b.start < now]
    assert replan(rig.mind, maria, now, cause="the stove at home is burning")
    assert [(b.id, b.start, b.end) for b in plan.blocks if b.end <= now] == before
    new = [b for b in plan.blocks if b.start >= now]
    assert new[0].start == now and new[0].source == "replan"
    _check_contiguous(plan)
    (event,) = rig.events("plan_revised")
    assert event["data"]["reason"] == "the stove at home is burning"
    assert "stove" in rig.requests("replan")[0].messages[0]["content"]


def test_resolve_action_sleep_and_steps():
    rig = Rig()
    maria = rig.agents["Maria"]
    plan = plan_new_day(rig.mind, maria, DAY)
    asleep = resolve_action(rig.mind, maria, plan.wake - timedelta(minutes=10))
    assert asleep.kind == "sleep" and asleep.place == "Oak Hill Dorm"
    assert asleep.object == "Oak Hill Dorm/Maria's room/bed"
    block = plan.blocks[1]
    rig.mind.now = block.start
    ensure_decomposed(rig.mind, maria, block, block.start)
    act = resolve_action(rig.mind, maria, block.start + timedelta(minutes=16))
    assert act.id == next(s.id for s in block.steps if s.start <= block.start + timedelta(minutes=16) < s.end)
