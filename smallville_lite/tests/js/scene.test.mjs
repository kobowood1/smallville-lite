// Unit tests for the town scene model (Phase 5). Run with: node --test tests/js
import test from "node:test";
import assert from "node:assert/strict";

import { addEvents, createIndex } from "../../src/smallville_lite/web/static/state.js";
import { buildTimeline, dialogueActive, MINUTE, placeSlot, planAt, positionAt, sceneAt, skipTarget }
  from "../../src/smallville_lite/web/static/scene.js";

const START = "2023-02-13T06:00:00";
const T = (hhmm, day = 13) => Date.parse(`2023-02-${day}T${hhmm}:00Z`);

const layout = {
  width: 400, height: 200,
  places: [
    { name: "Home", public: false, rect: [0, 0, 100, 100], door: [50, 100], areas: [],
      objects: [{ path: "Home/bed", name: "bed", at: [20, 40], icon: "🛏️", default_state: "idle" }] },
    { name: "Cafe", public: true, rect: [300, 0, 100, 100], door: [350, 100], areas: [],
      objects: [{ path: "Cafe/counter", name: "counter", at: [320, 40], icon: "🧾", default_state: "idle" }] },
  ],
};

let seq = 0;
const ev = (type, agent, tick, data, game_time = null) => ({
  seq: ++seq, run_id: "t", segment: 0, tick, type, agent, data,
  game_time: game_time ?? new Date(Date.parse(START + "Z") + (tick ?? 0) * 10 * MINUTE).toISOString().slice(0, 19),
});

function fixture() {
  seq = 0;
  const events = [
    ev("run_started", null, null, { run_id: "t", scenario: "x", stub: true, budget_usd: 1 }, START),
    ev("world_init", null, null, { start: START, tick_minutes: 10, town: "T", places: [], travel_ticks_default: 1 }, START),
    ev("agent_init", "Ann", null, { start: "Home/bed", age: 30, traits: "", bio: [], home: "Home", known_places: [] }, START),
    ev("agent_init", "Bob", null, { start: "Cafe", age: 30, traits: "", bio: [], home: "Cafe", known_places: [] }, START),
    ev("action_started", "Ann", 0, { action_id: "a0", description: "Ann is sleeping", emoji: "😴", label: "sleeping",
      place: "Home", object: "Home/bed", start: START, kind: "sleep" }),
    ev("action_started", "Bob", 0, { action_id: "b0", description: "Bob is serving", emoji: "☕", label: "serving",
      place: "Cafe", object: "Cafe/counter", start: START, kind: "planned" }),
    // Ann walks to the cafe: departs at tick 6 (07:00), arrives tick 7 (07:10)
    ev("action_started", "Ann", 6, { action_id: "a1", description: "Ann is getting coffee", emoji: "☕", label: "coffee",
      place: "Cafe", object: null, start: "2023-02-13T07:00:00", kind: "planned" }),
    ev("move_started", "Ann", 6, { from: "Home", to: "Cafe", depart_tick: 6, arrive_tick: 7 }),
    ev("move_arrived", "Ann", 7, { place: "Cafe" }),
    // a conversation at 07:20 with two lines
    ev("conversation_started", "Ann", 8, { conv_id: "c1", participants: ["Ann", "Bob"], initiator: "Ann", place: "Cafe", reason: "hi" }),
    ev("utterance", "Ann", 8, { conv_id: "c1", turn: 0, speaker: "Ann", listener: "Bob", text: "Hello Bob", end_conversation: false, retrieved_node_ids: [] }, "2023-02-13T07:20:00"),
    ev("utterance", "Bob", 8, { conv_id: "c1", turn: 1, speaker: "Bob", listener: "Ann", text: "Hi Ann, bye", end_conversation: true, retrieved_node_ids: [] }, "2023-02-13T07:21:00"),
    ev("conversation_ended", "Ann", 8, { conv_id: "c1", turns: 2, ended_by: "Bob", topic_label: "greetings", summary: "", duration_minutes: 2 }),
    ev("object_state_changed", null, 8, { object_path: "Cafe/counter", state: "busy", by: "Bob" }),
    ev("time_skipped", null, 20, { from_tick: 100, to_tick: 140, reason: "all_asleep" }),
  ];
  return addEvents(createIndex(), events);
}

test("agents start at what they are doing, not at the place centre", () => {
  const tl = buildTimeline(fixture(), layout);
  const ann = sceneAt(tl, T("06:00")).agents.find((a) => a.name === "Ann");
  assert.ok(Math.abs(ann.x - 20) < 12 && Math.abs(ann.y - 56) < 12, `Ann should start at her bed, got ${ann.x},${ann.y}`);
});

test("walking interpolates through both doors and ends at the destination", () => {
  const tl = buildTimeline(fixture(), layout);
  const keys = tl.agents.Ann.keys;
  const depart = T("07:00"), arrive = T("07:10");
  const atFromDoor = positionAt(keys, depart + 0.15 * (arrive - depart));
  assert.deepEqual([Math.round(atFromDoor.x), Math.round(atFromDoor.y)], [50, 100]);
  const atToDoor = positionAt(keys, arrive - 0.15 * (arrive - depart));
  assert.deepEqual([Math.round(atToDoor.x), Math.round(atToDoor.y)], [350, 100]);
  const mid = sceneAt(tl, T("07:05")).agents.find((a) => a.name === "Ann");
  assert.ok(mid.inTransit && mid.inTransit.to === "Cafe" && mid.emoji === "🚶");
  assert.ok(mid.x > 50 && mid.x < 350);
  const after = sceneAt(tl, T("07:15")).agents.find((a) => a.name === "Ann");
  const slot = placeSlot(layout.places[1], 0, 2);
  assert.deepEqual([after.x, after.y], [slot.x, slot.y]);
  assert.equal(after.inTransit, null);
});

test("speech bubbles: the latest line stays until the next one, partners know each other", () => {
  const tl = buildTimeline(fixture(), layout);
  let s = sceneAt(tl, T("07:20") + 30000, { bubbleMs: MINUTE });
  const ann = s.agents.find((a) => a.name === "Ann"), bob = s.agents.find((a) => a.name === "Bob");
  assert.equal(ann.speaking.text, "Hello Bob");
  assert.equal(bob.speaking, null);
  assert.equal(ann.partner.name, "Bob");
  assert.equal(s.links.length, 1);
  s = sceneAt(tl, T("07:21") + 30000, { bubbleMs: MINUTE });
  assert.equal(s.agents.find((a) => a.name === "Ann").speaking, null);
  assert.equal(s.agents.find((a) => a.name === "Bob").speaking.text, "Hi Ann, bye");
  s = sceneAt(tl, T("07:21") + 2 * MINUTE, { bubbleMs: MINUTE });
  assert.ok(s.agents.every((a) => !a.speaking));
  assert.ok(dialogueActive(tl, T("07:20")));
  assert.ok(!dialogueActive(tl, T("09:00")));
});

test("object states and night skips", () => {
  const tl = buildTimeline(fixture(), layout);
  assert.equal(sceneAt(tl, T("07:00")).objectStates["Cafe/counter"], undefined);
  assert.equal(sceneAt(tl, T("07:30")).objectStates["Cafe/counter"], "busy");
  const t0 = Date.parse(START + "Z");
  assert.equal(skipTarget(tl, t0 + 120 * 10 * MINUTE), t0 + 140 * 10 * MINUTE);
  assert.equal(skipTarget(tl, t0), null);
});

test("planAt applies revisions and detail steps", () => {
  const idx = createIndex();
  idx.plans = [
    { type: "plan_created", level: "hour", agent: "Ann", date: "2023-02-13", time: "2023-02-13T06:00:00",
      items: [{ id: "b1", start: "2023-02-13T07:00:00", end: "2023-02-13T09:00:00", description: "coffee", place: "Cafe" },
              { id: "b2", start: "2023-02-13T09:00:00", end: "2023-02-13T12:00:00", description: "work", place: "Home" }] },
    { type: "plan_created", level: "detail", agent: "Ann", date: "2023-02-13", time: "2023-02-13T07:00:00", block_id: "b1",
      items: [{ id: "s1", start: "2023-02-13T07:00:00", end: "2023-02-13T07:15:00", description: "order" }] },
    { type: "plan_revised", agent: "Ann", time: "2023-02-13T08:00:00", from_time: "2023-02-13T08:00:00", removed_ids: ["b2"],
      added: [{ id: "b3", start: "2023-02-13T08:00:00", end: "2023-02-13T12:00:00", description: "fire drill", place: "Cafe" }] },
  ];
  const before = planAt(idx, "Ann", T("07:05"));
  assert.deepEqual(before.map((b) => b.id), ["b1", "b2"]);
  assert.ok(before[0].current && before[0].steps[0].current);
  const after = planAt(idx, "Ann", T("08:30"));
  assert.deepEqual(after.map((b) => [b.id, b.end.slice(11, 16)]), [["b1", "08:00"], ["b3", "12:00"]]);
  assert.ok(after[1].current);
});
