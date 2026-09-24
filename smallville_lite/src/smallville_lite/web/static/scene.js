// Scene model for the visual town (Phase 5). Pure functions, no DOM: shared by every renderer
// and unit-tested under Node (tests/js/scene.test.mjs).
//
//   const tl = buildTimeline(idx, layout);      // from the event index (state.js) + map layout
//   const scene = sceneAt(tl, tMs, { bubbleMs }); // everything a renderer needs at game time tMs
//
// Times are milliseconds on the game clock (game time parsed as UTC, see state.js).

import { parseGameTime } from "./state.js";

export const MINUTE = 60000;
const WALK_IN_PLACE_MS = 2 * MINUTE;      // tween when moving between objects inside a place
const DOOR_FRACTION = 0.15;              // share of a trip spent walking to / from the doors

// ---------------------------------------------------------------------------
// building

export function buildTimeline(idx, layout) {
  const t0 = idx.start ? idx.start.getTime() : 0;
  const tickMs = (idx.tickMinutes || 10) * MINUTE;
  const tickTime = (tick) => t0 + tick * tickMs;
  const places = new Map(layout.places.map((p) => [p.name, p]));
  const objects = new Map();
  for (const p of layout.places) for (const o of p.objects) objects.set(o.path, { ...o, place: p.name });
  const names = [...idx.agentNames];

  const tl = {
    t0, tickMs, tMax: tickTime(idx.maxTick), agentNames: names, layout, places, objects,
    agents: {}, objectStates: {}, conversations: [], skips: [],
  };

  const slot = (name, place) => placeSlot(places.get(place), names.indexOf(name), names.length);
  const spotFor = (name, place, objectPath) => {
    const obj = objectPath ? objects.get(objectPath) : null;
    if (obj && obj.place === place) {
      const k = names.indexOf(name);
      return { x: obj.at[0] + ((k % 2) * 2 - 1) * 9, y: obj.at[1] + 16 + Math.floor(k / 2) * 4 };
    }
    return slot(name, place);
  };

  for (const name of names) {
    const a = idx.agents[name];
    const startPlace = a.init ? a.init.start.split("/")[0] : layout.places[0]?.name;
    const keys = [];
    const transits = [];
    let cur = spotFor(name, startPlace, null);
    let place = startPlace;
    let transitUntil = -Infinity;
    let pendingObject = null;
    const push = (t, p) => {
      const last = keys[keys.length - 1];
      keys.push({ t: last ? Math.max(t, last.t) : t, x: p.x, y: p.y });
    };
    push(t0, cur);

    // Merge this agent's actions and moves in log order.
    const events = [];
    for (const act of a.actions) events.push({ kind: "act", seq: act.seq ?? 0, t: parseGameTime(act.time).getTime(), act });
    for (const e of idx.stateEvents) {
      if (e.agent === name && e.type === "move_started") events.push({ kind: "move", seq: e.seq, t: tickTime(e.tick), e });
    }
    events.sort((x, y) => x.seq - y.seq);

    for (const ev of events) {
      if (ev.kind === "act") {
        const act = ev.act;
        if (ev.t < transitUntil || act.place !== place) { pendingObject = act.object; continue; }
        const next = spotFor(name, place, act.object);
        if (keys.length === 1 && ev.t <= t0) {        // what they are doing at the start: just be there
          cur = next;
          keys[0] = { t: t0, x: next.x, y: next.y };
          continue;
        }
        if (next.x !== cur.x || next.y !== cur.y) {
          push(ev.t, cur);
          push(ev.t + WALK_IN_PLACE_MS, next);
          cur = next;
        }
      } else {
        const d = ev.e.data;
        const tArr = tickTime(d.arrive_tick);
        const from = places.get(d.from), to = places.get(d.to);
        const dest = spotFor(name, d.to, pendingObject);
        const span = Math.max(1, tArr - ev.t);
        push(ev.t, cur);
        if (from) push(ev.t + span * DOOR_FRACTION, { x: from.door[0], y: from.door[1] });
        if (to) push(tArr - span * DOOR_FRACTION, { x: to.door[0], y: to.door[1] });
        push(tArr, dest);
        transits.push({ t0: ev.t, t1: tArr, from: d.from, to: d.to });
        cur = dest; place = d.to; transitUntil = tArr; pendingObject = null;
      }
    }
    const actions = a.actions
      .map((act) => ({ t: parseGameTime(act.time).getTime(), emoji: act.emoji, label: act.label,
                       description: act.description, place: act.place, kind: act.kind, object: act.object }))
      .sort((x, y) => x.t - y.t);
    tl.agents[name] = { keys, transits, actions };
  }

  for (const e of idx.stateEvents) {
    if (e.type !== "object_state_changed") continue;
    (tl.objectStates[e.data.object_path] ??= []).push({ t: tickTime(e.tick), state: e.data.state });
  }

  for (const c of idx.conversations.values()) {
    const start = parseGameTime(c.start).getTime();
    const turns = c.turns.map((u) => ({ t: parseGameTime(u.time).getTime(), speaker: u.speaker, text: u.text }));
    const lastTurn = turns.length ? turns[turns.length - 1].t : start;
    const end = c.end ? start + Math.max(c.end.duration_minutes, 1) * MINUTE : lastTurn + MINUTE;
    tl.conversations.push({ id: c.id, t0: start, t1: Math.max(end, lastTurn + MINUTE), participants: c.participants,
                            place: c.place, reason: c.reason, topic: c.end?.topic_label || null, turns });
  }
  tl.conversations.sort((x, y) => x.t0 - y.t0);

  for (const s of idx.skips || []) tl.skips.push({ t0: tickTime(s.from_tick), t1: tickTime(s.to_tick) });
  return tl;
}

// Where an agent stands in a place when not using a specific object: a small grid around the centre.
export function placeSlot(place, k, n) {
  if (!place) return { x: 0, y: 0 };
  const [x, y, w, h] = place.rect;
  const cx = x + w / 2, cy = y + 26 + (h - 26) / 2;
  const cols = Math.min(4, Math.max(1, n));
  const col = k % cols, row = Math.floor(k / cols);
  return { x: cx + (col - (cols - 1) / 2) * 34, y: cy + row * 34 - 10 };
}

// ---------------------------------------------------------------------------
// querying

function lastAtOrBefore(list, t, key = "t") {
  let lo = 0, hi = list.length - 1, best = -1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (list[mid][key] <= t) { best = mid; lo = mid + 1; } else hi = mid - 1;
  }
  return best;
}

export function positionAt(keys, t) {
  const i = lastAtOrBefore(keys, t);
  if (i < 0) return { x: keys[0].x, y: keys[0].y };
  const a = keys[i], b = keys[i + 1];
  if (!b || b.t <= a.t) return { x: a.x, y: a.y };
  const f = Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t)));
  return { x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f };
}

export function sceneAt(tl, t, opts = {}) {
  const bubbleMs = opts.bubbleMs ?? MINUTE;
  const active = tl.conversations.filter((c) => c.t0 <= t && t < c.t1);
  const speaking = new Map();
  for (const c of active) {
    // Only the most recent line of a conversation is on screen.
    // A line stays up until the next one; the last line stays for bubbleMs.
    const i = lastAtOrBefore(c.turns, t);
    if (i < 0) continue;
    const until = c.turns[i + 1] ? c.turns[i + 1].t : c.turns[i].t + bubbleMs;
    if (t < until) speaking.set(c.turns[i].speaker, { text: c.turns[i].text, conv: c.id });
  }
  const agents = tl.agentNames.map((name) => {
    const a = tl.agents[name];
    const pos = positionAt(a.keys, t);
    const tr = a.transits[lastAtOrBefore(a.transits, t, "t0")];
    const inTransit = tr && t < tr.t1 ? { from: tr.from, to: tr.to, progress: (t - tr.t0) / Math.max(1, tr.t1 - tr.t0) } : null;
    const act = a.actions[lastAtOrBefore(a.actions, t)] || null;
    const conv = active.find((c) => c.participants.includes(name));
    return {
      name, x: pos.x, y: pos.y, inTransit,
      emoji: inTransit ? "🚶" : conv ? "💬" : act?.emoji || "·",
      label: inTransit ? `walking to ${inTransit.to}` : act?.label || "",
      description: inTransit ? `${name} is on the way to ${inTransit.to}` : act?.description || "",
      place: inTransit ? null : act?.place ?? null,
      speaking: speaking.get(name) || null,
      conv: conv ? conv.id : null,
    };
  });
  const objectStates = {};
  for (const [path, list] of Object.entries(tl.objectStates)) {
    const i = lastAtOrBefore(list, t);
    if (i >= 0) objectStates[path] = list[i].state;
  }
  // Each speaker knows where their partner stands, so a renderer can keep bubbles off the partner.
  const byName = new Map(agents.map((a) => [a.name, a]));
  for (const c of active) {
    for (const name of c.participants) {
      const other = c.participants.find((n) => n !== name);
      const me = byName.get(name), them = byName.get(other);
      if (me && them) me.partner = { name: other, x: them.x, y: them.y };
    }
  }
  const links = active.map((c) => ({ conv: c.id, participants: c.participants }));
  return { t, agents, objectStates, links, conversations: active };
}

// True while dialogue lines are being spoken at t (playback slows down so they can be read).
export function dialogueActive(tl, t) {
  return tl.conversations.some((c) => c.turns.length && c.t0 <= t && t <= c.turns[c.turns.length - 1].t + MINUTE);
}

export function skipTarget(tl, t) {
  const s = tl.skips.find((x) => x.t0 <= t && t < x.t1);
  return s ? s.t1 : null;
}

// ---------------------------------------------------------------------------
// plans (for the agent panel): hour blocks for the day of t, with revisions and detail steps applied

export function planAt(idx, agent, t) {
  const day = new Date(t).toISOString().slice(0, 10);
  let blocks = null;
  const steps = new Map();
  for (const p of idx.plans) {
    if (p.agent !== agent || !p.time || parseGameTime(p.time).getTime() > t) continue;
    if (p.type === "plan_created" && p.level === "hour" && p.date === day) {
      blocks = p.items.map((b) => ({ ...b }));
      steps.clear();
    } else if (p.type === "plan_created" && p.level === "detail" && blocks) {
      const list = steps.get(p.block_id) || [];
      steps.set(p.block_id, list.concat(p.items));
    } else if (p.type === "plan_revised" && blocks && p.from_time.slice(0, 10) === day) {
      const cut = p.from_time;
      blocks = blocks.filter((b) => !p.removed_ids.includes(b.id));
      for (const b of blocks) if (b.start < cut && b.end > cut) b.end = cut;
      blocks = blocks.concat(p.added.map((b) => ({ ...b })));
    }
  }
  if (!blocks) return null;
  const iso = new Date(t).toISOString().slice(0, 19);
  blocks.sort((a, b) => a.start.localeCompare(b.start));
  return blocks.map((b) => ({
    ...b,
    current: b.start <= iso && iso < b.end,
    steps: (steps.get(b.id) || []).filter((s) => s.start < b.end)
      .map((s) => ({ ...s, current: s.start <= iso && iso < s.end })),
  }));
}
