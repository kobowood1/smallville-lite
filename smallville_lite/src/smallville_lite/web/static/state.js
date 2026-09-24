// Event-log reducer for the smallville_lite viewer (DESIGN §8.3).
// No DOM access: this module is shared by the web log (Phase 4) and the visual town (Phase 5).
//
//   const idx = createIndex();  addEvents(idx, events);   // incremental
//   reduceState(idx, tick)  -> where every agent is and what they are doing at `tick`
//   memoryAt(idx, agent, tick), reflectionForest(idx, agent), usage(idx), gameTime(idx, tick)

// Game time is naive local time in the log; treat it as UTC so Date math and toISOString() round-trip.
export function parseGameTime(s) {
  return s ? new Date(/[zZ]$|[+-]\d\d:\d\d$/.test(s) ? s : s + "Z") : null;
}

export function createIndex() {
  return {
    run: { runId: null, scenario: null, stub: null, budget: null, status: "running", reason: null, spent: 0 },
    world: null,               // world_init data
    start: null,               // Date of tick 0
    tickMinutes: 10,
    agents: {},                // name -> { init, actions:[], memory:[], byId:Map, lastAccess:Map(id -> [{tick,time}]) }
    agentNames: [],
    conversations: new Map(),  // conv_id -> { id, start, tick, participants, place, reason, initiator, turns:[], end }
    stateEvents: [],           // action_started / move_* / object_state_changed, in log order
    snapshots: [],             // state_snapshot events
    reflections: [],           // reflection events
    plans: [],                 // plan_created / plan_revised events
    llm: [],                   // llm_call data + tick + game_time
    embeds: [],
    fallbacks: [],
    budget: [],                // budget_warning / budget_exhausted
    reports: [],
    lastSeq: 0,
    maxTick: 0,
  };
}

function agentEntry(idx, name) {
  if (!idx.agents[name]) {
    idx.agents[name] = { init: null, actions: [], memory: [], byId: new Map(), lastAccess: new Map() };
    idx.agentNames = Object.keys(idx.agents).sort();
  }
  return idx.agents[name];
}

export function addEvents(idx, events) {
  for (const e of events) addEvent(idx, e);
  return idx;
}

export function addEvent(idx, e) {
  if (e.seq <= idx.lastSeq) return;
  idx.lastSeq = e.seq;
  if (typeof e.tick === "number") idx.maxTick = Math.max(idx.maxTick, e.tick);
  const d = e.data;
  switch (e.type) {
    case "run_started":
      Object.assign(idx.run, { runId: d.run_id, scenario: d.scenario, stub: d.stub, budget: d.budget_usd, status: "running" });
      break;
    case "run_resumed":
      Object.assign(idx.run, { status: "running", budget: d.budget_usd ?? idx.run.budget });
      break;
    case "run_ended":
      Object.assign(idx.run, { status: d.reason, reason: d.reason, spent: d.spent_usd });
      break;
    case "world_init":
      idx.world = d;
      idx.start = parseGameTime(d.start);
      idx.tickMinutes = d.tick_minutes;
      break;
    case "agent_init":
      agentEntry(idx, e.agent).init = d;
      break;
    case "action_started":
      agentEntry(idx, e.agent).actions.push({ ...d, tick: e.tick, time: e.game_time, agent: e.agent });
      idx.stateEvents.push(e);
      break;
    case "move_started":
    case "move_arrived":
    case "object_state_changed":
      idx.stateEvents.push(e);
      break;
    case "state_snapshot":
      idx.snapshots.push(e);
      break;
    case "memory_added": {
      const a = agentEntry(idx, e.agent);
      const node = { ...d.node, tick: e.tick };
      a.memory.push(node);
      a.byId.set(node.id, node);
      break;
    }
    case "memory_accessed": {
      const a = agentEntry(idx, e.agent);
      for (const r of d.results) {
        if (!a.lastAccess.has(r.node_id)) a.lastAccess.set(r.node_id, []);
        a.lastAccess.get(r.node_id).push({ tick: e.tick, time: e.game_time, purpose: d.purpose });
      }
      break;
    }
    case "conversation_started":
      idx.conversations.set(d.conv_id, {
        id: d.conv_id, tick: e.tick, start: e.game_time, participants: d.participants, place: d.place,
        reason: d.reason, initiator: d.initiator, turns: [], end: null,
      });
      break;
    case "utterance": {
      const c = idx.conversations.get(d.conv_id);
      if (c) c.turns.push({ ...d, time: e.game_time });
      break;
    }
    case "conversation_ended": {
      const c = idx.conversations.get(d.conv_id);
      if (c) c.end = d;
      break;
    }
    case "reflection":
      idx.reflections.push({ ...d, agent: e.agent, tick: e.tick, time: e.game_time });
      break;
    case "plan_created":
    case "plan_revised":
      idx.plans.push({ ...d, type: e.type, agent: e.agent, tick: e.tick, time: e.game_time });
      break;
    case "llm_call":
      idx.llm.push({ ...d, tick: e.tick, time: e.game_time });
      break;
    case "embed_call":
      idx.embeds.push({ ...d, tick: e.tick, time: e.game_time });
      break;
    case "llm_fallback":
      idx.fallbacks.push({ ...d, agent: e.agent, tick: e.tick, time: e.game_time });
      break;
    case "budget_warning":
    case "budget_exhausted":
      idx.budget.push({ type: e.type, ...d, tick: e.tick, time: e.game_time });
      break;
    case "report":
      idx.reports.push(d);
      break;
  }
}

export function gameTime(idx, tick) {
  if (!idx.start) return null;
  return new Date(idx.start.getTime() + tick * idx.tickMinutes * 60000);
}

// Where everyone is and what they are doing after `tick` has been committed.
export function reduceState(idx, tick) {
  let base = null;
  for (const s of idx.snapshots) {
    if (s.tick <= tick) base = s; else break;
  }
  const agents = {};
  const objects = {};
  if (base) {
    for (const [name, a] of Object.entries(base.data.agents)) agents[name] = { ...a };
    Object.assign(objects, base.data.object_states);
  } else {
    for (const name of idx.agentNames) {
      const init = idx.agents[name].init;
      agents[name] = { place: init ? init.start.split("/")[0] : null, area: null, in_transit: null,
                       action_id: null, description: null, emoji: null, label: null };
    }
  }
  const fromTick = base ? base.tick : -Infinity;
  for (const e of idx.stateEvents) {
    if (e.tick === null || e.tick <= fromTick) continue;
    if (e.tick > tick) break;
    const a = e.agent ? (agents[e.agent] ??= {}) : null;
    const d = e.data;
    if (e.type === "action_started") {
      Object.assign(a, { action_id: d.action_id, description: d.description, emoji: d.emoji, label: d.label,
                         kind: d.kind, object: d.object, action_place: d.place });
    } else if (e.type === "move_started") {
      a.in_transit = { from: d.from, to: d.to, depart_tick: d.depart_tick, arrive_tick: d.arrive_tick };
    } else if (e.type === "move_arrived") {
      Object.assign(a, { place: d.place, area: null, in_transit: null });
    } else if (e.type === "object_state_changed") {
      objects[d.object_path] = d.state;
    }
  }
  return { tick, time: gameTime(idx, tick), agents, objects };
}

export function memoryAt(idx, agent, tick = Infinity) {
  const a = idx.agents[agent];
  if (!a) return [];
  return a.memory.filter((n) => n.tick === null || n.tick === undefined || n.tick <= tick);
}

export function lastAccessed(idx, agent, nodeId, tick = Infinity) {
  const a = idx.agents[agent];
  const list = a ? a.lastAccess.get(nodeId) || [] : [];
  let last = null;
  for (const r of list) if (r.tick === null || r.tick <= tick) last = r;
  return last;
}

// Reflection trees (paper Fig. 7): roots are reflections no other reflection cites.
export function reflectionForest(idx, agent) {
  const a = idx.agents[agent];
  if (!a) return [];
  const reflections = a.memory.filter((n) => n.type === "reflection");
  const cited = new Set();
  for (const r of reflections) for (const ev of r.evidence) {
    const t = a.byId.get(ev);
    if (t && t.type === "reflection") cited.add(ev);
  }
  const build = (node, seen) => ({
    node,
    children: seen.has(node.id) ? [] : node.evidence.map((id) => a.byId.get(id)).filter(Boolean)
      .map((child) => build(child, new Set([...seen, node.id]))),
  });
  return reflections.filter((r) => !cited.has(r.id)).reverse().map((r) => build(r, new Set()));
}

export function usage(idx) {
  const total = { calls: 0, failed: 0, cost: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0 };
  const byTask = {}, byModel = {};
  const series = [];
  let cumulative = 0;
  const add = (bucket, d) => {
    bucket.calls += 1;
    bucket.failed += d.ok === false ? 1 : 0;
    bucket.cost += d.cost_usd || 0;
    bucket.input += d.input_tokens || 0;
    bucket.output += d.output_tokens || 0;
    bucket.cacheRead += d.cache_read_tokens || 0;
    bucket.cacheWrite += d.cache_write_tokens || 0;
  };
  const blank = () => ({ calls: 0, failed: 0, cost: 0, input: 0, output: 0, cacheRead: 0, cacheWrite: 0 });
  for (const d of idx.llm) {
    add(total, d);
    add(byTask[d.task] ??= blank(), d);
    add(byModel[d.model] ??= blank(), d);
    cumulative += d.cost_usd || 0;
    if (d.time) series.push({ time: parseGameTime(d.time), cost: cumulative });
  }
  const embedCost = idx.embeds.reduce((s, d) => s + (d.cost_usd || 0), 0);
  return { total, byTask, byModel, series, embedCost };
}

export function hitRate(b) {
  const denom = b.cacheRead + b.cacheWrite + b.input;
  return denom ? b.cacheRead / denom : 0;
}

export function conversationsFor(idx, agent = null) {
  const all = [...idx.conversations.values()];
  return agent ? all.filter((c) => c.participants.includes(agent)) : all;
}

export function commitmentsFor(idx, convId) {
  const out = [];
  for (const name of idx.agentNames) {
    for (const n of idx.agents[name].memory) {
      if (n.subtype === "commitment" && n.meta && n.meta.conv_id === convId) out.push({ agent: name, ...n });
    }
  }
  return out;
}
