// Visual town controller (Phase 5): playback clock, renderer, side panel.
// Reads only the event index (state.js) and the map layout; never touches the simulation.

import { buildTimeline, dialogueActive, MINUTE, planAt, sceneAt, skipTarget } from "./scene.js";
import { createRenderer } from "./renderer.js";
import { memoryAt } from "./state.js";

const READ_MS_PER_LINE = 3000;   // "slow down for dialogue": one line per ~3 real seconds

function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === null || v === undefined || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style") el.style.cssText = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
const hhmm = (t) => new Date(t).toISOString().slice(11, 16);
const stamp = (t) => new Date(t).toISOString().slice(5, 16).replace("T", " ");

function readPref(key, fallback) {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function writePref(key, value) {
  try { localStorage.setItem(key, value); } catch { /* private mode: not remembered */ }
}

export function agentColors(names) {
  const css = getComputedStyle(document.documentElement);
  const out = {};
  names.forEach((n, i) => {
    const slot = i < 8 ? i : "other";
    out[n] = { fill: css.getPropertyValue(`--agent-${slot}`).trim() || "#888",
               ink: css.getPropertyValue(`--agent-ink-${slot}`).trim() || "#fff" };
  });
  return out;
}

export class TownView {
  constructor({ els, onTimeChange = () => {}, onOpenMemory = () => {}, onOpenDialogue = () => {} }) {
    this.els = els;
    this.onTimeChange = onTimeChange;
    this.onOpenMemory = onOpenMemory;
    this.onOpenDialogue = onOpenDialogue;
    this.idx = null; this.layout = null; this.tl = null; this.renderer = null;
    this.t = 0; this.playing = false; this.visible = false; this.selected = null;
    this.lastFrame = null; this.lastPanel = 0; this.frameReq = null;
    this.kind = readPref("townRenderer", "three");
    if (els.renderer) {
      els.renderer.value = this.kind;
      els.renderer.addEventListener("change", async () => {
        this.kind = els.renderer.value;
        writePref("townRenderer", this.kind);
        await this.mountRenderer();
        this.draw(true);
      });
    }
    els.play.addEventListener("click", () => (this.playing ? this.pause() : this.play()));
    els.scrub.addEventListener("input", () => {
      if (!this.tl) return;                               // run still loading
      const f = +els.scrub.value / 1000;
      this.seek(this.tl.t0 + f * (this.tl.tMax - this.tl.t0));
      els.live.checked = false;
    });
    els.speed.addEventListener("change", () => this.draw(true));
    els.live.addEventListener("change", () => { if (els.live.checked && this.tl) { this.seek(this.tl.tMax); this.play(); } });
  }

  async open(idx, layout, { live }) {
    this.close();
    this.idx = idx; this.layout = layout;
    this.tl = buildTimeline(idx, layout);
    await this.mountRenderer();
    this.els.live.checked = !!live;
    this.t = live ? this.tl.tMax : this.tl.t0;
    if (live) this.play(); else this.pause();
    this.draw(true);
  }

  close() {
    this.pause();
    this.renderer?.destroy();
    this.renderer = null;
  }

  // WebGL first; if three.js cannot load (offline CDN) or WebGL is missing, fall back to SVG.
  async mountRenderer() {
    this.renderer?.destroy();
    this.renderer = null;
    const opts = { agents: this.tl.agentNames, colors: agentColors(this.tl.agentNames) };
    let kind = this.kind;
    try {
      this.renderer = await createRenderer(kind);
      this.renderer.mount(this.els.map, this.layout, opts);
    } catch (err) {
      console.warn(`town renderer "${kind}" unavailable, using svg:`, err);
      this.renderer?.destroy?.();
      kind = "svg";
      this.renderer = await createRenderer("svg");
      this.renderer.mount(this.els.map, this.layout, opts);
      if (this.els.notice) this.els.notice.textContent = "3D view unavailable here (needs WebGL and the three.js CDN); showing 2D.";
    }
    if (this.els.renderer) this.els.renderer.value = kind;
    this.renderer.onAgentClick((name) => this.select(name === this.selected ? null : name));
    this.renderer.setSelected(this.selected);
  }

  // New events arrived (live mode): rebuild the timeline, keep the playhead.
  refresh(idx) {
    if (!this.layout) return;
    this.idx = idx;
    this.tl = buildTimeline(idx, this.layout);
    if (this.els.live.checked && !this.playing) this.play();
    this.draw(true);
  }

  setVisible(v) {
    this.visible = v;
    if (v) { this.renderer?.resize(); this.draw(true); if (this.playing) this.loop(); }
    else if (this.frameReq) { cancelAnimationFrame(this.frameReq); this.frameReq = null; }
  }

  seek(t) {
    if (!this.tl) return;
    this.t = Math.min(this.tl.tMax, Math.max(this.tl.t0, t));
    this.draw(true);
  }

  seekTick(tick) { if (this.tl) this.seek(this.tl.t0 + tick * this.tl.tickMs); }

  play() {
    if (!this.tl) return;
    if (this.t >= this.tl.tMax && !this.els.live.checked) this.t = this.tl.t0;
    this.playing = true;
    this.els.play.textContent = "❚❚";
    this.els.play.setAttribute("aria-label", "Pause");
    this.lastFrame = null;
    if (this.visible) this.loop();
  }

  pause() {
    this.playing = false;
    if (this.els) { this.els.play.textContent = "▶"; this.els.play.setAttribute("aria-label", "Play"); }
    if (this.frameReq) { cancelAnimationFrame(this.frameReq); this.frameReq = null; }
  }

  select(name) {
    this.selected = name;
    this.renderer?.setSelected(name);
    this.panel();
  }

  // game milliseconds per real millisecond
  rate() {
    const speed = +this.els.speed.value;                 // game minutes per real second
    let r = (speed * MINUTE) / 1000;
    if (this.els.slow.checked && dialogueActive(this.tl, this.t)) r = Math.min(r, MINUTE / READ_MS_PER_LINE);
    return r;
  }

  loop() {
    if (this.frameReq) return;
    const step = (now) => {
      this.frameReq = null;
      if (!this.playing || !this.visible) return;
      const dt = this.lastFrame === null ? 0 : Math.min(250, now - this.lastFrame);
      this.lastFrame = now;
      let t = this.t + dt * this.rate();
      if (this.els.skip.checked) {
        const target = skipTarget(this.tl, t);
        if (target !== null) t = target;
      }
      if (t >= this.tl.tMax) {
        t = this.tl.tMax;
        if (!this.els.live.checked) this.pause();      // replay finished; live mode waits for new events
      }
      this.t = t;
      this.draw(false);
      if (this.playing) this.frameReq = requestAnimationFrame(step);
    };
    this.frameReq = requestAnimationFrame(step);
  }

  draw(force) {
    if (!this.tl || !this.renderer) return;
    const bubbleMs = Math.max(MINUTE, 4000 * (+this.els.speed.value * MINUTE) / 1000);
    const scene = sceneAt(this.tl, this.t, { bubbleMs });
    this.scene = scene;
    this.renderer.render(scene);
    const span = Math.max(1, this.tl.tMax - this.tl.t0);
    this.els.scrub.value = String(Math.round(((this.t - this.tl.t0) / span) * 1000));
    this.els.clock.textContent = `${new Date(this.t).toISOString().slice(0, 16).replace("T", " ")}`;
    const now = performance.now();
    if (force || now - this.lastPanel > 300) {
      this.lastPanel = now;
      this.panel();
      this.onTimeChange(Math.floor((this.t - this.tl.t0) / this.tl.tickMs));
    }
  }

  // ------------------------------------------------------------------ side panel
  panel() {
    const box = this.els.side;
    box.replaceChildren();
    if (!this.scene) return;
    if (this.selected) box.append(this.agentPanel(this.selected));
    else box.append(h("p", { class: "muted small" }, "Click an agent to see what they are doing, their plan for today, and their memories."));
    box.append(this.conversationPanel());
  }

  agentPanel(name) {
    const a = this.scene.agents.find((x) => x.name === name);
    const colors = agentColors(this.tl.agentNames);
    const tick = Math.floor((this.t - this.tl.t0) / this.tl.tickMs);
    const plan = planAt(this.idx, name, this.t) || [];
    const memories = memoryAt(this.idx, name, tick).slice(-6).reverse();
    return h("div", { class: "agent-panel" },
      h("div", { class: "row", style: "margin-bottom:6px" },
        h("span", { class: "swatch", style: `background:${colors[name].fill}` }), h("b", {}, name),
        h("button", { class: "small", onclick: () => this.onOpenMemory(name, tick) }, "Memory stream"),
        h("button", { class: "small", onclick: () => this.select(null) }, "✕")),
      h("div", { class: "now-line" }, h("span", { class: "big-emoji" }, a?.emoji || "·"), a?.description || ""),
      h("div", { class: "muted small" }, a?.inTransit ? `${a.inTransit.from} → ${a.inTransit.to} (${Math.round(a.inTransit.progress * 100)}%)` : a?.place || ""),
      h("h3", {}, "Plan for today"),
      plan.length ? h("ol", { class: "plan" }, plan.map((b) => h("li", { class: b.current ? "current" : "" },
        h("span", { class: "mono" }, `${b.start.slice(11, 16)}–${b.end.slice(11, 16)} `), b.description, h("span", { class: "muted" }, ` @ ${b.place}`),
        b.current && b.steps.length ? h("ul", { class: "steps" }, b.steps.map((s) => h("li", { class: s.current ? "current" : "" },
          h("span", { class: "mono" }, `${s.start.slice(11, 16)} `), `${s.emoji || ""} ${s.description}`))) : null)))
        : h("div", { class: "muted small" }, "No plan yet."),
      h("h3", {}, "Recent memories"),
      h("ul", { class: "mems" }, memories.map((m) => h("li", {}, h("span", { class: `badge b-${m.type}` }, m.subtype), " ", m.description,
        h("span", { class: "muted" }, ` · ${m.created.slice(11, 16)} · imp ${m.importance}`)))));
  }

  conversationPanel() {
    const t = this.t;
    let convs = this.tl.conversations.filter((c) => c.t0 <= t && t < c.t1);
    let label = "Talking now";
    if (!convs.length) {
      const past = this.tl.conversations.filter((c) => c.t0 <= t);
      convs = past.slice(-1);
      label = "Most recent conversation";
    }
    if (this.selected) {
      const mine = this.tl.conversations.filter((c) => c.t0 <= t && c.participants.includes(this.selected));
      if (mine.length && !convs.some((c) => c.participants.includes(this.selected))) { convs = mine.slice(-1); label = `${this.selected.split(" ")[0]}'s latest conversation`; }
    }
    const box = h("div", { class: "conv-panel" }, h("h3", {}, label));
    if (!convs.length) { box.append(h("div", { class: "muted small" }, "No conversations yet.")); return box; }
    for (const c of convs) {
      box.append(h("div", { class: "muted small" }, `${stamp(c.t0)} · ${c.participants.join(" & ")} · ${c.place}`));
      const lines = c.turns.filter((u) => u.t <= t);
      for (const u of lines) {
        box.append(h("div", { class: `line ${u.speaker === c.participants[0] ? "" : "right"}` },
          h("div", { class: "who" }, `${u.speaker.split(" ")[0]} · ${hhmm(u.t)}`), u.text));
      }
      if (lines.length < c.turns.length) box.append(h("div", { class: "muted small" }, "…"));
      box.append(h("button", { class: "small", onclick: () => this.onOpenDialogue(c.id) }, "Full transcript"));
    }
    return box;
  }
}
