// SVG implementation of the town renderer (see renderer.js for the contract).
// Static layers are built once in mount(); render() only moves sprites and toggles text/state.
// Speech bubbles are HTML overlays (SVG has no text wrapping) positioned from SVG coordinates.

const NS = "http://www.w3.org/2000/svg";

function el(tag, attrs = {}, parent = null) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (parent) parent.append(e);
  return e;
}

function text(parent, x, y, value, attrs = {}) {
  const t = el("text", { x, y, ...attrs }, parent);
  t.textContent = value;
  return t;
}

export class SvgRenderer {
  constructor() {
    this.container = null;
    this.svg = null;
    this.overlay = null;
    this.sprites = new Map();
    this.objects = new Map();
    this.links = new Map();
    this.clickHandler = () => {};
    this.selected = null;
  }

  mount(container, layout, { agents, colors }) {
    this.container = container;
    container.classList.add("svgtown");
    const svg = el("svg", { viewBox: `0 0 ${layout.width} ${layout.height}`, role: "img",
                            "aria-label": "Map of the town with the agents' positions" });
    this.svg = svg;
    const gPlaces = el("g", { class: "places" }, svg);
    for (const p of layout.places) {
      const [x, y, w, h] = p.rect;
      const g = el("g", { class: `place ${p.public ? "public" : "private"}` }, gPlaces);
      el("rect", { x, y, width: w, height: h, rx: 10, class: "place-rect" }, g);
      text(g, x + 10, y + 18, p.name, { class: "place-label" });
      for (const a of p.areas) {
        const [ax, ay, aw, ah] = a.rect;
        el("rect", { x: ax, y: ay, width: aw, height: ah, rx: 6, class: "area-rect" }, g);
        text(g, ax + 6, ay + 13, a.name, { class: "area-label" });
      }
      el("circle", { cx: p.door[0], cy: p.door[1], r: 4, class: "door" }, g);
      for (const o of p.objects) {
        const og = el("g", { class: "object", transform: `translate(${o.at[0]},${o.at[1]})` }, g);
        const title = el("title", {}, og);
        title.textContent = `${o.name} (${o.default_state})`;
        text(og, 0, 5, o.icon, { class: "object-icon", "text-anchor": "middle" });
        const badge = el("circle", { cx: 9, cy: -8, r: 4, class: "object-badge", visibility: "hidden" }, og);
        this.objects.set(o.path, { g: og, title, badge, name: o.name, def: o.default_state, state: null });
      }
    }
    this.gLinks = el("g", { class: "links" }, svg);
    const gAgents = el("g", { class: "agents" }, svg);
    for (const name of agents) {
      const c = colors[name];
      const g = el("g", { class: "agent", tabindex: 0, role: "button", "aria-label": name }, gAgents);
      el("circle", { r: 15, class: "agent-halo" }, g);
      el("circle", { r: 11, fill: c.fill, class: "agent-body" }, g);
      text(g, 0, 4, name.split(" ").map((s) => s[0]).join("").slice(0, 2), { class: "agent-initials", "text-anchor": "middle", fill: c.ink });
      const emoji = text(g, 0, -18, "", { class: "agent-emoji", "text-anchor": "middle" });
      text(g, 0, 26, name.split(" ")[0], { class: "agent-name", "text-anchor": "middle" });
      g.addEventListener("click", () => this.clickHandler(name));
      g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") this.clickHandler(name); });
      this.sprites.set(name, { g, emoji, color: c.fill, x: null, y: null, emojiText: "" });
    }
    this.overlay = document.createElement("div");
    this.overlay.className = "bubbles";
    container.append(svg, this.overlay);
    this.bubbles = new Map();
  }

  render(scene) {
    for (const a of scene.agents) {
      const s = this.sprites.get(a.name);
      if (!s) continue;
      if (s.x !== a.x || s.y !== a.y) {
        s.g.setAttribute("transform", `translate(${a.x.toFixed(1)},${a.y.toFixed(1)})`);
        s.x = a.x; s.y = a.y;
      }
      if (s.emojiText !== a.emoji) { s.emoji.textContent = a.emoji; s.emojiText = a.emoji; }
      s.g.classList.toggle("walking", !!a.inTransit);
      this.bubble(a);
    }
    for (const [path, o] of this.objects) {
      const state = scene.objectStates[path] ?? o.def;
      if (state === o.state) continue;
      o.state = state;
      const busy = state !== o.def;
      o.badge.setAttribute("visibility", busy ? "visible" : "hidden");
      o.g.classList.toggle("busy", busy);
      o.title.textContent = `${o.name} (${state})`;
    }
    const seen = new Set();
    for (const link of scene.links) {
      const [a, b] = link.participants.map((n) => scene.agents.find((x) => x.name === n));
      if (!a || !b) continue;
      seen.add(link.conv);
      let line = this.links.get(link.conv);
      if (!line) { line = el("line", { class: "talk-link" }, this.gLinks); this.links.set(link.conv, line); }
      line.setAttribute("x1", a.x); line.setAttribute("y1", a.y); line.setAttribute("x2", b.x); line.setAttribute("y2", b.y);
    }
    for (const [id, line] of this.links) if (!seen.has(id)) { line.remove(); this.links.delete(id); }
  }

  bubble(a) {
    let b = this.bubbles.get(a.name);
    if (!a.speaking) { if (b) b.el.hidden = true; return; }
    if (!b) {
      const div = document.createElement("div");
      div.className = "speech";
      div.style.setProperty("--who", this.sprites.get(a.name).color);
      this.overlay.append(div);
      b = { el: div, text: null };
      this.bubbles.set(a.name, b);
    }
    if (b.text !== a.speaking.text) { b.el.textContent = a.speaking.text; b.text = a.speaking.text; }
    b.el.hidden = false;
    // Open the bubble away from the partner so it never covers the person being addressed.
    const east = !a.partner || a.partner.x <= a.x;
    const below = !!a.partner && a.partner.y < a.y - 20 && Math.abs(a.partner.x - a.x) < 140;
    b.el.classList.toggle("east", east);
    b.el.classList.toggle("west", !east);
    b.el.classList.toggle("below", below);
    const p = this.toScreen(a.x, below ? a.y + 34 : a.y - 26);
    b.el.style.left = `${p.x}px`;
    b.el.style.top = `${p.y}px`;
  }

  toScreen(x, y) {
    const m = this.svg.getScreenCTM();
    const box = this.container.getBoundingClientRect();
    if (!m) return { x: 0, y: 0 };
    return { x: m.a * x + m.c * y + m.e - box.left, y: m.b * x + m.d * y + m.f - box.top };
  }

  setSelected(name) {
    this.selected = name;
    for (const [n, s] of this.sprites) s.g.classList.toggle("selected", n === name);
  }

  onAgentClick(fn) { this.clickHandler = fn; }

  resize() { /* the SVG scales with its container; bubbles are re-placed on the next render() */ }

  destroy() {
    this.svg?.remove();
    this.overlay?.remove();
    this.container?.classList.remove("svgtown");
    this.sprites.clear(); this.objects.clear(); this.links.clear();
  }
}
