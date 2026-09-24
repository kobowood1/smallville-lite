// WebGL (three.js) implementation of the town renderer — same contract as svg-renderer.js
// (see renderer.js). The town view hands it a scene every frame; this renderer adds its own
// animation on top: a day/night cycle driven by game time, lamps that come on at night,
// agents that bob and turn as they walk, lie down to sleep (with floating z's), pop their
// emoji when their action changes, and pulsing markers on objects in use.
//
// three.js is loaded from a CDN through the page's import map; if that or WebGL is unavailable
// createRenderer("three") throws and the town falls back to the SVG renderer.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const DAY_MS = 24 * 3600 * 1000;
const WALL_H = 12, WALL_T = 3, DOOR_W = 34, EMOJI_H = 26;
const PALETTE = {
  ground: 0xcfdcc4, publicFloor: 0xf2e6cf, privateFloor: 0xe3d3bd, area: 0xfaf4ea, wall: 0xb9a78f,
  skyDay: new THREE.Color(0xbfe0f7), skyDusk: new THREE.Color(0xf2b27e), skyNight: new THREE.Color(0x0f1426),
  lamp: 0xffc977, talk: 0xeb6834,
};

// ---------------------------------------------------------------- text / emoji sprites
const textureCache = new Map();

function canvasTexture(key, draw, w, h) {
  if (textureCache.has(key)) return textureCache.get(key);
  const c = document.createElement("canvas");
  const dpr = 2;
  c.width = w * dpr; c.height = h * dpr;
  const ctx = c.getContext("2d");
  ctx.scale(dpr, dpr);
  draw(ctx, w, h);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  const entry = { tex, aspect: w / h };
  textureCache.set(key, entry);
  return entry;
}

function emojiTexture(emoji) {
  return canvasTexture(`e:${emoji}`, (ctx, w, h) => {
    ctx.font = `${Math.floor(h * 0.8)}px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif`;
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(emoji, w / 2, h / 2 + h * 0.05);
  }, 64, 64);
}

function labelTexture(text, { size = 26, bold = true, pill = true } = {}) {
  const probe = document.createElement("canvas").getContext("2d");
  const font = `${bold ? 600 : 400} ${size}px system-ui,-apple-system,"Segoe UI",Roboto,sans-serif`;
  probe.font = font;
  const w = Math.ceil(probe.measureText(text).width) + (pill ? size : 8);
  const h = Math.ceil(size * 1.6);
  return canvasTexture(`l:${font}:${pill}:${text}`, (ctx) => {
    if (pill) {
      ctx.fillStyle = "rgba(255,255,255,0.88)";
      const r = h / 2;
      ctx.beginPath(); ctx.roundRect(1, 1, w - 2, h - 2, r); ctx.fill();
    }
    ctx.font = font; ctx.fillStyle = "#1b1b1a"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(text, w / 2, h / 2 + 1);
  }, w, h);
}

function sprite(entry, height, { depthTest = true } = {}) {
  const mat = new THREE.SpriteMaterial({ map: entry.tex, transparent: true, depthTest, depthWrite: false });
  const s = new THREE.Sprite(mat);
  s.scale.set(height * entry.aspect, height, 1);
  return s;
}

function setSpriteTexture(s, entry, height) {
  s.material.map = entry.tex;
  s.material.needsUpdate = true;
  s.scale.set(height * entry.aspect, height, 1);
}

// ---------------------------------------------------------------- renderer
export class ThreeRenderer {
  constructor() {
    this.sprites = new Map();
    this.objects = new Map();
    this.links = new Map();
    this.lamps = [];
    this.clickHandler = () => {};
    this.selected = null;
    this.scene3 = null;
    this.latest = null;
    this.frame = null;
    this.timer = new THREE.Timer();
    this.visible = true;
  }

  mount(container, layout, { agents, colors }) {
    this.container = container;
    this.layout = layout;
    container.classList.add("threetown");
    // throws without WebGL -> SVG fallback; preserveDrawingBuffer lets screenshots / "save image" capture the view
    const gl = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
    gl.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    gl.shadowMap.enabled = true;
    gl.shadowMap.type = THREE.PCFShadowMap;
    this.gl = gl;
    gl.domElement.setAttribute("role", "img");
    gl.domElement.setAttribute("aria-label", "3D view of the town with the agents' positions");
    this.overlay = document.createElement("div");
    this.overlay.className = "bubbles";
    container.append(gl.domElement, this.overlay);
    this.bubbles = new Map();

    const scene = new THREE.Scene();
    this.scene3 = scene;
    this.ox = layout.width / 2;
    this.oz = layout.height / 2;

    // lights: sky/ground fill + a sun that moves with the game clock
    this.hemi = new THREE.HemisphereLight(0xffffff, 0x8a7f6a, 1.0);
    scene.add(this.hemi);
    const sun = new THREE.DirectionalLight(0xfff2dc, 1.6);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const ext = Math.max(layout.width, layout.height) * 0.75;
    Object.assign(sun.shadow.camera, { left: -ext, right: ext, top: ext, bottom: -ext, near: 10, far: 3000 });
    sun.shadow.bias = -0.0008;
    scene.add(sun, sun.target);
    this.sun = sun;

    // ground
    const ground = new THREE.Mesh(new THREE.CircleGeometry(Math.hypot(layout.width, layout.height) * 0.6, 96),
      new THREE.MeshStandardMaterial({ color: PALETTE.ground, roughness: 1 }));
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    for (const p of layout.places) this.buildPlace(p);

    // agents
    const bodyGeo = new THREE.CapsuleGeometry(9, 15, 6, 18);
    const headGeo = new THREE.SphereGeometry(7.2, 22, 16);
    const noseGeo = new THREE.ConeGeometry(2.2, 5.5, 10);
    this.agentMeshes = [];
    for (const name of agents) {
      const color = new THREE.Color(colors[name].fill);
      const root = new THREE.Group();
      const body = new THREE.Group();
      const torso = new THREE.Mesh(bodyGeo, new THREE.MeshStandardMaterial({ color, roughness: 0.55 }));
      torso.position.y = 17.5;
      const head = new THREE.Mesh(headGeo, new THREE.MeshStandardMaterial({ color: color.clone().lerp(new THREE.Color(0xffe9d6), 0.65), roughness: 0.6 }));
      head.position.y = 37.5;
      const nose = new THREE.Mesh(noseGeo, new THREE.MeshStandardMaterial({ color: color.clone().multiplyScalar(0.7) }));
      nose.rotation.x = Math.PI / 2; nose.position.set(0, 37.5, 7.6);
      for (const m of [torso, head, nose]) { m.castShadow = true; m.userData.agent = name; }
      body.add(torso, head, nose);
      root.add(body);
      const ring = new THREE.Mesh(new THREE.RingGeometry(15, 19, 40),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2; ring.position.y = 3.6;
      root.add(ring);
      const emoji = sprite(emojiTexture("·"), EMOJI_H);
      emoji.position.y = 62;
      const label = sprite(labelTexture(name.split(" ")[0]), 14, { depthTest: false });
      label.position.y = 80;
      label.renderOrder = 10;
      root.add(emoji, label);
      scene.add(root);
      this.agentMeshes.push(torso, head, nose);
      this.sprites.set(name, {
        root, body, ring, emoji, label, color: colors[name].fill,
        pos: null, target: null, heading: 0, bob: 0, emojiText: null, pop: 0, zs: [], zTimer: Math.random(),
      });
    }

    this.gTalk = new THREE.Group();
    scene.add(this.gTalk);

    // camera + controls
    const cam = new THREE.OrthographicCamera(-1, 1, 1, -1, 1, 5000);
    cam.position.set(-220, 700, 620);
    cam.lookAt(0, 0, 0);
    this.camera = cam;
    const controls = new OrbitControls(cam, gl.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minPolarAngle = 0.25;
    controls.maxPolarAngle = 1.25;
    controls.minZoom = 0.6;
    controls.maxZoom = 4;
    controls.screenSpacePanning = false;
    this.controls = controls;

    // picking
    this.raycaster = new THREE.Raycaster();
    gl.domElement.addEventListener("pointerdown", (e) => { this.downAt = [e.clientX, e.clientY]; });
    gl.domElement.addEventListener("pointerup", (e) => {
      if (!this.downAt || Math.hypot(e.clientX - this.downAt[0], e.clientY - this.downAt[1]) > 5) return;
      const r = gl.domElement.getBoundingClientRect();
      const ndc = new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
      this.raycaster.setFromCamera(ndc, cam);
      const hit = this.raycaster.intersectObjects(this.agentMeshes, false)[0];
      if (hit) this.clickHandler(hit.object.userData.agent);
    });

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.visibility = new IntersectionObserver((entries) => {
      this.visible = entries.some((en) => en.isIntersecting);
      if (this.visible) this.start();
    });
    this.visibility.observe(container);
    this.resize();
    this.start();
  }

  buildPlace(p) {
    const [x, y, w, h] = p.rect;
    const cx = x + w / 2 - this.ox, cz = y + h / 2 - this.oz;
    const floor = new THREE.Mesh(new THREE.BoxGeometry(w, 3, h),
      new THREE.MeshStandardMaterial({ color: p.public ? PALETTE.publicFloor : PALETTE.privateFloor, roughness: 0.9 }));
    floor.position.set(cx, 1.5, cz);
    floor.receiveShadow = true;
    this.scene3.add(floor);
    for (const a of p.areas) {
      const [ax, ay, aw, ah] = a.rect;
      const tile = new THREE.Mesh(new THREE.BoxGeometry(aw, 0.6, ah),
        new THREE.MeshStandardMaterial({ color: PALETTE.area, roughness: 0.8 }));
      tile.position.set(ax + aw / 2 - this.ox, 3.3, ay + ah / 2 - this.oz);
      tile.receiveShadow = true;
      this.scene3.add(tile);
      const lab = sprite(labelTexture(a.name, { size: 20, bold: false, pill: false }), 10);
      lab.position.set(ax + 6 - this.ox + lab.scale.x / 2, 5, ay + 8 - this.oz);
      this.scene3.add(lab);
    }
    // walls with a gap at the door
    const wallMat = new THREE.MeshStandardMaterial({ color: PALETTE.wall, roughness: 0.85 });
    const [dx, dy] = p.door;
    const edges = [
      { a: [x, y], b: [x + w, y] }, { a: [x, y + h], b: [x + w, y + h] },
      { a: [x, y], b: [x, y + h] }, { a: [x + w, y], b: [x + w, y + h] },
    ];
    const dist = (e) => (e.a[1] === e.b[1] ? Math.abs(dy - e.a[1]) : Math.abs(dx - e.a[0]));
    const doorEdge = edges.reduce((best, e) => (dist(e) < dist(best) ? e : best), edges[0]);
    for (const e of edges) {
      const horizontal = e.a[1] === e.b[1];
      const segs = [];
      if (e === doorEdge) {
        const d = horizontal ? dx : dy, lo = horizontal ? e.a[0] : e.a[1], hi = horizontal ? e.b[0] : e.b[1];
        if (d - DOOR_W / 2 > lo) segs.push([lo, d - DOOR_W / 2]);
        if (d + DOOR_W / 2 < hi) segs.push([d + DOOR_W / 2, hi]);
      } else segs.push(horizontal ? [e.a[0], e.b[0]] : [e.a[1], e.b[1]]);
      for (const [s0, s1] of segs) {
        const len = s1 - s0;
        const wall = new THREE.Mesh(horizontal ? new THREE.BoxGeometry(len, WALL_H, WALL_T) : new THREE.BoxGeometry(WALL_T, WALL_H, len), wallMat);
        if (horizontal) wall.position.set((s0 + s1) / 2 - this.ox, WALL_H / 2 + 3, e.a[1] - this.oz);
        else wall.position.set(e.a[0] - this.ox, WALL_H / 2 + 3, (s0 + s1) / 2 - this.oz);
        wall.castShadow = true; wall.receiveShadow = true;
        this.scene3.add(wall);
      }
    }
    // name
    const name = sprite(labelTexture(p.name, { size: 30 }), 22, { depthTest: false });
    name.position.set(cx, 34, y - this.oz + 6);
    name.renderOrder = 5;
    this.scene3.add(name);
    // lamp by the door: a post, a bulb, and a warm light that comes on at night
    const lampX = dx - this.ox + (doorEdge.a[1] === doorEdge.b[1] ? DOOR_W / 2 + 8 : 0);
    const lampZ = dy - this.oz + (doorEdge.a[1] === doorEdge.b[1] ? 0 : DOOR_W / 2 + 8);
    const post = new THREE.Mesh(new THREE.CylinderGeometry(1, 1.2, 26, 8), new THREE.MeshStandardMaterial({ color: 0x4a4a48 }));
    post.position.set(lampX, 13, lampZ);
    post.castShadow = true;
    const bulbMat = new THREE.MeshStandardMaterial({ color: 0xfff3d6, emissive: PALETTE.lamp, emissiveIntensity: 0 });
    const bulb = new THREE.Mesh(new THREE.SphereGeometry(3, 12, 10), bulbMat);
    bulb.position.set(lampX, 27, lampZ);
    const light = new THREE.PointLight(PALETTE.lamp, 0, Math.max(w, h) * 1.1, 1.4);
    light.position.set(cx, 75, cz);
    this.scene3.add(post, bulb, light);
    this.lamps.push({ light, bulbMat });
    // objects: emoji billboards; a pulsing ring when in use
    for (const o of p.objects) {
      const s = sprite(emojiTexture(o.icon), 20);
      s.position.set(o.at[0] - this.ox, 13, o.at[1] - this.oz);
      const ring = new THREE.Mesh(new THREE.RingGeometry(8, 11.5, 32),
        new THREE.MeshBasicMaterial({ color: 0xfab219, transparent: true, opacity: 0, side: THREE.DoubleSide, depthWrite: false }));
      ring.rotation.x = -Math.PI / 2;
      ring.position.set(o.at[0] - this.ox, 3.9, o.at[1] - this.oz);
      this.scene3.add(s, ring);
      this.objects.set(o.path, { sprite: s, ring, def: o.default_state, busy: false, phase: Math.random() * 6 });
    }
  }

  // --------------------------------------------------------------- contract
  render(scene) {
    this.latest = scene;
    for (const a of scene.agents) {
      const s = this.sprites.get(a.name);
      if (!s) continue;
      const target = new THREE.Vector3(a.x - this.ox, 0, a.y - this.oz);
      if (!s.pos) { s.pos = target.clone(); s.root.position.copy(target); }
      s.target = target;
      s.state = a;
      if (s.emojiText !== a.emoji) {
        setSpriteTexture(s.emoji, emojiTexture(a.emoji || "·"), EMOJI_H);
        if (s.emojiText !== null) s.pop = 1;
        s.emojiText = a.emoji;
      }
    }
    for (const [path, o] of this.objects) {
      const state = scene.objectStates[path] ?? o.def;
      o.busy = state !== o.def;
    }
    this.links = new Map(scene.links.map((l) => [l.conv, l.participants]));
    if (!this.frame) this.start();
  }

  setSelected(name) { this.selected = name; }

  onAgentClick(fn) { this.clickHandler = fn; }

  resize() {
    if (!this.gl) return;
    const w = this.container.clientWidth || 800;
    const h = Math.round(w * (this.layout.height / this.layout.width));
    this.gl.setSize(w, h);
    this.fit(w / h);
  }

  // Frame the whole map for the current camera angle: project its corners into view space.
  fit(aspect) {
    const cam = this.camera;
    cam.updateMatrixWorld();
    const W = this.layout.width / 2, H = this.layout.height / 2;
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const x of [-W, W]) for (const y of [0, 70]) for (const z of [-H, H]) {
      const v = new THREE.Vector3(x, y, z).applyMatrix4(cam.matrixWorldInverse);
      minX = Math.min(minX, v.x); maxX = Math.max(maxX, v.x); minY = Math.min(minY, v.y); maxY = Math.max(maxY, v.y);
    }
    let fw = maxX - minX + 30, fh = maxY - minY + 30;
    if (fw / fh > aspect) fh = fw / aspect; else fw = fh * aspect;
    const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
    Object.assign(cam, { left: cx - fw / 2, right: cx + fw / 2, top: cy + fh / 2, bottom: cy - fh / 2 });
    cam.updateProjectionMatrix();
  }

  destroy() {
    if (this.frame) cancelAnimationFrame(this.frame);
    this.frame = null;
    this.resizeObserver?.disconnect();
    this.visibility?.disconnect();
    this.controls?.dispose();
    this.scene3?.traverse((obj) => {
      obj.geometry?.dispose?.();
      if (obj.material) (Array.isArray(obj.material) ? obj.material : [obj.material]).forEach((m) => m.dispose());
    });
    this.gl?.dispose();
    this.gl?.domElement.remove();
    this.overlay?.remove();
    this.container?.classList.remove("threetown");
    this.sprites.clear(); this.objects.clear();
  }

  // --------------------------------------------------------------- animation loop
  start() {
    if (this.frame || !this.gl) return;
    const tick = (now) => {
      this.frame = null;
      if (!this.visible || !this.gl) return;
      this.timer.update(now);
      const dt = Math.min(0.1, this.timer.getDelta());
      this.animate(dt, this.timer.getElapsed());
      this.controls.update();
      this.gl.render(this.scene3, this.camera);
      this.placeBubbles();
      this.frame = requestAnimationFrame(tick);
    };
    this.frame = requestAnimationFrame(tick);
  }

  animate(dt, time) {
    const scene = this.latest;
    if (scene) this.dayNight(scene.t);
    for (const [name, s] of this.sprites) {
      if (!s.target) continue;
      // follow the scene position smoothly; the scene is already interpolated in game time
      const before = s.root.position.clone();
      s.root.position.lerp(s.target, 1 - Math.pow(0.0005, dt));
      const moved = s.root.position.clone().sub(before);
      const speed = moved.length() / Math.max(dt, 1e-3);
      if (speed > 4) s.heading = Math.atan2(moved.x, moved.z);
      s.body.rotation.y += (((s.heading - s.body.rotation.y + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI) * Math.min(1, dt * 10);
      const asleep = s.emojiText === "😴";
      const talking = s.emojiText === "💬";
      if (speed > 4) s.bob += dt * 14;
      const bobY = speed > 4 ? Math.abs(Math.sin(s.bob)) * 3 : talking ? Math.abs(Math.sin(time * 6)) * 1.2 : Math.sin(time * 2 + s.zTimer * 6) * 0.4;
      const lie = asleep ? Math.PI / 2 : 0;
      s.body.rotation.z += (lie - s.body.rotation.z) * Math.min(1, dt * 4);
      s.body.position.y = asleep ? 6 : bobY;
      s.body.position.x = asleep ? -14 : 0;
      // emoji pop when the action changes
      s.pop = Math.max(0, s.pop - dt * 2.5);
      const popScale = 1 + Math.sin(s.pop * Math.PI) * 0.6;
      const e = emojiTexture(s.emojiText || "·");
      s.emoji.scale.set(EMOJI_H * e.aspect * popScale, EMOJI_H * popScale, 1);
      s.emoji.position.y = (asleep ? 40 : 62) + Math.sin(time * 2.2 + s.zTimer * 5) * 1.5;
      s.label.position.y = asleep ? 58 : 80;
      // selection ring
      const sel = name === this.selected;
      s.ring.material.opacity = sel ? 0.55 + Math.sin(time * 4) * 0.25 : 0;
      s.ring.scale.setScalar(sel ? 1 + Math.sin(time * 4) * 0.08 : 1);
      // sleeping z's
      if (asleep) {
        s.zTimer += dt;
        if (s.zTimer > 1.1) {
          s.zTimer = 0;
          const z = sprite(labelTexture("z", { size: 26, pill: false }), 12);
          z.position.set(s.root.position.x + 6, 34, s.root.position.z);
          z.userData.age = 0;
          this.scene3.add(z);
          s.zs.push(z);
        }
      }
      for (const z of [...s.zs]) {
        z.userData.age += dt;
        z.position.y += dt * 10;
        z.position.x += Math.sin(z.userData.age * 3) * dt * 6;
        z.material.opacity = Math.max(0, 1 - z.userData.age / 2.2);
        if (z.userData.age > 2.2) { this.scene3.remove(z); z.material.dispose(); s.zs.splice(s.zs.indexOf(z), 1); }
      }
    }
    for (const o of this.objects.values()) {
      o.phase += dt * 3;
      o.ring.material.opacity = o.busy ? 0.35 + Math.sin(o.phase) * 0.25 : 0;
      o.sprite.position.y = o.busy ? 15 + Math.sin(o.phase * 1.3) * 2 : 13;
    }
    this.drawLinks(time);
  }

  dayNight(t) {
    const hour = ((t % DAY_MS) + DAY_MS) % DAY_MS / 3600000;
    const daylight = Math.max(0, Math.sin(((hour - 6) / 12) * Math.PI));        // 0 at night, 1 at noon
    const twilight = Math.max(0, 1 - Math.abs(hour - 6.5) / 1.5) + Math.max(0, 1 - Math.abs(hour - 18.5) / 1.5);
    const sky = PALETTE.skyNight.clone().lerp(PALETTE.skyDay, Math.min(1, daylight * 1.6));
    sky.lerp(PALETTE.skyDusk, Math.min(0.6, twilight * 0.6));
    this.scene3.background = sky;
    this.hemi.intensity = 0.35 + 0.85 * daylight;
    this.hemi.color.setHSL(0.6, 0.3, 0.55 + 0.45 * daylight);
    this.sun.intensity = 0.15 + 1.6 * daylight;
    const angle = ((hour - 6) / 12) * Math.PI;
    this.sun.position.set(Math.cos(angle) * 700, 200 + Math.max(0.1, Math.sin(angle)) * 600, 300);
    this.sun.color.setHSL(0.1, 0.6, 0.75 + 0.2 * daylight);
    const night = 1 - Math.min(1, daylight * 3);
    for (const l of this.lamps) {
      l.light.intensity = night * 900;
      l.bulbMat.emissiveIntensity = night * 2.5;
    }
  }

  drawLinks(time) {
    this.gTalk.clear();
    for (const [, participants] of this.links) {
      const [a, b] = participants.map((n) => this.sprites.get(n)).filter(Boolean);
      if (!a || !b) continue;
      const p0 = a.root.position.clone().setY(30), p1 = b.root.position.clone().setY(30);
      const mid = p0.clone().add(p1).multiplyScalar(0.5).setY(42 + Math.sin(time * 3) * 2);
      const curve = new THREE.QuadraticBezierCurve3(p0, mid, p1);
      const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(24));
      const line = new THREE.Line(geo, new THREE.LineDashedMaterial({ color: PALETTE.talk, dashSize: 4, gapSize: 3 }));
      line.computeLineDistances();
      line.material.dashOffset = -time * 8;
      this.gTalk.add(line);
    }
  }

  placeBubbles() {
    const scene = this.latest;
    if (!scene) return;
    const r = this.gl.domElement.getBoundingClientRect();
    const box = this.container.getBoundingClientRect();
    const toScreen = (v) => {
      const p = v.clone().project(this.camera);
      return { x: (p.x + 1) / 2 * r.width + r.left - box.left, y: (1 - p.y) / 2 * r.height + r.top - box.top };
    };
    for (const a of scene.agents) {
      let b = this.bubbles.get(a.name);
      if (!a.speaking) { if (b) b.el.hidden = true; continue; }
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
      const me = this.sprites.get(a.name).root.position;
      const head = toScreen(me.clone().setY(90));
      const partner = a.partner ? this.sprites.get(a.partner.name)?.root.position : null;
      const ps = partner ? toScreen(partner.clone().setY(90)) : null;
      const east = !ps || ps.x <= head.x;
      const below = !!ps && ps.y < head.y - 20 && Math.abs(ps.x - head.x) < 140;
      b.el.classList.toggle("east", east);
      b.el.classList.toggle("west", !east);
      b.el.classList.toggle("below", below);
      const anchor = below ? toScreen(me.clone().setY(-6)) : head;
      b.el.style.left = `${anchor.x}px`;
      b.el.style.top = `${anchor.y}px`;
    }
  }
}
