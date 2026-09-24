// Renderer interface for the visual town (Phase 5).
//
// The town view never draws anything itself: it computes a scene with scene.js and hands it to a
// renderer. Any object with these methods can be plugged in (SVG today; a three.js renderer can
// implement the same contract later without touching the rest of the viewer):
//
//   mount(container: HTMLElement, layout, { agents: string[], colors: Record<string, {fill, ink}> })
//       Build static geometry (places, areas, objects) and one sprite per agent.
//   render(scene)
//       Called every animation frame with sceneAt(...) output: agent positions, emoji,
//       speech bubbles, conversation links, object states. Must be cheap and idempotent.
//   setSelected(name | null)     Highlight one agent (the one shown in the side panel).
//   onAgentClick(fn(name))       Register the click handler for agent sprites.
//   resize()                     Container size changed.
//   destroy()                    Remove everything this renderer added to the DOM.

export const RENDERERS = {
  three: () => import("./three-renderer.js").then((m) => new m.ThreeRenderer()),   // WebGL, animated
  svg: () => import("./svg-renderer.js").then((m) => new m.SvgRenderer()),         // 2D fallback
};

export async function createRenderer(kind = "svg") {
  const make = RENDERERS[kind];
  if (!make) throw new Error(`unknown renderer ${kind}; available: ${Object.keys(RENDERERS).join(", ")}`);
  const r = await make();
  for (const method of ["mount", "render", "setSelected", "onAgentClick", "resize", "destroy"]) {
    if (typeof r[method] !== "function") throw new Error(`renderer ${kind} does not implement ${method}()`);
  }
  return r;
}
