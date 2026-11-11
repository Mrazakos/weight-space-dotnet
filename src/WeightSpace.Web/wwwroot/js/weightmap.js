// The weight map: all 698,115 weights of the network as 24 rectangles, one per tensor, dark
// where a weight is zero. When an edit statement sets weights they light up: blue for positive,
// red for negative, with a short flash so the eye catches where the model is writing.

const W = 1600, H = 640;
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';
const THEMES = {
  light: { page: "#fcfcfb", panel: "#eeede8", edge: "#d6d5cd", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", pos: "#2a78d6", neg: "#d03b3b" },
  dark: { page: "#1a1a19", panel: "#242422", edge: "#383835", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", pos: "#3987e5", neg: "#e66767" },
};

const state = { canvas: null, ctx: null, panels: [], byKey: {}, dots: new Map(), theme: "light", raf: 0 };

export function init(canvas, layoutJson, theme) {
  state.canvas = canvas;
  canvas.width = W;
  canvas.height = H;
  state.ctx = canvas.getContext("2d");
  state.panels = JSON.parse(layoutJson);
  state.byKey = Object.fromEntries(state.panels.map(p => [p.key, p]));
  state.theme = theme === "dark" ? "dark" : "light";
  state.dots.clear();
  draw();
}

export function reset() { state.dots.clear(); draw(); }
export function setTheme(theme) { state.theme = theme === "dark" ? "dark" : "light"; draw(); }

// edits: [[tensor key, [flat indices], [values]], ...] as applied by one or more statements
export function add(edits) {
  const now = performance.now();
  for (const [key, indices, values] of edits) {
    for (let n = 0; n < indices.length; n++) {
      const id = key + ":" + indices[n];
      if (values[n] === 0) state.dots.delete(id);
      else state.dots.set(id, { key, index: indices[n], value: values[n], born: now });
    }
  }
  if (!state.raf) state.raf = requestAnimationFrame(animate);
}

function animate() {
  draw();
  const now = performance.now();
  let live = false;
  for (const d of state.dots.values()) if (now - d.born < 700) { live = true; break; }
  state.raf = live ? requestAnimationFrame(animate) : 0;
}

function draw() {
  const ctx = state.ctx, t = THEMES[state.theme], now = performance.now();
  if (!ctx) return;
  ctx.fillStyle = t.page;
  ctx.fillRect(0, 0, W, H);
  for (const p of state.panels) {
    ctx.fillStyle = t.panel;
    ctx.fillRect(p.x, p.y, p.w, p.h);
    if (p.gates) {                                  // the three gates of a GRU: reset, update, new
      ctx.strokeStyle = t.edge;
      ctx.lineWidth = 1;
      for (const k of [1, 2]) {
        ctx.beginPath();
        ctx.moveTo(p.x, p.y + p.h * k / 3);
        ctx.lineTo(p.x + p.w, p.y + p.h * k / 3);
        ctx.stroke();
      }
    }
    if (p.label) {
      ctx.font = `400 15px ${FONT}`;
      ctx.fillStyle = t.ink2;
      ctx.textAlign = "left";
      ctx.fillText(p.label, p.x, p.y - 10);
    }
  }
  const first = state.panels.find(p => p.gates);
  if (first) {
    ctx.font = `400 13px ${FONT}`;
    ctx.fillStyle = t.muted;
    ctx.textAlign = "right";
    ["reset", "update", "new"].forEach((g, k) => ctx.fillText(g, first.x - 4, first.y + first.h * (k + 0.5) / 3 + 4));
  }
  for (const d of state.dots.values()) {
    const p = state.byKey[d.key];
    if (!p) continue;
    const row = Math.floor(d.index / p.cols), col = d.index % p.cols;
    const x = p.x + (col + 0.5) / p.cols * p.w, y = p.y + (row + 0.5) / p.rows * p.h;
    const age = Math.min(1, (now - d.born) / 700);
    const r = 4 + (1 - age) * 10;
    ctx.globalAlpha = 0.35 + 0.65 * age;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fillStyle = d.value > 0 ? t.pos : t.neg;
    ctx.fill();
    ctx.globalAlpha = 1;
  }
  // legend
  ctx.font = `400 15px ${FONT}`;
  ctx.textAlign = "left";
  let x = 1420;
  for (const [colour, label] of [[t.pos, "positive weight"], [t.neg, "negative weight"]]) {
    ctx.beginPath();
    ctx.arc(x + 6, 150, 6, 0, Math.PI * 2);
    ctx.fillStyle = colour;
    ctx.fill();
    ctx.fillStyle = t.ink2;
    ctx.fillText(label, x + 20, 155);
    ctx.translate(0, 26);
  }
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.fillStyle = t.muted;
  ctx.fillText("grey: weight is 0", 1420, 210);
  ctx.fillText("In each GRU block: left, weights from the layer's input; right, from its own state; thin strips are biases.", 70, 628);
}

export function scrollToEnd(element) { if (element) element.scrollTop = element.scrollHeight; }
