// The pitch replay: 100 seconds of tracking, the reference labels, the hand-written network's
// labels and the trained network's, with the hand-written recurrent units drawn live.
// One 1920 x 1080 canvas, so that what is on screen is exactly what gets recorded for a slide.
//
// Colour carries one meaning throughout: blue = home, orange = away, grey = dead ball.

const W = 1920, H = 1080;
const THEMES = {
  light: {
    page: "#fcfcfb", ink: "#0b0b0b", ink2: "#52514e", muted: "#898781", grid: "#e1e0d9", axis: "#c3c2b7",
    home: "#2a78d6", away: "#eb6834", dead: "#b4b2aa", pitch: "#f1f4ee", line: "#c3c9bd", ball: "#0b0b0b",
    gap: "#f0efec",
  },
  dark: {
    page: "#1a1a19", ink: "#f0efec", ink2: "#c3c2b7", muted: "#898781", grid: "#2c2c2a", axis: "#383835",
    home: "#3987e5", away: "#d95926", dead: "#6f6e68", pitch: "#20241f", line: "#3d443a", ball: "#f0efec",
    gap: "#262624",
  },
};
const LABEL_TEXT = ["Home in possession", "Away in possession", "Dead ball"];
const PITCH = { x: 48, y: 124, w: 1180, h: 764 };
const PANEL = { x: 1272, y: 124, w: 600 };
const TIME = { x: 300, y: 912, w: 1572 };
const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

const state = {
  canvas: null, ctx: null, data: null, theme: "light", frame: 0, playing: false, speed: 1,
  last: 0, raf: 0, recorder: null, onEnd: null, controls: {},
};

export async function init(canvas, url, theme) {
  state.canvas = canvas;
  canvas.width = W;
  canvas.height = H;
  state.ctx = canvas.getContext("2d");
  state.theme = theme === "dark" ? "dark" : "light";
  wireControls();
  await load(url);
  cancelAnimationFrame(state.raf);
  state.last = performance.now();
  state.raf = requestAnimationFrame(tick);
  window.weightSpaceReplay = { state, record, play, pause, seek, ready: true };
}

export async function load(url) {
  const response = await fetch(url);
  const data = await response.json();
  data.gaps = runs(data.positions.ball.map(b => b[0] === null));
  state.data = data;
  state.frame = Math.min(state.frame, data.frames - 1);
  draw();
}

export function setTheme(theme) { state.theme = theme === "dark" ? "dark" : "light"; draw(); }
export function play() { if (state.frame >= state.data.frames - 1) state.frame = 0; state.playing = true; syncControls(); }
export function pause() { state.playing = false; syncControls(); }
export function seek(fraction) { state.frame = fraction * (state.data.frames - 1); draw(); syncControls(); }
export function setSpeed(speed) { state.speed = speed; }
export function dispose() { cancelAnimationFrame(state.raf); state.playing = false; }

// Record one full pass at normal speed and hand the file to the browser as a download.
export function record() {
  return new Promise(resolve => {
    const types = ["video/mp4;codecs=avc1.640028", "video/mp4", "video/webm;codecs=vp9", "video/webm"];
    const type = types.find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
    if (!type) { resolve(null); return; }
    const stream = state.canvas.captureStream(30);
    const recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: 5_000_000 });
    const chunks = [];
    recorder.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
    recorder.onstop = () => {
      const blob = new Blob(chunks, { type });
      const name = `replay-${state.data.id}-${state.theme}.${type.startsWith("video/mp4") ? "mp4" : "webm"}`;
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = name;
      document.body.appendChild(link);
      link.click();
      link.remove();
      state.recorder = null;
      syncControls();
      resolve(name);
    };
    state.recorder = recorder;
    state.frame = 0;
    state.speed = 1;
    state.playing = true;
    state.onEnd = () => setTimeout(() => recorder.stop(), 400);     // hold the last frame briefly
    draw();
    recorder.start(250);
    syncControls();
  });
}

function wireControls() {
  const find = id => document.getElementById(id);
  state.controls = { play: find("replay-play"), scrub: find("replay-scrub"), speed: find("replay-speed"), record: find("replay-record") };
  const c = state.controls;
  if (c.play) c.play.onclick = () => (state.playing ? pause() : play());
  if (c.scrub) c.scrub.oninput = () => { state.playing = false; seek(c.scrub.value / 1000); };
  if (c.speed) c.speed.onchange = () => setSpeed(parseFloat(c.speed.value));
  if (c.record) c.record.onclick = () => record();
}

function syncControls() {
  const c = state.controls;
  if (c.play) c.play.textContent = state.playing ? "Pause" : "Play";
  if (c.scrub && state.data) c.scrub.value = Math.round(1000 * state.frame / (state.data.frames - 1));
  if (c.record) { c.record.disabled = !!state.recorder; c.record.textContent = state.recorder ? "Recording…" : "Record video"; }
}

function tick(now) {
  const dt = Math.min(0.1, (now - state.last) / 1000);
  state.last = now;
  if (state.playing && state.data) {
    state.frame += dt * state.data.fps * state.speed;
    if (state.frame >= state.data.frames - 1) {
      state.frame = state.data.frames - 1;
      state.playing = false;
      if (state.onEnd) { state.onEnd(); state.onEnd = null; }
    }
    syncControls();
  }
  // Draw every tick, moving or not: a recording needs a steady stream of frames.
  if (state.data) draw();
  state.raf = requestAnimationFrame(tick);
}

function runs(mask) {                      // [start, end) of every true run
  const out = [];
  let start = -1;
  for (let i = 0; i <= mask.length; i++) {
    if (i < mask.length && mask[i]) { if (start < 0) start = i; }
    else if (start >= 0) { out.push([start, i]); start = -1; }
  }
  return out;
}

function draw() {
  const d = state.data, ctx = state.ctx, t = THEMES[state.theme];
  if (!d) return;
  const f = Math.max(0, Math.min(d.frames - 1, state.frame));
  const i = Math.floor(f), frac = f - i, j = Math.min(d.frames - 1, i + 1);
  const colours = [t.home, t.away, t.dead];

  ctx.fillStyle = t.page;
  ctx.fillRect(0, 0, W, H);

  // ---- header
  text(ctx, "A network nobody trained", 48, 62, 44, t.ink, "600");
  text(ctx, `${d.description}  ·  ${d.nonzero.toLocaleString("en-US")} of ${d.total.toLocaleString("en-US")} weights set`, 48, 100, 24, t.ink2);
  text(ctx, `${(f / d.fps).toFixed(1).padStart(5, " ")} s`, W - 48, 62, 44, t.ink, "600", "right");
  text(ctx, "held-out match, automatic tracking", W - 48, 100, 24, t.ink2, "400", "right");

  // ---- pitch
  const P = PITCH, L = d.pitch[0], Wd = d.pitch[1];
  const px = x => P.x + (x / 10 + L / 2) / L * P.w, py = y => P.y + (y / 10 + Wd / 2) / Wd * P.h;
  const m = v => v / L * P.w;                                         // metres to pixels
  ctx.fillStyle = t.pitch;
  roundRect(ctx, P.x, P.y, P.w, P.h, 10);
  ctx.fill();
  ctx.strokeStyle = t.line;
  ctx.lineWidth = 3;
  ctx.strokeRect(P.x + 14, P.y + 14, P.w - 28, P.h - 28);
  const fx = x => P.x + 14 + (x + L / 2) / L * (P.w - 28), fy = y => P.y + 14 + (y + Wd / 2) / Wd * (P.h - 28);
  line(ctx, fx(0), fy(-Wd / 2), fx(0), fy(Wd / 2));
  ctx.beginPath();
  ctx.arc(fx(0), fy(0), m(9.15), 0, Math.PI * 2);
  ctx.stroke();
  for (const side of [-1, 1]) {
    ctx.strokeRect(Math.min(fx(side * L / 2), fx(side * (L / 2 - 16.5))), fy(-20.16), m(16.5), fy(20.16) - fy(-20.16));
    ctx.strokeRect(Math.min(fx(side * L / 2), fx(side * (L / 2 - 5.5))), fy(-9.16), m(5.5), fy(9.16) - fy(-9.16));
  }
  // the frame of the pitch takes the colour of the hand-written network's current label
  ctx.strokeStyle = colours[d.labels.net[i]];
  ctx.lineWidth = 10;
  roundRect(ctx, P.x + 5, P.y + 5, P.w - 10, P.h - 10, 8);
  ctx.stroke();

  const qx = x => fx(x / 10), qy = y => fy(y / 10);
  const lerp = (a, b) => (a === null || b === null ? a : a + (b - a) * frac);
  for (const [team, colour] of [["home", t.home], ["away", t.away]]) {
    const a = d.positions[team][i], b = d.positions[team][j];
    for (let k = 0; k < a.length; k += 2) {
      if (a[k] === null) continue;
      const x = qx(lerp(a[k], b[k])), y = qy(lerp(a[k + 1], b[k + 1]));
      ctx.beginPath();
      ctx.arc(x, y, 14, 0, Math.PI * 2);
      ctx.fillStyle = colour;
      ctx.fill();
      ctx.lineWidth = 3;
      ctx.strokeStyle = t.pitch;
      ctx.stroke();
    }
  }
  const ball = d.positions.ball[i], ballNext = d.positions.ball[j];
  if (ball[0] !== null) {
    state.lastBall = [qx(lerp(ball[0], ballNext[0])), qy(lerp(ball[1], ballNext[1]))];
    ctx.beginPath();
    ctx.arc(state.lastBall[0], state.lastBall[1], 9, 0, Math.PI * 2);
    ctx.fillStyle = t.ball;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = t.pitch;
    ctx.stroke();
  } else {
    if (state.lastBall) {                       // where it was last seen, hollow
      ctx.beginPath();
      ctx.setLineDash([5, 5]);
      ctx.arc(state.lastBall[0], state.lastBall[1], 13, 0, Math.PI * 2);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = t.ink2;
      ctx.stroke();
      ctx.setLineDash([]);
    }
    pill(ctx, P.x + P.w / 2, P.y + 46, "ball not seen by the tracking", t);
  }

  // ---- right panel: three answers, the probabilities, the hand-written units
  let y = PANEL.y - 4;
  const rows = [["Reference labels", d.labels.truth[i]], ["Written by hand", d.labels.net[i]], ["Trained by SGD", d.labels.trained[i]]];
  for (const [title, label] of rows) {
    text(ctx, title, PANEL.x, y + 22, 24, t.ink2);
    ctx.fillStyle = colours[label];
    roundRect(ctx, PANEL.x, y + 36, 26, 44, 6);
    ctx.fill();
    text(ctx, LABEL_TEXT[label], PANEL.x + 42, y + 70, 36, t.ink, title === "Written by hand" ? "600" : "400");
    y += 102;
  }
  y += 6;
  line(ctx, PANEL.x, y, PANEL.x + PANEL.w, y, t.grid, 2);
  y += 36;
  text(ctx, "Hand-written network: probabilities", PANEL.x, y, 24, t.ink2);
  y += 14;
  ["home", "away", "dead"].forEach((name, c) => {
    const p = d.probs[i * 3 + c];
    text(ctx, name, PANEL.x, y + 26, 24, t.ink2);
    ctx.fillStyle = t.grid;
    roundRect(ctx, PANEL.x + 84, y + 6, PANEL.w - 170, 24, 4);
    ctx.fill();
    if (p > 0.004) {
      ctx.fillStyle = colours[c];
      roundRect(ctx, PANEL.x + 84, y + 6, Math.max(6, (PANEL.w - 170) * p), 24, 4);
      ctx.fill();
    }
    text(ctx, p.toFixed(2), PANEL.x + PANEL.w, y + 27, 24, t.ink, "400", "right");
    y += 37;
  });
  y += 10;
  line(ctx, PANEL.x, y, PANEL.x + PANEL.w, y, t.grid, 2);
  y += 36;
  text(ctx, "Hand-written recurrent units", PANEL.x, y, 24, t.ink2);
  y += 10;
  for (const trace of d.traces) {
    const v = trace.values[i], lo = trace.min, hi = trace.max, gx = PANEL.x, gw = PANEL.w;
    text(ctx, trace.label, gx, y + 28, 24, t.ink);
    text(ctx, trace.unit, gx + gw, y + 28, 22, t.muted, "400", "right");
    ctx.fillStyle = t.grid;
    roundRect(ctx, gx, y + 42, gw, 14, 7);
    ctx.fill();
    const zero = gx + (0 - lo) / (hi - lo) * gw, at = gx + (Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo) * gw;
    ctx.fillStyle = lo < 0 ? (v >= 0 ? t.home : t.away) : t.dead;
    roundRect(ctx, Math.min(zero, at), y + 42, Math.max(4, Math.abs(at - zero)), 14, 7);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(at, y + 49, 11, 0, Math.PI * 2);
    ctx.fillStyle = t.ink;
    ctx.fill();
    ctx.lineWidth = 3;
    ctx.strokeStyle = t.page;
    ctx.stroke();
    if (lo < 0) {
      text(ctx, "away", gx, y + 80, 20, t.muted);
      text(ctx, "home", gx + gw, y + 80, 20, t.muted, "400", "right");
    }
    y += lo < 0 ? 96 : 78;
  }

  // ---- timeline: first trace, then the three label tracks, with the ball gaps shaded
  const T = TIME, tx = k => T.x + k / (d.frames - 1) * T.w;
  const trace = d.traces[0];
  if (trace) {
    const top = T.y, h = 58, mid = v => top + h - (v - trace.min) / (trace.max - trace.min) * h;
    ctx.fillStyle = t.gap;
    for (const [a, b] of d.gaps) ctx.fillRect(tx(a), top, tx(b) - tx(a), h);
    line(ctx, T.x, mid(0), T.x + T.w, mid(0), t.axis, 1.5);
    ctx.beginPath();
    for (let k = 0; k < d.frames; k += 2) ctx[k ? "lineTo" : "moveTo"](tx(k), mid(Math.max(trace.min, Math.min(trace.max, trace.values[k]))));
    ctx.strokeStyle = t.ink;
    ctx.lineWidth = 2.5;
    ctx.lineJoin = "round";
    ctx.stroke();
    text(ctx, trace.label.split(" (")[0], 48, top + 26, 21, t.ink2);
    text(ctx, "shaded: ball not seen", 48, top + 52, 19, t.muted);
  }
  const tracks = [["Reference", d.labels.truth], ["Written by hand", d.labels.net], ["Trained", d.labels.trained]];
  tracks.forEach(([name, labels], r) => {
    const top = T.y + 70 + r * 32;
    text(ctx, name, 48, top + 20, 22, t.ink2, name === "Written by hand" ? "600" : "400");
    let start = 0;
    for (let k = 1; k <= d.frames; k++) {
      if (k < d.frames && labels[k] === labels[start]) continue;
      ctx.fillStyle = colours[labels[start]];
      const x0 = tx(start), x1 = tx(Math.min(k, d.frames - 1));
      ctx.fillRect(x0 + 1, top, Math.max(1, x1 - x0 - 2), 24);       // 2 px of surface between runs
      start = k;
    }
  });
  const hx = tx(f);
  line(ctx, hx, T.y - 6, hx, T.y + 70 + 3 * 32 - 4, t.ink, 3);
  ctx.beginPath();
  ctx.arc(hx, T.y - 6, 7, 0, Math.PI * 2);
  ctx.fillStyle = t.ink;
  ctx.fill();
}

function text(ctx, s, x, y, size, colour, weight = "400", align = "left") {
  ctx.font = `${weight} ${size}px ${FONT}`;
  ctx.fillStyle = colour;
  ctx.textAlign = align;
  ctx.textBaseline = "alphabetic";
  ctx.fillText(s, x, y);
}

function line(ctx, x0, y0, x1, y1, colour, width) {
  if (colour) ctx.strokeStyle = colour;
  if (width) ctx.lineWidth = width;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x1, y1);
  ctx.stroke();
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.roundRect(x, y, w, h, Math.min(r, Math.abs(w) / 2, Math.abs(h) / 2));
}

function pill(ctx, cx, cy, s, t) {
  ctx.font = `400 24px ${FONT}`;
  const w = ctx.measureText(s).width + 36;
  ctx.fillStyle = t.page;
  roundRect(ctx, cx - w / 2, cy - 22, w, 44, 22);
  ctx.fill();
  ctx.strokeStyle = t.axis;
  ctx.lineWidth = 2;
  ctx.stroke();
  text(ctx, s, cx, cy + 8, 24, t.ink2, "400", "center");
}
