// Screenshots and replay videos of the running app, saved to media/ for slides.
//
//   dotnet run --project src/WeightSpace.Web --no-launch-profile     (with ASPNETCORE_URLS=http://localhost:5199)
//   cd tools/capture && npm install
//   node capture.mjs shots                      pages and canvases as PNG
//   node capture.mjs videos [expA expB ...]     the pitch replay as MP4, light and dark stage
//
// It drives the Edge (or Chrome) that is already installed; nothing is downloaded. Videos are
// recorded by the page itself (canvas -> MediaRecorder), in real time: 100 s each.

import { chromium } from "playwright-core";
import { mkdirSync } from "node:fs";
import { resolve, dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const base = process.env.WEIGHTSPACE_URL ?? "http://localhost:5199";
const media = resolve(dirname(fileURLToPath(import.meta.url)), "..", "..", "media");
mkdirSync(media, { recursive: true });

async function launch() {
  for (const channel of ["msedge", "chrome"]) {
    try { return await chromium.launch({ channel, headless: true, args: ["--autoplay-policy=no-user-gesture-required"] }); }
    catch { /* try the next browser */ }
  }
  throw new Error("neither Edge nor Chrome could be started");
}

const [mode = "shots", ...rest] = process.argv.slice(2);
const browser = await launch();
const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1, acceptDownloads: true });
const page = await context.newPage();
page.on("pageerror", e => console.log("page error:", e.message));
page.on("console", m => { if (m.type() === "error") console.log("console error:", m.text()); });

const canvasPng = async (file, selector = "canvas.stage") => {
  const url = await page.evaluate(sel => document.querySelector(sel).toDataURL("image/png"), selector);
  const { writeFileSync } = await import("node:fs");
  writeFileSync(join(media, file), Buffer.from(url.split(",")[1], "base64"));
  console.log("wrote", file);
};

if (mode === "shots") {
  await page.goto(base + "/", { waitUntil: "networkidle" });
  await page.screenshot({ path: join(media, "page-home.png"), fullPage: true });

  await page.goto(base + "/results", { waitUntil: "networkidle" });
  await page.screenshot({ path: join(media, "page-results.png"), fullPage: true });

  for (const [net, theme, second] of [["expA", "light", 14], ["expB", "light", 14], ["expB", "dark", 62]]) {
    await page.goto(`${base}/replay?net=${net}&theme=${theme}`, { waitUntil: "networkidle" });
    await page.waitForFunction(() => window.weightSpaceReplay?.ready && window.weightSpaceReplay.state.data);
    await page.evaluate(s => window.weightSpaceReplay.seek(s / 100), second);
    await page.waitForTimeout(200);
    await canvasPng(`replay-${net}-${theme}-${second}s.png`);
  }
  await page.goto(`${base}/replay?net=expB`, { waitUntil: "networkidle" });
  await page.waitForFunction(() => window.weightSpaceReplay?.ready);
  await page.screenshot({ path: join(media, "page-replay.png"), fullPage: true });

  for (const [source, file] of [["A", "weights-expA.png"], ["B", "weights-expB.png"]]) {
    await page.goto(base + "/write", { waitUntil: "networkidle" });
    await page.waitForTimeout(800);
    await page.selectOption("select", source);
    await page.getByRole("button", { name: "Write the network" }).click();
    await page.getByText("Watch this network play").waitFor({ timeout: 120_000 });
    await page.waitForTimeout(1200);                                  // let the flashes settle
    await canvasPng(file);
    if (source === "A") await page.screenshot({ path: join(media, "page-write.png"), fullPage: true });
  }
} else if (mode === "videos") {
  const networks = rest.length ? rest : ["expA", "expB"];
  for (const net of networks) {
    for (const theme of ["light", "dark"]) {
      await page.goto(`${base}/replay?net=${net}&theme=${theme}`, { waitUntil: "networkidle" });
      await page.waitForFunction(() => window.weightSpaceReplay?.ready && window.weightSpaceReplay.state.data);
      const download = page.waitForEvent("download", { timeout: 180_000 });
      await page.evaluate(() => { window.weightSpaceReplay.record(); });
      const file = await download;
      await file.saveAs(join(media, file.suggestedFilename()));
      console.log("wrote", file.suggestedFilename());
    }
  }
}
await browser.close();
