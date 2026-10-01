// node tests/run-in-chrome.mjs <url> [timeoutSec] — opens url in Chrome via CDP and prints #log once window.testDone.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const CHROME = process.env.CHROME || ['/Applications/Google Chrome.app', '/Applications/Google Chrome 2.app']
  .map((a) => `${a}/Contents/MacOS/Google Chrome`).find((p) => fs.existsSync(p));
const [url, timeout = '300'] = process.argv.slice(2);
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'a2m-chrome-'));
// A fresh port per run: never attach to a stale browser left over from an earlier run.
const port = 9400 + Math.floor(Math.random() * 500);
const proc = spawn(CHROME, [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run',
  '--no-default-browser-check', '--enable-unsafe-webgpu', '--autoplay-policy=no-user-gesture-required', '--mute-audio', process.env.HEADFUL ? '' : '--headless=new', 'about:blank'].filter(Boolean),
  { stdio: 'ignore', detached: true });
const killAll = () => { try { process.kill(-proc.pid, 'SIGKILL'); } catch {} };
process.on('exit', killAll);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let targets;
for (let i = 0; i < 50; i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); break; } catch { await sleep(200); }
}
const page = targets.find((t) => t.type === 'page');
const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener('open', r));
let id = 0;
const waiters = new Map();
ws.addEventListener('message', (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && waiters.has(msg.id)) { waiters.get(msg.id)(msg); waiters.delete(msg.id); }
  if (msg.method === 'Runtime.consoleAPICalled' && process.env.VERBOSE) console.log('[console]', msg.params.args.map((a) => a.value).join(' '));
  if (msg.method === 'Runtime.exceptionThrown') { const d = msg.params.exceptionDetails; console.log('[exception]', d.exception?.description, '@', (d.url || '').split('/').pop() + ':' + (d.lineNumber + 1)); }
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; waiters.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
await send('Runtime.enable');
if (process.env.WIDTH) {
  await send('Emulation.setDeviceMetricsOverride', { width: +process.env.WIDTH, height: +(process.env.HEIGHT || 900), deviceScaleFactor: +(process.env.DPR || 2), mobile: false });
}
await send('Emulation.setEmulatedMedia', { features: [{ name: 'prefers-color-scheme', value: process.env.DARK ? 'dark' : 'light' }] });
if (process.env.INIT) await send('Page.enable'), await send('Page.addScriptToEvaluateOnNewDocument', { source: process.env.INIT });
await send('Page.navigate', { url });
if (process.env.SHOT) {
  // Screenshot mode: wait, capture, print the page text.
  await sleep(+(process.env.AFTER || 20) * 1000);
  if (process.env.EVAL) {
    const r = await send('Runtime.evaluate', { expression: process.env.EVAL, awaitPromise: true, returnByValue: true });
    console.log('EVAL ->', JSON.stringify(r.result?.result?.value ?? r.result?.exceptionDetails?.exception?.description));
    await sleep(+(process.env.EVAL_WAIT || 3) * 1000);
  }
  const shot = await send('Page.captureScreenshot', { format: 'png' });
  fs.writeFileSync(process.env.SHOT, Buffer.from(shot.result.data, 'base64'));
  const r = await send('Runtime.evaluate', { expression: 'document.body.innerText', returnByValue: true });
  console.log(r.result.result.value);
  killAll();
  process.exit(0);
}
const t0 = Date.now();
let out = '';
while (Date.now() - t0 < timeout * 1000) {
  await sleep(1000);
  const r = await send('Runtime.evaluate', { expression: 'window.testDone ? document.getElementById("log").textContent : null', returnByValue: true });
  if (r.result?.result?.value) { out = r.result.result.value; break; }
}
console.log(out || 'TIMEOUT');
killAll();
await sleep(500);
try { fs.rmSync(profile, { recursive: true, force: true }); } catch {}
process.exit(0);
