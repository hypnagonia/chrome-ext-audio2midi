// node tools/build.mjs — checks every UI locale against en.json, writes Chrome's
// _locales/<code>/messages.json (store listing) and packages dist/byear-<version>.zip.
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const dir = path.join(root, 'ui/locales');
const en = JSON.parse(fs.readFileSync(path.join(dir, 'en.json'), 'utf8'));
const holes = (s) => (s.match(/\{\w+\}/g) || []).sort().join();
let ok = true;
const fail = (msg) => { ok = false; console.error('✗', msg); };

const codes = fs.readdirSync(dir).filter((f) => f.endsWith('.json')).map((f) => f.slice(0, -5)).sort();
fs.rmSync(path.join(root, '_locales'), { recursive: true, force: true });
for (const code of codes) {
  const d = JSON.parse(fs.readFileSync(path.join(dir, `${code}.json`), 'utf8'));
  const missing = Object.keys(en).filter((k) => !(k in d));
  const extra = Object.keys(d).filter((k) => !(k in en));
  if (missing.length) fail(`${code}: missing ${missing.join(', ')}`);
  if (extra.length) fail(`${code}: unknown keys ${extra.join(', ')}`);
  for (const k of Object.keys(en)) if (k in d && holes(d[k]) !== holes(en[k])) fail(`${code}.${k}: placeholders ${holes(d[k])} vs ${holes(en[k])}`);
  if (!d['manifest.name'].startsWith('byEar')) fail(`${code}: name must start with byEar`);
  if ([...d['manifest.name']].length > 75) fail(`${code}: name over 75 chars`);
  if ([...d['manifest.description']].length > 132) fail(`${code}: description over 132 chars`);
  const messages = {
    extName: { message: d['manifest.name'] },
    extShortName: { message: 'byEar' },
    extDescription: { message: d['manifest.description'] },
    actionTitle: { message: d['manifest.action'] },
  };
  fs.mkdirSync(path.join(root, '_locales', code), { recursive: true });
  fs.writeFileSync(path.join(root, '_locales', code, 'messages.json'), JSON.stringify(messages, null, 2) + '\n');
}
if (!ok) process.exit(1);
console.log(`locales OK: ${codes.join(' ')}`);

const { version } = JSON.parse(fs.readFileSync(path.join(root, 'manifest.json'), 'utf8'));
const zip = path.join(root, 'dist', `byear-${version}.zip`);
fs.mkdirSync(path.dirname(zip), { recursive: true });
fs.rmSync(zip, { force: true });
execFileSync('zip', ['-qr', zip, 'manifest.json', 'background.js', 'capture-worklet.js', 'sidepanel.html', 'sidepanel.css',
  'sidepanel.js', 'engine', 'music', 'ui', 'icons', '_locales', '-x', '.*'], { cwd: root });
console.log(`packaged ${path.relative(root, zip)} (${(fs.statSync(zip).size / 1024).toFixed(0)} KB)`);
