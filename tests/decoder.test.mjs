// node tests/decoder.test.mjs <ref.json> — replays the Python token stream through NoteDecoder.
import fs from 'node:fs';
import { NoteDecoder } from '../engine/decoder.js';

const ref = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seeks = ref.tokens.filter((t) => Array.isArray(t)).map((t) => t[1]);
const dec = new NoteDecoder();
const evs = [];
let chunk = 0;
for (const t of ref.tokens) {
  if (Array.isArray(t)) {
    evs.push(...dec.boundary(t[1], chunk + 1 < seeks.length ? seeks[chunk + 1] : null));
    chunk++;
  } else evs.push(...dec.token(t));
}
evs.push(...dec.finish());
const got = evs.map((e) => (e.type === 'start'
  ? ['s', e.index, e.pitch, +e.time.toFixed(4), e.instrument]
  : ['e', e.index, +e.time.toFixed(4)]));
const a = JSON.stringify(got), b = JSON.stringify(ref.events);
if (a !== b) {
  const i = got.findIndex((g, k) => JSON.stringify(g) !== JSON.stringify(ref.events[k]));
  console.error('MISMATCH at', i, got[i], ref.events[i]);
  process.exit(1);
}
console.log(`decoder OK: ${got.length} events match`);
