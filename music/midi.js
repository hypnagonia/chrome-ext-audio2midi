// Standard MIDI file (type 1) writer: one track per instrument, GM program
// changes, drums on channel 10. Same layout as muscriptor's note_event2midi.

import { DRUM_PROGRAM, programForInstrument } from '../engine/vocab.js';

const TPB = 480;
const TEMPO = 500000; // 120 bpm, so 1 s = 960 ticks

function vlq(n) {
  const bytes = [n & 0x7f];
  while ((n >>= 7)) bytes.unshift((n & 0x7f) | 0x80);
  return bytes;
}

function track(events) {
  const body = [];
  for (const e of events) body.push(...vlq(e.delta), ...e.bytes);
  body.push(0, 0xff, 0x2f, 0);
  const len = body.length;
  return [0x4d, 0x54, 0x72, 0x6b, (len >>> 24) & 255, (len >>> 16) & 255, (len >>> 8) & 255, len & 255, ...body];
}

const text = (s) => Array.from(new TextEncoder().encode(s));

/** notes: [{instrument, pitch, start, end}] (seconds) -> Uint8Array */
export function notesToMidi(notes) {
  const byInst = new Map();
  for (const n of notes) {
    if (!byInst.has(n.instrument)) byInst.set(n.instrument, []);
    byInst.get(n.instrument).push(n);
  }
  const free = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15];
  const byProgram = new Map(); // tracks with the same GM sound can share a channel
  // More instruments than channels: share by program, else reuse round-robin.
  const channelFor = (program) => {
    if (byProgram.has(program)) return byProgram.get(program);
    const ch = free.length ? free.shift() : [...byProgram.values()][byProgram.size % 15];
    byProgram.set(program, ch);
    return ch;
  };
  const tracks = [];
  const tempo = [{ delta: 0, bytes: [0xff, 0x51, 3, (TEMPO >> 16) & 255, (TEMPO >> 8) & 255, TEMPO & 255] }];
  tracks.push(track(tempo));
  const toTick = (s) => Math.max(0, Math.round((s * TPB * 1e6) / TEMPO));
  for (const [inst, list] of byInst) {
    const program = programForInstrument(inst);
    const drums = program === DRUM_PROGRAM;
    const ch = drums ? 9 : channelFor(program);
    const name = text(inst.replace(/_/g, ' '));
    const raw = [];
    // Same pitch overlapping itself: end the earlier note where the next one starts,
    // otherwise its note-off would cut the later note short.
    const sorted = [...list].sort((a, b) => a.start - b.start);
    const nextStart = new Map();
    for (let i = sorted.length - 1; i >= 0; i--) {
      const n = sorted[i];
      const on = toTick(n.start);
      let off = toTick(drums ? n.start + 0.01 : Math.max(n.end ?? n.start + 0.01, n.start + 0.01));
      if (nextStart.has(n.pitch)) off = Math.min(off, nextStart.get(n.pitch));
      nextStart.set(n.pitch, on);
      if (off <= on) continue;
      raw.push({ tick: on, on: true, pitch: n.pitch }, { tick: off, on: false, pitch: n.pitch });
    }
    raw.sort((a, b) => a.tick - b.tick || (a.on === b.on ? 0 : a.on ? 1 : -1));
    const evs = [
      { delta: 0, bytes: [0xff, 0x03, ...vlq(name.length), ...name] },
      { delta: 0, bytes: [0xc0 | ch, drums ? 0 : program & 0x7f] },
    ];
    let last = 0;
    for (const r of raw) {
      evs.push({ delta: r.tick - last, bytes: [(r.on ? 0x90 : 0x80) | ch, r.pitch, r.on ? 100 : 0] });
      last = r.tick;
    }
    tracks.push(track(evs));
  }
  const header = [0x4d, 0x54, 0x68, 0x64, 0, 0, 0, 6, 0, 1, (tracks.length >> 8) & 255, tracks.length & 255, (TPB >> 8) & 255, TPB & 255];
  return new Uint8Array([...header, ...tracks.flat()]);
}
