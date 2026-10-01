// Chord and key estimation from transcribed notes (pitch-class template matching).

import { pcName } from './names.js';

// [suffix, intervals, complexity]
const QUALITIES = [
  ['', [0, 4, 7], 0],
  ['m', [0, 3, 7], 0],
  ['7', [0, 4, 7, 10], 1],
  ['maj7', [0, 4, 7, 11], 1],
  ['m7', [0, 3, 7, 10], 1],
  ['6', [0, 4, 7, 9], 1.5],
  ['m6', [0, 3, 7, 9], 1.5],
  ['sus4', [0, 5, 7], 1],
  ['sus2', [0, 2, 7], 1.2],
  ['dim', [0, 3, 6], 1],
  ['m7♭5', [0, 3, 6, 10], 1.5],
  ['dim7', [0, 3, 6, 9], 1.5],
  ['aug', [0, 4, 8], 1.5],
  ['add9', [0, 2, 4, 7], 1.5],
  ['5', [0, 7], 0.5],
];

export const isPitched = (inst) => inst !== 'drums';

/**
 * Duration-weighted pitch-class profile of notes overlapping [t0, t1), plus the
 * most distinct pitches sounding at the same instant (1 for a melody line).
 */
function profile(notes, t0, t1, now) {
  const pc = new Float64Array(12);
  let bass = Infinity;
  let bassW = 0;
  const edges = [];
  for (const n of notes) {
    const end = n.end ?? now;
    const ov = Math.min(end, t1) - Math.max(n.start, t0);
    if (ov <= 0) continue;
    pc[n.pitch % 12] += ov;
    if (n.pitch < bass) { bass = n.pitch; bassW = ov; }
    // Ignore overlaps under 30 ms: legato release tails, not harmony.
    const s0 = Math.max(n.start, t0);
    edges.push([s0, 1], [Math.max(s0 + 1e-3, Math.min(end, t1) - 0.03), -1]);
  }
  edges.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  let cur = 0;
  let poly = 0;
  for (const [, d] of edges) poly = Math.max(poly, (cur += d));
  return { pc, bass: bassW > 0 ? bass % 12 : null, poly };
}

/** Best chord for a pitch-class profile, or null when nothing chord-like sounds. */
export function matchChord(pc, bass) {
  const total = pc.reduce((a, b) => a + b, 0);
  if (total <= 0) return null;
  const norm = Array.from(pc, (v) => v / total);
  const present = norm.filter((v) => v > 0.06).length;
  if (present < 2) return null;
  let best = null;
  for (let root = 0; root < 12; root++) {
    for (const [suffix, iv, complexity] of QUALITIES) {
      if (suffix === '5' && present > 2) continue;
      let covered = 0;
      let missing = 0;
      for (const i of iv) {
        const w = norm[(root + i) % 12];
        covered += w;
        if (w < 0.04) missing++;
      }
      if (missing > (iv.length > 3 ? 1 : 0)) continue;
      let score = covered - 0.15 * missing - 0.03 * complexity;
      if (bass === root) score += 0.06;
      if (!best || score > best.score) best = { root, suffix, score, covered };
    }
  }
  if (!best || best.covered < 0.65) return null;
  return { root: best.root, suffix: best.suffix, bass: bass !== null && bass !== best.root ? bass : null };
}

/** {root, suffix, bass} spelled for display; bass is '' when in root position. */
export function chordParts(ch, useFlats = false) {
  return { root: pcName(ch.root, useFlats), suffix: ch.suffix, bass: ch.bass !== null ? pcName(ch.bass, useFlats) : '' };
}

export function chordName(ch, useFlats = false) {
  if (!ch) return '';
  const p = chordParts(ch, useFlats);
  return p.root + p.suffix + (p.bass ? `/${p.bass}` : '');
}

export const chordKey = (ch) => (ch ? `${ch.root}${ch.suffix}/${ch.bass}` : '');

/**
 * Chord segments for notes within [t0, t1) on a 0.25 s grid, merged when
 * consecutive windows agree. Returns [{start, end, chord}].
 */
export function chordSegments(notes, t0, t1, now = t1) {
  const step = 0.25;
  const wins = [];
  // One sweep through the notes in start order, keeping only those still sounding:
  // linear in the song length instead of scanning every note for every window.
  const sorted = [...notes].sort((a, b) => a.start - b.start);
  let next = 0;
  let active = [];
  for (let t = t0; t < t1 - 1e-6; t += step) {
    while (next < sorted.length && sorted[next].start < t + step) active.push(sorted[next++]);
    active = active.filter((n) => (n.end ?? now) > t);
    const { pc, bass, poly } = profile(active, t, t + step, now);
    wins.push({ start: t, end: t + step, chord: poly >= 2 ? matchChord(pc, bass) : null });
  }
  // A single-window blip between two equal chords (A B A) is passing tones.
  for (let i = 1; i + 1 < wins.length; i++) {
    const a = chordKey(wins[i - 1].chord);
    if (a && a === chordKey(wins[i + 1].chord)) wins[i].chord = wins[i - 1].chord;
  }
  const segs = [];
  for (const w of wins) {
    const last = segs[segs.length - 1];
    if (last && chordKey(last.chord) === chordKey(w.chord)) last.end = w.end;
    else segs.push({ ...w });
  }
  return segs.filter((s) => s.chord);
}

// Krumhansl–Kessler key profiles.
const MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const MINOR = [6.33, 2.68, 3.52, 5.38, 2.6, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

function corr(a, b) {
  const ma = a.reduce((s, v) => s + v, 0) / 12;
  const mb = b.reduce((s, v) => s + v, 0) / 12;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < 12; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return da && db ? num / Math.sqrt(da * db) : 0;
}

/** Estimated key from notes in [t0, t1): {tonic, minor, useFlats} or null. */
export function estimateKey(notes, t0, t1, now = t1) {
  const { pc } = profile(notes, t0, t1, now);
  if (pc.reduce((a, b) => a + b, 0) < 1) return null;
  let best = null;
  for (let k = 0; k < 12; k++) {
    for (const [minor, prof] of [[false, MAJOR], [true, MINOR]]) {
      const rot = Array.from({ length: 12 }, (_, i) => prof[(i - k + 12) % 12]);
      const r = corr(Array.from(pc), rot);
      if (!best || r > best.r) best = { tonic: k, minor, r };
    }
  }
  // Flat keys: F, B♭, E♭, A♭, D♭ major and their relative minors.
  const majorTonic = best.minor ? (best.tonic + 3) % 12 : best.tonic;
  return { ...best, useFlats: [5, 10, 3, 8, 1].includes(majorTonic) };
}
