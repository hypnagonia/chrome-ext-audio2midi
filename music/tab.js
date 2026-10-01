// Guitar / bass tablature: give every note a string and fret the way a player would.
//
// One scoring model, used two ways:
// - live: each new note group gets the cheapest shape given where the hand is now
//   (stable: what is on screen never reshuffles);
// - after refine: a Viterbi search over the whole part picks the cheapest path through
//   all groups, so riffs stay in one position and the hand doesn't jump around.

const GUITAR = [40, 45, 50, 55, 59, 64]; // E2 A2 D3 G3 B3 E4, low to high
const GUITAR_7 = [35, ...GUITAR]; // + low B1
const BASS = [28, 33, 38, 43]; // E1 A1 D2 G2
const BASS_5 = [23, ...BASS]; // + low B0
const BASS_PARTS = new Set(['acoustic_bass', 'electric_bass', 'contrabass']);
const GUITAR_PARTS = new Set(['acoustic_guitar', 'clean_electric_guitar', 'distorted_electric_guitar']);
const NAMES = { 35: 'B', 23: 'B', 40: 'E', 28: 'E', 45: 'A', 33: 'A', 50: 'D', 38: 'D', 55: 'G', 43: 'G', 59: 'B', 64: 'e' };
export const stringNames = (tuning) => tuning.map((p) => NAMES[p]);
const MAX_FRET = 22;
const TOGETHER = 0.03; // seconds: onsets this close are one chord
const B_STRING = 59; // the guitar's B string: a last resort
const B_PENALTY = 25;
const CANDIDATES = 10; // shapes kept per group for the whole-part search

/**
 * Strings for a part: bass parts get a bass, everything else a guitar. A guitar part that
 * goes below low E gets a 7-string (low B); a bass part below low E gets a 5-string.
 */
export function tuningFor(instrument, lowest = Infinity) {
  if (BASS_PARTS.has(instrument)) return lowest < BASS[0] && lowest >= BASS_5[0] ? BASS_5 : BASS;
  if (GUITAR_PARTS.has(instrument) && lowest < GUITAR[0] && lowest >= GUITAR_7[0]) return GUITAR_7;
  return GUITAR;
}

/** How hard a shape is to hold, on its own (lower is easier). */
function shapeCost(shape, tuning) {
  const placed = shape.filter(Boolean);
  const frets = placed.filter((p) => p.fret > 0).map((p) => p.fret);
  const lo = frets.length ? Math.min(...frets) : 0;
  const span = frets.length ? Math.max(...frets) - lo : 0;
  let cost = Math.max(0, span - 3) * 4; // stretch beyond four frets
  // Fingers: notes on the lowest fret can share one finger (a barre).
  const fingers = frets.length ? 1 + frets.filter((f) => f !== lo).length : 0;
  if (fingers > 4) cost += (fingers - 4) * 6;
  for (const p of placed) {
    if (p.fret > 15) cost += 1; // very high frets
    if (p.fret === 0) cost -= 0.2; // open strings ring and need no finger
    // Guitar B string: for single notes and two-note lines, only when nothing else works.
    // Chords keep their standard shapes (open C, Am, E, G need it).
    if (placed.length <= 2 && tuning[p.string] === B_STRING && tuning.includes(64)) cost += B_PENALTY;
    cost += p.fret * 0.01; // tie-break toward lower positions
  }
  // Strummed chords: strings skipped inside the shape are awkward.
  if (placed.length >= 3) {
    const used = new Set(placed.map((p) => p.string));
    const sMin = Math.min(...used), sMax = Math.max(...used);
    for (let st = sMin; st <= sMax; st++) if (!used.has(st)) cost += 0.6;
  }
  cost += shape.filter((p) => p === null).length * 3; // notes left out
  return cost;
}

/** Where the hand sits for a shape: its lowest fretted fret (null when it's all open strings). */
const anchor = (shape) => {
  const frets = shape.filter((p) => p && p.fret > 0).map((p) => p.fret);
  return frets.length ? Math.min(...frets) : null;
};

/** Moving the hand costs more the faster the passage. */
function moveCost(from, to, gap) {
  if (from == null || to == null) return 0;
  return Math.abs(from - to) * (gap < 0.25 ? 1 : gap < 1 ? 0.6 : 0.3);
}

/**
 * Every playable shape for notes struck together (sorted low to high): lower notes on lower
 * strings, one string each, avoiding `taken` strings. If they can't all fit, the lowest
 * notes are left out (null), as they are usually doubled by the bass.
 */
function shapes(notes, tuning, taken = new Set()) {
  if (!notes.length) return [[]];
  const out = [];
  const pick = [];
  const search = (i, minString) => {
    if (out.length > 400) return;
    if (i === notes.length) { out.push([...pick]); return; }
    for (let st = minString; st < tuning.length; st++) {
      const fret = notes[i].pitch - tuning[st];
      if (fret < 0 || fret > MAX_FRET || taken.has(st)) continue;
      pick.push({ string: st, fret });
      search(i + 1, st + 1);
      pick.pop();
    }
  };
  search(0, 0);
  if (out.length) return out;
  return shapes(notes.slice(1), tuning, taken).map((s) => [null, ...s]);
}

/** Note groups (struck together) of a part, from index i0, sorted by start. */
function groupsOf(list, i0 = 0, until = Infinity) {
  const groups = [];
  for (let i = i0; i < list.length && list[i].start <= until;) {
    let k = i + 1;
    while (k < list.length && list[k].start - list[i].start < TOGETHER) k++;
    groups.push(list.slice(i, k));
    i = k;
  }
  return groups;
}

export class Fingering {
  constructor() {
    this.parts = new Map(); // instrument -> {tuning, pos: WeakMap note -> {string, fret} | null}
  }

  _part(instrument, tuning) {
    let part = this.parts.get(instrument);
    // A new tuning (e.g. a low B showed up: 6 -> 7 strings) re-fingers the whole part.
    if (!part || part.tuning !== tuning) {
      part = { tuning, pos: new WeakMap() };
      this.parts.set(instrument, part);
    }
    return part;
  }

  /** Position of a note assigned earlier, or undefined. */
  get(note, instrument) {
    return this.parts.get(instrument)?.pos.get(note);
  }

  /**
   * Live: assign positions to notes of one part that start in [from, to]. `list` is the
   * part's notes sorted by start. Notes already placed keep their place.
   */
  assign(instrument, list, from, to, tuning = tuningFor(instrument)) {
    const { pos } = this._part(instrument, tuning);
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start < from) lo = mid + 1; else hi = mid; }
    // Hand position carried over from just before the window.
    let hand = null, last = -Infinity;
    for (let j = lo - 1; j >= 0 && j >= lo - 16; j--) {
      const p = pos.get(list[j]);
      if (p && p.fret > 0) { hand = p.fret; last = list[j].start; break; }
    }
    for (const group of groupsOf(list, lo, to)) {
      const todo = group.filter((n) => !pos.has(n)).sort((a, b) => a.pitch - b.pitch);
      if (todo.length) {
        const taken = new Set(group.map((n) => pos.get(n)?.string).filter((x) => x != null));
        const gap = group[0].start - last;
        let best = null;
        for (const shape of shapes(todo, tuning, taken)) {
          const cost = shapeCost(shape, tuning) + moveCost(hand, anchor(shape), gap);
          if (!best || cost < best.cost) best = { cost, shape };
        }
        todo.forEach((n, j) => pos.set(n, best?.shape[j] ?? null));
      }
      const a = anchor(group.map((n) => pos.get(n)).filter(Boolean));
      if (a != null) hand = a;
      last = group[0].start;
    }
  }

  /**
   * Whole part at once (after refine): Viterbi over note groups, each with its cheapest
   * candidate shapes, minimising shape difficulty plus hand movement along the song.
   */
  optimize(instrument, list, tuning = tuningFor(instrument)) {
    const part = { tuning, pos: new WeakMap() };
    this.parts.set(instrument, part);
    const groups = groupsOf(list).map((g) => [...g].sort((a, b) => a.pitch - b.pitch));
    if (!groups.length) return;
    const cands = groups.map((g) => shapes(g, tuning)
      .map((shape) => ({ shape, cost: shapeCost(shape, tuning), a: anchor(shape) }))
      .sort((x, y) => x.cost - y.cost)
      .slice(0, CANDIDATES));
    let prev = cands[0].map((c) => ({ total: c.cost, back: -1 }));
    const backs = [prev];
    for (let i = 1; i < groups.length; i++) {
      const gap = groups[i][0].start - groups[i - 1][0].start;
      const cur = cands[i].map((c) => {
        let best = { total: Infinity, back: 0 };
        cands[i - 1].forEach((p, j) => {
          const t = prev[j].total + moveCost(p.a, c.a, gap);
          if (t < best.total) best = { total: t, back: j };
        });
        return { total: best.total + c.cost, back: best.back };
      });
      backs.push(cur);
      prev = cur;
    }
    let j = prev.reduce((b, x, k) => (x.total < prev[b].total ? k : b), 0);
    for (let i = groups.length - 1; i >= 0; i--) {
      const shape = cands[i][j]?.shape;
      groups[i].forEach((n, k) => part.pos.set(n, shape?.[k] ?? null));
      j = backs[i][j].back;
    }
  }
}
