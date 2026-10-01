// Guitar / bass tablature: give every note a string and fret the way a player would.
//
// The model is a fretting hand: it sits in a position (index finger on one fret, a finger
// per fret above it) and shifts only when it must, the more reluctantly the faster the
// music; reach is measured physically (frets narrow up the neck); notes still ringing
// aren't cut off by the next note on the same string, so arpeggios spread over strings.
// A Viterbi search finds the cheapest path through each run of notes:
// - live: each new stretch of notes, continuing from where the hand was (what is already
//   on screen never reshuffles);
// - after refine: the whole part at once.

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
const B_PENALTY = 4; // about a hand shift: avoided whenever another string is in reach
const CANDIDATES = 10; // shapes kept per group for the whole-part search

/**
 * Strings for a part: bass parts get a bass, everything else a guitar. A guitar part that
 * goes below low E gets a 7-string (low B); a bass part below low E gets a 5-string.
 */
export function tuningFor(instrument, lowest = Infinity) {
  if (BASS_PARTS.has(instrument)) return lowest < BASS[0] ? BASS_5 : BASS;
  if (GUITAR_PARTS.has(instrument) && lowest < GUITAR[0]) return GUITAR_7;
  return GUITAR;
}

/** String index a user pinned a note to (stored as that string's open pitch), or -1. */
const forcedIndex = (n, tuning) => (n.forcedOpen == null ? -1 : tuning.indexOf(n.forcedOpen));

/**
 * Shapes for a group: notes the user pinned stay on their string; notes no string can
 * reach are left out (null) on their own instead of blanking the whole chord.
 * Returns [{positions: Map note -> pos|null, shape: [...pos]}].
 */
function groupShapes(group, tuning, taken = new Set()) {
  const fixed = new Map();
  const used = new Set(taken);
  for (const n of group) {
    const st = forcedIndex(n, tuning);
    const fret = n.pitch - tuning[st];
    if (st >= 0 && fret >= 0 && fret <= MAX_FRET && !used.has(st)) { fixed.set(n, { string: st, fret }); used.add(st); }
  }
  const reachable = (n) => tuning.some((open) => n.pitch - open >= 0 && n.pitch - open <= MAX_FRET);
  const rest = group.filter((n) => !fixed.has(n) && reachable(n)).sort((a, b) => a.pitch - b.pitch);
  return shapes(rest, tuning, used).map((shape) => {
    const positions = new Map(fixed);
    rest.forEach((n, j) => positions.set(n, shape[j] ?? null));
    for (const n of group) if (!positions.has(n)) positions.set(n, null);
    return { positions, shape: [...positions.values()] };
  });
}

/**
 * Physical reach between frets lo..hi, in widths of the first fret: frets get narrower up
 * the neck, so a five-fret span is a stretch at fret 1 but comfortable at fret 12.
 */
const reach = (lo, hi) => (2 ** (-(lo - 1) / 12) - 2 ** (-hi / 12)) / (1 - 2 ** (-1 / 12));
const COMFORT = 3.8; // one finger per fret over four frets in first position

/** How hard a shape is to hold, on its own (lower is easier). */
function shapeCost(shape, tuning) {
  const placed = shape.filter(Boolean);
  const fretted = placed.filter((p) => p.fret > 0);
  const bass = tuning[0] < 35 && tuning.length <= 5;
  let cost = 0;
  if (fretted.length) {
    const frets = fretted.map((p) => p.fret);
    const lo = Math.min(...frets), hi = Math.max(...frets);
    cost += Math.max(0, reach(lo, hi) - COMFORT) * 4; // stretch
    // Fingers: notes on the lowest fret can share one finger (a barre).
    const fingers = 1 + frets.filter((f) => f !== lo).length;
    if (fingers > 4) cost += (fingers - 4) * 6;
  }
  for (const p of placed) {
    // Up the neck: less comfortable, and past fret 15 hard to reach (more so on a bass).
    if (p.fret > 12) cost += (p.fret - 12) * (bass ? 0.25 : 0.15) + Math.max(0, p.fret - 15) * 0.3;
    if (p.fret === 0) cost -= 0.2; // open strings ring and need no finger
    // Guitar B string: for single notes and two-note lines, only when nothing else works.
    // Chords keep their standard shapes (open C, Am, E, G need it).
    if (placed.length <= 2 && tuning[p.string] === B_STRING && tuning.includes(64)) cost += B_PENALTY;
    cost += p.fret * 0.01; // tie-break toward lower positions
  }
  if (placed.length >= 2) {
    const sorted = [...placed].sort((x, y) => x.string - y.string);
    // Power chord / octave shape (root, fifth two frets up on the next string): the classic grip.
    const [r, f] = sorted;
    if (f.string === r.string + 1 && f.fret === r.fret + 2 && r.fret > 0) cost -= 0.4;
    // Strummed chords: strings skipped inside the shape are awkward.
    if (placed.length >= 3) {
      const used = new Set(placed.map((p) => p.string));
      for (let st = sorted[0].string; st <= sorted[sorted.length - 1].string; st++) if (!used.has(st)) cost += 0.6;
    }
  }
  cost += shape.filter((p) => p === null).length * 3; // notes left out
  return cost;
}

/** Fretted range of a shape ({lo, hi}, or nulls when it is all open strings). */
function span(shape) {
  let lo = null, hi = null;
  for (const p of shape) {
    if (!p || p.fret === 0) continue;
    lo = lo == null ? p.fret : Math.min(lo, p.fret);
    hi = hi == null ? p.fret : Math.max(hi, p.fret);
  }
  return { lo, hi };
}

/** Shifting the hand costs more the faster the passage. */
const urgency = (gap) => (gap < 0.15 ? 2 : gap < 0.3 ? 1.2 : gap < 0.6 ? 0.7 : gap < 1.5 ? 0.4 : 0.2);
const RING = 0.12; // seconds a note must still ring for cutting it to matter

/**
 * The step from where the player is (`carry`: hand position = index-finger fret, when the
 * hand last fretted, and the strings still ringing) to shape `c` at time `t`.
 * - The hand covers its position .. +3, with a small stretch either way; anything else is a
 *   shift, dearer the less time there is since the hand was last busy, so an open string in
 *   between buys time to move, as players use them.
 * - Open strings fit naturally near the nut; high up the neck they look out of place.
 * - Putting a note on a string that is still ringing cuts it short: arpeggios spread over
 *   strings, the way players let them ring.
 */
function step(carry, c, t) {
  let cost = 0, hand = carry.hand, busy = carry.busy;
  if (c.lo != null) {
    if (hand == null) hand = c.lo;
    else if (c.lo >= hand - 1 && c.hi <= hand + 4 && reach(Math.min(c.lo, hand), c.hi) <= COMFORT + 1.2) {
      if (c.lo < hand) cost += 0.3; // index reaches back
      if (c.hi > hand + 3) cost += 0.5; // pinky reaches forward
    } else {
      cost += urgency(t - busy) * (0.8 + 0.3 * Math.abs(c.lo - hand));
      hand = c.lo;
    }
    busy = t;
  }
  for (const p of c.shape) {
    if (!p) continue;
    if (p.fret === 0 && c.shape.length === 1) { // single notes (chords have their own shapes)
      const where = hand ?? 0;
      if (where <= 4) cost -= 0.4; // first position: open strings are the natural choice
      else if (where >= 9) cost += 0.4; // far up the neck: an open string is out of place
    }
    if ((carry.ring.get(p.string) ?? -Infinity) > t + RING) cost += 1.2;
  }
  return { cost, hand, busy };
}

/** Strings ringing after shape `c` (notes `positions`) is played at time t. */
function ringAfter(ring, positions, t) {
  const out = new Map();
  for (const [st, end] of ring) if (end > t) out.set(st, end);
  for (const [n, p] of positions) if (p) out.set(p.string, n.end ?? n.start + 1);
  return out;
}

/**
 * Best fingering for a run of note groups (Viterbi over each group's cheapest shapes):
 * shape difficulty plus hand shifts plus cut-off ringing notes, along the whole run.
 * `start` is the player's state before the run; `taken(i)` the strings group i can't use.
 * Returns one Map note -> position per group.
 */
function solve(groups, tuning, start, taken = () => undefined) {
  const cands = groups.map((g, i) => groupShapes(g, tuning, taken(i))
    .map((c) => ({ ...c, cost: shapeCost(c.shape, tuning), ...span(c.shape) }))
    .sort((x, y) => x.cost - y.cost)
    .slice(0, CANDIDATES));
  const backs = [];
  let prev = [{ total: 0, carry: start }];
  groups.forEach((g, i) => {
    const t = g[0].start;
    const cur = cands[i].map((c) => {
      let best = null;
      prev.forEach((p, j) => {
        const s = step(p.carry, c, t);
        const total = p.total + s.cost;
        if (!best || total < best.total) best = { total, back: j, hand: s.hand, busy: s.busy, ring: p.carry.ring };
      });
      return { total: best.total + c.cost, back: best.back, carry: { hand: best.hand, busy: best.busy, ring: ringAfter(best.ring, c.positions, t) } };
    });
    backs.push(cur);
    prev = cur;
  });
  const out = new Array(groups.length);
  let j = prev.reduce((b, x, k) => (x.total < prev[b].total ? k : b), 0);
  for (let i = groups.length - 1; i >= 0; i--) {
    out[i] = cands[i][j]?.positions ?? new Map();
    j = backs[i][j].back;
  }
  return out;
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

  /** Drop a note's position (its pitch was edited): it gets fingered again. */
  forget(note, instrument) {
    this.parts.get(instrument)?.pos.delete(note);
  }

  /**
   * Put a note on the next string in `dir` (+1 = higher string) that can play it and isn't
   * used by another note struck at the same time. Returns the new position, or null.
   */
  moveString(note, instrument, list, dir, tuning = tuningFor(instrument)) {
    const { pos } = this._part(instrument, tuning);
    const cur = pos.get(note);
    const taken = new Set(list.filter((m) => m !== note && Math.abs(m.start - note.start) < TOGETHER)
      .map((m) => pos.get(m)?.string).filter((x) => x != null));
    for (let st = (cur?.string ?? (dir > 0 ? -1 : tuning.length)) + dir; st >= 0 && st < tuning.length; st += dir) {
      const fret = note.pitch - tuning[st];
      if (fret < 0 || fret > MAX_FRET || taken.has(st)) continue;
      const p = { string: st, fret };
      pos.set(note, p);
      note.forcedOpen = tuning[st]; // pinned by its open pitch: survives a 6 <-> 7 string change
      return p;
    }
    return null;
  }

  /** Position of a note assigned earlier, or undefined. */
  get(note, instrument) {
    return this.parts.get(instrument)?.pos.get(note);
  }

  /**
   * Live: assign positions to notes of one part that start in [from, to]. `list` is the
   * part's notes sorted by start. Notes already placed keep their place; each new stretch
   * of notes (one chunk as it arrives) is solved as a whole, starting from where the hand was.
   */
  assign(instrument, list, from, to, tuning = tuningFor(instrument)) {
    const { pos } = this._part(instrument, tuning);
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start < from) lo = mid + 1; else hi = mid; }
    const groups = groupsOf(list, lo, to);
    for (let i = 0; i < groups.length;) {
      if (groups[i].every((n) => pos.has(n))) { i++; continue; }
      let k = i;
      while (k < groups.length && groups[k].some((n) => !pos.has(n))) k++;
      const run = groups.slice(i, k);
      const placed = solve(
        run.map((g) => g.filter((n) => !pos.has(n))),
        tuning,
        this._before(list, pos, run[0][0].start),
        (j) => new Set(run[j].map((n) => pos.get(n)?.string).filter((x) => x != null)),
      );
      run.forEach((g, j) => { for (const n of g) if (!pos.has(n)) pos.set(n, placed[j].get(n) ?? null); });
      i = k;
    }
  }

  /** The player's state just before time t: hand position and strings still ringing. */
  _before(list, pos, t) {
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start < t) lo = mid + 1; else hi = mid; }
    let hand = null, busy = -Infinity;
    const ring = new Map();
    for (let j = lo - 1; j >= 0 && j >= lo - 24; j--) {
      const p = pos.get(list[j]);
      if (!p) continue;
      if (hand == null && p.fret > 0) { hand = p.fret; busy = list[j].start; }
      const end = list[j].end ?? list[j].start + 1;
      if (end > t && !ring.has(p.string)) ring.set(p.string, end);
    }
    return { hand, busy, ring };
  }

  /** Whole part at once (after refine): the same search over the entire song. */
  optimize(instrument, list, tuning = tuningFor(instrument)) {
    const part = { tuning, pos: new WeakMap() };
    this.parts.set(instrument, part);
    const groups = groupsOf(list);
    if (!groups.length) return;
    const placed = solve(groups, tuning, { hand: null, busy: -Infinity, ring: new Map() });
    groups.forEach((g, i) => { for (const n of g) part.pos.set(n, placed[i].get(n) ?? null); });
  }
}
