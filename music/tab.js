// Guitar / bass tablature: assign each note a string and fret, keeping the hand near
// one position and giving notes struck together different strings.

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

/**
 * Strings for a part: bass parts get a bass, everything else a guitar. A guitar part that
 * goes below low E gets a 7-string (low B); a bass part below low E gets a 5-string.
 */
export function tuningFor(instrument, lowest = Infinity) {
  if (BASS_PARTS.has(instrument)) return lowest < BASS[0] && lowest >= BASS_5[0] ? BASS_5 : BASS;
  if (GUITAR_PARTS.has(instrument) && lowest < GUITAR[0] && lowest >= GUITAR_7[0]) return GUITAR_7;
  return GUITAR;
}

/**
 * Strings for notes struck together (sorted low to high): lower notes on lower strings,
 * each on its own string. Picks the shape nearest the hand with the smallest stretch.
 * If not every note fits, drops the lowest ones (they're usually doubled by the bass).
 */
function bestShape(notes, tuning, taken, hand) {
  if (!notes.length) return [];
  let best = null;
  const pick = [];
  const search = (i, minString) => {
    if (i === notes.length) {
      const frets = pick.filter((p) => p.fret > 0).map((p) => p.fret);
      const span = frets.length ? Math.max(...frets) - Math.min(...frets) : 0;
      // Near the hand, small stretch; on a tie the lower position (open strings) wins.
      const cost = pick.reduce((c, p) => c + p.fret * 0.01, 0)
        + frets.reduce((c, f) => c + Math.abs(f - hand) * 0.5 + (f > 15 ? 1 : 0), 0) + Math.max(0, span - 3) * 4;
      if (!best || cost < best.cost) best = { cost, shape: [...pick] };
      return;
    }
    for (let st = minString; st < tuning.length; st++) {
      const fret = notes[i].pitch - tuning[st];
      if (fret < 0 || fret > MAX_FRET || taken.has(st)) continue;
      pick.push({ string: st, fret });
      search(i + 1, st + 1);
      pick.pop();
    }
  };
  search(0, 0);
  if (best) return best.shape;
  // Too many notes for the strings: leave out the lowest and try again.
  const rest = bestShape(notes.slice(1), tuning, taken, hand);
  return rest ? [null, ...rest] : null;
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
   * Assign positions to notes of one instrument that start in [from, to].
   * `list` is that instrument's notes sorted by start. Already assigned notes are kept,
   * so the tab never reshuffles while you watch.
   */
  assign(instrument, list, from, to, tuning = tuningFor(instrument)) {
    const { pos } = this._part(instrument, tuning);
    this.pos = pos;
    // First note at or after `from`.
    let lo = 0, hi = list.length;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start < from) lo = mid + 1; else hi = mid; }
    // Hand position carried over from the note before the window.
    let hand = 5;
    for (let j = lo - 1; j >= 0 && j >= lo - 16; j--) {
      const p = this.pos.get(list[j]);
      if (p) { hand = p.fret || hand; break; }
    }
    for (let i = lo; i < list.length && list[i].start <= to;) {
      // Group notes struck together.
      let k = i + 1;
      while (k < list.length && list[k].start - list[i].start < TOGETHER) k++;
      const group = list.slice(i, k);
      const todo = group.filter((n) => !this.pos.has(n)).sort((a, b) => a.pitch - b.pitch);
      const taken = new Set(group.map((n) => this.pos.get(n)?.string).filter((x) => x != null));
      const shape = bestShape(todo, tuning, taken, hand);
      todo.forEach((n, j) => this.pos.set(n, shape?.[j] ?? null));
      const fretted = (shape || []).filter((p) => p && p.fret > 0).map((p) => p.fret);
      if (fretted.length) hand = Math.round(fretted.reduce((x, y) => x + y, 0) / fretted.length);
      i = k;
    }
  }
}
