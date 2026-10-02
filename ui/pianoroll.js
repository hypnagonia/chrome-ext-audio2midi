// Scrolling piano roll: pitch auto-fits the visible notes. Live, the right edge
// follows the newest transcribed notes; in review, the playhead sits 40% across.

import { instrumentColor, noteName } from './instruments.js';
import { chordName } from '../music/chords.js';
import { Fingering, tuningFor, stringNames } from '../music/tab.js';

const DRUM_LANE = 18;
const CHORD_LANE = 22;
const RULER = 16;
const TAB_CONTROLS = 44; // room under the staffs for the view/zoom controls
const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
export const ZOOM_STEPS = [1, 1.5, 2, 3, 5, 8, 12, 20, 30, 45, 60]; // seconds visible (close-up steps help fast tab passages)
export const TAB_MAX_WINDOW = 8; // tablature: zoom out only this far, so fret numbers keep room

const REDUCED_MOTION = matchMedia('(prefers-reduced-motion: reduce)');
const THEME_VARS = ['--faint', '--pending-pulse', '--paper', '--rule', '--rule-strong', '--ink', '--muted', '--live', '--roll-black', '--pending', '--pending-line', '--chord-font', '--label-font'];

/** Relative luminance (WCAG) of a #rrggbb colour. */
function luminance(hex) {
  const n = parseInt(hex.slice(1), 16);
  const lin = (c) => { c /= 255; return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; };
  return 0.2126 * lin(n >> 16) + 0.7152 * lin((n >> 8) & 255) + 0.0722 * lin(n & 255);
}

const labelColors = new Map();
/** Label text colour with the higher contrast on a note colour. */
function labelColor(color) {
  if (!labelColors.has(color)) {
    let pick = '#fff';
    if (/^#[0-9a-f]{6}$/i.test(color)) {
      const L = luminance(color);
      pick = (1.05) / (L + 0.05) >= (L + 0.05) / 0.05 ? '#fff' : '#0b0d12';
    }
    labelColors.set(color, pick);
  }
  return labelColors.get(color);
}

export class PianoRoll {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.window = 12; // seconds visible
    this.mode = 'roll'; // 'roll' (piano roll) or 'tab' (tablature)
    this.fingering = new Fingering();
    this.onZoom = () => {};
    this.onPan = () => {}; // (seconds) positive = later in time
    this.hits = []; // [{x, y, w, h, note}] from the last draw, for hover
    this.dirty = true; // set by callers when content changes; draw() clears it
    this.easing = false; // true while the pitch range is still animating
    this.textWidths = new Map();
    // Wheel over the roll scrolls the roll: pans time. Ctrl/Cmd + wheel (or pinch) zooms.
    canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      if (e.ctrlKey || e.metaKey) {
        this.zoom(e.deltaY > 0 ? 1 : -1);
        return;
      }
      // Like a DAW piano roll: wheel scrolls pitch (when zoomed in far enough to overflow),
      // Shift + wheel or a sideways swipe pans time.
      const scale = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.h : 1; // lines / pages -> px
      const sideways = e.shiftKey || Math.abs(e.deltaX) > Math.abs(e.deltaY);
      if (!sideways && this.vOverflow) {
        if (this.mode === 'tab') this.tabScroll -= e.deltaY * scale; // staffs scroll in pixels
        else this.pitchScroll -= (e.deltaY * scale) / this.rowH; // the roll in semitones
        this.dirty = true;
        return;
      }
      const px = sideways ? (e.deltaX || e.deltaY) : e.deltaY;
      this.onPan((px * scale / (this.w || 1)) * this.window);
    }, { passive: false });
    canvas.addEventListener('keydown', (e) => {
      if (e.altKey) return; // Alt + arrows edit the selected note (handled by the panel)
      const pan = { ArrowLeft: -0.1, ArrowRight: 0.1 }[e.key];
      if (pan) this.onPan(pan * this.window); // the canvas is left-to-right in every language
      else if ((e.key === 'ArrowUp' || e.key === 'ArrowDown') && this.vOverflow) {
        const dir = e.key === 'ArrowUp' ? 1 : -1;
        if (this.mode === 'tab') this.tabScroll += dir * (this.staffH || 92) / 4;
        else this.pitchScroll += dir * 2;
        this.dirty = true;
      }
      else if (e.key === '+' || e.key === '=') this.zoom(-1);
      else if (e.key === '-' || e.key === '_') this.zoom(1);
      else return;
      e.preventDefault();
      e.stopPropagation();
    });
    this.lo = 48;
    this.hi = 72;
    this.pitchScroll = 0; // semitones the user scrolled the pitch view (when it overflows)
    this.vOverflow = false;
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._watchDpr();
    this._resize();
  }

  /** dir -1 zooms in (fewer seconds), +1 zooms out. */
  zoom(dir) {
    const steps = this.mode === 'tab' ? ZOOM_STEPS.filter((s) => s <= TAB_MAX_WINDOW) : ZOOM_STEPS;
    const i = steps.findIndex((v) => v >= this.window);
    const next = steps[Math.max(0, Math.min(steps.length - 1, (i < 0 ? steps.length - 1 : i) + dir))];
    if (next !== this.window) {
      this.window = next;
      this.pitchScroll = 0;
      this.tabScroll = 0;
      this.onZoom(next);
    }
  }

  // Re-render sharply when the window moves to a screen with another pixel ratio.
  _watchDpr() {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    mq.addEventListener('change', () => { this._resize(); this._watchDpr(); }, { once: true });
  }

  /**
   * Tablature: one staff per audible instrument (4-string bass staff for bass parts,
   * 6-string guitar staff for everything else), fret numbers on the strings and a faint
   * line for how long each note rings. Staffs scroll vertically when they don't fit.
   */
  _drawTab(s, { t0, t1, x, top, bottom, w, v, hits }) {
    const { ctx } = this;
    const insts = (s.instruments || []).filter((i) => i !== 'drums' && s.audible(i) && s.byInst?.get(i)?.length);
    // Keep the last staff clear of the floating view/zoom controls, so its notes stay clickable.
    bottom -= TAB_CONTROLS;
    const areaH = bottom - top;
    this.vOverflow = false;
    if (!insts.length) return;
    const staffH = Math.max(92, areaH / insts.length);
    const totalH = staffH * insts.length;
    const maxScroll = Math.max(0, totalH - areaH);
    this.vOverflow = maxScroll > 0;
    this.staffH = staffH;
    this.tabScroll = Math.max(-maxScroll, Math.min(0, this.tabScroll || 0)); // pixels, <= 0
    let selectedBox = null;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, top, w, areaH);
    ctx.clip();
    insts.forEach((inst, idx) => {
      const staffTop = top + idx * staffH + this.tabScroll;
      if (staffTop > bottom || staffTop + staffH < top) return;
      const list = s.byInst.get(inst);
      const tuning = tuningFor(inst, this._lowest(inst, list));
      const names = stringNames(tuning);
      const color = instrumentColor(inst);
      const gap = Math.min(16, (staffH - 26) / (tuning.length - 1));
      const lineY = (str) => staffTop + 18 + (tuning.length - 1 - str) * gap; // high string on top
      const visibleY = (y0, y1) => y1 >= top && y0 <= bottom; // only what you can see is clickable
      // Staff label + string lines
      ctx.font = `bold 11px ${v['--label-font']}`;
      ctx.textBaseline = 'middle';
      ctx.fillStyle = color;
      ctx.fillRect(4, staffTop + 6, 8, 3);
      ctx.fillStyle = v['--muted'];
      ctx.fillText(s.label?.(inst) ?? inst, 16, staffTop + 8);
      ctx.fillStyle = v['--faint'] || v['--rule-strong'];
      for (let st = 0; st < tuning.length; st++) ctx.fillRect(14, Math.round(lineY(st)), w - 14, 1);
      // Notes in view (binary search: notes ring at most ~12 s)
      this.fingering.assign(inst, list, t0 - 12, t1, tuning);
      let i0 = 0, hi = list.length;
      while (i0 < hi) { const mid = (i0 + hi) >> 1; if (list[mid].start < t0 - 12) i0 = mid + 1; else hi = mid; }
      const size = Math.max(9, Math.min(13, gap));
      const font = `bold ${size}px ${v['--label-font']}`;
      const items = [];
      const lastRight = new Map(); // string -> right edge of the last number placed on it
      for (let i = i0; i < list.length && list[i].start <= t1; i++) {
        const n = list[i];
        const end = n.end ?? s.done;
        if (end < t0) continue;
        const p = this.fingering.get(n, inst);
        if (!p) continue;
        // A note that started off the left edge keeps its number, pinned at the edge.
        const nx = Math.max(x(n.start), 16);
        const ny = lineY(p.string);
        const text = String(p.fret);
        const tw = this._width(text, font);
        const tick = nx - 1 < (lastRight.get(p.string) ?? -Infinity); // too close to the previous number
        if (!tick) lastRight.set(p.string, nx + tw + 2);
        items.push({ n, nx, ny, end, text, tw, tick });
      }
      // Pass 1: ring lines and ticks. Pass 2: numbers on top, so nothing covers a digit.
      for (const it of items) {
        const len = Math.max(2, x(it.end) - it.nx);
        ctx.globalAlpha = 0.45;
        ctx.fillStyle = color;
        ctx.fillRect(it.nx, it.ny - 1, len, 3);
        ctx.globalAlpha = 1;
        // The ring line is clickable too (the number, pushed later, wins where they overlap).
        if (visibleY(it.ny - 5, it.ny + 5)) hits.push({ x: it.nx, y: it.ny - 5, w: len, h: 10, note: it.n });
        if (it.tick) {
          ctx.fillStyle = v['--ink'];
          ctx.fillRect(it.nx, it.ny - 3, 1.5, 6);
          if (visibleY(it.ny - 4, it.ny + 4)) hits.push({ x: it.nx - 2, y: it.ny - 4, w: 5, h: 8, note: it.n });
          if (it.n === s.selected) selectedBox = [it.nx - 3, it.ny - 5, 7.5, 10];
        }
      }
      ctx.font = font;
      for (const it of items) {
        if (it.tick) continue;
        ctx.fillStyle = v['--paper'];
        ctx.fillRect(it.nx - 2, it.ny - size / 2 - 1, it.tw + 4, size + 2); // break the string line
        ctx.fillStyle = v['--ink'];
        ctx.fillText(it.text, it.nx, it.ny + 0.5);
        const box = [it.nx - 2, it.ny - size / 2 - 1, it.tw + 4, size + 2];
        if (visibleY(box[1], box[1] + box[3])) hits.push({ x: box[0], y: box[1], w: box[2], h: box[3], note: it.n });
        if (it.n === s.selected) selectedBox = [box[0] - 1.5, box[1] - 1.5, box[2] + 3, box[3] + 3];
      }
      // String names last, on their own backing, so fret numbers never cover them.
      ctx.font = `10px ${v['--label-font']}`;
      for (let st = 0; st < tuning.length; st++) {
        ctx.fillStyle = v['--paper'];
        ctx.fillRect(0, lineY(st) - 6, 14, 12);
        ctx.fillStyle = v['--muted'];
        ctx.fillText(names[st], 3, lineY(st));
      }
    });
    if (selectedBox) { // after everything, so no later number covers it
      ctx.strokeStyle = v['--ink'];
      ctx.lineWidth = 1.5;
      ctx.strokeRect(...selectedBox);
    }
    ctx.restore();
    if (this.vOverflow) {
      const barH = Math.max(16, (areaH / totalH) * areaH);
      const barTop = top + (-this.tabScroll / totalH) * areaH;
      ctx.fillStyle = v['--rule-strong'];
      ctx.fillRect(w - 4, barTop, 3, barH);
    }
  }

  /**
   * Notes that may be in [t0, t1]: binary search in each part's start-sorted list (notes
   * last at most ~12 s), so a frame costs the same at minute 1 and at hour 1.
   */
  *_inView(s, t0, t1) {
    if (!s.byInst) { yield* s.notes; return; }
    for (const list of s.byInst.values()) {
      let lo = 0, hi = list.length;
      const from = t0 - 12;
      while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start < from) lo = mid + 1; else hi = mid; }
      for (let i = lo; i < list.length && list[i].start <= t1; i++) yield list[i];
    }
  }

  /** Tab: move a note to another string (same pitch). Returns its new position or null. */
  moveString(note, list, dir) {
    const p = this.fingering.moveString(note, note.instrument, list, dir, tuningFor(note.instrument, this._lowest(note.instrument, list)));
    if (p) this.dirty = true;
    return p;
  }

  /** Re-finger every part with the whole-song search (after refine, when all notes are known). */
  refinger(byInst) {
    for (const [inst, list] of byInst) {
      if (inst === 'drums' || !list.length) continue;
      this.fingering.optimize(inst, list, tuningFor(inst, this._lowest(inst, list)));
    }
    this.dirty = true;
  }

  /** Lowest pitch of a part (cached per list length; lists only grow or get replaced). */
  _lowest(inst, list) {
    const c = (this.lowCache ||= new Map()).get(inst);
    if (c && c.list === list && c.n === list.length) return c.low;
    // Ignore the lowest 1% so a stray mis-transcribed note doesn't flip the tuning.
    const pitches = list.map((n) => n.pitch).sort((a, b) => a - b);
    const low = pitches.length ? pitches[pitches.length >= 50 ? Math.floor(pitches.length * 0.01) : 0] : Infinity;
    this.lowCache.set(inst, { list, n: list.length, low });
    return low;
  }

  /** Right edge of the live view: the newest notes plus a thin strip of audio in flight. */
  liveEdge(now, done) {
    return Math.min(now, done + this.window * 0.12);
  }

  /**
   * The audible note under a canvas point (CSS px), or null. A near miss counts: the
   * closest note within a few pixels (more in tab view, where numbers are small).
   */
  noteAt(px, py) {
    for (let i = this.hits.length - 1; i >= 0; i--) {
      const r = this.hits[i];
      if (px >= r.x && px <= r.x + r.w && py >= r.y && py <= r.y + r.h) return r.note;
    }
    const reach = this.mode === 'tab' ? 10 : 5;
    let best = null, bestD = reach;
    for (const r of this.hits) {
      const dx = Math.max(r.x - px, 0, px - (r.x + r.w));
      const dy = Math.max(r.y - py, 0, py - (r.y + r.h));
      const d = Math.hypot(dx, dy);
      if (d < bestD) { bestD = d; best = r.note; }
    }
    return best;
  }

  _resize() {
    const r = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.w = r.width;
    this.h = r.height;
    this.canvas.width = Math.round(r.width * dpr);
    this.canvas.height = Math.round(r.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.dirty = true;
  }

  _theme() {
    if (!this.vars) {
      const css = getComputedStyle(this.canvas);
      this.vars = Object.fromEntries(THEME_VARS.map((n) => [n, css.getPropertyValue(n).trim()]));
      this.vars['--label-font'] ||= 'monospace';
    }
    return this.vars;
  }

  _width(text, font) {
    const k = `${font}|${text}`;
    if (!this.textWidths.has(k)) {
      this.ctx.font = font;
      this.textWidths.set(k, this.ctx.measureText(text).width);
    }
    return this.textWidths.get(k);
  }

  /**
   * @param s {notes, now, done, audible(inst)->bool, chords:[{start,end,chord}], flats, playhead, center, end}
   *   now: seconds of audio captured; done: seconds transcribed so far;
   *   end: live only, a fixed right edge while the user has scrolled back (null follows live).
   */
  draw(s) {
    const { ctx, w, h } = this;
    if (!w || !h) return;
    const v = this._theme();
    const t1 = s.center != null ? Math.max(s.center + this.window * 0.6, this.window)
      : s.end != null ? Math.max(s.end, this.window)
      : Math.max(this.liveEdge(s.now, s.done), this.window);
    const t0 = t1 - this.window;
    const x = (t) => ((t - t0) / this.window) * w;
    const top = CHORD_LANE;
    const bottom = h - DRUM_LANE - RULER;

    // Fit pitch range to the audible notes in view (eased so it does not jump);
    // soloing an instrument zooms the roll onto its range.
    let lo = Infinity, hi = -Infinity, alo = Infinity, ahi = -Infinity;
    const visible = [];
    for (const n of this._inView(s, t0, t1)) {
      if (n.start > t1 || (n.end ?? s.done) < t0) continue;
      visible.push(n);
      if (n.instrument === 'drums') continue;
      lo = Math.min(lo, n.pitch); hi = Math.max(hi, n.pitch);
      if (s.audible(n.instrument)) { alo = Math.min(alo, n.pitch); ahi = Math.max(ahi, n.pitch); }
    }
    if (alo <= ahi) { lo = alo; hi = ahi; }
    this.easing = false;
    const areaH = bottom - top;
    // Zoomed in: rows keep a minimum height (so note names fit) and the pitch view scrolls.
    const minRow = this.window <= 5 ? 14 : this.window <= 12 ? 9 : 0;
    this.vOverflow = false;
    let extent = null;
    if (lo <= hi) {
      const mid = (lo + hi) / 2;
      lo = Math.min(lo - 1, mid - 6);
      hi = Math.max(hi + 1, mid + 6);
      extent = [lo, hi];
      const fit = hi - lo + 1;
      const rows = minRow ? Math.min(fit, Math.floor(areaH / minRow)) : fit;
      if (rows < fit) {
        this.vOverflow = true;
        const maxScroll = (fit - rows) / 2;
        this.pitchScroll = Math.max(-maxScroll, Math.min(maxScroll, this.pitchScroll));
        const center = mid + this.pitchScroll;
        lo = center - (rows - 1) / 2;
        hi = center + (rows - 1) / 2;
      } else {
        this.pitchScroll = 0;
      }
      this.lo += (lo - this.lo) * 0.12;
      this.hi += (hi - this.hi) * 0.12;
      this.easing = Math.abs(lo - this.lo) > 0.05 || Math.abs(hi - this.hi) > 0.05;
    }
    const pLo = Math.floor(this.lo), pHi = Math.ceil(this.hi);
    const rowH = areaH / (pHi - pLo + 1);
    this.rowH = rowH;
    const y = (p) => bottom - (p - pLo + 1) * rowH;

    ctx.clearRect(0, 0, w, h);
    // Black-key rows and C lines
    ctx.font = `11px ${v['--label-font']}`;
    ctx.textBaseline = 'bottom';
    for (let p = pLo; p <= pHi && this.mode !== 'tab'; p++) {
      if (BLACK_KEYS.has(((p % 12) + 12) % 12)) {
        ctx.fillStyle = v['--roll-black'];
        ctx.fillRect(0, y(p), w, rowH);
      }
      if (p % 12 === 0) {
        ctx.fillStyle = v['--rule-strong'];
        ctx.fillRect(0, y(p) + rowH - 0.5, w, 1);
        ctx.fillStyle = v['--muted'];
        ctx.fillText(noteName(p), 4, y(p) + rowH - 1);
      }
    }
    // Second lines + time ruler (labels at least ~56 px apart)
    ctx.fillStyle = v['--rule'];
    const step = this.window > 30 ? 5 : 1;
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) ctx.fillRect(Math.round(x(t)), top, 1, bottom - top);
    const every = [1, 2, 5, 10, 15, 30, 60].find((k) => (k / this.window) * w >= 56) ?? 60;
    ctx.font = `11px ${v['--label-font']}`;
    ctx.textBaseline = 'middle';
    ctx.fillStyle = v['--muted'];
    for (let t = Math.ceil(Math.max(0, t0) / every) * every; t <= t1; t += every) {
      ctx.fillRect(Math.round(x(t)), h - RULER, 1, 4);
      ctx.fillText(fmtTime(t), Math.round(x(t)) + 4, h - RULER / 2 + 1);
    }
    ctx.fillStyle = v['--rule-strong'];
    ctx.fillRect(0, h - RULER, w, 1);

    // Not yet transcribed. While listening it breathes slowly, so you can see byEar is working.
    if (s.done < t1) {
      const xd = Math.max(0, x(s.done));
      ctx.fillStyle = v['--pending'];
      ctx.fillRect(xd, 0, w - xd, h);
      if (s.listening) {
        // Only the zone the next chunk will fill pulses, brighter and darker.
        const xe = Math.min(w, x(Math.min(s.now, s.done + (s.chunk || 5))));
        const pulse = REDUCED_MOTION.matches ? 0.5 : 0.5 + 0.5 * Math.sin(performance.now() / 380);
        ctx.fillStyle = `rgba(255, 255, 255, ${0.02 + 0.10 * pulse})`;
        ctx.fillRect(xd, 0, Math.max(0, xe - xd), h);
      }
      ctx.save();
      ctx.beginPath();
      ctx.rect(xd, 0, w - xd, h);
      ctx.clip();
      ctx.strokeStyle = v['--pending-line'];
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let i = -h; i < w; i += 8) {
        ctx.moveTo(xd + i, h);
        ctx.lineTo(xd + i + h, 0);
      }
      ctx.stroke();
      ctx.restore();
    }

    // Drum hits in their own lane
    const hits = [];
    for (const n of visible) {
      if (n.instrument !== 'drums') continue;
      const on = s.audible(n.instrument);
      ctx.globalAlpha = on ? 1 : 0.2;
      ctx.fillStyle = instrumentColor(n.instrument);
      const dx = x(n.start) - 1, dy = bottom + 3 + ((n.pitch % 6) / 6) * (DRUM_LANE - 6);
      ctx.fillRect(dx, dy, 3, 3);
      if (on) hits.push({ x: dx - 2, y: dy - 2, w: 7, h: 7, note: n });
    }
    ctx.globalAlpha = 1;

    if (this.mode === 'tab') {
      this._drawTab(s, { t0, t1, x, top, bottom, w, v, hits }); // has its own save/clip/restore
    } else {
    // Pitched notes, clipped to the note area
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, top, w, bottom - top);
    ctx.clip();
    const size = Math.max(7, Math.min(11, Math.floor(rowH - 1)));
    const font = `bold ${size}px ${v['--label-font']}`;
    ctx.textBaseline = 'middle';
    // Muted / not-soloed parts are drawn faintly underneath the audible ones.
    const pitched = visible.filter((n) => n.instrument !== 'drums');
    pitched.sort((a, b) => s.audible(a.instrument) - s.audible(b.instrument));
    for (const n of pitched) {
      const on = s.audible(n.instrument);
      const color = instrumentColor(n.instrument);
      const nx = x(n.start);
      const nw = Math.max(2, x(n.end ?? s.done) - nx - 1);
      ctx.globalAlpha = on ? 1 : 0.18;
      ctx.fillStyle = color;
      ctx.fillRect(nx, y(n.pitch) + 0.5, nw, Math.max(1.5, rowH - 1));
      ctx.globalAlpha = 1;
      if (!on) continue;
      if (y(n.pitch) + rowH >= top && y(n.pitch) <= bottom) hits.push({ x: nx, y: y(n.pitch), w: nw, h: Math.max(4, rowH), note: n });
      if (n === s.selected) {
        ctx.strokeStyle = v['--ink'];
        ctx.lineWidth = 1.5;
        ctx.strokeRect(nx - 1, y(n.pitch) - 0.5, nw + 2, Math.max(2.5, rowH));
      }
      // Label: name + octave when it fits in the visible part, else the letter alone.
      const shown = Math.min(nx + nw, w) - Math.max(nx, 0);
      if (rowH < 7 || shown < 8) continue;
      const full = noteName(n.pitch, s.flats);
      const letter = full.replace(/-?\d+$/, '');
      const text = this._width(full, font) + 4 <= shown ? full : this._width(letter, font) + 4 <= shown ? letter : '';
      if (!text) continue;
      ctx.font = font;
      ctx.fillStyle = labelColor(color);
      ctx.fillText(text, Math.max(nx, 0) + 2, y(n.pitch) + rowH / 2 + 0.5);
    }
    ctx.restore();
    }

    // Chord lane
    ctx.fillStyle = v['--paper'];
    ctx.fillRect(0, 0, w, CHORD_LANE);
    ctx.fillStyle = v['--rule-strong'];
    ctx.fillRect(0, CHORD_LANE - 1, w, 1);
    ctx.fillRect(0, bottom, w, 1);
    ctx.textBaseline = 'middle';
    const chordFont = `bold 14px ${v['--chord-font']}`;
    for (const seg of s.chords) {
      if (seg.end < t0 || seg.start > t1) continue;
      const sx = Math.max(x(seg.start), 0);
      ctx.fillStyle = v['--rule-strong'];
      if (x(seg.start) >= 0) ctx.fillRect(x(seg.start), 4, 1, CHORD_LANE - 8);
      // Label only when the whole name fits: a clipped "Gma" reads as a different chord.
      const name = chordName(seg.chord, s.flats);
      if (this._width(name, chordFont) + 8 > Math.min(x(seg.end), w) - sx) continue;
      ctx.font = chordFont;
      ctx.fillStyle = v['--ink'];
      ctx.fillText(name, sx + 4, CHORD_LANE / 2 + 1);
    }

    this.t0 = t0; // for pointer scrubbing
    this.hits = hits;
    // Vertical scrollbar: where the visible pitch rows sit within all the notes in view.
    if (this.mode !== 'tab' && this.vOverflow && extent) {
      const span = extent[1] - extent[0] + 1;
      const barTop = top + ((extent[1] - pHi) / span) * areaH;
      const barH = Math.max(16, ((pHi - pLo + 1) / span) * areaH);
      ctx.fillStyle = v['--rule-strong'];
      ctx.fillRect(w - 4, Math.max(top, Math.min(bottom - barH, barTop)), 3, barH);
    }
    if (s.playhead != null && s.playhead > t0) {
      ctx.fillStyle = v['--live'];
      ctx.fillRect(Math.round(x(s.playhead)), 0, 1.5, h);
    }
  }
}
