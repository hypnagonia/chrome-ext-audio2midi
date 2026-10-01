// Scrolling piano roll: pitch auto-fits the visible notes. Live, the right edge
// follows the newest transcribed notes; in review, the playhead sits 40% across.

import { instrumentColor, noteName } from './instruments.js';
import { chordName } from '../music/chords.js';

const DRUM_LANE = 18;
const CHORD_LANE = 22;
const BLACK_KEYS = new Set([1, 3, 6, 8, 10]);
export const ZOOM_STEPS = [3, 5, 8, 12, 20, 30, 45, 60]; // seconds visible

const THEME_VARS = ['--paper', '--rule', '--rule-strong', '--ink', '--muted', '--live', '--roll-black', '--pending', '--pending-line', '--chord-font', '--label-font'];

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
    this.onZoom = () => {};
    this.dirty = true; // set by callers when content changes; draw() clears it
    this.easing = false; // true while the pitch range is still animating
    this.textWidths = new Map();
    // Ctrl/Cmd + wheel, or trackpad pinch, zooms time.
    canvas.addEventListener('wheel', (e) => {
      if (!e.ctrlKey && !e.metaKey) return;
      e.preventDefault();
      this.zoom(e.deltaY > 0 ? 1 : -1);
    }, { passive: false });
    this.lo = 48;
    this.hi = 72;
    new ResizeObserver(() => this._resize()).observe(canvas);
    this._watchDpr();
    this._resize();
  }

  /** dir -1 zooms in (fewer seconds), +1 zooms out. */
  zoom(dir) {
    const i = ZOOM_STEPS.findIndex((v) => v >= this.window);
    const next = ZOOM_STEPS[Math.max(0, Math.min(ZOOM_STEPS.length - 1, (i < 0 ? ZOOM_STEPS.length - 1 : i) + dir))];
    if (next !== this.window) {
      this.window = next;
      this.onZoom(next);
    }
  }

  // Re-render sharply when the window moves to a screen with another pixel ratio.
  _watchDpr() {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    mq.addEventListener('change', () => { this._resize(); this._watchDpr(); }, { once: true });
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
   * @param s {notes, now, done, hidden:Set, chords:[{start,end,chord}], flats, playhead, center}
   *   now: seconds of audio captured; done: seconds transcribed so far.
   */
  draw(s) {
    const { ctx, w, h } = this;
    if (!w || !h) return;
    const v = this._theme();
    const t1 = s.center != null
      ? Math.max(s.center + this.window * 0.6, this.window)
      : Math.max(Math.min(s.now, s.done + this.window * 0.12), this.window);
    const t0 = t1 - this.window;
    const x = (t) => ((t - t0) / this.window) * w;
    const top = CHORD_LANE;
    const bottom = h - DRUM_LANE;

    // Fit pitch range to what is visible (eased so it does not jump).
    let lo = Infinity, hi = -Infinity;
    const visible = [];
    for (const n of s.notes) {
      if (n.start > t1 || (n.end ?? s.done) < t0 || s.hidden.has(n.instrument)) continue;
      visible.push(n);
      if (n.instrument !== 'drums') { lo = Math.min(lo, n.pitch); hi = Math.max(hi, n.pitch); }
    }
    this.easing = false;
    if (lo <= hi) {
      const mid = (lo + hi) / 2;
      lo = Math.min(lo - 1, mid - 6);
      hi = Math.max(hi + 1, mid + 6);
      this.lo += (lo - this.lo) * 0.08;
      this.hi += (hi - this.hi) * 0.08;
      this.easing = Math.abs(lo - this.lo) > 0.05 || Math.abs(hi - this.hi) > 0.05;
    }
    const pLo = Math.floor(this.lo), pHi = Math.ceil(this.hi);
    const rowH = (bottom - top) / (pHi - pLo + 1);
    const y = (p) => bottom - (p - pLo + 1) * rowH;

    ctx.clearRect(0, 0, w, h);
    // Black-key rows and C lines
    ctx.font = `11px ${v['--label-font']}`;
    ctx.textBaseline = 'bottom';
    for (let p = pLo; p <= pHi; p++) {
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
    // Second lines
    ctx.fillStyle = v['--rule'];
    const step = this.window > 30 ? 5 : 1;
    for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) ctx.fillRect(Math.round(x(t)), top, 1, bottom - top);

    // Not yet transcribed
    if (s.done < t1) {
      const xd = Math.max(0, x(s.done));
      ctx.fillStyle = v['--pending'];
      ctx.fillRect(xd, 0, w - xd, h);
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
    for (const n of visible) {
      if (n.instrument !== 'drums') continue;
      ctx.fillStyle = instrumentColor(n.instrument);
      ctx.fillRect(x(n.start) - 1, bottom + 3 + ((n.pitch % 6) / 6) * (DRUM_LANE - 6), 3, 3);
    }

    // Pitched notes, clipped to the note area
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, top, w, bottom - top);
    ctx.clip();
    const size = Math.max(7, Math.min(11, Math.floor(rowH - 1)));
    const font = `bold ${size}px ${v['--label-font']}`;
    ctx.textBaseline = 'middle';
    for (const n of visible) {
      if (n.instrument === 'drums') continue;
      const color = instrumentColor(n.instrument);
      const nx = x(n.start);
      const nw = Math.max(2, x(n.end ?? s.done) - nx - 1);
      ctx.fillStyle = color;
      ctx.fillRect(nx, y(n.pitch) + 0.5, nw, Math.max(1.5, rowH - 1));
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

    // Chord lane
    ctx.fillStyle = v['--paper'];
    ctx.fillRect(0, 0, w, CHORD_LANE);
    ctx.fillStyle = v['--rule-strong'];
    ctx.fillRect(0, CHORD_LANE - 1, w, 1);
    ctx.fillRect(0, bottom, w, 1);
    ctx.textBaseline = 'middle';
    ctx.font = `bold 14px ${v['--chord-font']}`;
    for (const seg of s.chords) {
      if (seg.end < t0 || seg.start > t1) continue;
      const sx = Math.max(x(seg.start), 0);
      ctx.fillStyle = v['--rule-strong'];
      if (x(seg.start) >= 0) ctx.fillRect(x(seg.start), 4, 1, CHORD_LANE - 8);
      if (x(seg.end) - sx < 24) continue; // too narrow to label legibly
      ctx.fillStyle = v['--ink'];
      ctx.save();
      ctx.beginPath();
      ctx.rect(sx, 0, x(seg.end) - sx, CHORD_LANE);
      ctx.clip();
      ctx.fillText(chordName(seg.chord, s.flats), sx + 4, CHORD_LANE / 2 + 1);
      ctx.restore();
    }

    this.t0 = t0; // for pointer scrubbing
    if (s.playhead != null && s.playhead > t0) {
      ctx.fillStyle = v['--live'];
      ctx.fillRect(Math.round(x(s.playhead)), 0, 1.5, h);
    }
  }
}
