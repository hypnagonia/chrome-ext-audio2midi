// Review playback: the recorded audio ("original") or the transcription ("midi")
// rendered by a small Web Audio synth. Both share one session timeline.

const LOOKAHEAD = 1.5; // seconds scheduled ahead
const TICK_MS = 200;

// Synth voices per instrument family: [oscillator, attack, decay, sustain, release, lowpass Hz]
const VOICES = {
  keys: ['triangle', 0.005, 0.9, 0.25, 0.25, 5000],
  epiano: ['sine', 0.005, 1.2, 0.3, 0.3, 4000],
  organ: ['square', 0.01, 0.05, 0.8, 0.08, 2400],
  mallet: ['sine', 0.002, 0.6, 0.0, 0.3, 8000],
  guitar: ['sawtooth', 0.004, 0.5, 0.35, 0.15, 2600],
  dist: ['sawtooth', 0.004, 0.2, 0.7, 0.1, 1800],
  bass: ['triangle', 0.005, 0.4, 0.6, 0.1, 900],
  strings: ['sawtooth', 0.08, 0.3, 0.8, 0.3, 3000],
  brass: ['sawtooth', 0.03, 0.2, 0.75, 0.12, 2200],
  reed: ['square', 0.02, 0.2, 0.7, 0.1, 2600],
  flute: ['sine', 0.04, 0.2, 0.8, 0.12, 6000],
  voice: ['triangle', 0.05, 0.2, 0.85, 0.15, 3200],
  synth: ['sawtooth', 0.01, 0.3, 0.6, 0.2, 3500],
};
const FAMILY = {
  acoustic_piano: 'keys', electric_piano: 'epiano', organ: 'organ', chromatic_percussion: 'mallet',
  acoustic_guitar: 'guitar', clean_electric_guitar: 'guitar', distorted_electric_guitar: 'dist', orchestral_harp: 'mallet',
  acoustic_bass: 'bass', electric_bass: 'bass', contrabass: 'bass', timpani: 'bass',
  violin: 'strings', viola: 'strings', cello: 'strings', string_ensemble: 'strings', synth_strings: 'strings', orchestra_hit: 'brass',
  trumpet: 'brass', trombone: 'brass', tuba: 'brass', french_horn: 'brass', brass_section: 'brass',
  soprano_and_alto_sax: 'reed', tenor_sax: 'reed', baritone_sax: 'reed', oboe: 'reed', english_horn: 'reed',
  bassoon: 'reed', clarinet: 'reed', flutes: 'flute', voice: 'voice', synth_lead: 'synth', synth_pad: 'strings',
};

export class Player {
  constructor() {
    this.ctx = null;
    this.mode = 'midi'; // or 'original'
    this.playing = false;
    this.pos = 0; // timeline seconds while paused
    this.duration = 0;
    this.segments = []; // [{seek, rate, data: Int16Array}]
    this.notes = [];
    this.audible = () => true; // (instrument) -> plays in MIDI mode (mute/solo)
    this.onEnd = () => {};
    this.gen = 0; // bumps on every play/pause so a stale play() can bail out
    this.nodes = new Set(); // started sources/oscillators, stopped on pause
    this.buffers = new WeakMap(); // segment -> AudioBuffer
    this.speed = 1; // 1, 0.75, 0.5
    this.el = null; // <audio> for slowed-down original audio (keeps pitch)
    this.elActive = false;
  }

  _ensureCtx() {
    if (this.ctx) return;
    this.ctx = new AudioContext({ latencyHint: 'playback' });
    const comp = this.ctx.createDynamicsCompressor();
    this.master = this.ctx.createGain();
    this.master.connect(comp).connect(this.ctx.destination);
    this.noise = this._noiseBuffer();
  }

  /** Sound one note now (click on a note), with its instrument's voice. */
  async preview(note) {
    this._ensureCtx();
    await this.ctx.resume();
    const bus = this.ctx.createGain();
    bus.gain.value = 0.25;
    bus.connect(this.master);
    const keep = this.bus;
    this.bus = bus;
    const length = Math.min(1.2, Math.max(0.3, (note.end ?? note.start + 0.5) - note.start));
    this._note(note, this.ctx.currentTime + 0.01, length);
    this.bus = keep;
    setTimeout(() => bus.disconnect(), 3000);
  }

  async setSpeed(speed) {
    const was = this.playing;
    this.pause();
    this.speed = speed;
    if (was) await this.play();
  }

  /** The recording as one WAV (silent gaps filled), for pitch-preserving slow playback. */
  /** Pitch-preserving slow playback needs one sample rate and a sane size; else null. */
  _elementOk() {
    const rates = new Set(this.segments.map((s) => s.rate));
    return rates.size === 1 && this.duration * [...rates][0] * 2 < 600e6;
  }

  _element() {
    const last = this.segments[this.segments.length - 1];
    // Keyed on the recording itself (a Clear makes a new array), not just its length.
    if (this.el && this.elFor === this.segments && this.elLast === last && this.elDuration === this.duration) return this.el;
    if (this.elUrl) URL.revokeObjectURL(this.elUrl);
    const rate = this.segments[0]?.rate || 48000;
    const n = Math.ceil(this.duration * rate);
    const buf = new ArrayBuffer(44 + n * 2);
    const v = new DataView(buf);
    const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
    str(0, 'RIFF'); v.setUint32(4, 36 + n * 2, true); str(8, 'WAVE'); str(12, 'fmt ');
    v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
    v.setUint32(24, rate, true); v.setUint32(28, rate * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
    str(36, 'data'); v.setUint32(40, n * 2, true);
    const pcm = new Int16Array(buf, 44, n);
    for (const seg of this.segments) {
      const at = Math.round(seg.seek * rate);
      if (at < n) pcm.set(seg.data.subarray(0, Math.min(seg.data.length, n - at)), at);
    }
    this.elUrl = URL.createObjectURL(new Blob([buf], { type: 'audio/wav' }));
    if (!this.el) {
      this.el = new Audio();
      this.el.preservesPitch = true;
      this.elSource = this.ctx.createMediaElementSource(this.el);
      this.elGain = this.ctx.createGain();
      this.elSource.connect(this.elGain).connect(this.master);
    }
    this.el.src = this.elUrl;
    this.elFor = this.segments;
    this.elLast = last;
    this.elDuration = this.duration;
    return this.el;
  }

  /** Timeline position now. */
  get time() {
    if (!this.playing) return this.pos;
    if (this.elActive) return Math.min(this.duration, this.el.currentTime);
    return Math.min(this.duration, Math.max(this.startPos, this.startPos + (this.ctx.currentTime - this.startAt) * this.speed));
  }

  load({ segments, notes, duration, audible }) {
    this.segments = segments;
    this.notes = [...notes].sort((a, b) => a.start - b.start);
    this.duration = duration;
    this.audible = audible ?? (() => true);
  }

  async play() {
    if (this.playing) return;
    const gen = ++this.gen;
    if (this.pos >= this.duration - 0.05) this.pos = 0;
    this._ensureCtx();
    await this.ctx.resume();
    if (gen !== this.gen || this.playing) return; // paused or replayed while resuming
    if (this.mode === 'original' && this.speed !== 1 && this.segments.length && this._elementOk()) {
      // Slowed-down original: an <audio> element keeps the pitch.
      const el = this._element();
      el.currentTime = this.pos;
      el.playbackRate = this.speed;
      el.preservesPitch = true;
      this.elGain.gain.value = 1;
      try {
        await el.play();
      } catch {
        return; // superseded by a newer play()/pause(), or the element refused: stay paused
      }
      if (gen !== this.gen) {
        if (!this.playing) el.pause(); // only if nothing newer is playing it
        return;
      }
      this.elActive = true;
      this.playing = true;
      clearInterval(this.timer);
      this.timer = setInterval(() => this._tick(), TICK_MS);
      return;
    }
    this.playing = true;
    this.startAt = this.ctx.currentTime + 0.05;
    this.startPos = this.pos;
    this.scheduledUntil = this.pos;
    this.bus = this.ctx.createGain();
    this.bus.gain.value = this.mode === 'midi' ? 0.22 : 1;
    this.bus.connect(this.master);
    clearInterval(this.timer);
    this._tick();
    this.timer = setInterval(() => this._tick(), TICK_MS);
  }

  pause() {
    this.gen++;
    clearInterval(this.timer);
    if (!this.playing) return;
    this.pos = this.time;
    this.playing = false;
    if (this.elActive) {
      this.el.pause();
      this.elActive = false;
      return;
    }
    const { bus, ctx } = this;
    const at = ctx.currentTime;
    bus.gain.setTargetAtTime(0, at, 0.015);
    for (const node of this.nodes) {
      try { node.stop(at + 0.08); } catch { /* not started yet */ }
    }
    this.nodes.clear();
    setTimeout(() => bus.disconnect(), 150);
  }

  seek(t) {
    const was = this.playing;
    this.pause();
    this.pos = Math.max(0, Math.min(this.duration, t));
    if (was) this.play();
  }

  setMode(mode) {
    if (mode === this.mode) return;
    const was = this.playing;
    this.pause();
    this.mode = mode;
    if (was) this.play();
  }

  stop() {
    this.pause();
    this.pos = 0;
  }

  _track(node) {
    this.nodes.add(node);
    node.onended = () => this.nodes.delete(node);
    return node;
  }

  _buffer(seg) {
    let buf = this.buffers.get(seg);
    if (!buf) {
      buf = this.ctx.createBuffer(1, seg.data.length, seg.rate);
      const ch = buf.getChannelData(0);
      for (let i = 0; i < seg.data.length; i++) ch[i] = seg.data[i] / 32768;
      this.buffers.set(seg, buf);
    }
    return buf;
  }

  _tick() {
    if (!this.playing) return;
    const now = this.time;
    if (now >= this.duration - 0.01 || (this.elActive && this.el.ended)) {
      this.pause();
      this.pos = this.duration;
      this.onEnd();
      return;
    }
    if (this.elActive) return; // the <audio> element plays itself
    const from = this.scheduledUntil;
    const to = Math.min(this.duration, now + LOOKAHEAD);
    if (to <= from) return;
    const first = from === this.startPos;
    this.scheduledUntil = to;
    const when = (t) => this.startAt + (t - this.startPos) / this.speed;
    if (this.mode === 'original') {
      for (const seg of this.segments) {
        const len = seg.data.length / seg.rate;
        // Each segment is scheduled once: when its start (or the play position) enters the window.
        const start = Math.max(seg.seek, this.startPos);
        if (start < from || start >= to || seg.seek + len <= this.startPos) continue;
        const src = this._track(this.ctx.createBufferSource());
        src.buffer = this._buffer(seg);
        src.connect(this.bus);
        // If the timer ran late, start now but skip ahead so the audio stays on the timeline.
        // Fallback for slow playback (mixed sample rates / huge sessions): speed changes pitch here.
        src.playbackRate.value = this.speed;
        const late = Math.max(0, this.ctx.currentTime - when(start));
        src.start(when(start) + late, start - seg.seek + late * this.speed);
      }
    } else {
      for (const n of this.notes) {
        if (n.start >= to) break;
        if (!this.audible(n.instrument)) continue;
        const end = n.end ?? n.start + 0.3;
        if (n.start >= from) this._note(n, when(n.start), (end - n.start) / this.speed);
        // Starting inside a held note: sound the rest of it.
        else if (first && n.start < this.startPos && end > this.startPos + 0.05 && n.instrument !== 'drums') {
          this._note(n, this.startAt, (end - this.startPos) / this.speed);
        }
      }
    }
  }

  _note(n, t, length) {
    const ctx = this.ctx;
    if (n.instrument === 'drums') return this._drum(n.pitch, t);
    const [type, a, d, s, r, cutoff] = VOICES[FAMILY[n.instrument] ?? 'synth'];
    const dur = Math.max(0.06, length, a + 0.02); // release never before the attack has finished
    const osc = this._track(ctx.createOscillator());
    osc.type = type;
    osc.frequency.value = 440 * 2 ** ((n.pitch - 69) / 12);
    const lp = ctx.createBiquadFilter();
    lp.type = 'lowpass';
    lp.frequency.value = cutoff;
    const g = ctx.createGain();
    const peak = n.pitch < 48 ? 0.5 : 0.32;
    g.gain.setValueAtTime(0, t);
    g.gain.linearRampToValueAtTime(peak, t + a);
    g.gain.setTargetAtTime(peak * s, t + a, d / 3);
    g.gain.setTargetAtTime(0, t + dur, r / 3);
    osc.connect(lp).connect(g).connect(this.bus);
    osc.start(t);
    osc.stop(t + dur + r * 2);
  }

  _drum(pitch, t) {
    const ctx = this.ctx;
    const g = ctx.createGain();
    g.connect(this.bus);
    if (pitch === 35 || pitch === 36) { // kick
      const o = ctx.createOscillator();
      o.frequency.setValueAtTime(140, t);
      o.frequency.exponentialRampToValueAtTime(45, t + 0.12);
      g.gain.setValueAtTime(0.9, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + 0.25);
      o.connect(g);
      this._track(o).start(t);
      o.stop(t + 0.3);
      return;
    }
    const hat = [42, 44, 46, 51, 53, 59].includes(pitch);
    const cym = [49, 52, 55, 57].includes(pitch);
    const src = this._track(ctx.createBufferSource());
    src.buffer = this.noise;
    const f = ctx.createBiquadFilter();
    f.type = hat || cym ? 'highpass' : 'bandpass';
    f.frequency.value = hat || cym ? 7000 : 1800;
    const len = cym ? 0.9 : pitch === 46 ? 0.3 : hat ? 0.05 : 0.14;
    g.gain.setValueAtTime(hat ? 0.25 : 0.5, t);
    g.gain.exponentialRampToValueAtTime(0.001, t + len);
    src.connect(f).connect(g);
    src.start(t);
    src.stop(t + len + 0.05);
  }

  _noiseBuffer() {
    const b = this.ctx.createBuffer(1, this.ctx.sampleRate, this.ctx.sampleRate);
    const d = b.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
    return b;
  }
}
