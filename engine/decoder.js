// Token stream -> note events. Port of OpenNoteTracker / decode_model_tokens
// from muscriptor/events.py, made incremental so chunks can arrive live.

import { VOCAB, FRAME_RATE, MIN_NOTE_SEC, instrumentForProgram } from './vocab.js';

const key = (program, pitch) => `${program}:${pitch}`;

export class NoteDecoder {
  constructor() {
    this.open = new Map(); // key -> {program, pitch, time}
    this.startEvents = new Map(); // key -> start event
    this.nextIndex = 0;
    this.seekTime = 0;
    this.nextSeekTime = null;
    this.startTick = 0;
    this.tick = 0;
    this.program = null;
    this.velocity = null;
    this.inPrologue = true;
    this.skipRest = false;
    this.tieSet = new Set();
    this.chunkStarted = false;
    this.keepLeadIn = true; // false: drop unknown held notes, exactly like the reference
  }

  /** Begin a chunk. Returns events closed by a malformed previous chunk. */
  boundary(seekTime, nextSeekTime) {
    const out = [];
    if (this.chunkStarted && this.inPrologue) this._endAll(this.seekTime, out);
    this.seekTime = seekTime;
    this.nextSeekTime = nextSeekTime;
    this.startTick = Math.round(seekTime * FRAME_RATE);
    this.tick = this.startTick;
    this.program = null;
    this.velocity = null;
    this.inPrologue = true;
    this.skipRest = false;
    this.tieSet = new Set();
    this.chunkStarted = true;
    return out;
  }

  /** Sorted [program, pitch] pairs currently open (the next chunk's tie prologue). */
  openKeys() {
    return [...this.open.values()].map((n) => [n.program, n.pitch]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  token(id) {
    const out = [];
    if (!(id >= 0 && id < VOCAB.length)) return out;
    const { type, value } = VOCAB[id];
    if (this.inPrologue) {
      if (type === 'tie' || type === 'shift') {
        // A shift before the tie: the model skipped the end of the prologue. The reference
        // decoder drops the rest of the chunk (a 5 s hole in the music); we read on instead.
        this.inPrologue = false;
        this.velocity = null;
        for (const [k, n] of [...this.open]) {
          if (!this.tieSet.has(k)) this._end(k, n, this.seekTime, out);
        }
        // Held notes we didn't know of (the first chunk: already sounding when the audio
        // starts) begin at the chunk start instead of being lost.
        for (const k of this.keepLeadIn ? this.tieSet : []) {
          if (this.open.has(k)) continue;
          const [program, pitch] = k.split(':').map(Number);
          if (program === 128) continue; // drums don't sustain
          this.open.set(k, { program, pitch, time: this.seekTime });
          const start = this._mint(pitch, this.seekTime, instrumentForProgram(program));
          this.startEvents.set(k, start);
          out.push(start);
        }
        if (type === 'tie') return out;
        this.program = null; // the shift itself is read below, as in the body
      } else {
        if (type === 'program') this.program = value;
        else if (type === 'pitch' && this.program !== null) this.tieSet.add(key(this.program, value));
        return out;
      }
    }
    if (this.skipRest) return out;

    if (type === 'shift') {
      if (value > 0) this.tick = this.startTick + value;
    } else if (type === 'program') {
      this.program = value;
    } else if (type === 'velocity') {
      this.velocity = value;
    } else if (type === 'drum') {
      const time = this.tick / FRAME_RATE;
      if (this.nextSeekTime === null || time < this.nextSeekTime) {
        const start = this._mint(value, time, 'drums');
        out.push(start, { type: 'end', index: start.index, time: time + MIN_NOTE_SEC });
      }
    } else if (type === 'pitch') {
      if (this.program === null || this.velocity === null) return out;
      const time = this.tick / FRAME_RATE;
      if (this.nextSeekTime !== null && time >= this.nextSeekTime) return out;
      const k = key(this.program, value);
      if (this.open.has(k)) this._end(k, this.open.get(k), time, out);
      if (this.velocity > 0) {
        this.open.set(k, { program: this.program, pitch: value, time });
        const start = this._mint(value, time, instrumentForProgram(this.program));
        this.startEvents.set(k, start);
        out.push(start);
      }
    }
    return out;
  }

  /** End of stream: close everything still open. */
  finish() {
    const out = [];
    if (this.chunkStarted && this.inPrologue) {
      this._endAll(this.seekTime, out);
    } else {
      for (const [k, n] of [...this.open]) this._end(k, n, n.time + MIN_NOTE_SEC, out);
    }
    return out;
  }

  _mint(pitch, time, instrument) {
    return { type: 'start', index: this.nextIndex++, pitch, time, instrument };
  }

  _end(k, _n, time, out) {
    this.open.delete(k);
    const start = this.startEvents.get(k);
    this.startEvents.delete(k);
    out.push({ type: 'end', index: start.index, time });
  }

  _endAll(time, out) {
    for (const [k, n] of [...this.open]) this._end(k, n, time, out);
  }
}
