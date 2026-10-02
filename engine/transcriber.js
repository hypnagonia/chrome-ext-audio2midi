// Chunk-by-chunk transcription with prelude forcing, mirroring
// TranscriptionModel._generate_token_stream for batch_size == 1.

import { NoteDecoder } from './decoder.js';
import { MAX_PREFILL } from './gpu.js';
import { autoLevel } from './level.js';
import { VOCAB, SAMPLE_RATE, FRAME_RATE, instrumentConditionRows, tieSectionTokens, forbiddenTokens, tokenId, programForInstrument } from './vocab.js';

const VOICE_INDEX_BASE = 1e9; // note indices of the separate voice pass
const NOTE_ON = new Set(['pitch', 'drum']);

/** True when generated tokens (after the forced prompt) contain any note or drum hit. */
function hasNotes(tokens, from) {
  for (let i = from; i < tokens.length; i++) if (NOTE_ON.has(VOCAB[tokens[i]]?.type)) return true;
  return false;
}

const VOICE_PROGRAM = programForInstrument('voice');
// Seconds later where the voice pass listens again after a dropout (first that hears singing wins).
const VOICE_RETRY_SHIFTS = [0.15, 0.5, 1];

/** Voice note-ons in generated tokens (after the forced prompt). */
function voiceNotes(tokens, from) {
  let program = null, n = 0;
  for (let i = from; i < tokens.length; i++) {
    const v = VOCAB[tokens[i]];
    if (v?.type === 'program') program = v.value;
    else if (v?.type === 'pitch' && program === VOICE_PROGRAM) n++;
  }
  return n;
}

const rms = (x) => {
  let e = 0;
  for (let i = 0; i < x.length; i += 4) e += x[i] * x[i];
  return Math.sqrt(e / Math.max(1, x.length / 4));
};

export class Transcriber {
  /** opts.autoLevel (default true): boost quiet chunks before the model hears them. */
  constructor(engine, mel, { autoLevel: level = true } = {}) {
    this.engine = engine;
    this.mel = mel;
    this.level = level ? autoLevel : (x) => x;
    this.instruments = null; // hard restriction (also conditions the model)
    this.hint = null; // conditioning only: tells the model what to expect, forbids nothing
    this.reset();
  }

  reset() {
    this.lastVoice = 0; // voice notes in the previous chunk (an empty voice pass after singing gets a retry)
    this.decoder = new NoteDecoder();
    this.chunks = 0;
    // The voice pass keeps its own continuity; its note indices never collide with the main pass.
    this.voiceDecoder = new NoteDecoder();
    this.voiceDecoder.nextIndex = VOICE_INDEX_BASE;
    this.voiceChunks = 0;
    this.keptVoice = new Set(); // voice-pass notes we kept (their ends follow later)
    this.droppedVoice = new Set(); // main-pass voice notes we replaced
  }

  /**
   * Instruments the model should expect, without forbidding others (MuScriptor's own
   * conditioning input). "voice" is special: it runs as a separate pass whose voice notes
   * are added, so the other instruments come out exactly as without the hint.
   */
  setHint(names) {
    const list = names && names.length ? names : [];
    this.voice = list.includes('voice');
    const rest = list.filter((n) => n !== 'voice');
    this.hint = rest.length ? rest : null;
  }

  /** Restrict (and condition) transcription to these instrument names; null for any. */
  setInstruments(names) {
    this.instruments = names && names.length ? names : null;
    this.engine.setForbidden(forbiddenTokens(this.instruments));
  }

  /**
   * One model pass over a chunk, continuing `decoder`. The tie prologue (notes held over
   * from the previous chunk) is forced, except on a first chunk where the model writes its
   * own (notes already sounding when the audio starts). A chunk that has sound but comes
   * back without a single note gets one more try from a clean start (held notes end at the
   * chunk start), unless we're in a hurry.
   */
  async _pass(decoder, mel, names, first, { loud = false, hurry = false } = {}) {
    const instRows = instrumentConditionRows(names);
    const prompt = first ? [] : this._prologue(decoder, mel, instRows);
    let out = await this.engine.generate({ mel, instRows, prompt });
    if (loud && !hurry && !hasNotes(out.tokens, prompt.length)) {
      const retry = await this.engine.generate({ mel, instRows, prompt: tieSectionTokens([]) });
      retry.generated += out.generated;
      out = retry;
    }
    const events = [];
    for (const t of out.tokens) events.push(...decoder.token(t));
    return { events, out };
  }

  /**
   * The tie prologue for `decoder`'s held notes. It must fit the prefill pass: if an
   * unusual pile of held notes would overflow it, sustain only what fits (the rest end at
   * the chunk start).
   */
  _prologue(decoder, mel, instRows) {
    const room = MAX_PREFILL - mel.frames - 2 - instRows.length;
    const keys = decoder.openKeys();
    let keep = keys.length;
    let prompt = tieSectionTokens(keys);
    while (prompt.length > room && keep > 0) prompt = tieSectionTokens(keys.slice(0, --keep));
    return prompt;
  }

  /**
   * The voice pass. Whether the model hears the singing in a chunk depends on where the
   * chunk starts, so when it comes back empty although singing is expected, it listens
   * again from a little later in the chunk. Returns {events, generated, found}.
   */
  async _voicePass(vdec, mel, leveled, seekTime, { hurry, expectVoice }) {
    const instRows = instrumentConditionRows(['voice']);
    const first = this.voiceChunks++ === 0;
    const prompt = first ? [] : this._prologue(vdec, mel, instRows);
    let out = await this.engine.generate({ mel, instRows, prompt });
    let generated = out.generated;
    if (expectVoice && !hurry && !voiceNotes(out.tokens, prompt.length)) {
      for (const shiftSec of VOICE_RETRY_SHIFTS) {
        const shift = Math.round(shiftSec * SAMPLE_RATE);
        const moved = new Float32Array(leveled.length);
        moved.set(leveled.subarray(shift));
        const retry = await this.engine.generate({ mel: this.mel.compute(moved), instRows, prompt: tieSectionTokens([]) });
        generated += retry.generated;
        if (!voiceNotes(retry.tokens, 1)) continue;
        out = retry;
        // Its times count from the shifted start (held notes end at the chunk start).
        vdec.startTick = Math.round((seekTime + shiftSec) * FRAME_RATE);
        vdec.tick = vdec.startTick;
        break;
      }
    }
    const events = [];
    for (const t of out.tokens) events.push(...vdec.token(t));
    return { events, generated, found: events.some((e) => e.type === 'start' && e.instrument === 'voice') };
  }

  /**
   * Main-pass events. While the voice pass runs, its voice notes replace the main pass's,
   * so those are dropped; the ends of dropped notes are always dropped too.
   */
  _main(events, voiceOn) {
    return events.filter((e) => {
      if (e.type === 'start') {
        if (voiceOn && e.instrument === 'voice') { this.droppedVoice.add(e.index); return false; }
        return true;
      }
      return !this.droppedVoice.delete(e.index);
    });
  }

  /** Voice-pass events: only the voice notes. */
  _onlyVoice(events) {
    return events.filter((e) => {
      if (e.type === 'start') {
        if (e.instrument !== 'voice') return false;
        this.keptVoice.add(e.index);
        return true;
      }
      return this.keptVoice.delete(e.index);
    });
  }

  /**
   * Transcribe one 5 s chunk. Pass samples = null to skip it (silence or
   * falling behind): open notes then end at this chunk's start.
   * If the GPU fails midway, what was decoded so far is still returned (failed: true),
   * so no note is left open in the panel.
   * @returns {events, stats, failed, error}
   */
  async processChunk(samples, seekTime, nextSeekTime, { hurry = false } = {}) {
    const decoder = this.decoder; // stays with this chunk even if reset() runs meanwhile
    const vdec = this.voiceDecoder;
    // Falling behind: the extra voice pass waits (its open notes end) rather than a chunk being lost.
    const voiceOn = this.voice && !this.instruments && !(hurry && samples);
    const mainEvents = decoder.boundary(seekTime, nextSeekTime);
    const voiceEvents = vdec.boundary(seekTime, nextSeekTime);
    const first = this.chunks++ === 0;
    let keepMainVoice = false;
    const result = () => ({ events: [...this._main(mainEvents, voiceOn && !keepMainVoice), ...this._onlyVoice(voiceEvents)] });
    if (!samples) return { ...result(), stats: null };
    const t0 = performance.now();
    let generated = 0, eos = false, tokens = [];
    let t1 = t0;
    try {
      const leveled = this.level(samples);
      const mel = this.mel.compute(leveled);
      t1 = performance.now();
      const opts = { loud: rms(leveled) > 0.01, hurry };
      const main = await this._pass(decoder, mel, this.instruments ?? this.hint, first, opts);
      mainEvents.push(...main.events);
      ({ generated, eos, tokens } = { generated: main.out.generated, eos: main.out.eos, tokens: main.out.tokens });
      if (voiceOn) {
        // Vocals were there a moment ago, or the main pass hears them now: an empty voice
        // pass is the model dropping out (it depends on where the chunk starts), so retry.
        const expectVoice = (this.lastVoice ?? 0) >= 3 || mainEvents.some((e) => e.type === 'start' && e.instrument === 'voice');
        const voice = await this._voicePass(vdec, mel, leveled, seekTime, { hurry, expectVoice });
        voiceEvents.push(...voice.events);
        this.lastVoice = voice.events.filter((e) => e.type === 'start').length;
        generated += voice.generated;
        // Still nothing, but the main pass heard singing: keep its voice notes for this chunk.
        if (!voice.found && expectVoice) keepMainVoice = true;
      }
    } catch (error) {
      return { ...result(), stats: null, failed: true, error };
    }
    return { ...result(), tokens, stats: { melMs: t1 - t0, genMs: performance.now() - t1, generated, eos } };
  }

  finish() {
    return [...this._main(this.decoder.finish(), false), ...this._onlyVoice(this.voiceDecoder.finish())];
  }

  /**
   * Transcribe one chunk on its own (no tie prologue from a neighbour), e.g.
   * to fill in a chunk that was skipped live. Notes still open at the end are
   * closed at the chunk end. Uses its own decoders, so live state is untouched.
   */
  async isolated(samples, seekTime, nextSeekTime) {
    const mel = this.mel.compute(this.level(samples));
    const once = async (names, base) => {
      const decoder = new NoteDecoder();
      decoder.nextIndex = base;
      const events = decoder.boundary(seekTime, nextSeekTime);
      events.push(...(await this._pass(decoder, mel, names, true)).events);
      decoder.boundary(nextSeekTime, null);
      events.push(...decoder.token(tokenId('tie', 0))); // empty tie set: close everything
      return events;
    };
    const events = await once(this.instruments ?? this.hint, 0);
    if (!this.voice || this.instruments) return events;
    const voice = await once(['voice'], VOICE_INDEX_BASE);
    const kept = new Set(voice.filter((e) => e.type === 'start' && e.instrument === 'voice').map((e) => e.index));
    const dropped = new Set(events.filter((e) => e.type === 'start' && e.instrument === 'voice').map((e) => e.index));
    return [...events.filter((e) => !dropped.has(e.index)), ...voice.filter((e) => kept.has(e.index))];
  }
}
