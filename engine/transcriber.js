// Chunk-by-chunk transcription with prelude forcing, mirroring
// TranscriptionModel._generate_token_stream for batch_size == 1.

import { NoteDecoder } from './decoder.js';
import { MAX_PREFILL } from './gpu.js';
import { autoLevel } from './level.js';
import { instrumentConditionRows, tieSectionTokens, forbiddenTokens, tokenId } from './vocab.js';

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
    this.decoder = new NoteDecoder();
    this.chunks = 0;
  }

  /**
   * Instruments the model should expect, without forbidding others. This is MuScriptor's
   * own conditioning input; e.g. a "voice" hint recovers vocals the model otherwise drops.
   */
  setHint(names) {
    this.hint = names && names.length ? names : null;
  }

  /** Restrict (and condition) transcription to these instrument names; null for any. */
  setInstruments(names) {
    this.instruments = names && names.length ? names : null;
    this.engine.setForbidden(forbiddenTokens(this.instruments));
  }

  /**
   * Transcribe one 5 s chunk. Pass samples = null to skip it (silence or
   * falling behind): open notes then end at this chunk's start.
   * @returns {events, stats}
   */
  async processChunk(samples, seekTime, nextSeekTime) {
    const decoder = this.decoder; // stays with this chunk even if reset() runs meanwhile
    const events = decoder.boundary(seekTime, nextSeekTime);
    const first = this.chunks++ === 0;
    if (!samples) return { events, stats: null };
    const t0 = performance.now();
    const mel = this.mel.compute(this.level(samples));
    const instRows = instrumentConditionRows(this.instruments ?? this.hint);
    let prompt = [];
    if (!first) {
      // The tie prologue must fit the prefill pass. If an unusual pile of held notes
      // would overflow it, sustain only what fits; the rest end at the chunk start.
      const room = MAX_PREFILL - mel.frames - 2 - instRows.length;
      const keys = decoder.openKeys();
      let keep = keys.length;
      prompt = tieSectionTokens(keys);
      while (prompt.length > room && keep > 0) prompt = tieSectionTokens(keys.slice(0, --keep));
    }
    const t1 = performance.now();
    const out = await this.engine.generate({ mel, instRows, prompt });
    for (const t of out.tokens) events.push(...decoder.token(t));
    const t2 = performance.now();
    return {
      events,
      tokens: out.tokens,
      stats: { melMs: t1 - t0, genMs: t2 - t1, generated: out.generated, eos: out.eos },
    };
  }

  finish() {
    return this.decoder.finish();
  }

  /**
   * Transcribe one chunk on its own (no tie prologue from a neighbour), e.g.
   * to fill in a chunk that was skipped live. Notes still open at the end are
   * closed at the chunk end. Uses its own decoder, so live state is untouched.
   */
  async isolated(samples, seekTime, nextSeekTime) {
    const decoder = new NoteDecoder();
    const events = decoder.boundary(seekTime, nextSeekTime);
    const out = await this.engine.generate({
      mel: this.mel.compute(this.level(samples)),
      instRows: instrumentConditionRows(this.instruments ?? this.hint),
    });
    for (const t of out.tokens) events.push(...decoder.token(t));
    decoder.boundary(nextSeekTime, null);
    events.push(...decoder.token(tokenId('tie', 0))); // empty tie set: close everything
    return events;
  }
}
