// Side panel: onboarding, model loading, audio capture and the live views.

import { PianoRoll, ZOOM_STEPS } from './ui/pianoroll.js';
import { Player } from './ui/player.js';
import { instrumentLabel, instrumentColor, noteName, drumName } from './ui/instruments.js';
import { chordSegments, chordParts, estimateKey } from './music/chords.js';
import { notesToMidi } from './music/midi.js';
import { INSTRUMENT_NAMES, SAMPLE_RATE, SEGMENT_SAMPLES } from './engine/vocab.js';
import { t, setLanguage, applyI18n, detectLanguage, LANGUAGES, lang } from './ui/i18n.js';
import { setNaming, defaultNaming, pcName, NAMING_MODES } from './music/names.js';

const $ = (id) => document.getElementById(id);
const isExtension = typeof chrome !== 'undefined' && !!chrome.tabCapture;
// Dev/test only (plain web page, not the extension): ?weights=<url>&src=<audio url>
const params = new URLSearchParams(isExtension ? '' : location.search);
const CHUNK_SEC = SEGMENT_SAMPLES / SAMPLE_RATE;
const CHORD_HISTORY_SEC = 30;
// Auto pause: this long below SILENCE_RMS (about -60 dBFS) pauses the timeline.
const SILENCE_RMS = 1e-3;
const SILENCE_SEC = 1.5;
const MAX_RECORDING_SEC = 60 * 60; // keep at most an hour of audio for review
const MODELS = {
  small: { label: 'pill.fast', mb: 209 },
  medium: { label: 'pill.accurate', mb: 615 },
};
// Single-line instruments: show their notes, not chord names.
const LINE_INSTRUMENTS = new Set([
  'acoustic_bass', 'electric_bass', 'contrabass', 'voice', 'trumpet', 'trombone', 'tuba', 'french_horn',
  'soprano_and_alto_sax', 'tenor_sax', 'baritone_sax', 'oboe', 'english_horn', 'bassoon', 'clarinet',
  'flutes', 'synth_lead', 'timpani',
]);

// ---------------------------------------------------------------- settings

const DEFAULTS = { accepted: false, token: '', model: 'small', f32: false, instruments: [], zoom: 12, lang: 'auto', naming: 'auto' };
let settings = { ...DEFAULTS };
const settingsStore = {
  async get() {
    if (isExtension) return { ...DEFAULTS, ...(await chrome.storage.local.get(null)) };
    try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('a2m') || '{}') }; } catch { return { ...DEFAULTS }; }
  },
  async set(patch) {
    Object.assign(settings, patch);
    if (isExtension) return chrome.storage.local.set(patch);
    try { localStorage.setItem('a2m', JSON.stringify(settings)); } catch { /* storage unavailable */ }
  },
};

// ---------------------------------------------------------------- UI helpers

const SCREENS = ['welcome', 'tokenScreen', 'loading', 'live'];
function showScreen(name) {
  for (const s of SCREENS) $(s).hidden = s !== name;
  $('dock').hidden = name !== 'live';
}

const ERROR_KEYS = {
  'bad-token': 'err.badToken', 'no-license': 'err.noLicense', download: 'err.download',
  'no-webgpu': 'err.noWebgpu', 'gpu-lost': 'err.gpuLost', 'cache-failed': 'msg.cacheFailed',
};
/** Worker/engine errors carry a code; translate it, else show the raw message. */
const errorText = (d) => (ERROR_KEYS[d.code] ? t(ERROR_KEYS[d.code], { status: d.status ?? '', error: d.error ?? '' }) : d.message);

function banner(text, kind = 'error') {
  $('banner').hidden = !text;
  $('bannerText').textContent = text || '';
  $('banner').className = `banner ${kind === 'info' ? 'info' : ''}`;
}
$('bannerClose').addEventListener('click', () => banner(''));

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

function chordHTML(ch, flats) {
  const p = chordParts(ch, flats);
  return `<span class="chord">${esc(p.root)}${p.suffix ? `<sup>${esc(p.suffix)}</sup>` : ''}${p.bass ? `<span class="slash">/${esc(p.bass)}</span>` : ''}</span>`;
}

function setArc(el, frac) {
  el.style.strokeDasharray = `${Math.max(0, Math.min(1, frac)) * 100} 100`;
}

function updatePill() {
  const pill = $('modelPill');
  pill.hidden = !modelInfo && !loading;
  pill.textContent = t(MODELS[settings.model]?.label ?? 'pill.fast');
  pill.classList.toggle('busy', loading);
  pill.title = modelInfo ? t('pill.title', { gpu: modelInfo.gpu || 'GPU' }) : '';
}

// ---------------------------------------------------------------- worker / model

const worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module' });
let modelInfo = null;
let loading = false;
let download = null; // {t0, loaded0} for the ETA

function loadModel(extra = {}) {
  player.pause();
  loading = true;
  modelInfo = null;
  download = null;
  banner('');
  showScreen('loading');
  setLoading(t('loading.ready'), '', null);
  updatePill();
  const msg = { type: 'load', model: settings.model, token: settings.token, f16: !settings.f32, ...extra };
  worker.postMessage(msg, msg.buffer ? [msg.buffer] : []);
}

function setLoading(title, detail, frac) {
  $('loadTitle').textContent = title;
  $('loadDetail').textContent = detail;
  $('loading').querySelector('.ring').classList.toggle('spin', frac === null);
  if (frac !== null) setArc($('loadArc'), frac);
}

function onProgress(m) {
  if (m.phase === 'download' && m.total) {
    const now = performance.now();
    if (!download) download = { t0: now, loaded0: m.loaded };
    const rate = (m.loaded - download.loaded0) / Math.max(1, now - download.t0); // bytes per ms
    const left = rate > 0 ? (m.total - m.loaded) / rate / 1000 : null;
    const eta = left === null || now - download.t0 < 1500 ? ''
      : left < 60 ? t('loading.etaSec', { n: Math.ceil(left / 5) * 5 }) : t('loading.etaMin', { n: Math.ceil(left / 60) });
    const progress = t('loading.progress', { done: (m.loaded / 1e6).toFixed(0), total: (m.total / 1e6).toFixed(0) });
    setLoading(t('loading.download'), `${[progress, eta].filter(Boolean).join(', ')}. ${t('loading.once')}`, m.frac);
  } else if (m.phase === 'download' || m.phase === 'config') {
    setLoading(t('loading.download'), t('loading.once'), null);
  } else if (m.phase === 'cache' || m.phase === 'file') {
    setLoading(t('loading.load'), '', null);
  } else if (m.phase === 'gpu') {
    setLoading(t('loading.gpu'), '', m.frac);
  }
}

worker.onmessage = ({ data }) => {
  switch (data.type) {
    case 'progress':
      onProgress(data);
      break;
    case 'ready':
      loading = false;
      modelInfo = data.info;
      worker.postMessage({ type: 'instruments', names: settings.instruments });
      showScreen('live');
      $('recBtn').disabled = !isExtension;
      updatePill();
      renderPanel();
      if (params.get('src')) startFile(params.get('src'));
      break;
    case 'loadError':
      loading = false;
      updatePill();
      if (data.code === 'needs-token') {
        showScreen('tokenScreen');
      } else {
        banner(errorText(data));
        showScreen(settings.token ? 'tokenScreen' : 'welcome');
      }
      break;
    case 'events':
      onEvents(data);
      break;
    case 'warning':
      banner(errorText(data), 'info');
      break;
    case 'cacheCleared':
      banner(t('msg.cacheCleared'), 'info');
      break;
    case 'error':
      banner(errorText(data));
      if (data.code === 'gpu-lost' && !loading) {
        if (capture) stopCapture();
        loadModel(); // the GPU reset (driver update, sleep…): bring the engine back
      }
      break;
  }
};

// ---------------------------------------------------------------- session state

const session = {
  notes: new Map(), // key -> {instrument, pitch, start, end|null}
  part: 0, // capture session id; bumps on every capture so note indices never collide
  minPart: 1, // events from older sessions (cleared) are ignored
  offset: 0, // session time where the current capture starts
  now: 0, // seconds of audio captured
  done: 0, // seconds transcribed
  chords: new Map(), // instrument -> segments
  ensemble: [],
  key: null,
  hidden: new Set(),
  dropped: 0,
  audio: [], // recorded chunks for review: [{seek, rate, data: Int16Array}]
  droppedSeeks: [], // [{part, seek}] chunks skipped live, transcribed once the session is final
  backfilling: 0,
  byInst: new Map(), // instrument -> notes, for the instrument rows
};
const player = new Player();
const reviewing = () => !capture && session.done > 0;

function addNote(k, note) {
  session.notes.set(k, note);
  if (!session.byInst.has(note.instrument)) session.byInst.set(note.instrument, []);
  const list = session.byInst.get(note.instrument);
  list.push(note);
  // Backfilled notes arrive out of order; keep each list sorted by start.
  if (list.length > 1 && list[list.length - 2].start > note.start) list.sort((a, b) => a.start - b.start);
}

function onEvents(msg) {
  if (msg.part < session.minPart) return; // from a session the user cleared
  if (msg.backfill) {
    session.backfilling = Math.max(0, session.backfilling - 1);
    if (!session.backfilling) banner('');
  }
  const prefix = msg.backfill ? `${msg.part}b${msg.backfill}` : `${msg.part}`;
  for (const ev of msg.events) {
    const k = `${prefix}:${ev.index}`;
    if (ev.type === 'start') addNote(k, { instrument: ev.instrument, pitch: ev.pitch, start: ev.time, end: null });
    else if (session.notes.has(k)) session.notes.get(k).end = ev.time;
  }
  if (msg.seek != null) session.done = Math.max(session.done, msg.seek + CHUNK_SEC);
  if (msg.dropped) {
    session.dropped++;
    session.droppedSeeks.push({ part: msg.part, seek: msg.seek });
  }
  if (msg.final) {
    session.done = Math.max(session.done, session.audioEnd ?? 0);
    backfill(msg.part); // now every skipped chunk of this session is known
  }
  roll.dirty = true;
  if (reviewing()) enterReview();
  else analyse();
  renderPanel();
}

/** Chords and key: the last 30 s while live, the whole session for review. */
function analyse(full = false) {
  const end = session.done;
  const from = full ? 0 : Math.max(0, Math.floor((end - CHORD_HISTORY_SEC) / 0.25) * 0.25);
  const byInst = new Map();
  const pitched = [];
  for (const n of session.notes.values()) {
    if ((n.end ?? end) < from) continue;
    if (!byInst.has(n.instrument)) byInst.set(n.instrument, []);
    byInst.get(n.instrument).push(n);
    if (n.instrument !== 'drums') pitched.push(n);
  }
  session.chords.clear();
  for (const [inst, notes] of byInst) {
    if (inst !== 'drums' && !LINE_INSTRUMENTS.has(inst)) session.chords.set(inst, chordSegments(notes, from, end, end));
  }
  session.ensemble = chordSegments(pitched, from, end, end);
  session.key = estimateKey(pitched, full ? 0 : Math.max(0, end - 45), end, end) ?? session.key;
}

// ---------------------------------------------------------------- rendering

const roll = new PianoRoll($('roll'));

/** The time the panel describes: the playhead in review, else the transcription head. */
const viewTime = () => (reviewing() ? player.time : session.done);

function currentSegment(segs, t = viewTime()) {
  if (reviewing()) return segs.find((s) => s.start <= t && t < s.end) ?? null;
  const last = segs[segs.length - 1];
  return last && last.end >= t - 1 ? last : null;
}

function renderPanel() {
  const flats = session.key?.useFlats ?? false;
  const hasNotes = session.notes.size > 0;
  const T = viewTime();
  const cur = currentSegment(session.ensemble, T);
  setHTML($('nowChord'), cur
    ? chordHTML(cur.chord, flats)
    : `<span class="idle">${capture ? esc(t(hasNotes ? 'live.noChord' : 'live.listening')) : ''}</span>`);
  const key = session.key;
  $('keyName').textContent = key ? t(key.minor ? 'key.minor' : 'key.major', { tonic: pcName(key.tonic, key.useFlats) }) : '';
  $('keyName').title = key ? t(reviewing() ? 'key.tipFull' : 'key.tipLive') : '';

  const behind = Math.max(0, session.now - session.done);
  let status = '';
  if (capture?.waiting) {
    status = t('live.waiting');
  } else if (capture) {
    status = hasNotes ? t('live.behind', { n: behind.toFixed(0) }) : capture.label;
    if (session.dropped) status += `, ${t('live.skipped', { n: session.dropped })}`;
  }
  setText($('heroStatus'), status);

  $('rollEmpty').textContent = hasNotes ? '' : capture
    ? t('live.emptyListening')
    : t(isExtension ? 'live.emptyIdle' : 'live.emptyDev');
  $('recBtn').classList.toggle('waiting', !!capture?.waiting);

  // Instruments by first appearance, drums last. Rows are reused so focus and hover survive updates.
  const insts = [...session.byInst.keys()]
    .sort((a, b) => (a === 'drums') - (b === 'drums') || session.byInst.get(a)[0].start - session.byInst.get(b)[0].start);
  const list = $('instruments');
  // Move only rows that are out of place: re-inserting a focused row would drop its focus.
  insts.forEach((inst, i) => {
    const row = instrumentRow(inst, flats, T);
    if (list.children[i] !== row) list.insertBefore(row, list.children[i] ?? null);
  });
  while (list.children.length > insts.length) list.lastChild.remove();
  roll.dirty = true;
  renderTransport();
  $('instruments').parentElement.hidden = insts.length === 0;
  $('exportBtn').disabled = !hasNotes;
  $('clearBtn').hidden = !hasNotes;
}

const rowCache = new Map(); // instrument -> <li>

/** Notes of `list` (sorted by start) that start at or before T. */
function upTo(list, T) {
  let lo = 0, hi = list.length;
  while (lo < hi) { const mid = (lo + hi) >> 1; if (list[mid].start <= T) lo = mid + 1; else hi = mid; }
  return list.slice(Math.max(0, lo - 64), lo);
}

function instrumentRow(inst, flats, T) {
  let li = rowCache.get(inst);
  if (!li) {
    li = document.createElement('li');
    li.className = 'inst';
    li.tabIndex = 0;
    li.setAttribute('role', 'switch');
    li.innerHTML = `<span class="dot" style="background:${instrumentColor(inst)}"></span>
      <span class="name"><span class="label"></span><span class="count"></span></span>
      <span class="current"></span><span class="trail"></span>`;
    const toggle = () => {
      if (session.hidden.has(inst)) session.hidden.delete(inst); else session.hidden.add(inst);
      roll.dirty = true;
      renderPanel();
    };
    li.addEventListener('click', toggle);
    li.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle(); } });
    rowCache.set(inst, li);
  }
  const off = session.hidden.has(inst);
  li.classList.toggle('off', off);
  li.setAttribute('aria-checked', String(!off));
  li.title = t(off ? 'insts.show' : 'insts.hide');
  const all = session.byInst.get(inst);
  const notes = upTo(all, T);
  const segs = (session.chords.get(inst) || []).filter((sg) => sg.start <= T);
  let current = '';
  let trail = '';
  if (inst === 'drums') {
    const hits = new Map();
    for (const n of notes) if (n.start >= T - CHUNK_SEC) hits.set(drumName(n.pitch), (hits.get(drumName(n.pitch)) || 0) + 1);
    trail = [...hits].sort((a, b) => b[1] - a[1]).slice(0, 4).map(([name, c]) => `<span>${esc(name)} ${c}</span>`).join('');
  } else if (segs.length) {
    const cur = currentSegment(segs, T);
    current = cur ? chordHTML(cur.chord, flats) : '';
    trail = segs.slice(-6).map((s) => chordHTML(s.chord, flats)).join('');
  } else {
    const last = notes.slice(-8);
    const lastNote = last[last.length - 1];
    if (lastNote && lastNote.start >= T - 2) current = `<span class="chord">${esc(noteName(lastNote.pitch, flats).replace(/-?\d+$/, ''))}</span>`;
    trail = `<span class="notes">${esc(last.map((n) => noteName(n.pitch, flats)).join('  '))}</span>`;
  }
  setText(li.querySelector('.label'), instrumentLabel(inst));
  setText(li.querySelector('.count'), String(all.length));
  setHTML(li.querySelector('.current'), current);
  setHTML(li.querySelector('.trail'), trail);
  return li;
}

// Only touch the DOM when content changes (keeps screen readers quiet and hover stable).
function setHTML(el, html) { if (el._html !== html) { el._html = html; el.innerHTML = html; } }
function setText(el, text) { if (el.textContent !== text) el.textContent = text; }

function frame() {
  if (capture) {
    session.now = session.offset + capture.captured / capture.rate;
    setArc($('chunkArc'), capture.fill / capture.chunk.length);
    capture.levelShown += (capture.level - capture.levelShown) * 0.3;
    $('recBtn').style.setProperty('--level', Math.min(1, capture.levelShown * 6).toFixed(3));
  }
  const animating = capture || player.playing || drag || scrubbing || roll.easing;
  if (!$('live').hidden && (animating || roll.dirty)) {
    roll.dirty = false;
    const review = reviewing();
    const t = player.time;
    roll.draw({
      notes: session.notes.values(),
      now: Math.max(session.now, session.done),
      done: session.done,
      hidden: session.hidden,
      chords: session.ensemble,
      flats: session.key?.useFlats ?? false,
      playhead: review ? t : null,
      center: review ? t : null,
    });
    if (review && !scrubbing) $('scrub').value = player.duration ? Math.round((t / player.duration) * 1000) : 0;
    if (review) {
      setText($('timeLabel'), `${fmtTime(t)} / ${fmtTime(player.duration)}`);
      $('scrub').setAttribute('aria-valuetext', fmtTime(t));
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
// Keep the "behind" readout ticking between chunks.
setInterval(() => { if (capture) renderPanel(); }, 1000);
// Follow the playhead: chords, key and instrument rows at the current time.
setInterval(() => { if (player.playing) renderPanel(); }, 250);

// ---------------------------------------------------------------- capture

let capture = null;


async function resample16k(samples, rate) {
  if (rate === SAMPLE_RATE) return samples;
  const oc = new OfflineAudioContext(1, SEGMENT_SAMPLES, SAMPLE_RATE);
  const buf = oc.createBuffer(1, samples.length, rate);
  buf.copyToChannel(samples, 0);
  const src = oc.createBufferSource();
  src.buffer = buf;
  src.connect(oc.destination);
  src.start();
  return (await oc.startRendering()).getChannelData(0);
}

/** Wire `makeSource(ctx)` into playback, the recorder and the 5 s chunker. */
async function startCapture(makeSource, label, cleanup) {
  const ctx = new AudioContext({ latencyHint: 'playback' });
  await ctx.audioWorklet.addModule('capture-worklet.js');
  const source = await makeSource(ctx);
  // Capturing a tab mutes it, so play it straight through: no added delay.
  source.connect(ctx.destination);
  const tap = new AudioWorkletNode(ctx, 'capture', { numberOfOutputs: 1 });
  const mute = ctx.createGain();
  mute.gain.value = 0;
  source.connect(tap).connect(mute).connect(ctx.destination);

  const rate = ctx.sampleRate;
  const chunkLen = Math.round(CHUNK_SEC * rate);
  player.pause();
  const part = ++session.part;
  session.offset = Math.max(session.done, session.now);
  worker.postMessage({ type: 'part', part });
  const c = {
    ctx, rate, label, cleanup, captured: 0, chunk: new Float32Array(chunkLen), fill: 0, index: 0,
    stopped: false, pending: Promise.resolve(), level: 0, levelShown: 0, quietFor: 0, waiting: false,
  };
  const send = async (samples, seek, final) => {
    let energy = 0;
    for (let i = 0; i < samples.length; i += 4) energy += samples[i] * samples[i];
    if (Math.sqrt(energy / (samples.length / 4)) < 1e-4) {
      worker.postMessage({ type: 'skip', part, seek, next: seek + CHUNK_SEC });
    } else {
      const copy = new Float32Array(await resample16k(samples, rate));
      worker.postMessage({ type: 'chunk', part, samples: copy, seek, next: seek + CHUNK_SEC }, [copy.buffer]);
    }
    if (final) worker.postMessage({ type: 'finish', part });
  };
  // Chunks are handed over in order even though resampling is async.
  c.flush = (final) => {
    const seek = session.offset + c.index * CHUNK_SEC;
    c.index++;
    const samples = c.chunk;
    c.chunk = new Float32Array(chunkLen);
    c.fill = 0;
    record(samples, seek, rate);
    c.pending = c.pending.then(() => send(samples, seek, final));
    return c.pending;
  };
  tap.port.onmessage = ({ data }) => {
    if (c.stopped) return;
    let e = 0;
    for (let i = 0; i < data.length; i += 8) e += data[i] * data[i];
    c.level = Math.sqrt(e / (data.length / 8));
    // Auto pause on silence, auto resume when sound returns.
    c.quietFor = c.level < SILENCE_RMS ? c.quietFor + data.length / rate : 0;
    if (c.waiting) {
      if (c.level < SILENCE_RMS) return;
      c.waiting = false;
      renderPanel();
    } else if (c.quietFor >= SILENCE_SEC) {
      c.waiting = true;
      if (c.fill > 0.5 * rate) {
        c.flush(false); // transcribe what was heard; the zero padding becomes timeline
        c.captured = c.index * chunkLen;
      } else {
        c.captured -= c.fill; // drop a sliver of near-silence
        c.chunk.fill(0);
        c.fill = 0;
      }
      renderPanel();
      return;
    }
    let off = 0;
    while (off < data.length) {
      const n = Math.min(data.length - off, chunkLen - c.fill);
      c.chunk.set(data.subarray(off, off + n), c.fill);
      c.fill += n;
      off += n;
      c.captured += n;
      if (c.fill === chunkLen) c.flush(false);
    }
  };
  capture = c;
  const rec = $('recBtn');
  rec.classList.add('on');
  rec.disabled = false;
  rec.dataset.i18nAttr = 'aria-label:dock.stopAria';
  $('recLabel').dataset.i18n = 'dock.stop';
  applyI18n(rec.parentElement);
  banner('');
  renderPanel();
}

async function stopCapture() {
  const c = capture;
  if (!c || c.stopped) return;
  c.stopped = true;
  capture = null;
  if (c.fill > 0.5 * c.rate) c.flush(true);
  else c.pending = c.pending.then(() => worker.postMessage({ type: 'finish', part: session.part }));
  c.cleanup?.();
  session.audioEnd = session.offset + c.captured / c.rate;
  const rec = $('recBtn');
  rec.classList.remove('on', 'waiting');
  rec.disabled = true; // until the last chunk is handed over
  rec.style.setProperty('--level', 0);
  rec.dataset.i18nAttr = 'aria-label:dock.listenAria';
  $('recLabel').dataset.i18n = 'dock.listen';
  applyI18n(rec.parentElement);
  setArc($('chunkArc'), 0);
  renderPanel();
  await c.pending;
  await c.ctx.close();
  rec.disabled = !isExtension;
  enterReview();
  renderPanel();
}

/** Keep the chunk for review playback: skip silence, cap the total length. */
function record(samples, seek, rate) {
  let peak = 0;
  for (let i = 0; i < samples.length; i += 16) peak = Math.max(peak, Math.abs(samples[i]));
  if (peak < 1e-4) return;
  if (session.audio.length * CHUNK_SEC >= MAX_RECORDING_SEC) return;
  session.audio.push({ seek, rate, data: toInt16(samples) });
}

const toInt16 = (f32) => {
  const out = new Int16Array(f32.length);
  for (let i = 0; i < f32.length; i++) out[i] = Math.max(-32768, Math.min(32767, Math.round(f32[i] * 32767)));
  return out;
};

/** Transcribe the chunks of `part` that were skipped live, from the recording. */
async function backfill(part) {
  const mine = session.droppedSeeks.filter((d) => d.part === part);
  session.droppedSeeks = session.droppedSeeks.filter((d) => d.part !== part);
  const segs = new Map(session.audio.map((a) => [a.seek, a]));
  for (const { seek } of mine) {
    const seg = segs.get(seek);
    if (!seg) continue;
    const f32 = new Float32Array(seg.data.length);
    for (let i = 0; i < f32.length; i++) f32[i] = seg.data[i] / 32768;
    const samples = new Float32Array(await resample16k(f32, seg.rate));
    if (part < session.minPart) return; // cleared while resampling
    session.backfilling++;
    worker.postMessage({ type: 'backfill', part, key: `${seek}`, samples, seek, next: seek + CHUNK_SEC }, [samples.buffer]);
  }
  if (session.backfilling) {
    banner(t('msg.backfill', { n: session.backfilling }), 'info');
    session.dropped = 0;
  }
}

/** Switch to review: full-session chords, and load the recording + notes into the player. */
function enterReview() {
  analyse(true);
  const last = session.audio[session.audio.length - 1];
  const duration = Math.max(session.done, last ? last.seek + last.data.length / last.rate : 0);
  player.load({ segments: session.audio, notes: [...session.notes.values()], duration, hidden: session.hidden });
}

const fmtTime = (t) => `${Math.floor(t / 60)}:${String(Math.floor(t % 60)).padStart(2, '0')}`;

function renderTransport() {
  const show = reviewing();
  $('transport').hidden = !show;
  $('roll').classList.toggle('scrubbable', show);
  if (!show) return;
  $('playBtn').classList.toggle('playing', player.playing);
  const pos = player.time;
  if (!scrubbing) $('scrub').value = player.duration ? Math.round((pos / player.duration) * 1000) : 0;
  setText($('timeLabel'), `${fmtTime(pos)} / ${fmtTime(player.duration)}`);
  $('scrub').setAttribute('aria-valuetext', fmtTime(pos));
  $('playBtn').setAttribute('aria-label', t(player.playing ? 'transport.pause' : 'transport.play'));
  const hasAudio = session.audio.length > 0;
  for (const b of $('transport').querySelectorAll('.switch button')) {
    b.setAttribute('aria-checked', String(b.dataset.mode === player.mode));
    if (b.dataset.mode === 'original') b.disabled = !hasAudio;
  }
  if (!hasAudio && player.mode === 'original') player.setMode('midi');
}

async function startTab() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab) throw new Error(t('err.noTab'));
  let streamId;
  try {
    streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: tab.id });
  } catch (e) {
    if (/invoked|activeTab|permission/i.test(e.message)) throw new Error(t('err.invoke'));
    if (/active stream/i.test(e.message)) throw new Error(t('err.busy'));
    throw new Error(t('err.capture', { error: e.message }));
  }
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { mandatory: { chromeMediaSource: 'tab', chromeMediaSourceId: streamId } },
      video: false,
    });
  } catch (e) {
    throw new Error(t('err.capture', { error: e.message }));
  }
  await startCapture((ctx) => ctx.createMediaStreamSource(stream), tab.title || t('live.thisTab'),
    () => stream.getTracks().forEach((t) => t.stop()));
  stream.getAudioTracks()[0].addEventListener('ended', stopCapture);
}

async function startFile(fileOrUrl) {
  const owned = typeof fileOrUrl !== 'string';
  const url = owned ? URL.createObjectURL(fileOrUrl) : fileOrUrl;
  const audio = new Audio(url);
  audio.crossOrigin = 'anonymous';
  await startCapture(async (ctx) => {
    const src = ctx.createMediaElementSource(audio);
    await audio.play();
    return src;
  }, owned ? fileOrUrl.name : url.split('/').pop(), () => {
    audio.pause();
    if (owned) URL.revokeObjectURL(url);
  });
  audio.addEventListener('ended', stopCapture);
}

// ---------------------------------------------------------------- controls

function showZoom(sec) {
  $('zoomLabel').textContent = t('zoom.seconds', { n: sec });
  $('zoomIn').disabled = sec <= ZOOM_STEPS[0];
  $('zoomOut').disabled = sec >= ZOOM_STEPS[ZOOM_STEPS.length - 1];
}
roll.onZoom = (sec) => {
  roll.dirty = true;
  showZoom(sec);
  settingsStore.set({ zoom: sec });
};
$('zoomIn').addEventListener('click', () => roll.zoom(-1));
$('zoomOut').addEventListener('click', () => roll.zoom(1));

$('agreeBtn').addEventListener('click', async () => {
  await settingsStore.set({ accepted: true });
  loadModel();
});

$('tokenForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  await settingsStore.set({ token: $('tokenInput').value.trim(), accepted: true });
  loadModel();
});

$('modelFile').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (file) loadModel({ buffer: await file.arrayBuffer() });
});

$('recBtn').addEventListener('click', async () => {
  if (capture) return stopCapture();
  $('recBtn').disabled = true;
  try {
    await startTab();
  } catch (e) {
    banner(e.message);
    $('recBtn').disabled = false;
  }
});

$('fileInput').addEventListener('change', async (e) => {
  const file = e.target.files[0];
  e.target.value = '';
  if (!file || !modelInfo) return;
  if (capture) await stopCapture();
  try { await startFile(file); } catch (err) { banner(t('err.capture', { error: err.message })); }
});

$('exportBtn').addEventListener('click', () => {
  const notes = [...session.notes.values()].map((n) => ({ ...n, end: n.end ?? session.done }));
  const blob = new Blob([notesToMidi(notes)], { type: 'audio/midi' });
  const a = document.createElement('a');
  const d = new Date();
  const pad = (x) => String(x).padStart(2, '0');
  a.href = URL.createObjectURL(blob);
  a.download = `byear-${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}-${pad(d.getHours())}${pad(d.getMinutes())}.mid`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
});

$('clearBtn').addEventListener('click', () => {
  player.stop();
  session.audio = [];
  session.droppedSeeks = [];
  session.notes.clear();
  session.byInst.clear();
  rowCache.clear();
  session.chords.clear();
  session.ensemble = [];
  session.key = null;
  session.dropped = 0;
  session.backfilling = 0;
  banner('');
  if (capture) {
    // Keep listening: only notes from earlier sessions are dropped from now on.
    session.minPart = session.part;
  } else {
    session.minPart = session.part + 1;
    worker.postMessage({ type: 'abort' });
    session.now = session.done = session.offset = 0;
    session.audioEnd = 0;
  }
  roll.dirty = true;
  renderPanel();
});

// Review transport
$('playBtn').addEventListener('click', async () => {
  if (player.playing) player.pause(); else await player.play();
  renderPanel();
});
player.onEnd = () => renderPanel();
for (const b of $('transport').querySelectorAll('.switch button')) {
  b.addEventListener('click', () => {
    player.setMode(b.dataset.mode);
    renderPanel();
  });
}
let scrubbing = false;
$('scrub').addEventListener('input', () => {
  scrubbing = true;
  player.pos = ($('scrub').value / 1000) * player.duration;
  if (!player.playing) renderPanel();
});
$('scrub').addEventListener('change', () => {
  scrubbing = false;
  player.seek(($('scrub').value / 1000) * player.duration);
  renderPanel();
});
// Click the piano roll to move the playhead there; drag to scrub.
let drag = null;
$('roll').addEventListener('pointerdown', (e) => {
  if (!reviewing()) return;
  drag = { x: e.clientX, t: player.time, was: player.playing, moved: false };
  $('roll').setPointerCapture(e.pointerId);
});
$('roll').addEventListener('pointermove', (e) => {
  if (!drag) return;
  if (!drag.moved && Math.abs(e.clientX - drag.x) < 4) return;
  if (!drag.moved) {
    drag.moved = true;
    player.pause();
  }
  const w = $('roll').getBoundingClientRect().width;
  player.pos = Math.max(0, Math.min(player.duration, drag.t - ((e.clientX - drag.x) / w) * roll.window));
  renderPanel();
});
$('roll').addEventListener('pointerup', async (e) => {
  if (!drag) return;
  const { moved, was } = drag;
  drag = null;
  if (moved) {
    if (was) await player.play();
  } else {
    const r = $('roll').getBoundingClientRect();
    player.seek(roll.t0 + ((e.clientX - r.left) / r.width) * roll.window); // keeps playing if it was
  }
  renderPanel();
});
$('roll').addEventListener('pointercancel', () => { drag = null; });
document.addEventListener('keydown', (e) => {
  if (!reviewing() || $('live').hidden || $('settings').open) return;
  if (e.target.closest('input, button, select, textarea, a, summary, label, dialog, [role=switch]')) return;
  if (e.key === ' ') { e.preventDefault(); $('playBtn').click(); return; }
  const dir = { ArrowLeft: -5, ArrowRight: 5 }[e.key];
  if (!dir) return;
  e.preventDefault();
  player.seek(player.time + (document.dir === 'rtl' ? -dir : dir));
  roll.dirty = true;
  renderPanel();
});

// Settings sheet
const chips = $('optInstruments');
function renderSettingsText() {
  chips.replaceChildren(...INSTRUMENT_NAMES.map((name) => {
    const l = document.createElement('label');
    l.innerHTML = `<input type="checkbox" value="${name}">${esc(instrumentLabel(name))}`;
    return l;
  }));
  $('fastDesc').textContent = t('settings.fastDesc', { mb: MODELS.small.mb });
  $('accurateDesc').textContent = t('settings.accurateDesc', { mb: MODELS.medium.mb });
  const detected = detectLanguage();
  $('optLang').replaceChildren(
    new Option(t('settings.languageAuto', { name: LANGUAGES[detected] }), 'auto'),
    ...Object.entries(LANGUAGES).map(([code, name]) => new Option(name, code)),
  );
  const example = t(`noteNames.${defaultNaming(lang)}`);
  $('optNaming').replaceChildren(
    new Option(t('settings.noteNamesAuto', { example }), 'auto'),
    ...NAMING_MODES.map((m) => new Option(t(`noteNames.${m}`), m)),
  );
}

/** Apply language + note-name settings to everything on screen. */
async function applyLocale() {
  await setLanguage(settings.lang === 'auto' ? detectLanguage() : settings.lang);
  setNaming(settings.naming === 'auto' ? defaultNaming(lang) : settings.naming, lang);
  applyI18n();
  renderSettingsText();
  $('welcomeFine').textContent = t('welcome.size', { mb: MODELS[settings.model].mb });
  showZoom(roll.window);
  updatePill();
  renderPanel();
}

const openSettings = () => {
  for (const r of document.querySelectorAll('input[name=model]')) r.checked = r.value === settings.model;
  $('optLang').value = settings.lang;
  $('optNaming').value = settings.naming;
  $('optToken').value = settings.token;
  $('optF32').checked = settings.f32;
  for (const box of chips.querySelectorAll('input')) box.checked = settings.instruments.includes(box.value);
  $('settings').returnValue = ''; // Esc keeps the previous value otherwise, which would save
  $('settings').showModal();
};
$('settingsBtn').addEventListener('click', openSettings);
$('modelPill').addEventListener('click', openSettings);

$('settings').addEventListener('close', async () => {
  if ($('settings').returnValue !== 'save') return;
  const next = {
    model: document.querySelector('input[name=model]:checked')?.value ?? settings.model,
    token: $('optToken').value.trim(),
    f32: $('optF32').checked,
    instruments: [...chips.querySelectorAll('input:checked')].map((b) => b.value),
    lang: $('optLang').value,
    naming: $('optNaming').value,
  };
  const localeChanged = next.lang !== settings.lang || next.naming !== settings.naming;
  const reload = next.model !== settings.model || next.f32 !== settings.f32 || (next.token !== settings.token && !modelInfo);
  await settingsStore.set(next);
  if (localeChanged) await applyLocale();
  worker.postMessage({ type: 'instruments', names: settings.instruments });
  if (reload && settings.accepted) {
    if (capture) await stopCapture();
    loadModel();
  }
});

$('clearCache').addEventListener('click', () => worker.postMessage({ type: 'clearCache' }));

// ---------------------------------------------------------------- boot

settings = await settingsStore.get();
roll.window = ZOOM_STEPS.includes(settings.zoom) ? settings.zoom : 12;
showZoom(roll.window);
await applyLocale();
if (params.get('weights')) loadModel({ url: new URL(params.get('weights'), location.href).href });
else if (settings.accepted) loadModel();
else showScreen('welcome');
