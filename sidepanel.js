// Side panel: onboarding, model loading, audio capture and the live views.

import { PianoRoll, ZOOM_STEPS, TAB_MAX_WINDOW } from './ui/pianoroll.js';
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
const PREROLL_SEC = 0.5; // audio kept from before the first sound (note attacks)
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

const DEFAULTS = {
  accepted: false, token: '', model: 'small', f32: false, instruments: [], zoom: 12, lang: 'auto', naming: 'auto',
  vocals: false, // tell the model the song has singing (conditioning hint)
  view: 'roll', // canvas: 'roll' (piano roll) or 'tab' (tablature)
  refine: true, // re-transcribe the whole recording after stop
};
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
  cancelRefine();
  session.backfilling = 0; // the worker drops queued jobs on load
  loading = true;
  modelInfo = null;
  download = null;
  banner('');
  showScreen('loading');
  setLoading(t('loading.ready'), '', null);
  updatePill();
  const msg = { type: 'load', model: settings.model, token: settings.token, f16: !settings.f32, autoLevel: settings.autoLevel !== false, ...extra };
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
  order: [], // channel order set by drag and drop (also the tab staff order)
  muted: new Set(), // channel strip M
  solo: new Set(), // channel strip S
  viewEnd: null, // live view: fixed right edge while scrolled back; null follows the newest notes
  dropped: 0,
  audio: [], // recorded chunks for review: [{seek, rate, data: Int16Array}]
  droppedSeeks: [], // [{part, seek}] chunks skipped live, transcribed once the session is final
  backfilling: 0,
  byInst: new Map(), // instrument -> notes, for the instrument rows
  version: 0, // bumps whenever notes change (chord cache)
  refine: null, // {parts:Set, runs, chunks, done, notes:Map} while re-transcribing after stop
  ignoreParts: new Set(), // sessions whose late results must be dropped (cancelled refine)
  selected: null, // the note clicked last (Alt+arrows change its pitch)
  edits: [], // user corrections, re-applied when refine replaces the notes
};

// ---------------------------------------------------------------- note editing

function selectNote(n) {
  session.selected = n;
  roll.dirty = true;
  if (n) player.preview(n);
}

function afterEdit() {
  session.version++;
  roll.dirty = true;
  if (reviewing()) enterReview(); else analyse();
  renderPanel();
}

/** Move the selected note by `semis` semitones (Alt+↑/↓, with Shift an octave). */
function transposeSelected(semis) {
  const n = session.selected;
  if (!n || n.instrument === 'drums') return;
  const to = Math.max(0, Math.min(127, n.pitch + semis));
  if (to === n.pitch) return;
  session.edits.push({ instrument: n.instrument, start: n.start, from: n.pitch, to });
  roll.fingering.forget(n, n.instrument);
  roll.lowCache?.delete(n.instrument); // the part's range (and so its tuning) may change
  delete n.forcedOpen;
  n.pitch = to;
  player.preview(n);
  afterEdit();
}

/** Tab view: move the selected note to the next string up/down that can play it. */
function moveSelectedString(dir) {
  const n = session.selected;
  if (!n || n.instrument === 'drums') return;
  const pos = roll.moveString(n, session.byInst.get(n.instrument), dir);
  if (!pos) return;
  session.edits.push({ instrument: n.instrument, start: n.start, from: n.pitch, to: n.pitch, open: n.forcedOpen });
  player.preview(n);
  roll.dirty = true;
}

function deleteSelected() {
  const n = session.selected;
  if (!n) return;
  let found = false;
  for (const [k, v] of session.notes) if (v === n) { session.notes.delete(k); found = true; break; }
  if (!found) { selectNote(null); return; } // a note from before a Clear/refine: nothing to delete
  const list = session.byInst.get(n.instrument);
  const i = list?.indexOf(n) ?? -1;
  if (i >= 0) list.splice(i, 1);
  if (list && !list.length) session.byInst.delete(n.instrument);
  session.edits.push({ instrument: n.instrument, start: n.start, from: n.pitch, deleted: true });
  session.selected = null;
  afterEdit();
}

/** After refine swaps the notes in, apply the user's corrections to the new notes. */
function reapplyEdits() {
  const used = new Set();
  for (const e of session.edits) {
    const list = session.byInst.get(e.instrument) || [];
    const n = list.find((m) => !used.has(m) && m.pitch === e.from && Math.abs(m.start - e.start) < 0.06);
    if (!n) continue;
    used.add(n);
    if (e.deleted) {
      for (const [k, v] of session.notes) if (v === n) { session.notes.delete(k); break; }
      list.splice(list.indexOf(n), 1);
      if (!list.length) session.byInst.delete(e.instrument);
    } else {
      n.pitch = e.to;
      if (e.open != null) n.forcedOpen = e.open; // the whole-song fingering keeps the string
    }
  }
}

// ---------------------------------------------------------------- hints & refine

/**
 * Hint for the refine pass: the instruments clearly present, never voice by itself.
 * "voice" is added only by the vocals toggle, and then runs as a separate pass, so the
 * toggle changes nothing but the voice track.
 */
function lineup() {
  const total = [...session.byInst.values()].reduce((a, l) => a + l.length, 0);
  const names = [...session.byInst.entries()]
    .filter(([inst, l]) => inst !== 'voice' && !inst.startsWith('program_') && l.length >= Math.max(12, total * 0.02))
    .map(([inst]) => inst);
  if (settings.vocals) names.push('voice');
  return names;
}

/** Re-transcribe the recording with the session's line-up as a hint, then swap the notes in. */
/** Starts refining when it is safe; returns false when it isn't (then live notes stay). */
function startRefine() {
  // Not while listening (it would replace notes mid-capture), not while chunks are being
  // filled in, and not on a capped recording (it doesn't cover the whole session).
  const complete = session.audio.length * CHUNK_SEC < MAX_RECORDING_SEC;
  if (!settings.refine || capture || !session.audio.length || !complete || session.backfilling) return false;
  refineRun();
  return true;
}

async function refineRun() {
  cancelRefine();
  const segs = [...session.audio].sort((a, b) => a.seek - b.seek);
  if (!segs.length) return;
  const hint = lineup();
  const runs = [];
  for (const seg of segs) {
    const run = runs[runs.length - 1];
    if (run && Math.abs(run[run.length - 1].seek + CHUNK_SEC - seg.seek) < 1e-6) run.push(seg); else runs.push([seg]);
  }
  const r = { parts: new Set(), runs: runs.length, chunks: segs.length, done: 0, notes: new Map(), failed: false };
  session.refine = r;
  banner(t('msg.refining', { pct: 0 }), 'info');
  for (const run of runs) {
    const part = ++session.part;
    r.parts.add(part);
    worker.postMessage({ type: 'part', part });
    worker.postMessage({ type: 'hint', part, names: hint });
    for (const seg of run) {
      const f32 = new Float32Array(seg.data.length);
      for (let i = 0; i < f32.length; i++) f32[i] = seg.data[i] / 32768;
      const samples = new Float32Array(await resample16k(f32, seg.rate));
      if (session.refine !== r) return; // cancelled meanwhile
      worker.postMessage({ type: 'chunk', part, samples, seek: seg.seek, next: seg.seek + CHUNK_SEC }, [samples.buffer]);
    }
    worker.postMessage({ type: 'finish', part });
  }
  worker.postMessage({ type: 'hint', part: session.part, names: null });
}

function onRefineEvents(msg) {
  const r = session.refine;
  if (msg.failed) r.failed = true;
  for (const ev of msg.events) {
    const k = `${msg.part}:${ev.index}`;
    if (ev.type === 'start') r.notes.set(k, { instrument: ev.instrument, pitch: ev.pitch, start: ev.time, end: null });
    else if (r.notes.has(k)) r.notes.get(k).end = ev.time;
  }
  if (msg.seek != null) {
    r.done++;
    banner(t('msg.refining', { pct: Math.round((100 * r.done) / r.chunks) }), 'info');
  }
  if (msg.final && --r.runs === 0 && r.failed) {
    // Something went wrong on the GPU: keep the live notes rather than a partial result.
    session.refine = null;
    banner(t('msg.refineFailed'));
    return;
  }
  if (msg.final && r.runs === 0) {
    // Swap in the refined transcription in one step.
    session.refine = null;
    session.version++;
    session.notes = new Map();
    session.byInst.clear();
    for (const [k, n] of r.notes) addNote(k, n);
    session.selected = null;
    reapplyEdits(); // keep the user's corrections
    roll.refinger(session.byInst); // whole-song fingering now that every note is known
    roll.dirty = true;
    enterReview();
    renderPanel();
    banner(t('msg.refined'), 'info');
    setTimeout(() => { if ($('bannerText').textContent === t('msg.refined')) banner(''); }, 4000);
  }
}

function cancelRefine() {
  const r = session.refine;
  if (!r) return;
  for (const p of r.parts) session.ignoreParts.add(p);
  session.refine = null;
  worker.postMessage({ type: 'abort', parts: [...r.parts] }); // only the refine's jobs
  banner('');
}
const player = new Player();
/** Mixer logic: with any solo, only soloed channels sound and show; else everything not muted. */
const audible = (inst) => (session.solo.size ? session.solo.has(inst) : !session.muted.has(inst));
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
  if (msg.part < session.minPart || session.ignoreParts.has(msg.part)) return; // cleared / cancelled
  if (session.ignoreParts.size > 64) for (const p of session.ignoreParts) if (p < session.minPart) session.ignoreParts.delete(p);
  session.version++;
  if (session.refine?.parts.has(msg.part)) return onRefineEvents(msg);
  if (msg.backfill) {
    session.backfilling = Math.max(0, session.backfilling - 1);
    if (!session.backfilling) banner('');
  }
  const prefix = msg.backfill ? `${msg.part}b${msg.backfill}` : `${msg.part}`;
  if (!isExtension && msg.seek != null) (session.statsLog ||= []).push([msg.seek, msg.events.filter((e) => e.type === 'start').length, msg.stats ? Math.round(msg.stats.genMs) : msg.dropped ? 'drop' : 'skip', msg.backlog]);
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
    // Now every chunk of this session is known: refine the whole recording, or at least
    // fill in the chunks that were skipped live.
    if (!startRefine()) backfill(msg.part);
  }
  roll.dirty = true;
  if (reviewing()) enterReview();
  else analyse();
  renderPanel();
}

/** Chords and key: the last 30 s while live, the whole session for review. */
let analysed = null; // {key} of the per-instrument chords/key last computed

function analyse(full = false) {
  const end = session.done;
  const from = full ? 0 : Math.max(0, Math.floor((end - CHORD_HISTORY_SEC) / 0.25) * 0.25);
  const byInst = new Map();
  for (const n of session.notes.values()) {
    if ((n.end ?? end) < from) continue;
    if (!byInst.has(n.instrument)) byInst.set(n.instrument, []);
    byInst.get(n.instrument).push(n);
  }
  // Per-instrument chords and the key depend only on the notes, not on mute/solo:
  // recompute them only when the notes changed.
  const cacheKey = `${session.version}|${from}|${end}`;
  if (analysed !== cacheKey) {
    analysed = cacheKey;
    session.chords.clear();
    for (const [inst, notes] of byInst) {
      if (inst !== 'drums' && !LINE_INSTRUMENTS.has(inst)) session.chords.set(inst, chordSegments(notes, from, end, end));
    }
    const all = [...byInst].filter(([i]) => i !== 'drums').flatMap(([, l]) => l);
    session.key = estimateKey(all, full ? 0 : Math.max(0, end - 45), end, end) ?? session.key;
  }
  // The big chord follows what you hear.
  const heard = [...byInst].filter(([i]) => i !== 'drums' && audible(i)).flatMap(([, l]) => l);
  session.ensemble = chordSegments(heard, from, end, end);
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
  const insts = orderedInstruments();
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
  $('clearBtn').disabled = !hasNotes;
  $('liveBtn').hidden = reviewing() || session.viewEnd == null;
}

const rowCache = new Map(); // instrument -> <li>

/** Channel order: the user's drag-and-drop order, then new parts by first appearance, drums last. */
function orderedInstruments() {
  const first = (i) => session.byInst.get(i)[0]?.start ?? Infinity;
  const auto = [...session.byInst.keys()]
    .sort((a, b) => (a === 'drums') - (b === 'drums') || first(a) - first(b));
  const placed = session.order.filter((i) => session.byInst.has(i));
  return [...placed, ...auto.filter((i) => !placed.includes(i))];
}

function moveChannel(inst, before) {
  const order = orderedInstruments().filter((i) => i !== inst);
  const at = before ? order.indexOf(before) : order.length;
  order.splice(at < 0 ? order.length : at, 0, inst);
  session.order = order;
  roll.dirty = true;
  renderPanel();
}

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
    li.innerHTML = `<span class="dot" style="background:${instrumentColor(inst)}"></span>
      <span class="ms">
        <button type="button" class="mute" data-i18n="insts.mute" data-i18n-attr="title:insts.muteTip,aria-label:insts.muteTip"></button>
        <button type="button" class="solo" data-i18n="insts.solo" data-i18n-attr="title:insts.soloTip,aria-label:insts.soloTip"></button>
      </span>
      <span class="name"></span>
      <span class="current"></span><span class="trail"></span>`;
    applyI18n(li);
    const toggle = (set) => {
      if (set.has(inst)) set.delete(inst); else set.add(inst);
      mixChanged();
    };
    // Any number of channels can be muted or soloed at once.
    // Drag a strip to reorder channels (and the tab staffs with them).
    li.draggable = true;
    li.addEventListener('dragstart', (e) => {
      e.dataTransfer.setData('text/x-byear-channel', inst);
      e.dataTransfer.effectAllowed = 'move';
      li.classList.add('dragging');
    });
    li.addEventListener('dragend', () => {
      li.classList.remove('dragging');
      for (const el of document.querySelectorAll('.inst.dropBefore, .inst.dropAfter')) el.classList.remove('dropBefore', 'dropAfter');
    });
    li.addEventListener('dragover', (e) => {
      if (!e.dataTransfer.types.includes('text/x-byear-channel')) return;
      e.preventDefault();
      const r = li.getBoundingClientRect();
      const after = (e.clientX - r.left > r.width / 2) !== (document.dir === 'rtl');
      li.classList.toggle('dropAfter', after);
      li.classList.toggle('dropBefore', !after);
    });
    li.addEventListener('dragleave', (e) => { if (!li.contains(e.relatedTarget)) li.classList.remove('dropBefore', 'dropAfter'); });
    li.addEventListener('drop', (e) => {
      const from = e.dataTransfer.getData('text/x-byear-channel');
      const after = li.classList.contains('dropAfter');
      li.classList.remove('dropBefore', 'dropAfter');
      if (!from || from === inst) return;
      e.preventDefault();
      const order = orderedInstruments().filter((i) => i !== from);
      moveChannel(from, after ? order[order.indexOf(inst) + 1] : inst);
    });
    // Keyboard: Alt + arrow on a strip's buttons moves the channel.
    li.addEventListener('keydown', (e) => {
      if (!e.altKey || (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight')) return;
      e.preventDefault();
      const order = orderedInstruments();
      const i = order.indexOf(inst);
      const dir = (e.key === 'ArrowRight') !== (document.dir === 'rtl') ? 1 : -1;
      const j = Math.max(0, Math.min(order.length - 1, i + dir));
      if (j === i) return;
      moveChannel(inst, dir > 0 ? order[j + 1] : order[j]);
      li.querySelector('.solo').focus();
    });
    li.querySelector('.mute').addEventListener('click', () => toggle(session.muted));
    li.querySelector('.solo').addEventListener('click', () => toggle(session.solo));
    rowCache.set(inst, li);
  }
  const muted = session.muted.has(inst);
  const soloed = session.solo.has(inst);
  li.classList.toggle('off', !audible(inst));
  li.querySelector('.mute').setAttribute('aria-pressed', String(muted));
  li.querySelector('.solo').setAttribute('aria-pressed', String(soloed));
  const all = session.byInst.get(inst);
  const notes = upTo(all, T);
  const segs = (session.chords.get(inst) || []).filter((sg) => sg.start <= T);
  let current = '';
  let trail = '';
  if (inst === 'drums') {
    const hits = new Map();
    for (const n of notes) if (n.start >= T - CHUNK_SEC) hits.set(drumName(n.pitch), (hits.get(drumName(n.pitch)) || 0) + 1);
    trail = [...hits].sort((a, b) => b[1] - a[1]).slice(0, 2).map(([name, c]) => `<span>${esc(name)} ${c}</span>`).join('');
  } else if (segs.length) {
    const cur = currentSegment(segs, T);
    current = cur ? chordHTML(cur.chord, flats) : '';
    trail = segs.slice(-4).map((s) => chordHTML(s.chord, flats)).join('');
  } else {
    const last = notes.slice(-8);
    const lastNote = last[last.length - 1];
    if (lastNote && lastNote.start >= T - 2) current = `<span class="chord">${esc(noteName(lastNote.pitch, flats).replace(/-?\d+$/, ''))}</span>`;
    trail = `<span class="notes">${esc(last.map((n) => noteName(n.pitch, flats)).join('  '))}</span>`;
  }
  // The strip is narrow: the name truncates, the full name and note count are on hover.
  setText(li.querySelector('.name'), instrumentLabel(inst));
  li.title = `${instrumentLabel(inst)}: ${all.length}`;
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
      audible,
      end: reviewing() ? null : session.viewEnd,
      listening: !!capture && !capture.waiting,
      instruments: orderedInstruments(),
      byInst: session.byInst,
      label: instrumentLabel,
      chunk: CHUNK_SEC,
      selected: session.selected,
      chords: session.ensemble,
      flats: session.key?.useFlats ?? false,
      playhead: review ? t : null,
      center: review ? t : null,
    });
    if (pointer) updateTip();
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
/** `begin` (optional) starts the sound once the whole graph is listening, so nothing is missed. */
async function startCapture(makeSource, label, cleanup, begin) {
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
  cancelRefine(); // listening again: the previous recording keeps its current notes
  const part = ++session.part;
  session.offset = Math.max(session.done, session.now);
  worker.postMessage({ type: 'part', part });
  worker.postMessage({ type: 'hint', part, names: null }); // no stale hint from an earlier refine
  const c = {
    part, ctx, rate, label, cleanup, captured: 0, chunk: new Float32Array(chunkLen), fill: 0, index: 0,
    stopped: false, pending: Promise.resolve(), level: 0, levelShown: 0, quietFor: 0,
    waiting: true, // nothing is recorded or timed until the first sound: no empty beginning
    preroll: [], // the last PREROLL_SEC of quiet audio while waiting
  };
  let lastHint = null;
  const send = async (samples, seek, final) => {
    const hint = settings.vocals ? ['voice'] : null; // live: main pass unhinted, plus the voice pass
    if (JSON.stringify(hint) !== JSON.stringify(lastHint)) {
      lastHint = hint;
      worker.postMessage({ type: 'hint', part, names: hint });
    }
    let energy = 0;
    for (let i = 0; i < samples.length; i += 4) energy += samples[i] * samples[i];
    if (Math.sqrt(energy / (samples.length / 4)) < 1e-4) {
      worker.postMessage({ type: 'skip', part, seek, next: seek + CHUNK_SEC, live: true });
    } else {
      const copy = new Float32Array(await resample16k(samples, rate));
      worker.postMessage({ type: 'chunk', part, samples: copy, seek, next: seek + CHUNK_SEC, live: true }, [copy.buffer]);
    }
    if (final) worker.postMessage({ type: 'finish', part });
  };
  // Chunks are handed over in order even though resampling is async.
  c.flush = (final) => {
    const seek = session.offset + c.index * CHUNK_SEC;
    c.index++;
    const samples = c.chunk;
    const filled = c.fill;
    c.chunk = new Float32Array(chunkLen);
    c.fill = 0;
    // The last chunk is recorded without its zero padding: review ends where the music did.
    record(final ? samples.subarray(0, filled) : samples, seek, rate);
    c.pending = c.pending.then(() => send(samples, seek, final));
    return c.pending;
  };
  const append = (data) => {
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
  tap.port.onmessage = ({ data }) => {
    if (c.stopped) return;
    let e = 0;
    for (let i = 0; i < data.length; i += 8) e += data[i] * data[i];
    c.level = Math.sqrt(e / (data.length / 8));
    // Auto pause on silence, auto resume when sound returns.
    c.quietFor = c.level < SILENCE_RMS ? c.quietFor + data.length / rate : 0;
    if (c.waiting) {
      if (c.level < SILENCE_RMS) {
        // Keep the last moment of "silence": it holds the start of the attack of the
        // note that wakes us up, which the model needs to hear that note at all.
        c.preroll.push(data);
        let held = c.preroll.reduce((a, b) => a + b.length, 0);
        while (held - c.preroll[0].length >= PREROLL_SEC * rate) held -= c.preroll.shift().length;
        return;
      }
      c.waiting = false;
      for (const block of c.preroll.splice(0)) append(block);
      renderPanel();
    } else if (c.quietFor >= SILENCE_SEC) {
      // Pause on silence. The chunk in progress is kept and keeps filling when the sound
      // returns, so no padded silence ever lands in the middle of the music. (The quiet we
      // waited through stays: splicing music edge to edge makes the model miss notes.)
      c.waiting = true;
      renderPanel();
      return;
    }
    append(data);
  };
  capture = c;
  try {
    if (begin) await begin();
  } catch (e) {
    await stopCapture(); // e.g. the file can't be played: don't stay "waiting" forever
    throw e;
  }
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
  else c.pending = c.pending.then(() => worker.postMessage({ type: 'finish', part: c.part }));
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
  player.load({ segments: session.audio, notes: [...session.notes.values()], duration, audible });
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
  await startCapture((ctx) => ctx.createMediaElementSource(audio), owned ? fileOrUrl.name : url.split('/').pop(), () => {
    audio.pause();
    if (owned) URL.revokeObjectURL(url);
  }, () => audio.play()); // play only after the tap is connected: the first note is captured too
  audio.addEventListener('ended', stopCapture);
}

// ---------------------------------------------------------------- controls

function showZoom(sec) {
  $('zoomLabel').textContent = t('zoom.seconds', { n: sec });
  $('zoomIn').disabled = sec <= ZOOM_STEPS[0];
  $('zoomOut').disabled = sec >= (roll.mode === 'tab' ? TAB_MAX_WINDOW : ZOOM_STEPS[ZOOM_STEPS.length - 1]);
}
roll.onZoom = (sec) => {
  roll.dirty = true;
  showZoom(sec);
  settingsStore.set({ zoom: sec });
};
$('zoomIn').addEventListener('click', () => roll.zoom(-1));

// Piano roll or tablature
function showView() {
  roll.mode = settings.view === 'tab' ? 'tab' : 'roll';
  roll.pitchScroll = 0;
  roll.tabScroll = 0;
  if (roll.mode === 'tab' && roll.window > TAB_MAX_WINDOW) roll.window = TAB_MAX_WINDOW;
  showZoom(roll.window);
  roll.dirty = true;
  for (const b of document.querySelectorAll('.viewSwitch button')) b.setAttribute('aria-checked', String(b.dataset.view === roll.mode));
}
for (const b of document.querySelectorAll('.viewSwitch button')) {
  b.addEventListener('click', async () => {
    await settingsStore.set({ view: b.dataset.view });
    showView();
  });
}
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
  // The file starts at the first note: no empty bars before the music.
  const first = Math.min(...[...session.notes.values()].map((n) => n.start));
  const notes = [...session.notes.values()].map((n) => ({ ...n, start: n.start - first, end: (n.end ?? session.done) - first }));
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
  session.muted.clear();
  session.solo.clear();
  session.order = [];
  session.selected = null;
  session.edits = [];
  session.version++;
  session.viewEnd = null;
  roll.tabScroll = 0;
  roll.pitchScroll = 0;
  session.chords.clear();
  session.ensemble = [];
  session.key = null;
  session.dropped = 0;
  session.backfilling = 0;
  cancelRefine();
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

// "vocals": tell the model the song has singing (MuScriptor's own conditioning input).
function showVocals() { $('vocalsBtn').setAttribute('aria-pressed', String(settings.vocals)); }
$('vocalsBtn').addEventListener('click', async () => {
  await settingsStore.set({ vocals: !settings.vocals });
  showVocals();
  // Applies to the next chunks while listening; in review, re-transcribe with the new hint.
  if (!capture && reviewing()) startRefine();
});

// Review transport
$('playBtn').addEventListener('click', async () => {
  if (player.playing) player.pause(); else await player.play();
  renderPanel();
});
player.onEnd = () => renderPanel();
// Playback speed: 100% -> 75% -> 50% (pitch stays the same for the original audio).
const SPEEDS = [1, 0.75, 0.5];
function showSpeed() { $('speedBtn').textContent = `${Math.round(player.speed * 100)}%`; }
$('speedBtn').addEventListener('click', async () => {
  await player.setSpeed(SPEEDS[(SPEEDS.indexOf(player.speed) + 1) % SPEEDS.length]);
  showSpeed();
  renderPanel();
});
showSpeed();
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
/** Mute/solo changed: chords, roll, playback and rows follow what you hear. */
function mixChanged() {
  roll.dirty = true;
  if (reviewing()) {
    enterReview();
    if (player.playing) player.seek(player.time); // reschedule with the new mix
  } else {
    analyse();
  }
  renderPanel();
}

// Scrolling over the roll pans it: live, it looks back in time; in review, it scrubs.
let wheelTimer = null;
let wheelResume = false;
roll.onPan = (d) => {
  if (reviewing()) {
    if (player.playing) { player.pause(); wheelResume = true; }
    player.pos = Math.max(0, Math.min(player.duration, player.pos + d));
    clearTimeout(wheelTimer);
    wheelTimer = setTimeout(async () => {
      if (wheelResume) await player.play();
      wheelResume = false;
      renderPanel();
    }, 300);
  } else {
    const edge = roll.liveEdge(Math.max(session.now, session.done), session.done);
    const next = (session.viewEnd ?? edge) + d;
    session.viewEnd = next >= edge - 0.05 ? null : Math.max(roll.window, next);
  }
  roll.dirty = true;
  renderPanel();
};
$('liveBtn').addEventListener('click', () => {
  session.viewEnd = null;
  roll.dirty = true;
  renderPanel();
});

// Hover a note: name, instrument, time. Re-checked every frame while the roll moves.
let pointer = null;
function updateTip() {
  const tip = $('noteTip');
  const n = pointer && !drag ? roll.noteAt(pointer.x, pointer.y) : null;
  if (!n) { tip.hidden = true; return; }
  const e = { offsetX: pointer.x, offsetY: pointer.y };
  const flats = session.key?.useFlats ?? false;
  const name = n.instrument === 'drums' ? drumName(n.pitch) : noteName(n.pitch, flats);
  tip.innerHTML = `<strong>${esc(name)}</strong><span>${esc(instrumentLabel(n.instrument))}, ${fmtTime(n.start)}</span><em>${esc(t(roll.mode === 'tab' ? 'note.hintTab' : 'note.hint'))}</em>`;
  tip.hidden = false;
  const r = $('roll').getBoundingClientRect();
  const left = Math.min(e.offsetX + 12, r.width - tip.offsetWidth - 4);
  const top = e.offsetY + 16 + tip.offsetHeight > r.height ? e.offsetY - tip.offsetHeight - 8 : e.offsetY + 16;
  tip.style.left = `${Math.max(4, left)}px`;
  tip.style.top = `${Math.max(4, top)}px`;
}
$('roll').addEventListener('pointermove', (e) => {
  pointer = { x: e.offsetX, y: e.offsetY };
  updateTip();
});
$('roll').addEventListener('pointerleave', () => {
  pointer = null;
  updateTip();
});

// Click the piano roll to move the playhead there; drag to scrub.
let drag = null;
$('roll').addEventListener('pointerdown', (e) => {
  const hit = roll.noteAt(e.offsetX, e.offsetY);
  if (!reviewing()) {
    selectNote(hit); // live: click a note to hear it
    return;
  }
  drag = { x: e.clientX, t: player.time, was: player.playing, moved: false, hit };
  try { $('roll').setPointerCapture(e.pointerId); } catch { /* synthetic pointer */ }
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
  const { moved, was, hit } = drag;
  drag = null;
  if (moved) {
    if (was) await player.play();
  } else if (hit) {
    selectNote(hit); // a note: hear it (and select it for editing)
  } else {
    selectNote(null);
    const r = $('roll').getBoundingClientRect();
    player.seek(roll.t0 + ((e.clientX - r.left) / r.width) * roll.window); // keeps playing if it was
  }
  renderPanel();
});
$('roll').addEventListener('pointercancel', () => { drag = null; });
/** Whether a key event comes from inside elements matching `selector` (events on document: no). */
const targetIn = (e, selector) => e.target instanceof Element && !!e.target.closest(selector);

// Selected note: Alt+↑/↓ semitone (with Shift an octave), Delete removes it, Esc deselects.
document.addEventListener('keydown', (e) => {
  if (!session.selected || $('live').hidden || $('settings').open) return;
  if (targetIn(e, 'input[type=text], input[type=password], textarea, select')) return;
  if (e.altKey && (e.key === 'ArrowUp' || e.key === 'ArrowDown')) {
    e.preventDefault();
    e.stopPropagation();
    transposeSelected((e.key === 'ArrowUp' ? 1 : -1) * (e.shiftKey ? 12 : 1));
  } else if (roll.mode === 'tab' && (e.key === 'ArrowUp' || e.key === 'ArrowDown') && !targetIn(e, 'input, [role=radio], select')) {
    // Tablature: same pitch, another string (up = higher string, as drawn).
    e.preventDefault();
    e.stopPropagation();
    moveSelectedString(e.key === 'ArrowUp' ? 1 : -1);
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    e.preventDefault();
    deleteSelected();
  } else if (e.key === 'Escape') {
    selectNote(null);
  }
}, true);

// Space is always play/stop, wherever focus is (except while typing).
document.addEventListener('keydown', (e) => {
  if (e.key !== ' ' || $('live').hidden || $('settings').open) return;
  if (targetIn(e, 'input[type=text], input[type=password], textarea, select')) return;
  e.preventDefault();
  e.stopPropagation();
  if (e.repeat) return;
  if (capture) stopCapture();
  else if (reviewing()) $('playBtn').click();
  else if (!$('recBtn').disabled) $('recBtn').click();
}, true);
document.addEventListener('keyup', (e) => { if (e.key === ' ' && !$('live').hidden) e.preventDefault(); }, true);
document.addEventListener('keydown', (e) => {
  if (!reviewing() || $('live').hidden || $('settings').open) return;
  if (targetIn(e, 'input, button, select, textarea, a, summary, label, dialog, [role=switch], canvas')) return;
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
  $('optRefine').checked = settings.refine;
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
    refine: $('optRefine').checked,
    instruments: [...chips.querySelectorAll('input:checked')].map((b) => b.value),
    lang: $('optLang').value,
    naming: $('optNaming').value,
  };
  const localeChanged = next.lang !== settings.lang || next.naming !== settings.naming;
  const reload = next.model !== settings.model || next.f32 !== settings.f32 || (next.token !== settings.token && !modelInfo);
  await settingsStore.set(next);
  if (localeChanged) await applyLocale();
  worker.postMessage({ type: 'instruments', names: settings.instruments });
  if (session.refine) startRefine(); // redo it with the new instruments throughout
  if (reload && settings.accepted) {
    if (capture) await stopCapture();
    loadModel();
  }
});

$('clearCache').addEventListener('click', () => worker.postMessage({ type: 'clearCache' }));

// ---------------------------------------------------------------- boot

// Test hook (dev page only, never in the extension): read the timeline state.
if (!isExtension) {
  window.__byearTime = () => player.time;
  window.__byearGaps = () => { const st = [...session.notes.values()].filter((n) => n.instrument !== 'drums').map((n) => n.start).sort((a, b) => a - b); let g = 0, at = 0; for (let i = 1; i < st.length; i++) if (st[i] - st[i - 1] > g) { g = st[i] - st[i - 1]; at = st[i - 1]; } return { longestGap: +g.toFixed(2), at: +at.toFixed(2), duration: player.duration }; };
  window.__byearSel = () => session.selected && { pitch: session.selected.pitch, instrument: session.selected.instrument };
  window.__byearState = () => ({ dropped: session.dropped, droppedSeeks: session.droppedSeeks.map((d) => d.seek), stats: session.statsLog, refine: !!session.refine, audio: session.audio.length, backfilling: session.backfilling, capture: !!capture, setting: settings.refine, reviewing: reviewing() });
  window.__byearPos = () => session.selected && roll.fingering.get(session.selected, session.selected.instrument);
  window.__byear = () => ({
    now: session.now, done: session.done, viewEnd: session.viewEnd, t0: roll.t0, mode: roll.mode, window: roll.window, notes: session.notes.size,
    first: [...session.notes.values()].sort((a, b) => a.start - b.start).slice(0, 6).map((n) => [+n.start.toFixed(2), n.pitch, n.instrument]),
  });
}

settings = await settingsStore.get();
roll.window = ZOOM_STEPS.includes(settings.zoom) ? settings.zoom : 12;
showZoom(roll.window);
await applyLocale();
showVocals();
showView();
if (params.get('weights')) loadModel({ url: new URL(params.get('weights'), location.href).href });
else if (settings.accepted) loadModel();
else showScreen('welcome');
