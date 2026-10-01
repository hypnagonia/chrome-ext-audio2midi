// Inference worker: owns the weights, the WebGPU engine and one ordered job queue.
//
// Protocol (panel -> worker), all queued in order except load/abort/instruments:
//   {type:'load', ...}                 serialized; a newer load supersedes an older one
//   {type:'part', part}                a new capture session starts (fresh decoder)
//   {type:'chunk'|'skip', part, seek, next, samples?}
//   {type:'finish', part}              end of a capture session
//   {type:'backfill', part, key, seek, next, samples}
//   {type:'hint', names}                conditioning-only instrument hint for the following chunks
//   {type:'abort'}                     drop everything still queued
//   {type:'instruments', names}        applied between jobs
// Every 'events' reply echoes `part` so the panel can tell sessions apart.

import { parseSafetensors } from './safetensors.js';
import { MelFrontend } from './mel.js';
import { Engine } from './gpu.js';
import { Transcriber } from './transcriber.js';

const CACHE = 'muscriptor-weights-v1';
// Token-free fp16 mirror first; the official gated repo needs a Hugging Face token.
const SOURCES = [
  { url: (size, file) => `https://huggingface.co/jenyasn/muscriptor-${size}-fp16/resolve/main/${file}`, token: false },
  { url: (size, file) => `https://huggingface.co/MuScriptor/muscriptor-${size}/resolve/main/${file}`, token: true },
];

// Errors carry a `code` the side panel translates; `message` is the English fallback.
class CodedError extends Error {
  constructor(message, code, extra = {}) {
    super(message);
    Object.assign(this, { code }, extra);
  }
}

let engine = null;
let transcriber = null;
const queue = [];
let pumping = null; // promise while the queue is being worked
let loadSeq = 0;
let loadChain = Promise.resolve();
let pendingInstruments; // applied between jobs so a chunk never mixes masks

const post = (msg, transfer) => self.postMessage(msg, transfer);

/** True when buf holds a complete safetensors file. */
function validSafetensors(buf) {
  try {
    const st = parseSafetensors(buf);
    for (const t of st.tensors.values()) if (t.offset + t.bytes > buf.byteLength) return false;
    return st.tensors.size > 0;
  } catch {
    return false;
  }
}

async function fetchWithProgress(url, token, phase, validate = null) {
  const cache = await caches.open(CACHE);
  const hit = await cache.match(url);
  if (hit) {
    post({ type: 'progress', phase: 'cache', frac: null });
    const buf = await hit.arrayBuffer();
    if (!validate || validate(buf)) return buf;
    await cache.delete(url); // corrupt cache entry: download again
  }
  const res = await fetch(url, { headers: token ? { Authorization: `Bearer ${token}` } : {} });
  if (res.status === 401 || res.status === 403) {
    throw res.status === 401
      ? new CodedError('Hugging Face rejected the token.', 'bad-token', { status: 401 })
      : new CodedError('The Hugging Face account has not accepted the MuScriptor license.', 'no-license', { status: 403 });
  }
  if (!res.ok) throw new CodedError(`Download failed (HTTP ${res.status}).`, 'download', { status: res.status });
  // content-length is the encoded size when the body is compressed: only trust it otherwise.
  const encoded = (res.headers.get('content-encoding') || 'identity') !== 'identity';
  const total = encoded ? 0 : Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const parts = [];
  let loaded = 0;
  let lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    const now = performance.now();
    if (now - lastPost > 150) {
      lastPost = now;
      post({ type: 'progress', phase, frac: total ? Math.min(1, loaded / total) : null, loaded, total });
    }
  }
  if (total && loaded !== total) throw new CodedError('Download incomplete.', 'download', { status: 'incomplete' });
  const blob = new Blob(parts);
  parts.length = 0;
  const buf = await blob.arrayBuffer();
  if (validate && !validate(buf)) throw new CodedError('Downloaded model is damaged.', 'download', { status: 'corrupt' });
  try {
    await cache.put(url, new Response(blob, { headers: { 'content-type': 'application/octet-stream' } }));
  } catch (e) {
    post({ type: 'warning', code: 'cache-failed', error: e.message, message: `Could not cache weights: ${e.message}` });
  }
  return buf;
}

function inferConfig(st, json) {
  if (json?.dim) return json;
  const dim = st.shape('emb.weight')[1];
  let layers = 0;
  while (st.has(`transformer.layers.${layers}.norm1.weight`)) layers++;
  return { dim, num_heads: dim / 64, num_layers: layers, card: st.shape('linear.weight')[0] };
}

async function download(model, token) {
  let lastError = null;
  for (const src of SOURCES) {
    if (src.token && !token) continue;
    const auth = src.token ? token : null;
    try {
      const json = JSON.parse(new TextDecoder().decode(await fetchWithProgress(src.url(model, 'config.json'), auth, 'config')));
      const buffer = await fetchWithProgress(src.url(model, 'model.safetensors'), auth, 'download', validSafetensors);
      return { json, buffer };
    } catch (e) {
      lastError = e;
    }
  }
  // The free mirror refused (not a network blip) and there is no token for the official repo.
  if (!token && [401, 403, 404].includes(lastError?.status)) {
    throw new CodedError('The free model download is not reachable.', 'needs-token');
  }
  throw lastError?.code ? lastError : new CodedError(lastError?.message || 'Download failed.', 'download', { status: 'network' });
}

async function load({ model, token, buffer, url, f16 = true, autoLevel = true }, seq) {
  const stale = () => seq !== loadSeq;
  queue.length = 0;
  await pumping;
  if (engine) {
    engine.destroy();
    engine = null;
    transcriber = null;
  }
  let json = null;
  if (!buffer && url) {
    post({ type: 'progress', phase: 'file', frac: null });
    const res = await fetch(url);
    if (!res.ok) throw new CodedError(`Could not load ${url} (${res.status})`, 'download', { status: res.status });
    buffer = await res.arrayBuffer();
  }
  if (!buffer) ({ json, buffer } = await download(model, token));
  if (stale()) return;
  if (!validSafetensors(buffer)) throw new CodedError('Not a valid model file.', 'download', { status: 'corrupt' });
  post({ type: 'progress', phase: 'gpu', frac: 0 });
  const st = parseSafetensors(buffer);
  const created = await Engine.create(st, inferConfig(st, json), {
    f16,
    onProgress: (frac) => post({ type: 'progress', phase: 'gpu', frac }),
  });
  if (stale()) {
    created.destroy();
    return;
  }
  engine = created;
  engine.device.lost.then((info) => {
    if (engine === created && info.reason !== 'destroyed') post({ type: 'error', code: 'gpu-lost', message: info.message });
  });
  const P = 'condition_provider.conditioners.self_wav.mel_spec_transform.';
  const mel = new MelFrontend(st.getF32(P + 'spectrogram.window').slice(), st.getF32(P + 'mel_scale.fb').slice());
  transcriber = new Transcriber(engine, mel, { autoLevel });
  transcriber.setInstruments(pendingInstruments ?? null);
  pendingInstruments = undefined;
  post({ type: 'ready', info: { gpu: `${engine.adapterInfo.vendor} ${engine.adapterInfo.architecture}`.trim(), f16 } });
}

/**
 * Only live chunks may be skipped (to keep up while listening), and only when a newer
 * live chunk of the same session is already waiting. Refine and backfill chunks never are.
 */
const superseded = (job) => job.type === 'chunk' && job.live
  && queue.some((j) => (j.type === 'chunk' || j.type === 'skip') && j.live && j.part === job.part);

async function runJob(job) {
  if (pendingInstruments !== undefined) {
    transcriber.setInstruments(pendingInstruments);
    pendingInstruments = undefined;
  }
  switch (job.type) {
    case 'part':
      transcriber.reset();
      return;
    case 'hint':
      transcriber.setHint(job.names);
      return;
    case 'finish':
      post({ type: 'events', part: job.part, seek: null, events: transcriber.finish(), final: true });
      transcriber.reset();
      return;
    case 'backfill': {
      const events = await transcriber.isolated(job.samples, job.seek, job.next);
      post({ type: 'events', part: job.part, seek: job.seek, events, backfill: job.key });
      return;
    }
    default: {
      // Falling behind: skip this chunk (its notes end at the chunk start) and catch up.
      const dropped = superseded(job);
      const r = await transcriber.processChunk(job.type === 'chunk' && !dropped ? job.samples : null, job.seek, job.next);
      post({ type: 'events', part: job.part, seek: job.seek, events: r.events, stats: r.stats, dropped, backlog: queue.length });
    }
  }
}

function pump() {
  if (pumping) return;
  pumping = (async () => {
    while (queue.length && transcriber) {
      const job = queue.shift();
      try {
        await runJob(job);
      } catch (e) {
        post({ type: 'error', message: e.message || String(e), code: e.code ?? null });
        // Keep the panel's bookkeeping consistent even when a job fails.
        if (job.type === 'backfill') post({ type: 'events', part: job.part, seek: null, events: [], backfill: job.key });
        if (job.type === 'chunk' || job.type === 'skip') post({ type: 'events', part: job.part, seek: job.seek, events: [] });
      }
    }
  })().finally(() => { pumping = null; });
}

self.onmessage = ({ data }) => {
  switch (data.type) {
    case 'load': {
      const seq = ++loadSeq;
      loadChain = loadChain.then(() => load(data, seq)).catch((e) => {
        if (seq === loadSeq) post({ type: 'loadError', message: e.message || String(e), code: e.code ?? null, status: e.status ?? null });
      });
      break;
    }
    case 'abort':
      queue.length = 0;
      break;
    case 'instruments':
      if (transcriber && !pumping) transcriber.setInstruments(data.names);
      else pendingInstruments = data.names;
      break;
    case 'part':
    case 'hint':
    case 'chunk':
    case 'skip':
    case 'finish':
    case 'backfill':
      queue.push(data);
      if (transcriber) pump();
      break;
    case 'clearCache':
      caches.delete(CACHE).then(() => post({ type: 'cacheCleared' }));
      break;
  }
};
