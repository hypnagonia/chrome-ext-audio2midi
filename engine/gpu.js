// WebGPU implementation of the MuScriptor decoder-only transformer
// (muscriptor/models/lm.py + modules/transformer.py).
//
// Weights live on the GPU as f16 (default) or f32; activations are f32.
// Generation keeps the sampled token, position and history on the GPU so
// several decode steps can be queued per submit without a CPU round trip.

import { VOCAB_LIMIT, EOS_ID } from './vocab.js';

const HEAD_DIM = 64;
const MAX_SEQ = 3072; // KV cache length: prefix (<=540) + 2000 generated tokens
export const MAX_PREFILL = 1024; // rows in one prefill pass
const STEPS_PER_SUBMIT = 16;
// Decode attention splits each head's keys over this many workgroups (flash-decoding).
const ATTN_SPLITS = 16;
const PART_STRIDE = 68; // per (head, split): max, sum, 64 accumulators, padding

const COMMON = `
struct P { M: u32, N: u32, K: u32, flags: u32, a: u32, b: u32, c: u32, d: u32 }
struct S { pos: u32, step: u32, token: u32, done: u32 }
fn erf_(x: f32) -> f32 {
  let z = abs(x);
  let t = 1.0 / (1.0 + 0.5 * z);
  let r = t * exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
    t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
    t * (-0.82215223 + t * 0.17087277)))))))));
  return select(r - 1.0, 1.0 - r, x >= 0.0);
}
fn gelu(x: f32) -> f32 { return 0.5 * x * (1.0 + erf_(x * 0.70710678118654752)); }
`;

const EPILOGUE = `
fn epi(r: u32, n: u32, v0: f32) {
  var v = v0;
  if ((p.flags & 1u) != 0u) { v += B[n]; }
  if ((p.flags & 2u) != 0u) { v = gelu(v); }
  let idx = r * p.N + n;
  if ((p.flags & 4u) != 0u) { Y[idx] = Y[idx] + v; } else { Y[idx] = v; }
}
`;

// Y[M,N] (+)= act(X[M,K] . W[N,K]^T + b), tiled 64x64 with 4x4 per thread.
const matmulWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
${EPILOGUE}
fn loadW(i: u32) -> f32 { ${f16 ? 'return unpack2x16float(W[i >> 1u])[i & 1u];' : 'return W[i];'} }
var<workgroup> xs: array<f32, 1088>;
var<workgroup> ws: array<f32, 1088>;
@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.y * 16u + lid.x;
  let row0 = wg.y * 64u;
  let col0 = wg.x * 64u;
  var acc: array<f32, 16>;
  for (var k0 = 0u; k0 < p.K; k0 += 16u) {
    for (var i = 0u; i < 4u; i++) {
      let e = tid + i * 256u;
      let r = e / 16u;
      let c = e % 16u;
      let gk = k0 + c;
      let gr = row0 + r;
      var xv = 0.0;
      if (gr < p.M && gk < p.K) { xv = X[gr * p.K + gk]; }
      xs[r * 17u + c] = xv;
      let gn = col0 + r;
      var wv = 0.0;
      if (gn < p.N && gk < p.K) { wv = loadW(gn * p.K + gk); }
      ws[r * 17u + c] = wv;
    }
    workgroupBarrier();
    for (var kk = 0u; kk < 16u; kk++) {
      var a: array<f32, 4>;
      var b: array<f32, 4>;
      for (var i = 0u; i < 4u; i++) {
        a[i] = xs[(lid.y + i * 16u) * 17u + kk];
        b[i] = ws[(lid.x + i * 16u) * 17u + kk];
      }
      for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) { acc[i * 4u + j] = fma(a[i], b[j], acc[i * 4u + j]); }
      }
    }
    workgroupBarrier();
  }
  for (var i = 0u; i < 4u; i++) {
    for (var j = 0u; j < 4u; j++) {
      let r = row0 + lid.y + i * 16u;
      let n = col0 + lid.x + j * 16u;
      if (r < p.M && n < p.N) { epi(r, n, acc[i * 4u + j]); }
    }
  }
}`;

// Single-row Y[N] (+)= act(W[N,K] . x + b): 8 rows per workgroup, 32 lanes per row.
const matvecWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<vec4f>;
@group(0) @binding(2) var<storage, read> W: array<${f16 ? 'vec4u' : 'vec4f'}>;
@group(0) @binding(3) var<storage, read> B: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
@group(0) @binding(5) var<storage, read> st: S;
${EPILOGUE}
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  if (st.done != 0u) { return; } // speculative steps after EOS: skip the weight reads
  let r = lid.x / 32u;
  let l = lid.x % 32u;
  let n = wg.x * 8u + r;
  var s = 0.0;
  if (n < p.N) {
${f16 ? `    let K8 = p.K / 8u;
    for (var i = l; i < K8; i += 32u) {
      let w = W[n * K8 + i];
      s += dot(vec4f(unpack2x16float(w.x), unpack2x16float(w.y)), X[2u * i]);
      s += dot(vec4f(unpack2x16float(w.z), unpack2x16float(w.w)), X[2u * i + 1u]);
    }` : `    let K4 = p.K / 4u;
    for (var i = l; i < K4; i += 32u) { s += dot(W[n * K4 + i], X[i]); }`}
  }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = 16u; st > 0u; st >>= 1u) {
    if (l < st) { red[lid.x] += red[lid.x + st]; }
    workgroupBarrier();
  }
  if (l == 0u && n < p.N) { epi(0u, n, red[lid.x]); }
}`;

// Row-wise LayerNorm: Y[row] = LN(X[row + a]).
const layernormWGSL = `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> G: array<f32>;
@group(0) @binding(3) var<storage, read> Bt: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<f32>;
var<workgroup> red: array<f32, 256>;
fn reduce(t: u32) -> f32 {
  for (var st = 128u; st > 0u; st >>= 1u) {
    if (t < st) { red[t] += red[t + st]; }
    workgroupBarrier();
  }
  let v = red[0];
  workgroupBarrier();
  return v;
}
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let D = p.N;
  let src = (wg.x + p.a) * D;
  let dst = wg.x * D;
  var s = 0.0;
  for (var i = lid.x; i < D; i += 256u) { s += X[src + i]; }
  red[lid.x] = s;
  workgroupBarrier();
  let mean = reduce(lid.x) / f32(D);
  var v = 0.0;
  for (var i = lid.x; i < D; i += 256u) { let d = X[src + i] - mean; v += d * d; }
  red[lid.x] = v;
  workgroupBarrier();
  let rstd = inverseSqrt(reduce(lid.x) / f32(D) + 1e-5);
  for (var i = lid.x; i < D; i += 256u) { Y[dst + i] = (X[src + i] - mean) * rstd * G[i] + Bt[i]; }
}`;

// X[i] += sinusoidal position embedding of (pos + row).
const addposWGSL = `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: S;
@group(0) @binding(2) var<storage, read> PE: array<f32>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i < p.M * p.N) { X[i] += PE[st.pos * p.N + i]; }
}`;

// Append this pass's K and V rows (from packed qkv) to the cache at pos.
const kvwriteWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: S;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read_write> Kc: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(4) var<storage, read_write> Vc: array<${f16 ? 'u32' : 'f32'}>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let D = p.N;
${f16 ? `  let H2 = D / 2u;
  let c = gid.x;
  if (c >= p.M * H2) { return; }
  let i = c / H2;
  let j = c % H2;
  let src = i * 3u * D + 2u * j;
  let dst = (st.pos + i) * H2 + j;
  Kc[dst] = pack2x16float(vec2f(QKV[src + D], QKV[src + D + 1u]));
  Vc[dst] = pack2x16float(vec2f(QKV[src + 2u * D], QKV[src + 2u * D + 1u]));` : `  let c = gid.x;
  if (c >= p.M * D) { return; }
  let i = c / D;
  let j = c % D;
  let src = i * 3u * D + j;
  let dst = (st.pos + i) * D + j;
  Kc[dst] = QKV[src + D];
  Vc[dst] = QKV[src + 2u * D];`}
}`;

// Causal attention, one workgroup per (query row, head); online softmax over
// blocks of 64 keys. Query row i sits at absolute position pos + i.
const attentionWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: S;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> Kc: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(4) var<storage, read> Vc: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(5) var<storage, read_write> O: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> ss: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let D = p.N;
  let i = wg.x;
  let h = wg.y;
  let t = lid.x;
  let nkeys = st.pos + i + 1u;
  qs[t] = QKV[i * 3u * D + h * 64u + t] * 0.125;
  workgroupBarrier();
  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  for (var b0 = 0u; b0 < nkeys; b0 += 64u) {
    let j = b0 + t;
    var s = -1e30;
    if (j < nkeys) {
      s = 0.0;
${f16 ? `      let base = (j * D + h * 64u) / 2u;
      for (var d = 0u; d < 32u; d++) {
        let kv = unpack2x16float(Kc[base + d]);
        s += qs[2u * d] * kv.x + qs[2u * d + 1u] * kv.y;
      }` : `      let base = j * D + h * 64u;
      for (var d = 0u; d < 64u; d++) { s += qs[d] * Kc[base + d]; }`}
    }
    ss[t] = s;
    workgroupBarrier();
    var bm = -1e30;
    for (var u = 0u; u < 64u; u++) { bm = max(bm, ss[u]); }
    let nm = max(m, bm);
    let corr = exp(m - nm);
    acc *= corr;
    l *= corr;
    workgroupBarrier();
    ss[t] = exp(s - nm); // each key's weight computed once, by its own thread
    workgroupBarrier();
    let cnt = min(64u, nkeys - b0);
    for (var u = 0u; u < cnt; u++) {
      let pe = ss[u];
      l += pe;
      let vi = (b0 + u) * D + h * 64u + t;
${f16 ? '      acc += pe * unpack2x16float(Vc[vi >> 1u])[vi & 1u];' : '      acc += pe * Vc[vi];'}
    }
    m = nm;
    workgroupBarrier();
  }
  O[i * D + h * 64u + t] = acc / l;
}`;

// Decode-only attention (one query row at pos): split the keys of each head
// over a = ATTN_SPLITS workgroups; each writes a partial (max, sum, acc[64]).
const attentionSplitWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: S;
@group(0) @binding(2) var<storage, read> QKV: array<f32>;
@group(0) @binding(3) var<storage, read> Kc: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(4) var<storage, read> Vc: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(5) var<storage, read_write> Part: array<f32>;
var<workgroup> qs: array<f32, 64>;
var<workgroup> ss: array<f32, 64>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let D = p.N;
  let split = wg.x;
  let h = wg.y;
  let t = lid.x;
  if (st.done != 0u) { return; }
  let nkeys = st.pos + 1u;
  let per = (nkeys + p.a - 1u) / p.a;
  let start = split * per;
  let end = min(nkeys, start + per);
  qs[t] = QKV[h * 64u + t] * 0.125;
  workgroupBarrier();
  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  for (var b0 = start; b0 < end; b0 += 64u) {
    let j = b0 + t;
    var s = -1e30;
    if (j < end) {
      s = 0.0;
${f16 ? `      let base = (j * D + h * 64u) / 2u;
      for (var d = 0u; d < 32u; d++) {
        let kv = unpack2x16float(Kc[base + d]);
        s += qs[2u * d] * kv.x + qs[2u * d + 1u] * kv.y;
      }` : `      let base = j * D + h * 64u;
      for (var d = 0u; d < 64u; d++) { s += qs[d] * Kc[base + d]; }`}
    }
    ss[t] = s;
    workgroupBarrier();
    var bm = -1e30;
    for (var u = 0u; u < 64u; u++) { bm = max(bm, ss[u]); }
    let nm = max(m, bm);
    let corr = exp(m - nm);
    acc *= corr;
    l *= corr;
    workgroupBarrier();
    ss[t] = exp(s - nm);
    workgroupBarrier();
    let cnt = min(64u, end - b0);
    for (var u = 0u; u < cnt; u++) {
      let pe = ss[u];
      l += pe;
      let vi = (b0 + u) * D + h * 64u + t;
${f16 ? '      acc += pe * unpack2x16float(Vc[vi >> 1u])[vi & 1u];' : '      acc += pe * Vc[vi];'}
    }
    m = nm;
    workgroupBarrier();
  }
  let o = (h * p.a + split) * ${PART_STRIDE}u;
  if (t == 0u) { Part[o] = m; Part[o + 1u] = l; }
  Part[o + 2u + t] = acc;
}`;

// Merge the per-split partials of each head into the attention output.
const attentionCombineWGSL = `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> Part: array<f32>;
@group(0) @binding(2) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
  let h = wg.x;
  let t = lid.x;
  var mx = -1e30;
  for (var s = 0u; s < p.a; s++) { mx = max(mx, Part[(h * p.a + s) * ${PART_STRIDE}u]); }
  var l = 0.0;
  var acc = 0.0;
  for (var s = 0u; s < p.a; s++) {
    let o = (h * p.a + s) * ${PART_STRIDE}u;
    let w = exp(Part[o] - mx);
    l += Part[o + 1u] * w;
    acc += Part[o + 2u + t] * w;
  }
  O[h * 64u + t] = acc / l;
}`;

// Greedy pick over logits[< a] not forbidden; advance pos by M and log the token.
const argmaxWGSL = `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> L: array<f32>;
@group(0) @binding(2) var<storage, read> F: array<u32>;
@group(0) @binding(3) var<storage, read_write> st: S;
@group(0) @binding(4) var<storage, read_write> H: array<u32>;
var<workgroup> bv: array<f32, 256>;
var<workgroup> bi: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3u) {
  let t = lid.x;
  var best = -3.4e38;
  var idx = 0xffffffffu;
  for (var i = t; i < p.N; i += 256u) {
    if (i < p.a && F[i] == 0u) {
      let v = L[i];
      if (v > best) { best = v; idx = i; }
    }
  }
  bv[t] = best;
  bi[t] = idx;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s >>= 1u) {
    if (t < s) {
      let ov = bv[t + s];
      let oi = bi[t + s];
      if (ov > bv[t] || (ov == bv[t] && oi < bi[t])) { bv[t] = ov; bi[t] = oi; }
    }
    workgroupBarrier();
  }
  if (t == 0u) {
    if (st.done != 0u) {
      H[st.step] = ${EOS_ID}u; // after EOS: only fill the history, state stays put
      st.step += 1u;
    } else {
      var tok = bi[0];
      if (tok == 0xffffffffu) { tok = ${EOS_ID}u; } // all logits NaN/masked: end the chunk
      H[st.step] = tok;
      st.token = tok;
      st.step += 1u;
      st.pos += p.M;
      if (tok == ${EOS_ID}u) { st.done = 1u; }
    }
  }
}`;

// X[0] = emb[token]
const embedWGSL = (f16) => `${COMMON}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> st: S;
@group(0) @binding(2) var<storage, read> E: array<${f16 ? 'u32' : 'f32'}>;
@group(0) @binding(3) var<storage, read_write> X: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let d = gid.x;
  if (d >= p.N) { return; }
  let i = st.token * p.N + d;
${f16 ? '  X[d] = unpack2x16float(E[i >> 1u])[i & 1u];' : '  X[d] = E[i];'}
}`;

function toF16(f32) {
  if (typeof Float16Array !== 'undefined') return new Uint16Array(new Float16Array(f32).buffer);
  const out = new Uint16Array(f32.length);
  const fv = new Float32Array(1);
  const iv = new Uint32Array(fv.buffer);
  for (let i = 0; i < f32.length; i++) {
    fv[0] = f32[i];
    const x = iv[0];
    const sign = (x >>> 16) & 0x8000;
    let exp = ((x >>> 23) & 0xff) - 127 + 15;
    let mant = x & 0x7fffff;
    if (exp >= 31) { out[i] = sign | 0x7c00; continue; }
    if (exp <= 0) {
      if (exp < -10) { out[i] = sign; continue; }
      mant |= 0x800000;
      const shift = 14 - exp;
      let h = mant >> shift;
      const rem = mant & ((1 << shift) - 1), halfway = 1 << (shift - 1);
      if (rem > halfway || (rem === halfway && (h & 1))) h++;
      out[i] = sign | h;
      continue;
    }
    let h = (exp << 10) | (mant >> 13);
    const rem = mant & 0x1fff;
    if (rem > 0x1000 || (rem === 0x1000 && (h & 1))) h++;
    out[i] = sign | h;
  }
  return out;
}

export class Engine {
  /**
   * @param st parsed safetensors (see safetensors.js)
   * @param cfg {dim, num_heads, num_layers, card}
   */
  static async create(st, cfg, { f16 = true, onProgress = () => {} } = {}) {
    if (!navigator.gpu) throw Object.assign(new Error('WebGPU is not available in this browser'), { code: 'no-webgpu' });
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw Object.assign(new Error('No WebGPU adapter found'), { code: 'no-webgpu' });
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
      },
    });
    const e = new Engine(device, cfg, f16, adapter.info);
    await e._upload(st, onProgress);
    e._buildDecode();
    return e;
  }

  constructor(device, cfg, f16, adapterInfo) {
    this.device = device;
    this.f16 = f16;
    this.adapterInfo = adapterInfo;
    this.D = cfg.dim;
    this.H = cfg.num_heads;
    this.L = cfg.num_layers;
    this.card = cfg.card;
    if (this.D / this.H !== HEAD_DIM) throw new Error(`head dim ${this.D / this.H} unsupported`);
    const mk = (code) => device.createComputePipeline({
      layout: 'auto',
      compute: { module: device.createShaderModule({ code }), entryPoint: 'main' },
    });
    this.pl = {
      matmul: mk(matmulWGSL(f16)),
      matvec: mk(matvecWGSL(f16)),
      layernorm: mk(layernormWGSL),
      addpos: mk(addposWGSL),
      kvwrite: mk(kvwriteWGSL(f16)),
      attention: mk(attentionWGSL(f16)),
      attentionSplit: mk(attentionSplitWGSL(f16)),
      attentionCombine: mk(attentionCombineWGSL),
      argmax: mk(argmaxWGSL),
      embed: mk(embedWGSL(f16)),
    };
    device.lost.then((info) => { this.lost = info; });
  }

  _buf(size, usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage });
  }

  _upload32(data) {
    const b = this._buf(data.byteLength);
    this.device.queue.writeBuffer(b, 0, data.buffer, data.byteOffset, data.byteLength);
    return b;
  }

  /** Upload a weight matrix by name in the engine's weight precision. */
  _uploadW(st, name) {
    if (this.f16) {
      const bits = st.getF16Bits(name);
      return this._upload32(bits ?? toF16(st.getF32(name)));
    }
    return this._upload32(st.getF32(name));
  }

  async _upload(st, onProgress) {
    const { D, L } = this;
    const P = 'condition_provider.conditioners.';
    const total = L + 2;
    this.layers = [];
    for (let i = 0; i < L; i++) {
      const p = `transformer.layers.${i}.`;
      this.layers.push({
        inProj: this._uploadW(st, p + 'self_attn.in_proj_weight'),
        outProj: this._uploadW(st, p + 'self_attn.out_proj.weight'),
        l1: this._uploadW(st, p + 'linear1.weight'),
        l2: this._uploadW(st, p + 'linear2.weight'),
        n1g: this._upload32(st.getF32(p + 'norm1.weight')),
        n1b: this._upload32(st.getF32(p + 'norm1.bias')),
        n2g: this._upload32(st.getF32(p + 'norm2.weight')),
        n2b: this._upload32(st.getF32(p + 'norm2.bias')),
      });
      onProgress((i + 1) / total);
      // Let the queue drain so the staging memory does not pile up.
      if (i % 4 === 3) await this.device.queue.onSubmittedWorkDone();
    }
    this.embCPU = st.getF32('emb.weight').slice();
    this.emb = this._uploadW(st, 'emb.weight');
    this.head = this._uploadW(st, 'linear.weight');
    this.outG = this._upload32(st.getF32('out_norm.weight'));
    this.outB = this._upload32(st.getF32('out_norm.bias'));
    this.melW = this._uploadW(st, P + 'self_wav.output_proj.weight');
    this.melB = this._upload32(st.getF32(P + 'self_wav.output_proj.bias'));
    this.instEmb = st.getF32(P + 'instrument_group.embed.weight').slice();
    this.dsEmb = st.getF32(P + 'dataset_name.embed.weight').slice();
    onProgress(1);

    // Sinusoidal positions, mirroring create_sin_embedding's fp32 math.
    const half = D / 2;
    const pe = new Float32Array(MAX_SEQ * D);
    const freq = new Float32Array(half);
    for (let j = 0; j < half; j++) freq[j] = Math.fround(10000 ** Math.fround(j / (half - 1)));
    for (let pos = 0; pos < MAX_SEQ; pos++) {
      for (let j = 0; j < half; j++) {
        const ph = Math.fround(pos / freq[j]);
        pe[pos * D + j] = Math.cos(ph);
        pe[pos * D + half + j] = Math.sin(ph);
      }
    }
    this.pe = this._upload32(pe);

    const kvBytes = MAX_SEQ * D * (this.f16 ? 2 : 4);
    for (const l of this.layers) {
      l.k = this._buf(kvBytes);
      l.v = this._buf(kvBytes);
    }
    const R = MAX_PREFILL;
    this.X = this._buf(R * D * 4);
    this.Hn = this._buf(R * D * 4);
    this.QKV = this._buf(R * 3 * D * 4);
    this.O = this._buf(R * D * 4);
    this.F = this._buf(R * 4 * D * 4);
    this.mel = this._buf(512 * 512 * 4);
    this.part = this._buf(this.H * ATTN_SPLITS * PART_STRIDE * 4);
    this.logits = this._buf(this.card * 4);
    this.forbid = this._upload32(new Uint32Array(this.card));
    this.state = this._buf(16);
    this.history = this._buf(4096 * 4);
    this.dummy = this._buf(16);
    await this.device.queue.onSubmittedWorkDone();
  }

  _uniform(vals) {
    const b = this.device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const a = new Uint32Array(8);
    a.set(vals);
    this.device.queue.writeBuffer(b, 0, a);
    return b;
  }

  _op(name, uni, buffers, dispatch) {
    const pipeline = this.pl[name];
    const entries = [{ binding: 0, resource: { buffer: this._uniform(uni) } }];
    buffers.forEach((buffer, i) => entries.push({ binding: i + 1, resource: { buffer } }));
    const bindGroup = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    return { pipeline, bindGroup, dispatch, uniform: entries[0].resource.buffer };
  }

  /** Ops for a forward pass over M rows already in X (at positions st.pos..). */
  _forwardOps(M, decode) {
    const { D, H } = this;
    const ops = [];
    const lin = (X, W, B, Y, N, K, flags) => (decode
      ? this._op('matvec', [1, N, K, flags], [X, W, B ?? this.dummy, Y, this.state], [Math.ceil(N / 8)])
      : this._op('matmul', [M, N, K, flags], [X, W, B ?? this.dummy, Y], [Math.ceil(N / 64), Math.ceil(M / 64)]));
    ops.push(this._op('addpos', [M, D], [this.state, this.pe, this.X], [Math.ceil((M * D) / 256)]));
    for (const l of this.layers) {
      ops.push(this._op('layernorm', [M, D, 0, 0, 0], [this.X, l.n1g, l.n1b, this.Hn], [M]));
      ops.push(lin(this.Hn, l.inProj, null, this.QKV, 3 * D, D, 0));
      ops.push(this._op('kvwrite', [M, D], [this.state, this.QKV, l.k, l.v],
        [Math.ceil((M * D) / (this.f16 ? 512 : 256))]));
      if (decode) {
        ops.push(this._op('attentionSplit', [1, D, H, 0, ATTN_SPLITS], [this.state, this.QKV, l.k, l.v, this.part], [ATTN_SPLITS, H]));
        ops.push(this._op('attentionCombine', [1, D, H, 0, ATTN_SPLITS], [this.part, this.O], [H]));
      } else {
        ops.push(this._op('attention', [M, D, H], [this.state, this.QKV, l.k, l.v, this.O], [M, H]));
      }
      ops.push(lin(this.O, l.outProj, null, this.X, D, D, 4));
      ops.push(this._op('layernorm', [M, D, 0, 0, 0], [this.X, l.n2g, l.n2b, this.Hn], [M]));
      ops.push(lin(this.Hn, l.l1, null, this.F, 4 * D, D, 2));
      ops.push(lin(this.F, l.l2, null, this.X, D, 4 * D, 4));
    }
    ops.push(this._op('layernorm', [1, D, 0, 0, M - 1], [this.X, this.outG, this.outB, this.Hn], [1]));
    ops.push(this._op('matvec', [1, this.card, D, 0], [this.Hn, this.head, this.dummy, this.logits, this.state],
      [Math.ceil(this.card / 8)]));
    ops.push(this._op('argmax', [M, this.card, 0, 0, VOCAB_LIMIT], [this.logits, this.forbid, this.state, this.history], [1]));
    return ops;
  }

  _buildDecode() {
    this.decodeOps = [
      this._op('embed', [1, this.D], [this.state, this.emb, this.X], [Math.ceil(this.D / 256)]),
      ...this._forwardOps(1, true),
    ];
  }

  _encode(pass, ops) {
    for (const op of ops) {
      pass.setPipeline(op.pipeline);
      pass.setBindGroup(0, op.bindGroup);
      pass.dispatchWorkgroups(...op.dispatch);
    }
  }

  /** Embedding rows for the non-mel prefix: dataset(null), instrument rows, token ids. */
  _prefixRows(instRows, tokens) {
    const { D } = this;
    const out = new Float32Array((1 + instRows.length + tokens.length) * D);
    out.set(this.dsEmb.subarray(D, 2 * D), 0); // dataset_name: always the null class (row 1)
    instRows.forEach((r, i) => out.set(this.instEmb.subarray(r * D, (r + 1) * D), (1 + i) * D));
    tokens.forEach((t, i) => out.set(this.embCPU.subarray(t * D, (t + 1) * D), (1 + instRows.length + i) * D));
    return out;
  }

  setForbidden(ids) {
    const mask = new Uint32Array(this.card);
    for (const i of ids) mask[i] = 1;
    this.device.queue.writeBuffer(this.forbid, 0, mask);
  }

  /**
   * Prefill one chunk and greedily decode until EOS.
   * @param mel {frames, data} log-mel features (501 x 512)
   * @param instRows instrument_group embedding rows
   * @param prompt forced tokens after the initial token (tie prologue)
   * @param maxGen token budget (gen_sequence length, as in LMModel.generate)
   * @returns {tokens, eos} prompt tokens followed by generated ones, EOS stripped
   */
  async generate({ mel, instRows, prompt = [], maxGen = 2000, debugLogits = false }) {
    if (this.lost) throw Object.assign(new Error(`GPU device lost: ${this.lost.message}`), { code: 'gpu-lost' });
    const { device, D } = this;
    const frames = mel.frames;
    const tokens = [this.card, ...prompt];
    const rows = this._prefixRows(instRows, tokens);
    const M = frames + rows.length / D;
    if (M > MAX_PREFILL) throw new Error(`prefill too long (${M} rows)`);
    const budget = Math.min(maxGen - prompt.length, MAX_SEQ - M - 1);

    device.queue.writeBuffer(this.mel, 0, mel.data);
    device.queue.writeBuffer(this.X, frames * D * 4, rows);
    device.queue.writeBuffer(this.state, 0, new Uint32Array(4));

    const melOp = this._op('matmul', [frames, D, 512, 1], [this.mel, this.melW, this.melB, this.X],
      [Math.ceil(D / 64), Math.ceil(frames / 64)]);
    const prefillOps = this._forwardOps(M, false);
    const transient = [melOp, ...prefillOps];

    let enqueued = 0;
    const pending = [];
    const submit = (first) => {
      const enc = device.createCommandEncoder();
      const start = enqueued;
      if (first) {
        let pass = enc.beginComputePass();
        this._encode(pass, [melOp]);
        pass.end();
        // Last mel frame is past the audio length: its projection is masked to zero.
        enc.clearBuffer(this.X, (frames - 1) * D * 4, D * 4);
        pass = enc.beginComputePass();
        this._encode(pass, prefillOps);
        pass.end();
        enqueued++;
        if (debugLogits) {
          const rb = device.createBuffer({ size: this.card * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
          enc.copyBufferToBuffer(this.logits, 0, rb, 0, this.card * 4);
          this._debugReadback = rb;
        }
      }
      const steps = Math.min(STEPS_PER_SUBMIT, budget - enqueued);
      if (steps > 0) {
        const pass = enc.beginComputePass();
        for (let s = 0; s < steps; s++) this._encode(pass, this.decodeOps);
        pass.end();
        enqueued += steps;
      }
      const n = enqueued - start;
      const rb = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      enc.copyBufferToBuffer(this.history, start * 4, rb, 0, n * 4);
      device.queue.submit([enc.finish()]);
      pending.push({ rb, done: rb.mapAsync(GPUMapMode.READ) });
    };

    const generated = [];
    let eos = false;
    try {
      submit(true);
      if (enqueued < budget) submit(false);
      while (pending.length) {
        const { rb, done } = pending.shift();
        await done;
        const vals = new Uint32Array(rb.getMappedRange().slice(0));
        rb.unmap();
        rb.destroy();
        if (eos) continue;
        for (const t of vals) {
          if (t === EOS_ID) { eos = true; break; }
          generated.push(t);
        }
        if (!eos && enqueued < budget) submit(false);
      }
    } finally {
      // Also on device loss / map failure: release everything this chunk allocated.
      for (const { rb } of pending) rb.destroy();
      for (const op of transient) op.uniform.destroy();
    }

    let logits = null;
    if (debugLogits) {
      await this._debugReadback.mapAsync(GPUMapMode.READ);
      logits = new Float32Array(this._debugReadback.getMappedRange().slice(0));
      this._debugReadback.destroy();
    }
    return { tokens: [...prompt, ...generated], eos, generated: generated.length, logits };
  }

  destroy() {
    this.device.destroy();
  }
}
