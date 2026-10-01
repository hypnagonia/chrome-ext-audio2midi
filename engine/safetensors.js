// Minimal safetensors reader over an ArrayBuffer.

function f16ToF32(h) {
  const s = (h & 0x8000) ? -1 : 1;
  const e = (h >> 10) & 0x1f;
  const m = h & 0x3ff;
  if (e === 0) return s * m * 2 ** -24;
  if (e === 31) return m ? NaN : s * Infinity;
  return s * (1 + m / 1024) * 2 ** (e - 15);
}

export function parseSafetensors(buffer) {
  const view = new DataView(buffer);
  const headerLen = Number(view.getBigUint64(0, true));
  const header = JSON.parse(new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen)));
  const base = 8 + headerLen;
  const tensors = new Map();
  for (const [name, info] of Object.entries(header)) {
    if (name === '__metadata__') continue;
    const [begin, end] = info.data_offsets;
    tensors.set(name, { dtype: info.dtype, shape: info.shape, offset: base + begin, bytes: end - begin });
  }
  // Legacy multi-codebook keys (emb.0.* / linears.0.*) -> single stream names.
  for (const [from, to] of [['emb.0.weight', 'emb.weight'], ['linears.0.weight', 'linear.weight']]) {
    if (tensors.has(from) && !tensors.has(to)) tensors.set(to, tensors.get(from));
  }

  /** Returns the tensor as Float32Array (copying/converting as needed). */
  function getF32(name) {
    const t = tensors.get(name);
    if (!t) throw new Error(`missing tensor ${name}`);
    const n = t.bytes / { F32: 4, F16: 2, BF16: 2 }[t.dtype];
    if (t.dtype === 'F32') {
      if (t.offset % 4 === 0) return new Float32Array(buffer, t.offset, n);
      return new Float32Array(buffer.slice(t.offset, t.offset + t.bytes));
    }
    const src = new Uint16Array(buffer.slice(t.offset, t.offset + t.bytes));
    const out = new Float32Array(n);
    if (t.dtype === 'F16') for (let i = 0; i < n; i++) out[i] = f16ToF32(src[i]);
    else {
      const u = new Uint32Array(out.buffer);
      for (let i = 0; i < n; i++) u[i] = src[i] << 16;
    }
    return out;
  }

  /** Raw fp16 bits when the tensor is stored as F16, else null. */
  function getF16Bits(name) {
    const t = tensors.get(name);
    if (!t || t.dtype !== 'F16') return null;
    return t.offset % 2 === 0
      ? new Uint16Array(buffer, t.offset, t.bytes / 2)
      : new Uint16Array(buffer.slice(t.offset, t.offset + t.bytes));
  }

  return { tensors, getF32, getF16Bits, has: (n) => tensors.has(n), shape: (n) => tensors.get(n)?.shape };
}
