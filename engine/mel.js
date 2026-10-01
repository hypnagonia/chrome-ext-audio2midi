// Log-mel features matching MelSpectrogramConditioner: n_fft 2048, hop 160,
// center + reflect pad, magnitude (power 1), HTK mel fb, log(x + 1e-6).

const N_FFT = 2048;
const HOP = 160;
const N_BINS = N_FFT / 2 + 1;
export const N_MELS = 512;

export class MelFrontend {
  /** window: Float32Array(2048); fb: Float32Array(1025 * 512), row-major [bin][mel]. */
  constructor(window, fb) {
    this.window = window;
    // Sparse filterbank: per mel band, contiguous nonzero bin range + weights.
    this.bands = [];
    for (let m = 0; m < N_MELS; m++) {
      let lo = -1, hi = -1;
      for (let b = 0; b < N_BINS; b++) {
        if (fb[b * N_MELS + m] !== 0) { if (lo < 0) lo = b; hi = b; }
      }
      const w = lo < 0 ? new Float32Array(0) : new Float32Array(hi - lo + 1);
      for (let b = lo; b <= hi && lo >= 0; b++) w[b - lo] = fb[b * N_MELS + m];
      this.bands.push({ lo, w });
    }
    // FFT tables
    const bits = Math.log2(N_FFT);
    this.rev = new Uint32Array(N_FFT);
    for (let i = 0; i < N_FFT; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(N_FFT / 2);
    this.sin = new Float64Array(N_FFT / 2);
    for (let i = 0; i < N_FFT / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / N_FFT);
      this.sin[i] = -Math.sin((2 * Math.PI * i) / N_FFT);
    }
    this.re = new Float64Array(N_FFT);
    this.im = new Float64Array(N_FFT);
    this.mag = new Float64Array(N_BINS);
  }

  /** samples: Float32Array -> {frames, data: Float32Array(frames * 512)} */
  compute(samples) {
    const n = samples.length;
    const pad = N_FFT / 2;
    const frames = Math.floor(n / HOP) + 1;
    const out = new Float32Array(frames * N_MELS);
    const at = (i) => {
      // reflect padding
      let j = i - pad;
      if (j < 0) j = -j;
      if (j >= n) j = 2 * (n - 1) - j;
      return samples[j];
    };
    const { re, im, rev, mag } = this;
    for (let f = 0; f < frames; f++) {
      const start = f * HOP;
      for (let i = 0; i < N_FFT; i++) {
        re[rev[i]] = at(start + i) * this.window[i];
        im[rev[i]] = 0;
      }
      this._fft();
      for (let b = 0; b < N_BINS; b++) mag[b] = Math.hypot(re[b], im[b]);
      const row = f * N_MELS;
      for (let m = 0; m < N_MELS; m++) {
        const { lo, w } = this.bands[m];
        let s = 0;
        for (let k = 0; k < w.length; k++) s += mag[lo + k] * w[k];
        out[row + m] = Math.log(s + 1e-6);
      }
    }
    return { frames, data: out };
  }

  _fft() {
    const { re, im } = this;
    for (let size = 2; size <= N_FFT; size <<= 1) {
      const half = size >> 1;
      const step = N_FFT / size;
      for (let i = 0; i < N_FFT; i += size) {
        for (let j = 0; j < half; j++) {
          const wr = this.cos[j * step], wi = this.sin[j * step];
          const a = i + j, b = a + half;
          const tr = re[b] * wr - im[b] * wi;
          const ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr; im[a] += ti;
        }
      }
    }
  }
}
