// node tests/mel.test.mjs <model.safetensors> <audio.f32> <ref.json>
import fs from 'node:fs';
import { parseSafetensors } from '../engine/safetensors.js';
import { MelFrontend } from '../engine/mel.js';

const [wPath, aPath, rPath] = process.argv.slice(2);
const buf = fs.readFileSync(wPath);
const st = parseSafetensors(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
const P = 'condition_provider.conditioners.self_wav.mel_spec_transform.';
const mel = new MelFrontend(st.getF32(P + 'spectrogram.window'), st.getF32(P + 'mel_scale.fb'));
const a = fs.readFileSync(aPath);
const audio = new Float32Array(a.buffer.slice(a.byteOffset, a.byteOffset + a.byteLength)).subarray(0, 80000);
const t0 = performance.now();
const { frames, data } = mel.compute(audio);
const ms = performance.now() - t0;
const ref = JSON.parse(fs.readFileSync(rPath, 'utf8'));
let maxErr = 0;
ref.mel.forEach((v, i) => { maxErr = Math.max(maxErr, Math.abs(v - data[i])); });
const tail = data.subarray((frames - 2) * 512);
ref.mel_last_rows.forEach((v, i) => { maxErr = Math.max(maxErr, Math.abs(v - tail[i])); });
const sum = data.reduce((s, v) => s + v, 0);
console.log(`frames=${frames} maxAbsErr=${maxErr.toExponential(2)} sum=${sum.toFixed(1)} ref=${ref.mel_sum.toFixed(1)} ${ms.toFixed(0)}ms`);
if (frames !== 501 || maxErr > 1e-3) process.exit(1);
