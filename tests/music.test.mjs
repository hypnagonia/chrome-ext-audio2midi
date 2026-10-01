// node tests/music.test.mjs [out.mid]
import fs from 'node:fs';
import assert from 'node:assert/strict';
import { chordSegments, chordName, estimateKey, matchChord } from '../music/chords.js';
import { pcName, setNaming } from '../music/names.js';
const keyName = (k) => `${pcName(k.tonic, k.useFlats)} ${k.minor ? 'minor' : 'major'}`;
import { notesToMidi } from '../music/midi.js';

const chord = (pitches, start, end, instrument = 'acoustic_piano') => pitches.map((pitch) => ({ pitch, start, end, instrument }));
// C – Am – F/A – G7 – Bdim, one second each
const notes = [
  ...chord([48, 64, 67, 72], 0, 1),
  ...chord([45, 60, 64, 69], 1, 2),
  ...chord([45, 60, 65, 69], 2, 3),
  ...chord([43, 59, 62, 65, 67], 3, 4),
  ...chord([59, 62, 65], 4, 5),
];
const segs = chordSegments(notes, 0, 5).map((s) => `${chordName(s.chord)}@${s.start}`);
assert.deepEqual(segs, ['C@0', 'Am@1', 'F/A@2', 'G7@3', 'Bdim@4']);
assert.equal(matchChord(new Float64Array(12).fill(0).map((_, i) => (i === 0 ? 1 : 0)), 0), null, 'single note is no chord');
assert.equal(chordName(matchChord(Float64Array.from([1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]), 0)), 'C5');
const cadence = [...chord([48, 64, 67, 72], 0, 1), ...chord([53, 60, 65, 69], 1, 2), ...chord([43, 59, 62, 65, 67], 2, 3), ...chord([48, 60, 64, 67], 3, 5)];
const key = estimateKey(cadence, 0, 5);
assert.equal(keyName(key), 'C major');
const fKey = estimateKey(chord([53, 57, 60], 0, 2).concat(chord([58, 62, 65], 2, 3), chord([48, 52, 55, 58], 3, 4)), 0, 4);
assert.equal(keyName(fKey), 'F major');
assert.equal(chordName(matchChord(Float64Array.from([0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0]), 3), fKey.useFlats), 'E♭', 'flat spelling');

// A walking bass line (one note at a time) is not a chord.
const walk = [40, 43, 45, 47, 45, 43].map((pitch, i) => ({ pitch, start: i * 0.2, end: i * 0.2 + 0.19, instrument: 'electric_bass' }));
assert.deepEqual(chordSegments(walk, 0, 1.2), [], 'monophonic line has no chords');
// ...but bass under sustained guitar dyads still yields a chord for the ensemble.
const ens = [...walk.slice(0, 1).map((n) => ({ ...n, end: 1 })), ...chord([55, 59], 0, 1, 'clean_electric_guitar')];
assert.equal(chordName(chordSegments(ens, 0, 1)[0].chord), 'Em');
// Staccato chord (notes shorter than the 30 ms overlap guard) still counts.
assert.equal(chordName(chordSegments(chord([60, 64, 67], 0, 0.02), 0, 0.25)[0].chord), 'C');
const midi = notesToMidi([...notes, { instrument: 'drums', pitch: 36, start: 0.5, end: 0.51 }, { instrument: 'electric_bass', pitch: 36, start: 0, end: 2 }]);
fs.writeFileSync(process.argv[2] || '/tmp/a2m-test.mid', midi);
// 17 instruments: no channel collision for different programs beyond what GM allows, file parses.
const many = ['acoustic_piano', 'electric_piano', 'organ', 'acoustic_guitar', 'clean_electric_guitar', 'distorted_electric_guitar',
  'electric_bass', 'violin', 'viola', 'cello', 'trumpet', 'trombone', 'flutes', 'clarinet', 'oboe', 'synth_lead', 'synth_pad']
  .map((instrument, i) => ({ instrument, pitch: 60 + i, start: i * 0.1, end: i * 0.1 + 1 }));
fs.writeFileSync('/tmp/a2m-many.mid', notesToMidi([...many, { instrument: 'acoustic_piano', pitch: 60, start: 0.5, end: 3 }]));
// Other note-name systems
setNaming('solfege', 'fr');
assert.equal(chordName(matchChord(Float64Array.from([1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0].map((v, i) => [2, 6, 9].includes(i) ? 1 : 0)), 2)), 'Ré');
setNaming('german', 'de');
assert.equal(pcName(10, true) + pcName(11), 'BH');
setNaming('letters', 'en');
console.log('music OK:', segs.join(' '), '| key', keyName(key), '| midi bytes', midi.length);
