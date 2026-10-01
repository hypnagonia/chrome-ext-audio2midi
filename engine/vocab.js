// MT3 token vocabulary and instrument groups, ported from
// muscriptor/tokenizer/notes.py and muscriptor/tokenizer/mt3.py.

export const DRUM_PROGRAM = 128;
export const MIN_NOTE_SEC = 0.01;
export const FRAME_RATE = 100;
export const SAMPLE_RATE = 16000;
export const SEGMENT_SAMPLES = 5 * SAMPLE_RATE;
export const VOCAB_LIMIT = 1393; // logits at or above this index are masked

const RANGES = [
  ['PAD', 0, 0], ['EOS', 0, 0], ['UNK', 0, 0],
  ['shift', 0, 1000],
  ['pitch', 0, 127],
  ['velocity', 0, 1],
  ['tie', 0, 0],
  ['program', 0, 129],
  ['drum', 0, 127],
];

/** Token index -> {type, value}. */
export const VOCAB = [];
const INDEX = new Map();
for (const [type, lo, hi] of RANGES) {
  for (let v = lo; v <= hi; v++) {
    INDEX.set(`${type}:${v}`, VOCAB.length);
    VOCAB.push({ type, value: v });
  }
}
export const EOS_ID = 1;
export const tokenId = (type, value) => INDEX.get(`${type}:${value}`);

// MT3_FULL_PLUS group -> GM programs. The model emits the first program of a group.
const GROUP_PROGRAMS = {
  0: [0, 1, 3, 6, 7], 1: [2, 4, 5], 2: [8, 9, 10, 11, 12, 13, 14, 15],
  3: [16, 17, 18, 19, 20, 21, 22, 23], 4: [24, 25], 5: [26, 27, 28], 6: [29, 30, 31],
  7: [32, 35], 8: [33, 34, 36, 37, 38, 39], 9: [40], 10: [41], 11: [42], 12: [43],
  13: [46], 14: [47], 15: [48, 49, 44, 45], 16: [50, 51], 17: [52, 53, 54], 18: [55],
  19: [56, 59], 20: [57], 21: [58], 22: [60], 23: [61, 62, 63], 24: [64, 65], 25: [66],
  26: [67], 27: [68], 28: [69], 29: [70], 30: [71], 31: [72, 73, 74, 75, 76, 77, 78, 79],
  32: [80, 81, 82, 83, 84, 85, 86, 87], 33: [88, 89, 90, 91, 92, 93, 94, 95], 34: [100], 35: [101],
};

export const GROUP_IDS = {
  acoustic_piano: 0, electric_piano: 1, chromatic_percussion: 2, organ: 3,
  acoustic_guitar: 4, clean_electric_guitar: 5, distorted_electric_guitar: 6,
  acoustic_bass: 7, electric_bass: 8, violin: 9, viola: 10, cello: 11, contrabass: 12,
  orchestral_harp: 13, timpani: 14, string_ensemble: 15, synth_strings: 16, voice: 17,
  orchestra_hit: 18, trumpet: 19, trombone: 20, tuba: 21, french_horn: 22,
  brass_section: 23, soprano_and_alto_sax: 24, tenor_sax: 25, baritone_sax: 26,
  oboe: 27, english_horn: 28, bassoon: 29, clarinet: 30, flutes: 31, synth_lead: 32,
  synth_pad: 33, drums: 36,
};
export const INSTRUMENT_NAMES = Object.keys(GROUP_IDS);

const PROGRAM_TO_NAME = new Map();
const NAME_TO_PROGRAM = new Map();
for (const [name, gid] of Object.entries(GROUP_IDS)) {
  const progs = GROUP_PROGRAMS[gid];
  if (progs) {
    PROGRAM_TO_NAME.set(progs[0], name);
    NAME_TO_PROGRAM.set(name, progs[0]);
  }
}

export function instrumentForProgram(program) {
  if (program === DRUM_PROGRAM) return 'drums';
  return PROGRAM_TO_NAME.get(program) ?? `program_${program}`;
}

export function programForInstrument(name) {
  if (name === 'drums') return DRUM_PROGRAM;
  if (NAME_TO_PROGRAM.has(name)) return NAME_TO_PROGRAM.get(name);
  if (name.startsWith('program_')) return Number(name.slice(8));
  return 0;
}

/** Rows of the instrument_group class embedding for conditioning (null -> row 1). */
export function instrumentConditionRows(names) {
  const known = (names || []).filter((n) => n in GROUP_IDS);
  if (known.length === 0) return [1];
  return known.map((n) => GROUP_IDS[n] + 2);
}

/** Tie prologue tokens declaring (program, pitch) pairs as sustained. */
export function tieSectionTokens(openKeys) {
  const sorted = [...openKeys].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const out = [];
  let program = null;
  for (const [prog, pitch] of sorted) {
    if (prog !== program) {
      out.push(tokenId('program', prog));
      program = prog;
    }
    out.push(tokenId('pitch', pitch));
  }
  out.push(tokenId('tie', 0));
  return out;
}

/** Token ids that may not be sampled when only `names` are allowed (null = all allowed). */
export function forbiddenTokens(names) {
  names = (names || []).filter((n) => n in GROUP_IDS);
  if (names.length === 0) return [];
  const allowDrums = names.includes('drums');
  const allowed = new Set(names.filter((n) => n !== 'drums').map((n) => GROUP_PROGRAMS[GROUP_IDS[n]][0]));
  const out = [];
  VOCAB.forEach((e, i) => {
    if (e.type === 'program' && !allowed.has(e.value)) out.push(i);
    else if (e.type === 'drum' && !allowDrums) out.push(i);
  });
  return out;
}
