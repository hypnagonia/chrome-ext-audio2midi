// Pitch-class spelling in the user's note-name system.

const SYSTEMS = {
  letters: {
    sharp: ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'B'],
    flat: ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B♭', 'B'],
  },
  // German / Central European: B natural is H, B flat is B.
  german: {
    sharp: ['C', 'C♯', 'D', 'D♯', 'E', 'F', 'F♯', 'G', 'G♯', 'A', 'A♯', 'H'],
    flat: ['C', 'D♭', 'D', 'E♭', 'E', 'F', 'G♭', 'G', 'A♭', 'A', 'B', 'H'],
  },
  solfege: {
    sharp: ['Do', 'Do♯', 'Re', 'Re♯', 'Mi', 'Fa', 'Fa♯', 'Sol', 'Sol♯', 'La', 'La♯', 'Si'],
    flat: ['Do', 'Re♭', 'Re', 'Mi♭', 'Mi', 'Fa', 'Sol♭', 'Sol', 'La♭', 'La', 'Si♭', 'Si'],
  },
};
// Accented solfège spellings by language.
const ACCENTS = {
  fr: { Re: 'Ré' },
  pt_BR: { Do: 'Dó', Re: 'Ré', Fa: 'Fá', La: 'Lá' },
};
const accented = (lang) => {
  const map = ACCENTS[lang];
  const fix = (n) => n.replace(/^(Do|Re|Fa|La)/, (m) => map[m] ?? m);
  return { sharp: SYSTEMS.solfege.sharp.map(fix), flat: SYSTEMS.solfege.flat.map(fix) };
};

export const NAMING_MODES = ['letters', 'solfege', 'german'];
let current = SYSTEMS.letters;

/** The system musicians use by default in a language. */
export function defaultNaming(lang) {
  if (['fr', 'es', 'it'].includes(lang)) return 'solfege';
  if (['de', 'pl'].includes(lang)) return 'german';
  return 'letters';
}

export function setNaming(mode, lang) {
  current = mode === 'solfege' && ACCENTS[lang] ? accented(lang) : SYSTEMS[mode] ?? SYSTEMS.letters;
}

export const pcName = (pc, flats = false) => (flats ? current.flat : current.sharp)[((pc % 12) + 12) % 12];
export const noteName = (pitch, flats = false) => pcName(pitch % 12, flats) + (Math.floor(pitch / 12) - 1);
