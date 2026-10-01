// Display names and ink colours per instrument, grouped by family hue.

import { t } from './i18n.js';
export { noteName } from '../music/names.js';

const COLORS = {
  acoustic_piano: '#2F6FDF', electric_piano: '#5B8DEF', organ: '#3D4FB8', chromatic_percussion: '#4AA3C9',
  acoustic_guitar: '#D9822B', clean_electric_guitar: '#E5A12E', distorted_electric_guitar: '#D2512B',
  acoustic_bass: '#A33A4A', electric_bass: '#BF4B5C', contrabass: '#8C2F45',
  violin: '#2E9E6A', viola: '#3BAA80', cello: '#22805A', orchestral_harp: '#6CB86A',
  string_ensemble: '#2FA39A', synth_strings: '#47B5A6', timpani: '#8A7A5A',
  voice: '#E0569B',
  trumpet: '#C9A227', trombone: '#B88A1E', tuba: '#9C7417', french_horn: '#D4B03C', brass_section: '#C69A2E',
  orchestra_hit: '#B5652A',
  soprano_and_alto_sax: '#1F9BB5', tenor_sax: '#1A86A0', baritone_sax: '#176F85', oboe: '#3FA6C7',
  english_horn: '#4BB3CF', bassoon: '#2B7E95', clarinet: '#5BB8D6', flutes: '#7CC6DF',
  synth_lead: '#8E5BE0', synth_pad: '#A47CE8',
  drums: '#7B8494',
};

export function instrumentLabel(name) {
  if (name.startsWith('program_')) return t('inst.program', { n: Number(name.slice(8)) + 1 });
  const label = t(`inst.${name}`);
  return label === `inst.${name}` ? name.replace(/_/g, ' ') : label;
}

export function instrumentColor(name) {
  if (COLORS[name]) return COLORS[name];
  let h = 0;
  for (const c of name) h = (h * 31 + c.charCodeAt(0)) % 360;
  return `hsl(${h} 55% 50%)`;
}

// General MIDI percussion names for the drum notes the model commonly emits.
const DRUMS = {
  35: 'kick', 36: 'kick', 37: 'rim', 38: 'snare', 39: 'clap', 40: 'snare', 41: 'lowTom', 42: 'hihat',
  43: 'lowTom', 44: 'pedalHat', 45: 'midTom', 46: 'openHat', 47: 'midTom', 48: 'highTom', 49: 'crash',
  50: 'highTom', 51: 'ride', 52: 'china', 53: 'rideBell', 54: 'tambourine', 55: 'splash', 56: 'cowbell', 57: 'crash', 59: 'ride',
};
export const drumName = (p) => (DRUMS[p] ? t(`drum.${DRUMS[p]}`) : t('drum.perc', { n: p }));
