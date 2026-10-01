// Minimal i18n: JSON dictionaries in ui/locales, {placeholder} interpolation,
// and DOM binding through data attributes:
//   data-i18n="key"                 text content (child [data-slot] elements fill {slot} placeholders)
//   data-i18n-attr="attr:key,..."   attribute values

export const LANGUAGES = {
  en: 'English',
  es: 'Español',
  pt_BR: 'Português (Brasil)',
  fr: 'Français',
  de: 'Deutsch',
  it: 'Italiano',
  ru: 'Русский',
  uk: 'Українська',
  pl: 'Polski',
  tr: 'Türkçe',
  ja: '日本語',
  ko: '한국어',
  zh_CN: '简体中文',
  zh_TW: '繁體中文',
  hi: 'हिन्दी',
  id: 'Bahasa Indonesia',
  vi: 'Tiếng Việt',
  ar: 'العربية',
};
const RTL = new Set(['ar']);

let dict = {};
let fallback = {};
export let lang = 'en';

/** Best supported language for the browser's preferences. */
export function detectLanguage(prefs = navigator.languages ?? [navigator.language]) {
  for (const raw of prefs) {
    const tag = raw.replace('-', '_');
    const [base, region] = tag.split('_');
    if (LANGUAGES[tag]) return tag;
    if (base === 'zh') return ['TW', 'HK', 'MO'].includes(region) || /Hant/.test(raw) ? 'zh_TW' : 'zh_CN';
    if (base === 'pt') return 'pt_BR';
    if (LANGUAGES[base]) return base;
  }
  return 'en';
}

async function fetchDict(code) {
  const res = await fetch(new URL(`./locales/${code}.json`, import.meta.url));
  if (!res.ok) throw new Error(`missing locale ${code}`);
  return res.json();
}

export async function setLanguage(code) {
  if (!Object.keys(fallback).length) fallback = await fetchDict('en');
  try {
    dict = code === 'en' ? fallback : await fetchDict(code);
    lang = code;
  } catch {
    dict = fallback;
    lang = 'en';
  }
  document.documentElement.lang = lang.replace('_', '-');
  document.documentElement.dir = RTL.has(lang) ? 'rtl' : 'ltr';
}

export function t(key, params = {}) {
  const s = dict[key] ?? fallback[key] ?? key;
  return s.replace(/\{(\w+)\}/g, (m, k) => (k in params ? String(params[k]) : m));
}

/** Translate every bound element under root. Safe to call again after a language change. */
export function applyI18n(root = document) {
  for (const el of root.querySelectorAll('[data-i18n]')) {
    if (!el._slots) {
      el._slots = new Map([...el.querySelectorAll(':scope > [data-slot]')].map((s) => [s.dataset.slot, s]));
    }
    const text = t(el.dataset.i18n);
    if (!el._slots.size) {
      el.textContent = text;
      continue;
    }
    const parts = text.split(/\{(\w+)\}/);
    el.replaceChildren(...parts.map((p, i) => (i % 2 ? el._slots.get(p) ?? `{${p}}` : p)).filter((n) => n !== ''));
  }
  for (const el of root.querySelectorAll('[data-i18n-attr]')) {
    for (const pair of el.dataset.i18nAttr.split(',')) {
      const [attr, key] = pair.split(':').map((s) => s.trim());
      el.setAttribute(attr, t(key));
    }
  }
}
