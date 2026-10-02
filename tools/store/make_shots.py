"""Renders the five Chrome Web Store screenshots (1280x800, 24-bit PNG, no alpha) into docs/store/.

Needs: a local server on :8765 at the repo root, tests/fixtures/{small-fp16.safetensors,song.mp3},
Pillow, and node. Each shot plays the song in a muted headless Chrome, so this takes a few minutes.
"""
import json
import os
import subprocess
import sys
import urllib.parse
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'store'
OUT.mkdir(parents=True, exist_ok=True)
PANEL = 'weights=tests/fixtures/small-fp16.safetensors&src=tests/fixtures/song.mp3'
REVIEW = """(async () => {
  const d = panel().document;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 40 && d.getElementById('transport').hidden; i++) await sleep(500);
  // Let the improving pass finish (final notes and fingering), then close its message.
  for (let i = 0; i < 240 && !/improved/i.test(d.getElementById('bannerText').textContent); i++) await sleep(500);
  await sleep(300);
  d.getElementById('bannerClose').click();
  d.querySelector('[data-mode=midi]').click();
  d.getElementById('speedBtn').click(); // 75%: practice speed
  const s = d.getElementById('scrub'); s.value = 700; s.dispatchEvent(new Event('change'));
  d.getElementById('playBtn').click(); await sleep(2500); d.getElementById('playBtn').click();
  return d.getElementById('timeLabel').textContent;
})()"""

# Tab view, in review, parked on a stretch with guitar and bass.
TAB = """(async () => {
  const d = panel().document;
  const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
  for (let i = 0; i < 40 && d.getElementById('transport').hidden; i++) await sleep(500);
  // Let the improving pass finish (final notes and fingering), then close its message.
  for (let i = 0; i < 240 && !/improved/i.test(d.getElementById('bannerText').textContent); i++) await sleep(500);
  await sleep(300);
  d.getElementById('bannerClose').click();
  // Solo the guitar: one clear staff, the way you'd practise a part.
  [...d.querySelectorAll('.inst')].find((r) => /guitar/i.test(r.textContent))?.querySelector('.solo')?.click();
  const s = d.getElementById('scrub'); s.value = 680; s.dispatchEvent(new Event('change'));
  await sleep(1500);
  return d.getElementById('timeLabel').textContent;
})()"""

SHOTS = [
    dict(name='1-live-chords', after=60, rollh=230, state=dict(accepted=True, zoom=8, vocals=True),
         title='Hear a song, see its chords.',
         sub='byEar listens to any tab (YouTube, Spotify, a lesson video) and writes out the chords, notes and key while the music plays.',
         points=['chords for every instrument', 'the key of the song', 'private: runs on your GPU']),
    dict(name='2-instruments', after=60, rollh=150, state=dict(accepted=True, zoom=12, vocals=True),
         title='Every instrument, its own channel.',
         sub='Voice, guitar, keys, bass and drums each get a channel strip with its chords or notes. Mute, solo, or drag them into your own order.',
         points=['mute and solo, like a mixer', 'vocals, chords and single lines', 'drum hits by name']),
    dict(name='3-tabs', after=63, state=dict(accepted=True, zoom=5, view='tab'), eval=TAB,
         title='Guitar and bass tabs.',
         sub='Switch to tab and every part is fingered the way a player would: hand positions, open strings where they fit, standard chord shapes.',
         points=['7-string guitar and 5-string bass', 'click a fret to hear it', 'move a note to another string']),
    dict(name='4-review-midi', after=63, state=dict(accepted=True, zoom=8, vocals=True), eval=REVIEW,
         title='Practise it, fix it, export it.',
         sub='When the song ends, play it back as MIDI or the original, slow it down, click any note to hear it, correct it, and export a MIDI file for your DAW.',
         points=['75% and 50% speed, same pitch', 'fix notes with Alt + arrow keys', 'one MIDI track per instrument']),
    dict(name='5-languages', after=57, state=dict(accepted=True, zoom=8, lang='fr'),
         title='Your language, your note names.',
         sub='18 languages, with notes named the way you learned them: C D E, Do Ré Mi, or C D E … H.',
         points=['English, Español, Français, Deutsch…', '日本語, 한국어, 中文, العربية…', 'switch any time in settings']),
]

only = sys.argv[1:]
for shot in SHOTS:
    if only and shot['name'] not in only:
        continue
    query = urllib.parse.urlencode({
        'title': shot['title'], 'sub': shot['sub'], 'points': '|'.join(shot['points']),
        'state': json.dumps(shot['state']), 'panel': PANEL, 'rollh': shot.get('rollh', ''),
    })
    raw = OUT / f"{shot['name']}@2x.png"
    env = dict(os.environ, WIDTH='1280', HEIGHT='800', DPR='2', SHOT=str(raw), AFTER=str(shot['after']))
    if shot.get('eval'):
        env.update(EVAL=shot['eval'], EVAL_WAIT='1')
    subprocess.run(['node', str(ROOT / 'tests' / 'run-in-chrome.mjs'), f'http://127.0.0.1:8765/tools/store/shot.html?{query}'],
                   env=env, check=True, stdout=subprocess.DEVNULL)
    img = Image.open(raw).convert('RGB').resize((1280, 800), Image.LANCZOS)
    img.save(OUT / f"{shot['name']}.png", optimize=True)
    raw.unlink()
    print('wrote', f"docs/store/{shot['name']}.png", img.size, img.mode)
