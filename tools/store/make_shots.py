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
  d.querySelector('[data-mode=midi]').click();
  const s = d.getElementById('scrub'); s.value = 700; s.dispatchEvent(new Event('change'));
  d.getElementById('playBtn').click(); await sleep(2500); d.getElementById('playBtn').click();
  return d.getElementById('timeLabel').textContent;
})()"""

# Hide the bass (a click on its row) so the roll zooms onto the guitar and every note gets a label.
HIDE_BASS = """(async () => {
  const d = panel().document;
  const row = [...d.querySelectorAll('.inst')].find((r) => /bass/i.test(r.textContent));
  row?.click();
  await new Promise((r) => setTimeout(r, 1800));
  return !!row;
})()"""

SHOTS = [
    dict(name='1-live-chords', after=60, rollh=230, state=dict(accepted=True, zoom=8),
         title='Hear a song, see its chords.',
         sub='byEar listens to any tab (YouTube, Spotify, a lesson video) and writes out the chords, notes and key while the music plays.',
         points=['chords for every instrument', 'the key of the song', 'private: runs on your GPU']),
    dict(name='2-instruments', after=60, rollh=150, state=dict(accepted=True, zoom=12),
         title='Every instrument, its own part.',
         sub='Guitar, bass, piano, voice, strings and drums: each one gets its own chords or notes. Click one to hide it.',
         points=['36 instrument types', 'chords and single lines', 'drum hits by name']),
    dict(name='3-notes', after=58, eval=HIDE_BASS, state=dict(accepted=True, zoom=5),
         title='Every note, labelled.',
         sub='Zoom into the piano roll to read each note by name and octave. Learn the part, then play it.',
         points=['zoom from 3 to 60 seconds', 'sharps and flats follow the key', 'clean dark piano roll']),
    dict(name='4-review-midi', after=63, state=dict(accepted=True, zoom=8), eval=REVIEW,
         title='Play it back as MIDI.',
         sub='When the song ends, switch between the original and the MIDI version, click to jump anywhere, and export a MIDI file for your DAW.',
         points=['original or midi, one click', 'export one track per instrument', 'nothing leaves your computer']),
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
