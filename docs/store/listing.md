# Chrome Web Store listing (v0.7.0)

## Name

byEar – Chord Finder & Audio to MIDI

## Summary (132 characters max)

Shows the chords, notes and key of the music in a browser tab, as a piano roll or guitar tabs. Runs on your device. Exports MIDI.

## Description

Hear a song, see its chords.

byEar listens to the music playing in any tab (YouTube, Spotify, a lesson video) and writes out the chords, notes and key while it plays, in a side panel next to the video.

WHAT YOU GET
• Chords for every instrument. Voice, guitar, keys, bass and drums each get their own channel strip with its chords or notes, plus mute and solo buttons like a mixer. Drag the channels into your own order.
• The vocal line. Switch on "vocals" for songs with singing and byEar follows the melody too.
• Guitar and bass tabs. Switch the canvas from "midi" to "tab". Parts are fingered the way a player would: hand positions, few jumps, open strings where they fit, standard chord shapes. Goes to 7-string guitar and 5-string bass when a part goes that low.
• A cleaner result after you stop. byEar writes along a few seconds behind the music, then listens to the whole recording again for a better transcription.
• Practise. Play back the original audio or the MIDI, at 100%, 75% or 50% (same pitch), and click anywhere to jump there.
• Hear and fix notes. Click a note to hear it. Alt + arrow keys move it a semitone, Delete removes it, and in tab view the arrow keys move it to another string. Your fixes go into the chords and the export.
• Export MIDI. One click, one track per instrument, ready for your DAW.
• 18 languages, with note names the way you learned them: C D E, Do Ré Mi, or C D E … H.

PRIVATE BY DESIGN
The music is transcribed on your own computer, on your GPU (WebGPU). No audio is uploaded anywhere, there's no account, and no tracking. The only download is the transcription model itself, once, from Hugging Face.

GOOD TO KNOW
• Needs a GPU with WebGPU (most computers from the last few years).
• Notes appear about 5 seconds after you hear them: byEar listens in 5-second slices.
• Settings let you pick the "accurate" model (catches more, needs a stronger GPU) and tell byEar which instruments to listen for.

Made by jenyadoesapps — https://jenyadoesapps.com
Transcription model: MuScriptor (open source).

## Screenshots (1280×800, 24-bit PNG, no alpha)

1. `1-live-chords.png` — Hear a song, see its chords.
2. `2-instruments.png` — Every instrument, its own channel.
3. `3-tabs.png` — Guitar and bass tabs.
4. `4-review-midi.png` — Practise it, fix it, export it.
5. `5-languages.png` — Your language, your note names.

Promo tile: `promo-tile-440x280.png` (unchanged).

## What changed since the last submission (v0.2.0 → v0.7.0)

New
• Guitar and bass tablature, with natural fingering: hand positions, open strings where they fit, standard chord shapes; 7-string guitar and 5-string bass.
• Practice speed: 75% and 50%, the original keeps its pitch.
• Click a note to hear it; fix it with Alt + arrow keys or Delete; move it to another string in tab view.
• MIDI playback is the default after a song; the vocal line plays as a violin, a little louder than the band.
• Drag channel strips to reorder them (tab staffs follow). Default order: voice, guitar, keys, bass, drums.
• Zoom in to 1 second, for fast passages.

Better
• Vocals: a dedicated voice pass that leaves the other instruments untouched, and it no longer drops out on some 5-second slices.
• No more missing 5-second slices in the notes, live or after the improving pass.
• Quiet audio is levelled automatically; the first notes of a song are no longer lost.
• Plain part names: "Guitar" and "Bass" instead of electric/acoustic/distorted variants.
• Easier note picking in tab view; smaller tooltips that don't cover the notes.
• Faster on long sessions (an hour of music stays smooth).
• The progress message pulses while the transcription is being improved.

Privacy: unchanged. Audio never leaves the device; the only network request is the model download from Hugging Face.
