"""Draws the byEar icon (an ear whose curl turns into piano-roll notes) at every size Chrome needs.

python3 tools/make_icons.py [preview.png]   (needs Pillow)
"""
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw

S = 1024
INK = (238, 238, 238, 255)
NOTES = [(91, 141, 239), (229, 161, 46), (46, 158, 106)]  # piano, guitar, strings inks


def arc(cx, cy, r, a0, a1, n=60):
    return [(cx + r * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + r * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def bezier(p0, p1, p2, p3, n=40):
    out = []
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        out.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                    u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))
    return out


def stroke(d, pts, w):
    """Round-capped stroke: stamp discs densely along the polyline (no join artifacts)."""
    step = w / 10
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / step))
        for i in range(n + 1):
            x, y = x0 + (x1 - x0) * i / n, y0 + (y1 - y0) * i / n
            d.ellipse([x - w / 2, y - w / 2, x + w / 2, y + w / 2], fill=INK)


def icon(size):
    im = Image.new('RGBA', (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, S - 1, S - 1], radius=230, fill=(17, 17, 17, 255), outline=(52, 52, 52, 255), width=12)
    w = 84 if size >= 48 else 104  # thicker strokes survive downscaling
    # Outer ear: from the front (left), over the top, down the back, curling into the lobe.
    cx, cy, r = 470, 430, 230
    outer = arc(cx, cy, r, 160, 360)
    outer += bezier((cx + r, cy), (cx + r, cy + 170), (600, 600), (560, 700))
    outer += bezier((560, 700), (520, 790), (410, 800), (380, 720))
    stroke(d, outer, w)
    # Inner curl.
    inner = arc(cx, cy, 105, 190, 360) + bezier((cx + 105, cy), (cx + 105, cy + 90), (520, 560), (450, 570))
    stroke(d, inner, w)
    # What the ear heard: three note bars.
    for (x0, y, x1), col in zip([(760, 290, 930), (790, 445, 900), (760, 600, 880)], NOTES):
        d.rounded_rectangle([x0, y, x1, y + 78], radius=26, fill=col + (255,))
    return im.resize((size, size), Image.LANCZOS)


root = Path(__file__).resolve().parent.parent
for size in (16, 32, 48, 128):
    icon(size).save(root / 'icons' / f'{size}.png')
if len(sys.argv) > 1:
    sheet = Image.new('RGBA', (512 + 128 + 48 + 32 + 16 + 64, 512), (40, 40, 40, 255))
    x = 0
    for size in (512, 128, 48, 32, 16):
        sheet.paste(icon(size), (x, 0))
        x += size + 16
    sheet.save(sys.argv[1])
print('icons written')
