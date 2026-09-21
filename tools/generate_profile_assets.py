from __future__ import annotations

import html
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from scipy.ndimage import binary_closing, binary_fill_holes, label
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
PORTRAIT = ROOT / "assets" / "source-portrait.png"
LOGO_DIR = ROOT / "tools" / "logo-masks"
OUT = ROOT / "assets"
OUT.mkdir(parents=True, exist_ok=True)

W, H = 1180, 610
PW, PH = 300, 340
PX, PY = 86, 170
RNG = np.random.default_rng(1709)


def prepare_portrait() -> tuple[np.ndarray, np.ndarray]:
    im = Image.open(PORTRAIT).convert("RGB")
    side = min(im.size)
    left = (im.width - side) // 2
    im = im.crop((left, 0, left + side, side)).resize((PW, PH), Image.Resampling.LANCZOS)
    rgb = np.asarray(im).astype(np.float32)
    gray = ImageOps.grayscale(im)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.3)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=2))
    g = np.asarray(gray).astype(np.float32) / 255.0

    corners = np.concatenate([
        rgb[:35, :35].reshape(-1, 3), rgb[:35, -35:].reshape(-1, 3),
        rgb[-35:, :35].reshape(-1, 3), rgb[-35:, -35:].reshape(-1, 3),
    ])
    bg = np.median(corners, axis=0)
    dist = np.linalg.norm(rgb - bg, axis=2)
    yy, xx = np.mgrid[:PH, :PW]
    width = np.where(yy < 55, 60, np.where(yy < 190, 91, np.where(yy < 250, 125, 150)))
    silhouette = np.abs(xx - PW / 2) < width
    mask = (dist > 24) & silhouette
    mask = binary_closing(mask, iterations=3)
    mask = binary_fill_holes(mask)
    labs, count = label(mask)
    if count:
        sizes = np.bincount(labs.ravel())
        sizes[0] = 0
        mask = labs == sizes.argmax()

    dark = floyd_steinberg(g, mask=mask)
    light_full = floyd_steinberg(1.0 - g, mask=np.ones_like(mask, dtype=bool))
    # Keep every subject dot for facial detail, then sample only the background.
    # This preserves the light-mode backdrop without creating a 3 MB SVG.
    light = np.zeros_like(light_full)
    sy, sx = np.where((light_full > 0) & mask)
    subject_cap = min(15000, len(sy))
    if subject_cap:
        weights = (1.0 - g[sy, sx]) + 0.08
        weights = weights / weights.sum()
        selected = RNG.choice(len(sy), subject_cap, replace=False, p=weights)
        light[sy[selected], sx[selected]] = 1
    remaining = max(0, 19000 - int(light.sum()))
    by, bx = np.where((light_full > 0) & ~mask)
    if remaining and len(by):
        keep = RNG.choice(len(by), min(remaining, len(by)), replace=False)
        light[by[keep], bx[keep]] = 1
    np.save(OUT / "portrait-dark.npy", dark)
    np.save(OUT / "portrait-light.npy", light)
    return dark, light


def floyd_steinberg(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    a = values.copy()
    result = np.zeros(a.shape, dtype=np.uint8)
    for y in range(a.shape[0]):
        forward = y % 2 == 0
        xs = range(a.shape[1]) if forward else range(a.shape[1] - 1, -1, -1)
        for x in xs:
            if not mask[y, x]:
                a[y, x] = 0
                continue
            old = a[y, x]
            new = 1.0 if old >= 0.5 else 0.0
            result[y, x] = int(new)
            err = old - new
            if forward:
                neighbors = [(x + 1, y, 7 / 16), (x - 1, y + 1, 3 / 16), (x, y + 1, 5 / 16), (x + 1, y + 1, 1 / 16)]
            else:
                neighbors = [(x - 1, y, 7 / 16), (x + 1, y + 1, 3 / 16), (x, y + 1, 5 / 16), (x - 1, y + 1, 1 / 16)]
            for nx, ny, weight in neighbors:
                if 0 <= nx < a.shape[1] and 0 <= ny < a.shape[0] and mask[ny, nx]:
                    a[ny, nx] = np.clip(a[ny, nx] + err * weight, 0, 1)
    return result


def path_for(points: np.ndarray, scale: float = 1.0, ox: float = 0, oy: float = 0) -> str:
    if len(points) == 0:
        return ""
    return "".join(f"M{ox + x * scale:.2f},{oy + y * scale:.2f}h{0.86 * scale:.2f}" for y, x in points)


def portrait_layers(bits: np.ndarray, color: str) -> str:
    pts = np.argwhere(bits > 0)
    intro_groups = RNG.integers(0, 60, len(pts))
    noise = RNG.normal(0, 4, len(pts))
    band_values = (pts[:, 1] * 0.47 + pts[:, 0] * 0.31 + noise)
    lo, hi = band_values.min(), band_values.max()
    bands = np.clip(((band_values - lo) / max(1e-6, hi - lo) * 93).astype(int), 0, 93)

    chunks = [f'<g transform="translate({PX} {PY})" fill="none" stroke="{color}" stroke-width="1.12" shape-rendering="crispEdges">']
    chunks.append('<g id="portrait-intro">')
    for i in range(60):
        p = pts[intro_groups == i]
        if not len(p):
            continue
        begin = 0.16 + (i % 12) * 0.065 + (i // 12) * 0.025
        chunks.append(f'<path d="{path_for(p)}" opacity="0"><animate attributeName="opacity" values="0;1" dur="1.6s" begin="{begin:.2f}s" fill="freeze" /></path>')
    chunks.append('</g><g id="portrait-loop">')
    for i in range(94):
        p = pts[bands == i]
        if not len(p):
            continue
        cx = p[:, 1].mean()
        cy = p[:, 0].mean()
        dx = (PW / 2 - cx) * 0.42 + RNG.normal(0, 2.0)
        dy = (PH / 2 - cy) * 0.42 + RNG.normal(0, 2.0)
        chunks.append(
            f'<path d="{path_for(p)}" opacity="1">'
            f'<animate attributeName="opacity" values="1;1;0;0;0;0;0;0;1;1" keyTimes="0;0.19;0.27;0.40;0.50;0.63;0.73;0.86;0.94;1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />'
            f'<animateTransform attributeName="transform" type="translate" values="0 0;0 0;{dx:.1f} {dy:.1f};0 0" keyTimes="0;0.19;0.27;1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />'
            '</path>'
        )
    chunks.append('</g></g>')
    return "".join(chunks)


def logo_points(name: str, n: int = 900) -> np.ndarray:
    im = Image.open(LOGO_DIR / f"{name}.png").convert("RGBA")
    a = np.asarray(im)[:, :, 3]
    pts = np.argwhere(a > 80)
    if len(pts) < n:
        idx = RNG.choice(len(pts), n, replace=True)
    else:
        idx = RNG.choice(len(pts), n, replace=False)
    p = pts[idx].astype(float)
    p[:, [0, 1]] = p[:, [1, 0]]
    p -= p.mean(axis=0)
    span = max(np.ptp(p[:, 0]), np.ptp(p[:, 1]))
    return p / max(span, 1) * 176 + np.array([PW / 2, PH / 2])


def match(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    out = np.empty_like(b)
    out[rows] = b[cols]
    return out


def travellers(color: str) -> str:
    logos = [logo_points("python"), logo_points("pytorch"), logo_points("espressif")]
    logos[1] = match(logos[0], logos[1])
    logos[2] = match(logos[1], logos[2])
    start = np.column_stack([RNG.uniform(50, PW - 50, 900), RNG.uniform(55, PH - 40, 900)])
    vals = []
    for i in range(900):
        coords = [start[i], start[i], logos[0][i], logos[0][i], logos[1][i], logos[1][i], logos[2][i], logos[2][i], start[i], start[i]]
        vals.append(";".join(f"{x:.1f},{y:.1f}" for x, y in coords))
    chunks = [f'<g transform="translate({PX} {PY})" fill="{color}" opacity="0">',
              '<animate attributeName="opacity" values="0;0;1;1;1;1;1;1;0;0" keyTimes="0;0.19;0.27;0.40;0.50;0.63;0.73;0.86;0.94;1" dur="14.2s" begin="3.2s" repeatCount="indefinite" />']
    for i, v in enumerate(vals):
        chunks.append(f'<circle cx="0" cy="0" r="1.35"><animate attributeName="cx" values="{v.replace(",", " ").split(";")[0].split()[0]};{v.replace(",", " ").split(";")[1].split()[0]}" dur="0.1s" fill="freeze" />')
        coords = v.split(';')
        xs = ';'.join(c.split(',')[0] for c in coords)
        ys = ';'.join(c.split(',')[1] for c in coords)
        chunks[-1] = f'<circle cx="{start[i,0]:.1f}" cy="{start[i,1]:.1f}" r="1.35"><animate attributeName="cx" values="{xs}" keyTimes="0;0.19;0.27;0.40;0.50;0.63;0.73;0.86;0.94;1" dur="14.2s" begin="3.2s" repeatCount="indefinite" /><animate attributeName="cy" values="{ys}" keyTimes="0;0.19;0.27;0.40;0.50;0.63;0.73;0.86;0.94;1" dur="14.2s" begin="3.2s" repeatCount="indefinite" /></circle>'
    chunks.append('</g>')
    return ''.join(chunks)


ROWS = [
    ("Subject", "Yousef Mohammed"),
    ("Role", "IoT & AI Developer"),
    ("Origin", "Cairo, Egypt"),
    ("Education", "Helwan National University"),
    ("Status", "Building + Learning + Shipping"),
    ("ToolChain", "VS Code · Git · Linux · Arduino"),
    ("Core.Lang", "Python · C++"),
    ("Core.AI", "PyTorch · TensorFlow · NLP"),
    ("Core.Embedded", "ESP32 · Arduino"),
    ("Core.Data", "Pandas · NumPy · Scikit-learn"),
    ("Core.Vision", "OpenCV"),
    ("Grid.Mail", "ym9159303@gmail.com"),
    ("Grid.Portfolio", "Coming soon"),
    ("Grid.LinkedIn", "linkedin.com/in/joox"),
    ("Grid.GitHub", "github.com/yousefJooX"),
    ("Grid.Instagram", "instagram.com/joox.cmd"),
]


def info_panel(text: str, muted: str, chrome: str, accent: str) -> str:
    chunks = [f'<text x="480" y="145" fill="{chrome}" font-size="13" font-weight="700" letter-spacing="2">SYSTEM.INFO</text>',
              f'<rect x="1040" y="126" width="73" height="25" rx="12" fill="#EF4444" opacity=".16"/><circle cx="1054" cy="138.5" r="4" fill="#EF4444"><animate attributeName="opacity" values="1;.25;1" dur="1.4s" repeatCount="indefinite"/></circle><text x="1064" y="143" fill="#EF4444" font-size="12" font-weight="700">LIVE</text>',
              f'<rect x="480" y="158" width="164" height="29" rx="14.5" fill="{chrome}" opacity=".14"/><text x="496" y="178" fill="{chrome}" font-size="14" font-weight="700">@yousefJooX</text>']
    y = 211
    for label_text, value in ROWS:
        label_safe = html.escape(label_text)
        value_safe = html.escape(value)
        chunks.append(f'<text x="480" y="{y}" fill="{muted}" font-size="14">{label_safe}</text>')
        chunks.append(f'<line x1="{610 if len(label_text) < 11 else 640}" y1="{y-4}" x2="777" y2="{y-4}" stroke="{muted}" stroke-opacity=".32" stroke-dasharray="2 5"/>')
        chunks.append(f'<text x="1107" y="{y}" text-anchor="end" textLength="310" lengthAdjust="spacingAndGlyphs" fill="{text}" font-size="14">{value_safe}</text>')
        y += 23
    return ''.join(chunks)


def make_svg(bits: np.ndarray, mode: str) -> str:
    dark = mode == "dark"
    bg = "#0A101F" if dark else "#F8FAFC"
    panel = "#101827" if dark else "#FFFFFF"
    border = "#25324A" if dark else "#CBD5E1"
    text = "#F8FAFC" if dark else "#0F172A"
    muted = "#94A3B8" if dark else "#475569"
    portrait = "#A78BFA" if dark else "#7C3AED"
    chrome = "#22D3EE" if dark else "#0891B2"
    accent = "#10B981"
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1180" height="610" viewBox="0 0 1180 610">
<rect width="1180" height="610" rx="24" fill="{bg}"/>
<rect x="25" y="25" width="1130" height="560" rx="18" fill="{panel}" stroke="{border}" stroke-width="2"/>
<rect x="25" y="25" width="1130" height="54" rx="18" fill="{border}" opacity=".38"/>
<circle cx="54" cy="52" r="7" fill="#EF4444"/><circle cx="78" cy="52" r="7" fill="#F59E0B"/><circle cx="102" cy="52" r="7" fill="#10B981"/>
<text x="590" y="58" text-anchor="middle" fill="{muted}" font-size="14" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">profile.sh --live</text>
<rect x="58" y="110" width="365" height="438" rx="13" fill="{bg}" stroke="{border}"/>
<text x="82" y="143" fill="{chrome}" font-size="13" font-weight="700" letter-spacing="2" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">VISUAL.MAP</text>
<line x1="454" y1="109" x2="454" y2="549" stroke="{border}"/>
{portrait_layers(bits, portrait)}
{travellers(accent)}
<rect x="67" y="520" width="346" height="17" rx="8.5" fill="{border}" opacity=".42"/>
<rect x="67" y="520" width="252" height="17" rx="8.5" fill="{chrome}" opacity=".65"><animate attributeName="width" values="98;252;330;252" dur="5s" repeatCount="indefinite"/></rect>
<g font-family="ui-monospace,SFMono-Regular,Menlo,monospace">{info_panel(text, muted, chrome, accent)}</g>
<text x="58" y="570" fill="{muted}" font-size="11" font-family="ui-monospace,SFMono-Regular,Menlo,monospace">PYTHON · PYTORCH · ESP32 // CAIRO, EG</text>
</svg>'''


def main() -> None:
    dark, light = prepare_portrait()
    (OUT / "dark.svg").write_text(make_svg(dark, "dark"), encoding="utf-8")
    (OUT / "light.svg").write_text(make_svg(light, "light"), encoding="utf-8")
    print("Generated", OUT / "dark.svg", OUT / "light.svg")


if __name__ == "__main__":
    main()
