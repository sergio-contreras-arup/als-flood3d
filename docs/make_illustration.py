"""Generate docs/img/ilustracion.svg: isometric sketch of what als-flood3d computes.

The ground "point cloud" is a toy terrain with a valley; its colour is the real
Eq. 1 (ACU) evaluated on those points. Roofs use RUN colours (sloped = high,
flat = low). Run: python docs/make_illustration.py
"""

from math import cos, exp, pi, radians, sin
from pathlib import Path

import numpy as np

OUT = Path(__file__).parent / "img" / "ilustracion.svg"
W, H = 1200, 720
S = 22.0  # px per world unit
ZS = 1.0  # vertical exaggeration
OX, OY = 450.0, 170.0
C30, S30 = cos(radians(30)), sin(radians(30))

BG, PANEL, LINE, TEXT, MUTED = "#0f1216", "#171b21", "#2b313b", "#e6e9ee", "#9aa3af"
ACU_RAMP = ["#08306b", "#1561a8", "#4292c6", "#9ecae1", "#d6e6f2"]  # 0 m -> cap
RUN_HIGH, RUN_MID, RUN_LOW = "#fca50a", "#dd513a", "#420a68"
GROUND_SAFE = "#6b6f76"
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def P(x, y, z):
    return OX + (x - y) * C30 * S, OY + (x + y) * S30 * S - z * ZS * S


def pts(seq):
    return " ".join(f"{a:.1f},{b:.1f}" for a, b in seq)


def lerp_hex(stops, t):
    t = min(max(t, 0.0), 1.0) * (len(stops) - 1)
    i = min(int(t), len(stops) - 2)
    f = t - i
    a = [int(stops[i][k : k + 2], 16) for k in (1, 3, 5)]
    b = [int(stops[i + 1][k : k + 2], 16) for k in (1, 3, 5)]
    return "#" + "".join(f"{round(a[k] + (b[k] - a[k]) * f):02x}" for k in range(3))


# ---------------------------------------------------------------- terrain
def valley_x(y):
    return 12.0 + 1.4 * sin(y / 3.2)


def terrain(x, y):
    d = x - valley_x(y)
    return 1.3 + 0.015 * y - 2.5 * exp(-(d * d) / (2 * 1.9**2))


BUILDINGS = [  # x0, y0, w, d, h, ridge (0 = flat)
    (1.2, 1.6, 3.4, 3.0, 3.4, 0.0),
    (1.4, 7.6, 3.0, 4.0, 2.3, 1.4),
    (5.2, 14.2, 2.6, 2.6, 5.6, 0.0),
    (5.6, 3.2, 2.0, 2.4, 1.9, 1.0),
    (16.6, 12.2, 3.0, 3.2, 2.2, 1.3),
    (17.0, 6.6, 2.6, 2.8, 2.6, 0.0),
    (16.2, 17.0, 2.6, 2.2, 2.0, 1.1),
]
TREES = [(8.6, 6.2), (8.9, 10.8), (15.0, 8.4), (14.6, 15.6), (9.0, 17.4), (2.6, 18.0)]


def in_footprint(x, y, pad=0.15):
    return any(x0 - pad <= x <= x0 + w + pad and y0 - pad <= y <= y0 + d + pad for x0, y0, w, d, *_ in BUILDINGS)


step = 0.5
gx, gy = np.meshgrid(np.arange(0, 20.01, step), np.arange(0, 20.01, step))
gx, gy = gx.ravel(), gy.ravel()
gz = np.array([terrain(a, b) for a, b in zip(gx, gy)])

# Eq. 1 on the toy cloud: ACU = z - min z within r1 (XY)
R1 = 12.0  # schematic r1: large enough for every point to see the valley floor
d2 = (gx[:, None] - gx[None, :]) ** 2 + (gy[:, None] - gy[None, :]) ** 2
acu = gz - np.where(d2 <= R1**2, gz[None, :], np.inf).min(axis=1)
CAP = 1.4

prims = []  # (depth, svg)

for x, y, z, a in zip(gx, gy, gz, acu):
    if in_footprint(x, y):
        continue
    sx, sy = P(x, y, z)
    col = lerp_hex(ACU_RAMP, a / CAP) if a < CAP else GROUND_SAFE
    r = 2.3 if a < CAP else 1.9
    prims.append((x + y, f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{r}" fill="{col}"/>'))


def poly(points3d, fill, extra=""):
    return f'<polygon points="{pts(P(*p) for p in points3d)}" fill="{fill}" {extra}/>'


def building_svg(x0, y0, w, d, h, ridge):
    base = min(terrain(x0 + i * w, y0 + j * d) for i in (0, 1) for j in (0, 1)) - 0.1
    top = base + h
    x1, y1 = x0 + w, y0 + d
    out = []
    right = [(x1, y0, base), (x1, y1, base), (x1, y1, top), (x1, y0, top)]
    front = [(x0, y1, base), (x1, y1, base), (x1, y1, top), (x0, y1, top)]
    if ridge:
        xm = x0 + w / 2
        out.append(poly([(x0, y0, top), (xm, y0, top + ridge), (xm, y1, top + ridge), (x0, y1, top)], RUN_MID))
        out.append(poly([(xm, y0, top + ridge), (x1, y0, top), (x1, y1, top), (xm, y1, top + ridge)], RUN_HIGH))
        out.append(poly(right, "#8e959f"))
        out.append(poly(front + [(xm, y1, top + ridge)], "#b9bfc7"))
        out.append(poly(front + [(xm, y1, top + ridge)], "url(#dots)"))
        # runoff arrows down the visible roof plane
        for t in (0.3, 0.7):
            yy = y0 + t * d
            a, b = P(xm + 0.25, yy, top + ridge - 0.25 * ridge / (w / 2)), P(x1 - 0.2, yy, top + 0.2 * ridge / (w / 2))
            out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                       f'stroke="#fff4d6" stroke-width="2" marker-end="url(#arrow-light)"/>')
    else:
        out.append(poly(right, "#8e959f"))
        out.append(poly(front, "#b9bfc7"))
        out.append(poly(front, "url(#dots)"))
        out.append(poly([(x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)], RUN_LOW))
        out.append(poly([(x0, y0, top), (x1, y0, top), (x1, y1, top), (x0, y1, top)], "url(#dots)"))
    out.append(poly(right, "url(#dots)"))
    return x1 + y1, "\n".join(out)


for b in BUILDINGS:
    prims.append(building_svg(*b))

rng = np.random.default_rng(4)
for tx, ty in TREES:
    tz = terrain(tx, ty)
    trunk_a, trunk_b = P(tx, ty, tz), P(tx, ty, tz + 1.2)
    parts = [f'<line x1="{trunk_a[0]:.1f}" y1="{trunk_a[1]:.1f}" x2="{trunk_b[0]:.1f}" y2="{trunk_b[1]:.1f}" stroke="#5b4636" stroke-width="2.5"/>']
    v = rng.normal(size=(70, 3))
    v *= (rng.random(70) ** (1 / 3) * 0.95 / np.linalg.norm(v, axis=1))[:, None]
    v = v[np.argsort(v[:, 0] + v[:, 1])]
    for dx, dy, dz in v:
        sx, sy = P(tx + dx, ty + dy, tz + 2.0 + dz * 1.1)
        shade = lerp_hex(["#1f5b2a", "#2e7d32", "#5fa463"], (dz + 1) / 2)
        parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="2.4" fill="{shade}"/>')
    prims.append((tx + ty + 1.0, "\n".join(parts)))

# runoff paths on the ground: follow the terrain towards the valley
for sx0, sy0 in [(8.6, 8.4), (8.9, 13.4), (16.0, 8.0), (16.4, 15.4)]:
    path, x, y = [], sx0, sy0
    for _ in range(14):
        path.append(P(x, y, terrain(x, y) + 0.05))
        gxd = (terrain(x + 0.05, y) - terrain(x - 0.05, y)) / 0.1
        gyd = (terrain(x, y + 0.05) - terrain(x, y - 0.05)) / 0.1
        n = (gxd**2 + gyd**2) ** 0.5 or 1
        x, y = x - 0.32 * gxd / n, y - 0.32 * gyd / n
        if abs(x - valley_x(y)) < 0.6:
            break
    d = "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in path)
    prims.append((sx0 + sy0 + 3, f'<path d="{d}" fill="none" stroke="{RUN_HIGH}" stroke-width="2.6" '
                                 f'stroke-linecap="round" marker-end="url(#arrow)"/>'))

prims.sort(key=lambda p: p[0])

# ---------------------------------------------------------------- ACU annotation
px, py = 9.0, 0.6
pz = terrain(px, py)
near = (gx - px) ** 2 + (gy - py) ** 2 <= 4.5**2  # nearest part of the valley floor, for a readable sketch
k = int(np.argmin(np.where(near, gz, np.inf)))
mx, my, mz = gx[k], gy[k], gz[k]
ARC_R = 3.6
ring = [P(px + ARC_R * cos(t), py + ARC_R * sin(t), pz) for t in np.linspace(-0.9, 2.4, 40)]
pa, pm = P(px, py, pz), P(mx, my, mz)
pm_up = P(mx, my, pz)
annot = f"""
<polyline points="{pts(ring)}" fill="none" stroke="#7fb3ff" stroke-width="2" stroke-dasharray="7 5"/>
<line x1="{pa[0]:.1f}" y1="{pa[1]:.1f}" x2="{pm_up[0]:.1f}" y2="{pm_up[1]:.1f}" stroke="#4f9cf9" stroke-width="1" stroke-dasharray="3 3"/>
<line x1="{pm_up[0]:.1f}" y1="{pm_up[1]:.1f}" x2="{pm[0]:.1f}" y2="{pm[1]:.1f}" stroke="#ffffff" stroke-width="2" marker-start="url(#tick)" marker-end="url(#tick)"/>
<circle cx="{pa[0]:.1f}" cy="{pa[1]:.1f}" r="5" fill="#ffffff" stroke="#0f1216" stroke-width="2"/>
<circle cx="{pm[0]:.1f}" cy="{pm[1]:.1f}" r="5" fill="#08306b" stroke="#ffffff" stroke-width="2"/>
<text x="{pa[0]:.1f}" y="{pa[1] - 12:.1f}" text-anchor="middle" font-size="13" fill="{TEXT}" font-weight="600">P</text>
<text x="{pm_up[0] + 10:.1f}" y="{(pm_up[1] + pm[1]) / 2 + 4:.1f}" font-size="13" fill="{TEXT}" stroke="#0f1216" stroke-width="4" stroke-linejoin="round" paint-order="stroke" font-family="{MONO}">ACU = z − z<tspan baseline-shift="sub" font-size="10">min</tspan></text>
"""
ring_label = ring[20]

# ---------------------------------------------------------------- rain
rain = []
for _ in range(80):
    x = rng.uniform(70, 840)
    y = rng.uniform(150, 300)
    rain.append(f'<line x1="{x:.0f}" y1="{y:.0f}" x2="{x - 5:.0f}" y2="{y + 16:.0f}"/>')


# ---------------------------------------------------------------- markers + key
def marker(world, letter, dx=0, dy=0):
    sx, sy = P(*world)
    sx, sy = sx + dx, sy + dy
    return (f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="11" fill="#ffffff" stroke="{BG}" stroke-width="2"/>'
            f'<text x="{sx:.1f}" y="{sy + 4.5:.1f}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{BG}">{letter}</text>')


b_gable, b_flat = BUILDINGS[6], BUILDINGS[2]
t_veg = TREES[5]
vy = 17.5
KEY = [
    ("A", "Vaguada / cauce", "mínimo local del terreno → ACU ≈ 0", "#4292c6",
     (valley_x(vy), vy, terrain(valley_x(vy), vy)), (0, 0)),
    ("B", "Tejado inclinado", "plano y con pendiente → RUN alto", RUN_HIGH,
     (b_gable[0] + b_gable[2] * 0.75, b_gable[1] + b_gable[3] * 0.5, terrain(b_gable[0], b_gable[1]) + b_gable[4] + 0.4), (0, -4)),
    ("C", "Cubierta plana", "verticality ≈ 0 → RUN bajo", "#7b4bb0",
     (b_flat[0] + b_flat[2] / 2, b_flat[1] + b_flat[3] / 2, terrain(b_flat[0], b_flat[1]) + b_flat[4]), (0, 0)),
    ("D", "Calles y taludes", "superficies continuas → vías de escorrentía", RUN_HIGH,
     (8.6, 8.4, terrain(8.6, 8.4)), (-14, -6)),
    ("E", "Vegetación", "scattering alto → frena y dispersa el agua", "#2e7d32",
     (t_veg[0], t_veg[1], terrain(*t_veg) + 3.2), (0, 0)),
]
markers = "\n".join(marker(w, l, *off) for l, _, _, _, w, off in KEY)
key_items = []
for i, (l, t, body, col, _, _) in enumerate(KEY):
    y = 300 + i * 62
    key_items.append(
        f'<circle cx="880" cy="{y}" r="11" fill="#ffffff"/>'
        f'<text x="880" y="{y + 4.5}" text-anchor="middle" font-size="12.5" font-weight="700" fill="{BG}">{l}</text>'
        f'<rect x="900" y="{y - 9}" width="4" height="34" rx="2" fill="{col}"/>'
        f'<text x="914" y="{y + 1}" font-size="14" font-weight="600" fill="{TEXT}">{t}</text>'
        f'<text x="914" y="{y + 20}" font-size="12" fill="{MUTED}">{body}</text>')
callouts = markers + "\n" + "\n".join(key_items)


# ---------------------------------------------------------------- legend
def bar(x, y, stops, w=220):
    gid = f"g{abs(hash(tuple(stops))) % 10**6}"
    st = "".join(f'<stop offset="{i / (len(stops) - 1):.2f}" stop-color="{c}"/>' for i, c in enumerate(stops))
    return (f'<defs><linearGradient id="{gid}">{st}</linearGradient></defs>'
            f'<rect x="{x}" y="{y}" width="{w}" height="10" rx="3" fill="url(#{gid})"/>')


legend = f"""
<g font-family="{FONT}">
  <text x="70" y="62" font-size="26" font-weight="700" fill="{TEXT}">als-flood3d</text>
  <text x="70" y="88" font-size="14" fill="{MUTED}">Riesgo por lluvia extrema calculado directamente sobre la nube LiDAR</text>
  <text x="70" y="107" font-size="14" fill="{MUTED}">Balado &amp; Biljecki (2026) · sin DTM ni malla intermedios</text>

  <g transform="translate(860,48)">
    <rect width="300" height="200" rx="10" fill="{PANEL}" stroke="{LINE}"/>
    <text x="20" y="30" font-size="13" font-weight="700" fill="{TEXT}">Acumulación · ACU</text>
    <text x="20" y="49" font-size="12" fill="{MUTED}" font-family="{MONO}">z − min{{z suelo, r1 = 100 m}}</text>
    {bar(20, 60, ACU_RAMP[::-1][::-1], 260)}
    <text x="20" y="86" font-size="11" fill="{MUTED}">0 m · acumula</text>
    <text x="280" y="86" font-size="11" fill="{MUTED}" text-anchor="end">alto · seguro</text>
    <text x="20" y="122" font-size="13" font-weight="700" fill="{TEXT}">Escorrentía · RUN</text>
    <text x="20" y="141" font-size="12" fill="{MUTED}" font-family="{MONO}">vert^0.1 · plan · |scat − 1|</text>
    {bar(20, 152, ["#000004", "#420a68", "#932667", "#dd513a", "#fca50a", "#fcffa4"], 260)}
    <text x="20" y="178" font-size="11" fill="{MUTED}">bajo</text>
    <text x="280" y="178" font-size="11" fill="{MUTED}" text-anchor="end">alto</text>
  </g>
</g>
"""

steps = [("1", "Nube ALS clasificada", "LAS/LAZ · suelo = clase 2"),
         ("2", "ACU (Ec. 1) + RUN (Ec. 2–7)", "por punto, sobre la propia nube"),
         ("3", "LAZ enriquecido + visor 3D", "CloudCompare / QGIS / HTML")]
flow = [f'<g font-family="{FONT}">']
bx, bw, gap = 70, 330, 35
for i, (n, t, s) in enumerate(steps):
    x = bx + i * (bw + gap)
    flow.append(f'<rect x="{x}" y="632" width="{bw}" height="56" rx="10" fill="{PANEL}" stroke="{LINE}"/>')
    flow.append(f'<circle cx="{x + 28}" cy="660" r="13" fill="#4f9cf9"/>'
                f'<text x="{x + 28}" y="665" text-anchor="middle" font-size="13" font-weight="700" fill="#0b1a2e">{n}</text>')
    flow.append(f'<text x="{x + 52}" y="656" font-size="14" font-weight="600" fill="{TEXT}">{t}</text>')
    flow.append(f'<text x="{x + 52}" y="675" font-size="12" fill="{MUTED}">{s}</text>')
    if i < 2:
        flow.append(f'<line x1="{x + bw + 6}" y1="660" x2="{x + bw + gap - 6}" y2="660" stroke="{MUTED}" '
                    f'stroke-width="2" marker-end="url(#arrow-muted)"/>')
flow.append("</g>")

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img"
  aria-label="Ilustración isométrica de una ciudad: el agua se acumula en la vaguada (ACU) y escurre por tejados inclinados y calles (RUN)">
<title>als-flood3d: acumulación (ACU) y escorrentía (RUN) sobre una nube LiDAR urbana</title>
<defs>
  <pattern id="dots" width="5" height="5" patternUnits="userSpaceOnUse">
    <circle cx="2.5" cy="2.5" r="0.9" fill="#ffffff" fill-opacity="0.28"/>
  </pattern>
  <marker id="arrow" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{RUN_HIGH}"/></marker>
  <marker id="arrow-light" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="#fff4d6"/></marker>
  <marker id="arrow-muted" viewBox="0 0 10 10" refX="7" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>
  <marker id="tick" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="8" markerHeight="8" orient="auto">
    <path d="M5,0 L5,10" stroke="#ffffff" stroke-width="2"/></marker>
</defs>
<rect width="{W}" height="{H}" rx="16" fill="{BG}"/>
<g stroke="#7fb3ff" stroke-opacity="0.35" stroke-width="1.2" stroke-linecap="round">{''.join(rain)}</g>
<g>{''.join(s for _, s in prims)}</g>
<g font-family="{FONT}">{annot}
<text x="{ring_label[0] - 8:.1f}" y="{ring_label[1] + 18:.1f}" text-anchor="end" font-size="12" fill="#7fb3ff" stroke="#0f1216" stroke-width="4" stroke-linejoin="round" paint-order="stroke" font-family="{MONO}">r1 = 100 m (XY)</text>
</g>
<g font-family="{FONT}">{callouts}</g>
{legend}
{''.join(flow)}
</svg>
"""
OUT.write_text(svg, encoding="utf-8")
print(f"{OUT} ({len(svg) / 1024:.0f} KB)")
