"""Synthetic classified ALS-like scene for testing and demos.

Contains: gently sloped ground with a 5 m deep depression, a flat-roof building,
a gable-roof building and a tree. Each element carries a label so tests can
check the expected behaviour of ACU and RUN.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GROUND, VEGETATION, BUILDING = 2, 5, 6  # ASPRS classes


@dataclass
class SyntheticScene:
    xyz: np.ndarray
    classification: np.ndarray
    label: np.ndarray  # element name per point

    DEPRESSION_CENTER = (300.0, 300.0)


def _ground_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    cx, cy = SyntheticScene.DEPRESSION_CENTER
    slope = 0.01 * x
    depression = -5.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * 25.0**2))
    return slope + depression


def _box_walls(rng, x0, y0, w, h, z_base, z_top, density):
    pts = []
    for (xa, ya), (xb, yb) in [((x0, y0), (x0 + w, y0)), ((x0 + w, y0), (x0 + w, y0 + h)),
                               ((x0 + w, y0 + h), (x0, y0 + h)), ((x0, y0 + h), (x0, y0))]:
        length = np.hypot(xb - xa, yb - ya)
        n = int(length * (z_top - z_base) * density)
        t, z = rng.random(n), rng.uniform(z_base, z_top, n)
        pts.append(np.column_stack([xa + t * (xb - xa), ya + t * (yb - ya), z]))
    return np.vstack(pts)


def make_scene(size: float = 400.0, density: float = 2.0, noise: float = 0.03, seed: int = 0) -> SyntheticScene:
    rng = np.random.default_rng(seed)
    parts: list[tuple[np.ndarray, int, str]] = []

    # Ground (excluding building footprints)
    n = int(size * size * density)
    gx, gy = rng.uniform(0, size, n), rng.uniform(0, size, n)
    flat_fp = (gx > 80) & (gx < 120) & (gy > 80) & (gy < 120)
    gable_fp = (gx > 80) & (gx < 120) & (gy > 280) & (gy < 320)
    keep = ~(flat_fp | gable_fp)
    gx, gy = gx[keep], gy[keep]
    parts.append((np.column_stack([gx, gy, _ground_z(gx, gy)]), GROUND, "ground"))

    # Flat-roof building, 40x40 m, 15 m high
    base = float(_ground_z(np.array([100.0]), np.array([100.0]))[0])
    n = int(40 * 40 * density)
    rx, ry = rng.uniform(80, 120, n), rng.uniform(80, 120, n)
    parts.append((np.column_stack([rx, ry, np.full(n, base + 15.0)]), BUILDING, "flat_roof"))
    parts.append((_box_walls(rng, 80, 80, 40, 40, base, base + 15.0, density * 0.3), BUILDING, "wall"))

    # Gable-roof building, ridge along y, eaves 10 m, ridge 20 m (45 deg roof)
    base = float(_ground_z(np.array([100.0]), np.array([300.0]))[0])
    rx, ry = rng.uniform(80, 120, n), rng.uniform(280, 320, n)
    rz = base + 10.0 + (20.0 - np.abs(rx - 100.0)) * 0.5
    parts.append((np.column_stack([rx, ry, rz]), BUILDING, "gable_roof"))

    # Tree: noisy sphere of foliage
    n = 1500
    v = rng.normal(size=(n, 3))
    v *= (rng.random(n) ** (1 / 3) * 5.0 / np.linalg.norm(v, axis=1))[:, None]
    tz = float(_ground_z(np.array([250.0]), np.array([100.0]))[0])
    parts.append((v + [250.0, 100.0, tz + 9.0], VEGETATION, "tree"))

    xyz = np.vstack([p for p, _, _ in parts])
    xyz += rng.normal(scale=noise, size=xyz.shape)
    cls = np.concatenate([np.full(len(p), c) for p, c, _ in parts])
    label = np.concatenate([np.full(len(p), name) for p, _, name in parts])
    return SyntheticScene(xyz=xyz, classification=cls, label=label)
