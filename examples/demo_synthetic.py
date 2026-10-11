"""Run the method on a synthetic scene and save LAZ + top-view figures.

    python examples/demo_synthetic.py --out-dir out/

Requires matplotlib for the figures.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from als_flood3d import FloodRiskConfig, compute_flood_risk
from als_flood3d.io import PointCloud, write_enriched
from als_flood3d.synthetic import make_scene

ap = argparse.ArgumentParser()
ap.add_argument("--out-dir", default="out")
ap.add_argument("--density", type=float, default=2.0)
args = ap.parse_args()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)

scene = make_scene(density=args.density)
res = compute_flood_risk(scene.xyz, scene.classification, FloodRiskConfig(r1=100.0, r2=4.0))
write_enriched(out / "synthetic_enriched.laz", PointCloud(scene.xyz, scene.classification), res.as_fields())

# Paper visualisation: ACU colour gradient capped at 10 m (low = accumulation risk).
order = np.argsort(scene.xyz[:, 2])  # draw higher points on top, like a top view
fig, axes = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
for ax, (name, values, cmap, vmax) in zip(
    axes,
    [("Water accumulation ACU [m] (capped at 10 m)", res.acu, "Blues_r", 10.0),
     ("Runoff potential RUN [-]", res.run, "inferno", 1.0)],
):
    sc = ax.scatter(*scene.xyz[order, :2].T, c=values[order], s=0.3, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(name)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, shrink=0.8)
fig.savefig(out / "synthetic_topview.png", dpi=150)

print(f"Wrote {out / 'synthetic_enriched.laz'} and {out / 'synthetic_topview.png'}")
for label in np.unique(scene.label):
    m = scene.label == label
    print(f"  {label:<11} ACU median={np.nanmedian(res.acu[m]):6.2f} m   RUN median={np.nanmedian(res.run[m]):.3f}")
