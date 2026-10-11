"""Command line entry point: ``als-flood3d input.laz output.laz --r2 2``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from .io import read_point_clouds, rgb_of, write_enriched
from .pipeline import FloodRiskConfig, compute_flood_risk

# r2 used in the paper for each dataset density (Table 2).
R2_PRESETS = {"vigo": 10.0, "hongkong": 5.0, "rotterdam": 2.0}


def _summary(name: str, v: np.ndarray) -> str:
    f = v[np.isfinite(v)]
    if f.size == 0:
        return f"  {name:<12} all NaN"
    p = np.percentile(f, [5, 50, 95])
    return f"  {name:<12} min={f.min():8.3f}  p5={p[0]:8.3f}  p50={p[1]:8.3f}  p95={p[2]:8.3f}  max={f.max():8.3f}  NaN={v.size - f.size}"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description="3D extreme rainfall risk mapping on ALS point clouds (Balado & Biljecki, 2026)."
    )
    ap.add_argument("input", nargs="+", help="LAS/LAZ (classified) or ASCII x y z classification; several tiles are merged")
    ap.add_argument("output", help="LAS/LAZ (extra dims) or .txt/.csv")
    ap.add_argument("--r1", type=float, default=100.0, help="XY radius for water accumulation [m] (default 100)")
    ap.add_argument("--r2", type=float, default=None, help="3D radius for geometric features [m] (default 2)")
    ap.add_argument("--preset", choices=sorted(R2_PRESETS), help="use the paper's r2 for a density similar to this dataset")
    ap.add_argument("--grid", type=float, default=10.0, help="ground subsampling cell [m] (default 10)")
    ap.add_argument("--ground-class", type=int, nargs="+", default=[2], help="ground class codes (default 2)")
    ap.add_argument("--acu-method", choices=["grid", "kdtree"], default="grid")
    ap.add_argument("--voxel", type=float, default=None,
                    help="thin the input to one point per voxel of this size [m] (for clouds much denser than ~20 pt/m2)")
    ap.add_argument("--html", default=None, help="also write a self-contained 3D HTML viewer")
    ap.add_argument("--html-points", type=int, default=2_000_000, help="max points embedded in the viewer")
    ap.add_argument("--workers", type=int, default=-1)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(message)s")

    r2 = args.r2 if args.r2 is not None else R2_PRESETS.get(args.preset, 2.0)
    cfg = FloodRiskConfig(
        r1=args.r1,
        r2=r2,
        grid_cell=args.grid,
        ground_classes=tuple(args.ground_class),
        acu_method=args.acu_method,
        workers=args.workers,
    )

    cloud = read_point_clouds(args.input, voxel=args.voxel)
    res = compute_flood_risk(cloud.xyz, cloud.classification, cfg)
    write_enriched(args.output, cloud, res.as_fields())

    print(f"{len(cloud.xyz):,} points -> {args.output}  (r1={cfg.r1} m, r2={cfg.r2} m, grid={cfg.grid_cell} m)")
    for k, v in res.as_fields().items():
        print(_summary(k, v))
    print("  " + "  ".join(f"{k}={v:.1f}s" for k, v in res.timings.items()))

    if args.html:
        from .viewer import export_html

        info = {
            "Ficheros": ", ".join(Path(p).name for p in args.input),
            "r1 (ACU)": f"{cfg.r1:g} m",
            "r2 (features)": f"{cfg.r2:g} m",
            "Malla suelo": f"{cfg.grid_cell:g} m",
            "Vóxel": f"{args.voxel:g} m" if args.voxel else "—",
        }
        title = Path(args.html).stem.replace("_", " ")
        n = export_html(args.html, cloud.xyz, res.acu, res.run, cloud.classification, rgb_of(cloud),
                        max_points=args.html_points, title=title, info=info)
        print(f"Viewer: {args.html} ({n:,} points)")


if __name__ == "__main__":
    main()
