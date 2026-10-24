"""Self-contained HTML 3D viewer for enriched point clouds (three.js, WebGL).

The cloud is randomly subsampled to ``max_points``, quantised and embedded in
the HTML as base64, so the file opens with a double click (three.js is loaded
from a CDN, so an internet connection is required).

Binary layout (little endian), n = number of points:
    pos  uint16[n*3]  XYZ quantised over the bounding box
    acu  uint16[n]    ACU in centimetres, 65535 = NaN
    rgb  uint8[n*3]   colour (zeros if absent)
    cls  uint8[n]     ASPRS class (0 if absent)
    run  uint8[n]     RUN * 254, 255 = NaN
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import json
import re
from importlib import resources
from pathlib import Path

import numpy as np

ACU_NAN = 65535
RUN_NAN = 255


def _pack(xyz, acu, run, classification, rgb) -> tuple[bytes, dict]:
    n = len(xyz)
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    pos = np.round((xyz - lo) / span * 65535).astype("<u2")

    acu_q = np.full(n, ACU_NAN, dtype="<u2")
    ok = np.isfinite(acu)
    acu_q[ok] = np.clip(np.round(acu[ok] * 100), 0, ACU_NAN - 1)

    run_q = np.full(n, RUN_NAN, dtype=np.uint8)
    ok = np.isfinite(run)
    run_q[ok] = np.clip(np.round(run[ok] * 254), 0, 254)

    rgb = np.zeros((n, 3), np.uint8) if rgb is None else rgb.astype(np.uint8)
    cls = np.zeros(n, np.uint8) if classification is None else classification.astype(np.uint8)

    blob = b"".join(a.tobytes() for a in (pos, acu_q, rgb, cls, run_q))
    meta = {"n": n, "min": lo.tolist(), "max": hi.tolist(), "hasRgb": bool(rgb.any())}
    return blob, meta


def export_html(
    path: str | Path,
    xyz: np.ndarray,
    acu: np.ndarray,
    run: np.ndarray,
    classification: np.ndarray | None = None,
    rgb: np.ndarray | None = None,
    max_points: int = 2_000_000,
    title: str = "Riesgo por lluvia extrema",
    info: dict | None = None,
    seed: int = 0,
) -> int:
    """Write the viewer and return the number of points embedded."""
    n_total = len(xyz)
    idx = np.arange(n_total)
    if n_total > max_points:
        idx = np.sort(np.random.default_rng(seed).choice(n_total, max_points, replace=False))
    take = lambda a: None if a is None else np.asarray(a)[idx]  # noqa: E731

    blob, meta = _pack(np.asarray(xyz, np.float64)[idx], take(acu), take(run), take(classification), take(rgb))
    finite_run = run[np.isfinite(run)]
    meta.update(
        title=title,
        nTotal=int(n_total),
        info=info or {},
        runRange=[float(np.percentile(finite_run, 2)), float(np.percentile(finite_run, 99.5))] if finite_run.size else [0, 1],
    )

    template = resources.files("als_flood3d").joinpath("viewer_template.html").read_text(encoding="utf-8")
    # Escape at generation time so a malicious string in title/meta (e.g. an
    # input file name) cannot inject markup or break out of the JSON <script> block.
    meta_json = json.dumps(meta).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    subs = {
        "__TITLE__": html_mod.escape(title),
        "__META__": meta_json,
        "__DATA__": base64.b64encode(blob).decode("ascii"),
    }
    # Single pass so a value equal to a placeholder token (e.g. a file named
    # "__DATA__") cannot be rewritten by a later substitution.
    rendered = re.sub(r"__TITLE__|__META__|__DATA__", lambda m: subs[m.group(0)], template)
    Path(path).write_text(rendered, encoding="utf-8")
    return len(idx)


def main(argv: list[str] | None = None) -> None:
    """``als-flood3d-view enriched.laz viewer.html``: viewer from an already enriched LAS/LAZ."""
    from .io import read_point_cloud, rgb_of

    ap = argparse.ArgumentParser(description="Build an HTML 3D viewer from an enriched LAS/LAZ (ACU/RUN extra dims).")
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--max-points", type=int, default=2_000_000)
    ap.add_argument("--title", default=None)
    args = ap.parse_args(argv)

    cloud = read_point_cloud(args.input)
    if cloud.las is None or "ACU" not in cloud.las.point_format.dimension_names:
        raise SystemExit("Input must be a LAS/LAZ produced by als-flood3d (with ACU and RUN dimensions).")
    n = export_html(
        args.output,
        cloud.xyz,
        np.asarray(cloud.las["ACU"], np.float64),
        np.asarray(cloud.las["RUN"], np.float64),
        cloud.classification,
        rgb_of(cloud),
        max_points=args.max_points,
        title=args.title or Path(args.input).stem,
    )
    print(f"{n:,} of {len(cloud.xyz):,} points -> {args.output}")


if __name__ == "__main__":
    main()
