"""Water accumulation indicator (paper Section 3.1, Eq. 1).

ACU(x, y, z) = z - min{ z_i : sqrt((x_i - x)^2 + (y_i - y)^2) <= r1 }

Following the paper's optimisations:
  * the neighbourhood search is done in the XY plane only,
  * only ground points are candidates for the minimum,
  * the ground is subsampled on a regular grid (10 m by default).

Low ACU values mean the point lies close to the lowest ground elevation of its
surroundings, i.e. a likely water accumulation zone.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def ground_grid_subsample(ground_xyz: np.ndarray, cell: float = 10.0) -> np.ndarray:
    """Keep the lowest ground point of every ``cell`` x ``cell`` XY cell.

    Keeping the lowest point (instead of the centroid) preserves the minima that
    Eq. 1 looks for.
    """
    ground_xyz = np.asarray(ground_xyz, dtype=np.float64)
    if ground_xyz.size == 0:
        return ground_xyz.reshape(0, 3)
    ij = np.floor((ground_xyz[:, :2] - ground_xyz[:, :2].min(axis=0)) / cell).astype(np.int64)
    key = ij[:, 0] * (ij[:, 1].max() + 1) + ij[:, 1]
    # Sort by cell, then by z: the first point of each cell is its lowest one.
    order = np.lexsort((ground_xyz[:, 2], key))
    first = np.ones(order.size, dtype=bool)
    first[1:] = key[order][1:] != key[order][:-1]
    return ground_xyz[order[first]]


def _disk_footprint(radius_cells: float) -> np.ndarray:
    r = int(np.ceil(radius_cells))
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (xx**2 + yy**2) <= radius_cells**2


def _acu_grid(xyz: np.ndarray, ground_xyz: np.ndarray, r1: float, cell: float) -> np.ndarray:
    """Raster approximation: min-filter with a disk footprint over a ground-min grid.

    The ground minimum is rasterised at ``cell`` resolution (equivalent to the
    paper's 10 m subsampling) and every point takes the neighbourhood minimum of
    the cell it falls in. Positional error is bounded by the cell size.
    """
    origin = np.minimum(xyz[:, :2].min(axis=0), ground_xyz[:, :2].min(axis=0))
    top = np.maximum(xyz[:, :2].max(axis=0), ground_xyz[:, :2].max(axis=0))
    shape = (np.floor((top - origin) / cell).astype(np.int64) + 1)[::-1]  # rows=y, cols=x

    g_ij = np.floor((ground_xyz[:, :2] - origin) / cell).astype(np.int64)
    zmin = np.full(shape, np.inf)
    np.minimum.at(zmin, (g_ij[:, 1], g_ij[:, 0]), ground_xyz[:, 2])

    neigh_min = ndimage.minimum_filter(
        zmin, footprint=_disk_footprint(r1 / cell), mode="constant", cval=np.inf
    )

    p_ij = np.floor((xyz[:, :2] - origin) / cell).astype(np.int64)
    m = neigh_min[p_ij[:, 1], p_ij[:, 0]]
    acu = xyz[:, 2] - m
    acu[~np.isfinite(m)] = np.nan
    return acu


def _acu_kdtree(
    xyz: np.ndarray, ground_xyz: np.ndarray, r1: float, chunk_size: int, workers: int
) -> np.ndarray:
    """Exact Eq. 1 against the subsampled ground points (as the reference MATLAB code)."""
    tree = cKDTree(ground_xyz[:, :2])
    gz = ground_xyz[:, 2]
    acu = np.full(len(xyz), np.nan)
    for start in range(0, len(xyz), chunk_size):
        sl = slice(start, start + chunk_size)
        neigh = tree.query_ball_point(xyz[sl, :2], r=r1, workers=workers, return_sorted=False)
        counts = np.fromiter((len(n) for n in neigh), dtype=np.int64, count=len(neigh))
        has = counts > 0
        if not has.any():
            continue
        flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in neigh[has]])
        offsets = np.concatenate(([0], np.cumsum(counts[has])[:-1]))
        zmin = np.minimum.reduceat(gz[flat], offsets)
        chunk_acu = np.full(counts.size, np.nan)
        chunk_acu[has] = xyz[sl, 2][has] - zmin
        acu[sl] = chunk_acu
    return acu


def water_accumulation(
    xyz: np.ndarray,
    ground_xyz: np.ndarray,
    r1: float = 100.0,
    grid_cell: float = 10.0,
    method: str = "grid",
    chunk_size: int = 200_000,
    workers: int = -1,
    clip_negative: bool = True,
) -> np.ndarray:
    """Compute ACU (Eq. 1) for every point.

    Parameters
    ----------
    xyz : (N, 3) points to enrich (the whole cloud).
    ground_xyz : (M, 3) ground-classified points (not yet subsampled).
    r1 : XY search radius in metres (paper: 100 m).
    grid_cell : ground subsampling spacing in metres (paper: 10 m).
    method : ``"grid"`` (fast raster min-filter, default) or ``"kdtree"``
        (exact XY range search on subsampled ground points, slower).
    clip_negative : points below every ground sample (noise, underpasses) get 0.

    Returns
    -------
    (N,) float64 array; NaN where no ground point lies within ``r1``.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    ground_xyz = np.asarray(ground_xyz, dtype=np.float64)
    if len(ground_xyz) == 0:
        raise ValueError("No ground points available: water accumulation needs ground-classified points.")

    if method == "grid":
        acu = _acu_grid(xyz, ground_xyz, r1, grid_cell)
    elif method == "kdtree":
        sub = ground_grid_subsample(ground_xyz, grid_cell) if grid_cell > 0 else ground_xyz
        acu = _acu_kdtree(xyz, sub, r1, chunk_size, workers)
    else:
        raise ValueError(f"Unknown method {method!r}; use 'grid' or 'kdtree'.")

    if clip_negative:
        np.maximum(acu, 0.0, out=acu, where=np.isfinite(acu))
    return acu
