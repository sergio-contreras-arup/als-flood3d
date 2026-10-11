"""Eigen-based geometric features (paper Section 3.2, Eq. 2-6).

For each point P the covariance matrix C of its spherical neighbourhood of
radius r2 is computed (Eq. 3) and its eigenvalues obtained (Eq. 4), sorted as
lambda1 <= lambda2 <= lambda3. Then:

    verticality = 1 - |N_z|                     (Eq. 2, N = normal = eigvec of lambda1)
    planarity   = (lambda2 - lambda1) / lambda3 (Eq. 5)
    scattering  = lambda1 / lambda3             (Eq. 6)

These match the definitions used by CloudCompare's "Compute geometric features",
which the authors used to produce the inputs of their MATLAB code.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GeometricFeatures:
    verticality: np.ndarray
    planarity: np.ndarray
    scattering: np.ndarray
    eigenvalues: np.ndarray  # (N, 3) ascending
    normals: np.ndarray  # (N, 3)
    n_neighbors: np.ndarray  # (N,)


def _neighbourhood_covariances(
    xyz: np.ndarray, query: slice, tree: cKDTree, radius: float, workers: int
) -> tuple[np.ndarray, np.ndarray]:
    q = xyz[query]
    neigh = tree.query_ball_point(q, r=radius, workers=workers, return_sorted=False)
    counts = np.fromiter((len(n) for n in neigh), dtype=np.int64, count=len(neigh))
    flat = np.concatenate([np.asarray(n, dtype=np.int64) for n in neigh])
    owner = np.repeat(np.arange(len(q)), counts)
    offsets = np.concatenate(([0], np.cumsum(counts)[:-1]))

    # Offsets relative to the query point keep values small (precision) and the
    # covariance is translation invariant.
    d = xyz[flat] - q[owner]
    del flat, owner
    s1 = np.add.reduceat(d, offsets, axis=0)  # (n, 3)
    # Only the 6 unique second moments, one at a time, to bound memory.
    s2 = np.empty((len(q), 3, 3))
    for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
        s2[:, a, b] = s2[:, b, a] = np.add.reduceat(d[:, a] * d[:, b], offsets)

    k = counts[:, None]
    mean = s1 / k
    cov = s2 / k[:, :, None] - mean[:, :, None] * mean[:, None, :]
    return cov, counts


def geometric_features(
    xyz: np.ndarray,
    radius: float,
    chunk_size: int = 20_000,
    workers: int = -1,
    min_neighbors: int = 3,
) -> GeometricFeatures:
    """Compute verticality, planarity and scattering with a radius neighbourhood.

    ``radius`` is the paper's r2, adapted to point density
    (Vigo 10 m @ 0.5-2 pt/m2, Hong Kong 5 m @ 4-5 pt/m2, Rotterdam 2 m @ 10-20 pt/m2).
    Points with fewer than ``min_neighbors`` neighbours get NaN.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    n = len(xyz)
    tree = cKDTree(xyz)

    evals = np.full((n, 3), np.nan)
    normals = np.full((n, 3), np.nan)
    counts = np.zeros(n, dtype=np.int64)

    def process(start: int) -> None:
        sl = slice(start, min(start + chunk_size, n))
        cov, cnt = _neighbourhood_covariances(xyz, sl, tree, radius, workers=1)
        counts[sl] = cnt
        ok = cnt >= min_neighbors
        if not ok.any():
            return
        w, v = np.linalg.eigh(cov[ok])  # ascending eigenvalues
        idx = np.arange(sl.start, sl.stop)[ok]
        evals[idx] = np.clip(w, 0.0, None)
        normals[idx] = v[:, :, 0]

    # Chunks write disjoint slices; numpy/LAPACK release the GIL, so threads scale.
    n_threads = (os.cpu_count() or 1) if workers == -1 else max(1, workers)
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        list(pool.map(process, range(0, n, chunk_size)))

    l1, l2, l3 = evals[:, 0], evals[:, 1], evals[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        planarity = np.where(l3 > 0, (l2 - l1) / l3, np.nan)
        scattering = np.where(l3 > 0, l1 / l3, np.nan)
    verticality = 1.0 - np.abs(normals[:, 2])

    return GeometricFeatures(
        verticality=verticality,
        planarity=planarity,
        scattering=scattering,
        eigenvalues=evals,
        normals=normals,
        n_neighbors=counts,
    )
