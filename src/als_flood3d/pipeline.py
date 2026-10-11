"""End-to-end workflow (paper Figure 1): ALS cloud -> ACU + RUN per point."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .accumulation import water_accumulation
from .features import geometric_features
from .runoff import VERTICALITY_EXPONENT, runoff_potential

log = logging.getLogger(__name__)

ASPRS_GROUND = 2


@dataclass
class FloodRiskConfig:
    r1: float = 100.0  # XY radius for water accumulation (m)
    r2: float = 2.0  # 3D radius for geometric features (m), adapt to density
    grid_cell: float = 10.0  # ground subsampling spacing (m)
    ground_classes: tuple[int, ...] = (ASPRS_GROUND,)
    acu_method: str = "grid"  # "grid" (fast) | "kdtree" (exact on subsampled ground)
    verticality_exponent: float = VERTICALITY_EXPONENT
    workers: int = -1
    feature_chunk_size: int = 20_000


@dataclass
class FloodRiskResult:
    acu: np.ndarray
    run: np.ndarray
    verticality: np.ndarray
    planarity: np.ndarray
    scattering: np.ndarray
    timings: dict[str, float] = field(default_factory=dict)

    def as_fields(self) -> dict[str, np.ndarray]:
        return {
            "ACU": self.acu,
            "RUN": self.run,
            "verticality": self.verticality,
            "planarity": self.planarity,
            "scattering": self.scattering,
        }


def ground_mask(classification: np.ndarray | None, ground_classes: tuple[int, ...]) -> np.ndarray:
    if classification is None:
        raise ValueError(
            "The point cloud has no classification; water accumulation needs ground points "
            "(classify first, e.g. with PDAL filters.smrf / CSF)."
        )
    return np.isin(classification, ground_classes)


def compute_flood_risk(
    xyz: np.ndarray,
    classification: np.ndarray | None,
    config: FloodRiskConfig | None = None,
) -> FloodRiskResult:
    cfg = config or FloodRiskConfig()
    xyz = np.asarray(xyz, dtype=np.float64)
    timings: dict[str, float] = {}

    t = time.perf_counter()
    gmask = ground_mask(classification, cfg.ground_classes)
    log.info("Water accumulation: %d points, %d ground, r1=%.1f m", len(xyz), gmask.sum(), cfg.r1)
    acu = water_accumulation(
        xyz, xyz[gmask], r1=cfg.r1, grid_cell=cfg.grid_cell, method=cfg.acu_method, workers=cfg.workers
    )
    timings["accumulation_s"] = time.perf_counter() - t

    t = time.perf_counter()
    log.info("Geometric features: r2=%.2f m", cfg.r2)
    feats = geometric_features(xyz, cfg.r2, chunk_size=cfg.feature_chunk_size, workers=cfg.workers)
    timings["features_s"] = time.perf_counter() - t

    run = runoff_potential(feats.verticality, feats.planarity, feats.scattering, cfg.verticality_exponent)

    return FloodRiskResult(
        acu=acu,
        run=run,
        verticality=feats.verticality,
        planarity=feats.planarity,
        scattering=feats.scattering,
        timings=timings,
    )
