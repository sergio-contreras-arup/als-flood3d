"""Surface runoff potential (paper Section 3.2, Eq. 7).

RUN(P) = verticality(P)^0.1 * planarity(P) * |scattering(P) - 1|

The 0.1 exponent amplifies the small slopes that dominate urban ground
(roads, sidewalks); planarity rewards continuous surfaces; (1 - scattering)
penalises rough/cluttered geometry such as vegetation.
"""

from __future__ import annotations

import numpy as np

VERTICALITY_EXPONENT = 0.1


def runoff_potential(
    verticality: np.ndarray,
    planarity: np.ndarray,
    scattering: np.ndarray,
    exponent: float = VERTICALITY_EXPONENT,
) -> np.ndarray:
    v = np.clip(np.asarray(verticality, dtype=np.float64), 0.0, 1.0)
    return v**exponent * np.asarray(planarity, dtype=np.float64) * np.abs(np.asarray(scattering) - 1.0)
