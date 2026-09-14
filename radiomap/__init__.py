"""Sparse radio map reconstruction on a Sionna RT street canyon.

Classical spatial interpolation baselines (nearest neighbour, IDW, thin-plate
RBF, ordinary kriging, detrended kriging) scored two ways on the same map:

* ``random``  -- observations drawn uniformly, scored on the rest (gap filling)
* ``block``   -- a contiguous square removed entirely, scored only inside it
                 (extrapolation into unseen ground)

The package is deliberately split so that the ray tracing (which needs Sionna)
is isolated in :mod:`radiomap.scene`.  Everything else runs on numpy/scipy
alone, so the reconstruction study can be re-run from a cached ``.npz`` map
without installing Sionna at all.
"""

from .config import SceneConfig, SweepConfig
from .interpolators import (
    NearestNeighbour,
    InverseDistanceWeighting,
    ThinPlateRBF,
    OrdinaryKriging,
    DetrendedKriging,
    METHODS,
)
from .metrics import rmse, mae, to_db, to_linear
from .sampling import random_design, block_design, block_origins
from .trend import LogDistanceTrend
from .variogram import empirical_variogram, fit_spherical, SphericalVariogram

__version__ = "1.0.0"

__all__ = [
    "SceneConfig",
    "SweepConfig",
    "NearestNeighbour",
    "InverseDistanceWeighting",
    "ThinPlateRBF",
    "OrdinaryKriging",
    "DetrendedKriging",
    "METHODS",
    "rmse",
    "mae",
    "to_db",
    "to_linear",
    "random_design",
    "block_design",
    "block_origins",
    "LogDistanceTrend",
    "empirical_variogram",
    "fit_spherical",
    "SphericalVariogram",
]
