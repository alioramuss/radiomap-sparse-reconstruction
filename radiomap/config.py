"""Configuration for the scene and for the reconstruction sweep.

Every number that appears in the interim report lives here, so that a reader
can check the setup in one place rather than hunting through the scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class SceneConfig:
    """Ray tracing setup.

    Defaults reproduce the interim report: Sionna RT 2.0.1, the built-in
    ``simple_street_canyon`` scene, one isotropic rooftop transmitter.
    """

    scene_name: str = "simple_street_canyon"
    frequency_hz: float = 3.5e9
    tx_position: Sequence[float] = (-33.0, 11.0, 32.0)
    tx_power_dbm: float = 30.0

    # Measurement plane.
    plane_height_m: float = 1.5
    cell_size_m: float = 1.0

    # Solver.
    max_depth: int = 3
    samples_per_tx: int = int(2e8)
    los: bool = True
    specular_reflection: bool = True
    diffraction: bool = True
    diffuse_reflection: bool = False
    # NOTE: Sionna's default is refraction=True.  Left on, rays leak into
    # building interiors, those cells stop being structurally unreachable, and
    # the accessibility mask quietly stops meaning what you think it means.
    refraction: bool = False
    seed: int = 1

    def __post_init__(self) -> None:
        if self.cell_size_m <= 0:
            raise ValueError("cell_size_m must be positive")
        if self.samples_per_tx <= 0:
            raise ValueError("samples_per_tx must be positive")

    @property
    def wavelength_m(self) -> float:
        return 299_792_458.0 / self.frequency_hz


@dataclass(frozen=True)
class SweepConfig:
    """Reconstruction sweep setup."""

    sampling_fractions: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20)

    # Random design.
    random_seeds: Sequence[int] = (0, 1, 2, 3, 4)

    # Block design.  The block side is set to roughly twice the correlation
    # range fitted from the variogram (27.8 m at 3.5 GHz on this scene).  A
    # block narrower than the correlation range is still measuring
    # interpolation, not extrapolation.
    block_side_m: float = 60.0
    n_block_positions: int = 6
    block_seeds: Sequence[int] = (0, 1, 2)

    # Hyperparameters are chosen by k-fold cross-validation on the observed
    # cells only, never against the held-out truth.
    cv_folds: int = 5
    cv_seed: int = 12345

    # The variogram is fitted on a 10% draw, which is what a real user would
    # have, rather than on the full map.
    variogram_fit_fraction: float = 0.10
    variogram_fit_seed: int = 999

    methods: Sequence[str] = field(
        default_factory=lambda: (
            "nearest",
            "idw",
            "rbf",
            "ok",
            "dk",
        )
    )
