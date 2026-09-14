"""Ray tracing with Sionna RT, and the accessibility mask.

This is the only module that needs Sionna.  It writes a plain ``.npz`` that
every other module reads, so the reconstruction study can be re-run, checked or
extended without a Sionna install.

Two things worth knowing before reading the numbers:

* ``RadioMap.path_gain`` is a **linear** power ratio, not dB.  On this scene it
  is of order 1e-9.
* ``refraction`` defaults to **on** in Sionna.  Left on, rays pass through
  building walls and cells inside buildings stop being structurally
  unreachable, so the accessibility mask silently stops meaning "outdoors".
  It is switched off here on purpose.

Cells under building geometry are found by casting a vertical ray upward from
each cell centre, not by thresholding the gain.  Thresholding would throw away
genuine deep-shadow street cells, which are exactly the cells the hold-out
experiment is about.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .config import SceneConfig
from .metrics import to_db


@dataclass
class RadioMapData:
    """A traced map, reduced to what the reconstruction study needs."""

    xy: np.ndarray          # (n_valid, 2) cell centres in metres
    gain_db: np.ndarray     # (n_valid,) path gain in dB
    grid_shape: tuple[int, int]
    valid_mask: np.ndarray  # (ny, nx) bool, True where a cell is usable
    building_mask: np.ndarray  # (ny, nx) bool, True where a cell is under a building
    tx_xy: tuple[float, float]
    config: dict

    @property
    def n_valid(self) -> int:
        return int(self.xy.shape[0])

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            xy=self.xy,
            gain_db=self.gain_db,
            grid_shape=np.asarray(self.grid_shape),
            valid_mask=self.valid_mask,
            building_mask=self.building_mask,
            tx_xy=np.asarray(self.tx_xy),
            config=np.asarray([repr(self.config)], dtype=object),
        )

    @staticmethod
    def load(path: str) -> "RadioMapData":
        d = np.load(path, allow_pickle=True)
        return RadioMapData(
            xy=d["xy"],
            gain_db=d["gain_db"],
            grid_shape=tuple(int(v) for v in d["grid_shape"]),
            valid_mask=d["valid_mask"],
            building_mask=d["building_mask"],
            tx_xy=tuple(float(v) for v in d["tx_xy"]),
            config=eval(str(d["config"][0])) if "config" in d else {},
        )

    def to_grid(self, values: np.ndarray, fill: float = np.nan) -> np.ndarray:
        """Scatter a per-valid-cell vector back onto the (ny, nx) grid."""
        out = np.full(self.grid_shape, fill, dtype=float)
        out[self.valid_mask] = values
        return out


# --------------------------------------------------------------------------
# Sionna path
# --------------------------------------------------------------------------

def trace(cfg: SceneConfig | None = None, *, verbose: bool = True) -> RadioMapData:
    """Trace the scene and return the valid cells.

    Requires ``sionna >= 1.0`` (developed against Sionna RT 2.0.1).
    """
    cfg = cfg or SceneConfig()

    import sionna  # noqa: F401
    from sionna.rt import (
        PlanarArray,
        RadioMapSolver,
        Receiver,
        Transmitter,
        load_scene,
    )
    import sionna.rt as rt

    scene = load_scene(getattr(rt.scene, cfg.scene_name))
    scene.frequency = cfg.frequency_hz

    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = scene.tx_array

    scene.add(
        Transmitter(
            name="tx",
            position=list(cfg.tx_position),
            power_dbm=cfg.tx_power_dbm,
        )
    )

    solver = RadioMapSolver()
    rm = solver(
        scene,
        max_depth=cfg.max_depth,
        cell_size=(cfg.cell_size_m, cfg.cell_size_m),
        samples_per_tx=int(cfg.samples_per_tx),
        los=cfg.los,
        specular_reflection=cfg.specular_reflection,
        diffuse_reflection=cfg.diffuse_reflection,
        refraction=cfg.refraction,
        diffraction=cfg.diffraction,
        seed=cfg.seed,
    )

    gain_linear = np.asarray(rm.path_gain).squeeze()      # (ny, nx)
    centres = np.asarray(rm.cell_centers)                 # (ny, nx, 3)
    if verbose:
        print(f"traced grid {gain_linear.shape}, {gain_linear.size} cells")

    building = building_mask_from_scene(scene, centres)
    gain_db = to_db(gain_linear)

    valid = (~building) & np.isfinite(gain_db)
    xy = centres[..., :2][valid]

    data = RadioMapData(
        xy=xy.astype(float),
        gain_db=gain_db[valid].astype(float),
        grid_shape=tuple(gain_linear.shape),
        valid_mask=valid,
        building_mask=building,
        tx_xy=(float(cfg.tx_position[0]), float(cfg.tx_position[1])),
        config=dict(cfg.__dict__),
    )
    if verbose:
        frac = 100.0 * data.n_valid / gain_linear.size
        print(
            f"{building.sum()} cells under buildings, "
            f"{data.n_valid} valid ({frac:.1f}% of grid), "
            f"gain {data.gain_db.min():.1f} to {data.gain_db.max():.1f} dB, "
            f"sd {data.gain_db.std():.1f} dB"
        )
    return data


def building_mask_from_scene(scene, cell_centres: np.ndarray) -> np.ndarray:
    """True where a vertical ray from the cell centre hits geometry above it.

    Falls back to an all-False mask (with a warning) if the Mitsuba scene is not
    reachable, rather than silently substituting a gain threshold.
    """
    try:
        import mitsuba as mi
        import drjit as dr
    except Exception:  # pragma: no cover
        print("WARNING: mitsuba unavailable, building mask is empty")
        return np.zeros(cell_centres.shape[:2], dtype=bool)

    ny, nx = cell_centres.shape[:2]
    pts = cell_centres.reshape(-1, 3).astype(np.float32)

    try:
        mi_scene = scene.mi_scene
    except AttributeError:  # pragma: no cover - older Sionna
        mi_scene = getattr(scene, "mitsuba_scene", None)
    if mi_scene is None:  # pragma: no cover
        print("WARNING: could not reach the Mitsuba scene, building mask is empty")
        return np.zeros((ny, nx), dtype=bool)

    origin = mi.Point3f(pts[:, 0], pts[:, 1], pts[:, 2])
    direction = mi.Vector3f(0.0, 0.0, 1.0)
    ray = mi.Ray3f(o=origin, d=direction)
    si = mi_scene.ray_intersect(ray)
    hit = np.asarray(dr.numpy(si.is_valid()), dtype=bool)
    return hit.reshape(ny, nx)


# --------------------------------------------------------------------------
# Sionna-free path, for tests and for anyone who wants to read the pipeline
# before installing a ray tracer
# --------------------------------------------------------------------------

def synthetic_canyon(
    nx: int = 187,
    ny: int = 122,
    *,
    seed: int = 0,
    tx_xy: tuple[float, float] = (-33.0, 11.0),
    correlation_range_m: float = 28.0,
) -> RadioMapData:
    """A piecewise-smooth stand-in for a traced street canyon.

    Log-distance path loss, plus a correlated random field, plus two hard
    shadow edges and a building block.  It is *not* a substitute for the ray
    tracer and no result in the report comes from it.  It exists so that the
    sampling designs, the interpolators and the metrics can be exercised
    deterministically, which is what the test suite does.
    """
    rng = np.random.default_rng(seed)

    x = np.arange(nx, dtype=float) - nx / 2.0
    y = np.arange(ny, dtype=float) - ny / 2.0
    xx, yy = np.meshgrid(x, y)

    d = np.maximum(np.hypot(xx - tx_xy[0], yy - tx_xy[1]), 1.0)
    trend = -28.6 - 40.9 * np.log10(d)

    # Correlated field by filtering white noise.
    from scipy.ndimage import gaussian_filter

    noise = rng.standard_normal((ny, nx))
    smooth = gaussian_filter(noise, sigma=correlation_range_m / 3.0, mode="reflect")
    smooth = smooth / (smooth.std() + 1e-12) * 8.0

    # Two hard shadow boundaries: the thing smooth interpolators cannot see.
    shadow = np.zeros((ny, nx))
    shadow[yy > 0.30 * xx + 18.0] -= 14.0
    shadow[xx > 42.0] -= 9.0

    gain_db = trend + smooth + shadow

    building = np.zeros((ny, nx), dtype=bool)
    building[int(0.18 * ny): int(0.36 * ny), int(0.18 * nx): int(0.52 * nx)] = True
    building[int(0.62 * ny): int(0.84 * ny), int(0.44 * nx): int(0.80 * nx)] = True

    valid = ~building
    xy = np.stack([xx[valid], yy[valid]], axis=1)

    return RadioMapData(
        xy=xy,
        gain_db=gain_db[valid],
        grid_shape=(ny, nx),
        valid_mask=valid,
        building_mask=building,
        tx_xy=tx_xy,
        config={"synthetic": True, "seed": seed},
    )
