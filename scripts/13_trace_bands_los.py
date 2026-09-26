"""Trace the ground-truth map at 3.5, 28 and 39 GHz and mark line of sight.

This is the input to the 3GPP 38.901 UMi comparison (scripts 14 and 15).
Everything about the solver is the interim setup: 2e8 rays per tx, max depth 3,
LoS + specular + diffraction, refraction and diffuse scattering OFF, grid fixed
in metres.  The per-band convergence study (scripts 06 to 09) is what licenses
using 2e8 at all three bands.

Line of sight is decided GEOMETRICALLY, one deterministic shadow ray per cell
from the cell centre at 1.5 m to the transmitter, not by looking at which cells
the solver's LoS paths happened to reach.  38.901 switches formula on LOS vs
NLOS, so the classification has to be exact and seed-free.

    python scripts/13_trace_bands_los.py --out data/bands_los.npz
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

import numpy as np

from radiomap.config import SceneConfig
from radiomap.metrics import to_db

BANDS_GHZ = (3.5, 28.0, 39.0)


def los_mask(mi_scene, centres: np.ndarray, tx: np.ndarray) -> np.ndarray:
    """True where the straight segment cell centre -> tx is unobstructed."""
    import mitsuba as mi

    pts = centres.reshape(-1, 3).astype(np.float64)
    vec = tx[None, :] - pts
    dist = np.linalg.norm(vec, axis=1)
    d = vec / dist[:, None]
    # Nudge the origin off the ground plane and stop just short of the tx so
    # neither endpoint counts as an obstruction.
    eps = 1e-3
    o = pts + eps * d
    ray = mi.Ray3f(
        o=mi.Point3f(o[:, 0].astype(np.float32), o[:, 1].astype(np.float32), o[:, 2].astype(np.float32)),
        d=mi.Vector3f(d[:, 0].astype(np.float32), d[:, 1].astype(np.float32), d[:, 2].astype(np.float32)),
    )
    ray.maxt = mi.Float((dist - 2 * eps).astype(np.float32))
    blocked = np.asarray(mi_scene.ray_test(ray), dtype=bool)
    return (~blocked).reshape(centres.shape[:2])


def trace_band(freq_ghz: float, n_rays: int, seed: int, max_depth: int = 3):
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene
    import sionna.rt as rt
    from radiomap.scene import building_mask_from_scene

    cfg = SceneConfig(frequency_hz=freq_ghz * 1e9, samples_per_tx=n_rays, seed=seed,
                      max_depth=max_depth)
    scene = load_scene(getattr(rt.scene, cfg.scene_name))
    scene.frequency = cfg.frequency_hz
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    scene.add(Transmitter(name="tx", position=list(cfg.tx_position), power_dbm=cfg.tx_power_dbm))

    t0 = time.time()
    rm = RadioMapSolver()(
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
    secs = time.time() - t0
    gain = to_db(np.asarray(rm.path_gain).squeeze())
    centres = np.asarray(rm.cell_centers)
    building = building_mask_from_scene(scene, centres)
    los = los_mask(scene.mi_scene, centres, np.asarray(cfg.tx_position, dtype=float))
    return gain, centres, building, los, secs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/bands_los.npz")
    ap.add_argument("--rays", type=float, default=2e8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-depth", type=int, default=3,
                    help="3 is the interim setup; 5 is the truncation check")
    ap.add_argument("--bands", default="3.5,28,39")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    out = {}
    ref_centres = ref_building = ref_los = None
    bands = tuple(float(b) for b in args.bands.split(","))
    for f in bands:
        gain, centres, building, los, secs = trace_band(f, int(args.rays), args.seed, args.max_depth)
        print(f"{f:>5g} GHz  solve {secs:5.1f} s  finite cells {np.isfinite(gain).sum()}")
        if ref_centres is None:
            ref_centres, ref_building, ref_los = centres, building, los
        else:
            # Geometry must not move with frequency; if it did, nothing below
            # would be comparable cell for cell.
            assert np.allclose(centres, ref_centres), "grid moved between bands"
            assert np.array_equal(building, ref_building), "building mask moved"
            assert np.array_equal(los, ref_los), "LOS mask moved"
        out[f"gain_db_{f:g}"] = gain
        out[f"solve_s_{f:g}"] = secs

    cfg = SceneConfig()
    np.savez_compressed(
        args.out,
        centres=ref_centres,
        building=ref_building,
        los=ref_los,
        tx=np.asarray(cfg.tx_position, dtype=float),
        bands_ghz=np.asarray(bands),
        max_depth=args.max_depth,
        rays=args.rays,
        seed=args.seed,
        **out,
    )
    street = ~ref_building
    print(f"street cells {street.sum()}, of which LOS {int((ref_los & street).sum())} "
          f"({100 * (ref_los & street).sum() / street.sum():.1f}%)")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
