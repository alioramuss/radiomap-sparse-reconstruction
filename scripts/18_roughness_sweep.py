"""Part two of the scattering study: S made frequency dependent from roughness.

Script 11 held the scattering coefficient S equal across bands to isolate the
estimator question.  Here S is set per band from a surface roughness sigma_h
through the Rayleigh criterion and the Ament attenuation factor (``rayleigh_S``
in script 11, normal incidence), at sigma_h = 0.2, 0.5, 1.0 and 2.0 mm, and the
three bands are traced cell for cell on the same grid as the UMi comparison.

The question it answers is the hypothesis left open in the README: diffraction
alone makes deep NLOS cells scale as 30log10(f), against UMi's 21.3log10(f).
If roughness-driven diffuse scattering is what fills the shadow in a measured
channel, a frequency-dependent S should pull that coefficient back toward 2.13.

Script 11 showed the scattering-on estimator is heavy tailed, so everything
downstream (script 19) uses robust statistics: the per-cell MEDIAN over three
independent runs, and medians over cells.  A fourth run at 2e7 rays per config
measures how far those medians still drift with ray count.

S is applied uniformly to every material, as in script 11.  Normal incidence is
the largest S a given roughness can imply; at oblique incidence the effective
roughness shrinks by cos(theta), so these S values are an upper bound.

    python scripts/18_roughness_sweep.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import time
from importlib import import_module

import numpy as np

from radiomap.config import SceneConfig
from radiomap.metrics import to_db

rayleigh_S = import_module("11_scattering_sweep").rayleigh_S
los_mask = import_module("13_trace_bands_los").los_mask

BANDS_GHZ = (3.5, 28.0, 39.0)
ROUGHNESS_MM = (0.0, 0.2, 0.5, 1.0, 2.0)   # 0.0 = scattering off, the UMi baseline
MAIN_RAYS = int(1e8)
MAIN_SEEDS = (1, 2, 3)
DRIFT_RAYS = int(2e7)
DRIFT_SEED = 4


def trace_once(freq_ghz: float, s: float, n_rays: int, seed: int, want_geometry: bool):
    from sionna.rt import PlanarArray, RadioMapSolver, Transmitter, load_scene
    import sionna.rt as rt
    from radiomap.scene import building_mask_from_scene

    cfg = SceneConfig(frequency_hz=freq_ghz * 1e9, samples_per_tx=n_rays, seed=seed,
                      diffuse_reflection=s > 0.0, scattering_coefficient=s)
    scene = load_scene(getattr(rt.scene, cfg.scene_name))
    scene.frequency = cfg.frequency_hz
    for obj in scene.objects.values():
        obj.radio_material.scattering_coefficient = s
    scene.tx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    scene.rx_array = scene.tx_array
    scene.add(Transmitter(name="tx", position=list(cfg.tx_position), power_dbm=cfg.tx_power_dbm))

    t0 = time.time()
    rm = RadioMapSolver()(
        scene, max_depth=cfg.max_depth, cell_size=(cfg.cell_size_m, cfg.cell_size_m),
        samples_per_tx=int(cfg.samples_per_tx), los=cfg.los,
        specular_reflection=cfg.specular_reflection,
        diffuse_reflection=cfg.diffuse_reflection, refraction=cfg.refraction,
        diffraction=cfg.diffraction, seed=cfg.seed,
    )
    secs = time.time() - t0
    gain = to_db(np.asarray(rm.path_gain).squeeze()).astype(np.float32)
    geom = None
    if want_geometry:
        centres = np.asarray(rm.cell_centers)
        geom = dict(centres=centres,
                    building=building_mask_from_scene(scene, centres),
                    los=los_mask(scene.mi_scene, centres, np.asarray(cfg.tx_position, float)),
                    tx=np.asarray(cfg.tx_position, float))
    return gain, secs, geom


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="data/roughness")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    s_table = {f"{f:g}": {f"{h:g}": (rayleigh_S(f, h) if h > 0 else 0.0) for h in ROUGHNESS_MM}
               for f in BANDS_GHZ}
    with open(os.path.join(args.cache_dir, "s_table.json"), "w") as fh:
        json.dump(s_table, fh, indent=2)
    print("S implied by roughness, normal incidence (3.5 / 28 / 39 GHz):")
    for h in ROUGHNESS_MM[1:]:
        print(f"  {h:>3g} mm  " + "  ".join(f"{s_table[f'{f:g}'][f'{h:g}']:.3f}" for f in BANDS_GHZ))

    geom_path = os.path.join(args.cache_dir, "geometry.npz")
    jobs = [(f, h, MAIN_RAYS, sd) for h in ROUGHNESS_MM for f in BANDS_GHZ for sd in MAIN_SEEDS]
    jobs += [(f, h, DRIFT_RAYS, DRIFT_SEED) for h in ROUGHNESS_MM for f in BANDS_GHZ]
    for i, (f, h, n, sd) in enumerate(jobs, 1):
        out = os.path.join(args.cache_dir, f"f{f:g}_h{h:g}_n{n}_s{sd}.npz")
        if os.path.exists(out):
            continue
        s = s_table[f"{f:g}"][f"{h:g}"]
        want_geom = not os.path.exists(geom_path)
        gain, secs, geom = trace_once(f, s, n, sd, want_geom)
        if geom is not None:
            np.savez_compressed(geom_path, **geom)
        np.savez_compressed(out, gain_db=gain, S=s, solve_seconds=secs)
        print(f"[{i:>2}/{len(jobs)}] {f:>4g} GHz  sigma_h {h:>3g} mm  S {s:.3f}  "
              f"{n:.0e} rays  seed {sd}  {secs:6.1f} s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
