"""Does diffuse scattering change the ray budget, and does it do so per band?

Prof Thompson's suggestion after the multiband result: look at diffuse
scattering, on the expectation that it is more prominent below 6 GHz than at
millimetre wave.

Two things have to be said before any number here means anything.

* Sionna ships every ITU material with a scattering coefficient of ZERO, so
  ``diffuse_reflection=True`` on its own changes nothing.  Verified directly:
  flipping the flag at a fixed seed moves the map by 7e-6 dB, which is float
  noise.  S must be set explicitly, which makes this a modelling choice and not
  a flag.

* S is the fraction of the reflected FIELD that goes diffuse, and the specular
  part is scaled by sqrt(1 - S^2).  Raising S redistributes energy, it does not
  add energy.

This script does PART ONE only: S held equal across all three bands.  That
isolates the estimator question (does scattering cost more rays, and does the
cost differ by band) with the physics held constant.  Part two, where S is made
frequency dependent from the Rayleigh criterion, is a separate run, because
``rayleigh_S`` below shows the two parts pull in opposite directions.

    python scripts/11_scattering_sweep.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import itertools
import time

import numpy as np

from radiomap.config import SceneConfig
from radiomap.scene import trace

BANDS_GHZ = [3.5, 28.0, 39.0]
S_VALUES = [0.0, 0.3, 0.6]
RAY_COUNTS = [5e6, 2e7, 5e7, 1e8]
SEEDS = (1, 2, 3)

C = 299792458.0


def rayleigh_S(freq_ghz: float, sigma_mm: float, theta_deg: float = 0.0) -> float:
    """Scattering coefficient implied by surface roughness (Ament).

    The specular lobe is attenuated by rho_s = exp(-8 pi^2 sigma^2 cos^2 / lambda^2)
    and the energy that leaves it goes diffuse, so S = sqrt(1 - rho_s^2).

    NOTE the direction of this: a fixed surface looks ROUGHER at shorter
    wavelength, so this predicts MORE diffuse scattering at millimetre wave, not
    less.  That is the opposite of the expectation this study is testing, and
    reconciling the two is the point of part two.
    """
    lam_mm = C / (freq_ghz * 1e9) * 1000.0
    rho = np.exp(-8 * np.pi**2 * sigma_mm**2 * np.cos(np.radians(theta_deg)) ** 2 / lam_mm**2)
    return float(np.sqrt(max(0.0, 1.0 - rho**2)))


def run_point(freq_ghz: float, s: float, n_rays: float, cache_dir: str) -> dict:
    grids, buildings, times = [], [], []
    for seed in SEEDS:
        tag = f"f{freq_ghz:g}_S{s:g}_n{int(n_rays):d}_s{seed}"
        cache = os.path.join(cache_dir, tag + ".npz")
        if os.path.exists(cache):
            d = np.load(cache)
            grids.append(d["grid"]); buildings.append(d["building"])
            times.append(float(d["solve_seconds"]))
            continue
        cfg = SceneConfig(
            frequency_hz=freq_ghz * 1e9,
            samples_per_tx=int(n_rays),
            seed=seed,
            diffuse_reflection=s > 0.0,
            scattering_coefficient=s,
        )
        t0 = time.perf_counter()
        m = trace(cfg, verbose=False)
        el = time.perf_counter() - t0
        grid = m.to_grid(m.gain_db)
        np.savez_compressed(cache, grid=grid.astype(np.float32),
                            building=m.building_mask, solve_seconds=el)
        grids.append(grid); buildings.append(m.building_mask); times.append(el)

    street = ~buildings[0]
    ok = street & np.all([np.isfinite(g) for g in grids], axis=0)
    pairs = [float(np.sqrt(np.mean((grids[i][ok] - grids[j][ok]) ** 2)))
             for i, j in itertools.combinations(range(len(grids)), 2)]
    vals = grids[0][ok]
    return {
        "frequency_ghz": freq_ghz,
        "scattering_coefficient": s,
        "rays_per_tx": int(n_rays),
        "empty_street_cells_pct": 100.0 * (1.0 - ok.sum() / max(street.sum(), 1)),
        "run_to_run_rmse_db": float(np.mean(pairs)),
        "run_to_run_rmse_sd_db": float(np.std(pairs, ddof=1)),
        "gain_mean_db": float(np.mean(vals)),
        "gain_sd_db": float(np.std(vals)),
        "solve_seconds": float(np.mean(times)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/scattering_sweep.csv")
    p.add_argument("--cache-dir", default="data/scattering")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    print("Rayleigh/Ament S implied by surface roughness, normal incidence:")
    for sig in (0.5, 1.0):
        vals = " ".join(f"{rayleigh_S(f, sig):6.3f}" for f in BANDS_GHZ)
        print(f"  sigma_h = {sig:.1f} mm -> S = {vals}   (3.5 / 28 / 39 GHz)")
    print(flush=True)

    rows = []
    for freq, s in itertools.product(BANDS_GHZ, S_VALUES):
        for n in RAY_COUNTS:
            r = run_point(freq, s, n, args.cache_dir)
            rows.append(r)
            print(f"{freq:5.1f} GHz  S={s:.1f}  {int(n):>11,} rays  "
                  f"run-to-run {r['run_to_run_rmse_db']:6.3f} dB  "
                  f"mean {r['gain_mean_db']:8.2f} dB  sd {r['gain_sd_db']:5.2f}  "
                  f"empty {r['empty_street_cells_pct']:5.2f}%  "
                  f"{r['solve_seconds']:5.1f}s/seed", flush=True)
            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader(); w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
