"""Seed-to-seed ray-count convergence, separately at 3.5, 28 and 39 GHz.

The interim report established a noise floor at 3.5 GHz and then assumed the
2e8 rays that converged there would carry over to the higher bands.  Tianrun
Qi's reading is that the risk runs the other way: lower frequencies are the
ray-count-sensitive ones and the millimetre bands are less so.  Either way the
assumption is untested, so this measures it per band before anything is
compared against 3GPP 38.901 UMi.

Method is the one from ``scripts/02_ray_convergence.py``, with two changes:

* three seeds instead of two at every ray count except the most expensive, so
  the noise estimate gets an uncertainty of its own rather than resting on a
  single pair;
* the grid is held fixed in METRES (1 m cells) across all three bands, stated
  deliberately.  Fixing it in wavelengths instead would hold the cell constant
  in lambda but change the cell count by more than a hundredfold between 3.5
  and 39 GHz, and would be asking a different question.  Metres is what a real
  measurement campaign faces.

    python scripts/06_multiband_convergence.py --out results/convergence_multiband.csv
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import itertools
import json
import time

import numpy as np

from radiomap.config import SceneConfig
from radiomap.scene import trace

BANDS_GHZ = [3.5, 28.0, 39.0]
RAY_COUNTS = [2e5, 1e6, 5e6, 2e7, 5e7, 1e8, 2e8, 5e8]
# Three seeds everywhere except the most expensive count, which stays at two.
SEEDS_DEFAULT = (1, 2, 3)
SEEDS_EXPENSIVE = (1, 2)
EXPENSIVE_FROM = 5e8

FIELDS = [
    "frequency_ghz",
    "rays_per_tx",
    "n_seeds",
    "street_cells",
    "empty_street_cells_pct",
    "seed_to_seed_rmse_db",
    "seed_to_seed_rmse_sd_db",
    "one_map_vs_mean_db",
    "gain_mean_db",
    "gain_sd_db",
    "solve_seconds",
]


def seeds_for(n_rays: float) -> tuple[int, ...]:
    return SEEDS_EXPENSIVE if n_rays >= EXPENSIVE_FROM else SEEDS_DEFAULT


def run_point(freq_ghz: float, n_rays: float, cache_dir: str) -> dict:
    """Trace the same scene with several seeds and measure how far they differ."""
    seeds = seeds_for(n_rays)
    grids, buildings, solve_times = [], [], []

    for seed in seeds:
        tag = f"f{freq_ghz:g}_n{int(n_rays):d}_s{seed}"
        cache = os.path.join(cache_dir, tag + ".npz")
        if os.path.exists(cache):
            d = np.load(cache)
            grids.append(d["grid"])
            buildings.append(d["building"])
            solve_times.append(float(d["solve_seconds"]))
            continue

        cfg = SceneConfig(
            frequency_hz=freq_ghz * 1e9,
            samples_per_tx=int(n_rays),
            seed=seed,
        )
        t0 = time.perf_counter()
        m = trace(cfg, verbose=False)
        elapsed = time.perf_counter() - t0

        grid = m.to_grid(m.gain_db)
        np.savez_compressed(
            cache,
            grid=grid.astype(np.float32),
            building=m.building_mask,
            solve_seconds=elapsed,
        )
        grids.append(grid)
        buildings.append(m.building_mask)
        solve_times.append(elapsed)

    # The building mask is pure geometry and must not move with seed or band.
    for b in buildings[1:]:
        assert np.array_equal(b, buildings[0]), "building mask moved between traces"

    street = ~buildings[0]
    reached_in_all = street & np.all([np.isfinite(g) for g in grids], axis=0)
    empty_frac = 1.0 - reached_in_all.sum() / max(street.sum(), 1)

    # Every distinct pair of seeds gives one estimate of the noise.
    pair_rmse = [
        float(np.sqrt(np.mean((grids[i][reached_in_all] - grids[j][reached_in_all]) ** 2)))
        for i, j in itertools.combinations(range(len(grids)), 2)
    ]
    seed_rmse = float(np.mean(pair_rmse))
    seed_rmse_sd = float(np.std(pair_rmse, ddof=1)) if len(pair_rmse) > 1 else float("nan")

    vals = grids[0][reached_in_all]
    return {
        "frequency_ghz": freq_ghz,
        "rays_per_tx": int(n_rays),
        "n_seeds": len(seeds),
        "street_cells": int(street.sum()),
        "empty_street_cells_pct": 100.0 * empty_frac,
        "seed_to_seed_rmse_db": seed_rmse,
        "seed_to_seed_rmse_sd_db": seed_rmse_sd,
        "one_map_vs_mean_db": seed_rmse / np.sqrt(2.0),
        "gain_mean_db": float(np.mean(vals)),
        "gain_sd_db": float(np.std(vals)),
        "solve_seconds": float(np.mean(solve_times)),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/convergence_multiband.csv")
    p.add_argument("--cache-dir", default="data/multiband")
    p.add_argument("--bands", type=float, nargs="*", default=BANDS_GHZ)
    p.add_argument("--rays", type=float, nargs="*", default=RAY_COUNTS)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)

    rows: list[dict] = []
    for freq in args.bands:
        for n_rays in args.rays:
            t0 = time.perf_counter()
            row = run_point(freq, n_rays, args.cache_dir)
            rows.append(row)
            print(
                f"{freq:5.1f} GHz  {int(n_rays):>12,} rays  "
                f"empty {row['empty_street_cells_pct']:5.2f}%  "
                f"seed-to-seed {row['seed_to_seed_rmse_db']:6.3f} dB "
                f"(sd {row['seed_to_seed_rmse_sd_db']:.3f})  "
                f"one-vs-mean {row['one_map_vs_mean_db']:6.3f} dB  "
                f"field sd {row['gain_sd_db']:5.2f} dB  "
                f"{row['solve_seconds']:6.1f}s/seed  "
                f"[wall {time.perf_counter() - t0:.0f}s]",
                flush=True,
            )
            # Written after every point, so a partial run is still usable.
            with open(args.out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                w.writeheader()
                w.writerows(rows)

    print(json.dumps({"rows": len(rows), "out": args.out}))


if __name__ == "__main__":
    main()
