"""How much of the radio map is simulator noise?

Traces the same scene twice with different seeds at each ray count.  The two
maps should be identical and are not; the RMSE between them measures the Monte
Carlo noise directly, and a single map sits that distance divided by sqrt(2)
from the converged mean.

    python scripts/02_ray_convergence.py --out results/convergence.csv

This is worth running before comparing reconstruction errors of a few dB.  At
2e5 rays -- the count used by the closest published Sionna RT reconstruction
study -- the simulator disagrees with itself by more than the effect being
measured.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import time

import numpy as np

from radiomap.config import SceneConfig
from radiomap.scene import trace

RAY_COUNTS = [2e5, 1e6, 5e6, 2e7, 5e7, 1e8, 2e8, 5e8]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/convergence.csv")
    p.add_argument("--frequency-ghz", type=float, default=3.5)
    p.add_argument(
        "--rays",
        type=float,
        nargs="*",
        default=RAY_COUNTS,
        help="ray counts to sweep",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []

    for n_rays in args.rays:
        t0 = time.perf_counter()
        maps = []
        for seed in (1, 2):
            cfg = SceneConfig(
                frequency_hz=args.frequency_ghz * 1e9,
                samples_per_tx=int(n_rays),
                seed=seed,
            )
            maps.append(trace(cfg, verbose=False))
        elapsed = 0.5 * (time.perf_counter() - t0)

        a, b = maps
        # Compare on open-street cells reached in BOTH traces.  A cell missed by
        # one seed is a different fact about the solver and is counted
        # separately as the empty-cell fraction.
        grid_a = a.to_grid(a.gain_db)
        grid_b = b.to_grid(b.gain_db)
        street = (~a.building_mask) & (~b.building_mask)
        both = street & np.isfinite(grid_a) & np.isfinite(grid_b)

        empty_frac = 1.0 - (both.sum() / max(street.sum(), 1))
        seed_rmse = float(np.sqrt(np.mean((grid_a[both] - grid_b[both]) ** 2)))
        one_vs_mean = seed_rmse / np.sqrt(2.0)

        row = {
            "rays_per_tx": int(n_rays),
            "empty_street_cells_pct": 100.0 * empty_frac,
            "seed_to_seed_rmse_db": seed_rmse,
            "one_map_vs_mean_db": one_vs_mean,
            "solve_seconds": elapsed,
        }
        rows.append(row)
        print(
            f"{int(n_rays):>12,} rays  "
            f"empty {row['empty_street_cells_pct']:5.2f}%  "
            f"seed-to-seed {seed_rmse:5.2f} dB  "
            f"one-vs-mean {one_vs_mean:5.2f} dB  "
            f"{elapsed:.0f}s"
        )

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
