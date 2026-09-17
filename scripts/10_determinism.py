"""How much of the run-to-run spread is the seed, and how much is the solver?

Sionna RT 2.0.1 on the LLVM backend does not reproduce a map from its seed.
Eight runs at an identical seed give eight distinct maps.  That is worth
quantifying rather than mentioning, because it decides whether the convergence
table measures what it claims.

Two questions:

* how big is the same-seed spread next to the different-seed spread, and
* does it fall as N^-1/2 as well?  If it does, the unseeded part is simply a
  fraction of the sampling and the convergence table is unaffected: a user who
  runs the solver once draws from the total, which is what different-seed pairs
  measure.  If instead it flattened out, there would be a reproducibility floor
  that no ray count removes, and that would matter.

    python scripts/10_determinism.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import itertools

import numpy as np

from radiomap.config import SceneConfig
from radiomap.scene import trace

RAY_COUNTS = [5e6, 2e7, 5e7]
N_REPEATS = 4
FIXED_SEED = 7
DIFFERENT_SEEDS = (11, 12, 13)


def grid(seed: int, n: float, freq_ghz: float) -> np.ndarray:
    m = trace(
        SceneConfig(frequency_hz=freq_ghz * 1e9, samples_per_tx=int(n), seed=seed),
        verbose=False,
    )
    return m.to_grid(m.gain_db)


def pair_rmse(x: np.ndarray, y: np.ndarray) -> float:
    both = np.isfinite(x) & np.isfinite(y)
    return float(np.sqrt(np.mean((x[both] - y[both]) ** 2)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="results/determinism.csv")
    p.add_argument("--frequency-ghz", type=float, default=3.5)
    p.add_argument("--rays", type=float, nargs="*", default=RAY_COUNTS)
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []

    for n in args.rays:
        same = [grid(FIXED_SEED, n, args.frequency_ghz) for _ in range(N_REPEATS)]
        same_pairs = [pair_rmse(a, b) for a, b in itertools.combinations(same, 2)]
        # Pairs that landed on the same execution order agree to float noise;
        # they are not evidence of determinism, only of a repeated schedule.
        nonzero = [v for v in same_pairs if v > 1e-4]

        diff = [grid(s, n, args.frequency_ghz) for s in DIFFERENT_SEEDS]
        diff_pairs = [pair_rmse(a, b) for a, b in itertools.combinations(diff, 2)]

        row = {
            "rays_per_tx": int(n),
            "same_seed_rmse_db": float(np.mean(nonzero)) if nonzero else 0.0,
            "same_seed_pairs_differing": f"{len(nonzero)}/{len(same_pairs)}",
            "different_seed_rmse_db": float(np.mean(diff_pairs)),
        }
        row["fraction_of_total"] = (
            row["same_seed_rmse_db"] / row["different_seed_rmse_db"]
            if row["different_seed_rmse_db"]
            else float("nan")
        )
        rows.append(row)
        print(
            f"{int(n):>12,} rays   same-seed {row['same_seed_rmse_db']:6.3f} dB "
            f"({row['same_seed_pairs_differing']} pairs differ)   "
            f"different-seed {row['different_seed_rmse_db']:6.3f} dB   "
            f"ratio {row['fraction_of_total']:.2f}",
            flush=True,
        )

    n = np.array([r["rays_per_tx"] for r in rows], dtype=float)
    for label, key in (("same-seed", "same_seed_rmse_db"), ("different-seed", "different_seed_rmse_db")):
        y = np.array([r[key] for r in rows], dtype=float)
        if np.all(y > 0):
            alpha = float(np.polyfit(np.log10(n), np.log10(y), 1)[0])
            print(f"{label} spread scales as N^{alpha:.3f}")

    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
