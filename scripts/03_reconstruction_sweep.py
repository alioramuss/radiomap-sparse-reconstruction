"""Run both sampling designs across all methods and sampling fractions.

    python scripts/03_reconstruction_sweep.py --map data/map_3p5GHz.npz \
        --out results/sweep.csv

Add ``--synthetic`` to run the whole pipeline on the stand-in field in
``radiomap.scene.synthetic_canyon`` without Sionna.  That is for checking the
plumbing, not for producing results: no number in the report comes from it.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from radiomap.config import SweepConfig
from radiomap.interpolators import METHOD_LABELS
from radiomap.scene import RadioMapData, synthetic_canyon
from radiomap.sweep import block_sweep, random_sweep, summarise, write_csv
from radiomap.variogram import empirical_variogram, fit_spherical


def report_variogram(rm: RadioMapData, cfg: SweepConfig) -> None:
    """Fit the variogram on a 10% draw, which is what a real user would have."""
    rng = np.random.default_rng(cfg.variogram_fit_seed)
    idx = rng.choice(
        rm.n_valid,
        size=max(50, int(cfg.variogram_fit_fraction * rm.n_valid)),
        replace=False,
    )
    lags, gamma, counts = empirical_variogram(rm.xy[idx], rm.gain_db[idx])
    model = fit_spherical(lags, gamma, counts)
    print(
        f"variogram on a {100 * cfg.variogram_fit_fraction:.0f}% draw: "
        f"range {model.rng:.1f} m, nugget {model.nugget:.2f}, "
        f"sill {model.sill:.1f} (nugget is {100 * model.nugget / max(model.total_sill, 1e-9):.1f}% of total)"
    )
    print(
        f"block side {cfg.block_side_m:.0f} m is "
        f"{cfg.block_side_m / max(model.rng, 1e-9):.1f}x the correlation range"
    )


def print_table(summary, design: str, cfg: SweepConfig) -> None:
    print(f"\nRMSE in dB, mean +/- sd -- {design}")
    head = "  ".join(f"{100 * f:>4.0f}%" for f in cfg.sampling_fractions)
    print(f"{'method':<20}{head}")
    for m in cfg.methods:
        cells = []
        for f in cfg.sampling_fractions:
            mean, sd, _n = summary.get((design, m, f), (np.nan, np.nan, 0))
            cells.append(f"{mean:5.2f}+-{sd:<4.2f}")
        print(f"{METHOD_LABELS.get(m, m):<20}" + " ".join(cells))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="data/map_3p5GHz.npz")
    p.add_argument("--out", default="results/sweep.csv")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--design", choices=("both", "random", "block"), default="both")
    args = p.parse_args()

    cfg = SweepConfig()
    rm = synthetic_canyon() if args.synthetic else RadioMapData.load(args.map)
    print(
        f"{rm.n_valid} valid cells of {rm.grid_shape[0] * rm.grid_shape[1]} "
        f"({100 * rm.n_valid / (rm.grid_shape[0] * rm.grid_shape[1]):.1f}%), "
        f"{rm.gain_db.min():.0f} to {rm.gain_db.max():.0f} dB, "
        f"sd {rm.gain_db.std():.1f} dB"
    )
    report_variogram(rm, cfg)

    rows = []
    if args.design in ("both", "random"):
        rows += random_sweep(rm, cfg)
    if args.design in ("both", "block"):
        rows += block_sweep(rm, cfg)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_csv(rows, args.out)

    summary = summarise(rows)
    for design in ("random", "block"):
        if any(k[0] == design for k in summary):
            print_table(summary, design, cfg)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
