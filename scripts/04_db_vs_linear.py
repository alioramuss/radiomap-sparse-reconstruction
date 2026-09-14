"""Ablation: does it matter whether you interpolate in dB or in linear power?

Everything in the main sweep interpolates in dB, which is the field convention
and the domain where the residuals are approximately Gaussian.  Part of the
literature scores in linear power instead, which weights cells near the
transmitter far more heavily.  No one appears to have quantified the
difference, and it is a small ablation to run.

    python scripts/04_db_vs_linear.py --map data/map_3p5GHz.npz
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import numpy as np

from radiomap.config import SweepConfig
from radiomap.crossval import select
from radiomap.interpolators import METHOD_LABELS, METHODS
from radiomap.metrics import rmse, to_db, to_linear
from radiomap.sampling import random_design
from radiomap.scene import RadioMapData, synthetic_canyon


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="data/map_3p5GHz.npz")
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--fraction", type=float, default=0.05)
    args = p.parse_args()

    cfg = SweepConfig()
    rm = synthetic_canyon() if args.synthetic else RadioMapData.load(args.map)

    print(f"fraction {100 * args.fraction:.0f}%, both scored as RMSE in dB\n")
    print(f"{'method':<20}{'fit in dB':>12}{'fit in linear':>16}")

    for method in cfg.methods:
        extra = {"tx_xy": rm.tx_xy} if method == "dk" else {}
        db_errs, lin_errs = [], []

        for seed in cfg.random_seeds:
            obs, sc = random_design(rm.n_valid, args.fraction, seed)
            xy_o, xy_h = rm.xy[obs], rm.xy[sc]
            z_o, z_h = rm.gain_db[obs], rm.gain_db[sc]

            params, _ = select(
                method, xy_o, z_o, folds=cfg.cv_folds, seed=cfg.cv_seed, extra=extra
            )
            m = METHODS[method](**{**params, **extra}).fit(xy_o, z_o)
            db_errs.append(rmse(z_h, m.predict(xy_h)))

            # Same split, same hyperparameters, fitted on linear power and
            # converted back so the two are scored on identical footing.
            lin_o = to_linear(z_o)
            m2 = METHODS[method](**{**params, **extra}).fit(xy_o, lin_o)
            pred_lin = np.maximum(m2.predict(xy_h), 1e-30)
            lin_errs.append(rmse(z_h, to_db(pred_lin)))

        print(
            f"{METHOD_LABELS.get(method, method):<20}"
            f"{np.mean(db_errs):>9.2f} dB{np.mean(lin_errs):>13.2f} dB"
        )


if __name__ == "__main__":
    main()
