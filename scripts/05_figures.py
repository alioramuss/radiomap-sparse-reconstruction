"""Figures: the error curves, and one hold-out block opened up.

    python scripts/05_figures.py --map data/map_3p5GHz.npz \
        --sweep results/sweep.csv --outdir figures

Figure 2 is the pair of error-vs-fraction curves, shaded by one standard
deviation.  Figure 3 is truth / prediction / error inside a single hold-out
block, which is where the 12 dB number stops being abstract: the error is not
diffuse, it is the shadow boundaries drawn in.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radiomap.config import SweepConfig
from radiomap.crossval import select
from radiomap.interpolators import METHOD_LABELS, METHODS
from radiomap.metrics import rmse
from radiomap.sampling import block_design, block_origins
from radiomap.scene import RadioMapData, synthetic_canyon


def load_sweep(path: str):
    agg = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            agg[(row["design"], row["method"], float(row["fraction"]))].append(
                float(row["rmse"])
            )
    return agg


def figure_curves(agg, cfg: SweepConfig, outdir: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=False)
    fracs = np.array(cfg.sampling_fractions) * 100.0

    for ax, design, title in zip(
        axes,
        ("random", "block"),
        ("Random sampling", "Contiguous 60 m hold-out"),
    ):
        for method in cfg.methods:
            means, sds = [], []
            for f in cfg.sampling_fractions:
                v = agg.get((design, method, f), [])
                means.append(np.mean(v) if v else np.nan)
                sds.append(np.std(v) if v else np.nan)
            means, sds = np.array(means), np.array(sds)
            ax.plot(fracs, means, marker="o", ms=4, label=METHOD_LABELS[method])
            ax.fill_between(fracs, means - sds, means + sds, alpha=0.12, lw=0)
        ax.set_xscale("log")
        ax.set_xticks(fracs)
        ax.set_xticklabels([f"{f:g}%" for f in fracs])
        ax.set_xlabel("cells observed")
        ax.set_ylabel("RMSE (dB)")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, frameon=False)

    fig.suptitle(
        "Same map, same methods, two designs. Bands are one standard deviation.",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, "fig2_error_curves.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def figure_block(rm: RadioMapData, cfg: SweepConfig, outdir: str, fraction: float) -> None:
    origin = block_origins(rm.xy, cfg.block_side_m, cfg.n_block_positions)[0]
    observed, scored = block_design(rm.xy, origin, cfg.block_side_m, fraction, 0)

    extra = {"tx_xy": rm.tx_xy}
    params, _ = select(
        "dk", rm.xy[observed], rm.gain_db[observed],
        folds=cfg.cv_folds, seed=cfg.cv_seed, extra=extra,
    )
    model = METHODS["dk"](**{**params, **extra})
    model.fit(rm.xy[observed], rm.gain_db[observed])
    pred = model.predict(rm.xy[scored])
    truth = rm.gain_db[scored]
    err = pred - truth

    xy = rm.xy[scored]
    x = np.unique(xy[:, 0])
    y = np.unique(xy[:, 1])
    ix = np.searchsorted(x, xy[:, 0])
    iy = np.searchsorted(y, xy[:, 1])

    def grid(v):
        g = np.full((len(y), len(x)), np.nan)
        g[iy, ix] = v
        return g

    lo = float(np.nanpercentile(truth, 2))
    hi = float(np.nanpercentile(truth, 98))
    lim = float(np.nanpercentile(np.abs(err), 98))

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    for ax, (g, title, kw) in zip(
        axes,
        [
            (grid(truth), "Truth", dict(cmap="viridis", vmin=lo, vmax=hi)),
            (grid(pred), "Prediction", dict(cmap="viridis", vmin=lo, vmax=hi)),
            (grid(err), "Error", dict(cmap="RdBu_r", vmin=-lim, vmax=lim)),
        ],
    ):
        im = ax.imshow(g, origin="lower", **kw)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, label="dB")

    fig.suptitle(
        f"Inside one {cfg.block_side_m:.0f} m hold-out block at "
        f"{100 * fraction:.0f}% sampling: RMSE {rmse(truth, pred):.2f} dB. "
        "The error is the shadow boundaries, drawn in.",
        fontsize=10,
    )
    fig.tight_layout()
    path = os.path.join(outdir, "fig3_block_error.png")
    fig.savefig(path, dpi=160)
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--map", default="data/map_3p5GHz.npz")
    p.add_argument("--sweep", default="results/sweep.csv")
    p.add_argument("--outdir", default="figures")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--synthetic", action="store_true")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = SweepConfig()
    rm = synthetic_canyon() if args.synthetic else RadioMapData.load(args.map)

    if os.path.exists(args.sweep):
        figure_curves(load_sweep(args.sweep), cfg, args.outdir)
    else:
        print(f"no sweep at {args.sweep}, skipping the error curves")

    figure_block(rm, cfg, args.outdir, args.fraction)


if __name__ == "__main__":
    main()
