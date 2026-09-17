"""Figure: the noise floor is the same in all three bands, and the map is not.

    python scripts/08_multiband_figure.py

(a) the noise floor on log-log axes against the N^-1/2 reference;
(b) each band's noise divided by the 3.5 GHz value, which is where the answer
    actually lives: the three curves sit within a few percent of each other, so
    the scale is linear and narrow rather than log;
(c) the spread of the map itself, which does move with frequency.  Panel (c) is
    the reason (b) was worth measuring: the field genuinely gets harder edged at
    millimetre wave without that changing what the solver costs.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Colourblind-safe, and distinguishable in greyscale print.
BAND_STYLE = {
    "3.5": ("#0072B2", "o", "3.5 GHz"),
    "28": ("#D55E00", "s", "28 GHz"),
    "39": ("#009E73", "^", "39 GHz"),
}
REFERENCE_BAND = "3.5"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary", default="results/multiband_summary.json")
    p.add_argument("--out", default="figures/fig_multiband_convergence.png")
    args = p.parse_args()

    with open(args.summary, encoding="utf-8") as fh:
        summary = json.load(fh)
    bands = summary["bands"]

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.3))

    ref = np.array(
        [r["seed_to_seed_rmse_db"] for r in bands[REFERENCE_BAND]["rows"]], dtype=float
    )

    for key, band in bands.items():
        colour, marker, label = BAND_STYLE.get(key, ("#555555", "o", f"{key} GHz"))
        rows = band["rows"]
        n = np.array([r["rays_per_tx"] for r in rows], dtype=float)
        rmse = np.array([r["seed_to_seed_rmse_db"] for r in rows], dtype=float)

        ax1.plot(n, rmse, marker=marker, color=colour, lw=1.6, ms=5,
                 mfc="none", label=f"{label}  ($N^{{{band['alpha']:.2f}}}$)")
        ax2.plot(n, rmse / ref, marker=marker, color=colour, lw=1.6, ms=5,
                 mfc="none", label=label)

    n_ref = np.array([5e6, 5e8])
    first = bands[REFERENCE_BAND]
    anchor = first["c"] * (5e6 ** first["alpha"])
    ax1.plot(n_ref, anchor * (n_ref / 5e6) ** -0.5, "k--", lw=1.0, alpha=0.6,
             label="$N^{-1/2}$ reference")

    ax1.set(xscale="log", yscale="log", xlabel="Rays per transmitter",
            ylabel="Run-to-run RMSE (dB)", title="(a) Simulator noise floor")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, loc="lower left")

    ax2.axhline(1.0, color="k", lw=0.8, alpha=0.5)
    ax2.axhspan(0.95, 1.05, color="0.85", alpha=0.5, zorder=0)
    ax2.set(xscale="log", xlabel="Rays per transmitter",
            ylabel="Noise relative to 3.5 GHz", ylim=(0.88, 1.12),
            title="(b) Same floor in every band")
    ax2.text(3e5, 1.093, "shaded band: ±5%", fontsize=8, color="0.35")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8, loc="lower right")

    freqs = np.array([float(k) for k in bands], dtype=float)
    order = np.argsort(freqs)
    sds = np.array([bands[k]["field_sd_db"] for k in bands], dtype=float)[order]
    for f, sd, key in zip(freqs[order], sds, np.array(list(bands))[order]):
        colour = BAND_STYLE.get(key, ("#555555",))[0]
        ax3.bar(f"{f:g}", sd, color=colour, width=0.55)
        ax3.text(f"{f:g}", sd + 0.3, f"{sd:.1f}", ha="center", fontsize=9)
    ax3.set(xlabel="Carrier frequency (GHz)", ylabel="Spread of the map (dB)",
            ylim=(0, max(sds) * 1.18), title="(c) The map does change")
    ax3.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
