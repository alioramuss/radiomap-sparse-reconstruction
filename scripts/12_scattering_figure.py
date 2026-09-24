"""Figure and summary for the diffuse scattering result.

(a) the noise floor against ray count for each S: with scattering off the
    estimator obeys N^-1/2, with scattering on it essentially stops converging.
(b) where that comes from: the per-cell disagreement is heavy tailed.  The
    middle of the map is fine even with scattering on.  A few percent of cells,
    the ones only diffuse paths ever reach, disagree by tens of dB and they are
    what any RMSE is actually reporting.
(c) the scattering coefficient that surface roughness implies, which rises
    steeply with frequency and so runs opposite to the expectation under test.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from importlib import import_module
rayleigh_S = import_module("11_scattering_sweep").rayleigh_S

BAND_COLOUR = {3.5: "#0072B2", 28.0: "#D55E00", 39.0: "#009E73"}
S_STYLE = {0.0: ("-", "o"), 0.3: ("--", "s"), 0.6: (":", "^")}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/scattering_sweep.csv")
    p.add_argument("--cache-dir", default="data/scattering")
    p.add_argument("--out", default="figures/fig_scattering.png")
    args = p.parse_args()

    rows = [{k: float(v) for k, v in r.items()} for r in csv.DictReader(open(args.csv))]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

    # (a) noise floor
    for f in (3.5, 28.0, 39.0):
        for S in (0.0, 0.3, 0.6):
            sub = sorted([r for r in rows if r["frequency_ghz"] == f
                          and r["scattering_coefficient"] == S],
                         key=lambda r: r["rays_per_tx"])
            n = [r["rays_per_tx"] for r in sub]
            y = [r["run_to_run_rmse_db"] for r in sub]
            ls, mk = S_STYLE[S]
            ax1.plot(n, y, ls, marker=mk, ms=4, lw=1.5, color=BAND_COLOUR[f], mfc="none",
                     label=f"{f:g} GHz, S={S:g}" if f == 3.5 or S == 0.0 else None)
    nref = np.array([5e6, 1e8])
    ax1.plot(nref, 2.5 * (nref / 5e6) ** -0.5, "k--", lw=1.0, alpha=0.5, label="$N^{-1/2}$")
    ax1.set(xscale="log", yscale="log", xlabel="Rays per transmitter",
            ylabel="Run-to-run RMSE (dB)", title="(a) Scattering breaks convergence")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7, ncol=2, loc="lower left")

    # (b) tails
    qs = np.array([50, 75, 90, 95, 99, 99.9])
    for f in (3.5, 39.0):
        for S in (0.0, 0.6):
            a = np.load(os.path.join(args.cache_dir, f"f{f:g}_S{S:g}_n100000000_s1.npz"))
            b = np.load(os.path.join(args.cache_dir, f"f{f:g}_S{S:g}_n100000000_s2.npz"))
            ok = (~a["building"]) & np.isfinite(a["grid"]) & np.isfinite(b["grid"])
            d = np.abs(a["grid"][ok].astype(float) - b["grid"][ok].astype(float))
            ax2.plot(qs, np.percentile(d, qs), marker="o" if S == 0 else "^", ms=4,
                     lw=1.5, color=BAND_COLOUR[f], ls="-" if S == 0 else ":",
                     mfc="none", label=f"{f:g} GHz, S={S:g}")
    ax2.set(yscale="log", xlabel="Percentile of street cells",
            ylabel="|run-to-run difference| (dB)",
            title="(b) The middle is fine, the tail is not")
    ax2.grid(True, which="both", alpha=0.25)
    ax2.legend(fontsize=8, loc="upper left")

    # (c) roughness prediction
    freqs = np.linspace(2, 42, 250)
    for sigma, style in ((0.5, "-"), (1.0, "--")):
        ax3.plot(freqs, [rayleigh_S(f, sigma) for f in freqs], style, color="#444444",
                 lw=1.6, label=f"$\\sigma_h$ = {sigma:g} mm")
    for f in (3.5, 28.0, 39.0):
        ax3.axvline(f, color=BAND_COLOUR[f], alpha=0.35, lw=1.2)
        ax3.text(f + 0.7, 0.92, f"{f:g} GHz", color=BAND_COLOUR[f], fontsize=8)
    # The hypothesis under test says scattering WEAKENS with frequency.  Roughness
    # says the opposite, and steeply, so draw both so the conflict is visible.
    ax3.annotate("", xy=(40, 0.12), xytext=(6, 0.62),
                 arrowprops=dict(arrowstyle="->", color="#B00020", lw=1.8, ls="--"))
    ax3.text(11.5, 0.30, "hypothesis under test:\nweaker at mmWave", fontsize=8,
             color="#B00020", ha="left")
    ax3.text(20.0, 0.80, "roughness model", fontsize=8, color="#333333", rotation=14)
    ax3.set(xlabel="Carrier frequency (GHz)", ylabel="Implied scattering coefficient $S$",
            ylim=(0, 1.05), title="(c) Roughness points the other way")
    ax3.grid(True, alpha=0.25)
    ax3.legend(fontsize=8, loc="lower right")

    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
