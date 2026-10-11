"""Figure for the roughness sweep (scripts 18 and 19).

(a) NLOS bias against 38.901 UMi as roughness rises, per band.
(b) share of NLOS cells inside UMi's own one-sigma shadow fading band; the
    model itself implies 68%.
(c) the per-cell NLOS frequency coefficient, median and 10th to 90th
    percentile, against the single-edge 3.0 and UMi's 2.13.
(d) how far to trust (a) to (c): run-to-run disagreement in the far zone and
    how much the far-zone median still moves between 2e7 and 1e8 rays.

    python scripts/20_roughness_figure.py
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BAND_COLOUR = {3.5: "#0072B2", 28.0: "#D55E00", 39.0: "#009E73"}
BAND_MARK = {3.5: "o", 28.0: "s", 39.0: "^"}
ROUGH = (0.0, 0.2, 0.5, 1.0, 2.0)
INK, MUTED = "#222222", "#6b6b6b"


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=10.5, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.set_xticks(range(len(ROUGH)))
    ax.set_xticklabels(["off"] + [f"{h:g}" for h in ROUGH[1:]])
    ax.set_xlabel("Surface roughness $\\sigma_h$ (mm)", fontsize=9, color=INK)
    ax.grid(True, axis="y", alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=8.5, colors=MUTED)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/roughness_summary.json")
    ap.add_argument("--out", default="figures/fig_roughness.png")
    args = ap.parse_args()
    D = json.load(open(args.summary))
    rows = D["rows"]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    x = np.arange(len(ROUGH))

    def series(f, key):
        return [next(r for r in rows if r["band_ghz"] == f and r["roughness_mm"] == h)[key] for h in ROUGH]

    fig, axs = plt.subplots(2, 2, figsize=(11, 8.2))
    (a, b), (c, d) = axs

    for f in BAND_COLOUR:
        kw = dict(color=BAND_COLOUR[f], marker=BAND_MARK[f], ms=6, lw=2, mfc="white", mew=1.6)
        ya = series(f, "nlos_umi_bias_mean_db")
        a.plot(x, ya, **kw, label=f"{f:g} GHz")
        yb = [100 * v for v in series(f, "nlos_within_1sigma")]
        b.plot(x, yb, **kw, label=f"{f:g} GHz")
        d.plot(x, series(f, "nlos_far_run_diff_median_db"), **kw, label=f"{f:g} GHz: run-to-run, median |diff|")
        d.plot(x, [-v for v in series(f, "nlos_far_drift_2e7_minus_1e8_median_db")],
               color=BAND_COLOUR[f], marker=BAND_MARK[f], ms=5, lw=1.4, ls=":", mfc=BAND_COLOUR[f],
               label=f"{f:g} GHz: median rise 2e7 to 1e8 rays")

    a.axhline(0, color=MUTED, lw=1)
    style(a, "(a) NLOS mean bias against UMi", "Ray-traced PL minus UMi (dB)")
    a.legend(fontsize=8, frameon=False, loc="upper right")

    b.axhline(68.3, color=MUTED, lw=1, ls="--")
    b.text(0.05, 69.5, "68% expected if UMi's own statistics held", fontsize=8, color=MUTED)
    style(b, "(b) NLOS cells within UMi's $\\pm1\\sigma$ (7.82 dB)", "Share of NLOS cells (%)")
    b.set_ylim(0, 80)
    b.legend(fontsize=8, frameon=False, loc="lower right")

    fc = D["freq_coef"]
    med = [fc[f"{h:g}_nlos"]["median"] for h in ROUGH]
    lo = [fc[f"{h:g}_nlos"]["p10"] for h in ROUGH]
    hi = [fc[f"{h:g}_nlos"]["p90"] for h in ROUGH]
    far = [fc[f"{h:g}_nlos_far"]["median"] for h in ROUGH]
    c.fill_between(x, lo, hi, color="#8c8c8c", alpha=0.18, lw=0, label="all NLOS, 10th to 90th pct")
    c.plot(x, med, color=INK, marker="o", ms=6, lw=2, mfc="white", mew=1.6, label="all NLOS, median")
    c.plot(x, far, color=INK, marker="D", ms=5, lw=1.4, ls="--", mfc="white", label="far NLOS (>80 m), median")
    for yv, txt in ((3.0, "single diffracting edge, 3.0"), (2.13, "UMi NLOS, 2.13"), (2.0, "free space, 2.0")):
        c.hlines(yv, -0.3, len(ROUGH) - 0.85, color=MUTED, lw=0.9, ls=(0, (4, 3)))
        c.text(len(ROUGH) - 0.8, yv + {3.0: 0.0, 2.13: 0.06, 2.0: -0.06}[yv], txt, fontsize=7.5,
               color=MUTED, ha="left", va="center")
    style(c, "(c) How NLOS path loss scales with frequency", "dB of path loss per dB of $f_c$")
    c.set_ylim(0, 3.4)
    c.set_xlim(-0.3, len(ROUGH) + 0.75)
    c.legend(fontsize=8, frameon=False, loc="lower left")

    style(d, "(d) How far to trust it: far NLOS zone", "dB")
    d.set_ylim(0, 5.6)
    d.legend(fontsize=7.2, frameon=False, loc="upper left", ncol=2)

    fig.suptitle("Roughness-driven diffuse scattering, simple_street_canyon, Sionna RT 2.0.1 "
                 "(median of 3 runs at 1e8 rays; S from Rayleigh/Ament at normal incidence)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=170)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
