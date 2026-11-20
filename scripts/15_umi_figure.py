"""Figure for the 38.901 reference-model comparison (--model umi|uma).

(a) path loss against 3D distance at 28 GHz, ray-traced cells with the UMi
    LOS and NLOS means and their one-sigma shadow fading bands;
(b) NLOS residual (ray traced minus UMi) against distance, median and
    interquartile range per band;
(c) the per-cell frequency coefficient, LOS and NLOS, with free space (2.0),
    the 38.901 UMi NLOS value (2.13) and single-edge diffraction (3.0) marked;
(d) where in the canyon each coefficient sits.

    python scripts/15_umi_figure.py --data data/bands_los.npz --outdir figures
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from radiomap import threegpp as g

BAND_COLOUR = {3.5: "#0072B2", 28.0: "#D55E00", 39.0: "#009E73"}
LOS_C, NLOS_C = "#0072B2", "#D55E00"
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#e6e6e6"
H_UT = 1.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bands_los.npz")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--band", type=float, default=28.0)
    ap.add_argument("--model", choices=("umi", "uma"), default="umi",
                    help="38.901 reference model. UMa is nominal for a 25 m "
                         "base station, UMi for 10 m; this transmitter is a "
                         "32 m rooftop site, so UMa is the closer match.")
    ap.add_argument("--ieee", action="store_true",
                    help="size for an IEEE double-column figure* (7.16 in) "
                         "with fonts legible at that width")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = np.load(args.data)
    bands = [float(b) for b in d["bands_ghz"]]
    c, tx, los, bld = d["centres"], d["tx"], d["los"], d["building"]
    d2d = np.hypot(c[..., 0] - tx[0], c[..., 1] - tx[1])
    d3d = np.sqrt(d2d ** 2 + (tx[2] - H_UT) ** 2)
    pl = {f: -d[f"gain_db_{f:g}"] for f in bands}
    m = ~bld & (d2d >= 10)
    for f in bands:
        m &= np.isfinite(pl[f])
    hbs = float(tx[2])

    # An IEEE double-column figure is 7.16 in wide. Rendering at screen size and
    # letting LaTeX shrink it is what makes figure text unreadable in print, so
    # render at the final width and set the fonts for it.
    base = 8.0 if args.ieee else 8.5
    plt.rcParams.update({"font.size": base, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.titlesize": base + 1.0,
                         "axes.titleweight": "bold", "axes.titlelocation": "left"})
    figsize = (7.16, 5.6) if args.ieee else (10.5, 8.2)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    (a, b), (cax, dax) = axs

    # (a) scatter at one band
    f = args.band
    dd = np.linspace(d3d[m].min(), d3d[m].max(), 200)
    d2 = np.sqrt(np.maximum(dd ** 2 - (hbs - H_UT) ** 2, 0))
    for cls, cm, col in (("NLOS", m & ~los, NLOS_C), ("LOS", m & los, LOS_C)):
        a.scatter(d3d[cm], pl[f][cm], s=1.2, alpha=0.25, color=col, lw=0, rasterized=True)
    MODEL = {"umi": (g.umi_los, g.umi_nlos), "uma": (g.uma_los, g.uma_nlos)}[args.model]
    MNAME = args.model.upper().replace("UMI", "UMi").replace("UMA", "UMa")
    for cls, fn, col in (("los", MODEL[0], LOS_C), ("nlos", MODEL[1], NLOS_C)):
        mu = fn(d2, dd, f, hbs, H_UT)
        s = g.SIGMA_SF[(args.model, cls)]
        a.plot(dd, mu, color=col, lw=2)
        a.fill_between(dd, mu - s, mu + s, color=col, alpha=0.12, lw=0)
    a.set_xscale("log")
    a.set_ylabel("path loss (dB)")
    a.set_title(f"(a) Cells vs {MNAME}, {f:g} GHz" if args.ieee
                else f"(a) Ray-traced cells vs 38.901 {MNAME} at {f:g} GHz")
    a.set_xlabel("3D distance (m)" if args.ieee else
                 "3D distance (m)   ·   dots: Sionna RT cells, lines: model mean, "
                 "band: ±1σ shadow fading")
    a.text(0.98, 0.05, "LOS", color=LOS_C, transform=a.transAxes, ha="right", fontweight="bold")
    a.text(0.98, 0.12, "NLOS", color=NLOS_C, transform=a.transAxes, ha="right", fontweight="bold")
    a.invert_yaxis()

    # (b) NLOS residual vs distance
    edges = np.arange(10, d2d[m].max() + 10, 10)
    mid = 0.5 * (edges[1:] + edges[:-1])
    cm = m & ~los
    for f in bands:
        mu = MODEL[1](d2d, d3d, f, hbs, H_UT)
        r = pl[f] - mu
        med, q1, q3 = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = cm & (d2d >= lo) & (d2d < hi)
            if sel.sum() < 30:
                med.append(np.nan); q1.append(np.nan); q3.append(np.nan); continue
            v = r[sel]
            med.append(np.median(v)); q1.append(np.percentile(v, 25)); q3.append(np.percentile(v, 75))
        b.plot(mid, med, color=BAND_COLOUR[f], lw=2, label=f"{f:g} GHz")
        b.fill_between(mid, q1, q3, color=BAND_COLOUR[f], alpha=0.12, lw=0)
    s = g.SIGMA_SF[(args.model, "nlos")]
    b.axhspan(-s, s, color="#999999", alpha=0.12, lw=0)
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xlabel("2D distance from transmitter (m)")
    b.set_ylabel(f"ray traced − {MNAME} NLOS (dB)")
    b.set_title("(b) NLOS residual versus distance" if args.ieee
                else "(b) NLOS residual versus distance, per band")
    b.text(0.02, 0.04,
           f"grey: ±{s} dB {MNAME} NLOS shadow fading" if args.ieee
           else f"grey band: ±{s} dB, the model's own NLOS shadow fading",
           transform=b.transAxes, fontsize=base - 1.0, color=MUTED)
    b.legend(frameon=False, loc="upper left")

    # (c) frequency coefficient histogram
    x = 10 * np.log10(np.asarray(bands))
    xc = x - x.mean()
    bins = np.linspace(1.4, 3.4, 81)
    for cls, cm2, col in (("LOS", m & los, LOS_C), ("NLOS", m & ~los, NLOS_C)):
        Y = np.stack([pl[f][cm2] for f in bands], axis=1)
        sl = (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)
        cax.hist(np.clip(sl, bins[0], bins[-1]), bins=bins, color=col, alpha=0.75,
                 label=f"{cls}  (median {np.median(sl):.2f})", density=True)
    # At the narrow width the three labels crowd each other, so the leftmost
    # one hangs into the empty left tail instead of across the 2.13 line.
    # UMa's NLOS frequency term is 20log10(f), so its reference coincides with
    # free space; UMi's is 21.3log10(f).  Mark whichever model is in use.
    ref = 2.0 if args.model == "uma" else 2.13
    marks = [(2.0, "free space\n" + ("= UMa NLOS" if args.model == "uma" else "= UMi LOS"),
              "right" if args.ieee else "left"),
             (3.0, "single-edge\ndiffraction", "left")]
    if args.model == "umi":
        marks.insert(1, (2.13, "UMi NLOS", "left"))
    for v, lab, side in marks:
        cax.axvline(v, color=INK, lw=0.9, ls="--")
        dx = 0.02 if side == "left" else -0.03
        cax.text(v + dx, cax.get_ylim()[1] * (0.70 if v == 2.13 else 0.93),
                 lab, fontsize=base - 1.0, color=INK, va="top", ha=side)
    cax.set_xlabel(
        "frequency coefficient (dB of path loss per dB of $f_c$)\n"
        "values below 1.4 gathered into the first bin"
        if args.ieee else
        "frequency coefficient: dB of path loss per dB of carrier frequency\n"
        "(values below 1.4 are gathered into the first bin)")
    cax.set_ylabel("density of cells")
    cax.set_title("(c) Frequency scaling, cell by cell" if args.ieee
                  else "(c) How path loss scales with frequency, cell by cell")
    cax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.36, 0.62))

    # (d) where
    Y = np.stack([pl[f] for f in bands], axis=-1)
    sl_grid = ((Y - Y.mean(-1, keepdims=True)) @ xc) / (xc @ xc)
    sl_grid = np.where(m, sl_grid, np.nan)
    ext = [c[..., 0].min() - 0.5, c[..., 0].max() + 0.5, c[..., 1].min() - 0.5, c[..., 1].max() + 0.5]
    bg = np.where(bld, 1.0, np.nan)
    dax.imshow(bg, origin="lower", extent=ext, cmap="Greys", vmin=0, vmax=3, interpolation="nearest")
    im = dax.imshow(sl_grid, origin="lower", extent=ext, cmap="cividis", vmin=1.8, vmax=3.1,
                    interpolation="nearest")
    th = np.linspace(0, 2*np.pi, 361)
    dax.plot(tx[0] + 80*np.cos(th), tx[1] + 80*np.sin(th), color="white",
             lw=1.2, ls="--", alpha=0.9)
    dax.plot(tx[0] + 80*np.cos(th), tx[1] + 80*np.sin(th), color=INK,
             lw=0.6, ls="--", alpha=0.6)
    dax.plot(tx[0], tx[1], marker="*", ms=13, color="white", mec=INK, mew=1)
    # the 80 m ring runs off the scene, so clip back to the map extent
    dax.set_xlim(ext[0], ext[1]); dax.set_ylim(ext[2], ext[3])
    dax.set_title("(d) Where each regime sits" if args.ieee
                  else "(d) Where each regime sits (grey: buildings, star: tx)")
    dax.set_xlabel("x (m)")
    dax.set_ylabel("y (m)")
    cb = fig.colorbar(im, ax=dax, fraction=0.035, pad=0.02)
    cb.set_label("frequency coefficient", color=INK)

    for ax in (a, b, cax):
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    if not args.ieee:
        # In the paper the caption carries this, and a suptitle inside a
        # figure* only steals height from the panels.
        fig.suptitle("Sionna RT street canyon against 3GPP 38.901 UMi at 3.5, 28 and 39 GHz",
                     x=0.01, ha="left", fontsize=base + 2.0, fontweight="bold", color=INK)
    rect = (0, 0, 1, 1) if args.ieee else (0, 0, 1, 0.96)
    fig.tight_layout(rect=rect, w_pad=1.4, h_pad=1.4)
    tag = "" if args.model == "umi" else f"_{args.model}"
    name = (f"umi_comparison{tag}_ieee.png" if args.ieee
            else f"umi_comparison{tag}.png")
    out = os.path.join(args.outdir, name)
    fig.savefig(out, dpi=400 if args.ieee else 170)
    print("wrote", out)


if __name__ == "__main__":
    main()
