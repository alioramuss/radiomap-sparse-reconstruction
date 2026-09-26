"""Sionna RT against 3GPP 38.901 UMi-Street Canyon at 3.5, 28 and 39 GHz.

The comparison Tianrun Qi proposed on 14 September: how far the ray-traced
street canyon sits from the statistical model in each band, and how that gap
changes with frequency.  Reads the output of scripts/13_trace_bands_los.py.

Residual convention: residual = PL_raytraced - PL_model, in dB.  Positive means
the ray tracer loses MORE than the model says.

What is computed, per band and separately for LOS and NLOS cells:

1. bias, spread and RMSE of the residual, and what fraction of cells fall
   within one shadow-fading sigma of the model (68% if the model's own
   statistics held on this scene);
2. the path loss exponent fitted to the ray-traced cells, next to the
   model's (UMi: 2.1 LOS, 3.53 NLOS);
3. the frequency coefficient per cell: path loss regressed on 10 log10(fc)
   across the three bands, so free space would give exactly 2.0 and the
   38.901 NLOS formula 2.13;
4. a naive cross-frequency translation: take the 3.5 GHz map, add
   20 log10(f / 3.5) and score it against the traced map at 28 and 39 GHz,
   next to the 38.901 prediction for the same cells.  A first look at Qi's
   stretch goal of predicting one band from another.

UMi is fitted for a 10 m base station and this transmitter is on a 32 m
rooftop, so UMa (25 m nominal) is run as a sensitivity check alongside.

    python scripts/14_umi_comparison.py --data data/bands_los.npz --outdir results
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json

import numpy as np

from radiomap import threegpp as g

H_UT = 1.5


def load(path):
    d = np.load(path)
    bands = [float(b) for b in d["bands_ghz"]]
    c = d["centres"]
    tx = d["tx"]
    d2d = np.hypot(c[..., 0] - tx[0], c[..., 1] - tx[1])
    d3d = np.sqrt(d2d ** 2 + (tx[2] - H_UT) ** 2)
    pl = {f: -d[f"gain_db_{f:g}"] for f in bands}
    return dict(bands=bands, d2d=d2d, d3d=d3d, los=d["los"], building=d["building"],
                pl=pl, h_bs=float(tx[2]), centres=c)


def base_mask(D):
    """Street cells inside the 38.901 distance range, reached in every band."""
    m = ~D["building"] & g.valid("umi", D["d2d"], 3.5)
    for f in D["bands"]:
        m &= np.isfinite(D["pl"][f])
    return m


def fit_exponent(pl, d3d):
    """PL = A + 10 n log10(d3d); returns (n, A, residual sd)."""
    x = 10 * np.log10(d3d)
    A = np.vstack([x, np.ones_like(x)]).T
    (n, a), *_ = np.linalg.lstsq(A, pl, rcond=None)
    return float(n), float(a), float(np.std(pl - (n * x + a)))


def stats(res, sigma):
    return dict(
        n=int(res.size),
        bias_db=float(np.mean(res)),
        median_db=float(np.median(res)),
        sd_db=float(np.std(res)),
        rmse_db=float(np.sqrt(np.mean(res ** 2))),
        within_1sigma=float(np.mean(np.abs(res) <= sigma)),
        p05_db=float(np.percentile(res, 5)),
        p95_db=float(np.percentile(res, 95)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bands_los.npz")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    D = load(args.data)
    m = base_mask(D)
    los = D["los"]
    classes = {"los": m & los, "nlos": m & ~los}
    summary = {"cells": {k: int(v.sum()) for k, v in classes.items()},
               "h_bs_m": D["h_bs"], "h_ut_m": H_UT, "bands_ghz": D["bands"],
               "excluded_d2d_below_10m": int((~D["building"] & (D["d2d"] < 10)).sum())}

    rows = []
    for model in ("umi", "uma"):
        for f in D["bands"]:
            pl_model = g.path_loss(model, los, D["d2d"], D["d3d"], f, D["h_bs"], H_UT)
            for cls, cm in classes.items():
                res = D["pl"][f][cm] - pl_model[cm]
                s = stats(res, g.SIGMA_SF[(model, cls)])
                n_rt, a_rt, sd_rt = fit_exponent(D["pl"][f][cm], D["d3d"][cm])
                rows.append(dict(model=model, band_ghz=f, cls=cls,
                                 sigma_sf_db=g.SIGMA_SF[(model, cls)],
                                 fitted_exponent=n_rt, fitted_intercept_db=a_rt,
                                 fit_residual_sd_db=sd_rt, **s))

    with open(os.path.join(args.outdir, "umi_comparison.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ---- frequency coefficient per cell -------------------------------------
    x = 10 * np.log10(np.asarray(D["bands"]))
    xc = x - x.mean()
    freq = {}
    for cls, cm in classes.items():
        Y = np.stack([D["pl"][f][cm] for f in D["bands"]], axis=1)
        slope = (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)   # dB per dB of fc
        pair_lo = (Y[:, 1] - Y[:, 0]) / (x[1] - x[0])
        pair_hi = (Y[:, 2] - Y[:, 1]) / (x[2] - x[1])
        freq[cls] = dict(
            median=float(np.median(slope)), p10=float(np.percentile(slope, 10)),
            p90=float(np.percentile(slope, 90)), mean=float(np.mean(slope)),
            median_3p5_to_28=float(np.median(pair_lo)),
            median_28_to_39=float(np.median(pair_hi)),
        )
        np.save(os.path.join(args.outdir, f"freq_slope_{cls}.npy"), slope)
    summary["frequency_coefficient"] = freq
    summary["frequency_coefficient_model"] = {"free_space": 2.0, "umi_los": 2.0,
                                              "umi_nlos": 2.13, "uma_nlos": 2.0}

    # ---- naive cross-frequency translation ----------------------------------
    trans = {}
    f0 = D["bands"][0]
    for f in D["bands"][1:]:
        shift = 20 * np.log10(f / f0)
        pl_umi = g.path_loss("umi", los, D["d2d"], D["d3d"], f, D["h_bs"], H_UT)
        for cls, cm in classes.items():
            e_tr = D["pl"][f][cm] - (D["pl"][f0][cm] + shift)
            e_umi = D["pl"][f][cm] - pl_umi[cm]
            trans[f"{f:g}_{cls}"] = dict(
                translate_rmse_db=float(np.sqrt(np.mean(e_tr ** 2))),
                translate_bias_db=float(np.mean(e_tr)),
                translate_sd_db=float(np.std(e_tr)),
                umi_rmse_db=float(np.sqrt(np.mean(e_umi ** 2))),
            )
    summary["cross_frequency_translation_from_3p5"] = trans

    with open(os.path.join(args.outdir, "umi_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    # ---- print ---------------------------------------------------------------
    print(f"cells: LOS {summary['cells']['los']}, NLOS {summary['cells']['nlos']} "
          f"(d2D >= 10 m; {summary['excluded_d2d_below_10m']} street cells closer excluded)")
    for model in ("umi", "uma"):
        print(f"\n{model.upper()}  residual = raytraced PL - model PL")
        print(f"{'band':>6} {'cls':>5} {'bias':>7} {'sd':>6} {'rmse':>6} {'in1sig':>7} {'n_fit':>6} {'model n':>8}")
        for r in rows:
            if r["model"] != model:
                continue
            mn = {("umi", "los"): 2.1, ("umi", "nlos"): 3.53, ("uma", "los"): 2.2, ("uma", "nlos"): 3.908}[(model, r["cls"])]
            print(f"{r['band_ghz']:>6g} {r['cls']:>5} {r['bias_db']:>7.2f} {r['sd_db']:>6.2f} "
                  f"{r['rmse_db']:>6.2f} {100*r['within_1sigma']:>6.1f}% {r['fitted_exponent']:>6.2f} {mn:>8.2f}")
    print("\nfrequency coefficient (dB of PL per dB of fc):")
    for cls, v in freq.items():
        print(f"  {cls:>4}: median {v['median']:.3f} [p10 {v['p10']:.3f}, p90 {v['p90']:.3f}]  "
              f"3.5->28 {v['median_3p5_to_28']:.3f}  28->39 {v['median_28_to_39']:.3f}")
    print("\ntranslate 3.5 GHz map by 20log10(f/3.5) vs 38.901 UMi, RMSE dB:")
    for k, v in trans.items():
        print(f"  {k:>9}: translate {v['translate_rmse_db']:.2f} (bias {v['translate_bias_db']:+.2f})  "
              f"UMi {v['umi_rmse_db']:.2f}")


if __name__ == "__main__":
    main()
