"""Analysis of the roughness sweep (script 18), with its own known-answer checks.

Robust throughout, because script 11 showed the scattering-on estimator is
heavy tailed: the map for each configuration is the per-cell MEDIAN of three
independent 1e8-ray runs, and every summary is a median over cells.

Cells are the UMi comparison's cells (street, d2D >= 10 m, reached with
scattering off in every band), so the scattering-off column reproduces the
README and every other column is a cell-for-cell change against it.  NLOS is
split by 2D distance into the reflection-lit near zone (< 45 m), the middle
(45 to 80 m) and the diffraction-dominated far zone (> 80 m), as in the README.

    python scripts/19_roughness_analysis.py
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

BANDS = (3.5, 28.0, 39.0)
ROUGH = (0.0, 0.2, 0.5, 1.0, 2.0)
SEEDS = (1, 2, 3)
H_UT = 1.5


def load_cfg(cache, f, h, n, seeds):
    grids = [np.load(os.path.join(cache, f"f{f:g}_h{h:g}_n{n}_s{s}.npz"))["gain_db"].astype(float)
             for s in seeds]
    return np.stack(grids)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/roughness")
    ap.add_argument("--out", default="results/roughness_summary.json")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    G = np.load(os.path.join(args.cache_dir, "geometry.npz"))
    S = json.load(open(os.path.join(args.cache_dir, "s_table.json")))
    c, tx = G["centres"], G["tx"]
    d2d = np.hypot(c[..., 0] - tx[0], c[..., 1] - tx[1])
    d3d = np.sqrt(d2d ** 2 + (tx[2] - H_UT) ** 2)
    los, building = G["los"], G["building"]

    runs = {(f, h): load_cfg(args.cache_dir, f, h, int(1e8), SEEDS) for f in BANDS for h in ROUGH}
    drift = {(f, h): load_cfg(args.cache_dir, f, h, int(2e7), (4,))[0] for f in BANDS for h in ROUGH}
    med = {k: np.median(v, axis=0) for k, v in runs.items()}   # nan if any run missed the cell

    base = ~building & g.valid("umi", d2d, 3.5)
    for f in BANDS:
        base &= np.isfinite(med[(f, 0.0)])
    zones = {
        "los": base & los,
        "nlos": base & ~los,
        "nlos_near": base & ~los & (d2d < 45),
        "nlos_mid": base & ~los & (d2d >= 45) & (d2d <= 80),
        "nlos_far": base & ~los & (d2d > 80),
    }
    out = {"cells": {k: int(v.sum()) for k, v in zones.items()}, "S": S, "bands": {}, "freq_coef": {}}

    # cells only reached once scattering is on
    street_d = ~building & g.valid("umi", d2d, 3.5)
    out["newly_reached"] = {
        f"{f:g}_{h:g}": int((street_d & ~np.isfinite(med[(f, 0.0)]) & np.isfinite(med[(f, h)])).sum())
        for f in BANDS for h in ROUGH[1:]}
    out["unreached_off"] = {f"{f:g}": int((street_d & ~np.isfinite(med[(f, 0.0)])).sum()) for f in BANDS}

    rows = []
    for f in BANDS:
        pl_umi = g.path_loss("umi", los, d2d, d3d, f, float(tx[2]), H_UT)
        for h in ROUGH:
            pl = -med[(f, h)]
            r = dict(band_ghz=f, roughness_mm=h, S=S[f"{f:g}"][f"{h:g}"])
            for z, m in zones.items():
                ok = m & np.isfinite(pl)
                res = pl[ok] - pl_umi[ok]
                r[f"{z}_cells"] = int(ok.sum())
                r[f"{z}_gain_change_median_db"] = float(np.median(med[(f, h)][ok] - med[(f, 0.0)][ok]))
                r[f"{z}_umi_bias_mean_db"] = float(np.mean(res))
                r[f"{z}_umi_bias_median_db"] = float(np.median(res))
                sig = g.SIGMA_SF[("umi", "los" if z == "los" else "nlos")]
                r[f"{z}_within_1sigma"] = float(np.mean(np.abs(res) <= sig))
                # reliability of the estimator at this configuration
                a, b = runs[(f, h)][0][ok], runs[(f, h)][1][ok]
                dd = np.abs(a - b)
                dd = dd[np.isfinite(dd)]
                r[f"{z}_run_diff_median_db"] = float(np.median(dd))
                r[f"{z}_run_diff_p99_db"] = float(np.percentile(dd, 99))
                dr = drift[(f, h)][ok] - med[(f, h)][ok]
                r[f"{z}_drift_2e7_minus_1e8_median_db"] = float(np.nanmedian(dr))
            rows.append(r)
    with open(args.out.replace(".json", ".csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    out["rows"] = rows

    # per-cell frequency coefficient at each roughness (S differs per band, as the physics says)
    x = 10 * np.log10(np.asarray(BANDS)); xc = x - x.mean()
    for h in ROUGH:
        Y_all = np.stack([-med[(f, h)] for f in BANDS], axis=-1)
        for z in ("los", "nlos", "nlos_near", "nlos_mid", "nlos_far"):
            m = zones[z] & np.all(np.isfinite(Y_all), axis=-1)
            Y = Y_all[m]
            slope = (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)
            out["freq_coef"][f"{h:g}_{z}"] = dict(
                median=float(np.median(slope)), p10=float(np.percentile(slope, 10)),
                p90=float(np.percentile(slope, 90)),
                share_near_3=float(np.mean(np.abs(slope - 3.0) <= 0.1)))
        np.save(os.path.join(os.path.dirname(args.out), f"freq_slope_h{h:g}_nlos.npy"),
                ((Y_all[zones['nlos']] - Y_all[zones['nlos']].mean(1, keepdims=True)) @ xc / (xc @ xc)))

    # ---- known-answer checks -------------------------------------------------
    checks = []
    def chk(name, ok, detail):
        checks.append((name, bool(ok), detail)); print(f"{'pass' if ok else 'FAIL'}  {name}  {detail}")

    readme_bias = {3.5: 6.96, 28.0: 11.08, 39.0: 11.78}
    for f in BANDS:
        rb = next(r for r in rows if r["band_ghz"] == f and r["roughness_mm"] == 0.0)["nlos_umi_bias_mean_db"]
        chk(f"S=0 reproduces README NLOS bias at {f:g} GHz", abs(rb - readme_bias[f]) < 0.5,
            f"{rb:+.2f} vs {readme_bias[f]:+.2f} dB")
    fc0 = out["freq_coef"]["0_nlos"]["median"]
    chk("S=0 reproduces README NLOS frequency coefficient", abs(fc0 - 2.99) < 0.05, f"{fc0:.3f} vs 2.99")
    fcl = out["freq_coef"]["0_los"]["median"]
    chk("LOS frequency coefficient is free space", abs(fcl - 2.0) < 0.06, f"{fcl:.3f}")
    s_rows = [[S[f"{f:g}"][f"{h:g}"] for h in ROUGH] for f in BANDS]
    chk("S rises with roughness and with frequency",
        all(np.all(np.diff(r) >= 0) for r in s_rows) and all(
            S["3.5"][k] <= S["28"][k] <= S["39"][k] for k in S["3.5"]), "monotone")
    for f in BANDS:
        sd = float(np.std(np.stack([runs[(f, 0.0)][i][zones['nlos']] for i in range(3)]), axis=0).mean())
        chk(f"S=0 runs agree at {f:g} GHz (sanity on the baseline)", sd < 0.6, f"mean per-cell sd {sd:.2f} dB")
    los_change = max(abs(r["los_gain_change_median_db"]) for r in rows)
    chk("LOS cells barely move with scattering (direct path dominates)", los_change < 1.5,
        f"largest median change {los_change:.2f} dB")
    out["checks"] = [dict(name=n, passed=p, detail=d) for n, p, d in checks]
    print(f"{sum(p for _, p, _ in checks)}/{len(checks)} checks passed")

    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # ---- print ---------------------------------------------------------------
    print("\ncells:", out["cells"])
    print("\nNLOS far (>80 m): median gain change vs S=0 | mean UMi bias | run diff med/p99 | drift")
    for r in rows:
        print(f"{r['band_ghz']:>5g} GHz  {r['roughness_mm']:>3g} mm  S {r['S']:.3f}  "
              f"far {r['nlos_far_gain_change_median_db']:+6.2f}  bias {r['nlos_far_umi_bias_mean_db']:+6.2f}  "
              f"near {r['nlos_near_gain_change_median_db']:+6.2f}  all-NLOS bias {r['nlos_umi_bias_mean_db']:+6.2f} "
              f"in1sig {100*r['nlos_within_1sigma']:4.1f}%  run {r['nlos_far_run_diff_median_db']:.2f}/"
              f"{r['nlos_far_run_diff_p99_db']:.1f}  drift {r['nlos_far_drift_2e7_minus_1e8_median_db']:+.2f}")
    print("\nNLOS frequency coefficient, median [p10, p90]  (UMi 2.13, single edge 3.0)")
    for h in ROUGH:
        a = out["freq_coef"][f"{h:g}_nlos"]; b = out["freq_coef"][f"{h:g}_nlos_far"]
        print(f"  {h:>3g} mm: all NLOS {a['median']:.2f} [{a['p10']:.2f}, {a['p90']:.2f}]  "
              f"far {b['median']:.2f} [{b['p10']:.2f}, {b['p90']:.2f}]  near-3.0 share {100*a['share_near_3']:.0f}%")
    print("newly reached cells:", out["newly_reached"], "unreached with S=0:", out["unreached_off"])


if __name__ == "__main__":
    main()
