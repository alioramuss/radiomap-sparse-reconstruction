"""Is the NLOS gap to 38.901 a max-depth truncation artefact?

Everything so far traces at max depth 3.  If the far NLOS cells are dark only
because paths needing a fourth or fifth interaction were cut off, raising the
depth should close a good part of the gap.  This compares depth 3 against
depth 5 at 3.5 and 28 GHz, same rays, same seed, same cells.

    python scripts/13_trace_bands_los.py --out data/bands_los_depth5.npz --max-depth 5 --bands 3.5,28
    python scripts/17_depth_check.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

import numpy as np

from radiomap import threegpp as g

H_UT = 1.5


def main(d3_path="data/bands_los.npz", d5_path="data/bands_los_depth5.npz", out="results/depth_check.json"):
    a, b = np.load(d3_path), np.load(d5_path)
    assert np.array_equal(a["los"], b["los"]) and np.array_equal(a["building"], b["building"])
    c, tx, los, bld = a["centres"], a["tx"], a["los"], a["building"]
    d2d = np.hypot(c[..., 0] - tx[0], c[..., 1] - tx[1])
    d3d = np.sqrt(d2d ** 2 + (tx[2] - H_UT) ** 2)
    res = {}
    for f in [float(x) for x in b["bands_ghz"]]:
        g3, g5 = a[f"gain_db_{f:g}"], b[f"gain_db_{f:g}"]
        m = ~bld & (d2d >= 10) & np.isfinite(g3) & np.isfinite(g5)
        mu = g.path_loss("umi", los, d2d, d3d, f, float(tx[2]), H_UT)
        row = {}
        for cls, cm in (("los", m & los), ("nlos", m & ~los)):
            r3 = -g3[cm] - mu[cm]
            r5 = -g5[cm] - mu[cm]
            gain_up = g5[cm] - g3[cm]          # dB of extra power from depth 4-5 paths
            far = cm & (d2d >= 80)
            row[cls] = dict(
                bias_depth3_db=float(r3.mean()), bias_depth5_db=float(r5.mean()),
                rmse_depth3_db=float(np.sqrt((r3 ** 2).mean())), rmse_depth5_db=float(np.sqrt((r5 ** 2).mean())),
                median_power_added_db=float(np.median(gain_up)),
                p90_power_added_db=float(np.percentile(gain_up, 90)),
                far_nlos_bias_depth3_db=float((-g3[far] - mu[far]).mean()) if cls == "nlos" else None,
                far_nlos_bias_depth5_db=float((-g5[far] - mu[far]).mean()) if cls == "nlos" else None,
            )
        newly = int((~bld & (d2d >= 10) & ~np.isfinite(g3) & np.isfinite(g5)).sum())
        row["cells_reached_only_at_depth5"] = newly
        res[f"{f:g}"] = row
        n = row["nlos"]
        print(f"{f:>5g} GHz NLOS  bias {n['bias_depth3_db']:+.2f} -> {n['bias_depth5_db']:+.2f} dB,  "
              f"RMSE {n['rmse_depth3_db']:.2f} -> {n['rmse_depth5_db']:.2f},  "
              f"power added median {n['median_power_added_db']:.2f} dB (p90 {n['p90_power_added_db']:.2f}),  "
              f"far (>=80 m) bias {n['far_nlos_bias_depth3_db']:+.2f} -> {n['far_nlos_bias_depth5_db']:+.2f},  "
              f"newly reached {newly}")
        l = row["los"]
        print(f"{'':>5}     LOS   bias {l['bias_depth3_db']:+.2f} -> {l['bias_depth5_db']:+.2f} dB")
    with open(out, "w") as fh:
        json.dump(res, fh, indent=2)


if __name__ == "__main__":
    main()
