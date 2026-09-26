"""Known-answer checks for the 38.901 comparison (scripts 13 to 15).

The formula checks run without data.  The data checks need
data/bands_los.npz from scripts/13_trace_bands_los.py and are skipped if it
is missing.

    python scripts/16_umi_checks.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from radiomap import threegpp as g

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def formula_checks():
    # Hand-computed from Table 7.4.1-1.
    v = float(g.umi_los(50.0, 100.0, 28.0, 10.0, 1.5))
    check("UMi LOS, d3D 100 m, 28 GHz = 32.4 + 42 + 20log10(28)", abs(v - 103.343) < 1e-3, f"{v:.3f}")

    v = float(g.umi_nlos(50.0, 100.0, 28.0, 10.0, 1.5))
    check("UMi NLOS, d3D 100 m, 28 GHz = 22.4 + 70.6 + 21.3log10(28)", abs(v - 123.824) < 1e-3, f"{v:.3f}")

    v = float(g.breakpoint_m(3.5, 10.0, 1.5))
    check("UMi breakpoint, 10 m / 1.5 m / 3.5 GHz = 210.1 m", abs(v - 210.14) < 0.05, f"{v:.2f} m")

    for name, fn, hbs in (("UMi", g.umi_los, 10.0), ("UMa", g.uma_los, 25.0)):
        dbp = float(g.breakpoint_m(3.5, hbs, 1.5))
        d3 = np.hypot(dbp, hbs - 1.5)
        lo = float(fn(dbp * (1 - 1e-9), d3, 3.5, hbs, 1.5))
        hi = float(fn(dbp * (1 + 1e-9), d3, 3.5, hbs, 1.5))
        check(f"{name} LOS is continuous at the breakpoint", abs(lo - hi) < 1e-3, f"{lo:.4f} vs {hi:.4f}")

    d2 = np.linspace(10, 5000, 400)
    ok = True
    for model, hbs in (("umi", 10.0), ("uma", 25.0), ("umi", 32.0)):
        for f in (0.5, 3.5, 28.0, 39.0, 100.0):
            d3 = np.hypot(d2, hbs - 1.5)
            ok &= bool(np.all(g.MODELS[(model, "nlos")](d2, d3, f, hbs, 1.5)
                              >= g.MODELS[(model, "los")](d2, d3, f, hbs, 1.5) - 1e-9))
    check("NLOS is never below LOS, both models, 0.5 to 100 GHz", ok)

    # Frequency coefficient of the formulas themselves, by finite difference.
    f1, f2 = 3.5, 39.0
    x = 10 * np.log10(f2 / f1)
    s_los = float(g.umi_los(50, 60, f2, 32, 1.5) - g.umi_los(50, 60, f1, 32, 1.5)) / x
    check("UMi LOS scales as 2.0 dB per dB of fc", abs(s_los - 2.0) < 1e-9, f"{s_los:.4f}")
    s_nlos = float(g.umi_nlos(120, 125, f2, 32, 1.5) - g.umi_nlos(120, 125, f1, 32, 1.5)) / x
    check("UMi NLOS (where PL' binds) scales as 2.13", abs(s_nlos - 2.13) < 1e-9, f"{s_nlos:.4f}")

    # The regression used for the per-cell coefficient recovers a planted slope.
    bands = np.array([3.5, 28.0, 39.0])
    xx = 10 * np.log10(bands)
    xc = xx - xx.mean()
    rng = np.random.default_rng(0)
    Y = 50 + rng.normal(0, 5, (100, 1)) + np.outer(np.full(100, 2.7), xx)
    sl = (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)
    check("per-cell regression recovers a planted 2.7", np.allclose(sl, 2.7), f"{sl.mean():.6f}")


def data_checks(path="data/bands_los.npz"):
    if not os.path.exists(path):
        print(f"[SKIP] data checks, {path} not found")
        return
    d = np.load(path)
    bands = [float(b) for b in d["bands_ghz"]]
    c, tx, los, bld = d["centres"], d["tx"], d["los"], d["building"]
    d3 = np.linalg.norm(c - tx, axis=-1)

    check("no LOS cell sits under a building", not np.any(los & bld), f"{int((los & bld).sum())}")

    for f in bands:
        gdb = d[f"gain_db_{f:g}"]
        fs = -(20 * np.log10(d3) + 20 * np.log10(f * 1e9) - 147.55)
        m = los & ~bld & np.isfinite(gdb)
        r = gdb[m] - fs[m]
        med = float(np.median(r))
        within = float(np.mean(np.abs(r) <= 6.0))
        check(f"{f:g} GHz: LOS cells sit on free space (median within 2 dB, 98% within 6 dB)",
              abs(med) < 2.0 and within > 0.98, f"median {med:+.2f} dB, {100*within:.1f}% within 6 dB")

    m = ~bld & los
    for f in bands:
        m &= np.isfinite(d[f"gain_db_{f:g}"])
    xx = 10 * np.log10(np.asarray(bands))
    xc = xx - xx.mean()
    Y = np.stack([-d[f"gain_db_{f:g}"][m] for f in bands], axis=1)
    sl = (Y - Y.mean(1, keepdims=True)) @ xc / (xc @ xc)
    check("LOS cells follow 20log10(fc) (median coefficient within 0.05 of 2.0)",
          abs(np.median(sl) - 2.0) < 0.05, f"{np.median(sl):.3f}")

    ns = ~bld & ~los
    for f in bands:
        ns &= np.isfinite(d[f"gain_db_{f:g}"])
    frac = float(np.mean(ns[~bld]))
    check("NLOS street cells reached in every band are the large majority", frac > 0.75, f"{100*frac:.1f}%")


if __name__ == "__main__":
    formula_checks()
    data_checks()
    n = len(RESULTS)
    print(f"\n{sum(RESULTS)}/{n} checks passed")
    sys.exit(0 if all(RESULTS) else 1)
