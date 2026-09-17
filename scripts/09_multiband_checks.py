"""Known-answer checks on the multiband convergence result, in the style of tests_pipeline.py.

Every number in the multiband table rests on assumptions that can be checked
rather than assumed.  This is that check.  It retraces a small amount (the
determinism and independence checks need fresh traces at a cheap ray count) and
otherwise works from the cached maps.

    python scripts/09_multiband_checks.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import json

import numpy as np

from radiomap.config import SceneConfig
from radiomap.scene import trace

CACHE = "data/multiband"
PASS, FAIL = [], []


def check(name: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(name)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))


def load(freq: float, n: int, seed: int):
    return np.load(os.path.join(CACHE, f"f{freq:g}_n{n:d}_s{seed}.npz"))


def fit(n, y, lo):
    m = n >= lo
    a, b = np.polyfit(np.log10(n[m]), np.log10(y[m]), 1)
    pred = a * np.log10(n[m]) + b
    ss_res = np.sum((np.log10(y[m]) - pred) ** 2)
    ss_tot = np.sum((np.log10(y[m]) - np.log10(y[m]).mean()) ** 2)
    return float(a), float(1 - ss_res / ss_tot)


def main() -> None:
    with open("results/multiband_summary.json", encoding="utf-8") as fh:
        summary = json.load(fh)
    bands = {float(k): v for k, v in summary["bands"].items()}

    # 1 and 2. What is the seed actually controlling?  If repeating a seed
    #    reproduces the map, "seed-to-seed RMSE" measures what it says.  If it
    #    does not, the quantity is really run-to-run and the label is wrong.
    #    Either way the comparison must be made on cells both runs reached.
    def run(seed: int):
        m = trace(
            SceneConfig(frequency_hz=3.5e9, samples_per_tx=int(5e6), seed=seed),
            verbose=False,
        )
        return m.to_grid(m.gain_db)

    a, a2, c = run(7), run(7), run(8)

    def pair_rmse(x, y):
        both = np.isfinite(x) & np.isfinite(y)
        return float(np.sqrt(np.mean((x[both] - y[both]) ** 2))), float(
            np.mean(np.isfinite(x) != np.isfinite(y))
        )

    same_rmse, same_cov = pair_rmse(a, a2)
    diff_rmse, diff_cov = pair_rmse(a, c)

    # NOTE: this one does not pass, and it is a finding rather than a bug in
    # the study.  Sionna RT 2.0.1 on the LLVM backend is not bit reproducible:
    # two runs at an identical seed differ.  Recorded, reported, and the reason
    # the sweep's column is relabelled run-to-run rather than seed-to-seed.
    check("1. repeating a seed reproduces the map exactly", same_rmse < 1e-9,
          f"same-seed rmse {same_rmse:.3f} dB, coverage differs on "
          f"{100 * same_cov:.2f}% of cells  <-- EXPECTED FAIL, see note")

    # What the measurement actually needs: changing the seed must produce a
    # genuinely different draw, not a near-repeat.  It does, by a wide margin.
    check("2. changing the seed produces a materially different draw",
          diff_rmse > 2.0 * same_rmse,
          f"different-seed {diff_rmse:.3f} dB vs same-seed {same_rmse:.3f} dB "
          f"(ratio {diff_rmse / same_rmse:.2f})")

    # 3. The building mask is geometry and must not move with band or seed.
    masks = [np.load(p)["building"] for p in sorted(glob.glob(os.path.join(CACHE, "*.npz")))]
    check("3. building mask identical across every band, ray count and seed",
          all(np.array_equal(m, masks[0]) for m in masks),
          f"{len(masks)} traces, {int(masks[0].sum())} cells under buildings")

    # 4. Coverage is geometry, so it must not depend on frequency.  Comparing
    #    reached-cell patterns directly is not enough on its own: two runs at
    #    the SAME frequency already disagree about which marginal cells got a
    #    ray, because of the non-determinism check 1 just measured.  The test
    #    is therefore whether cross-band disagreement is any larger than the
    #    same-band disagreement between two seeds.
    for n in (int(2e5), int(5e6), int(2e8)):
        within = float(np.mean(
            np.isfinite(load(3.5, n, 1)["grid"]) != np.isfinite(load(3.5, n, 2)["grid"])
        ))
        across = max(
            float(np.mean(
                np.isfinite(load(3.5, n, 1)["grid"]) != np.isfinite(load(f, n, 1)["grid"])
            ))
            for f in (28.0, 39.0)
        )
        check(f"4. coverage at {n:.0e} rays: cross-band spread is no worse than within-band",
              across <= 1.15 * max(within, 1e-9),
              f"within-band {100 * within:.2f}%, cross-band {100 * across:.2f}%")

    # 4b. And the aggregate that the table actually reports must agree.
    for n in (int(2e5), int(5e6), int(2e8)):
        fr = [
            float(np.mean(~np.isfinite(load(f, n, 1)["grid"])[~load(f, n, 1)["building"]]))
            for f in (3.5, 28.0, 39.0)
        ]
        check(f"4b. empty-cell fraction at {n:.0e} rays agrees across bands",
              (max(fr) - min(fr)) < 0.005,
              " / ".join(f"{100 * v:.2f}%" for v in fr))

    # 5. The Monte Carlo law.  Fitted from 5e6 the exponent comes out near
    #    -0.55; restricting to the fully covered regime should move it toward
    #    the theoretical -0.5, which tells us the excess is residual coverage
    #    error at the low end and not something structural.
    for f, band in bands.items():
        n = np.array([r["rays_per_tx"] for r in band["rows"]], dtype=float)
        y = np.array([r["seed_to_seed_rmse_db"] for r in band["rows"]], dtype=float)
        a5, r5 = fit(n, y, 5e6)
        a20, r20 = fit(n, y, 2e7)
        check(f"5. {f:g} GHz exponent moves toward -1/2 on the fully covered points",
              abs(a20 + 0.5) < abs(a5 + 0.5),
              f"N>=5e6: {a5:.3f} (R2 {r5:.4f})   N>=2e7: {a20:.3f} (R2 {r20:.4f})")

    # 6. Free-space scaling.  Mean gain should drop by 20 log10(f2/f1) between
    #    bands if the solver is doing the obvious thing.  A large departure
    #    would mean the frequency is not reaching the propagation model.
    for f in (28.0, 39.0):
        got = bands[3.5]["field_mean_db"] - bands[f]["field_mean_db"]
        want = 20 * np.log10(f / 3.5)
        check(f"6. {f:g} GHz mean gain follows 20log10(f) to within 6 dB",
              abs(got - want) < 6.0,
              f"measured {got:.1f} dB, free space {want:.1f} dB")

    # 7. The map genuinely gets harder edged at mmWave, which is the thing that
    #    made the band-independent answer worth checking in the first place.
    sds = [bands[f]["field_sd_db"] for f in (3.5, 28.0, 39.0)]
    check("7. field spread increases with frequency", sds[0] < sds[1] < sds[2],
          "sd " + " -> ".join(f"{s:.1f}" for s in sds) + " dB")

    # 8. The required ray counts agree across bands to within 10%, which is the
    #    claim the email will make.
    need = [bands[f]["rays_for_target_db"]["0.25"] for f in (3.5, 28.0, 39.0)]
    spread = (max(need) - min(need)) / min(need)
    check("8. rays needed for 0.25 dB agree across bands to within 10%", spread < 0.10,
          " / ".join(f"{v:.2e}" for v in need) + f"  spread {100 * spread:.1f}%")

    # 9. The noise estimate is stable: the spread across seed pairs is small
    #    next to the estimate itself, so a two-seed number was not a fluke.
    worst_rel = 0.0
    for band in bands.values():
        for r in band["rows"]:
            sd, m = r["seed_to_seed_rmse_sd_db"], r["seed_to_seed_rmse_db"]
            if sd == sd:  # not nan
                worst_rel = max(worst_rel, sd / m)
    check("9. pairwise noise estimates agree to better than 5%", worst_rel < 0.05,
          f"worst relative spread {100 * worst_rel:.1f}%")

    print(f"\n{len(PASS)}/{len(PASS) + len(FAIL)} checks passed")
    if FAIL:
        print("FAILED: " + ", ".join(FAIL))
        sys.exit(1)


if __name__ == "__main__":
    main()
