"""Turn the multiband convergence traces into the answer: what does each band need?

Reads the cached traces written by ``scripts/06_multiband_convergence.py`` and
reports, per band:

* the seed-to-seed noise against ray count, with the single-trace empty-cell
  fraction (seed-count independent, so it can be compared with the two-seed
  table in the interim report);
* a fit of the Monte Carlo law, noise proportional to N^alpha, over the
  converged regime.  alpha should be -1/2 for an unbiased estimator; a band
  where it is not has something other than sampling noise in it;
* the ray count each band needs to put a single map a given distance from the
  converged mean, read off the fit.

    python scripts/07_multiband_analysis.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import glob
import json
import re

import numpy as np

# Fit the power law only where the estimator is actually in its asymptotic
# regime.  Below this the map is still full of unreached cells and the error is
# dominated by coverage, not by variance.
FIT_FROM_RAYS = 5e6

# Targets for "one map vs the converged mean", in dB.
TARGETS_DB = [1.0, 0.5, 0.25, 0.1]


def single_trace_empty_pct(cache_dir: str, freq_ghz: float, n_rays: int) -> float:
    """Mean over seeds of the fraction of street cells that trace never reached.

    The sweep's own column requires a cell to be reached in *every* seed, which
    makes it depend on how many seeds were run.  This one does not, so it is
    the figure to compare against the interim report.
    """
    pattern = os.path.join(cache_dir, f"f{freq_ghz:g}_n{n_rays:d}_s*.npz")
    fracs = []
    for path in sorted(glob.glob(pattern)):
        d = np.load(path)
        street = ~d["building"]
        reached = street & np.isfinite(d["grid"])
        fracs.append(1.0 - reached.sum() / max(street.sum(), 1))
    return 100.0 * float(np.mean(fracs)) if fracs else float("nan")


def fit_power_law(n: np.ndarray, rmse: np.ndarray) -> tuple[float, float, float]:
    """Least squares on log10(rmse) = log10(c) + alpha * log10(n).

    Returns (alpha, c, r_squared).
    """
    x, y = np.log10(n), np.log10(rmse)
    alpha, intercept = np.polyfit(x, y, 1)
    pred = alpha * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(alpha), float(10.0**intercept), r2


def rays_for_target(alpha: float, c: float, target_db: float) -> float:
    """Invert the fit: rays needed for one map to sit target_db from the mean.

    The fit is on the seed-to-seed RMSE; a single map sits that over sqrt(2)
    from the converged mean, so the seed-to-seed target is target * sqrt(2).
    """
    return float(10.0 ** ((np.log10(target_db * np.sqrt(2.0)) - np.log10(c)) / alpha))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", default="results/convergence_multiband.csv")
    p.add_argument("--cache-dir", default="data/multiband")
    p.add_argument("--out-json", default="results/multiband_summary.json")
    p.add_argument("--out-md", default="results/multiband_tables.md")
    args = p.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as fh:
        rows = [
            {k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()}
            for r in csv.DictReader(fh)
        ]

    bands = sorted({r["frequency_ghz"] for r in rows})
    summary: dict = {"bands": {}, "fit_from_rays": FIT_FROM_RAYS}
    md: list[str] = []

    for freq in bands:
        br = sorted((r for r in rows if r["frequency_ghz"] == freq), key=lambda r: r["rays_per_tx"])
        n = np.array([r["rays_per_tx"] for r in br], dtype=float)
        rmse = np.array([r["seed_to_seed_rmse_db"] for r in br], dtype=float)

        for r in br:
            r["single_trace_empty_pct"] = single_trace_empty_pct(
                args.cache_dir, freq, int(r["rays_per_tx"])
            )

        mask = n >= FIT_FROM_RAYS
        alpha, c, r2 = fit_power_law(n[mask], rmse[mask])
        needed = {f"{t:g}": rays_for_target(alpha, c, t) for t in TARGETS_DB}

        summary["bands"][f"{freq:g}"] = {
            "alpha": alpha,
            "c": c,
            "r_squared": r2,
            "n_fit_points": int(mask.sum()),
            "rays_for_target_db": needed,
            "field_sd_db": br[-1]["gain_sd_db"],
            "field_mean_db": br[-1]["gain_mean_db"],
            "rows": br,
        }

        md.append(f"### {freq:g} GHz\n")
        md.append(
            "| Rays per tx | Street cells with no ray | Seed-to-seed RMSE | One map vs. mean | Solve time |"
        )
        md.append("|---|---|---|---|---|")
        for r in br:
            sd = r["seed_to_seed_rmse_sd_db"]
            sd_txt = f" ±{sd:.2f}" if np.isfinite(sd) else ""
            md.append(
                f"| {r['rays_per_tx']:.0e} | {r['single_trace_empty_pct']:.2f}% | "
                f"{r['seed_to_seed_rmse_db']:.2f}{sd_txt} dB | "
                f"{r['one_map_vs_mean_db']:.2f} dB | {r['solve_seconds']:.0f} s |"
            )
        md.append("")
        md.append(
            f"Fit over N >= {FIT_FROM_RAYS:.0e}: RMSE proportional to N^{alpha:.3f} "
            f"(R^2 = {r2:.4f}, {int(mask.sum())} points). "
            f"Field sd {br[-1]['gain_sd_db']:.1f} dB.\n"
        )

    md.append("### Rays needed, read off each band's fit\n")
    md.append("| Target, one map vs. mean | " + " | ".join(f"{b:g} GHz" for b in bands) + " |")
    md.append("|---|" + "---|" * len(bands))
    for t in TARGETS_DB:
        cells = []
        for b in bands:
            v = summary["bands"][f"{b:g}"]["rays_for_target_db"][f"{t:g}"]
            cells.append(f"{v:.2e}")
        md.append(f"| {t:g} dB | " + " | ".join(cells) + " |")
    md.append("")

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.out_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    print("\n".join(md))
    print(f"wrote {args.out_json} and {args.out_md}")


if __name__ == "__main__":
    main()
