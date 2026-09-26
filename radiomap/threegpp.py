"""3GPP TR 38.901 (v17) Table 7.4.1-1 path loss, UMi-Street Canyon and UMa.

Only what the comparison needs: the mean path loss in dB for LOS and NLOS and
the shadow fading standard deviation that goes with each.  fc is in GHz,
distances and heights in metres, exactly as the table writes them.

Two things the table says that are easy to lose:

* the NLOS formula is max(PL_LOS, PL'_NLOS), so NLOS can never come out below
  LOS at the same distance;
* the breakpoint uses EFFECTIVE heights h' = h - h_E, with h_E = 1 m for UMi
  (and taken as 1 m for UMa here, which is its value with the highest
  probability at short range).

Validity ranges from the table are returned by ``valid`` so the caller can
state which cells sit outside them rather than silently extrapolating.
"""

from __future__ import annotations

import numpy as np

C = 299_792_458.0

SIGMA_SF = {
    ("umi", "los"): 4.0,
    ("umi", "nlos"): 7.82,
    ("uma", "los"): 4.0,
    ("uma", "nlos"): 6.0,
}

# Nominal base station height the model was fitted for.
NOMINAL_HBS = {"umi": 10.0, "uma": 25.0}


def breakpoint_m(fc_ghz, h_bs, h_ut, h_e: float = 1.0):
    return 4.0 * (h_bs - h_e) * (h_ut - h_e) * (np.asarray(fc_ghz) * 1e9) / C


def umi_los(d2d, d3d, fc_ghz, h_bs, h_ut):
    d_bp = breakpoint_m(fc_ghz, h_bs, h_ut)
    pl1 = 32.4 + 21.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (32.4 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.5 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    return np.where(np.asarray(d2d) <= d_bp, pl1, pl2)


def umi_nlos(d2d, d3d, fc_ghz, h_bs, h_ut):
    pl_prime = 35.3 * np.log10(d3d) + 22.4 + 21.3 * np.log10(fc_ghz) - 0.3 * (h_ut - 1.5)
    return np.maximum(umi_los(d2d, d3d, fc_ghz, h_bs, h_ut), pl_prime)


def uma_los(d2d, d3d, fc_ghz, h_bs, h_ut):
    d_bp = breakpoint_m(fc_ghz, h_bs, h_ut)
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc_ghz)
           - 9.0 * np.log10(d_bp ** 2 + (h_bs - h_ut) ** 2))
    return np.where(np.asarray(d2d) <= d_bp, pl1, pl2)


def uma_nlos(d2d, d3d, fc_ghz, h_bs, h_ut):
    pl_prime = 13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc_ghz) - 0.6 * (h_ut - 1.5)
    return np.maximum(uma_los(d2d, d3d, fc_ghz, h_bs, h_ut), pl_prime)


MODELS = {
    ("umi", "los"): umi_los,
    ("umi", "nlos"): umi_nlos,
    ("uma", "los"): uma_los,
    ("uma", "nlos"): uma_nlos,
}


def path_loss(model: str, los: np.ndarray, d2d, d3d, fc_ghz, h_bs, h_ut):
    """Per-cell mean path loss, picking the LOS or NLOS formula by ``los``."""
    pl_los = MODELS[(model, "los")](d2d, d3d, fc_ghz, h_bs, h_ut)
    pl_nlos = MODELS[(model, "nlos")](d2d, d3d, fc_ghz, h_bs, h_ut)
    return np.where(los, pl_los, pl_nlos)


def valid(model: str, d2d, fc_ghz) -> np.ndarray:
    """Table 7.4.1-1 ranges: 10 m <= d2D <= 5 km, 0.5 <= fc <= 100 GHz."""
    d2d = np.asarray(d2d)
    ok_f = 0.5 <= fc_ghz <= 100.0
    return (d2d >= 10.0) & (d2d <= 5000.0) & ok_f
