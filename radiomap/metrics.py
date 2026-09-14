"""Error metrics and the dB / linear conversion.

Sionna returns ``path_gain`` as a *linear* power ratio, not dB.  The values on
this scene are of order 1e-9.  The documentation does not say so, and it is an
easy thing to lose an afternoon to, so the conversion lives in one place.
"""

from __future__ import annotations

import numpy as np

#: Anything at or below this linear gain is treated as "no ray reached here".
GAIN_FLOOR = 1e-30


def to_db(linear_gain: np.ndarray, floor: float = GAIN_FLOOR) -> np.ndarray:
    """Linear power ratio -> dB.

    Cells with zero (or negative, which should not happen) gain are returned as
    ``nan`` rather than ``-inf``, so they propagate visibly instead of poisoning
    an average silently.
    """
    g = np.asarray(linear_gain, dtype=float)
    out = np.full(g.shape, np.nan, dtype=float)
    ok = g > floor
    out[ok] = 10.0 * np.log10(g[ok])
    return out


def to_linear(gain_db: np.ndarray) -> np.ndarray:
    """dB -> linear power ratio."""
    return np.power(10.0, np.asarray(gain_db, dtype=float) / 10.0)


def rmse(truth: np.ndarray, pred: np.ndarray) -> float:
    """Root mean squared error over finite pairs."""
    t = np.asarray(truth, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    ok = np.isfinite(t) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.sqrt(np.mean((t[ok] - p[ok]) ** 2)))


def mae(truth: np.ndarray, pred: np.ndarray) -> float:
    """Mean absolute error over finite pairs."""
    t = np.asarray(truth, dtype=float).ravel()
    p = np.asarray(pred, dtype=float).ravel()
    if t.shape != p.shape:
        raise ValueError(f"shape mismatch: {t.shape} vs {p.shape}")
    ok = np.isfinite(t) & np.isfinite(p)
    if not ok.any():
        return float("nan")
    return float(np.mean(np.abs(t[ok] - p[ok])))
