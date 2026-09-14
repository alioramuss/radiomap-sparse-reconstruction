"""Log-distance path-loss trend, used by the detrended kriging baseline.

Fitting ``gain_dB = a + b log10(d)`` and kriging the residual is the smallest
possible way of giving a spatial interpolator something other than smoothness
to work with.  On this scene it does not improve the mean error much, but it
sharply reduces the variance across hold-out block positions -- which is a
different and arguably more useful kind of win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LogDistanceTrend:
    """gain_dB ~= intercept + slope * log10(distance_to_tx_in_m)."""

    intercept: float = 0.0
    slope: float = 0.0
    tx_xy: tuple[float, float] = (0.0, 0.0)
    min_distance_m: float = 1.0

    def distances(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        d = np.linalg.norm(xy - np.asarray(self.tx_xy, dtype=float), axis=1)
        return np.maximum(d, self.min_distance_m)

    def fit(self, xy: np.ndarray, gain_db: np.ndarray) -> "LogDistanceTrend":
        d = self.distances(xy)
        x = np.log10(d)
        y = np.asarray(gain_db, dtype=float)
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < 2:
            raise ValueError("need at least two finite points to fit a trend")
        slope, intercept = np.polyfit(x[ok], y[ok], 1)
        self.slope = float(slope)
        self.intercept = float(intercept)
        return self

    def predict(self, xy: np.ndarray) -> np.ndarray:
        return self.intercept + self.slope * np.log10(self.distances(xy))

    def residual(self, xy: np.ndarray, gain_db: np.ndarray) -> np.ndarray:
        return np.asarray(gain_db, dtype=float) - self.predict(xy)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"LogDistanceTrend(gain_dB = {self.intercept:.1f} "
            f"{self.slope:+.1f} log10(d))"
        )
