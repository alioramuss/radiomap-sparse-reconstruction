"""Empirical variogram and a spherical model fit.

The spherical model is used rather than the exponential one because its range
parameter is the distance at which correlation actually reaches zero, so the
number can be quoted directly ("correlation range 27.8 m") without the factor
of three that the exponential model's practical range carries.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SphericalVariogram:
    """gamma(h) = nugget + sill * (1.5 h/a - 0.5 (h/a)^3) for h < a, else nugget + sill."""

    nugget: float
    sill: float
    rng: float  # the 'a' above, in metres

    def __call__(self, h: np.ndarray) -> np.ndarray:
        h = np.asarray(h, dtype=float)
        a = max(self.rng, 1e-9)
        ratio = np.clip(h / a, 0.0, 1.0)
        shape = 1.5 * ratio - 0.5 * ratio**3
        g = self.nugget + self.sill * shape
        return np.where(h <= 0, 0.0, g)

    def covariance(self, h: np.ndarray) -> np.ndarray:
        """Covariance implied by the model, C(h) = (nugget + sill) - gamma(h)."""
        total = self.nugget + self.sill
        return total - self(h)

    @property
    def total_sill(self) -> float:
        return self.nugget + self.sill


def empirical_variogram(
    xy: np.ndarray,
    values: np.ndarray,
    *,
    n_lags: int = 30,
    max_dist: float | None = None,
    max_pairs: int = 4_000_000,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Isotropic empirical semivariogram.

    Returns ``(lag_centres, gamma, counts)``.  Pairs are subsampled when the
    full pair set would be larger than ``max_pairs``, which keeps this usable
    on the 10% draw (about 1750 points, so 1.5M pairs) without special cases.
    """
    xy = np.asarray(xy, dtype=float)
    v = np.asarray(values, dtype=float)
    n = len(xy)
    if n < 10:
        raise ValueError("need at least 10 points to fit a variogram")
    if rng is None:
        rng = np.random.default_rng(0)

    n_pairs_full = n * (n - 1) // 2
    if n_pairs_full <= max_pairs:
        i, j = np.triu_indices(n, k=1)
    else:
        # Sample pairs uniformly with replacement; at these counts the
        # duplicate rate is negligible and the estimator is unbiased.
        i = rng.integers(0, n, size=max_pairs)
        j = rng.integers(0, n, size=max_pairs)
        keep = i != j
        i, j = i[keep], j[keep]

    d = np.linalg.norm(xy[i] - xy[j], axis=1)
    sq = 0.5 * (v[i] - v[j]) ** 2

    if max_dist is None:
        # Only the short lags carry information about the range.  Fitting out
        # into the tail biases the range upward, badly: on the synthetic check
        # in tests_pipeline.py, taking the 40th percentile instead of the 25th
        # turns a planted 15 m range into 22 m.
        max_dist = float(np.percentile(d, 25.0))

    keep = (d > 0) & (d <= max_dist)
    d, sq = d[keep], sq[keep]

    edges = np.linspace(0.0, max_dist, n_lags + 1)
    idx = np.digitize(d, edges) - 1
    idx = np.clip(idx, 0, n_lags - 1)

    counts = np.bincount(idx, minlength=n_lags)
    sums = np.bincount(idx, weights=sq, minlength=n_lags)
    dsums = np.bincount(idx, weights=d, minlength=n_lags)

    ok = counts > 0
    gamma = np.full(n_lags, np.nan)
    centres = np.full(n_lags, np.nan)
    gamma[ok] = sums[ok] / counts[ok]
    centres[ok] = dsums[ok] / counts[ok]

    return centres[ok], gamma[ok], counts[ok]


def fit_spherical(
    lags: np.ndarray,
    gamma: np.ndarray,
    counts: np.ndarray | None = None,
    *,
    range_bounds: tuple[float, float] = (1.0, 500.0),
) -> SphericalVariogram:
    """Weighted least-squares fit of the spherical model.

    Weights follow Cressie: proportional to the pair count and inversely to the
    squared model value, so the short lags -- which are where the information
    about the range actually is -- dominate the fit.  Plain pair-count weighting
    lets the tail pull the range upward by 40% or more on a field with a known
    answer; check 08 in ``tests_pipeline.py`` pins this down.

    Across ten independent realisations of a field with a planted 15 m range,
    this fitter recovers 15.6 +/- 1.0 m; check 08 runs exactly that.
    """
    lags = np.asarray(lags, dtype=float)
    gamma = np.asarray(gamma, dtype=float)
    n = np.ones_like(gamma) if counts is None else np.asarray(counts, dtype=float)

    sill0 = float(np.nanmax(gamma))
    rng0 = float(np.clip(np.nanmedian(lags), *range_bounds))

    def residual(p):
        nugget, sill, a = p
        model = SphericalVariogram(nugget, sill, a)
        m = model(lags)
        w = np.sqrt(n) / np.maximum(m, 1e-12)
        w = w / w.max()
        return w * (m - gamma)

    sol = least_squares(
        residual,
        x0=[0.0, max(sill0, 1e-9), rng0],
        bounds=(
            [0.0, 1e-12, range_bounds[0]],
            [max(sill0, 1e-9), 10.0 * max(sill0, 1e-9), range_bounds[1]],
        ),
        method="trf",
    )
    nugget, sill, a = sol.x
    return SphericalVariogram(float(nugget), float(sill), float(a))
