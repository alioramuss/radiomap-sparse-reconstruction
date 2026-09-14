"""Reconstruction baselines.

All five share one interface::

    model = Method(**hyperparameters)
    model.fit(xy_observed, values_observed)
    prediction = model.predict(xy_target)

``values`` are in dB throughout.  Interpolating in dB is the field convention
and the domain where the residuals are approximately Gaussian; part of the
literature scores in linear power instead, which weights cells near the
transmitter far more heavily.  ``scripts/04_db_vs_linear.py`` runs that
ablation rather than assuming the two agree.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from .trend import LogDistanceTrend
from .variogram import SphericalVariogram, empirical_variogram, fit_spherical


class _Base:
    """Shared plumbing: store the training set, build a KD-tree."""

    def __init__(self) -> None:
        self.xy_: np.ndarray | None = None
        self.z_: np.ndarray | None = None
        self.tree_: cKDTree | None = None

    def fit(self, xy: np.ndarray, z: np.ndarray):
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z, dtype=float)
        if xy.ndim != 2 or xy.shape[1] != 2:
            raise ValueError("xy must have shape (n, 2)")
        if xy.shape[0] != z.shape[0]:
            raise ValueError("xy and z must have the same length")
        self.xy_, self.z_ = xy, z
        self.tree_ = cKDTree(xy)
        return self

    def _check_fitted(self) -> None:
        if self.xy_ is None:
            raise RuntimeError("call fit() before predict()")

    @property
    def name(self) -> str:
        return type(self).__name__


class NearestNeighbour(_Base):
    """Piecewise-constant baseline. Included as the floor, not as a contender."""

    def predict(self, xy: np.ndarray) -> np.ndarray:
        self._check_fitted()
        _, idx = self.tree_.query(np.asarray(xy, dtype=float), k=1)
        return self.z_[idx]


class InverseDistanceWeighting(_Base):
    """Shepard's method on the ``k`` nearest observations."""

    def __init__(self, power: float = 2.0, k: int = 16) -> None:
        super().__init__()
        self.power = float(power)
        self.k = int(k)

    def predict(self, xy: np.ndarray) -> np.ndarray:
        self._check_fitted()
        xy = np.asarray(xy, dtype=float)
        k = min(self.k, len(self.xy_))
        d, idx = self.tree_.query(xy, k=k)
        if k == 1:
            d, idx = d[:, None], idx[:, None]

        out = np.empty(len(xy), dtype=float)
        exact = d[:, 0] <= 0.0
        out[exact] = self.z_[idx[exact, 0]]

        far = ~exact
        if far.any():
            w = 1.0 / np.power(d[far], self.power)
            out[far] = np.sum(w * self.z_[idx[far]], axis=1) / np.sum(w, axis=1)
        return out


class ThinPlateRBF(_Base):
    """Thin-plate spline, local by default.

    ``neighbors`` keeps the solve local; a global thin-plate spline on 3500
    observations is a dense 3500x3500 solve for every sampling fraction and
    seed, and it extrapolates no better.
    """

    def __init__(self, smoothing: float = 0.0, neighbors: int = 64) -> None:
        super().__init__()
        self.smoothing = float(smoothing)
        self.neighbors = int(neighbors)
        self._rbf: RBFInterpolator | None = None

    def fit(self, xy: np.ndarray, z: np.ndarray):
        super().fit(xy, z)
        n = len(self.xy_)
        neighbors = None if self.neighbors >= n else self.neighbors
        self._rbf = RBFInterpolator(
            self.xy_,
            self.z_,
            kernel="thin_plate_spline",
            smoothing=self.smoothing,
            neighbors=neighbors,
        )
        return self

    def predict(self, xy: np.ndarray) -> np.ndarray:
        self._check_fitted()
        return np.asarray(self._rbf(np.asarray(xy, dtype=float)), dtype=float)


@dataclass
class _KrigingHyper:
    k: int = 32
    variogram: SphericalVariogram | None = None


class OrdinaryKriging(_Base):
    """Ordinary kriging with a moving neighbourhood.

    Global kriging would need an n^3 solve at these sample counts (up to ~3500
    observations at 20%), repeated for every fraction, seed and block position.
    The moving neighbourhood of the ``k`` closest observations is the standard
    remedy and, on a field with a 28 m correlation range and 1 m cells, it is
    numerically indistinguishable from the global solve.
    """

    def __init__(self, k: int = 32, variogram: SphericalVariogram | None = None) -> None:
        super().__init__()
        self.k = int(k)
        self.variogram = variogram

    def fit(self, xy: np.ndarray, z: np.ndarray):
        super().fit(xy, z)
        if self.variogram is None:
            lags, gamma, counts = empirical_variogram(self.xy_, self.z_)
            self.variogram = fit_spherical(lags, gamma, counts)
        return self

    def predict(self, xy: np.ndarray, chunk: int = 2048) -> np.ndarray:
        self._check_fitted()
        xy = np.asarray(xy, dtype=float)
        k = min(self.k, len(self.xy_))
        out = np.empty(len(xy), dtype=float)

        for lo in range(0, len(xy), chunk):
            hi = min(lo + chunk, len(xy))
            block = xy[lo:hi]
            d0, idx = self.tree_.query(block, k=k)
            if k == 1:
                d0, idx = d0[:, None], idx[:, None]

            nb = self.xy_[idx]                     # (m, k, 2)
            zb = self.z_[idx]                      # (m, k)
            m = len(block)

            # Pairwise distances between the neighbours of each target.
            dd = np.linalg.norm(nb[:, :, None, :] - nb[:, None, :, :], axis=-1)
            gamma_mat = self.variogram(dd)         # (m, k, k)
            np.einsum("ijj->ij", gamma_mat)[...] = 0.0

            gamma_0 = self.variogram(d0)           # (m, k)

            # Lagrange-augmented ordinary kriging system.
            A = np.zeros((m, k + 1, k + 1), dtype=float)
            A[:, :k, :k] = gamma_mat
            A[:, :k, k] = 1.0
            A[:, k, :k] = 1.0
            # Tiny ridge on the covariance block keeps a duplicated observation
            # from making the system singular.
            A[:, :k, :k] += 1e-10 * np.eye(k)[None, :, :]

            b = np.ones((m, k + 1), dtype=float)
            b[:, :k] = gamma_0

            try:
                sol = np.linalg.solve(A, b[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:  # pragma: no cover - defensive
                sol = np.linalg.lstsq(A.reshape(-1, k + 1), b.ravel(), rcond=None)[0]
                sol = sol.reshape(m, k + 1)

            w = sol[:, :k]
            pred = np.sum(w * zb, axis=1)

            # An exact hit on an observation must return that observation.
            exact = d0[:, 0] <= 0.0
            pred[exact] = zb[exact, 0]
            out[lo:hi] = pred

        return out


class DetrendedKriging(OrdinaryKriging):
    """Fit a log-distance path-loss trend, krige the residual, add it back."""

    def __init__(
        self,
        k: int = 32,
        variogram: SphericalVariogram | None = None,
        tx_xy: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        super().__init__(k=k, variogram=variogram)
        self.tx_xy = tx_xy
        self.trend_: LogDistanceTrend | None = None

    def fit(self, xy: np.ndarray, z: np.ndarray):
        xy = np.asarray(xy, dtype=float)
        z = np.asarray(z, dtype=float)
        self.trend_ = LogDistanceTrend(tx_xy=self.tx_xy).fit(xy, z)
        residual = self.trend_.residual(xy, z)
        # The variogram of the residual is not the variogram of the field, so a
        # variogram handed in for the raw field must not be reused here.
        self.variogram = None
        super().fit(xy, residual)
        return self

    def predict(self, xy: np.ndarray, chunk: int = 2048) -> np.ndarray:
        if self.trend_ is None:
            raise RuntimeError("call fit() before predict()")
        xy = np.asarray(xy, dtype=float)
        return super().predict(xy, chunk=chunk) + self.trend_.predict(xy)


#: Registry used by the sweep scripts and the cross-validation helper.
METHODS: dict[str, type] = {
    "nearest": NearestNeighbour,
    "idw": InverseDistanceWeighting,
    "rbf": ThinPlateRBF,
    "ok": OrdinaryKriging,
    "dk": DetrendedKriging,
}

#: Display names, in the order the report tabulates them.
METHOD_LABELS: dict[str, str] = {
    "nearest": "Nearest neighbour",
    "idw": "IDW",
    "rbf": "RBF",
    "ok": "Ordinary kriging",
    "dk": "Detrended kriging",
}
