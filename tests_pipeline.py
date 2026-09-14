"""Known-answer checks for the reconstruction pipeline.

Nineteen checks, each of which fails loudly if the thing it is checking breaks.
They are deliberately not a coverage exercise: every one of them encodes a fact
that has to be true for a number in the report to mean what it says.

Run with::

    python tests_pipeline.py

No pytest required, no Sionna required.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np

from radiomap.crossval import cv_score, kfold_indices, select
from radiomap.interpolators import (
    DetrendedKriging,
    InverseDistanceWeighting,
    NearestNeighbour,
    OrdinaryKriging,
    ThinPlateRBF,
)
from radiomap.metrics import mae, rmse, to_db, to_linear
from radiomap.sampling import block_design, block_origins, random_design
from radiomap.scene import synthetic_canyon
from radiomap.trend import LogDistanceTrend
from radiomap.variogram import (
    SphericalVariogram,
    empirical_variogram,
    fit_spherical,
)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _scatter(n=400, seed=0, extent=120.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-extent / 2, extent / 2, size=(n, 2))


def _spherical_field(xy, rng_m=15.0, sill=1.0, nugget=0.0, seed=0):
    """Draw a Gaussian field with an exact spherical covariance."""
    model = SphericalVariogram(nugget=nugget, sill=sill, rng=rng_m)
    d = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    cov = model.covariance(d)
    cov[np.diag_indices_from(cov)] = model.total_sill
    cov += 1e-8 * np.eye(len(xy))
    L = np.linalg.cholesky(cov)
    z = np.random.default_rng(seed).standard_normal(len(xy))
    return L @ z


# --------------------------------------------------------------------------
# 1-5: every interpolator reproduces its own observations
# --------------------------------------------------------------------------

@check("01 nearest neighbour is exact on its own observations")
def _():
    xy = _scatter(seed=1)
    z = _spherical_field(xy, seed=1)
    m = NearestNeighbour().fit(xy, z)
    assert np.allclose(m.predict(xy), z, atol=1e-10)


@check("02 IDW is exact on its own observations")
def _():
    xy = _scatter(seed=2)
    z = _spherical_field(xy, seed=2)
    m = InverseDistanceWeighting(power=2.0, k=16).fit(xy, z)
    assert np.allclose(m.predict(xy), z, atol=1e-8)


@check("03 thin-plate RBF is exact on its own observations")
def _():
    xy = _scatter(n=200, seed=3)
    z = _spherical_field(xy, seed=3)
    m = ThinPlateRBF(smoothing=0.0, neighbors=64).fit(xy, z)
    assert np.max(np.abs(m.predict(xy) - z)) < 1e-6


@check("04 ordinary kriging is exact on its own observations")
def _():
    xy = _scatter(n=200, seed=4)
    z = _spherical_field(xy, seed=4)
    m = OrdinaryKriging(k=32).fit(xy, z)
    assert np.max(np.abs(m.predict(xy) - z)) < 1e-6


@check("05 detrended kriging is exact on its own observations")
def _():
    xy = _scatter(n=200, seed=5)
    z = -30.0 - 40.0 * np.log10(np.maximum(np.linalg.norm(xy, axis=1), 1.0))
    z = z + _spherical_field(xy, seed=5)
    m = DetrendedKriging(k=32, tx_xy=(0.0, 0.0)).fit(xy, z)
    assert np.max(np.abs(m.predict(xy) - z)) < 1e-6


# --------------------------------------------------------------------------
# 6-7: the thin-plate spline reproduces a linear field exactly, IDW does not
# --------------------------------------------------------------------------

@check("06 thin-plate spline is exact on a linear field")
def _():
    xy = _scatter(n=300, seed=6)
    z = 3.0 + 1.5 * xy[:, 0] - 2.25 * xy[:, 1]
    target = _scatter(n=120, seed=60, extent=80.0)
    truth = 3.0 + 1.5 * target[:, 0] - 2.25 * target[:, 1]
    m = ThinPlateRBF(smoothing=0.0, neighbors=64).fit(xy, z)
    err = rmse(truth, m.predict(target))
    assert err < 1e-6, err


@check("07 IDW is not exact on a linear field")
def _():
    xy = _scatter(n=300, seed=6)
    z = 3.0 + 1.5 * xy[:, 0] - 2.25 * xy[:, 1]
    target = _scatter(n=120, seed=60, extent=80.0)
    truth = 3.0 + 1.5 * target[:, 0] - 2.25 * target[:, 1]
    m = InverseDistanceWeighting(power=2.0, k=16).fit(xy, z)
    err = rmse(truth, m.predict(target))
    assert err > 1.0, err


# --------------------------------------------------------------------------
# 8-10: the variogram fitter
# --------------------------------------------------------------------------

@check("08 variogram fitter recovers a planted correlation range")
def _():
    # One realisation is a noisy thing to assert on, so this averages ten.
    # Expect about 15.6 +/- 1.0 m for a planted 15 m.
    recovered = []
    for seed in range(1, 11):
        xy = _scatter(n=900, seed=seed, extent=150.0)
        z = _spherical_field(xy, rng_m=15.0, sill=1.0, seed=seed)
        lags, gamma, counts = empirical_variogram(xy, z, n_lags=25)
        recovered.append(fit_spherical(lags, gamma, counts).rng)
    m, s = float(np.mean(recovered)), float(np.std(recovered))
    print(f"        recovered range {m:.1f} +/- {s:.1f} m (planted 15.0 m)")
    assert 13.5 < m < 16.5, m


@check("09 variogram nugget is ~0 for a noiseless field")
def _():
    xy = _scatter(n=900, seed=9, extent=150.0)
    z = _spherical_field(xy, rng_m=20.0, sill=1.0, nugget=0.0, seed=9)
    lags, gamma, counts = empirical_variogram(xy, z, n_lags=25)
    model = fit_spherical(lags, gamma, counts)
    print(f"        nugget {model.nugget:.3f} of total sill {model.total_sill:.3f}")
    assert model.nugget < 0.15 * model.total_sill, model.nugget


@check("10 variogram nugget picks up added white noise")
def _():
    xy = _scatter(n=900, seed=10, extent=150.0)
    clean = _spherical_field(xy, rng_m=20.0, sill=1.0, seed=10)
    noisy = clean + np.random.default_rng(101).standard_normal(len(xy)) * 0.7
    lags, gamma, counts = empirical_variogram(xy, noisy, n_lags=25)
    model = fit_spherical(lags, gamma, counts)
    print(f"        nugget {model.nugget:.3f} (added noise variance 0.49)")
    assert model.nugget > 0.2, model.nugget


# --------------------------------------------------------------------------
# 11: detrending helps when extrapolating a trend
# --------------------------------------------------------------------------

@check("11 detrended kriging beats ordinary kriging extrapolating a pure trend")
def _():
    rng = np.random.default_rng(11)
    xy = rng.uniform(5.0, 60.0, size=(500, 2))
    far = rng.uniform(80.0, 130.0, size=(200, 2))

    def field(p):
        d = np.maximum(np.linalg.norm(p, axis=1), 1.0)
        return -28.6 - 40.9 * np.log10(d)

    z, truth = field(xy), field(far)
    ok = OrdinaryKriging(k=32).fit(xy, z)
    dk = DetrendedKriging(k=32, tx_xy=(0.0, 0.0)).fit(xy, z)
    e_ok, e_dk = rmse(truth, ok.predict(far)), rmse(truth, dk.predict(far))
    print(f"        OK {e_ok:.2f} dB vs DK {e_dk:.2f} dB")
    assert e_dk < 0.5 * e_ok, (e_ok, e_dk)


# --------------------------------------------------------------------------
# 12-14: the sampling designs
# --------------------------------------------------------------------------

@check("12 random design never scores a cell it observed")
def _():
    for seed in range(5):
        obs, sc = random_design(10_000, 0.05, seed)
        assert np.intersect1d(obs, sc).size == 0
        assert len(obs) + len(sc) == 10_000


@check("13 block design never leaks an observation into the hold-out block")
def _():
    rm = synthetic_canyon(seed=13)
    origins = block_origins(rm.xy, 60.0, 6)
    for origin in origins:
        for seed in (0, 1, 2):
            obs, sc = block_design(rm.xy, origin, 60.0, 0.20, seed)
            assert np.intersect1d(obs, sc).size == 0
            inside = (
                (rm.xy[obs, 0] >= origin[0])
                & (rm.xy[obs, 0] < origin[0] + 60.0)
                & (rm.xy[obs, 1] >= origin[1])
                & (rm.xy[obs, 1] < origin[1] + 60.0)
            )
            assert not inside.any(), "observation found inside the hold-out block"


@check("14 block design scores only cells inside the block")
def _():
    rm = synthetic_canyon(seed=14)
    origin = block_origins(rm.xy, 60.0, 6)[0]
    _, sc = block_design(rm.xy, origin, 60.0, 0.10, 0)
    assert sc.size > 200
    assert np.all(rm.xy[sc, 0] >= origin[0])
    assert np.all(rm.xy[sc, 0] < origin[0] + 60.0)
    assert np.all(rm.xy[sc, 1] >= origin[1])
    assert np.all(rm.xy[sc, 1] < origin[1] + 60.0)


# --------------------------------------------------------------------------
# 15-16: cross-validation
# --------------------------------------------------------------------------

@check("15 cross-validation prefers a sensible IDW power over a degenerate one")
def _():
    rm = synthetic_canyon(seed=15)
    obs, _ = random_design(rm.n_valid, 0.05, 0)
    xy, z = rm.xy[obs], rm.gain_db[obs]
    sensible = cv_score("idw", {"power": 2.0, "k": 16}, xy, z, folds=5)
    degenerate = cv_score("idw", {"power": 0.01, "k": 16}, xy, z, folds=5)
    print(f"        p=2 {sensible:.2f} dB vs p=0.01 {degenerate:.2f} dB")
    assert sensible < degenerate
    best, _score = select("idw", xy, z, folds=5)
    assert best["power"] >= 1.0, best


@check("16 k-fold splits are a partition of the observations")
def _():
    for folds in (3, 5, 10):
        seen = []
        for train, test in kfold_indices(257, folds, seed=7):
            assert np.intersect1d(train, test).size == 0
            assert len(train) + len(test) == 257
            seen.append(test)
        allt = np.sort(np.concatenate(seen))
        assert np.array_equal(allt, np.arange(257))


# --------------------------------------------------------------------------
# 17: the moving neighbourhood is not the reason for any result
# --------------------------------------------------------------------------

@check("17 moving-neighbourhood kriging matches the global solve on a small problem")
def _():
    xy = _scatter(n=90, seed=17, extent=60.0)
    z = _spherical_field(xy, rng_m=20.0, seed=17)
    target = _scatter(n=60, seed=170, extent=40.0)
    vg = SphericalVariogram(0.0, 1.0, 20.0)
    local = OrdinaryKriging(k=32, variogram=vg).fit(xy, z).predict(target)
    glob = OrdinaryKriging(k=90, variogram=vg).fit(xy, z).predict(target)
    d = float(np.max(np.abs(local - glob)))
    print(f"        max |local - global| = {d:.3f} dB")
    assert d < 0.5, d


# --------------------------------------------------------------------------
# 18-19: units and the accessibility mask
# --------------------------------------------------------------------------

@check("18 dB/linear conversion round-trips and the metrics are right")
def _():
    g = np.array([1e-9, 5e-10, 2.5e-8])
    assert np.allclose(to_linear(to_db(g)), g, rtol=1e-12)
    assert np.isclose(to_db(np.array([1e-9]))[0], -90.0)
    t = np.array([1.0, 2.0, 3.0])
    p = np.array([2.0, 4.0, 6.0])
    assert np.isclose(rmse(t, p), np.sqrt((1 + 4 + 9) / 3))
    assert np.isclose(mae(t, p), 2.0)
    # a cell no ray reached must be nan, not -inf
    assert np.isnan(to_db(np.array([0.0]))[0])


@check("19 building cells are excluded from the valid set")
def _():
    rm = synthetic_canyon(seed=19)
    assert rm.building_mask.any()
    assert rm.n_valid == int((~rm.building_mask).sum())
    assert rm.valid_mask.sum() == rm.n_valid
    grid = rm.to_grid(rm.gain_db)
    assert np.all(np.isnan(grid[rm.building_mask]))
    assert np.all(np.isfinite(grid[rm.valid_mask]))


# --------------------------------------------------------------------------

def main() -> int:
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
        else:
            print(f"pass  {name}")
    total = len(CHECKS)
    print("-" * 62)
    print(f"{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
