"""Hyperparameter selection by k-fold cross-validation on the observations only.

The held-out truth is never touched.  This matters more than it sounds: with
the hold-out design the tempting shortcut is to tune against the block, which
turns an extrapolation experiment into an interpolation one with extra steps.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from .interpolators import (
    DetrendedKriging,
    InverseDistanceWeighting,
    METHODS,
    NearestNeighbour,
    OrdinaryKriging,
    ThinPlateRBF,
)
from .metrics import rmse

#: Search grids.  Deliberately small -- these are baselines, and an expensive
#: search would make the comparison about tuning budget rather than method.
DEFAULT_GRIDS: dict[str, list[dict[str, Any]]] = {
    "nearest": [{}],
    "idw": [
        {"power": p, "k": k}
        for p in (1.0, 1.5, 2.0, 3.0, 4.0)
        for k in (8, 16, 32)
    ],
    "rbf": [
        {"smoothing": s, "neighbors": n}
        for s in (0.0, 1.0, 10.0)
        for n in (32, 64)
    ],
    "ok": [{"k": k} for k in (32, 64)],
    "dk": [{"k": k} for k in (32, 64)],
}


def kfold_indices(n: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``[(train_idx, test_idx), ...]``."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    splits = np.array_split(perm, folds)
    out = []
    for f in range(folds):
        test = np.sort(splits[f])
        train = np.sort(np.concatenate([splits[g] for g in range(folds) if g != f]))
        out.append((train, test))
    return out


def cv_score(
    method: str,
    params: dict[str, Any],
    xy: np.ndarray,
    z: np.ndarray,
    *,
    folds: int = 5,
    seed: int = 12345,
    extra: dict[str, Any] | None = None,
) -> float:
    """Mean RMSE across folds for one hyperparameter setting."""
    cls = METHODS[method]
    extra = extra or {}
    errors = []
    for train, test in kfold_indices(len(xy), folds, seed):
        model = cls(**{**params, **extra})
        model.fit(xy[train], z[train])
        errors.append(rmse(z[test], model.predict(xy[test])))
    return float(np.mean(errors))


def select(
    method: str,
    xy: np.ndarray,
    z: np.ndarray,
    *,
    grid: Iterable[dict[str, Any]] | None = None,
    folds: int = 5,
    seed: int = 12345,
    extra: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    """Pick the hyperparameters with the lowest cross-validated RMSE."""
    grid = list(grid if grid is not None else DEFAULT_GRIDS[method])
    best, best_score = grid[0], np.inf
    for params in grid:
        try:
            score = cv_score(
                method, params, xy, z, folds=folds, seed=seed, extra=extra
            )
        except Exception:  # a degenerate setting should lose, not crash the sweep
            score = np.inf
        if np.isfinite(score) and score < best_score:
            best, best_score = params, score
    return best, float(best_score)
