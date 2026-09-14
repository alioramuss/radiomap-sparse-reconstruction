"""The reconstruction sweep itself: both designs, all methods, all fractions."""

from __future__ import annotations

import time
from typing import Any, Iterator

import numpy as np

from .config import SweepConfig
from .crossval import select
from .interpolators import METHODS
from .metrics import mae, rmse
from .sampling import block_design, block_origins, random_design
from .scene import RadioMapData


def _extra_kwargs(method: str, rm: RadioMapData) -> dict[str, Any]:
    """Per-method arguments that are not hyperparameters."""
    if method == "dk":
        return {"tx_xy": rm.tx_xy}
    return {}


def run_once(
    rm: RadioMapData,
    method: str,
    observed: np.ndarray,
    scored: np.ndarray,
    cfg: SweepConfig,
) -> dict[str, Any]:
    """Fit one method on one (observed, scored) split and score it."""
    xy_obs, z_obs = rm.xy[observed], rm.gain_db[observed]
    xy_hid, z_hid = rm.xy[scored], rm.gain_db[scored]

    extra = _extra_kwargs(method, rm)
    params, cv = select(
        method,
        xy_obs,
        z_obs,
        folds=cfg.cv_folds,
        seed=cfg.cv_seed,
        extra=extra,
    )

    t0 = time.perf_counter()
    model = METHODS[method](**{**params, **extra})
    model.fit(xy_obs, z_obs)
    pred = model.predict(xy_hid)
    elapsed = time.perf_counter() - t0

    return {
        "method": method,
        "n_observed": int(len(observed)),
        "n_scored": int(len(scored)),
        "params": params,
        "cv_rmse": cv,
        "rmse": rmse(z_hid, pred),
        "mae": mae(z_hid, pred),
        "seconds": elapsed,
    }


def random_sweep(
    rm: RadioMapData,
    cfg: SweepConfig | None = None,
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    cfg = cfg or SweepConfig()
    rows: list[dict[str, Any]] = []
    for frac in cfg.sampling_fractions:
        for seed in cfg.random_seeds:
            observed, scored = random_design(rm.n_valid, frac, seed)
            for method in cfg.methods:
                row = run_once(rm, method, observed, scored, cfg)
                row.update(design="random", fraction=frac, seed=seed, block=-1)
                rows.append(row)
                if verbose:
                    print(
                        f"random f={frac:<5} seed={seed} {method:<8} "
                        f"RMSE={row['rmse']:6.2f} dB  ({row['seconds']:.1f}s)"
                    )
    return rows


def block_sweep(
    rm: RadioMapData,
    cfg: SweepConfig | None = None,
    *,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    cfg = cfg or SweepConfig()
    origins = block_origins(rm.xy, cfg.block_side_m, cfg.n_block_positions)
    if verbose:
        print(f"{len(origins)} hold-out block positions: {origins}")

    rows: list[dict[str, Any]] = []
    for frac in cfg.sampling_fractions:
        for b, origin in enumerate(origins):
            for seed in cfg.block_seeds:
                observed, scored = block_design(
                    rm.xy, origin, cfg.block_side_m, frac, seed
                )
                for method in cfg.methods:
                    row = run_once(rm, method, observed, scored, cfg)
                    row.update(design="block", fraction=frac, seed=seed, block=b)
                    rows.append(row)
                    if verbose:
                        print(
                            f"block  f={frac:<5} b={b} seed={seed} {method:<8} "
                            f"RMSE={row['rmse']:6.2f} dB  ({row['seconds']:.1f}s)"
                        )
    return rows


def summarise(rows: list[dict[str, Any]]) -> dict[tuple[str, str, float], tuple[float, float, int]]:
    """``(design, method, fraction) -> (mean RMSE, sd RMSE, n runs)``."""
    buckets: dict[tuple[str, str, float], list[float]] = {}
    for r in rows:
        key = (r["design"], r["method"], r["fraction"])
        buckets.setdefault(key, []).append(r["rmse"])
    return {
        k: (float(np.mean(v)), float(np.std(v, ddof=0)), len(v))
        for k, v in buckets.items()
    }


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    import csv

    fields = [
        "design",
        "fraction",
        "seed",
        "block",
        "method",
        "n_observed",
        "n_scored",
        "rmse",
        "mae",
        "cv_rmse",
        "seconds",
        "params",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["params"] = repr(r.get("params", {}))
            w.writerow(out)
