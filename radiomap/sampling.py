"""The two sampling designs.

``random_design``
    Draw observations uniformly from the valid cells, score on the rest.  Every
    unobserved cell has an observation nearby, so this measures gap filling.
    It is what nearly all of this literature does.

``block_design``
    Remove a contiguous square entirely, draw observations from what remains,
    and score only inside the square.  Nothing inside is ever observed, so this
    measures extrapolation into unseen ground.

Both return disjoint (observed, scored) index pairs into the valid-cell arrays.
The disjointness is asserted here rather than trusted, because a single leaked
observation inside a hold-out block would turn a 12 dB result into a 3 dB one
and look entirely plausible.
"""

from __future__ import annotations

import numpy as np


def random_design(
    n_valid: int,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(observed_idx, scored_idx)`` for the random design."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be in (0, 1)")
    rng = np.random.default_rng(seed)
    n_obs = max(1, int(round(fraction * n_valid)))
    if n_obs >= n_valid:
        raise ValueError("fraction leaves nothing to score on")
    perm = rng.permutation(n_valid)
    observed = np.sort(perm[:n_obs])
    scored = np.sort(perm[n_obs:])
    _assert_disjoint(observed, scored)
    return observed, scored


def block_origins(
    xy: np.ndarray,
    side_m: float,
    n_positions: int,
    *,
    min_cells: int = 200,
) -> list[tuple[float, float]]:
    """Candidate lower-left corners for the hold-out block.

    Positions are spread along the long axis of the valid region, which on a
    street canyon means along the street.  Only positions whose block contains
    at least ``min_cells`` valid cells are kept, so a block that lands mostly on
    building footprints is not counted as a replicate.
    """
    xy = np.asarray(xy, dtype=float)
    x0, y0 = xy.min(axis=0)
    x1, y1 = xy.max(axis=0)

    span_x, span_y = x1 - x0, y1 - y0
    long_axis = 0 if span_x >= span_y else 1

    lo = (x0, y0)[long_axis]
    hi = (x1, y1)[long_axis] - side_m
    if hi <= lo:
        raise ValueError("block side is larger than the scene")

    other_lo = (x0, y0)[1 - long_axis]
    other_hi = (x1, y1)[1 - long_axis] - side_m
    other = 0.5 * (other_lo + max(other_lo, other_hi))

    out: list[tuple[float, float]] = []
    # Oversample the axis, then keep the first n_positions that are populated
    # enough to be worth scoring.
    for t in np.linspace(lo, hi, 3 * n_positions):
        origin = (t, other) if long_axis == 0 else (other, t)
        inside = _inside_block(xy, origin, side_m)
        if inside.sum() >= min_cells:
            if all(abs(t - (o[long_axis])) > 0.5 * side_m for o in out):
                out.append((float(origin[0]), float(origin[1])))
        if len(out) == n_positions:
            break

    if not out:
        raise ValueError("no block position contains enough valid cells")
    return out


def block_design(
    xy: np.ndarray,
    origin: tuple[float, float],
    side_m: float,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(observed_idx, scored_idx)`` for the contiguous hold-out design.

    ``fraction`` is applied to the cells *outside* the block, so the observation
    budget is comparable to the random design at the same fraction.
    """
    xy = np.asarray(xy, dtype=float)
    inside = _inside_block(xy, origin, side_m)
    outside_idx = np.flatnonzero(~inside)
    scored = np.flatnonzero(inside)

    if scored.size == 0:
        raise ValueError("hold-out block contains no valid cells")

    rng = np.random.default_rng(seed)
    n_obs = max(1, int(round(fraction * xy.shape[0])))
    n_obs = min(n_obs, outside_idx.size)
    observed = np.sort(rng.permutation(outside_idx)[:n_obs])

    _assert_disjoint(observed, scored)
    return observed, scored


def _inside_block(xy: np.ndarray, origin, side_m: float) -> np.ndarray:
    ox, oy = origin
    return (
        (xy[:, 0] >= ox)
        & (xy[:, 0] < ox + side_m)
        & (xy[:, 1] >= oy)
        & (xy[:, 1] < oy + side_m)
    )


def _assert_disjoint(observed: np.ndarray, scored: np.ndarray) -> None:
    if np.intersect1d(observed, scored).size:
        raise AssertionError("observed and scored index sets overlap")
