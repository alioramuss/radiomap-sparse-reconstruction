"""Trace the ground-truth radio map and cache it.

    python scripts/01_trace_map.py --out data/map_3p5GHz.npz

Needs Sionna.  Takes about 86 s on two CPU cores at the default 2e8 rays.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from radiomap.config import SceneConfig
from radiomap.scene import trace


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data/map_3p5GHz.npz")
    p.add_argument("--frequency-ghz", type=float, default=3.5)
    p.add_argument("--rays", type=float, default=2e8)
    p.add_argument("--cell-size", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    cfg = SceneConfig(
        frequency_hz=args.frequency_ghz * 1e9,
        samples_per_tx=int(args.rays),
        cell_size_m=args.cell_size,
        seed=args.seed,
    )
    print(cfg)

    rm = trace(cfg)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rm.save(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
