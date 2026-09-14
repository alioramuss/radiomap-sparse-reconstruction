# Sparse radio map reconstruction on a Sionna RT street canyon

Classical interpolation baselines on a ray-traced radio map, scored two ways:
filling scattered gaps, and predicting into a block of street that was never
measured. **The two answers differ by a factor of five.**

Interim results for the first of two tasks suggested by Tianrun Qi (Southeast
University, visiting IDCoM at the University of Edinburgh), hosted by
Prof John Thompson. The write-up is in `report/`; the findings are summarised
below.

---

## The three findings

**1. The ground truth is noisier than the thing being measured**, at the ray
counts published work uses. At 2×10⁵ rays — the count used by the closest
published Sionna RT reconstruction study — 29% of open-street cells receive no
ray at all, and the cells that do disagree between random seeds by 4.75 dB.
Reconstruction errors in this literature are typically 2–5 dB. Fixing it costs
86 seconds of compute, so it seemed worth fixing before anything else.

**2. The contiguous hold-out is not a harder version of random sampling. It is
a different problem.** Errors are about five times larger, and they stop
improving with more data. Going from 1% to 20% of the map cuts random-sampling
error by 63%; the same twentyfold increase buys 2.3 dB under hold-out, and the
curve is flat from 5% onward.

**3. Where the gap is matters more than which interpolator fills it.** Under
hold-out the standard deviation across block positions (1.5–8.5 dB) is larger
than the gaps between methods, so most of the method ranking is not resolved.

---

## Results

### Ray count convergence

Two independent traces of the identical scene, compared over open-street cells
reached in both. A single map sits the seed-to-seed RMSE divided by √2 from the
converged mean.

| Rays per tx | Street cells with no ray | Seed-to-seed RMSE | One map vs. mean | Solve time |
|---|---|---|---|---|
| 2×10⁵ | 28.94% | 4.75 dB | 3.36 dB | <1 s |
| 1×10⁶ | 9.10% | 4.08 dB | 2.89 dB | 1 s |
| 5×10⁶ | 0.98% | 2.49 dB | 1.76 dB | 3 s |
| 2×10⁷ | 0.20% | 1.15 dB | 0.81 dB | 9 s |
| 5×10⁷ | 0.18% | 0.67 dB | 0.47 dB | 22 s |
| 1×10⁸ | 0.18% | 0.46 dB | 0.32 dB | 44 s |
| **2×10⁸** | **0.18%** | **0.32 dB** | **0.23 dB** | **86 s** |
| 5×10⁸ | 0.18% | 0.20 dB | 0.14 dB | 269 s |

From 10⁶ rays upward the error falls as *N*<sup>−1/2</sup>, as it should for a
Monte Carlo estimator. This study runs at 2×10⁸, where the ground truth sits
0.23 dB from the converged mean — roughly an order of magnitude below the
smallest reconstruction error below.

A second, independent check agrees: the fitted variogram has a nugget of
essentially zero. Independent per-cell noise would appear as variance that does
not vanish at zero lag.

### Reconstruction error

RMSE in dB, mean ± standard deviation. Random: 5 seeds. Block: 6 block
positions × 3 seeds = 18 runs per cell.

**Random sampling**

| Method | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Nearest neighbour | 7.44 ±0.50 | 5.74 ±0.39 | 4.27 ±0.12 | 3.51 ±0.19 | 2.97 ±0.13 |
| IDW | 6.15 ±0.41 | 4.81 ±0.33 | 3.48 ±0.10 | 2.82 ±0.17 | 2.32 ±0.08 |
| RBF | 5.60 ±0.40 | 4.40 ±0.21 | 3.12 ±0.12 | 2.56 ±0.08 | **2.08 ±0.10** |
| Ordinary kriging | 5.75 ±0.30 | 4.44 ±0.25 | 3.11 ±0.10 | 2.56 ±0.08 | **2.08 ±0.08** |
| Detrended kriging | 5.79 ±0.36 | 4.45 ±0.25 | 3.12 ±0.10 | 2.57 ±0.08 | **2.08 ±0.08** |

**Contiguous 60 m hold-out**

| Method | 1% | 2% | 5% | 10% | 20% |
|---|---|---|---|---|---|
| Nearest neighbour | 14.57 ±4.24 | 13.56 ±4.37 | 12.10 ±3.71 | 12.12 ±4.27 | 11.94 ±3.92 |
| IDW | 14.39 ±4.18 | 13.62 ±4.85 | 11.68 ±4.23 | 11.27 ±4.41 | 11.27 ±3.99 |
| RBF | 15.63 ±7.51 | 12.24 ±3.47 | 10.36 ±3.19 | 10.00 ±3.32 | 10.88 ±8.47 |
| Ordinary kriging | 13.29 ±3.46 | 12.32 ±3.64 | 10.73 ±3.87 | 10.33 ±4.14 | **9.68 ±2.56** |
| Detrended kriging | **12.01 ±1.51** | 12.70 ±1.91 | 10.76 ±2.52 | 10.59 ±2.88 | 10.04 ±1.77 |

Two things survive the noise:

- **RBF is the least reliable extrapolator.** Best mean at 5% and 10%, worst
  variance at 1% and 20% (±8.47 dB). A thin-plate spline has nothing holding it
  down outside the convex hull of its data.
- **Detrending buys stability, not accuracy.** Fitting
  `gain_dB = −28.6 − 40.9 log₁₀(d)` before kriging the residual gives roughly
  the same mean error but halves to quarters the variance across block
  positions. The trend takes the residual standard deviation from 17.7 to
  14.5 dB and the correlation range from 27.8 m to 16.2 m.

### Why the hold-out is so much harder

The truth inside a block is not a smooth field. It is piecewise smooth, cut by
sharp edges where a diffraction wedge or a specular reflection boundary starts
and stops — the geometry of the canyon written into the map. IDW, RBF and
kriging all assume the field varies smoothly with distance. Under random
sampling that assumption survives, because observations land on both sides of
every edge. Under hold-out there is nothing inside the block to pin anything
down, so every method returns a smooth surface and the error map is the edges
themselves at ±15 dB.

This is why more samples outside do not help. The missing information is not
"what is the level around here", which the surroundings already answer. It is
"where does the shadow boundary run", which is a fact about geometry that no
amount of sampling elsewhere in the street recovers.

---

## Setup

| | |
|---|---|
| scene | `simple_street_canyon` (Sionna built-in) |
| solver | Sionna RT 2.0.1 |
| carrier | 3.5 GHz |
| measurement plane | 1.5 m |
| cell size | 1.0 m |
| grid | 122 × 187 |
| tx position | (−33, 11, 32) m |
| tx power | 30 dBm, isotropic |
| max depth | 3 |
| rays per tx | 2 × 10⁸ |
| propagation | LoS + specular reflection + diffraction; diffuse scattering and refraction **off** |
| valid cells | 17,547 (76.9% of grid), −150 to −73 dB, sd 17.6 dB |
| solver time | 86 s, 2 CPU cores |

Of the 22,814 cells, 5,236 sit under building geometry and are excluded, found
by casting a vertical ray from each cell centre. A handful more open-street
cells are never reached by any ray.

### Two things the Sionna docs do not say

- `path_gain` is returned as a **linear** power ratio, not dB. Values on this
  scene are of order 10⁻⁹.
- `refraction` defaults to **on**. Left on, rays leak into building interiors
  and those cells stop being structurally unreachable, which quietly changes
  what an accessibility mask means. It is switched off here on purpose.

---

## Install and run

```bash
git clone https://github.com/alioramuss/radiomap-sparse-reconstruction
cd radiomap-sparse-reconstruction
pip install -r requirements.txt
```

Everything except `scripts/01_trace_map.py` and `scripts/02_ray_convergence.py`
runs on numpy, scipy and matplotlib alone, so the reconstruction study can be
re-run from a cached `.npz` map without installing Sionna at all.

```bash
# 1. trace the ground-truth map               (needs Sionna, ~86 s)
python scripts/01_trace_map.py --out data/map_3p5GHz.npz

# 2. ray-count convergence                    (needs Sionna, ~15 min:
#    two seeds at each of eight ray counts)
python scripts/02_ray_convergence.py --out results/convergence.csv

# 3. the sweep: both designs, all methods     (no Sionna, ~1-2 h on 2 cores;
#    575 fits -- 125 random, 450 block)
python scripts/03_reconstruction_sweep.py --map data/map_3p5GHz.npz \
    --out results/sweep.csv

# 4. figures 2 and 3 from the report          (no Sionna, minutes)
python scripts/05_figures.py --map data/map_3p5GHz.npz \
    --sweep results/sweep.csv --outdir figures

# 5. dB vs linear ablation                    (no Sionna, minutes)
python scripts/04_db_vs_linear.py --map data/map_3p5GHz.npz
```

To check the plumbing without a ray tracer or a cached map, every script that
does not need Sionna accepts `--synthetic`, which substitutes a piecewise-smooth
stand-in field defined in `radiomap/scene.py`. **No result in the report comes
from it** — it exists so the pipeline can be exercised deterministically.

```bash
python scripts/03_reconstruction_sweep.py --synthetic --design random
```

There is also `notebooks/sparse_radio_map_reconstruction.ipynb`, which runs top
to bottom on a free Colab CPU runtime.

---

## Tests

```bash
python tests_pipeline.py      # 19/19 checks passed
```

Nineteen known-answer checks, no pytest and no Sionna required. Each one
encodes a fact that has to be true for a number above to mean what it says:

- every interpolator reproduces its own observations (checks 1–5)
- thin-plate splines are exact on a linear field, and IDW is not (6–7)
- the variogram fitter recovers a planted 15 m correlation range as
  15.6 ± 1.0 m across ten realisations, reports a near-zero nugget on a
  noiseless field, and picks up added white noise as a nugget (8–10)
- detrended kriging beats ordinary kriging when extrapolating a pure trend (11)
- the random design never scores a cell it observed; the block design never
  leaks an observation into the hold-out block, and scores only inside it (12–14)
- cross-validation prefers a sensible IDW power over a degenerate one, and the
  k-fold splits partition the observations (15–16)
- the moving neighbourhood matches the global kriging solve to better than
  0.5 dB on a small problem, so it is not the reason for any result (17)
- the dB/linear conversion round-trips, a cell no ray reached is `nan` and not
  `-inf`, and the metrics are right (18)
- building cells are excluded from the valid set (19)

---

## Decisions a reader might want to argue with

- Errors are scored on **unobserved valid cells only**. Including observed cells
  would flatter IDW, RBF and kriging, all of which reproduce their own
  observations exactly, and the flattery would grow with sampling density.
- Cells under building geometry are excluded by **vertical ray casting**, not by
  thresholding the gain. Thresholding would throw away genuine deep-shadow
  street cells, which are exactly the cells the hold-out experiment is about.
- Kriging uses a **moving neighbourhood** of the 32 or 64 closest observations,
  chosen by cross-validation. Global kriging would need an n³ solve at these
  sample counts.
- The variogram is fitted on a **10% draw**, which is what a real user would
  have, not on the full map.
- Hyperparameters for every method are chosen by 5-fold cross-validation **on
  the observations only**, never against the held-out truth.
- The block side (60 m) is about **twice the 27.8 m correlation range** fitted
  from the variogram. A block narrower than the correlation range is still
  measuring interpolation.

---

## What this does not show

One scene, one transmitter position, one frequency, one solver, no measurement
noise added on top of the simulation. The block results come from six positions
in a single street layout, and the spread across those six is large enough that
a different layout could plausibly reorder the methods. Nothing here should be
read as a general claim about interpolators.

### Next

- **Does the required sampling density scale with frequency?** Theory says the
  estimation problem is invariant to scaling wavelength and distance together,
  so spacing should track λ and density should go as *f*². At 28 and 39 GHz the
  canyon becomes near-binary line of sight and the map develops harder edges, so
  the smoothness assumption should break down faster than that law predicts.
  The grid has to be fixed in wavelengths or in metres, deliberately and
  stated, because the two give different questions.
- **Does a building mask close the gap?** The missing information is geometric.
  Giving the interpolator the building footprints — free in a simulation, cheap
  in reality — is the smallest change that could test that diagnosis.
- **dB or linear.** Everything here interpolates in dB, which is the field
  convention; part of the literature scores in linear power instead, which
  weights cells near the transmitter far more heavily, and no one appears to
  have quantified the difference. `scripts/04_db_vs_linear.py` is that
  ablation, written but not yet run.

---

## Layout

```
radiomap/
  config.py         every number from the report, in one place
  scene.py          Sionna RT tracing, building mask, synthetic stand-in
  sampling.py       random and contiguous hold-out designs
  interpolators.py  NN, IDW, thin-plate RBF, ordinary and detrended kriging
  variogram.py      empirical variogram + spherical model fit
  trend.py          log-distance path-loss trend
  crossval.py       k-fold hyperparameter selection
  metrics.py        RMSE, MAE, dB/linear
  sweep.py          the experiment loop
scripts/
  01_trace_map.py           trace and cache the ground-truth map  (Sionna)
  02_ray_convergence.py     the seed-to-seed noise floor          (Sionna)
  03_reconstruction_sweep.py  both designs, all methods, all fractions
  04_db_vs_linear.py        the dB vs linear power ablation
  05_figures.py             figures 2 and 3 from the report
notebooks/          self-contained Colab notebook
report/             the interim write-up sent to IDCoM, 13 September 2026
tests_pipeline.py   19 known-answer checks
```

## Licence

MIT. See `LICENSE`.
