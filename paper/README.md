# Paper draft

`main.tex` is a complete IEEE conference-format draft (IEEEtran), eight pages.
The venue is not settled: it was drafted with a letters format in mind, but at
this length a four-page letter would mean cutting the convergence and depth
sections.

## Reference model

The transmitter is a rooftop site at 32 m, standing on the footprint of a
21.8 m building. 38.901 specifies UMi street canyon at a 10 m base station and
UMa at 25 m, so UMa is the closer match and is the paper's primary reference;
UMi is reported alongside it. A matched-height UMi run is not possible in this
scene, since lowering the transmitter to 10 m puts it inside the building: it
then reaches 988 of 22,814 cells with no line of sight at all.

This matters when regenerating anything, because several scripts take a
`--model` flag and the paper's numbers are the `uma` ones.

## Compiling

Overleaf is the easiest route: create a project, upload `main.tex`, `refs.bib`
and the two figures in a `figures/` folder, and it will find IEEEtran without
any setup. Compile twice so the bibliography resolves.

## Figures

Neither figure is committed, since `figures/*.png` is gitignored. Generate
them into `paper/figures/`:

```bash
python scripts/15_umi_figure.py --data data/bands_los.npz \
       --outdir paper/figures --ieee --model uma

python scripts/19_roughness_analysis.py --model uma \
       --out results/roughness_summary_uma.json
python scripts/20_roughness_figure.py --model uma \
       --summary results/roughness_summary_uma.json \
       --out paper/figures/fig_roughness_uma_ieee.png --ieee
```

| File the paper expects | Produced by |
|---|---|
| `umi_comparison_uma_ieee.png` | `scripts/15_umi_figure.py --model uma --ieee` |
| `fig_roughness_uma_ieee.png` | `scripts/20_roughness_figure.py --model uma --ieee` |

The `--ieee` flag renders at 7.16 in, the width of an IEEE double-column
`figure*`, with fonts set for that size. Render at the final width rather than
letting LaTeX shrink a screen-sized figure, which is what makes figure text
unreadable in print. Both are 2x2 panel figures, so they span both columns; at
a single column each panel would be 1.75 in wide and unreadable.

## Which script makes which table

| Table | Script |
|---|---|
| I, convergence | `scripts/06_multiband_convergence.py` |
| II and III, model comparison and frequency slope | `scripts/14_umi_comparison.py` |
| IV, interaction depth | `scripts/13_trace_bands_los.py --max-depth 5` |
| V, roughness sweep | `scripts/18_roughness_sweep.py`, then `scripts/19_roughness_analysis.py --model uma` |

Data files the scripts expect are regenerated rather than committed; `data/`
and `results/` are gitignored apart from this structure.
