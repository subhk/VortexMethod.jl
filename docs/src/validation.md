# Validation

This page outlines simple validation steps and how to produce figures for the docs.

## Kelvin–Helmholtz (KH) run

Run the advanced example with MPI to produce a time-series JLD2 file with snapshots and (optionally) kinetic energy in the params:

```
mpirun -n 4 julia --project examples/advanced_kh3d.jl
```

By default, it writes to `checkpoints/advanced_series.jld2`. You can change the output path in the example script if desired.

## Kinetic energy plot

Use the provided plotting script to generate a KE vs time graphic into the docs assets. Set `SERIES_FILE` and `OUTPUT_PNG`:

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
OUTPUT_PNG=docs/src/assets/ke_series.png \
julia --project examples/plot_series_ke.jl
```

Then reference it here:

![](assets/ke_series.png)

## Extracting figures from the thesis (optional)

If the thesis figures are available in `mstock_dissertation.pdf`, extract selected images using `pdfimages` (Poppler) or `pdftoppm`:

- macOS (Homebrew): `brew install poppler`
- Ubuntu/Debian: `sudo apt-get install poppler-utils`

Extract embedded images:

```
pdfimages -png mstock_dissertation.pdf docs/src/assets/fig
```

or rasterize specific pages to PNGs (e.g., pages 10–12):

```
pdftoppm -png -f 10 -l 12 mstock_dissertation.pdf docs/src/assets/page
```

After saving, include figures in the appropriate pages (Theory/Remeshing/Parallelization). For the following thesis figure numbers, save with these suggested filenames:

- 1.23 → docs/src/assets/fig_1_23.png (Theory)
- 3.19 → docs/src/assets/fig_3_19.png (Theory/Parallelization)
- 3.26 → docs/src/assets/fig_3_26.png (Remeshing)
- 3.52 → docs/src/assets/fig_3_52.png (Remeshing)

![](assets/fig_1_23.png)

### Direct page extraction (user-provided pages)

You shared the page numbers for these figures: 3.19, 1.23, 3.26, 3.52 → pages 90, 98, 118, 140.

Extract at 300 dpi and save with the suggested filenames:

```
# Figure 3.19 on page 90
pdftoppm -png -r 300 -f 90 -l 90 mstock_dissertation.pdf docs/src/assets/fig_3_19
mv docs/src/assets/fig_3_19-090.png docs/src/assets/fig_3_19.png || \
  mv docs/src/assets/fig_3_19-90.png docs/src/assets/fig_3_19.png

# Figure 1.23 on page 98
pdftoppm -png -r 300 -f 98 -l 98 mstock_dissertation.pdf docs/src/assets/fig_1_23
mv docs/src/assets/fig_1_23-098.png docs/src/assets/fig_1_23.png || \
  mv docs/src/assets/fig_1_23-98.png docs/src/assets/fig_1_23.png

# Figure 3.26 on page 118
pdftoppm -png -r 300 -f 118 -l 118 mstock_dissertation.pdf docs/src/assets/fig_3_26
mv docs/src/assets/fig_3_26-118.png docs/src/assets/fig_3_26.png

# Figure 3.52 on page 140
pdftoppm -png -r 300 -f 140 -l 140 mstock_dissertation.pdf docs/src/assets/fig_3_52
mv docs/src/assets/fig_3_52-140.png docs/src/assets/fig_3_52.png
```

## Notes

- Ensure that any figure usage aligns with the thesis’ distribution rights.
- For reproducibility, note the exact example parameters used to generate the series file.

## Results (example figures)

- Gamma magnitude snapshot:

Generate and embed a snapshot plot of |γ| at a chosen time (snapshot index):

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
SNAP_INDEX=10 \
OUTPUT_PNG=docs/src/assets/snapshot_gamma.png \
julia --project examples/plot_snapshot_gamma.jl
```

Then include it here:

![](assets/snapshot_gamma.png)

If you share the thesis figure numbers/pages to include (and captions), we’ll add them directly to the relevant sections with proper references.
