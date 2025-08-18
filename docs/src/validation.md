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

```
![](assets/ke_series.png)
```

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

After saving, include figures in the appropriate pages (Theory/Remeshing/Parallelization), e.g.:

```
![](assets/fig-0001.png)
```

## Notes

- Ensure that any figure usage aligns with the thesis’ distribution rights.
- For reproducibility, note the exact example parameters used to generate the series file.

