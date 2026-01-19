# Validation

This page describes validation test cases and comparisons with analytical solutions, following [Stock (2006), §3.1.2, §3.4.4–3.4.5, §3.7.2](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837).

## Element Discretization Tests

### Stretch Parallel to Vorticity

A flat vortex sheet with ``\boldsymbol{\gamma}(t=0) = 1.0\hat{\mathbf{j}}`` is subjected to an artificial velocity field causing stretch parallel to the vorticity direction.

**Expected result:** ``\partial\boldsymbol{\gamma}/\partial t = 0`` (constant vortex sheet strength)

This test validates that the circulation-based discretization correctly handles the interplay between vortex stretching and the in-sheet dilatation term.

### Stretch Transverse to Vorticity

A flat vortex sheet with ``\boldsymbol{\gamma}(t=0) = 1.0\hat{\mathbf{i}}`` experiences strain transverse to the vorticity direction.

**Expected behavior:**
- Peak extensional strain (y=0.25): ``\gamma_x = e^{-2\pi t}`` (exponential decay)
- Peak compressional strain (y=0.75): ``\gamma_x = e^{+2\pi t}`` (exponential growth)

This test verifies that the method handles transverse strain—something neither standard vortex filament nor vortex particle methods can easily account for.

## Vortex Ring Circulation Conservation

### Setup

A unit-radius sphere centered at the origin with initial vortex sheet strength satisfying the no-through-flow condition in potential flow:

```math
\boldsymbol{\gamma} = -\frac{3}{2}r\,(\hat{\mathbf{n}} \times \mathbf{U}_\infty)
```

For ``\mathbf{U}_\infty = -\hat{\mathbf{k}}``, this gives ``\Gamma_{\text{ring}} = 3``.

### Circulation Measurement

The vortex ring circulation is computed as:

```math
\Gamma_{\text{ring}} = \sum_{p=1}^{N} \frac{1}{2\pi\|\mathbf{x}_p\|^2} \sum_{i=1}^{3} \Gamma_{p,i}\,(\Delta\mathbf{l}_{p,i} \times \mathbf{x}_p) \cdot \hat{\mathbf{k}}
```

### Expected Results

- Initial discretization error: second-order in mean triangle edge length
- Circulation should remain constant (within discretization error) throughout the simulation
- Different midpoint methods (geometric, spline, cylindrical) should all conserve circulation

## Baroclinic Validation (Rayleigh–Taylor)

### Stable Oscillation

For a sinusoidally-perturbed interface with lighter fluid above (``A > 0``):

| Parameter | Value |
|-----------|-------|
| Domain | [0:1]×[0:1]×[-4:4] |
| Perturbation | ``z = 0.01\sin(2\pi x)`` |
| Wavenumber | ``k = 2\pi`` |
| Boussinesq coefficient | ``\theta = 1`` |
| **Expected period** | ``\tau = 2\pi/\sqrt{2\pi} \approx 2.507`` |

### Unstable Growth

For heavier fluid above (``A < 0``):

```math
\text{Magnification} = \frac{f_{\max}}{F_0} = \cosh(t\sqrt{2\pi})
```

### Convergence

- M4' kernel: lowest errors, approximately second-order convergence for stable period
- Area-weighting and Peskin: approximately first-order convergence
- Regularization effect: longer periods (stable) and smaller growth rates (unstable) than theory

## Kelvin–Helmholtz Instability

### 2D Periodic Shear Layer

Setup following Krasny (1986) and Tryggvason (1989):

```math
z'(x,t=0) = 0.01\sin(2\pi x), \quad \gamma_y(x,t=0) = 1.0
```

**Dimensionless parameters:**
- ``\delta^* = \delta/\lambda``: regularization length / wavelength
- Typical values: ``\delta^* = 0.05`` to ``0.2``

**Expected behavior:**
- Roll-up time scales with ``\delta^*``
- Higher resolution (smaller ``\delta^*``) → sharper roll-up
- Regularization prevents finite-time singularity

### 3D Doubly-Periodic Shear Layer

Including spanwise perturbation at the most unstable 3D mode (Pierrehumbert & Widnall, 1982):

```math
x' = x + 0.01\sin(2\pi x), \quad z' = z + 0.01\sin(2\pi x) + 0.01\sin(4\pi y)
```

**Expected phenomena:**
- Initial 2D roll-up
- Development of streamwise ribs
- 3D instability growth

### KH with Stabilizing Stratification

At the critical Richardson number (``Ri_c = 1/4``), the shear layer is marginally stable. Linear theory gives the growth rate:

```math
n = \frac{ik\,\Delta u\,(\rho_2 - \rho_1)}{2(\rho_1 + \rho_2)} + \sqrt{\frac{k^2(\Delta u)^2\rho_1\rho_2}{(\rho_1+\rho_2)^2} - \frac{gk(\rho_2-\rho_1)}{\rho_1+\rho_2}}
```

## Running Validation Tests

### Kelvin–Helmholtz (KH) run

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
