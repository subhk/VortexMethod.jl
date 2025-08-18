# Theory Overview

This package implements a 3D Lagrangian vortex method with periodic boundaries. The core ideas are:

- Vorticity is carried on a moving, triangulated sheet (Lagrangian elements). Each triangle has a vector strength `γ` (element vorticity) that induces a velocity field.
- To compute velocities, element vorticity is spread to a regular 3D grid using a smooth, compactly-supported kernel (Peskin-style discrete delta). The resulting grid vorticity field is converted to a velocity field by solving a Poisson equation for the vector potential via FFTs, after assembling the curl RHS.
- Grid velocities are interpolated back to the Lagrangian nodes. Time integration moves nodes forward and updates element strengths consistent with conserved circulation and optional baroclinic sources or subfilter dissipation.

## Mathematical pipeline

1. Spreading (Lagrangian → Eulerian):
   - Each triangle is subdivided (4 sub-triangles) to reduce quadrature error.
   - The kernel distributes each element’s vorticity to nearby grid cells within `delr` cells in each direction.
   - Periodicity is enforced by tiling in ±1 in x and y (and periodic wrap in z).

2. Curl RHS on the grid:
   - Given vorticity `ω = (ωx,ωy,ωz)` on the grid, the RHS of the vector Poisson equation is `-curl(ω)`.
   - We use centered finite differences (4th-order in the interior, downwinded near boundaries with periodic wrap).

3. Poisson solve (periodic):
   - Spectral inversion in Fourier space with full 3D `FFTW.fft` (or a discrete Laplacian symbol when `mode=:fd`).
   - Zero-mean constraint by setting the k=0 mode to zero.

4. Interpolation (Eulerian → Lagrangian):
   - Use the same family of kernels to interpolate grid velocity to node positions (with periodic tiling in x,y).

5. Time stepping:
   - Two-stage RK2 is provided (with optional dissipation models). Adaptive `dt` via a CFL-like condition using the maximum grid speed.
   - Baroclinic source can update element vorticity (Atwood–gravity term) between stages.

## Kernels

The kernel module (`VortexMethod.Kernels`) provides interchangeable, compactly supported weighting functions:

- `PeskinStandard(delr=4.0)`: cosine-based 1D weights multiplied in x/y/z.
- `PeskinCosine(delr)`: smoother cosine taper.
- `M4Prime(delr=2.0)`: B-spline-like compact kernel (4th-order characteristics).
- `AreaWeighting(delr)`: simple hat-function scaling with element area.

The support radius `delr` defines the number of grid spacings over which the kernel has support.

## Remeshing notions (see “Remeshing” page for details)

- Lagrangian mesh quality degrades over time as nodes move and the sheet rolls up. Two complementary paths are implemented:
  1) Baseline length/flip/collapse pass (simple and fast), and
  2) Advanced quality- and flow-aware refinement.
- The advanced method uses periodic, minimum-image metrics to evaluate shape quality and local curvature; it also uses the Frobenius norm of the velocity gradient from a cached grid velocity to detect under-resolved regions.

## Figures from the thesis

Place figures (e.g., kernel shapes, flow schematics, sheet roll-up) from the thesis PDF (`mstock_dissertation.pdf`) into `docs/src/assets/`.

Thesis figures to include here:

- Figure 1.23 (theory overview; suggested filename: `fig_1_23.png`):

![](assets/fig_1_23.png)

- Figure 3.19 (supporting theory/parallel pipeline; suggested filename: `fig_3_19.png`):

![](assets/fig_3_19.png)

Add short captions below each image once extracted.
