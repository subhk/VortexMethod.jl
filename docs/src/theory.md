# Theory Overview

This package implements a 3D Lagrangian vortex sheet method based on the numerical methods developed in [Stock (2006)](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837). The core ideas are:

- Vorticity is carried on a moving, triangulated sheet (Lagrangian elements). Each triangular element stores scalar-valued circulations on its edges, which can be converted to vortex sheet strength or total vorticity.
- To compute velocities, element vorticity is spread to a regular 3D grid using a smooth, compactly-supported kernel. The resulting grid vorticity field is converted to a velocity field by solving a Poisson equation via FFTs.
- Grid velocities are interpolated back to the Lagrangian nodes. Time integration moves nodes forward and updates element strengths consistent with conserved circulation and optional baroclinic sources or subfilter dissipation.

---

## Governing Equations

!!! info "Key Physical Insight"
    Unlike standard vorticity methods, the vortex sheet formulation includes an **in-sheet dilatation term** that accounts for area changes of the sheet. This allows circulation (not just vorticity) to be the primary conserved quantity.

### Vortex Sheet Strength Evolution

The fundamental equation governing the dynamic evolution of vortex sheet strength is:

```math
\frac{D\boldsymbol{\gamma}}{Dt} = \underbrace{\boldsymbol{\gamma} \cdot \nabla\mathbf{u}}_{\text{stretch}} - \underbrace{\boldsymbol{\gamma}(\mathbf{P} \cdot \nabla \cdot \mathbf{u})}_{\text{dilatation}} + \underbrace{2A\,\hat{\mathbf{n}} \times (\bar{\mathbf{a}} - \mathbf{g})}_{\text{baroclinic}}
```

where:
- ``\boldsymbol{\gamma}`` is the vortex sheet strength
- ``\mathbf{P} = \mathbf{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}^T`` is the tangential projection operator
- ``A`` is the Atwood number (density ratio)
- ``\hat{\mathbf{n}}`` is the sheet normal
- ``\bar{\mathbf{a}}`` is the average acceleration, ``\mathbf{g}`` is gravitational acceleration

The dilatation term distinguishes the vortex sheet strength equation from the standard vorticity evolution equation.

### Circulation Evolution

The equivalent equation for circulation (which notably contains no vortex stretching term):

```math
\frac{D\Gamma}{Dt} = 2A\left(\frac{D\phi}{Dt} - \frac{1}{2}|\mathbf{u}|^2 + \frac{1}{8}|\Delta\mathbf{u}|^2 - \mathbf{g}\cdot\mathbf{x}\right)
```

!!! tip "Conservation Property"
    In **homogeneous** (``A = 0``), **inviscid** flow, circulation is exactly conserved: ``D\Gamma/Dt = 0``. This is the fundamental reason for using circulation-based discretization.

### Kinematic Velocity Equation

The velocity field is related to the vorticity field through:

```math
\nabla^2\mathbf{u} = -\nabla \times \boldsymbol{\omega}
```

This vector Poisson equation is solved on a regular grid using FFT-based spectral methods.

---

## Element Discretization

!!! note "Why Triangles?"
    Triangular elements provide a flexible, unstructured representation of arbitrarily-shaped vortex sheets. Unlike vortex filaments or particles, triangles can naturally represent **both** stretching along vorticity **and** transverse strain.

### Triangular Elements with Edge Circulations

The vortex sheet is discretized into flat triangular elements, each defined by connectivity to three Lagrangian nodes. Each triangular element ``p`` stores scalar-valued circulations ``\Gamma_{p,1..3}`` on its edges.

The key relationship between different vorticity representations:

```math
\boldsymbol{\alpha}_p = v_p\,\boldsymbol{\omega}_p = a_p\,\boldsymbol{\gamma}_p = \sum_{i=1}^{3} \Gamma_{p,i}\,\Delta\mathbf{l}_{p,i}
```

where:
- ``v_p`` is element volume (with regularization thickness)
- ``a_p`` is element area
- ``\boldsymbol{\omega}_p`` is vorticity
- ``\boldsymbol{\gamma}_p`` is vortex sheet strength
- ``\Delta\mathbf{l}_{p,i}`` are the edge vectors

### Circulation Assignment

When setting edge circulations from a given total vorticity, the following overconstrained system is solved:

```math
\begin{bmatrix}
\Delta l_{p,1,x} & \Delta l_{p,2,x} & \Delta l_{p,3,x} \\
\Delta l_{p,1,y} & \Delta l_{p,2,y} & \Delta l_{p,3,y} \\
\Delta l_{p,1,z} & \Delta l_{p,2,z} & \Delta l_{p,3,z} \\
1 & 1 & 1
\end{bmatrix}
\begin{pmatrix}
\Gamma_{p,1} \\
\Gamma_{p,2} \\
\Gamma_{p,3}
\end{pmatrix}
=
\begin{pmatrix}
\alpha_{p,x} \\
\alpha_{p,y} \\
\alpha_{p,z} \\
0
\end{pmatrix}
```

When the total vorticity is planar to the triangular element, the same total vorticity can be recovered. If not planar, the vorticity is reoriented to lie in the element plane.

!!! warning "Non-Planar Vorticity"
    When the desired vorticity is not in the plane of the triangle, the solver will project it onto the element plane. This is physically correct for a thin vortex sheet but means that out-of-plane vorticity components cannot be exactly represented on a single element.

---

## Mathematical Pipeline

The complete algorithm consists of five stages executed every time step:

### 1. Spreading (Lagrangian → Eulerian)

The vorticity field is created by interpolating element strengths onto a regular grid:

```math
\boldsymbol{\omega}(\mathbf{x}) = \frac{1}{\Delta x^3} \sum_{i=1}^{N} \frac{a_i\,\boldsymbol{\gamma}_i}{M} \sum_{j=1}^{M} \tilde{\delta}\left(\frac{\mathbf{x} - \mathbf{x}_{i,j}}{\Delta x}\right)
```

where:
- ``N`` is the number of triangular elements
- ``M`` is the number of sub-triangle partitions (for smooth spreading)
- ``\tilde{\delta}`` is the particle-grid operator (interpolation kernel)
- ``\mathbf{x}_{i,j}`` is the center of sub-triangle ``j`` of element ``i``

Each triangle is subdivided into ``M`` equal-area sub-elements to ensure smoothness when elements span multiple grid cells.

### 2. Curl RHS on the Grid

Given vorticity ``\boldsymbol{\omega} = (\omega_x, \omega_y, \omega_z)`` on the grid, the RHS of the vector Poisson equation is ``-\nabla\times\boldsymbol{\omega}``, computed using second-order centered finite differences.

### 3. Poisson Solve

Spectral inversion in Fourier space with full 3D FFT. The zero-mean constraint is enforced by setting the ``k=0`` mode to zero.

### 4. Interpolation (Eulerian → Lagrangian)

The same family of kernels interpolates grid velocity back to node positions with periodic tiling.

### 5. Time Stepping

Two-stage RK2 with optional dissipation models. Adaptive ``\Delta t`` via CFL-like condition using maximum grid speed. Baroclinic sources update element vorticity between stages.

---

## Interpolation Kernels

!!! info "Role of Kernels"
    The interpolation kernel serves as a **regularization** of the singular Biot-Savart integral. Larger kernel support produces smoother fields but at higher computational cost. The kernel also acts as an implicit **low-pass filter** for LES-style simulations.

The kernel module (`VortexMethod.Kernels`) provides interchangeable, compactly supported weighting functions. Most 3D rectangular interpolation methods are tensor products of 1D functions:

```math
\tilde{\delta}\left(\frac{\mathbf{x} - \mathbf{x}(s)}{\Delta x}\right) = \delta\left(\frac{x - x(s)}{\Delta x}\right) \delta\left(\frac{y - y(s)}{\Delta y}\right) \delta\left(\frac{z - z(s)}{\Delta z}\right)
```

### Area-Weighting (Cloud-In-Cell)

The simplest method, originating from the convolution of two top-hat functions:

```math
\delta(x) = \begin{cases}
1 - |x| & : |x| \leq 1 \\
0 & : |x| > 1
\end{cases}
```

- **Support:** 2 grid points per dimension (8 evaluations total)
- **Accuracy:** First-order
- **Usage:** `AreaWeighting(delr=1.0)`

### M4' Kernel

A B-spline-like kernel from Monaghan (1985):

```math
\delta(x) = \begin{cases}
0 & : |x| > 2 \\
\frac{1}{2}(2-|x|)^2(1-|x|) & : 1 \leq |x| \leq 2 \\
1 - \frac{5x^2}{2} + \frac{3|x|^3}{2} & : |x| \leq 1
\end{cases}
```

- **Support:** 4 grid points per dimension (64 evaluations total)
- **Accuracy:** Third-order
- **Usage:** `M4Prime(delr=2.0)`

### Peskin Function

A smoothing kernel with variable radius ``\varepsilon``:

```math
\delta(x) = \begin{cases}
\frac{1}{2\varepsilon}\left[1 + \cos\left(\frac{\pi x}{\varepsilon}\right)\right] & : |x| \leq \varepsilon \\
0 & : |x| > \varepsilon
\end{cases}
```

- **Support:** Tunable via ``\varepsilon`` (typically 2–4 grid spacings)
- **Accuracy:** First-order, but smoother than area-weighting
- **Usage:** `PeskinStandard(delr=4.0)`, `PeskinCosine(delr=3.0)`

### Kernel Selection Guidelines

| Kernel | Support | Accuracy | Speed | Smoothness | Best For |
|--------|---------|----------|-------|------------|----------|
| Area-Weighting | 2Δx | 1st | Fastest | Low | Quick tests |
| M4' | 4Δx | 3rd | Moderate | Good | Production runs |
| Peskin (ε=2) | 4Δx | 1st | Moderate | Excellent | Sensitive flows |
| Peskin (ε=3) | 6Δx | 1st | Slower | Best | High-quality visualization |

The M4' kernel provides the best accuracy-to-cost ratio for most simulations. Use larger Peskin radii when smooth vorticity fields are essential.

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

_Figure 1.23: High-level overview of the Lagrangian–Eulerian vortex method pipeline (spreading → curl RHS → Poisson → interpolation → time stepping)._ 

- Figure 3.19 (supporting theory/parallel pipeline; suggested filename: `fig_3_19.png`):

![](assets/fig_3_19.png)

_Figure 3.19: Parallelization sketch showing MPI distribution for spreading/interpolation and centralized Poisson solve/broadcast._

Add short captions below each image once extracted.
