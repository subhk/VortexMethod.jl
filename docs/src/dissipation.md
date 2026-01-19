# Sub-Filter Scale Dissipation

This page describes the sub-filter scale (SFS) dissipation models for Large Eddy Simulation (LES), based on [Stock (2006), §3.8](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837) and Mansfield et al. (1996).

## Motivation

In regularized vortex methods, the finite support of interpolation kernels acts as an implicit filter. Sub-filter scale motions are not resolved and their effects must be modeled to:

1. Provide energy dissipation at small scales
2. Prevent unphysical buildup of enstrophy
3. Enable physically meaningful turbulence statistics

## Smagorinsky-Equivalent Model

The filtered vorticity evolution equation with SFS dissipation:

```math
\frac{D\bar{\omega}_i}{Dt} = \bar{\boldsymbol{\omega}} \cdot \nabla\bar{u}_i + \nu\nabla^2\bar{\omega}_i + \nu_T\left(\frac{\partial\bar{\omega}_i}{\partial x_j\partial x_j} - \frac{\partial\bar{\omega}_j}{\partial x_i\partial x_j}\right)
```

### Eddy Viscosity

The turbulent (eddy) viscosity is computed from the filtered rate-of-strain:

```math
\nu_T = (c_T\delta)^2\sqrt{2\bar{S}_{ij}\bar{S}_{ij}}
```

where:
- ``c_T \approx 0.15`` is the model constant (similar to Smagorinsky constant)
- ``\delta`` is the filter width (typically the regularization length)
- ``\bar{S}_{ij}`` is the filtered rate-of-strain tensor

### Rate-of-Strain Tensor

```math
\bar{S}_{ij} = \frac{1}{2}\left(\frac{\partial\bar{u}_i}{\partial x_j} + \frac{\partial\bar{u}_j}{\partial x_i}\right)
```

The magnitude is:

```math
|\bar{S}| = \sqrt{2\bar{S}_{ij}\bar{S}_{ij}}
```

## Implementation

### Lagrangian-to-Eulerian-to-Lagrangian Approach

1. **Compute vorticity on grid** from Lagrangian elements (standard spreading)
2. **Compute velocity derivatives** on the grid using finite differences
3. **Compute eddy viscosity** at each grid point
4. **Compute vorticity diffusion** term on the grid
5. **Interpolate change** back to Lagrangian elements

This follows the grid-based approach from Mansfield et al. (1996), but interpolates back to existing elements instead of remeshing with replacement particles.

### Algorithm

```julia
function apply_sfs_dissipation!(eleGma, tri, nodeX, nodeY, nodeZ, domain, gr, dt;
                                 c_T=0.15, delta=nothing)
    # Default filter width = grid spacing
    delta = isnothing(delta) ? grid_spacing(domain, gr)[1] : delta

    # 1. Spread vorticity to grid
    VorX, VorY, VorZ = spread_vorticity_to_grid(eleGma, tri, nodeX, nodeY, nodeZ, domain, gr)

    # 2. Solve for velocity field
    Ux, Uy, Uz = poisson_velocity_fft(curl_rhs(VorX, VorY, VorZ, domain, gr)..., domain)

    # 3. Compute velocity gradients
    dudx, dudy, dudz = gradient(Ux, domain, gr)
    dvdx, dvdy, dvdz = gradient(Uy, domain, gr)
    dwdx, dwdy, dwdz = gradient(Uz, domain, gr)

    # 4. Rate-of-strain magnitude
    S_mag = compute_strain_magnitude(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)

    # 5. Eddy viscosity field
    nu_T = (c_T * delta)^2 .* S_mag

    # 6. Vorticity diffusion (Laplacian)
    dωdt_x = nu_T .* laplacian(VorX, domain, gr)
    dωdt_y = nu_T .* laplacian(VorY, domain, gr)
    dωdt_z = nu_T .* laplacian(VorZ, domain, gr)

    # 7. Interpolate back to elements and update
    update_element_vorticity!(eleGma, tri, nodeX, nodeY, nodeZ, dωdt_x, dωdt_y, dωdt_z, domain, gr, dt)
end
```

### Rate-of-Strain Computation

```julia
function compute_strain_magnitude(dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz)
    # Sij = 0.5 * (dui/dxj + duj/dxi)
    S11 = dudx
    S22 = dvdy
    S33 = dwdz
    S12 = 0.5 * (dudy + dvdx)
    S13 = 0.5 * (dudz + dwdx)
    S23 = 0.5 * (dvdz + dwdy)

    # |S| = sqrt(2 * Sij * Sij)
    return sqrt.(2.0 .* (S11.^2 + S22.^2 + S33.^2 + 2.0*(S12.^2 + S13.^2 + S23.^2)))
end
```

## Available Dissipation Models

The `VortexMethod.Dissipation` module provides several models:

### `NoDissipation` (Default)

No sub-filter dissipation. Suitable for:
- Validation against inviscid theory
- Short-duration simulations
- Very high resolution runs

### `SmagorinskyModel`

Standard Smagorinsky model with constant ``c_T``:

```julia
model = SmagorinskyModel(c_T=0.15)
rk2_step_with_dissipation!(nodes..., tri, eleGma, domain, gr, dt; dissipation_model=model)
```

### `DynamicSmagorinsky`

Germano-style dynamic procedure to compute ``c_T`` locally:

```julia
model = DynamicSmagorinsky(test_filter_ratio=2.0)
```

The dynamic model:
- Computes ``c_T`` from resolved scales
- Reduces dissipation in laminar regions
- More accurate for transitional flows

### `VortexStretchingDissipation`

Dissipation proportional to local vortex stretching:

```julia
model = VortexStretchingDissipation(c_stretch=0.1)
```

Targets regions with intense vortex stretching, which often indicates under-resolved dynamics.

### `MixedScaleModel`

Combines strain-based and vorticity-based indicators:

```julia
model = MixedScaleModel(c_strain=0.1, c_vorticity=0.1)
```

## Filter Width Selection

The filter width ``\delta`` should match the regularization scale:

| Kernel | Recommended ``\delta`` |
|--------|------------------------|
| Area-Weighting | ``\Delta x`` |
| M4' | ``2\Delta x`` |
| Peskin (ε=2) | ``2\Delta x`` |
| Peskin (ε=3) | ``3\Delta x`` |

Using ``\delta`` smaller than the kernel support can cause under-dissipation; larger values over-dissipate.

## Effect on Simulations

### Without SFS Dissipation

- Enstrophy can grow unboundedly
- Small-scale "noise" develops
- Energy piles up at grid scale

### With SFS Dissipation

- Controlled enstrophy decay
- Smooth vorticity fields
- Physical energy cascade

### Tuning ``c_T``

| Value | Effect |
|-------|--------|
| 0.1 | Light dissipation, preserves more small scales |
| 0.15 | Standard value from isotropic turbulence |
| 0.2 | Stronger dissipation, more stable but may be over-diffusive |

## Integration with Time Stepping

```julia
# Option 1: Use the combined function
rk2_step_with_dissipation!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt;
                           dissipation_model=SmagorinskyModel(0.15))

# Option 2: Apply separately
rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt)
apply_dissipation!(eleGma, tri, nodeX, nodeY, nodeZ, domain, gr, dt;
                   model=SmagorinskyModel(0.15))
```

## Validation

### Decaying Isotropic Turbulence

Compare energy decay rate ``E(t) \sim t^{-n}`` with experimental/DNS values:
- Without dissipation: ``n < 1`` (insufficient decay)
- With Smagorinsky: ``n \approx 1.2-1.4`` (reasonable)

### Kelvin–Helmholtz Roll-Up

Compare:
- Roll-up time
- Maximum vorticity
- Momentum thickness growth

Properly tuned dissipation should not significantly delay roll-up but should prevent spurious secondary instabilities.

## References

- Mansfield, J. R., Knio, O. M., & Meneveau, C. (1996). A dynamic LES scheme for the vorticity transport equation. Part I: Formulation. *Journal of Computational Physics*, 126(1), 59–76.
- Stock, M. J. (2006). *A regularized inviscid vortex sheet method for three dimensional flows with density interfaces*. Ph.D. Thesis, California Institute of Technology.
- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. *Monthly Weather Review*, 91(3), 99–164.
