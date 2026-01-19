```@meta
CurrentModule = VortexMethod
```

# VortexMethod.jl

A Julia implementation of a 3D Lagrangian vortex sheet method for simulating inviscid, incompressible flows with density interfaces.

## Overview

This package implements the regularized vortex sheet method developed in [Stock (2006)](https://resolver.caltech.edu/CaltechETD:etd-05312006-165837). Key features:

- **Lagrangian vortex sheets:** Vorticity carried on triangulated surfaces with edge-based circulation discretization
- **Vortex-in-Cell (VIC):** Fast O(N log N) velocity computation via FFT-based Poisson solvers
- **Adaptive remeshing:** Edge splitting and node merging to maintain mesh quality
- **Baroclinic effects:** Vorticity generation at density interfaces (Rayleigh–Taylor, Richtmyer–Meshkov)
- **Sub-filter dissipation:** LES-style Smagorinsky models for turbulent flows
- **MPI parallelization:** Scalable spreading, interpolation, and communication

## Documentation Structure

| Page | Description |
|------|-------------|
| [Theory](theory.md) | Governing equations, element discretization, interpolation kernels |
| [Boundary Conditions](boundary_conditions.md) | Periodic, open, and wall boundary treatments |
| [Baroclinic Effects](baroclinic.md) | Density discontinuities and vorticity generation |
| [Dissipation Models](dissipation.md) | Sub-filter scale dissipation for LES |
| [Remeshing](remeshing.md) | Edge splitting, node merging, quality metrics |
| [Parallelization](parallelization.md) | MPI implementation details |
| [Validation](validation.md) | Test cases and comparison with theory |
| [Usage](usage.md) | Getting started and workflow examples |
| [API](api.md) | Function reference |

## Quick Start

!!! tip "Getting Started"
    Make sure to run `julia --project -e 'using Pkg; Pkg.instantiate()'` before your first use.

```julia
using VortexMethod

# Define domain and grid
domain = DomainSpec(1.0, 1.0, 4.0)  # Lx, Ly, Lz
gr = GridSpec(64, 64, 256)          # nx, ny, nz

# Initialize sheet (nodes and triangles)
nodeX, nodeY, nodeZ, tri = create_initial_sheet(...)
eleGma = initialize_vortex_strength(tri, ...)

# Time stepping with MPI
dt = 0.01
for step in 1:nsteps
    rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt)
    remesh_pass!(nodeX, nodeY, nodeZ, tri, ds_max, ds_min; domain)
    wrap_nodes!(nodeX, nodeY, nodeZ, domain)
end
```

!!! note "MPI Support"
    For parallel execution, launch with `mpirun -n <nprocs> julia --project your_script.jl`

## Key References

- Stock, M. J. (2006). *A regularized inviscid vortex sheet method for three dimensional flows with density interfaces*. Ph.D. Thesis, California Institute of Technology.
- Cottet, G.-H., & Koumoutsakos, P. D. (2000). *Vortex Methods: Theory and Practice*. Cambridge University Press.

```@index
```

```@autodocs
Modules = [VortexMethod]
```

## Notes on Remeshing and Metrics

- Periodic metrics: Advanced remeshing and sheet quality now use periodic, minimum-image geometry in x/y and symmetric in z ([-Lz,Lz]).
- Thresholds (defaults):
  - max_aspect_ratio=3.0, max_skewness=0.8
  - min_angle_quality=0.4, min_jacobian_quality=0.4
  - grad_threshold=0.2 (Frobenius norm of ∇U), curvature_threshold=0.6 rad
- Kernel-based interpolation options are available via `Kernels`.
- New helper: `grid_velocity(eleGma, triXC, triYC, triZC, domain, gr)` computes `(Ux,Uy,Uz)` for reuse.

### Tuning Thresholds

- Coarser meshes typically require looser thresholds (e.g., AR≤4.0) to avoid over-refinement; finer meshes can tolerate tighter thresholds (e.g., AR≤3.0).
- Set `grad_threshold` proportional to expected shear; use a smaller value for gentle flows and larger for strongly sheared layers.
- `curvature_threshold` is a dihedral angle; values around 0.5–0.8 rad (≈29–46°) work well for sheet roll-up detection.
- You can pass these per-call via `flow_adaptive_remesh!` or rely on the defaults.

### Velocity Reuse

- Use `make_velocity_sampler(eleGma, triXC, triYC, triZC, domain, gr)` to build a closure `(x,y,z)->(u,v,w)` backed by a single spread/Poisson solve.
- This reduces repeated Poisson solves during remeshing and sheet tracking in the same time step.
