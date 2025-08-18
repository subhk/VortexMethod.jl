# VortexMethod.jl

[![CI](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/VortexMethod.jl/dev/)
[![Coverage](https://codecov.io/gh/subhk/VortexMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/VortexMethod.jl)

A 3D Lagrangian vortex method with MPI parallelism, periodic domains, multiple interpolation kernels, advanced remeshing, dissipation models, and vortex-sheet utilities. The implementation mirrors a well-tested Python reference (see `python/`) and a corresponding thesis, with additional engineering for performance and reproducibility.

## Highlights

- Lagrangian–Eulerian pipeline: spread element vorticity → curl RHS → Poisson solve → interpolate → RK time stepping.
- Multiple kernels: Peskin-style, cosine, M4', area-weighted (configurable support radius).
- Poisson solvers: periodic FFT + “advanced” iterative/multigrid interfaces and a hybrid fallback.
- Remeshing: baseline edge split/flip/collapse and thesis-style flow/quality thresholds with periodic metrics.
- Dissipation: Smagorinsky, dynamic, vortex-stretching, and mixed-scale models.
- Vortex sheets: Lagrangian/Eulerian/Hybrid structures, curvature/reconnection utilities.
- Checkpointing: CSV/MAT/JLD2; time-series with random-access snapshots.
- MPI: parallel spreading/interpolation; global reductions for diagnostics.

## Install

```
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Run examples (MPI)

```
mpirun -n 4 julia --project examples/simple3d.jl
mpirun -n 4 julia --project examples/kh3d.jl
mpirun -n 4 julia --project examples/advanced_kh3d.jl
```

## Minimal usage

```julia
using VortexMethod

domain = default_domain(); gr = default_grid()

# Build a structured sheet and an initial element vorticity
Nx, Ny = 64, 64
nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
eleGma = zeros(Float64, size(tri,1), 3); eleGma[:,2] .= 1.0

# Single RK2 step (adaptive dt and periodic BCs handled internally)
dt = 1e-3
rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt; adaptive=true, CFL=0.5, poisson_mode=:fd)

# Advanced remeshing with cached velocity sampler
vel = make_velocity_sampler(eleGma, triXC, triYC, triZC, domain, gr)
tri, changed = VortexMethod.RemeshAdvanced.flow_adaptive_remesh!(
    nodeX, nodeY, nodeZ, tri, vel, domain;
    max_aspect_ratio=3.0, min_angle_quality=0.4, min_jacobian_quality=0.4,
    max_skewness=0.8, grad_threshold=0.2, curvature_threshold=0.6,
)

# Always keep particles periodic after manual edits
wrap_nodes!(nodeX, nodeY, nodeZ, domain)
```

## Documentation

- Dev docs: https://subhk.github.io/VortexMethod.jl/dev/ (or build locally):

```
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

Docs include: theory, remeshing criteria, parallelization model, usage recipes, and a full API reference.

## Validation

- Generate a KH time series: `mpirun -n 4 julia --project examples/advanced_kh3d.jl`
- Plot KE into docs assets:

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
OUTPUT_PNG=docs/src/assets/ke_series.png \
julia --project examples/plot_series_ke.jl
```

- Snapshot of |γ|:

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
SNAP_INDEX=10 \
OUTPUT_PNG=docs/src/assets/snapshot_gamma.png \
julia --project examples/plot_snapshot_gamma.jl
```

## Notes

- Periodic BCs in all directions. Spreading/interpolation use compact kernels over sub-triangle quadrature; FFT Poisson uses full 3D transforms with k=0 handling.
- MPI: strided distribution (grid/nodes) + `Allreduce`; FFT currently on rank 0 with broadcast. Advanced solvers available for large cases.
- See `python/` for the original reference implementation.

## License

This project is licensed under the MIT License. See `LICENSE`.
