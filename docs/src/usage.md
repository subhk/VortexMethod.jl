# Usage Guide

## Installation

```
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Running examples with MPI

```
mpirun -n 4 julia --project examples/simple3d.jl
mpirun -n 4 julia --project examples/kh3d.jl
mpirun -n 4 julia --project examples/advanced_kh3d.jl
```

## Core API

- Domain/grid: `DomainSpec(Lx,Ly,Lz)`, `GridSpec(nx,ny,nz)`, `default_domain()`, `default_grid()`
- Periodic helpers: `wrap_point(x,y,z, dom)`, `wrap_nodes!(nodeX,nodeY,nodeZ, dom)`
- Spreading/interpolation (MPI): `spread_vorticity_to_grid_mpi`, `interpolate_node_velocity_mpi`
- Poisson: `poisson_velocity_fft` / `_mpi` (periodic), or see `PoissonAdvanced` for alternatives
- Time stepping: `rk2_step!`, `rk2_step_with_dissipation!`; helper `node_velocities`
- Velocity reuse: `grid_velocity` to compute `(Ux,Uy,Uz)` once; `make_velocity_sampler` to build `(x,y,z)->(u,v,w)` closures
- Remeshing: `VortexMethod.Remesh.*` and `VortexMethod.RemeshAdvanced.*`
- Vortex sheets: see `VortexMethod.VortexSheets`
- Checkpoints: `save_checkpoint!`, `save_state!`, `save_state_timeseries!`

## Minimal workflow

1. Build or load a triangulated sheet (node arrays + `tri` connectivity). Initialize `eleGma` (nt×3).
2. Compute node velocities and advance with `rk2_step!` (or the variant with dissipation).
3. Periodically remesh using the baseline or advanced methods. Preserve circulation via `node_circulation_from_ele_gamma` and `ele_gamma_from_node_circ`.
4. Save checkpoints regularly (rank 0) via `save_state!` or `save_state_timeseries!`.

## Advanced remeshing recipe

```
Ux,Uy,Uz = grid_velocity(eleGma, triXC, triYC, triZC, dom, gr)
vel = make_velocity_sampler(eleGma, triXC, triYC, triZC, dom, gr)
tri_new, changed = VortexMethod.RemeshAdvanced.flow_adaptive_remesh!(
    nodeX, nodeY, nodeZ, tri, vel, dom;
    max_aspect_ratio=3.0, min_angle_quality=0.4, min_jacobian_quality=0.4,
    max_skewness=0.8, grad_threshold=0.2, curvature_threshold=0.6,
)
```

## Periodicity and particle management

- After inserting, deleting, or moving nodes manually, call `wrap_nodes!` to keep positions in-range.
- Use `wrap_point` for single updates (e.g., after reconnection).

## Figures from the thesis

You can place figures into `docs/src/assets/` and reference them in Markdown with `![](assets/...)`. If you provide figure numbers/pages, we can add captions and cross-references.

