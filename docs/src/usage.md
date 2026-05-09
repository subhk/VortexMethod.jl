# Usage Guide

## Installation

!!! tip "First-Time Setup"
    Make sure to instantiate the project before your first use to download all dependencies.

```
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Running examples

```
julia --project examples/simple3d.jl
mpirun -n 4 julia --project examples/kh3d.jl
mpirun -n 4 julia --project examples/advanced_kh3d.jl
```

## Core API

- High-level interface: `RectilinearGrid`, `VortexSheetModel`, `Simulation`, `set!`, `time_step!`, `run!`
- Domain/grid: `DomainSpec(Lx,Ly,Lz)`, `GridSpec(nx,ny,nz)`, `default_domain()`, `default_grid()`
- Periodic helpers: `wrap_point(x,y,z, domain)`, `wrap_nodes!(nodeX,nodeY,nodeZ, domain)`
- Spreading/interpolation (MPI): `spread_vorticity_to_grid_mpi`, `interpolate_node_velocity_mpi`
- Poisson: `poisson_velocity_fft` / `_mpi` (periodic), or adaptive Poisson solvers for alternatives
- Time stepping: `rk2_step!`, `rk2_step_with_dissipation!`; helper `node_velocities`
- Velocity reuse: `grid_velocity` to compute `(Ux,Uy,Uz)` once; `make_velocity_sampler` to build `(x,y,z)->(u,v,w)` closures
- Remeshing: `VortexMethod.Remeshing.*`
- Vortex sheets: see `VortexMethod.Sheets`
- Checkpoints: `save_checkpoint!`, `save_state!`, `save_state_timeseries!`

## Minimal workflow

!!! info "Basic Simulation Loop"
    A typical simulation involves: initialize mesh -> time step -> remesh -> checkpoint -> repeat.

For the high-level interface:

```julia
using VortexMethod

grid = RectilinearGrid(size=(32, 32, 63), x=(0, 1), y=(0, 1), z=(-1, 1))
model = VortexSheetModel(; grid, sheet_size=(32, 32), Γ=(0, 1, 0))
simulation = Simulation(model; Δt=1e-3, stop_iteration=10)
run!(simulation)
```

For direct array control:

1. Build or load a triangulated sheet with node arrays, `tri` connectivity, and
   `eleGma` element circulation with shape `nt x 3`.
2. Compute node velocities and advance with `rk2_step!` or
   `rk2_step_with_dissipation!`.
3. Periodically remesh using the baseline or advanced methods. These functions
   accept `eleGma` and return updated connectivity plus updated circulation.
4. Save checkpoints regularly via `save_state!` or `save_state_timeseries!`.

## Advanced remeshing recipe

```
vel = make_velocity_sampler(eleGma, triXC, triYC, triZC, domain, gr)
tri_new, eleGma_new, changed = VortexMethod.Remeshing.flow_adaptive_remesh!(
    nodeX, nodeY, nodeZ, tri, eleGma, vel, domain;
    max_aspect_ratio=3.0, min_angle_quality=0.4, min_jacobian_quality=0.4,
    max_skewness=0.8, grad_threshold=0.2, curvature_threshold=0.6,
)
```

## Periodicity and particle management

!!! warning "Always Wrap Nodes"
    After inserting, deleting, or moving nodes manually, always call `wrap_nodes!` to ensure positions remain within the periodic domain.

- After inserting, deleting, or moving nodes manually, call `wrap_nodes!` to keep positions in-range.
- Use `wrap_point` for single updates (e.g., after reconnection).

## Figures from the thesis

You can place figures into `docs/src/assets/` and reference them in Markdown with `![](assets/...)`. If you provide figure numbers/pages, we can add captions and cross-references.
