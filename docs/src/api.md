# API Reference

This page lists the main user-facing entry points. See docstrings and source for details.

## High-level simulation interface

- Topologies: `Periodic`, `Bounded`, `Flat`
- `RectilinearGrid(; size, x=(0,1), y=(0,1), z=(-1,1), extent=nothing, topology=(Periodic, Periodic, Periodic))`
  currently validates periodic topology for the FFT-based solver path.
- `VortexSheetModel(; grid, sheet_size=(16,16), circulation=(0,1,0), amp=1e-2, kwargs...)`
- `set!(model; circulation=...)` or `set!(model; Γ=...)`
- `time_step!(model, Δt; kwargs...)`
- `Simulation(model; Δt, stop_iteration=nothing, stop_time=Inf)`
- `run!(simulation)`

## Domain and grid

- `DomainSpec(Lx,Ly,Lz)`, `GridSpec(nx,ny,nz)`
- `default_domain()`, `default_grid()`
- `grid_vectors(domain, gr)`, `grid_spacing(domain, gr)`, `grid_mesh(domain, gr)`
- `wrap_point(x,y,z, domain)`, `wrap_nodes!(nodeX,nodeY,nodeZ, domain)`

## Kernels

`VortexMethod.Kernels`:

- Types: `KernelType`, `PeskinStandard`, `PeskinCosine`, `M4Prime`, `AreaWeighting`
- Helpers: `kernel_function`, `kernel_support_radius`

## Spreading and interpolation (MPI)

`VortexMethod.Peskin3D`:

- `spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)`
- `interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)`
- Kernel variants: `spread_vorticity_to_grid_kernel_mpi`, `interpolate_node_velocity_kernel_mpi`

## Poisson solvers

`VortexMethod.Poisson3D`:

- `curl_rhs_centered(VorX,VorY,VorZ, dx,dy,dz)`
- `curl_rhs_centered!(workspace, u_rhs,v_rhs,w_rhs, VorX,VorY,VorZ, dx,dy,dz)`
- `poisson_velocity_fft(u_rhs,v_rhs,w_rhs, domain; mode=:spectral)`
- `poisson_velocity_fft_mpi(u_rhs,v_rhs,w_rhs, domain; mode=:spectral)`
- `poisson_velocity_pencil_fft(u_rhs,v_rhs,w_rhs, domain; mode=:spectral)`

`VortexMethod.PoissonAdvanced`:

- Types: `FFTSolver`, `IterativeSolver`, `MultigridSolver`, `HybridSolver`
- BCs: `PeriodicBC`, `DirichletBC`, `NeumannBC`
- `solve_poisson!(solver, u_rhs, v_rhs, w_rhs, domain)`
- `solve_poisson_adaptive!(u_rhs, v_rhs, w_rhs, domain; bc=PeriodicBC())`

## Time stepping

`VortexMethod.TimeStepper`:

- `node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, domain, gr; poisson_mode=:spectral)`
- `rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt; kwargs...)`
- `rk2_step_with_dissipation!(...; dissipation_model=NoDissipation(), kwargs...)`
- Velocity helpers: `grid_velocity(...)`, `make_velocity_sampler(...)`

## Dissipation models

`VortexMethod.Dissipation`:

- Types: `DissipationModel`, `NoDissipation`, `SmagorinskyModel`, `DynamicSmagorinsky`, `VortexStretchingDissipation`, `MixedScaleModel`
- Functions: `apply_dissipation!(...)`, `compute_eddy_viscosity(...)`, `filter_width(...)`

## Remeshing

`VortexMethod.Remesh`:

- `remesh_pass!(nodeX,nodeY,nodeZ, tri, eleGma, ds_max, ds_min; domain, ...)`
  returns `(tri_new, eleGma_new, changed)`
- Utilities: `detect_max_edge_length`, `detect_min_edge_length`

`VortexMethod.RemeshAdvanced`:

- `compute_mesh_quality(triXC,triYC,triZC, domain)` (periodic) and variants
- `flow_adaptive_remesh!(nodeX,nodeY,nodeZ, tri, eleGma, velocity_field, domain; thresholds...)`
  returns `(tri_new, eleGma_new, changed)`
- `curvature_based_remesh!(nodeX,nodeY,nodeZ, tri, eleGma, domain; thresholds...)`
  returns `(tri_new, eleGma_new, changed)`

## Vortex sheets

`VortexMethod.VortexSheets`:

- Types: `LagrangianSheet`, `EulerianSheet`, `HybridSheet`
- Evolution: `evolve_sheet!(sheet, evolution, velocity_field, dt, domain)` with `Classical`, `Adaptive`, or `HighOrder` strategies
- Analysis: `compute_sheet_curvature(...)`, `detect_sheet_rollup(...)`
- Reconnection/smoothing: `check_sheet_reconnection!(..., domain)`, `reconnect_sheet_nodes!(..., domain)`, `smooth_local_curvature!(..., domain)`

## Checkpointing

`VortexMethod.Checkpoint`:

- Single-snapshot: `save_checkpoint!`, `save_checkpoint_jld2!`
- Time series (JLD2): `save_state_timeseries!`, `series_times`, `load_series_snapshot`, `load_series_nearest_time`
- Helpers: `mesh_stats(...)` with periodic overload when `domain` is provided
