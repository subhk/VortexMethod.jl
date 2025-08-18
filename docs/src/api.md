# API Reference

This page lists the main user-facing entry points. See docstrings and source for details.

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
- `poisson_velocity_fft(u_rhs,v_rhs,w_rhs, domain; mode=:spectral)`
- `poisson_velocity_fft_mpi(u_rhs,v_rhs,w_rhs, domain; mode=:spectral)`

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

- `remesh_pass!(nodeX,nodeY,nodeZ, tri, ds_max, ds_min; domain, ...)`
- Utilities: `detect_max_edge_length`, `detect_min_edge_length`

`VortexMethod.RemeshAdvanced`:

- `compute_mesh_quality(triXC,triYC,triZC, domain)` (periodic) and variants
- `flow_adaptive_remesh!(nodeX,nodeY,nodeZ, tri, velocity_field, domain; thresholds...)`
- `curvature_based_remesh!(...)`, `anisotropic_remesh!(...)`, periodic-aware

## Vortex sheets

`VortexMethod.VortexSheets`:

- Types: `LagrangianSheet`, `EulerianSheet`, `HybridSheet`
- Evolution: `evolve_sheet!(sheet, evolution, velocity_field, dt, domain)` with `Classical`, `Adaptive`, or `HighOrder` strategies
- Analysis: `compute_sheet_curvature(...)`, `detect_sheet_rollup(...)`
- Reconnection/smoothing: `check_sheet_reconnection!(..., domain)`, `reconnect_sheet_nodes!(..., domain)`, `smooth_local_curvature!(..., domain)`

## Checkpointing

`VortexMethod.Checkpoint`:

- Single-snapshot: `save_checkpoint!` (CSV), `save_checkpoint_mat!` (MAT), `save_checkpoint_jld2!` (JLD2)
- Time series (JLD2): `save_state_timeseries!`, `series_times`, `load_series_snapshot`, `load_series_nearest_time`
- Helpers: `mesh_stats(...)` with periodic overload when `domain` is provided
