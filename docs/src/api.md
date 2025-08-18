# API Reference

This page lists the main user-facing entry points. See docstrings and source for details.

## Domain and grid

- `DomainSpec(Lx,Ly,Lz)`, `GridSpec(nx,ny,nz)`
- `default_domain()`, `default_grid()`
- `grid_vectors(dom, gr)`, `grid_spacing(dom, gr)`, `grid_mesh(dom, gr)`
- `wrap_point(x,y,z, dom)`, `wrap_nodes!(nodeX,nodeY,nodeZ, dom)`

## Kernels

`VortexMethod.Kernels`:

- Types: `KernelType`, `PeskinStandard`, `PeskinCosine`, `M4Prime`, `AreaWeighting`
- Helpers: `kernel_function`, `kernel_support_radius`

## Spreading and interpolation (MPI)

`VortexMethod.Peskin3D`:

- `spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)`
- `interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, dom, gr)`
- Kernel variants: `spread_vorticity_to_grid_kernel_mpi`, `interpolate_node_velocity_kernel_mpi`

## Poisson solvers

`VortexMethod.Poisson3D`:

- `curl_rhs_centered(VorX,VorY,VorZ, dx,dy,dz)`
- `poisson_velocity_fft(u_rhs,v_rhs,w_rhs, dom; mode=:spectral)`
- `poisson_velocity_fft_mpi(u_rhs,v_rhs,w_rhs, dom; mode=:spectral)`

`VortexMethod.PoissonAdvanced`:

- Types: `FFTSolver`, `IterativeSolver`, `MultigridSolver`, `HybridSolver`
- BCs: `PeriodicBC`, `DirichletBC`, `NeumannBC`
- `solve_poisson!(solver, u_rhs,v_rhs,w_rhs, dom)`
- `solve_poisson_adaptive!(u_rhs,v_rhs,w_rhs, dom; bc=PeriodicBC())`

## Time stepping

`VortexMethod.TimeStepper`:

- `node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr; poisson_mode=:spectral)`
- `rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; kwargs...)`
- `rk2_step_with_dissipation!(...; dissipation_model=NoDissipation(), kwargs...)`
- Velocity helpers: `grid_velocity(...)`, `make_velocity_sampler(...)`

## Dissipation models

`VortexMethod.Dissipation`:

- Types: `DissipationModel`, `NoDissipation`, `SmagorinskyModel`, `DynamicSmagorinsky`, `VortexStretchingDissipation`, `MixedScaleModel`
- Functions: `apply_dissipation!(...)`, `compute_eddy_viscosity(...)`, `filter_width(...)`

## Remeshing

`VortexMethod.Remesh`:

- `remesh_pass!(nodeX,nodeY,nodeZ, tri, ds_max, ds_min; dom, ...)`
- Utilities: `detect_max_edge_length`, `detect_min_edge_length`

`VortexMethod.RemeshAdvanced`:

- `compute_mesh_quality(triXC,triYC,triZC, dom)` (periodic) and variants
- `flow_adaptive_remesh!(nodeX,nodeY,nodeZ, tri, velocity_field, dom; thresholds...)`
- `curvature_based_remesh!(...)`, `anisotropic_remesh!(...)`, periodic-aware

## Vortex sheets

`VortexMethod.VortexSheets`:

- Types: `LagrangianSheet`, `EulerianSheet`, `HybridSheet`
- Evolution: `evolve_sheet!(sheet, evolution, velocity_field, dt, dom)` with `Classical`, `Adaptive`, or `HighOrder` strategies
- Analysis: `compute_sheet_curvature(...)`, `detect_sheet_rollup(...)`
- Reconnection/smoothing: `check_sheet_reconnection!(..., dom)`, `reconnect_sheet_nodes!(..., dom)`, `smooth_local_curvature!(..., dom)`

## Checkpointing

`VortexMethod.Checkpoint`:

- Single-snapshot: `save_checkpoint!` (CSV), `save_checkpoint_mat!` (MAT), `save_checkpoint_jld2!` (JLD2)
- Time series (JLD2): `save_state_timeseries!`, `series_times`, `load_series_snapshot`, `load_series_nearest_time`
- Helpers: `mesh_stats(...)` with periodic overload when `dom` is provided

