# Codebase Structure

This page maps the public API to the source layout so implementation work can
start from the right module.

## Package entry point

`src/VortexMethod.jl` is the package entry point. It includes the implementation
modules, re-exports the main public names, and makes the high-level interface
available at the top level.

The most common user-facing layer is in `src/interface.jl`:

- `RectilinearGrid` wraps `DomainSpec` and `GridSpec`.
- `VortexSheetModel` owns node arrays, triangle connectivity, element
  circulation, solver options, and the simulation clock.
- `Simulation`, `time_step!`, and `run!` provide the high-level stepping loop.

The lower-level API remains available for direct array control and tests.

## Numerical pipeline

The solver is organized around the Lagrangian sheet to Eulerian grid pipeline:

| Stage | Source files | Role |
|:------|:-------------|:-----|
| Domain and mesh setup | `src/domain.jl`, `src/mesh.jl` | Domain lengths, grid shape, wrapping, and structured sheet construction |
| Geometry and circulation | `src/physics/circulation.jl`, `src/kernels/grid_transfer.jl` | Triangle centroids, areas, normals, element circulation, and circulation transport helpers |
| Kernel operations | `src/kernels/kernels.jl`, `src/kernels/grid_transfer.jl` | Regularized kernel spreading and grid-to-node interpolation |
| Velocity solve | `src/poisson/Poisson.jl`, `src/poisson/fft.jl`, `src/poisson/solvers.jl` | Curl RHS construction and periodic FFT, pencil FFT, or advanced Poisson solves |
| Time integration | `src/timestep.jl` | RK2 stepping, velocity reuse helpers, baroclinic forcing, and optional dissipation |
| Remeshing | `src/remeshing/Remeshing.jl`, `src/remeshing/basic.jl`, `src/remeshing/quality.jl` | Circulation-aware edge splitting, collapse, and flow-adaptive refinement |
| Physics extensions | `src/physics/dissipation.jl`, `src/sheets/sheets.jl`, `src/particle_management.jl` | SFS dissipation, sheet evolution utilities, reconnection, and smoothing |
| Diagnostics and I/O | `src/energy.jl`, `src/checkpoint.jl` | Energy diagnostics, mesh statistics, checkpoints, and time series output |
| Performance utilities | `src/diagnostics/performance.jl`, `src/diagnostics/fast_linalg.jl`, `src/core/layout.jl` | Allocation reduction, cache-friendly storage, and small linear algebra helpers |

## Tests and examples

`test/runtests.jl` is the suite entry point. The tests are grouped by solver
area, including:

- `test/test_interface.jl` for the high-level interface.
- `test/test_remesh.jl` and `test/test_stock_regressions.jl` for remeshing,
  circulation preservation, and periodic geometry regressions.
- `test/test_poisson.jl`, `test/test_parallel_fft.jl`, and related MPI tests
  for velocity solves.
- `test/test_performance.jl` for allocation-sensitive paths.

Examples live under `examples/`. `examples/simple3d.jl` uses the high-level
interface, while the KH examples show lower-level array workflows with remeshing,
checkpointing, and MPI-oriented runs.

## Where to add new code

- Add user-facing constructors, naming conveniences, or simulation-loop features
  in `src/interface.jl`, then export them from `src/VortexMethod.jl`.
- Add numerical kernels near the stage they affect instead of routing through
  the high-level interface.
- Keep topology-changing mesh changes in `src/remeshing/basic.jl` or
  `src/remeshing/quality.jl`, wired through `src/remeshing/Remeshing.jl`,
  and carry `eleGma` through the return values.
- Add focused tests beside the affected subsystem and update this docs page if
  the source layout or public workflow changes.
