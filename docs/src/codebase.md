# Codebase Structure

This page maps the public API to the source layout so implementation work can
start from the right module.

## Package entry point

`src/VortexMethod.jl` is the package entry point. It includes the implementation
modules, re-exports the main public names, and makes the high-level interface
available at the top level.

The most common user-facing layer is in `src/integration/interface.jl`:

- `RectilinearGrid` wraps `DomainSpec` and `GridSpec`.
- `VortexSheetModel` owns node arrays, triangle connectivity, element
  circulation, solver options, and the simulation clock.
- `Simulation`, `time_step!`, and `run!` provide the high-level stepping loop.

The lower-level API remains available for direct array control and tests.

## Numerical pipeline

The solver is organized around the Lagrangian sheet to Eulerian grid pipeline:

| Stage | Source files | Role |
|:------|:-------------|:-----|
| Domain and mesh setup | `src/core/domain.jl`, `src/core/mesh.jl` | Domain lengths, grid shape, wrapping, and structured sheet construction |
| Geometry and circulation | `src/physics/circulation.jl`, `src/kernels/grid_transfer.jl` | Triangle centroids, areas, normals, element circulation, and circulation transport helpers |
| Kernel operations | `src/kernels/kernels.jl`, `src/kernels/grid_transfer.jl` | Regularized kernel spreading and grid-to-node interpolation |
| Velocity solve | `src/poisson/Poisson.jl`, `src/poisson/fft.jl`, `src/poisson/solvers.jl` | Curl RHS construction and periodic FFT, pencil FFT, or adaptive Poisson solves |
| Time integration | `src/integration/timestep.jl` | RK2 stepping, velocity reuse helpers, baroclinic forcing, and optional dissipation |
| Remeshing | `src/remeshing/Remeshing.jl`, `src/remeshing/basic.jl`, `src/remeshing/quality.jl` | Circulation-aware edge splitting, collapse, and flow-adaptive refinement |
| Physics extensions | `src/physics/dissipation.jl`, `src/sheets/sheets.jl`, `src/particle_management.jl` | SFS dissipation, sheet evolution utilities, reconnection, and smoothing |
| Diagnostics and I/O | `src/physics/energy.jl`, `src/io/checkpoint.jl` | Energy diagnostics, mesh statistics, checkpoints, and time series output |
| Performance utilities | `src/diagnostics/performance.jl`, `src/diagnostics/fast_linalg.jl`, `src/core/layout.jl` | Allocation reduction, cache-friendly storage, and small linear algebra helpers |

## Subsystem folders and namespaces

Implementation code is grouped by subsystem folder. Most subsystem internals
remain available through qualified names, while the most common functions are
also re-exported from `VortexMethod`.

| Folder | Main namespace | Contains |
|:-------|:---------------|:---------|
| `src/core/` | `VortexMethod.DomainImpl`, `VortexMethod.Mesh`, `VortexMethod.Layout`, `VortexMethod.Workspace` | Domain/grid types, periodic wrapping, structured mesh helpers, layout storage, and reusable workspaces |
| `src/kernels/` | `VortexMethod.Kernels`, `VortexMethod.GridTransfer` | Regularized kernels, spreading to grids, interpolation back to nodes, and workspace-backed grid transfer |
| `src/poisson/` | `VortexMethod.Poisson` | FFT, pencil FFT, iterative, multigrid, hybrid, and adaptive Poisson velocity solves |
| `src/integration/` | `VortexMethod.TimeStepper`, `VortexMethod.UserInterface` | RK2 stepping, velocity sampling, high-level model construction, and simulation loops |
| `src/remeshing/` | `VortexMethod.Remeshing` | Baseline edge split/collapse remeshing, mesh quality metrics, and flow-adaptive refinement |
| `src/sheets/` | `VortexMethod.Sheets` | Sheet model types, sheet evolution, rollup detection, reconnection, and smoothing |
| `src/physics/` | `VortexMethod.Circulation`, `VortexMethod.Dissipation`, `VortexMethod.Energy` | Circulation transfer, baroclinic forcing, subfilter dissipation, and kinetic-energy diagnostics |
| `src/io/` | `VortexMethod.Checkpoint` | Checkpoint files, JLD2 time series, snapshot loading, and mesh statistics |
| `src/diagnostics/` | `VortexMethod.Diagnostics`, `VortexMethod.FastLinAlg` | Allocation/performance diagnostics and small dense linear algebra helpers |

## Where to start

- Domain/grid setup: start in `src/core/domain.jl` for `DomainSpec`,
  `GridSpec`, grid vectors, grid spacing, and periodic wrapping; use
  `src/core/mesh.jl` for structured sheet generation.
- Kernel/grid transfer: start in `src/kernels/kernels.jl` for kernel types and
  support radii, then `src/kernels/grid_transfer.jl` for
  `VortexMethod.GridTransfer` spreading, interpolation, and workspace-backed
  methods.
- Poisson solves: start in `src/poisson/Poisson.jl`, which includes
  `src/poisson/fft.jl` and `src/poisson/solvers.jl` and exposes
  `VortexMethod.Poisson`.
- Time stepping: start in `src/integration/timestep.jl` for
  `VortexMethod.TimeStepper`, `node_velocities`, `grid_velocity`,
  `make_velocity_sampler`, and RK2 updates.
- Remeshing: start in `src/remeshing/Remeshing.jl` for the
  `VortexMethod.Remeshing` namespace, then follow into `basic.jl` for baseline
  remeshing or `quality.jl` for quality and adaptive methods.
- Checkpointing: start in `src/io/checkpoint.jl` for
  `VortexMethod.Checkpoint`, including single snapshots, JLD2 time series, and
  restart helpers.
- Diagnostics: start in `src/diagnostics/performance.jl` for
  `VortexMethod.Diagnostics`, `src/diagnostics/fast_linalg.jl` for
  `VortexMethod.FastLinAlg`, and `src/physics/energy.jl` for kinetic-energy
  diagnostics.

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
  in `src/integration/interface.jl`, then export them from `src/VortexMethod.jl`.
- Add numerical kernels near the stage they affect instead of routing through
  the high-level interface.
- Keep topology-changing mesh changes in `src/remeshing/basic.jl` or
  `src/remeshing/quality.jl`, wired through `src/remeshing/Remeshing.jl`,
  and carry `eleGma` through the return values.
- Add focused tests beside the affected subsystem and update this docs page if
  the source layout or public workflow changes.
