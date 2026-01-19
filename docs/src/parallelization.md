# Parallelization

The implementation uses MPI to parallelize the most expensive parts of the pipeline.

!!! info "MPI-Based Parallelism"
    The parallelization strategy uses MPI with embarrassingly parallel work distribution for spreading and interpolation, avoiding complex halo exchanges or domain decomposition.

## Spreading vorticity to the grid

- Each rank handles a strided subset of grid points (round-robin across the flattened `(k,j,i)` index ordering).
- For each grid point, it collects contributions from nearby elements via the chosen kernel and periodic tiles.
- Local contributions are accumulated in a rank-local buffer; a final `MPI.Allreduce` (SUM) builds the global grid vorticity.

Benefits: embarrassingly parallel, no halo exchange, and no custom domain decomposition needed.

## Poisson solve

!!! note "Centralized Solve"
    The FFT-based Poisson solver currently runs on rank 0 and broadcasts results. For very large problems, consider `PoissonAdvanced` for distributed solvers.

- The FFT-based Poisson solver currently runs on rank 0, computes the full solution `(Ux,Uy,Uz)`, and broadcasts to all ranks.
- For very large problems, `PoissonAdvanced` provides interfaces for iterative or multigrid solvers. A hybrid solver can try FFT first and fall back if the residual is above a threshold.

## Interpolation to nodes

- Each rank handles a strided subset of nodes and accumulates interpolated velocities from nearby grid cells using the same kernel family.
- Per-rank results are combined with `MPI.Allreduce`.

## Time stepping and adaptivity

- Node velocities are computed in parallel, and RK2 updates are carried out locally with final periodic wraps per step.
- Adaptive `dt` is computed from the global maximum grid speed via `MPI.Allreduce` (MAX).

## Checkpointing and I/O

!!! tip "Rank 0 I/O"
    All file I/O is performed on rank 0 to avoid contention and ensure consistent checkpoint files.

- Checkpoints (CSV, MAT, JLD2) are written on rank 0 to avoid file contention. Rank 0 aggregates necessary data and saves.

## Figures from the thesis

Recommended figures:

- Schematic of the Lagrangian–Eulerian pipeline.
- Strong/weak scaling curves (if available).

Place images extracted from the thesis into `docs/src/assets/` and reference here.

