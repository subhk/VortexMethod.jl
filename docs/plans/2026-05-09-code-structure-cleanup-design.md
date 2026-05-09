# Code Structure Cleanup Design

## Goal

Simplify the source tree so users and developers can find the relevant code by concept. This is a breaking cleanup: old module names and file names may change, examples and tests should be updated to teach the new structure, and compatibility shims are not required unless they make the migration easier to review.

## Recommended Approach

Use a domain-first package structure and merge duplicate basic/advanced concepts. The cleanup should improve navigation without redesigning the numerical methods at the same time.

The new tree should be organized around the areas developers search for:

```text
src/
  VortexMethod.jl
  core/
    domain.jl
    mesh.jl
    workspace.jl
    layout.jl
  kernels/
    kernels.jl
    peskin.jl
  poisson/
    poisson.jl
    fft.jl
    iterative.jl
    workspace.jl
  integration/
    timestep.jl
    interface.jl
  physics/
    circulation.jl
    dissipation.jl
    energy.jl
  remeshing/
    remeshing.jl
    quality.jl
    adaptive.jl
  sheets/
    vortex_sheets.jl
  io/
    checkpoint.jl
  diagnostics/
    performance.jl
```

## Module and API Boundaries

Use folders as the main navigation mechanism. Avoid adding many nested Julia modules unless they clearly reduce coupling. `src/VortexMethod.jl` should become a readable dependency map that includes subsystem files in order.

Old duplicate concepts should be folded into one subsystem:

- `poisson3d.jl` and `poisson_advanced.jl` become the `poisson/` subsystem.
- `remesh.jl` and `remesh_advanced.jl` become the `remeshing/` subsystem.
- `peskin3d.jl` moves under `kernels/`, preferably as `peskin.jl` or `grid_transfer.jl`.
- `soa_layout.jl` becomes `core/layout.jl`.
- `cache_optimization.jl` should be split: reusable optimized kernels move near the code they optimize, while exploratory wrappers move to `diagnostics/` or are removed if unused.

The public API should be curated. Export user-facing workflow functions and core types, but stop exporting low-level implementation helpers by default. Tests can use module-qualified internals where needed.

## Migration Plan

1. Create the new directory layout and move files with minimal behavioral edits.
2. Update `VortexMethod.jl` includes and exports so the package precompiles.
3. Merge Poisson code into one `poisson/` subsystem.
4. Merge remeshing code into one `remeshing/` subsystem.
5. Rename files away from broad or vague names such as `advanced` and `optimization`.
6. Update tests, examples, and docs to use the new paths and public names.
7. Remove stale exports and stale references once the suite is green.

## Testing Gates

Run these checks before calling the cleanup complete:

```bash
julia --project test/runtests.jl
```

Also run the KH example smoke tests and a stale-name sweep for:

- `PoissonAdvanced`
- `RemeshAdvanced`
- `poisson3d`
- `poisson_advanced`
- `remesh_advanced`
- `peskin3d`
- `cache_optimization`

The implementation should keep the existing numerical behavior covered by tests while making the package easier to navigate.
