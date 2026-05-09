# Code Structure Cleanup Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reorganize `src/` around developer-facing concepts and remove duplicate "advanced" vs core subsystem names.

**Architecture:** Use folders as the primary navigation mechanism and a small set of subsystem modules for conceptual boundaries. Merge Poisson and remeshing variants into single subsystems, rename vague/legacy files, then update tests, examples, and docs to teach the new layout.

**Tech Stack:** Julia package layout, Julia modules/includes, Documenter docs, existing `Test` suite, existing KH example scripts.

### Task 1: Add Structure Regression Tests

**Files:**
- Modify: `test/runtests.jl`
- Create: `test/test_code_structure.jl`

**Step 1: Write the failing test**

Create `test/test_code_structure.jl`:

```julia
using Test

repo_root = dirname(@__DIR__)
src_root = joinpath(repo_root, "src")

@testset "source tree is organized by subsystem" begin
    expected_files = [
        "core/domain.jl",
        "core/mesh.jl",
        "core/workspace.jl",
        "core/layout.jl",
        "kernels/kernels.jl",
        "kernels/grid_transfer.jl",
        "poisson/Poisson.jl",
        "poisson/fft.jl",
        "poisson/solvers.jl",
        "integration/timestep.jl",
        "integration/interface.jl",
        "physics/circulation.jl",
        "physics/dissipation.jl",
        "physics/energy.jl",
        "remeshing/Remeshing.jl",
        "remeshing/basic.jl",
        "remeshing/quality.jl",
        "sheets/sheets.jl",
        "io/checkpoint.jl",
        "diagnostics/performance.jl",
        "diagnostics/fast_linalg.jl",
    ]

    for relpath in expected_files
        @test isfile(joinpath(src_root, relpath))
    end
end

@testset "legacy duplicate subsystem files are removed" begin
    removed_files = [
        "peskin3d.jl",
        "poisson3d.jl",
        "poisson_advanced.jl",
        "remesh.jl",
        "remesh_advanced.jl",
        "vortex_sheets.jl",
        "soa_layout.jl",
        "cache_optimization.jl",
    ]

    for relpath in removed_files
        @test !isfile(joinpath(src_root, relpath))
    end
end

@testset "legacy module names are gone from source, tests, examples, and docs" begin
    roots = ["src", "test", "examples", joinpath("docs", "src")]
    stale_patterns = [
        "PoissonAdvanced",
        "RemeshAdvanced",
        "Poisson3D",
        "Peskin3D",
        "VortexSheets",
        "SoALayout",
        "CacheOptimization",
        "poisson3d",
        "poisson_advanced",
        "remesh_advanced",
        "peskin3d",
        "vortex_sheets",
        "cache_optimization",
    ]

    for root in roots
        for (dir, _, filenames) in walkdir(joinpath(repo_root, root))
            for filename in filenames
                file = joinpath(dir, filename)
                if endswith(file, ".jl") || endswith(file, ".md")
                    text = read(file, String)
                    for pattern in stale_patterns
                        @test !occursin(pattern, text)
                    end
                end
            end
        end
    end
end
```

Add the new test near the top of `test/runtests.jl`, after `include("test_domain.jl")`:

```julia
include("test_code_structure.jl")
```

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project -e 'using Test; include("test/test_code_structure.jl")'
```

Expected: FAIL because the new directories do not exist and legacy files/names are still present.

**Step 3: Commit**

```bash
git add test/test_code_structure.jl test/runtests.jl
git commit -m "test: add code structure expectations"
```

### Task 2: Move Files Into the New Directory Layout

**Files:**
- Move: `src/domain.jl` -> `src/core/domain.jl`
- Move: `src/mesh.jl` -> `src/core/mesh.jl`
- Move: `src/workspace.jl` -> `src/core/workspace.jl`
- Move: `src/soa_layout.jl` -> `src/core/layout.jl`
- Move: `src/kernels.jl` -> `src/kernels/kernels.jl`
- Move: `src/peskin3d.jl` -> `src/kernels/grid_transfer.jl`
- Move: `src/poisson3d.jl` -> `src/poisson/fft.jl`
- Move: `src/poisson_advanced.jl` -> `src/poisson/solvers.jl`
- Move: `src/timestep.jl` -> `src/integration/timestep.jl`
- Move: `src/interface.jl` -> `src/integration/interface.jl`
- Move: `src/circulation.jl` -> `src/physics/circulation.jl`
- Move: `src/dissipation.jl` -> `src/physics/dissipation.jl`
- Move: `src/energy.jl` -> `src/physics/energy.jl`
- Move: `src/remesh.jl` -> `src/remeshing/basic.jl`
- Move: `src/remesh_advanced.jl` -> `src/remeshing/quality.jl`
- Move: `src/vortex_sheets.jl` -> `src/sheets/sheets.jl`
- Move: `src/checkpoint.jl` -> `src/io/checkpoint.jl`
- Move: `src/performance.jl` -> `src/diagnostics/performance.jl`
- Move: `src/fast_linalg.jl` -> `src/diagnostics/fast_linalg.jl`
- Leave for now: `src/particle_management.jl`
- Leave for now: `src/cache_optimization.jl`

**Step 1: Move files mechanically**

Run:

```bash
mkdir -p src/core src/kernels src/poisson src/integration src/physics src/remeshing src/sheets src/io src/diagnostics
git mv src/domain.jl src/core/domain.jl
git mv src/mesh.jl src/core/mesh.jl
git mv src/workspace.jl src/core/workspace.jl
git mv src/soa_layout.jl src/core/layout.jl
git mv src/kernels.jl src/kernels/kernels.jl
git mv src/peskin3d.jl src/kernels/grid_transfer.jl
git mv src/poisson3d.jl src/poisson/fft.jl
git mv src/poisson_advanced.jl src/poisson/solvers.jl
git mv src/timestep.jl src/integration/timestep.jl
git mv src/interface.jl src/integration/interface.jl
git mv src/circulation.jl src/physics/circulation.jl
git mv src/dissipation.jl src/physics/dissipation.jl
git mv src/energy.jl src/physics/energy.jl
git mv src/remesh.jl src/remeshing/basic.jl
git mv src/remesh_advanced.jl src/remeshing/quality.jl
git mv src/vortex_sheets.jl src/sheets/sheets.jl
git mv src/checkpoint.jl src/io/checkpoint.jl
git mv src/performance.jl src/diagnostics/performance.jl
git mv src/fast_linalg.jl src/diagnostics/fast_linalg.jl
```

**Step 2: Update root includes only**

Edit `src/VortexMethod.jl` include paths to point at the moved files. Keep all module names and exports unchanged for this task.

**Step 3: Run package import**

Run:

```bash
julia --project -e 'using VortexMethod; println("import ok")'
```

Expected: PASS with `import ok`.

**Step 4: Run focused structural test**

Run:

```bash
julia --project -e 'using Test; include("test/test_code_structure.jl")'
```

Expected: still FAIL because stale module names and `cache_optimization.jl` remain. File-existence checks should mostly pass except the Poisson and remeshing wrapper files that are created in later tasks.

**Step 5: Commit**

```bash
git add src test
git commit -m "refactor: move source files into subsystem directories"
```

### Task 3: Create Poisson Subsystem and Remove PoissonAdvanced/Poisson3D

**Files:**
- Create: `src/poisson/Poisson.jl`
- Modify: `src/poisson/fft.jl`
- Modify: `src/poisson/solvers.jl`
- Modify: `src/integration/timestep.jl`
- Modify: `src/physics/dissipation.jl`
- Modify: `src/physics/energy.jl`
- Modify: `src/VortexMethod.jl`
- Modify tests using `VortexMethod.Poisson3D` or `VortexMethod.PoissonAdvanced`

**Step 1: Add Poisson wrapper module**

Create `src/poisson/Poisson.jl`:

```julia
module Poisson

using FFTW
using MPI
using SparseArrays
using LinearAlgebra
using ..DomainImpl

include("fft.jl")
include("solvers.jl")

export curl_rhs_centered, curl_rhs_centered!, PoissonWorkspace,
       poisson_velocity_fft, poisson_velocity_fft!, poisson_velocity_fft_mpi,
       poisson_velocity_fft_mpi!, poisson_velocity_pencil_fft,
       poisson_velocity_pencil_fft!,
       PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
       HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
       solve_poisson!, solve_poisson_advanced!, solve_poisson_adaptive!,
       solve_poisson_advanced_mpi!

end

using .Poisson: curl_rhs_centered, curl_rhs_centered!, PoissonWorkspace,
                poisson_velocity_fft, poisson_velocity_fft!, poisson_velocity_fft_mpi,
                poisson_velocity_fft_mpi!, poisson_velocity_pencil_fft,
                poisson_velocity_pencil_fft!,
                PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
                HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
                solve_poisson!, solve_poisson_advanced!, solve_poisson_adaptive!,
                solve_poisson_advanced_mpi!
```

**Step 2: Remove nested module wrappers from Poisson files**

In `src/poisson/fft.jl`:

- Remove `module Poisson3D`.
- Remove `using ..DomainImpl`; it is supplied by the wrapper.
- Remove the final `using .Poisson3D: ...` block.
- Remove the final `end # module`.

In `src/poisson/solvers.jl`:

- Remove `module PoissonAdvanced`.
- Remove duplicate `using FFTW`, `using MPI`, `using SparseArrays`, `using LinearAlgebra`, and `using ..DomainImpl`.
- Replace `using ..Poisson3D: poisson_velocity_fft` with no import; `poisson_velocity_fft` is in the same `Poisson` module.
- Remove the final `using .PoissonAdvanced: ...` block.
- Remove the final `end # module`.

**Step 3: Update imports**

Replace:

```julia
using ..Poisson3D
```

with:

```julia
using ..Poisson
```

in `src/integration/timestep.jl`, `src/physics/dissipation.jl`, and `src/physics/energy.jl`.

Replace calls like:

```julia
Poisson3D.poisson_velocity_fft(...)
```

with:

```julia
Poisson.poisson_velocity_fft(...)
```

**Step 4: Update root include**

In `src/VortexMethod.jl`, include only:

```julia
include("poisson/Poisson.jl")
```

Remove separate includes for `poisson/fft.jl` and `poisson/solvers.jl`.

**Step 5: Update tests**

Replace test references:

```julia
VortexMethod.Poisson3D
VortexMethod.PoissonAdvanced
```

with:

```julia
VortexMethod.Poisson
```

where module-qualified internals are needed.

**Step 6: Run Poisson tests**

Run:

```bash
julia --project -e 'using VortexMethod, Test; include("test/test_poisson.jl"); include("test/test_parallel_fft.jl")'
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src test
git commit -m "refactor: merge Poisson solvers into one subsystem"
```

### Task 4: Create Remeshing Subsystem and Remove RemeshAdvanced

**Files:**
- Create: `src/remeshing/Remeshing.jl`
- Modify: `src/remeshing/basic.jl`
- Modify: `src/remeshing/quality.jl`
- Modify: `src/sheets/sheets.jl`
- Modify: `src/VortexMethod.jl`
- Modify tests/examples/docs using `Remesh` or `RemeshAdvanced`

**Step 1: Add Remeshing wrapper module**

Create `src/remeshing/Remeshing.jl`:

```julia
module Remeshing

using LinearAlgebra
using StaticArrays
using ..DomainImpl
using ..Peskin3D
using ..Circulation

include("basic.jl")
include("quality.jl")

export detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!, remesh_pass!,
       MeshQuality, compute_mesh_quality, quality_based_remesh!,
       element_quality_metrics, element_quality_metrics_periodic,
       anisotropic_remesh!, curvature_based_remesh!, flow_adaptive_remesh!,
       quality_split_triangle!

end

using .Remeshing: detect_max_edge_length, detect_min_edge_length,
                  element_splitting!, edge_flip_small_edge!, remesh_pass!,
                  MeshQuality, compute_mesh_quality, quality_based_remesh!,
                  element_quality_metrics, element_quality_metrics_periodic,
                  anisotropic_remesh!, curvature_based_remesh!, flow_adaptive_remesh!,
                  quality_split_triangle!
```

**Step 2: Remove nested module wrappers**

In `src/remeshing/basic.jl`:

- Remove `module Remesh`.
- Remove wrapper-level `using` lines that are now in `Remeshing.jl`.
- Remove final `using .Remesh: ...` block.
- Remove final `end # module`.

In `src/remeshing/quality.jl`:

- Remove `module RemeshAdvanced`.
- Remove wrapper-level `using` lines that are now in `Remeshing.jl`.
- Remove final `using .RemeshAdvanced: ...` block.
- Remove final `end # module`.

**Step 3: Update dependencies**

In `src/sheets/sheets.jl`, replace:

```julia
using ..RemeshAdvanced: element_quality_metrics_periodic
```

with:

```julia
using ..Remeshing: element_quality_metrics_periodic
```

**Step 4: Update root include**

In `src/VortexMethod.jl`, include only:

```julia
include("remeshing/Remeshing.jl")
```

Remove separate includes for `remeshing/basic.jl` and `remeshing/quality.jl`.

**Step 5: Update references**

Replace:

```julia
VortexMethod.RemeshAdvanced
VortexMethod.Remesh
```

with:

```julia
VortexMethod.Remeshing
```

in tests, examples, and docs.

**Step 6: Run remeshing tests**

Run:

```bash
julia --project -e 'using VortexMethod, Test; include("test/test_remesh.jl"); include("test/test_stock_regressions.jl")'
```

Expected: PASS.

**Step 7: Commit**

```bash
git add src test examples docs/src
git commit -m "refactor: merge remeshing APIs into one subsystem"
```

### Task 5: Rename Grid Transfer, Sheets, Layout, and Diagnostics Modules

**Files:**
- Modify: `src/kernels/grid_transfer.jl`
- Modify: `src/core/layout.jl`
- Modify: `src/sheets/sheets.jl`
- Modify: `src/diagnostics/performance.jl`
- Modify: `src/VortexMethod.jl`
- Modify tests/examples/docs using old module names

**Step 1: Rename module wrappers**

Perform these module renames:

```text
Peskin3D -> GridTransfer
VortexSheets -> Sheets
SoALayout -> Layout
Performance -> Diagnostics
```

For each file:

- Change the `module ...` declaration.
- Change the final `using .OldModule: ...` block to `using .NewModule: ...`.
- Update internal qualified calls.

**Step 2: Update imports**

Replace:

```julia
using ..Peskin3D
using ..VortexSheets
```

with:

```julia
using ..GridTransfer
using ..Sheets
```

where those imports exist.

**Step 3: Decide cache optimization fate**

Move reusable cache helpers into `src/diagnostics/performance.jl` only if they are still used. Otherwise remove `src/cache_optimization.jl` and its exports from `src/VortexMethod.jl`.

Before removal, run:

```bash
rg -n "TiledPoissonSolver|BlockedSpreadingKernel|CacheAwareMesh|tiled_curl_computation|blocked_kernel_evaluation|cache_optimized_interpolation|hierarchical_grid_traversal" src test examples docs/src
```

If the only references are exports or unused tests, remove the stale code. If tests cover it, move the covered functions into `src/diagnostics/performance.jl`.

**Step 4: Run targeted tests**

Run:

```bash
julia --project -e 'using VortexMethod, Test; include("test/test_performance.jl"); include("test/mpi_parallel_correctness.jl")'
```

Expected: PASS for non-MPI tests. If `mpi_parallel_correctness.jl` requires MPI launcher context, use the full test command from `test/runtests.jl` with `VORTEXMETHOD_RUN_MPI_TESTS=true`.

**Step 5: Commit**

```bash
git add src test examples docs/src
git commit -m "refactor: rename internal subsystem modules"
```

### Task 6: Curate Root Exports and Root Include Map

**Files:**
- Modify: `src/VortexMethod.jl`
- Modify tests that relied on removed top-level exports

**Step 1: Rewrite include map**

Organize `src/VortexMethod.jl` includes by subsystem:

```julia
include("core/domain.jl")
include("core/mesh.jl")
include("core/workspace.jl")
include("core/layout.jl")

include("kernels/kernels.jl")
include("kernels/grid_transfer.jl")

include("poisson/Poisson.jl")
include("physics/circulation.jl")
include("physics/dissipation.jl")
include("physics/energy.jl")

include("integration/timestep.jl")
include("integration/interface.jl")

include("remeshing/Remeshing.jl")
include("sheets/sheets.jl")
include("io/checkpoint.jl")
include("particle_management.jl")
include("diagnostics/fast_linalg.jl")
include("diagnostics/performance.jl")
```

Adjust order if imports require it. Keep the order explicit and commented by subsystem.

**Step 2: Reduce top-level exports**

Keep top-level exports for:

- Core user types: `DomainSpec`, `GridSpec`, `VortexWorkspace`, `Simulation`, `Clock`, `VortexSheetModel`.
- Main user functions: `structured_mesh`, `time_step!`, `run!`, `set!`, `rk2_step!`, `rk2_step_with_dissipation!`.
- Main numerical APIs already used by examples: `poisson_velocity_fft`, `poisson_velocity_fft!`, `node_velocities`, `grid_velocity`.
- Public physics/remeshing/sheet APIs documented in `docs/src`.

Remove top-level exports for low-level helpers unless tests/examples/docs clearly treat them as public. Those helpers remain accessible by module-qualified names such as `VortexMethod.GridTransfer.triangle_areas`.

**Step 3: Update tests for module-qualified internals**

For low-level helper tests, replace top-level calls with subsystem-qualified calls. Example:

```julia
VortexMethod.GridTransfer.triangle_areas(...)
VortexMethod.Remeshing.quality_based_remesh!(...)
VortexMethod.Poisson.PoissonWorkspace{Float64}
```

**Step 4: Run import/export checks**

Run:

```bash
julia --project -e 'using VortexMethod; println(names(VortexMethod) |> length)'
julia --project -e 'using VortexMethod, Test; include("test/test_interface.jl")'
```

Expected: package imports and high-level interface tests pass.

**Step 5: Commit**

```bash
git add src test
git commit -m "refactor: curate public exports"
```

### Task 7: Update Examples and Documentation

**Files:**
- Modify: `examples/*.jl`
- Modify: `examples/*.md`
- Modify: `docs/src/*.md`

**Step 1: Replace old module names**

Run:

```bash
rg -n "PoissonAdvanced|RemeshAdvanced|Poisson3D|Peskin3D|VortexSheets|SoALayout|CacheOptimization|poisson3d|poisson_advanced|remesh_advanced|peskin3d|vortex_sheets|cache_optimization" examples docs/src
```

Update examples and docs to use:

```text
VortexMethod.Poisson
VortexMethod.Remeshing
VortexMethod.GridTransfer
VortexMethod.Sheets
VortexMethod.Layout
VortexMethod.Diagnostics
```

**Step 2: Update codebase documentation**

Update `docs/src/codebase.md` so it lists the new subsystem folders and explains where to start for:

- domain/grid setup
- kernel/grid transfer
- Poisson solves
- time stepping
- remeshing
- checkpointing
- diagnostics

**Step 3: Run docs smoke check**

Run:

```bash
julia --project=docs docs/make.jl
```

Expected: Documenter build passes.

**Step 4: Run KH example smoke tests**

Run:

```bash
julia --project examples/test_kh_examples.jl
```

Expected: PASS.

**Step 5: Commit**

```bash
git add examples docs/src
git commit -m "docs: update examples for subsystem layout"
```

### Task 8: Final Stale-Name Sweep and Full Verification

**Files:**
- Modify any file still containing stale names from the sweep.

**Step 1: Run stale-name sweep**

Run:

```bash
rg -n "PoissonAdvanced|RemeshAdvanced|Poisson3D|Peskin3D|VortexSheets|SoALayout|CacheOptimization|poisson3d|poisson_advanced|remesh_advanced|peskin3d|vortex_sheets|cache_optimization" src test examples docs/src
```

Expected: no output.

**Step 2: Run structural test**

Run:

```bash
julia --project -e 'using Test; include("test/test_code_structure.jl")'
```

Expected: PASS.

**Step 3: Run full suite**

Run:

```bash
julia --project test/runtests.jl
```

Expected: PASS.

**Step 4: Run optional MPI suite**

Run if MPI is available:

```bash
VORTEXMETHOD_RUN_MPI_TESTS=true julia --project test/runtests.jl
```

Expected: PASS.

**Step 5: Run diff hygiene**

Run:

```bash
git diff --check
git status --short
```

Expected: no whitespace errors. `git status --short` should show only intended tracked changes or unrelated pre-existing untracked files.

**Step 6: Commit**

```bash
git add src test examples docs/src
git commit -m "refactor: finish code structure cleanup"
```

### Task 9: Post-Cleanup Developer Notes

**Files:**
- Modify: `docs/src/codebase.md`
- Modify: `docs/src/api.md`

**Step 1: Add a short navigation guide**

In `docs/src/codebase.md`, add a section:

```markdown
## Where to Start

- Domain and grid setup: `src/core/`
- Grid transfer and Peskin kernels: `src/kernels/`
- Poisson solvers and RHS construction: `src/poisson/`
- Time stepping: `src/integration/`
- Circulation, dissipation, and energy: `src/physics/`
- Remeshing and quality checks: `src/remeshing/`
- Vortex sheet tracking: `src/sheets/`
- Checkpointing: `src/io/`
- Profiling and small optimized kernels: `src/diagnostics/`
```

**Step 2: Run docs and full tests**

Run:

```bash
julia --project=docs docs/make.jl
julia --project test/runtests.jl
```

Expected: both pass.

**Step 3: Commit**

```bash
git add docs/src/codebase.md docs/src/api.md
git commit -m "docs: add source navigation guide"
```
