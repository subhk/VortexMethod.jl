# VortexMethod.jl

[![CI](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/VortexMethod.jl/dev/)
[![Coverage](https://codecov.io/gh/subhk/VortexMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/VortexMethod.jl)

A high-performance 3D Lagrangian vortex method with MPI parallelism, periodic domains, multiple interpolation kernels, advanced remeshing, dissipation models, particle management, and efficient JLD2-based checkpointing. 

## Highlights

- **High-Performance Computing**: Lagrangian–Eulerian pipeline with MPI parallelism and optimized FFT-based Poisson solvers
- **Multiple Interpolation Kernels**: Peskin-style, cosine, M4', area-weighted (configurable support radius)
- **Poisson Solvers**: Spectral FFT with periodic boundary conditions, iterative/multigrid interfaces
- **Remeshing**: Edge split/flip/collapse with flow-adaptive and quality-based criteria using periodic metrics  
- **Particle Management**: Automatic insertion/removal, circulation conservation, density redistribution for optimal resolution
- **Dissipation Models**: Smagorinsky, dynamic, vortex-stretching, and mixed-scale turbulence models
- **Vortex Sheets**: Lagrangian/Eulerian/Hybrid structures with curvature and reconnection utilities
- **Modern I/O**: JLD2-based checkpointing with clean filenames (`chkpt_1.jld2`) and time-series with random-access snapshots
- **MPI Parallelism**: Scalable parallel spreading/interpolation with global reductions for diagnostics

## Install

```julia
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Run examples (MPI)

```
mpirun -n 4 julia --project examples/simple3d.jl
mpirun -n 4 julia --project examples/kh3d.jl
mpirun -n 4 julia --project examples/advanced_kh3d.jl
```

## Minimal usage

```julia
using VortexMethod

domain = default_domain(); gr = default_grid()

# Build a structured sheet and an initial element vorticity
Nx, Ny = 64, 64
nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
eleGma = zeros(Float64, size(tri,1), 3); eleGma[:,2] .= 1.0

# Single RK2 step with spectral Poisson solver (high accuracy)
dt = 1e-3
rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt; 
         adaptive=true, CFL=0.5, poisson_mode=:spectral)

# Adaptive particle management for optimal resolution
insert_criteria = ParticleInsertionCriteria(max_particles=50000, max_particle_spacing=0.02)
removal_criteria = ParticleRemovalCriteria(weak_circulation_threshold=1e-8)
n_changed = adaptive_particle_control!(nodeX, nodeY, nodeZ, tri, eleGma, domain; 
                                      insert_criteria=insert_criteria, 
                                      removal_criteria=removal_criteria)

# Advanced remeshing with flow adaptation
vel = make_velocity_sampler(eleGma, triXC, triYC, triZC, domain, gr)
tri, changed = VortexMethod.RemeshAdvanced.flow_adaptive_remesh!(
    nodeX, nodeY, nodeZ, tri, vel, domain;
    max_aspect_ratio=3.0, min_angle_quality=0.4, grad_threshold=0.2
)

# Save checkpoint with clean filename
save_checkpoint!("output/", 1, nodeX, nodeY, nodeZ, tri, eleGma)
# Creates: output/chkpt_1.jld2

# Always keep particles periodic after manual edits
wrap_nodes!(nodeX, nodeY, nodeZ, domain)
```

## Advanced Usage Examples

### Particle Management
```julia
# Maintain target particle count with automatic insertion/removal
target_count = 10000
tolerance = 0.1  # ±10%
n_change = maintain_particle_count!(nodeX, nodeY, nodeZ, tri, eleGma, domain, 
                                    target_count, tolerance)

# Insert vortex blob at specific location
center = (0.5, 0.5, 0.0)
strength = (0.0, 0.0, 1.0)  # ωz = 1
radius = 0.1
n_particles = 100
n_inserted = insert_vortex_blob_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain,
                                         center, strength, radius, n_particles)

# Redistribute particles for uniform spacing
final_count = redistribute_particles_periodic!(nodeX, nodeY, nodeZ, tri, eleGma, domain)
```

### Modern Checkpointing with Configurable Limits
```julia
# Save simulation state with metadata
save_state!("checkpoints/", 0.0, nodeX, nodeY, nodeZ, tri, eleGma;
           domain=domain, grid=gr, dt=dt, CFL=0.5, step=1)

# Time-series storage with automatic file rollover
for step in 1:5000
    # ... time stepping ...
    if step % 10 == 0
        file = save_state_timeseries!("series.jld2", step*dt, nodeX, nodeY, nodeZ, tri, eleGma;
                                      domain=domain, grid=gr, step=step, max_snapshots=500)
        # Automatically creates series.jld2, series_001.jld2, series_002.jld2, etc.
        # when each file reaches 500 snapshots
    end
end

# Load specific snapshot by time
times, steps, count = series_times("series_002.jld2")  # From specific file
idx, snapshot = load_series_nearest_time("series_002.jld2", 5.0)

# Get information about all files in a series
info = get_series_info("series.jld2")
println("Total snapshots across all files: $(info.total_snapshots)")
println("Files: $(info.files)")
println("Snapshots per file: $(info.file_counts)")
```

## Documentation

- Dev docs: https://subhk.github.io/VortexMethod.jl/dev/ (or build locally):

```
julia --project=docs -e 'using Pkg; Pkg.instantiate(); include("docs/make.jl")'
```

Docs include: theory, remeshing criteria, parallelization model, usage recipes, and a full API reference.

## Validation

- Generate a KH time series: `mpirun -n 4 julia --project examples/advanced_kh3d.jl`
- Plot KE into docs assets:

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
OUTPUT_PNG=docs/src/assets/ke_series.png \
julia --project examples/plot_series_ke.jl
```

- Snapshot of |γ|:

```
SERIES_FILE=checkpoints/advanced_series.jld2 \
SNAP_INDEX=10 \
OUTPUT_PNG=docs/src/assets/snapshot_gamma.png \
julia --project examples/plot_snapshot_gamma.jl
```

