# Kelvin-Helmholtz Instability Examples

This directory contains three Kelvin-Helmholtz (KH) instability examples that demonstrate different features of the VortexMethod.jl package, with particular emphasis on parallel FFT computation using PencilFFTs.

## Examples Overview

### 1. `kh3d.jl` - Basic KH Simulation with Parallel FFT Support

**Features:**
- Simple Kelvin-Helmholtz vortex sheet simulation
- Parallel FFT support via PencilFFTs
- Adaptive time stepping and remeshing
- JLD2 checkpointing with energy monitoring

**Usage:**
```bash
# Serial FFT (original behavior)
julia kh3d.jl

# Parallel FFT with MPI
mpirun -n 4 julia kh3d.jl --parallel-fft

# Alternative parallel flag
mpirun -n 8 julia kh3d.jl --parallel
```

### 2. `kh3d_parallel.jl` - Enhanced Parallel Simulation

**Features:**
- Comprehensive command-line configuration
- Performance monitoring and timing analysis
- Side-by-side performance comparison mode
- Enhanced initial conditions with perturbations
- Detailed metadata logging

**Usage:**
```bash
# Basic parallel run
mpirun -n 4 julia kh3d_parallel.jl --parallel-fft

# High resolution simulation
mpirun -n 8 julia kh3d_parallel.jl --parallel-fft --nx=128 --ny=128 --steps=100

# Performance comparison (runs both serial and parallel)
mpirun -n 4 julia kh3d_parallel.jl --compare-performance

# Custom parameters
julia kh3d_parallel.jl --nx=96 --ny=96 --dt=5e-4 --save-interval=0.05 --poisson-mode=spectral
```

**Command-line Options:**
- `--parallel-fft`: Use PencilFFTs for distributed FFT
- `--compare-performance`: Run both serial and parallel for timing comparison
- `--nx=N`, `--ny=N`: Mesh resolution (default: 64x64)
- `--steps=N`: Number of time steps (default: 50)
- `--dt=X`: Time step size (default: 1e-3)
- `--save-interval=X`: Save interval in physical time (default: 0.1)
- `--poisson-mode=MODE`: `spectral` or `fd` (default: fd)
- `--output-dir=DIR`: Output directory (default: checkpoints)

### 3. `advanced_kh3d.jl` - Advanced Features Demo

**Features:**
- M4' interpolation kernel for enhanced accuracy
- Dynamic Smagorinsky dissipation model
- Hybrid Poisson solver (FFT + iterative)
- Quality-based mesh adaptation
- Parallel FFT support

**Usage:**
```bash
# Advanced simulation with parallel FFT
mpirun -n 4 julia advanced_kh3d.jl --parallel-fft

# Serial advanced simulation
julia advanced_kh3d.jl
```

## Parallel FFT Performance

### When to Use Parallel FFT

**Use PencilFFTs (`--parallel-fft`) when:**
- Running on multiple MPI ranks (>= 4 recommended)
- Large grid sizes (>= 64³ recommended)
- Memory per rank is limited
- Want to scale to many cores/nodes

**Use Serial FFT (default) when:**
- Running on single rank or few ranks
- Small to medium grid sizes
- Memory is abundant
- Quick prototyping or testing

### Expected Performance Benefits

For typical KH simulations:

| Grid Size | MPI Ranks | Expected Speedup |
|-----------|-----------|------------------|
| 64³       | 4         | 1.5-2x          |
| 128³      | 8         | 3-5x            |
| 256³      | 16        | 6-10x           |

*Actual speedups depend on hardware, network, and problem characteristics*

## Output and Visualization

### Checkpoint Files

All examples save data in JLD2 format:
- **Series files**: `series.jld2` - Time series in single file
- **Individual checkpoints**: `chkpt_tXXXXXX_YYYYMMDD_HHMMSS.jld2`

### Loading Data

```julia
using VortexMethod

# Load latest checkpoint
ck = load_latest_jld2("checkpoints")
nodeX, nodeY, nodeZ = ck.nodeX, ck.nodeY, ck.nodeZ
tri, eleGma = ck.tri, ck.eleGma

# Load from time series
times, steps, count = series_times("checkpoints/series.jld2")
snap = load_series_snapshot("checkpoints/series.jld2", 10)  # 10th snapshot
```

### Performance Analysis

The `kh3d_parallel.jl` example provides detailed timing analysis:

```
PERFORMANCE SUMMARY
============================================================
Total simulation time: 45.32 seconds
MPI ranks: 4
FFT mode: PencilFFTs (parallel)

Time per step:
  Average: 0.8912 s
  Min:     0.8234 s  
  Max:     1.0123 s

FFT/Poisson solve time:
  Average: 0.3456 s
  Total:   17.28 s (38.1%)

Remeshing time:
  Average: 0.1234 s
  Total:   6.17 s (13.6%)
============================================================
```

## Troubleshooting

### Common Issues

1. **PencilFFTs not found**
   ```
   ERROR: KeyError(Base.PkgId(Base.UUID("4a48f351-57a6-4416-9ec4-c37015456aae"), "PencilFFTs"))
   ```
   **Solution**: Install PencilFFTs: `julia -e "using Pkg; Pkg.add(\"PencilFFTs\")"`

2. **MPI errors with PencilFFTs**
   ```
   ERROR: MPI not properly initialized
   ```
   **Solution**: Ensure MPI is properly configured and use `mpirun`

3. **Memory issues with large grids**
   **Solution**: Use more MPI ranks or enable parallel FFT to distribute memory

### Performance Tips

1. **Grid Size**: Use grid sizes that are powers of 2 for optimal FFT performance
2. **MPI Ranks**: For PencilFFTs, use ranks that divide evenly into grid dimensions
3. **Memory**: Monitor memory usage; parallel FFT reduces per-rank memory requirements
4. **I/O**: Use series format for better I/O performance with many time steps

## References

- **PencilFFTs.jl**: https://github.com/jipolanco/PencilFFTs.jl
- **VortexMethod.jl**: https://github.com/your-repo/VortexMethod.jl
- **Kelvin-Helmholtz Instability**: Classical fluid dynamics instability example