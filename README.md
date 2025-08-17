# VortexMethod

[![Build Status](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Build Status](https://app.travis-ci.com/subhk/VortexMethod.jl.svg?branch=main)](https://app.travis-ci.com/subhk/VortexMethod.jl)
[![Coverage](https://codecov.io/gh/subhk/VortexMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/VortexMethod.jl)

3D vortex method utilities with MPI-enabled spreading and interpolation, inspired by the Python reference in `python/` and the accompanying thesis. This initial implementation supports:

- Spreading triangular element vorticity to a periodic 3D grid (MPI parallel over grid points)
- FFT-based Poisson solve for velocity from vorticity’s curl RHS
- Interpolation of grid velocity back to Lagrangian nodes (MPI parallel over nodes)

The API is minimal and meant to be composed with your meshing and time-stepping code.

## Quick Start

- Instantiate and run with MPI (e.g., 4 ranks):

```
julia --project -e 'using Pkg; Pkg.instantiate()'
mpirun -n 4 julia --project examples/simple3d.jl
mpirun -n 4 julia --project examples/kh3d.jl
```

## API

- `DomainSpec(Lx,Ly,Lz)`, `GridSpec(nx,ny,nz)`, `default_domain()`, `default_grid()`
- `init_mpi!()`, `finalize_mpi!()`
- `spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, grid)` → `(VorX,VorY,VorZ)` as `(nz,ny,nx)` arrays
- `curl_rhs_centered(VorX,VorY,VorZ, dx,dy,dz)` → `(u_rhs, v_rhs, w_rhs)`
- `poisson_velocity_fft(u_rhs,v_rhs,w_rhs, dom)` → `(Ux,Uy,Uz)`
- `poisson_velocity_fft_mpi(u_rhs,v_rhs,w_rhs, dom)` → `(Ux,Uy,Uz)` (rank-0 solve + broadcast)
- `interpolate_node_velocity_mpi(Ux,Uy,Uz, nodeX,nodeY,nodeZ, dom, grid)` → `(u,v,w)`

Element vorticity `eleGma` is an `nt×3` matrix of vortex strength vectors per triangle. Triangle coordinates are `nt×3` matrices for `x`, `y`, `z` (vertex columns 1:3).

## Notes

- Periodic BCs in all directions. The FFT Poisson solver uses standard spectral inversion, and the spread/interpolation applies a 3D Peskin kernel with 4-way triangle subdivision (mirrors the Python code’s `delr=4`).
- MPI parallelization uses simple strided work distribution and `Allreduce` to combine results; no external MPI FFTs are required. The FFT solve currently runs on a single rank; results are replicated to all ranks after reductions.
- For realistic problems, integrate your meshing/remeshing and time stepping based on the existing Python scripts as reference.
