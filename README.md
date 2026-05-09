# VortexMethod.jl

[![CI](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/subhk/VortexMethod.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://subhk.github.io/VortexMethod.jl)
[![Coverage](https://codecov.io/gh/subhk/VortexMethod.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/subhk/VortexMethod.jl)

VortexMethod.jl is a Julia implementation of 3D Lagrangian vortex-sheet methods for inviscid, incompressible flows with density interfaces. It combines vortex-in-cell spreading, FFT-based Poisson solves, adaptive remeshing, MPI parallelism, and checkpointing tools.

## Features

- Lagrangian vortex sheets on triangulated surfaces
- Periodic-domain vortex-in-cell spreading and interpolation
- FFT, MPI, and pencil-FFT velocity solve paths
- Flow-adaptive remeshing and particle management
- Dissipation and baroclinic-interface utilities
- JLD2 checkpoints and time-series snapshots

## Install

```sh
julia --project -e 'using Pkg; Pkg.instantiate()'
```

## Quick Start

```julia
using VortexMethod

grid = RectilinearGrid(size=(25, 50, 99),
                       x=(0, 1),
                       y=(0, 1),
                       z=(-1, 1),
                       topology=(Periodic, Periodic, Periodic))

model = VortexSheetModel(; grid,
                         sheet_size=(64, 64),
                         Γ=(0.0, 1.0, 0.0))

simulation = Simulation(model; Δt=1e-3, stop_iteration=10)
run!(simulation)
```

## Reference

This package follows the regularized vortex sheet method developed in:

Stock, M. J. (2006). *A regularized inviscid vortex sheet method for three dimensional flows with density interfaces*. Ph.D. Thesis, California Institute of Technology.
