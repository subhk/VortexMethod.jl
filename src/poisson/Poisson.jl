module Poisson

using FFTW
using MPI
using PencilFFTs
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
       solve_poisson!, solve_poisson_adaptive!, solve_poisson_mpi!

end

using .Poisson: curl_rhs_centered, curl_rhs_centered!, PoissonWorkspace,
                poisson_velocity_fft, poisson_velocity_fft!, poisson_velocity_fft_mpi,
                poisson_velocity_fft_mpi!, poisson_velocity_pencil_fft,
                poisson_velocity_pencil_fft!,
                PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
                HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
                solve_poisson!, solve_poisson_adaptive!, solve_poisson_mpi!
