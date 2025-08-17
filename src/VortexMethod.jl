module VortexMethod

using FFTW
using MPI

include("domain.jl")
include("peskin3d.jl")
include("poisson3d.jl")

export DomainSpec, GridSpec,
       default_domain, default_grid,
       init_mpi!, finalize_mpi!,
       spread_vorticity_to_grid_mpi,
       poisson_velocity_fft,
       interpolate_node_velocity_mpi

end
