module VortexMethod

using FFTW
using MPI

include("domain.jl")
include("peskin3d.jl")
include("poisson3d.jl")
include("mesh.jl")
include("timestep.jl")
include("remesh.jl")
include("circulation.jl")
include("checkpoint.jl")
include("energy.jl")

export DomainSpec, GridSpec,
       default_domain, default_grid,
       init_mpi!, finalize_mpi!,
       spread_vorticity_to_grid_mpi,
       poisson_velocity_fft, poisson_velocity_fft_mpi,
       interpolate_node_velocity_mpi,
       structured_mesh,
       node_velocities, rk2_step!,
       node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
       save_checkpoint!, save_checkpoint_mat!, save_checkpoint_jld2!, load_latest_checkpoint
       , load_latest_jld2, load_checkpoint_jld2, save_state!, mesh_stats, save_state_timeseries!, series_times, load_series_snapshot, load_series_nearest_time,
       grid_ke, gamma_ke

end
