module VortexMethod

using FFTW
using MPI

include("domain.jl")
include("kernels.jl")
include("peskin3d.jl")
include("poisson3d.jl")
include("mesh.jl")
# Dependencies needed by TimeStepper
include("circulation.jl")
include("dissipation.jl")
# Time integration routines
include("timestep.jl")
# Remeshing and advanced solvers
include("remesh.jl")
include("remesh_advanced.jl")
include("poisson_advanced.jl")
include("vortex_sheets.jl")
include("checkpoint.jl")
include("energy.jl")
include("particle_management.jl")

export DomainSpec, GridSpec,
       default_domain, default_grid,
       wrap_point, wrap_nodes!,
       init_mpi!, finalize_mpi!,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_kernel_mpi,
       poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft,
       interpolate_node_velocity_mpi, interpolate_node_velocity_kernel_mpi,
       structured_mesh,
       node_velocities, rk2_step!, rk2_step_with_dissipation!,
       grid_velocity, make_velocity_sampler,
       node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,
       save_checkpoint!, save_checkpoint_jld2!, load_latest_checkpoint,
       load_latest_jld2, load_checkpoint_jld2, load_latest_checkpoint_jld2, load_checkpoint,
       save_state!, mesh_stats, save_state_timeseries!, series_times, load_series_snapshot, load_series_nearest_time,
       find_series_files, get_series_info,
       grid_ke, gamma_ke,
       KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,
       MeshQuality, compute_mesh_quality, quality_based_remesh!,
       element_quality_metrics, anisotropic_remesh!,
       curvature_based_remesh!, flow_adaptive_remesh!,
       DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
       VortexStretchingDissipation, MixedScaleModel,
       apply_dissipation!, compute_eddy_viscosity, filter_width,
       PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
       HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
       solve_poisson_adaptive!, solve_poisson_advanced_mpi!,
       VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet,
       HybridSheet, evolve_sheet!, track_sheet_interface!,
       compute_sheet_curvature, detect_sheet_rollup, adaptive_sheet_tracking!,
       insert_particles_periodic!, remove_particles_periodic!, 
       compact_mesh!, adaptive_particle_control!,
       ParticleInsertionCriteria, ParticleRemovalCriteria,
       insert_vortex_blob_periodic!, remove_weak_vortices!,
       maintain_particle_count!, redistribute_particles_periodic!

end
