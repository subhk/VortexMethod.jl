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

# Exports organized by source file for better maintainability

# domain.jl - Domain specification and periodic boundary conditions
export DomainSpec, GridSpec,
       default_domain, default_grid,
       wrap_point, wrap_nodes!,

# kernels.jl - Interpolation kernels and spreading functions  
       KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,

# peskin3d.jl - MPI parallel spreading and interpolation
       init_mpi!, finalize_mpi!,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_kernel_mpi,
       interpolate_node_velocity_mpi, interpolate_node_velocity_kernel_mpi,

# poisson3d.jl - FFT-based Poisson solvers
       poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft,

# mesh.jl - Mesh generation and connectivity
       structured_mesh,

# circulation.jl - Circulation management and transport
       node_circulation_from_ele_gamma, ele_gamma_from_node_circ, transport_ele_gamma,

# dissipation.jl - Turbulence models and viscosity
       DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
       VortexStretchingDissipation, MixedScaleModel,
       apply_dissipation!, compute_eddy_viscosity, filter_width,

# timestep.jl - Time integration and velocity computation
       node_velocities, rk2_step!, rk2_step_with_dissipation!,
       grid_velocity, make_velocity_sampler,

# remesh.jl - Basic remeshing operations
       MeshQuality, compute_mesh_quality, quality_based_remesh!,

# remesh_advanced.jl - Advanced remeshing with flow adaptation
       element_quality_metrics, anisotropic_remesh!,
       curvature_based_remesh!, flow_adaptive_remesh!,

# poisson_advanced.jl - Advanced Poisson solvers
       PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
       HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
       solve_poisson_adaptive!, solve_poisson_advanced_mpi!,

# vortex_sheets.jl - Vortex sheet tracking and evolution
       VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet,
       HybridSheet, evolve_sheet!, track_sheet_interface!,
       compute_sheet_curvature, detect_sheet_rollup, adaptive_sheet_tracking!,

# checkpoint.jl - JLD2-based checkpointing and time series
       save_checkpoint!, save_checkpoint_jld2!, load_latest_checkpoint,
       load_latest_jld2, load_checkpoint_jld2, load_latest_checkpoint_jld2, load_checkpoint,
       save_state!, mesh_stats, save_state_timeseries!, series_times, load_series_snapshot, load_series_nearest_time,
       find_series_files, get_series_info,

# energy.jl - Energy calculations and diagnostics
       grid_ke, gamma_ke,

# particle_management.jl - Adaptive particle insertion/removal
       insert_particles_periodic!, remove_particles_periodic!, 
       compact_mesh!, adaptive_particle_control!,
       ParticleInsertionCriteria, ParticleRemovalCriteria,
       insert_vortex_blob_periodic!, remove_weak_vortices!,
       maintain_particle_count!, redistribute_particles_periodic!

end
