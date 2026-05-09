module VortexMethod

using FFTW
using MPI

# Core data model and layouts
include("core/domain.jl")
include("core/mesh.jl")
include("core/layout.jl")

# Kernels and grid transfer
include("kernels/kernels.jl")

# Physics
include("physics/circulation.jl")

# Workspace depends on circulation geometry; grid transfer defines workspace-backed methods.
include("core/workspace.jl")
include("kernels/grid_transfer.jl")

# Poisson solvers
include("poisson/Poisson.jl")

# Physics modules depending on grid transfer and Poisson
include("physics/dissipation.jl")
include("physics/energy.jl")

# Time integration and user interface
include("integration/timestep.jl")
include("integration/interface.jl")

# Remeshing, sheets, I/O, and diagnostics
include("remeshing/Remeshing.jl")
include("sheets/sheets.jl")
include("io/checkpoint.jl")
include("particle_management.jl")
include("diagnostics/fast_linalg.jl")
include("diagnostics/performance.jl")

# Curated public surface. Subsystem internals remain available by qualified
# names, for example VortexMethod.GridTransfer.triangle_areas.
export DomainSpec, GridSpec, default_domain, default_grid,
       grid_vectors, grid_spacing, grid_mesh,
       wrap_point, wrap_nodes!,
       structured_mesh,
       VortexWorkspace,

       KernelType, PeskinStandard, PeskinCosine, M4Prime, AreaWeighting,

       init_mpi!, finalize_mpi!,
       spread_vorticity_to_grid_mpi, spread_vorticity_to_grid_mpi!,
       spread_vorticity_to_grid_kernel_mpi, spread_vorticity_to_grid_kernel_mpi!,
       interpolate_node_velocity_mpi, interpolate_node_velocity_mpi!,
       interpolate_node_velocity_kernel_mpi, interpolate_node_velocity_kernel_mpi!,

       poisson_velocity_fft, poisson_velocity_fft!,
       poisson_velocity_fft_mpi, poisson_velocity_fft_mpi!,
       poisson_velocity_pencil_fft, poisson_velocity_pencil_fft!,

       node_circulation_from_ele_gamma, ele_gamma_from_node_circ,
       transport_ele_gamma, baroclinic_ele_gamma,
       node_circulation_from_ele_gamma_mpi, ele_gamma_from_node_circ_mpi,
       transport_ele_gamma_mpi, baroclinic_ele_gamma_mpi,
       node_circulation_from_ele_gamma!, ele_gamma_from_node_circ!,

       DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
       VortexStretchingDissipation, MixedScaleModel,
       apply_dissipation!, compute_eddy_viscosity, filter_width,

       grid_ke, gamma_ke,

       node_velocities, node_velocities!,
       grid_velocity, grid_velocity!,
       make_velocity_sampler,
       rk2_step!, rk2_step_with_dissipation!,

       Periodic, Bounded, Flat, RectilinearGrid, Clock, VortexSheetModel,
       Simulation, set!, time_step!, run!,

       detect_max_edge_length, detect_min_edge_length, remesh_pass!,
       compute_mesh_quality, anisotropic_remesh!,
       curvature_based_remesh!, flow_adaptive_remesh!,

       VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet,
       HybridSheet, evolve_sheet!, track_sheet_interface!,
       compute_sheet_curvature, detect_sheet_rollup,
       check_sheet_reconnection!, reconnect_sheet_nodes!,
       adaptive_sheet_tracking!,

       save_checkpoint!, save_checkpoint_jld2!, load_latest_checkpoint,
       load_latest_jld2, load_checkpoint_jld2, load_latest_checkpoint_jld2,
       load_checkpoint, save_state!, mesh_stats, save_state_timeseries!,
       series_times, load_series_snapshot, load_series_nearest_time,
       find_series_files, get_series_info,

       insert_particles_periodic!, remove_particles_periodic!,
       compact_mesh!, adaptive_particle_control!,
       ParticleInsertionCriteria, ParticleRemovalCriteria,
       insert_vortex_blob_periodic!, remove_weak_vortices!,
       maintain_particle_count!, redistribute_particles_periodic!,
       insert_particles_periodic_mpi!, remove_particles_periodic_mpi!,
       adaptive_particle_control_mpi!, maintain_particle_count_mpi!,
       redistribute_particles_periodic_mpi!

end
