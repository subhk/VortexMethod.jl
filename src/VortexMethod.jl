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
# HPC optimization modules
include("performance.jl")
include("fast_linalg.jl")
include("soa_layout.jl")
include("cache_optimization.jl")

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
       triangle_normals, baroclinic_ele_gamma, TriangleGeometry, compute_triangle_geometry,
       node_circulation_from_ele_gamma_mpi, ele_gamma_from_node_circ_mpi,
       triangle_normals_mpi, baroclinic_ele_gamma_mpi, transport_ele_gamma_mpi,

# dissipation.jl - Turbulence models and viscosity
       DissipationModel, NoDissipation, SmagorinskyModel, DynamicSmagorinsky,
       VortexStretchingDissipation, MixedScaleModel,
       apply_dissipation!, compute_eddy_viscosity, filter_width,

# timestep.jl - Time integration and velocity computation
       node_velocities, rk2_step!, rk2_step_with_dissipation!,
       grid_velocity, make_velocity_sampler,

# remesh.jl - Basic remeshing operations
       detect_max_edge_length, detect_min_edge_length,
       element_splitting!, edge_flip_small_edge!, remesh_pass!,

# remesh_advanced.jl - Advanced remeshing with flow adaptation
       MeshQuality, compute_mesh_quality, quality_based_remesh!,
       element_quality_metrics, element_quality_metrics_periodic, anisotropic_remesh!,
       curvature_based_remesh!, flow_adaptive_remesh!,

# poisson_advanced.jl - Advanced Poisson solvers
       PoissonSolver, FFTSolver, IterativeSolver, MultigridSolver,
       HybridSolver, BoundaryCondition, PeriodicBC, DirichletBC, NeumannBC,
       solve_poisson_adaptive!, solve_poisson_advanced_mpi!,

# vortex_sheets.jl - Vortex sheet tracking and evolution
       VortexSheet, SheetEvolution, LagrangianSheet, EulerianSheet,
       HybridSheet, evolve_sheet!, track_sheet_interface!,
       compute_sheet_curvature, detect_sheet_rollup, check_sheet_reconnection!,
       reconnect_sheet_nodes!, adaptive_sheet_tracking!, compute_mesh_quality_sheet,

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
       maintain_particle_count!, redistribute_particles_periodic!,
       # MPI synchronized versions (Approach B: identical operations on all ranks)
       insert_particles_periodic_mpi!, remove_particles_periodic_mpi!,
       adaptive_particle_control_mpi!, maintain_particle_count_mpi!,
       redistribute_particles_periodic_mpi!,

# performance.jl - Performance monitoring and profiling
       @vortex_time, PerformanceCounters, reset_counters!, print_performance_report,
       enable_profiling!, disable_profiling!,

# fast_linalg.jl - Optimized linear algebra for small matrices
       solve_3x3!, solve_4x3!, fast_inv_3x3!, fast_det_3x3, fast_cross_product!,
       batch_solve_3x3!, TriangleMatrix3x3, EdgeVectorCache, fast_triangle_area,

# soa_layout.jl - Structure of Arrays memory layout
       TriangleSoA, NodeSoA, VorticitySoA, VelocitySoA,
       aos_to_soa!, soa_to_aos!, vectorized_kernel_eval!,
       soa_triangle_areas!, soa_circulation_solve!, create_soa_layout,

# cache_optimization.jl - Cache-aware algorithms
       TiledPoissonSolver, BlockedSpreadingKernel, CacheAwareMesh,
       tiled_curl_computation!, blocked_kernel_evaluation!,
       cache_optimized_interpolation!, hierarchical_grid_traversal

end
