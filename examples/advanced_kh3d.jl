# Enhanced Kelvin-Helmholtz instability with all advanced features
# Demonstrates: multiple kernels, advanced remeshing, dissipation models, 
# Poisson solvers, and vortex sheet tracking

using VortexMethod
using MPI
using Printf

init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

# Advanced configuration
dom = default_domain()
gr = default_grid()

# Solver options
kernel_type = M4Prime(2.0)  # Use M4' kernel for better accuracy
dissipation_model = DynamicSmagorinsky(0.17, 10)  # Dynamic Smagorinsky model
poisson_solver = HybridSolver(
    FFTSolver(:spectral, PeriodicBC()),
    IterativeSolver(:cg, 1e-8, 1000, :jacobi, PeriodicBC()),
    1e-6
)

# Mesh resolution
Nx = 64
Ny = 64

nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; dom=dom)

nt = size(tri,1)
eleGma = zeros(Float64, nt, 3)

# Enhanced initial condition with multiple vortex sheets
for t in 1:nt
    cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
    cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
    
    # Primary KH sheet
    sheet1_strength = tanh(20*(cy - π/2)) 
    
    # Secondary sheet for more complex dynamics
    sheet2_strength = 0.3 * tanh(15*(cy - 3*π/2))
    
    # Add small perturbation for instability growth
    perturbation = 0.1 * sin(4*cx) * exp(-(cy - π)^2)
    
    eleGma[t,2] = sheet1_strength + sheet2_strength + perturbation
end

# Initialize vortex sheet tracking
vortex_sheet = VortexSheet(nodeX, nodeY, nodeZ, tri, eleGma)

dt = 1e-3
nsteps = 100
Atg = 0.1  # Moderate baroclinic forcing

# Advanced remeshing parameters
remesh_every = 2
ar_max = 3.0
quality_threshold = 0.4
curvature_threshold = 0.8

# Monitoring and output
save_interval = 0.05
series_file = "checkpoints/advanced_series.jld2"
next_save_t = save_interval
time = 0.0

if rank == 0
    println("="^60)
    println("ADVANCED KELVIN-HELMHOLTZ 3D SIMULATION")
    println("Features: M4' kernel, Dynamic Smagorinsky, Hybrid Poisson solver")
    println("Advanced remeshing with quality metrics")
    println("="^60)
    println("Nx=$(Nx) Ny=$(Ny) nt=$(nt) dt=$(dt) steps=$(nsteps)")
    println("Kernel: $(typeof(kernel_type))")
    println("Dissipation: $(typeof(dissipation_model))")
    println("Poisson: $(typeof(poisson_solver))")
end

for it in 1:nsteps
    # Enhanced time stepping with all advanced features
    dt_used = rk2_step_with_dissipation!(
        nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt, dissipation_model;
        At=Atg, adaptive=true, CFL=0.4, poisson_mode=:spectral, kernel=kernel_type
    )
    time += dt_used
    
    # Update triangle coordinates
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    
    # Advanced quality-based remeshing
    if it % remesh_every == 0
        # Compute mesh quality
        qualities = compute_mesh_quality(triXC, triYC, triZC)
        min_quality = minimum([q.jacobian_quality for q in qualities])
        max_aspect = maximum([q.aspect_ratio for q in qualities])
        
        if min_quality < quality_threshold || max_aspect > ar_max
            # Store circulation before remeshing
            nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
            
            # Apply flow-adaptive remeshing
            velocity_field(x, y, z) = begin
                # Simple velocity approximation for remeshing guidance
                u, v, w = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr)
                # Find nearest node (simplified)
                return (0.0, 0.0, 0.0)  # Placeholder
            end
            
            # Thesis-style thresholds: aspect ratio, angle/jacobian quality,
            # Frobenius norm grad threshold, and curvature (dihedral angle in radians)
            tri_new, changed = flow_adaptive_remesh!(
                nodeX, nodeY, nodeZ, tri, velocity_field, dom;
                max_aspect_ratio=3.0,
                max_skewness=0.8,
                min_angle_quality=0.4,
                min_jacobian_quality=0.4,
                grad_threshold=0.2,         # ||∇U||_F
                curvature_threshold=0.6,    # radians (~34°)
                max_elements=10000
            )
            
            if changed
                tri = tri_new
                nt = size(tri, 1)
                
                # Rebuild triangle coordinates
                triXC = Array{Float64}(undef, nt, 3)
                triYC = similar(triXC)
                triZC = similar(triXC)
                @inbounds for k in 1:3, t in 1:nt
                    v = tri[t,k]
                    triXC[t,k] = nodeX[v]
                    triYC[t,k] = nodeY[v]
                    triZC[t,k] = nodeZ[v]
                end
                
                # Reconstruct vorticity from conserved circulation
                eleGma = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
                
                # Update vortex sheet
                vortex_sheet = VortexSheet(nodeX, nodeY, nodeZ, tri, eleGma)
                
                if rank == 0
                    println("  Remeshing applied: $(nt) elements (quality=$(Printf.@sprintf("%.3f", min_quality)), AR=$(Printf.@sprintf("%.2f", max_aspect)))")
                end
            end
        end
    end
    
    # Vortex sheet tracking and analysis
    if it % 5 == 0
        # Track sheet interface evolution
        adaptive_sheet_tracking!(vortex_sheet, 
            (x,y,z) -> node_velocities(eleGma, triXC, triYC, triZC, [x], [y], [z], dom, gr),
            dt_used, dom; curvature_threshold=curvature_threshold, quality_threshold=quality_threshold)
        
        # Detect rollup regions
        rollup_regions = detect_sheet_rollup(vortex_sheet; vorticity_threshold=1.5)
        num_rollup = sum(rollup_regions)
        
        # Compute sheet curvature
        curvatures = compute_sheet_curvature(vortex_sheet)
        max_curvature = maximum(curvatures)
        
        if rank == 0 && it % 10 == 0
            println("step $it: time=$(Printf.@sprintf("%.4f", time))")
            println("  Elements: $nt, Rollup regions: $num_rollup")
            println("  Max curvature: $(Printf.@sprintf("%.3f", max_curvature))")
            println("  x-range: [$(Printf.@sprintf("%.3f", minimum(nodeX))), $(Printf.@sprintf("%.3f", maximum(nodeX)))]")
            println("  y-range: [$(Printf.@sprintf("%.3f", minimum(nodeY))), $(Printf.@sprintf("%.3f", maximum(nodeY)))]")
            println("  z-range: [$(Printf.@sprintf("%.3f", minimum(nodeZ))), $(Printf.@sprintf("%.3f", maximum(nodeZ)))]")
        end
    end
    
    # Enhanced diagnostics and saving
    if rank == 0 && time >= next_save_t
        # Compute advanced diagnostics
        KE = gamma_ke(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=:spectral)
        
        # Compute effective dissipation rate
        eddy_viscosity = compute_eddy_viscosity(dissipation_model, eleGma, triXC, triYC, triZC, dom, gr)
        mean_eddy_visc = mean(eddy_viscosity)
        
        # Mesh quality statistics
        qualities = compute_mesh_quality(triXC, triYC, triZC)
        quality_stats = (
            min_jacobian = minimum([q.jacobian_quality for q in qualities]),
            mean_aspect = mean([q.aspect_ratio for q in qualities]),
            max_skewness = maximum([q.skewness for q in qualities])
        )
        
        # Vortex sheet statistics
        curvatures = compute_sheet_curvature(vortex_sheet)
        sheet_stats = (
            max_curvature = maximum(curvatures),
            mean_curvature = mean(curvatures),
            interface_nodes = sum(vortex_sheet.interface_markers),
            mean_age = mean(vortex_sheet.age)
        )
        
        # Save comprehensive state
        save_state_timeseries!(series_file, time, nodeX, nodeY, nodeZ, tri, eleGma;
                              dom=dom, grid=gr, dt=dt_used, CFL=0.4, adaptive=true,
                              poisson_mode=:spectral, remesh_every=remesh_every, 
                              save_interval=save_interval, ar_max=ar_max, step=it,
                              params_extra=(; 
                                  Atg=Atg, Nx=Nx, Ny=Ny, KE=KE,
                                  kernel_type=string(typeof(kernel_type)),
                                  dissipation_model=string(typeof(dissipation_model)),
                                  poisson_solver=string(typeof(poisson_solver)),
                                  mean_eddy_visc=mean_eddy_visc,
                                  quality_stats=quality_stats,
                                  sheet_stats=sheet_stats
                              ))
        
        println("Advanced checkpoint saved (t=$(round(time,digits=4))): $series_file")
        println("  KE: $(Printf.@sprintf("%.6f", KE)), Mean ν_sgs: $(Printf.@sprintf("%.2e", mean_eddy_visc))")
        println("  Quality - min: $(Printf.@sprintf("%.3f", quality_stats.min_jacobian)), mean AR: $(Printf.@sprintf("%.2f", quality_stats.mean_aspect))")
        println("  Sheet - max κ: $(Printf.@sprintf("%.3f", sheet_stats.max_curvature)), interface nodes: $(sheet_stats.interface_nodes)")
        
        next_save_t += save_interval
    end
end

if rank == 0
    println("="^60)
    println("ADVANCED SIMULATION COMPLETED")
    println("Final time: $(round(time, digits=4))")
    println("Final elements: $nt")
    println("="^60)
end

finalize_mpi!()
