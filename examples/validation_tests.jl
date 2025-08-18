# Comprehensive validation test cases based on thesis
# Tests various components of the advanced vortex method implementation

using VortexMethod
using MPI
using Printf
using LinearAlgebra

init_mpi!()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)

function run_validation_tests()
    if rank == 0
        println("="^60)
        println("VORTEX METHOD VALIDATION TESTS")
        println("Based on thesis test cases and benchmarks")
        println("="^60)
    end
    
    # Test 1: Kernel comparison
    test_interpolation_kernels()
    
    # Test 2: Advanced remeshing
    test_advanced_remeshing()
    
    # Test 3: Dissipation models
    test_dissipation_models()
    
    # Test 4: Multiple Poisson solvers
    test_poisson_solvers()
    
    # Test 5: Vortex sheet evolution
    test_vortex_sheet_tracking()
    
    # Test 6: Taylor-Green vortex benchmark
    test_taylor_green_vortex()
    
    # Test 7: Lamb vortex benchmark
    test_lamb_vortex()
    
    if rank == 0
        println("="^60)
        println("VALIDATION TESTS COMPLETED")
        println("="^60)
    end
end

# Test 1: Compare different interpolation kernels
function test_interpolation_kernels()
    if rank == 0
        println("\n[Test 1] Interpolation Kernel Comparison")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, π)
    gr = GridSpec(32, 32, 16)
    
    # Create simple test mesh
    Nx, Ny = 16, 16
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
    
    # Initial vorticity (simple Gaussian)
    nt = size(tri, 1)
    eleGma = zeros(Float64, nt, 3)
    for t in 1:nt
        cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
        cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
        r2 = (cx - π)^2 + (cy - π)^2
        eleGma[t,3] = exp(-r2)  # z-component vorticity
    end
    
    # Test different kernels
    kernels = [PeskinStandard(), PeskinCosine(), M4Prime(), AreaWeighting()]
    kernel_names = ["PeskinStandard", "PeskinCosine", "M4Prime", "AreaWeighting"]
    
    errors = Float64[]
    for (i, kernel) in enumerate(kernels)
        VorX, VorY, VorZ = spread_vorticity_to_grid_kernel_mpi(eleGma, triXC, triYC, triZC, domain, gr, kernel)
        
        # Compute L2 error (simplified)
        total_vorticity = sum(VorZ) * (domain.Lx/gr.nx) * (domain.Ly/gr.ny) * (2*domain.Lz/gr.nz)
        expected_vorticity = sum(eleGma[:, 3])
        error = abs(total_vorticity - expected_vorticity) / abs(expected_vorticity)
        push!(errors, error)
        
        if rank == 0
            println("  $(kernel_names[i]): Relative error = $(Printf.@sprintf("%.2e", error))")
        end
    end
    
    if rank == 0
        best_kernel = kernel_names[argmin(errors)]
        println("  Best kernel: $best_kernel (lowest error)")
    end
end

# Test 2: Advanced remeshing quality metrics
function test_advanced_remeshing()
    if rank == 0
        println("\n[Test 2] Advanced Remeshing Quality Metrics")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, π)
    
    # Create highly distorted mesh
    Nx, Ny = 8, 8
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain, amp=0.5)
    
    # Apply additional distortion
    for i in 1:length(nodeX)
        nodeX[i] += 0.3 * sin(3*nodeX[i]) * cos(2*nodeY[i])
        nodeY[i] += 0.2 * cos(4*nodeX[i]) * sin(3*nodeY[i])
    end
    
    # Rebuild triangle coordinates
    for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    
    # Compute quality metrics before remeshing
    qualities_before = compute_mesh_quality(triXC, triYC, triZC)
    min_quality_before = minimum([q.jacobian_quality for q in qualities_before])
    mean_aspect_before = mean([q.aspect_ratio for q in qualities_before])
    
    # Apply advanced remeshing
    eleGma = zeros(Float64, size(tri,1), 3)
    tri_new, changed = VortexMethod.RemeshAdvanced.quality_split_triangle!(
        nodeX, nodeY, nodeZ, tri, 1, domain)
    
    if changed
        # Recompute triangle coordinates
        nt_new = size(tri_new, 1)
        triXC_new = Array{Float64}(undef, nt_new, 3)
        triYC_new = similar(triXC_new)
        triZC_new = similar(triXC_new)
        for k in 1:3, t in 1:nt_new
            v = tri_new[t,k]
            triXC_new[t,k] = nodeX[v]
            triYC_new[t,k] = nodeY[v]
            triZC_new[t,k] = nodeZ[v]
        end
        
        # Compute quality metrics after remeshing
        qualities_after = compute_mesh_quality(triXC_new, triYC_new, triZC_new)
        min_quality_after = minimum([q.jacobian_quality for q in qualities_after])
        mean_aspect_after = mean([q.aspect_ratio for q in qualities_after])
        
        if rank == 0
            println("  Elements before: $(size(tri,1)), after: $(nt_new)")
            println("  Min quality: $(Printf.@sprintf("%.3f", min_quality_before)) → $(Printf.@sprintf("%.3f", min_quality_after))")
            println("  Mean aspect ratio: $(Printf.@sprintf("%.3f", mean_aspect_before)) → $(Printf.@sprintf("%.3f", mean_aspect_after))")
        end
    end
end

# Test 3: Dissipation model effects
function test_dissipation_models()
    if rank == 0
        println("\n[Test 3] Dissipation Model Comparison")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, π)
    gr = GridSpec(16, 16, 8)
    
    # Create test mesh with strong vorticity
    Nx, Ny = 8, 8
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
    
    nt = size(tri, 1)
    eleGma_original = zeros(Float64, nt, 3)
    for t in 1:nt
        cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
        cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
        r2 = (cx - π)^2 + (cy - π)^2
        eleGma_original[t,3] = 10.0 * exp(-r2)  # Strong initial vorticity
    end
    
    # Test different dissipation models
    models = [NoDissipation(), SmagorinskyModel(0.17), VortexStretchingDissipation(0.1, 1.0)]
    model_names = ["No Dissipation", "Smagorinsky", "Vortex Stretching"]
    
    dt = 0.01
    for (i, model) in enumerate(models)
        eleGma_test = copy(eleGma_original)
        apply_dissipation!(model, eleGma_test, triXC, triYC, triZC, domain, gr, dt)
        
        energy_loss = sum(eleGma_original.^2) - sum(eleGma_test.^2)
        energy_loss_percent = 100 * energy_loss / sum(eleGma_original.^2)
        
        if rank == 0
            println("  $(model_names[i]): Energy loss = $(Printf.@sprintf("%.2f", energy_loss_percent))%")
        end
    end
end

# Test 4: Poisson solver comparison
function test_poisson_solvers()
    if rank == 0
        println("\n[Test 4] Poisson Solver Comparison")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, π)
    nx, ny, nz = 16, 16, 8
    
    # Create test RHS (known solution: sin(x)sin(y)sin(z))
    u_rhs = zeros(Float64, nz, ny, nx)
    v_rhs = zeros(Float64, nz, ny, nx)
    w_rhs = zeros(Float64, nz, ny, nx)
    
    dx = domain.Lx / (nx - 1)
    dy = domain.Ly / (ny - 1)
    dz = 2 * domain.Lz / (nz - 1)
    
    for k in 1:nz, j in 1:ny, i in 1:nx
        x = (i-1) * dx
        y = (j-1) * dy
        z = (k-1) * dz - domain.Lz
        
        # RHS for ∇²u = -3*sin(x)sin(y)sin(z)
        u_rhs[k,j,i] = -3 * sin(x) * sin(y) * sin(z)
    end
    
    # Test different solvers
    if rank == 0
        solvers = [FFTSolver(:spectral), FFTSolver(:fd)]
        solver_names = ["FFT Spectral", "FFT Finite Difference"]
        
        for (i, solver) in enumerate(solvers)
            start_time = time()
            ux, uy, uz = solve_poisson!(solver, u_rhs, v_rhs, w_rhs, domain)
            solve_time = time() - start_time
            
            # Compute error against analytical solution
            error = 0.0
            for k in 1:nz, j in 1:ny, i in 1:nx
                x = (i-1) * dx
                y = (j-1) * dy
                z = (k-1) * dz - domain.Lz
                analytical = sin(x) * sin(y) * sin(z)
                error += (ux[k,j,i] - analytical)^2
            end
            error = sqrt(error / (nx*ny*nz))
            
            println("  $(solver_names[i]): Error = $(Printf.@sprintf("%.2e", error)), Time = $(Printf.@sprintf("%.3f", solve_time))s")
        end
    end
end

# Test 5: Vortex sheet tracking
function test_vortex_sheet_tracking()
    if rank == 0
        println("\n[Test 5] Vortex Sheet Tracking")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, π)
    
    # Create simple sheet
    Nx, Ny = 8, 4
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
    
    # Initialize sheet
    eleGma = zeros(Float64, size(tri,1), 3)
    eleGma[:, 3] .= 1.0  # Uniform z-vorticity
    
    sheet = VortexSheet(nodeX, nodeY, nodeZ, tri, eleGma)
    
    # Simple velocity field (uniform translation)
    velocity_field(x, y, z) = (0.1, 0.05, 0.0)
    
    # Track evolution
    dt = 0.1
    n_steps = 5
    
    initial_center = mean(sheet.nodes, dims=1)
    
    for step in 1:n_steps
        evolve_sheet!(sheet, VortexSheets.ClassicalEvolution(), velocity_field, dt, domain)
    end
    
    final_center = mean(sheet.nodes, dims=1)
    displacement = norm(final_center - initial_center)
    expected_displacement = norm([0.1, 0.05, 0.0]) * dt * n_steps
    
    if rank == 0
        println("  Sheet displacement: $(Printf.@sprintf("%.3f", displacement))")
        println("  Expected displacement: $(Printf.@sprintf("%.3f", expected_displacement))")
        println("  Relative error: $(Printf.@sprintf("%.2f", abs(displacement - expected_displacement)/expected_displacement * 100))%")
    end
end

# Test 6: Taylor-Green vortex benchmark
function test_taylor_green_vortex()
    if rank == 0
        println("\n[Test 6] Taylor-Green Vortex Benchmark")
        println("-"^40)
    end
    
    domain = DomainSpec(2π, 2π, 2π)
    gr = GridSpec(16, 16, 16)
    
    # Create mesh
    Nx, Ny = 8, 8
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
    
    # Taylor-Green initial condition
    nt = size(tri, 1)
    eleGma = zeros(Float64, nt, 3)
    for t in 1:nt
        cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
        cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
        cz = (triZC[t,1] + triZC[t,2] + triZC[t,3]) / 3
        
        # Taylor-Green vorticity
        eleGma[t,1] = 2 * cos(cx) * sin(cy) * sin(cz)
        eleGma[t,2] = -sin(cx) * cos(cy) * sin(cz)
        eleGma[t,3] = 0.0
    end
    
    # Compute initial kinetic energy
    initial_ke = gamma_ke(eleGma, triXC, triYC, triZC, domain, gr)
    
    # Evolve for one time step
    dt = 0.01
    dissipation_model = SmagorinskyModel(0.17)
    
    rk2_step_with_dissipation!(nodeX, nodeY, nodeZ, tri, eleGma, domain, gr, dt, dissipation_model)
    
    # Recompute triangle coordinates
    for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    
    final_ke = gamma_ke(eleGma, triXC, triYC, triZC, domain, gr)
    energy_decay = (initial_ke - final_ke) / initial_ke
    
    if rank == 0
        println("  Initial kinetic energy: $(Printf.@sprintf("%.6f", initial_ke))")
        println("  Final kinetic energy: $(Printf.@sprintf("%.6f", final_ke))")
        println("  Energy decay rate: $(Printf.@sprintf("%.2f", energy_decay * 100))%")
    end
end

# Test 7: Lamb vortex benchmark
function test_lamb_vortex()
    if rank == 0
        println("\n[Test 7] Lamb Vortex Benchmark")
        println("-"^40)
    end
    
    domain = DomainSpec(4π, 4π, π)
    gr = GridSpec(32, 32, 8)
    
    # Create fine mesh around vortex core
    Nx, Ny = 16, 16
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; domain=domain)
    
    # Lamb vortex initial condition (Gaussian vorticity)
    nt = size(tri, 1)
    eleGma = zeros(Float64, nt, 3)
    Γ = 1.0  # Circulation
    σ = 0.5  # Core radius
    
    for t in 1:nt
        cx = (triXC[t,1] + triXC[t,2] + triXC[t,3]) / 3
        cy = (triYC[t,1] + triYC[t,2] + triYC[t,3]) / 3
        
        # Center vortex at domain center
        rx = cx - 2π
        ry = cy - 2π
        r2 = rx^2 + ry^2
        
        # Gaussian vorticity profile
        eleGma[t,3] = (Γ / (π * σ^2)) * exp(-r2 / σ^2)
    end
    
    # Check circulation conservation
    total_circulation = sum(eleGma[:, 3]) * mean([
        0.5 * norm(cross([triXC[t,2] - triXC[t,1], triYC[t,2] - triYC[t,1], triZC[t,2] - triZC[t,1]],
                         [triXC[t,3] - triXC[t,1], triYC[t,3] - triYC[t,1], triZC[t,3] - triZC[t,1]]))
        for t in 1:nt
    ])
    
    circulation_error = abs(total_circulation - Γ) / Γ
    
    if rank == 0
        println("  Target circulation: $(Printf.@sprintf("%.6f", Γ))")
        println("  Computed circulation: $(Printf.@sprintf("%.6f", total_circulation))")
        println("  Circulation error: $(Printf.@sprintf("%.2f", circulation_error * 100))%")
    end
end

# Run all tests
run_validation_tests()

finalize_mpi!()
