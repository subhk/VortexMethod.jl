#!/usr/bin/env julia

"""
Enhanced Kelvin-Helmholtz 3D Instability with Parallel FFT Support

This example demonstrates:
1. PencilFFTs parallel FFT computation for large-scale simulations
2. Performance monitoring and timing comparisons
3. Configurable solver options and adaptive parameters
4. Advanced checkpointing and visualization capabilities
5. Command-line parameter control

Usage:
    # Serial FFT (original)
    julia kh3d_parallel.jl
    
    # Parallel FFT with 4 MPI ranks  
    mpirun -n 4 julia kh3d_parallel.jl --parallel-fft
    
    # High resolution with custom parameters
    mpirun -n 8 julia kh3d_parallel.jl --parallel-fft --nx=128 --ny=128 --steps=100
    
    # Performance comparison mode
    mpirun -n 4 julia kh3d_parallel.jl --compare-performance

Command-line options:
    --parallel-fft          Use PencilFFTs for distributed FFT computation
    --compare-performance   Run both serial and parallel FFT for timing comparison
    --nx=N                  Mesh resolution in x direction (default: 64)
    --ny=N                  Mesh resolution in y direction (default: 64)  
    --steps=N               Number of time steps (default: 50)
    --dt=X                  Time step size (default: 1e-3)
    --save-interval=X       Save interval in physical time (default: 0.1)
    --poisson-mode=MODE     Poisson mode: spectral or fd (default: fd)
    --output-dir=DIR        Output directory (default: checkpoints)
"""

using VortexMethod
using MPI
using Printf
using Dates

# Parse command line arguments
function parse_args()
    args = Dict{String,Any}(
        "parallel_fft" => false,
        "compare_performance" => false,
        "nx" => 64,
        "ny" => 64,
        "steps" => 50,
        "dt" => 1e-3,
        "save_interval" => 0.1,
        "poisson_mode" => :fd,
        "output_dir" => "checkpoints"
    )
    
    for arg in ARGS
        if arg == "--parallel-fft"
            args["parallel_fft"] = true
        elseif arg == "--compare-performance"
            args["compare_performance"] = true
        elseif startswith(arg, "--nx=")
            args["nx"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--ny=")
            args["ny"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--steps=")
            args["steps"] = parse(Int, split(arg, "=")[2])
        elseif startswith(arg, "--dt=")
            args["dt"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--save-interval=")
            args["save_interval"] = parse(Float64, split(arg, "=")[2])
        elseif startswith(arg, "--poisson-mode=")
            mode_str = split(arg, "=")[2]
            args["poisson_mode"] = mode_str == "spectral" ? :spectral : :fd
        elseif startswith(arg, "--output-dir=")
            args["output_dir"] = split(arg, "=")[2]
        end
    end
    
    return args
end

# Performance monitoring structure
mutable struct PerformanceMonitor
    fft_times::Vector{Float64}
    remesh_times::Vector{Float64}
    total_step_times::Vector{Float64}
    energy_times::Vector{Float64}
    start_time::Float64
    
    PerformanceMonitor() = new(Float64[], Float64[], Float64[], Float64[], time())
end

function log_timing!(monitor::PerformanceMonitor, category::Symbol, elapsed_time::Float64)
    if category == :fft
        push!(monitor.fft_times, elapsed_time)
    elseif category == :remesh
        push!(monitor.remesh_times, elapsed_time)
    elseif category == :step
        push!(monitor.total_step_times, elapsed_time)
    elseif category == :energy
        push!(monitor.energy_times, elapsed_time)
    end
end

function print_performance_summary(monitor::PerformanceMonitor, rank::Int, nprocs::Int, parallel_fft::Bool)
    if rank != 0
        return
    end
    
    total_time = time() - monitor.start_time
    fft_mode = parallel_fft ? "PencilFFTs (parallel)" : "FFTW (serial)"
    
    println("\n" * "="^60)
    println("PERFORMANCE SUMMARY")
    println("="^60)
    println("Total simulation time: $(round(total_time, digits=2)) seconds")
    println("MPI ranks: $nprocs")
    println("FFT mode: $fft_mode")
    println()
    
    if !isempty(monitor.total_step_times)
        println("Time per step:")
        println("  Average: $(round(mean(monitor.total_step_times), digits=4)) s")
        println("  Min:     $(round(minimum(monitor.total_step_times), digits=4)) s")
        println("  Max:     $(round(maximum(monitor.total_step_times), digits=4)) s")
    end
    
    if !isempty(monitor.fft_times)
        println("FFT/Poisson solve time:")
        println("  Average: $(round(mean(monitor.fft_times), digits=4)) s")
        println("  Total:   $(round(sum(monitor.fft_times), digits=2)) s ($(round(100*sum(monitor.fft_times)/total_time, digits=1))%)")
    end
    
    if !isempty(monitor.remesh_times)
        println("Remeshing time:")
        println("  Average: $(round(mean(monitor.remesh_times), digits=4)) s")
        println("  Total:   $(round(sum(monitor.remesh_times), digits=2)) s ($(round(100*sum(monitor.remesh_times)/total_time, digits=1))%)")
    end
    
    if !isempty(monitor.energy_times)
        println("Energy computation time:")
        println("  Average: $(round(mean(monitor.energy_times), digits=4)) s")
        println("  Total:   $(round(sum(monitor.energy_times), digits=2)) s ($(round(100*sum(monitor.energy_times)/total_time, digits=1))%)")
    end
    
    println("="^60)
end

function run_kh_simulation(args::Dict, parallel_fft::Bool, label::String="")
    # Initialize MPI
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    monitor = PerformanceMonitor()
    
    if rank == 0
        if !isempty(label)
            println("\n" * "="^60)
            println("RUNNING: $label")
            println("="^60)
        end
        println("Enhanced KH 3D with $(parallel_fft ? "Parallel" : "Serial") FFT")
        println("Nx=$(args["nx"]) Ny=$(args["ny"]) steps=$(args["steps"]) dt=$(args["dt"])")
        println("MPI ranks: $nprocs")
        println("Poisson mode: $(args["poisson_mode"])")
        if parallel_fft
            println("Using PencilFFTs for distributed FFT computation")
        else
            println("Using FFTW with rank-0 computation and broadcast")
        end
        println()
    end
    
    # Setup domain and grid
    dom = default_domain()
    gr = default_grid()
    
    # Create mesh
    Nx, Ny = args["nx"], args["ny"]
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(Nx, Ny; dom=dom)
    
    # Initialize vorticity with Kelvin-Helmholtz profile
    nt = size(tri, 1)
    eleGma = zeros(Float64, nt, 3)
    
    # Enhanced KH initial condition with perturbation
    for t in 1:nt
        x_center = sum(triXC[t, :]) / 3
        y_center = sum(triYC[t, :]) / 3
        
        # Base shear layer
        if 0.3 < y_center < 0.7
            eleGma[t, 1] = 2.0 * tanh(10 * (y_center - 0.5))  # x-component
        end
        
        # Add small perturbation to trigger instability  
        perturbation = 0.1 * sin(4π * x_center) * exp(-20 * (y_center - 0.5)^2)
        eleGma[t, 2] = perturbation  # y-component
    end
    
    # Simulation parameters
    dt = args["dt"]
    nsteps = args["steps"]
    save_interval = args["save_interval"]
    poisson_mode = args["poisson_mode"]
    
    # Enhanced controls
    Atg = 0.0                    # Atwood*gravity for baroclinic source
    remesh_every = 1             # remesh every step for KH instability
    ar_max = 4.0                 # aspect ratio threshold
    ke_stride = 5                # compute KE every Nth save
    save_series = true           # use JLD2 series format
    
    # Output setup
    output_dir = args["output_dir"]
    mkpath(output_dir)
    series_file = joinpath(output_dir, "kh3d_$(parallel_fft ? "parallel" : "serial")_series.jld2")
    
    time = 0.0
    next_save_t = save_interval
    
    # Main simulation loop
    for it in 1:nsteps
        step_start = time()
        
        # Time stepping with performance monitoring
        fft_start = time()
        dt_used = rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom, gr, dt; 
                           At=Atg, adaptive=true, CFL=0.5, 
                           poisson_mode=poisson_mode, parallel_fft=parallel_fft)
        fft_time = time() - fft_start
        log_timing!(monitor, :fft, fft_time)
        
        time += dt_used
        
        # Update triangle coordinates
        @inbounds for k in 1:3, t in 1:size(tri,1)
            v = tri[t,k]
            triXC[t,k] = nodeX[v]
            triYC[t,k] = nodeY[v]
            triZC[t,k] = nodeZ[v]
        end
        
        # Mesh quality diagnostics
        dx, dy, dz = grid_spacing(dom, gr)
        ds_max = 0.80 * max(dx, dy)
        ds_min = 0.05 * max(dx, dy)
        
        # Remeshing with performance monitoring
        if it % remesh_every == 0
            remesh_start = time()
            nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
            tri, changed = VortexMethod.Remesh.remesh_pass!(nodeX, nodeY, nodeZ, tri, ds_max, ds_min; 
                                                           dom=dom, ar_max=ar_max)
            if changed
                nt = size(tri, 1)
                triXC = Array{Float64}(undef, nt, 3)
                triYC = similar(triXC)
                triZC = similar(triXC)
                @inbounds for k in 1:3, t in 1:nt
                    v = tri[t,k]
                    triXC[t,k] = nodeX[v]
                    triYC[t,k] = nodeY[v] 
                    triZC[t,k] = nodeZ[v]
                end
                eleGma = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
            end
            remesh_time = time() - remesh_start
            log_timing!(monitor, :remesh, remesh_time)
        end
        
        step_time = time() - step_start
        log_timing!(monitor, :step, step_time)
        
        # Progress reporting
        if rank == 0 && it % 10 == 0
            println("Step $it: t=$(round(time, digits=4)), dt=$(round(dt_used, digits=6)), " *
                   "step_time=$(round(step_time, digits=3))s, " *
                   "fft_time=$(round(fft_time, digits=3))s")
        end
        
        # Saving and energy computation
        if rank == 0 && (time >= next_save_t)
            save_count = Int(floor(time / save_interval))
            KE = nothing
            
            if save_count % ke_stride == 0
                energy_start = time()
                KE = gamma_ke(eleGma, triXC, triYC, triZC, dom, gr; 
                             poisson_mode=poisson_mode, parallel_fft=parallel_fft)
                energy_time = time() - energy_start
                log_timing!(monitor, :energy, energy_time)
            end
            
            # Enhanced metadata
            params_extra = (
                Atg=Atg, Nx=Nx, Ny=Ny, KE=KE,
                parallel_fft=parallel_fft,
                fft_time=fft_time,
                step_time=step_time,
                nprocs=nprocs,
                julia_version=string(VERSION),
                timestamp=string(now())
            )
            
            base = save_state_timeseries!(series_file, time, nodeX, nodeY, nodeZ, tri, eleGma;
                                         dom=dom, grid=gr, dt=dt_used, CFL=0.5, adaptive=true,
                                         poisson_mode=poisson_mode, remesh_every=remesh_every, 
                                         save_interval=save_interval, ar_max=ar_max, step=it,
                                         params_extra=params_extra)
            
            println("  Checkpoint saved (t=$(round(time,digits=4))): $series_file")
            if KE !== nothing
                println("  Kinetic Energy: $(round(KE, digits=6))")
            end
            next_save_t += save_interval
        end
    end
    
    # Performance summary
    print_performance_summary(monitor, rank, nprocs, parallel_fft)
    
    finalize_mpi!()
    return monitor
end

function main()
    args = parse_args()
    
    if args["compare_performance"]
        # Run both serial and parallel for comparison
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println("Performance comparison mode: running both serial and parallel FFT")
        end
        
        # Serial run
        monitor_serial = run_kh_simulation(args, false, "SERIAL FFT BENCHMARK")
        
        # Small delay between runs
        sleep(2)
        
        # Parallel run  
        monitor_parallel = run_kh_simulation(args, true, "PARALLEL FFT BENCHMARK")
        
        # Comparison summary
        if MPI.Comm_rank(MPI.COMM_WORLD) == 0
            println("\n" * "="^60)
            println("PERFORMANCE COMPARISON")
            println("="^60)
            if !isempty(monitor_serial.fft_times) && !isempty(monitor_parallel.fft_times)
                serial_avg = mean(monitor_serial.fft_times)
                parallel_avg = mean(monitor_parallel.fft_times)
                speedup = serial_avg / parallel_avg
                println("Average FFT time:")
                println("  Serial:   $(round(serial_avg, digits=4)) s")
                println("  Parallel: $(round(parallel_avg, digits=4)) s")
                println("  Speedup:  $(round(speedup, digits=2))x")
            end
            println("="^60)
        end
    else
        # Single run
        run_kh_simulation(args, args["parallel_fft"])
    end
end

# Helper function for statistics (simple implementation)
mean(x) = sum(x) / length(x)

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end