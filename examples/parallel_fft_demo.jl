#!/usr/bin/env julia

"""
Demonstration of parallel FFT using PencilFFTs vs serial FFT with FFTW

This example shows how to use the new parallel_fft=true option to enable
PencilFFTs for distributed FFT computations across MPI ranks.

Usage:
    # Serial FFT (original)
    julia parallel_fft_demo.jl
    
    # Parallel FFT with 4 MPI ranks
    mpirun -n 4 julia parallel_fft_demo.jl --parallel-fft

Requirements:
    - PencilFFTs.jl package installed
    - MPI configured properly
"""

using VortexMethod
using MPI

function main()
    # Parse command line arguments
    parallel_fft = "--parallel-fft" in ARGS
    
    # Initialize MPI
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    nprocs = MPI.Comm_size(comm)
    
    if rank == 0
        if parallel_fft
            println("Running with PencilFFTs (parallel) on $nprocs ranks")
        else
            println("Running with FFTW (serial FFT, rank 0 only) on $nprocs ranks")
        end
    end
    
    # Setup domain and grid
    dom = default_domain()
    gr = GridSpec(32, 32, 63)  # Small grid for demo
    
    # Create simple test mesh - single vortex ring
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = structured_mesh(8, 8; dom=dom)
    
    # Create element vorticity (simple vortex ring pattern)
    nt = size(tri, 1)
    eleGma = zeros(Float64, nt, 3)
    for t in 1:nt
        x_center = sum(triXC[t, :]) / 3
        y_center = sum(triYC[t, :]) / 3
        z_center = sum(triZC[t, :]) / 3
        
        # Vortex ring strength based on distance from center
        r = sqrt((x_center - 0.5)^2 + (y_center - 0.5)^2)
        if 0.2 < r < 0.4
            eleGma[t, 3] = 0.1 * exp(-10 * (r - 0.3)^2)  # z-component vorticity
        end
    end
    
    # Time the velocity computation
    if rank == 0
        println("Computing node velocities...")
        t_start = time()
    end
    
    # Compute velocities using either serial or parallel FFT
    u, v, w = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr; 
                             poisson_mode=:spectral, parallel_fft=parallel_fft)
    
    if rank == 0
        t_end = time()
        println("Computation completed in $(t_end - t_start) seconds")
        
        # Print some statistics
        u_max = maximum(abs.(u))
        v_max = maximum(abs.(v))
        w_max = maximum(abs.(w))
        
        println("Max velocities: u=$u_max, v=$v_max, w=$w_max")
        
        if parallel_fft
            println("✓ Used PencilFFTs for distributed parallel FFT")
        else
            println("✓ Used FFTW with rank-0 computation and broadcast")
        end
    end
    
    # Finalize MPI
    finalize_mpi!()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end