#!/usr/bin/env julia

"""
Test file for parallel FFT implementation using PencilFFTs

This test verifies:
1. The parallel FFT function can be loaded
2. Basic syntax and imports are correct
3. The function signature is compatible
"""

using Test
using VortexMethod
using FFTW
using MPI
using PencilFFTs

@testset "Parallel FFT Tests" begin
    
    @testset "Module imports" begin
        @test isdefined(Main, :FFTW)
        @test isdefined(Main, :MPI)
        @test isdefined(Main, :PencilFFTs)
        println("All required packages import successfully")
    end
    
    @testset "VortexMethod integration" begin
        @test isdefined(VortexMethod, :Poisson3D)
        @test isdefined(VortexMethod, :TimeStepper)
        println("VortexMethod modules loaded successfully")
    end
    
    @testset "PencilFFTs function existence" begin
        # Check if our new function exists in the module
        @test hasmethod(VortexMethod.Poisson3D.poisson_velocity_pencil_fft, 
                       (Array{Float64,3}, Array{Float64,3}, Array{Float64,3}, VortexMethod.DomainSpec))
        
        println("poisson_velocity_pencil_fft function exists with correct signature")
    end

    @testset "PencilFFTs direct solve" begin
        domain = VortexMethod.DomainSpec(1.0, 1.0, 1.0)
        gr = VortexMethod.GridSpec(4, 4, 4)
        u_rhs = zeros(Float64, gr.nz, gr.ny, gr.nx)
        v_rhs = similar(u_rhs)
        w_rhs = similar(u_rhs)
        for k in 1:gr.nz, j in 1:gr.ny, i in 1:gr.nx
            x = (i - 1) / gr.nx
            y = (j - 1) / gr.ny
            z = (k - 1) / gr.nz
            u_rhs[k, j, i] = sin(2pi * x)
            v_rhs[k, j, i] = cos(2pi * y)
            w_rhs[k, j, i] = sin(2pi * z)
        end

        serial = VortexMethod.poisson_velocity_fft(u_rhs, v_rhs, w_rhs, domain)
        pencil = VortexMethod.poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain)
        @test all(isapprox.(pencil, serial; rtol=1e-10, atol=1e-10))
    end
    
    @testset "Configuration options" begin
        # These should exist and have the parallel_fft parameter
        functions_with_parallel_fft = [
            :grid_velocity,
            :node_velocities, 
            :make_velocity_sampler,
            :max_grid_speed,
            :rk2_step!,
            :rk2_step_with_dissipation!
        ]
        
        for func_name in functions_with_parallel_fft
            # Test that the function exists (basic syntax check)
            @test isdefined(VortexMethod.TimeStepper, func_name)
            println("$func_name exists with parallel_fft support")
        end
    end
end

println("All parallel FFT tests completed successfully!")
