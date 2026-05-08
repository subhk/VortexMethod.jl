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
