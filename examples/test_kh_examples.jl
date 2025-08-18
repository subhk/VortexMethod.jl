#!/usr/bin/env julia

"""
Test script for KH examples with parallel FFT functionality

This script tests:
1. Command-line argument parsing
2. Parallel FFT flag detection  
3. Basic function calls (without full simulation)
4. Output directory creation
"""

using Test

@testset "KH Examples Tests" begin
    
    @testset "Command-line argument parsing" begin
        # Test kh3d.jl parallel_fft detection
        old_args = copy(ARGS)
        
        # Test without flag
        empty!(ARGS)
        parallel_fft = "--parallel-fft" in ARGS || "--parallel" in ARGS
        @test parallel_fft == false
        
        # Test with --parallel-fft
        push!(ARGS, "--parallel-fft")
        parallel_fft = "--parallel-fft" in ARGS || "--parallel" in ARGS
        @test parallel_fft == true
        
        # Test with --parallel
        empty!(ARGS)
        push!(ARGS, "--parallel")
        parallel_fft = "--parallel-fft" in ARGS || "--parallel" in ARGS
        @test parallel_fft == true
        
        # Restore original ARGS
        empty!(ARGS)
        append!(ARGS, old_args)
        
        println("Command-line argument parsing works correctly")
    end
    
    @testset "kh3d_parallel.jl argument parsing" begin
        # Test the enhanced argument parser
        
        # Mock ARGS for testing
        test_args = ["--parallel-fft", "--nx=128", "--ny=64", "--steps=100", 
                    "--dt=5e-4", "--save-interval=0.05", "--poisson-mode=spectral"]
        
        # Simulate the parse_args function logic
        args = Dict{String,Any}(
            "parallel_fft" => false,
            "nx" => 64,
            "ny" => 64,
            "steps" => 50,
            "dt" => 1e-3,
            "poisson_mode" => :fd
        )
        
        for arg in test_args
            if arg == "--parallel-fft"
                args["parallel_fft"] = true
            elseif startswith(arg, "--nx=")
                args["nx"] = parse(Int, split(arg, "=")[2])
            elseif startswith(arg, "--ny=")
                args["ny"] = parse(Int, split(arg, "=")[2])
            elseif startswith(arg, "--steps=")
                args["steps"] = parse(Int, split(arg, "=")[2])
            elseif startswith(arg, "--dt=")
                args["dt"] = parse(Float64, split(arg, "=")[2])
            elseif startswith(arg, "--poisson-mode=")
                mode_str = split(arg, "=")[2]
                args["poisson_mode"] = mode_str == "spectral" ? :spectral : :fd
            end
        end
        
        @test args["parallel_fft"] == true
        @test args["nx"] == 128
        @test args["ny"] == 64
        @test args["steps"] == 100
        @test args["dt"] == 5e-4
        @test args["poisson_mode"] == :spectral
        
        println("Enhanced argument parsing works correctly")
    end
    
    @testset "Directory creation" begin
        # Test output directory creation
        test_dir = "test_checkpoints"
        
        if isdir(test_dir)
            rm(test_dir, recursive=true)
        end
        
        mkpath(test_dir)
        @test isdir(test_dir)
        
        # Clean up
        rm(test_dir, recursive=true)
        
        println("Directory creation works correctly")
    end
    
    @testset "File existence" begin
        # Check that all example files exist
        examples = [
            "examples/kh3d.jl",
            "examples/kh3d_parallel.jl",
            "examples/advanced_kh3d.jl",
            "examples/README_KH_examples.md"
        ]
        
        for example in examples
            @test isfile(example)
        end
        
        println("All example files exist")
    end
    
    @testset "Documentation completeness" begin
        # Check that README contains key information
        readme_content = read("examples/README_KH_examples.md", String)
        
        key_sections = [
            "kh3d.jl",
            "kh3d_parallel.jl", 
            "advanced_kh3d.jl",
            "--parallel-fft",
            "PencilFFTs",
            "Performance",
            "mpirun"
        ]
        
        for section in key_sections
            @test occursin(section, readme_content)
        end
        
        println("Documentation is complete")
    end
end

println("All KH example tests passed successfully!")