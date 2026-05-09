using VortexMethod
using Test
using MPI

# Clean up any leftover test files before starting
test_files_to_clean = [
    "checkpoints/test_series.jld2",
    "checkpoints/test_series_new.jld2", 
]

for file in test_files_to_clean
    isfile(file) && rm(file)
end

@testset "VortexMethod.jl" begin
    include("test_interface.jl")
    include("test_domain.jl")
    include("test_poisson.jl")
    include("test_remesh.jl")
    include("test_checkpoint.jl")
    include("test_particle_management.jl")
    include("test_stock_regressions.jl")
    include("test_parallel_fft.jl")
    include("test_performance.jl")

    if get(ENV, "VORTEXMETHOD_RUN_MPI_TESTS", "false") == "true"
        mpi_test = joinpath(@__DIR__, "mpi_parallel_correctness.jl")
        repo_root = dirname(@__DIR__)
        cmd = `$(MPI.mpiexec()) -n 2 $(Base.julia_cmd()) --project=$repo_root $mpi_test`
        @testset "MPI parallel correctness launcher" begin
            @test success(cmd)
        end
    end
end
