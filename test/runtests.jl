using VortexMethod
using Test

# Clean up any leftover test files before starting
test_files_to_clean = [
    "checkpoints/test_series.jld2",
    "checkpoints/test_series_new.jld2", 
]

for file in test_files_to_clean
    isfile(file) && rm(file)
end

@testset "VortexMethod.jl" begin
    include("test_domain.jl")
    include("test_poisson.jl")
    include("test_remesh.jl")
    include("test_checkpoint.jl")
    include("test_particle_management.jl")
end
