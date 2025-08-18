using VortexMethod
using Test

@testset "VortexMethod.jl" begin
    include("test_domain.jl")
    include("test_poisson.jl")
    include("test_remesh.jl")
    include("test_checkpoint.jl")
end
