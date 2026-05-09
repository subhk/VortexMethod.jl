@testset "VortexMethod high-level interface" begin
    grid = VortexMethod.RectilinearGrid(
        size=(6, 6, 5),
        x=(0.0, 1.0),
        y=(0.0, 1.0),
        z=(-1.0, 1.0),
        topology=(VortexMethod.Periodic, VortexMethod.Periodic, VortexMethod.Periodic),
    )

    @test grid.domain == VortexMethod.DomainSpec(1.0, 1.0, 1.0)
    @test grid.grid == VortexMethod.GridSpec(6, 6, 5)
    @test VortexMethod.RectilinearGrid isa UnionAll
    if VortexMethod.RectilinearGrid isa UnionAll
        @test grid isa VortexMethod.RectilinearGrid{Float64}
    end

    model = VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        Γ=(0.0, 1.0, 0.0),
        amp=0.0,
    )

    @test model.grid === grid
    @test VortexMethod.VortexSheetModel isa UnionAll
    if VortexMethod.VortexSheetModel isa UnionAll
        @test model isa VortexMethod.VortexSheetModel{Float64, Float64}
    end
    @test hasproperty(model, :workspace)
    if hasproperty(model, :workspace)
        @test model.workspace isa VortexMethod.VortexWorkspace{Float64}
    end
    @test model.clock.iteration == 0
    @test model.clock.time == 0.0
    @test size(model.eleGma, 1) == size(model.tri, 1)
    @test all(model.eleGma[:, 1] .== 0.0)
    @test all(model.eleGma[:, 2] .== 1.0)
    @test all(model.eleGma[:, 3] .== 0.0)

    VortexMethod.set!(model; Γ=(1.0, 2.0, 3.0))
    @test all(model.eleGma[:, 1] .== 1.0)
    @test all(model.eleGma[:, 2] .== 2.0)
    @test all(model.eleGma[:, 3] .== 3.0)

    Γ = fill(0.25, size(model.eleGma))
    VortexMethod.set!(model; circulation=Γ)
    @test model.eleGma == Γ

    model_with_circulation_alias = VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        circulation=(0.0, 0.5, 0.0),
        amp=0.0,
    )
    @test all(model_with_circulation_alias.eleGma[:, 2] .== 0.5)

    model_with_vector_at = VortexMethod.VortexSheetModel(; grid, sheet_size=(4, 4), At=zeros(size(model.tri, 1)))
    if VortexMethod.VortexSheetModel isa UnionAll
        @test model_with_vector_at isa VortexMethod.VortexSheetModel{Float64, Vector{Float64}}
    end
    @test model_with_vector_at.At isa Vector{Float64}

    simulation = VortexMethod.Simulation(model; Δt=0.0, stop_iteration=1)
    VortexMethod.run!(simulation)

    @test simulation.model === model
    @test model.clock.iteration == 1
    @test model.clock.time == 0.0
    @test model.clock.last_Δt == 0.0
end

@testset "No external interface naming in repository text" begin
    repo_root = dirname(@__DIR__)
    this_file = @__FILE__
    forbidden = "Ocean" * "anigans"
    for (root, _, files) in walkdir(repo_root)
        occursin(".git", root) && continue
        for file in files
            path = joinpath(root, file)
            path == this_file && continue
            isfile(path) || continue
            if splitext(file)[2] in (".jl", ".md")
                @test !occursin(forbidden, read(path, String))
            end
        end
    end
end
