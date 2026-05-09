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
    @test_throws ArgumentError VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        Γ=(0.0, 1.0, 0.0),
        circulation=(0.0, 0.5, 0.0),
        amp=0.0,
    )

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

@testset "Workspace-backed time_step! matches allocating RK2 path" begin
    domain = VortexMethod.default_domain()
    grid = VortexMethod.RectilinearGrid(size=(6, 6, 6))
    nodeX, nodeY, nodeZ, tri, _, _, _ =
        VortexMethod.structured_mesh(4, 4; domain=domain, amp=0.0)
    eleGma = fill(0.2, size(tri, 1), 3)
    expected_x = copy(nodeX)
    expected_y = copy(nodeY)
    expected_z = copy(nodeZ)
    expected_Γ = copy(eleGma)

    VortexMethod.rk2_step!(expected_x, expected_y, expected_z, tri, expected_Γ,
                           domain, grid.grid, 0.01)

    model = VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        Γ=(0.0, 0.2, 0.0),
        amp=0.0,
    )
    model.nodeX .= nodeX
    model.nodeY .= nodeY
    model.nodeZ .= nodeZ
    model.eleGma .= eleGma
    VortexMethod.time_step!(model, 0.01)

    @test model.nodeX ≈ expected_x
    @test model.nodeY ≈ expected_y
    @test model.nodeZ ≈ expected_z
    @test model.eleGma ≈ expected_Γ
end

@testset "Workspace-backed kernel and Smagorinsky branches match allocating paths" begin
    domain = VortexMethod.default_domain()
    grid = VortexMethod.RectilinearGrid(size=(6, 6, 6))
    nodeX, nodeY, nodeZ, tri, _, _, _ =
        VortexMethod.structured_mesh(4, 4; domain=domain, amp=0.0)
    eleGma = fill(0.2, size(tri, 1), 3)

    expected_x = copy(nodeX)
    expected_y = copy(nodeY)
    expected_z = copy(nodeZ)
    expected_Γ = copy(eleGma)
    VortexMethod.rk2_step_with_dissipation!(
        expected_x, expected_y, expected_z, tri, expected_Γ,
        domain, grid.grid, 0.001, VortexMethod.NoDissipation();
        kernel=VortexMethod.PeskinCosine(),
    )

    kernel_model = VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        Γ=(0.2, 0.2, 0.2),
        amp=0.0,
        kernel=VortexMethod.PeskinCosine(),
    )
    kernel_model.nodeX .= nodeX
    kernel_model.nodeY .= nodeY
    kernel_model.nodeZ .= nodeZ
    kernel_model.eleGma .= eleGma
    VortexMethod.time_step!(kernel_model, 0.001)

    @test kernel_model.nodeX ≈ expected_x
    @test kernel_model.nodeY ≈ expected_y
    @test kernel_model.nodeZ ≈ expected_z
    @test kernel_model.eleGma ≈ expected_Γ

    expected_x .= nodeX
    expected_y .= nodeY
    expected_z .= nodeZ
    expected_Γ .= eleGma
    VortexMethod.rk2_step_with_dissipation!(
        expected_x, expected_y, expected_z, tri, expected_Γ,
        domain, grid.grid, 0.001, VortexMethod.SmagorinskyModel(),
    )

    smag_model = VortexMethod.VortexSheetModel(;
        grid,
        sheet_size=(4, 4),
        Γ=(0.2, 0.2, 0.2),
        amp=0.0,
        dissipation=VortexMethod.SmagorinskyModel(),
    )
    smag_model.nodeX .= nodeX
    smag_model.nodeY .= nodeY
    smag_model.nodeZ .= nodeZ
    smag_model.eleGma .= eleGma
    VortexMethod.time_step!(smag_model, 0.001)

    @test smag_model.nodeX ≈ expected_x rtol=1e-10 atol=1e-12
    @test smag_model.nodeY ≈ expected_y rtol=1e-10 atol=1e-12
    @test smag_model.nodeZ ≈ expected_z rtol=1e-10 atol=1e-12
    @test smag_model.eleGma ≈ expected_Γ rtol=1e-10 atol=1e-12
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
