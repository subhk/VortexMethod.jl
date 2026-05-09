using StaticArrays

@testset "Type stability and allocations" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.GridSpec(6, 6, 6)
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = VortexMethod.structured_mesh(4, 4; domain=domain)
    eleGma = reshape(sin.(1:(size(tri, 1) * 3)), size(tri, 1), 3)
    geom = VortexMethod.compute_triangle_geometry(triXC, triYC, triZC)
    triC = VortexMethod.Peskin3D.triangle_centroids(triXC, triYC, triZC)
    subC = VortexMethod.Peskin3D.build_all_subcentroids(triXC, triYC, triZC)
    areas = VortexMethod.Peskin3D.triangle_areas(triXC, triYC, triZC)
    ds = VortexMethod.grid_spacing(domain, gr)
    coord = (0.2, 0.3, 0.0)

    @test @inferred(VortexMethod.Peskin3D.peskin_grid_sum(eleGma, triC, subC, coord, ds, areas; domain=domain)) isa NTuple{3,Float64}
    @test @inferred(VortexMethod.Peskin3D.peskin_grid_sum_kernel(eleGma, triC, subC, coord, ds, areas, VortexMethod.PeskinCosine(); domain=domain)) isa NTuple{3,Float64}

    VortexMethod.Peskin3D.peskin_grid_sum(eleGma, triC, subC, coord, ds, areas; domain=domain)
    GC.gc()
    @test @allocated(VortexMethod.Peskin3D.peskin_grid_sum(eleGma, triC, subC, coord, ds, areas; domain=domain)) < 1024

    VortexMethod.Circulation.node_circulation_from_ele_gamma(geom, eleGma)
    GC.gc()
    @test @allocated(VortexMethod.Circulation.node_circulation_from_ele_gamma(geom, eleGma)) < 10_000

    Ux = reshape(sin.(1:216), 6, 6, 6)
    Uy = reshape(cos.(1:216), 6, 6, 6)
    Uz = similar(Ux)
    fill!(Uz, 0.25)
    VortexMethod.grid_ke(Ux, Uy, Uz, domain, gr)
    GC.gc()
    @test @allocated(VortexMethod.grid_ke(Ux, Uy, Uz, domain, gr)) < 1024

    VortexMethod.grid_spacing(domain, gr)
    GC.gc()
    @test @allocated(VortexMethod.grid_spacing(domain, gr)) < 128

    velocity_field = (x, y, z) -> (x + y, y + z, z + x)
    remesh_nodeX, remesh_nodeY, remesh_nodeZ, remesh_tri, _, _, _ =
        VortexMethod.structured_mesh(8, 8; domain=domain)
    remesh_eleGma = zeros(size(remesh_tri, 1), 3)
    VortexMethod.RemeshAdvanced.anisotropic_remesh!(
        remesh_nodeX, remesh_nodeY, remesh_nodeZ, remesh_tri, remesh_eleGma,
        velocity_field, domain; refinement_threshold=1e9,
    )
    GC.gc()
    @test @allocated(VortexMethod.RemeshAdvanced.anisotropic_remesh!(
        remesh_nodeX, remesh_nodeY, remesh_nodeZ, remesh_tri, remesh_eleGma,
        velocity_field, domain; refinement_threshold=1e9,
    )) < 8_000

    split_nodeX, split_nodeY, split_nodeZ, split_tri, _, _, _ =
        VortexMethod.structured_mesh(8, 8; domain=domain)
    split_eleGma = ones(size(split_tri, 1), 3)
    VortexMethod.RemeshAdvanced.quality_based_remesh!(
        copy(split_nodeX), copy(split_nodeY), copy(split_nodeZ), copy(split_tri),
        split_eleGma, domain; max_aspect_ratio=0.1, max_elements=10_000,
    )
    GC.gc()
    @test @allocated(VortexMethod.RemeshAdvanced.quality_based_remesh!(
        copy(split_nodeX), copy(split_nodeY), copy(split_nodeZ), copy(split_tri),
        split_eleGma, domain; max_aspect_ratio=0.1, max_elements=10_000,
    )) < 100_000
end

@testset "Workspace spread/interpolation and SVector kernels" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.GridSpec(4, 4, 4)
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC =
        VortexMethod.structured_mesh(3, 3; domain=domain, amp=0.0)
    eleGma = fill(0.2, size(tri, 1), 3)
    ws = VortexMethod.VortexWorkspace(Float64, length(nodeX), size(tri, 1), gr.nx, gr.ny, gr.nz, domain)

    ζx, ζy, ζz = VortexMethod.spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    VortexMethod.spread_vorticity_to_grid_mpi!(ws, eleGma, triXC, triYC, triZC, domain, gr)
    @test ws.geom_dirty[] == false
    @test ws.ζx ≈ ζx
    @test ws.ζy ≈ ζy
    @test ws.ζz ≈ ζz

    ws.geom_dirty[] = true
    VortexMethod.spread_vorticity_to_grid_mpi!(ws, eleGma, triXC, triYC, triZC, domain, gr)
    GC.gc()
    ws.geom_dirty[] = true
    @test @allocated(VortexMethod.spread_vorticity_to_grid_mpi!(ws, eleGma, triXC, triYC, triZC, domain, gr)) < 1024

    old_triC = copy(ws.triC)
    ws.geom_dirty[] = false
    VortexMethod.spread_vorticity_to_grid_mpi!(ws, eleGma, triXC .+ 0.125, triYC, triZC, domain, gr)
    @test ws.triC == old_triC

    Ux = reshape(Float64.(1:(gr.nx * gr.ny * gr.nz)), gr.nz, gr.ny, gr.nx)
    Uy = fill(0.25, gr.nz, gr.ny, gr.nx)
    Uz = fill(-0.5, gr.nz, gr.ny, gr.nx)
    u, v, w = VortexMethod.interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)
    VortexMethod.interpolate_node_velocity_mpi!(ws, Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)
    @test ws.u1 ≈ collect(u)
    @test ws.v1 ≈ collect(v)
    @test ws.w1 ≈ collect(w)
    GC.gc()
    @test @allocated(VortexMethod.interpolate_node_velocity_mpi!(ws, Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)) < 1024

    p1 = SVector{3,Float64}(0.1, 0.2, 0.3)
    p2 = SVector{3,Float64}(0.4, 0.5, 0.1)
    p3 = SVector{3,Float64}(0.2, 0.7, 0.4)
    @test VortexMethod.Peskin3D.bary_point(p1, p2, p3, 1, 1, 2) isa SVector{3,Float64}
    @test VortexMethod.Peskin3D.centroid3(p1, p2, p3) isa SVector{3,Float64}

    weights = zeros(Float32, 2)
    dx = Float32[0.0, 0.25]
    dy = Float32[0.0, 0.25]
    dz = Float32[0.0, 0.25]
    VortexMethod.kernel_function_vec!(weights, VortexMethod.PeskinStandard(), dx, dy, dz, Float32(1), Float32(1), Float32(1))
    @test all(isfinite, weights)

    triC32 = Float32.(VortexMethod.Peskin3D.triangle_centroids(triXC, triYC, triZC; domain=domain))
    subC32 = Float32.(VortexMethod.Peskin3D.build_all_subcentroids(triXC, triYC, triZC; domain=domain, subsegments=2))
    areas32 = Float32.(VortexMethod.Peskin3D.triangle_areas(triXC, triYC, triZC; domain=domain))
    eleGma32 = Float32.(eleGma)
    acc32 = SVector{3,Float32}(0, 0, 0)
    coord32 = SVector{3,Float32}(0.2, 0.3, 0.0)
    eps32 = SVector{3,Float32}(0.5, 0.5, 0.5)
    shift32 = SVector{3,Float32}(0, 0, 0)
    result32 = VortexMethod.Peskin3D.peskin_add_nearby_kernel!(
        acc32, eleGma32, triC32, subC32, areas32, coord32,
        VortexMethod.PeskinStandard(), eps32, shift32,
    )
    @test result32 isa SVector{3,Float32}
end

@testset "Vortex sheet tracking allocations" begin
    domain = VortexMethod.default_domain()
    nodeX, nodeY, nodeZ, tri, _, _, _ =
        VortexMethod.structured_mesh(20, 20; domain=domain, amp=0.0)
    eleGma = ones(size(tri, 1), 3)
    velocity_field = (x, y, z) -> (0.1, 0.2, 0.3)

    function make_sheet()
        return VortexMethod.VortexSheet(copy(nodeX), copy(nodeY), copy(nodeZ), copy(tri), copy(eleGma))
    end

    sheet = make_sheet()
    VortexMethod.VortexSheets.evolve_sheet_rk2!(sheet, velocity_field, 0.01, domain)
    GC.gc()
    @test @allocated(VortexMethod.VortexSheets.evolve_sheet_rk2!(sheet, velocity_field, 0.01, domain)) < 5_000

    sheet = make_sheet()
    VortexMethod.VortexSheets.evolve_sheet_rk4!(sheet, velocity_field, 0.01, domain)
    GC.gc()
    @test @allocated(VortexMethod.VortexSheets.evolve_sheet_rk4!(sheet, velocity_field, 0.01, domain)) < 5_000

    sheet = make_sheet()
    VortexMethod.compute_sheet_curvature(sheet)
    GC.gc()
    @test @allocated(VortexMethod.compute_sheet_curvature(sheet)) < 100_000

    VortexMethod.detect_sheet_rollup(sheet)
    GC.gc()
    @test @allocated(VortexMethod.detect_sheet_rollup(sheet)) < 10_000
end

@testset "High-level workspace hot paths" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.GridSpec(6, 6, 6)
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC =
        VortexMethod.structured_mesh(4, 4; domain=domain, amp=0.0)
    eleGma = fill(0.2, size(tri, 1), 3)

    sampler = VortexMethod.make_velocity_sampler(eleGma, triXC, triYC, triZC, domain, gr)
    sampler(0.1, 0.2, 0.0)
    GC.gc()
    @test @allocated(sampler(0.1, 0.2, 0.0)) < 512

    model = VortexMethod.VortexSheetModel(;
        grid=VortexMethod.RectilinearGrid(size=(6, 6, 6)),
        sheet_size=(4, 4),
        Γ=(0.0, 0.2, 0.0),
        amp=0.0,
    )
    VortexMethod.time_step!(model, 0.0)
    GC.gc()
    @test @allocated(VortexMethod.time_step!(model, 0.0)) < 80_000

    kernel_model = VortexMethod.VortexSheetModel(;
        grid=VortexMethod.RectilinearGrid(size=(6, 6, 6)),
        sheet_size=(4, 4),
        Γ=(0.0, 0.2, 0.0),
        amp=0.0,
        kernel=VortexMethod.PeskinCosine(),
    )
    VortexMethod.time_step!(kernel_model, 0.0)
    GC.gc()
    @test @allocated(VortexMethod.time_step!(kernel_model, 0.0)) < 120_000

    smag_model = VortexMethod.VortexSheetModel(;
        grid=VortexMethod.RectilinearGrid(size=(6, 6, 6)),
        sheet_size=(4, 4),
        Γ=(0.0, 0.2, 0.0),
        amp=0.0,
        dissipation=VortexMethod.SmagorinskyModel(),
    )
    VortexMethod.time_step!(smag_model, 0.0)
    GC.gc()
    @test @allocated(VortexMethod.time_step!(smag_model, 0.0)) < 250_000

    advanced_dissipation_models = (
        VortexMethod.DynamicSmagorinsky(),
        VortexMethod.VortexStretchingDissipation(),
        VortexMethod.MixedScaleModel(),
    )
    for dissipation in advanced_dissipation_models
        advanced_model = VortexMethod.VortexSheetModel(;
            grid=VortexMethod.RectilinearGrid(size=(6, 6, 6)),
            sheet_size=(4, 4),
            Γ=(0.0, 0.2, 0.0),
            amp=0.0,
            dissipation=dissipation,
        )
        @test fieldtype(typeof(advanced_model), :dissipation) === typeof(dissipation)
        @test fieldtype(typeof(advanced_model), :kernel) === typeof(VortexMethod.PeskinStandard())
        @test @inferred(VortexMethod.time_step!(advanced_model, 0.0)) == 0.0
        GC.gc()
        @test @allocated(VortexMethod.time_step!(advanced_model, 0.0)) < 150_000
    end

    simulation = VortexMethod.Simulation(model; Δt=0.0, stop_iteration=1)
    @test fieldtype(typeof(simulation), :model) === typeof(model)
end

@testset "fast_linalg hot-path solver assertions removed" begin
    source = read(joinpath(dirname(@__DIR__), "src", "diagnostics", "fast_linalg.jl"), String)
    for name in ("solve_4x3!", "batch_solve_3x3!", "solve_3x3!")
        start = findfirst("function $name", source)
        @test start !== nothing
        tail = source[last(start):end]
        next_function = findnext("\nfunction ", tail, 2)
        body = next_function === nothing ? tail : tail[1:first(next_function)-1]
        @test !occursin("@assert", body)
    end
end
