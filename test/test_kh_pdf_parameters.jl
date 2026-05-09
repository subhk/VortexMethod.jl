include(joinpath(@__DIR__, "..", "examples", "kh_stock_setup.jl"))

@testset "Stock KH PDF parameters" begin
    @test STOCK_KH_DOMAIN == VortexMethod.DomainSpec(1.0, 1.0, 2.0)
    @test STOCK_KH_GRID == VortexMethod.GridSpec(29, 29, 116)
    @test STOCK_KH_MESH_NX == 30
    @test STOCK_KH_MESH_NY == 30
    @test STOCK_KH_PERTURBATION_AMPLITUDE == 0.01
    @test STOCK_KH_INITIAL_GAMMA == (0.0, 1.0, 0.0)

    dx, dy, dz = VortexMethod.grid_spacing(STOCK_KH_DOMAIN, STOCK_KH_GRID)
    @test dx == 1 / 29
    @test dy == 1 / 29
    @test dz == 1 / 29

    split_threshold, merge_threshold = stock_kh_remesh_thresholds(STOCK_KH_DOMAIN, STOCK_KH_GRID)
    @test split_threshold ≈ 0.8 / 29
    @test merge_threshold ≈ 0.2 / 29

    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = VortexMethod.structured_mesh(
        STOCK_KH_MESH_NX,
        STOCK_KH_MESH_NY;
        domain=STOCK_KH_DOMAIN,
        amp=STOCK_KH_PERTURBATION_AMPLITUDE,
    )

    i = 7
    j = 11
    idx = (j - 1) * STOCK_KH_MESH_NX + i
    x0 = (i - 1) / (STOCK_KH_MESH_NX - 1)
    y0 = (j - 1) / (STOCK_KH_MESH_NY - 1)
    @test nodeX[idx] ≈ x0 + 0.01 * sin(2π * x0)
    @test nodeY[idx] ≈ y0
    @test nodeZ[idx] ≈ 0.01 * sin(2π * x0) + 0.01 * sin(4π * y0)
    @test minimum(nodeZ) > -0.03
    @test maximum(nodeZ) < 0.03

    eleGma = zeros(Float64, size(tri, 1), 3)
    initialize_stock_kh_gamma!(eleGma)
    @test all(eleGma[:, 1] .== 0.0)
    @test all(eleGma[:, 2] .== 1.0)
    @test all(eleGma[:, 3] .== 0.0)
end
