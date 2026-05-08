using LinearAlgebra
using Statistics

function _tri_coords(nodeX, nodeY, nodeZ, tri)
    triXC = Matrix{Float64}(undef, size(tri, 1), 3)
    triYC = similar(triXC)
    triZC = similar(triXC)
    @inbounds for t in 1:size(tri, 1), k in 1:3
        v = tri[t, k]
        triXC[t, k] = nodeX[v]
        triYC[t, k] = nodeY[v]
        triZC[t, k] = nodeZ[v]
    end
    return triXC, triYC, triZC
end

function _total_circulation(nodeX, nodeY, nodeZ, tri, eleGma, domain)
    triXC, triYC, triZC = _tri_coords(nodeX, nodeY, nodeZ, tri)
    areas = VortexMethod.Peskin3D.triangle_areas(triXC, triYC, triZC; domain=domain)
    return vec(sum(areas .* eleGma; dims=1))
end

@testset "Stock regression fixes" begin
    @testset "remeshing carries circulation" begin
        domain = VortexMethod.default_domain()
        nodeX = [0.1, 0.75, 0.1]
        nodeY = [0.1, 0.1, 0.75]
        nodeZ = [0.0, 0.0, 0.0]
        tri = [1 2 3]
        eleGma = [1.2 -0.4 0.3]

        before = _total_circulation(copy(nodeX), copy(nodeY), copy(nodeZ), tri, eleGma, domain)
        tri2, eleGma2, changed = VortexMethod.Remesh.remesh_pass!(
            nodeX, nodeY, nodeZ, tri, eleGma, 0.2, 1e-6;
            domain=domain, compact=false, max_flips=0, max_merges=0,
        )
        after = _total_circulation(nodeX, nodeY, nodeZ, tri2, eleGma2, domain)

        @test changed
        @test size(eleGma2, 1) == size(tri2, 1)
        @test isapprox(after, before; rtol=1e-12, atol=1e-12)
    end

    @testset "legacy topology-only remesh APIs are removed" begin
        @test !hasmethod(
            VortexMethod.Remesh.remesh_pass!,
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Int}, Float64, Float64},
        )
        @test !hasmethod(
            VortexMethod.RemeshAdvanced.quality_based_remesh!,
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Int}, VortexMethod.DomainSpec},
        )
        @test !hasmethod(
            VortexMethod.RemeshAdvanced.anisotropic_remesh!,
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Int}, Function, VortexMethod.DomainSpec},
        )
        @test !hasmethod(
            VortexMethod.RemeshAdvanced.curvature_based_remesh!,
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Int}, VortexMethod.DomainSpec},
        )
        @test !hasmethod(
            VortexMethod.RemeshAdvanced.flow_adaptive_remesh!,
            Tuple{Vector{Float64}, Vector{Float64}, Vector{Float64}, Matrix{Int}, Function, VortexMethod.DomainSpec},
        )
    end

    @testset "generic transport rejects topology changes" begin
        domain = VortexMethod.default_domain()
        oldX = [0.1 0.7 0.1]
        oldY = [0.1 0.1 0.7]
        oldZ = [0.0 0.0 0.0]
        eleGma = [1.0 2.0 3.0]
        newX = [
            0.1 0.4 0.1
            0.4 0.7 0.4
            0.1 0.4 0.1
            0.4 0.4 0.1
        ]
        newY = [
            0.1 0.1 0.4
            0.1 0.1 0.4
            0.4 0.4 0.7
            0.1 0.4 0.4
        ]
        newZ = zeros(4, 3)

        @test_throws DimensionMismatch VortexMethod.transport_ele_gamma(
            eleGma, oldX, oldY, oldZ, newX, newY, newZ; domain=domain,
        )
        @test_throws ArgumentError VortexMethod.transport_ele_gamma(
            eleGma, oldX, oldY, oldZ, newX, newY, newZ; domain=domain, method=:nearest,
        )
    end

    @testset "node merge redistributes collapsed vorticity over survivors" begin
        domain = VortexMethod.DomainSpec(10.0, 10.0, 10.0)
        nodeX = [1.00, 1.01, 1.00, 1.00, 0.00, 2.00]
        nodeY = [1.00, 1.00, 2.00, 0.00, 1.50, 0.50]
        nodeZ = zeros(6)
        tri = [
            1 2 3
            2 1 4
            1 5 3
            2 4 6
        ]
        eleGma = [
            1.0 0.0 0.0
            0.0 2.0 0.0
            0.0 0.0 3.0
            4.0 0.0 0.0
        ]

        triXC0, triYC0, triZC0 = _tri_coords(nodeX, nodeY, nodeZ, tri)
        old_areas = VortexMethod.Peskin3D.triangle_areas(triXC0, triYC0, triZC0; domain=domain)
        old_total = vec(sum(old_areas .* eleGma; dims=1))

        tri2, eleGma2, changed = VortexMethod.Remesh.remesh_pass!(
            nodeX, nodeY, nodeZ, tri, eleGma, 100.0, 0.02;
            domain=domain, compact=false, max_flips=0, max_merges=1,
        )
        triXC2, triYC2, triZC2 = _tri_coords(nodeX, nodeY, nodeZ, tri2)
        new_areas = VortexMethod.Peskin3D.triangle_areas(triXC2, triYC2, triZC2; domain=domain)
        survivor_total = vec(sum(new_areas .* eleGma[3:4, :]; dims=1))
        redistribution = (old_total .- survivor_total) ./ sum(new_areas)
        expected = copy(eleGma[3:4, :])
        expected .+= reshape(redistribution, 1, 3)

        @test changed
        @test size(tri2, 1) == 2
        @test isapprox(eleGma2, expected; rtol=1e-12, atol=1e-12)
        @test isapprox(_total_circulation(nodeX, nodeY, nodeZ, tri2, eleGma2, domain),
                       old_total; rtol=1e-12, atol=1e-12)
    end

    @testset "advanced remeshing carries circulation" begin
        domain = VortexMethod.default_domain()
        nodeX = [0.1, 0.8, 0.12]
        nodeY = [0.1, 0.1, 0.18]
        nodeZ = [0.0, 0.0, 0.0]
        tri = [1 2 3]
        eleGma = [0.2 0.5 -0.3]

        before = _total_circulation(copy(nodeX), copy(nodeY), copy(nodeZ), tri, eleGma, domain)
        tri2, eleGma2, changed = VortexMethod.RemeshAdvanced.quality_based_remesh!(
            nodeX, nodeY, nodeZ, tri, eleGma, domain;
            max_aspect_ratio=1.01, max_elements=16,
        )
        after = _total_circulation(nodeX, nodeY, nodeZ, tri2, eleGma2, domain)

        @test changed
        @test size(eleGma2, 1) == size(tri2, 1)
        @test isapprox(after, before; rtol=1e-12, atol=1e-12)
    end

    @testset "advanced split copies parent strength to children" begin
        domain = VortexMethod.DomainSpec(10.0, 10.0, 10.0)
        nodeX = [0.0, 1.0, 0.0, 0.166, 0.168, 0.167]
        nodeY = [0.0, 0.0, 0.01, 0.0061, 0.0061, 0.0078]
        nodeZ = zeros(6)
        tri = [
            1 2 3
            4 5 6
        ]
        parent_gamma = [1.0, -2.0, 0.5]
        neighbor_gamma = [-3.0, 4.0, 2.0]
        eleGma = [
            parent_gamma'
            neighbor_gamma'
        ]

        tri2, eleGma2, changed = VortexMethod.RemeshAdvanced.quality_based_remesh!(
            nodeX, nodeY, nodeZ, tri, eleGma, domain;
            max_aspect_ratio=10.0,
            max_skewness=Inf,
            min_angle_quality=0.0,
            min_jacobian_quality=0.0,
            max_elements=8,
        )

        @test changed
        @test size(tri2, 1) == 5
        @test isapprox(eleGma2[1, :], parent_gamma; atol=0.0)
        @test isapprox(eleGma2[2, :], neighbor_gamma; atol=0.0)
        @test all(isapprox(vec(eleGma2[t, :]), parent_gamma; atol=0.0) for t in 3:5)
    end

    @testset "periodic triangle geometry unwraps boundary crossings" begin
        domain = VortexMethod.default_domain()
        triXC = [0.98 0.02 0.98]
        triYC = [0.0 0.0 0.2]
        triZC = [0.0 0.0 0.0]

        geom = VortexMethod.compute_triangle_geometry(triXC, triYC, triZC; domain=domain)
        areas = VortexMethod.Peskin3D.triangle_areas(triXC, triYC, triZC; domain=domain)
        centroids = VortexMethod.Peskin3D.triangle_centroids(triXC, triYC, triZC; domain=domain)

        @test isapprox(geom.areas[1], 0.004; atol=1e-12)
        @test isapprox(areas[1], 0.004; atol=1e-12)
        @test isapprox(centroids[1, 1], 0.9933333333333333; atol=1e-12)
    end

    @testset "kernel support includes z periodic images" begin
        domain = VortexMethod.default_domain()
        triC = [0.5 0.5 0.95]
        subC = Array{Float64}(undef, 1, 1, 3)
        subC[1, 1, :] .= (0.5, 0.5, 0.95)
        areas = [1.0]
        eleGma = [1.0 0.0 0.0]

        sx, sy, sz = VortexMethod.Peskin3D.peskin_grid_sum(
            eleGma, triC, subC, (0.5, 0.5, -0.98), (0.1, 0.1, 0.1), areas;
            delr=1.0, domain=domain,
        )

        @test sx > 0.0
        @test sy == 0.0
        @test sz == 0.0
    end

    @testset "subtriangle sampling supports M by M partitions" begin
        p1 = (0.0, 0.0, 0.0)
        p2 = (3.0, 0.0, 0.0)
        p3 = (0.0, 3.0, 0.0)

        c2 = VortexMethod.Peskin3D.subtriangle_centroids(p1, p2, p3, 2)
        c3 = VortexMethod.Peskin3D.subtriangle_centroids(p1, p2, p3, 3)

        @test size(c2) == (4, 3)
        @test size(c3) == (9, 3)
        @test isapprox(vec(mean(c3; dims=1)), [1.0, 1.0, 0.0]; atol=1e-12)
        @test isapprox(c2, VortexMethod.Peskin3D.subtriangle_centroids4(p1, p2, p3); atol=1e-12)
    end

    @testset "baroclinicity accepts per-element Atwood numbers" begin
        triXC = [0.0 1.0 0.0; 0.0 1.0 0.0]
        triYC = [0.0 0.0 1.0; 0.0 0.0 1.0]
        triZC = [0.0 0.0 1.0; 0.0 0.0 1.0]

        dG = VortexMethod.baroclinic_ele_gamma([0.5, 1.0], 0.25, triXC, triYC, triZC)

        @test isapprox(dG[2, 1], 2dG[1, 1]; atol=1e-12)
        @test isapprox(dG[2, 2], 2dG[1, 2]; atol=1e-12)
        @test all(dG[:, 3] .== 0.0)
    end

    @testset "SFS dissipation uses Eulerian grid tendency" begin
        domain = VortexMethod.default_domain()
        gr = VortexMethod.GridSpec(8, 8, 8)
        triXC = [0.20 0.45 0.20; 0.45 0.45 0.20]
        triYC = [0.20 0.20 0.45; 0.20 0.45 0.45]
        triZC = [0.00 0.00 0.00; 0.00 0.00 0.00]
        eleGma0 = [0.8 -0.2 0.4; -0.3 0.5 0.1]
        model = VortexMethod.SmagorinskyModel(0.17)
        dt = 0.05

        eleGma = VortexMethod.apply_dissipation!(model, copy(eleGma0), triXC, triYC, triZC, domain, gr, dt)
        local_decay = copy(eleGma0)
        for t in 1:size(local_decay, 1)
            local_decay[t, :] .*= exp(-(model.Cs^2) * norm(eleGma0[t, :]) * dt)
        end

        @test all(isfinite, eleGma)
        @test !isapprox(eleGma, local_decay; rtol=1e-8, atol=1e-10)
    end

    @testset "periodic FFT Poisson rejects nonperiodic boundary requests" begin
        domain = VortexMethod.default_domain()
        rhs = zeros(Float64, 4, 4, 4)

        @test_throws ArgumentError VortexMethod.poisson_velocity_fft(rhs, rhs, rhs, domain; boundary_condition=:wall)
    end
end
