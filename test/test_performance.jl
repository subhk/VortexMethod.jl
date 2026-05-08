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
