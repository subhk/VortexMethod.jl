@testset "Remeshing baseline" begin
    domain = VortexMethod.default_domain()
    Nx, Ny = 8, 8
    nodeX, nodeY, nodeZ, tri, triXC, triYC, triZC = VortexMethod.structured_mesh(Nx, Ny; domain=domain)
    nt0 = size(tri,1)
    # Force splits by setting ds_max below typical edge length
    dx,dy,dz = VortexMethod.grid_spacing(domain, VortexMethod.default_grid())
    ds_max = 0.5 * max(dx,dy)
    ds_min = 1e-6
    tri2, changed = VortexMethod.Remesh.remesh_pass!(nodeX, nodeY, nodeZ, tri, ds_max, ds_min; domain=domain, compact=true)
    @test changed
    @test size(tri2,1) > nt0
    # Nodes should be wrapped into domain
    @test all(0 .<= nodeX .< domain.Lx)
    @test all(0 .<= nodeY .< domain.Ly)
    @test all(-domain.Lz .<= nodeZ .<= domain.Lz)
end
