@testset "Domain + Periodicity" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.default_grid()
    x,y,z = VortexMethod.grid_vectors(domain, gr)
    dx,dy,dz = VortexMethod.grid_spacing(domain, gr)
    @test length(x) == gr.nx && length(y) == gr.ny && length(z) == gr.nz
    @test isapprox(x[2]-x[1], dx; atol=1e-12)
    @test isapprox(y[2]-y[1], dy; atol=1e-12)
    @test isapprox(z[2]-z[1], dz; atol=1e-12)

    # wrap_point and wrap_nodes!
    xw,yw,zw = VortexMethod.wrap_point(-0.1, domain.Ly+0.1, domain.Lz+0.5, domain)
    @test 0.0 <= xw < domain.Lx
    @test 0.0 <= yw < domain.Ly
    @test -domain.Lz <= zw <= domain.Lz

    nodeX = [domain.Lx+1e-3, -1e-3]
    nodeY = [domain.Ly+1e-3, -1e-3]
    nodeZ = [domain.Lz+1e-3, -domain.Lz-1e-3]
    VortexMethod.wrap_nodes!(nodeX, nodeY, nodeZ, domain)
    @test all(0 .<= nodeX .< domain.Lx)
    @test all(0 .<= nodeY .< domain.Ly)
    @test all(-domain.Lz .<= nodeZ .<= domain.Lz)
end
