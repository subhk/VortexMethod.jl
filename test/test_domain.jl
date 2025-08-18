@testset "Domain + Periodicity" begin
    dom = VortexMethod.default_domain()
    gr = VortexMethod.default_grid()
    x,y,z = VortexMethod.grid_vectors(dom, gr)
    dx,dy,dz = VortexMethod.grid_spacing(dom, gr)
    @test length(x) == gr.nx && length(y) == gr.ny && length(z) == gr.nz
    @test isapprox(x[2]-x[1], dx; atol=1e-12)
    @test isapprox(y[2]-y[1], dy; atol=1e-12)
    @test isapprox(z[2]-z[1], dz; atol=1e-12)

    # wrap_point and wrap_nodes!
    xw,yw,zw = VortexMethod.wrap_point(-0.1, dom.Ly+0.1, dom.Lz+0.5, dom)
    @test 0.0 <= xw < dom.Lx
    @test 0.0 <= yw < dom.Ly
    @test -dom.Lz <= zw <= dom.Lz

    nodeX = [dom.Lx+1e-3, -1e-3]
    nodeY = [dom.Ly+1e-3, -1e-3]
    nodeZ = [dom.Lz+1e-3, -dom.Lz-1e-3]
    VortexMethod.wrap_nodes!(nodeX, nodeY, nodeZ, dom)
    @test all(0 .<= nodeX .< dom.Lx)
    @test all(0 .<= nodeY .< dom.Ly)
    @test all(-dom.Lz .<= nodeZ .<= dom.Lz)
end

