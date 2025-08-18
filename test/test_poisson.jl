@testset "Poisson spectral inversion" begin
    # Manufactured solution: Ux = sin(2π x / Lx), Uy=Uz=0
    dom = VortexMethod.default_domain()
    nx,ny,nz = 16, 16, 16
    gr = VortexMethod.GridSpec(nx,ny,nz)
    x,y,z = VortexMethod.grid_vectors(dom, gr)
    kx = 2pi / dom.Lx
    Ux_true = Array{Float64}(undef, nz, ny, nx)
    Uy_true = zeros(Float64, nz, ny, nx)
    Uz_true = zeros(Float64, nz, ny, nx)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        Ux_true[k,j,i] = sin(kx * x[i])
    end
    # Laplacian(Ux) = -kx^2 * Ux
    u_rhs = -kx^2 .* Ux_true
    v_rhs = zeros(size(Ux_true)); w_rhs = zeros(size(Ux_true))
    Ux, Uy, Uz = VortexMethod.poisson_velocity_fft(u_rhs, v_rhs, w_rhs, dom; mode=:spectral)
    err = maximum(abs.(Ux .- Ux_true))
    @test err < 1e-6
    @test maximum(abs.(Uy)) < 1e-12
    @test maximum(abs.(Uz)) < 1e-12
end

