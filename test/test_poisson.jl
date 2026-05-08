@testset "Poisson spectral inversion" begin
    # Manufactured solution: Ux = sin(2π x / Lx), Uy=Uz=0
    domain = VortexMethod.default_domain()
    nx,ny,nz = 16, 16, 16
    gr = VortexMethod.GridSpec(nx,ny,nz)
    x,y,z = VortexMethod.grid_vectors(domain, gr)
    kx = 2pi / domain.Lx
    Ux_true = Array{Float64}(undef, nz, ny, nx)
    Uy_true = zeros(Float64, nz, ny, nx)
    Uz_true = zeros(Float64, nz, ny, nx)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        Ux_true[k,j,i] = sin(kx * x[i])
    end
    # Laplacian(Ux) = -kx^2 * Ux
    u_rhs = -kx^2 .* Ux_true
    v_rhs = zeros(size(Ux_true)); w_rhs = zeros(size(Ux_true))
    Ux, Uy, Uz = VortexMethod.poisson_velocity_fft(u_rhs, v_rhs, w_rhs, domain; mode=:spectral)
    err = maximum(abs.(Ux .- Ux_true))
    @test err < 1e-6
    @test maximum(abs.(Uy)) < 1e-12
    @test maximum(abs.(Uz)) < 1e-12
end

@testset "Curl RHS uses second-order periodic differences" begin
    domain = VortexMethod.default_domain()
    nx, ny, nz = 6, 5, 4
    dx, dy, dz = VortexMethod.grid_spacing(domain, VortexMethod.GridSpec(nx, ny, nz))
    VorX = Array{Float64}(undef, nz, ny, nx)
    VorY = similar(VorX)
    VorZ = similar(VorX)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        VorX[k,j,i] = sin(0.2i + 0.3j + 0.5k)
        VorY[k,j,i] = cos(0.4i - 0.2j + 0.3k)
        VorZ[k,j,i] = sin(0.1i - 0.4j + 0.6k)
    end

    prev(i, n) = i == 1 ? n : i - 1
    next(i, n) = i == n ? 1 : i + 1

    u_exp = similar(VorX)
    v_exp = similar(VorX)
    w_exp = similar(VorX)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        dZdy = (VorZ[k,next(j,ny),i] - VorZ[k,prev(j,ny),i]) / (2dy)
        dYdz = (VorY[next(k,nz),j,i] - VorY[prev(k,nz),j,i]) / (2dz)
        dXdz = (VorX[next(k,nz),j,i] - VorX[prev(k,nz),j,i]) / (2dz)
        dZdx = (VorZ[k,j,next(i,nx)] - VorZ[k,j,prev(i,nx)]) / (2dx)
        dYdx = (VorY[k,j,next(i,nx)] - VorY[k,j,prev(i,nx)]) / (2dx)
        dXdy = (VorX[k,next(j,ny),i] - VorX[k,prev(j,ny),i]) / (2dy)
        u_exp[k,j,i] = -(dZdy - dYdz)
        v_exp[k,j,i] = -(dXdz - dZdx)
        w_exp[k,j,i] = -(dYdx - dXdy)
    end

    u_rhs, v_rhs, w_rhs = VortexMethod.curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)

    @test u_rhs ≈ u_exp
    @test v_rhs ≈ v_exp
    @test w_rhs ≈ w_exp
end

@testset "Curl RHS workspace does not retain old derivative buffers" begin
    @test fieldcount(VortexMethod.Poisson3D.PoissonWorkspace{Float64}) == 0
end
