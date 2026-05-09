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

@testset "In-place Poisson solver matches allocating solver" begin
    domain = VortexMethod.default_domain()
    gr = VortexMethod.GridSpec(6, 6, 6)
    nz, ny, nx = gr.nz, gr.ny, gr.nx
    u_rhs = reshape(sin.(1:(nx * ny * nz)), nz, ny, nx)
    v_rhs = reshape(cos.(1:(nx * ny * nz)), nz, ny, nx)
    w_rhs = fill(0.125, nz, ny, nx)

    Ux, Uy, Uz = VortexMethod.poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain)
    outx = similar(u_rhs)
    outy = similar(u_rhs)
    outz = similar(u_rhs)
    workx = Array{ComplexF64}(undef, nz, ny, nx)
    worky = similar(workx)
    workz = similar(workx)

    VortexMethod.poisson_velocity_fft_mpi!(outx, outy, outz, workx, worky, workz,
                                           u_rhs, v_rhs, w_rhs, domain)
    @test outx ≈ Ux
    @test outy ≈ Uy
    @test outz ≈ Uz

    GC.gc()
    @test @allocated(VortexMethod.poisson_velocity_fft_mpi!(outx, outy, outz, workx, worky, workz,
                                                            u_rhs, v_rhs, w_rhs, domain)) < 10_000
end

@testset "Advanced Poisson solvers use concrete dispatch" begin
    domain = VortexMethod.default_domain()
    nz, ny, nx = 4, 4, 4
    u_rhs = reshape(sin.(1:(nx * ny * nz)), nz, ny, nx)
    v_rhs = reshape(cos.(1:(nx * ny * nz)), nz, ny, nx)
    w_rhs = fill(0.125, nz, ny, nx)

    fft_solver = VortexMethod.FFTSolver()
    ux, uy, uz = @inferred VortexMethod.Poisson.solve_poisson!(
        fft_solver, u_rhs, v_rhs, w_rhs, domain,
    )
    @test size(ux) == size(u_rhs)
    @test size(uy) == size(v_rhs)
    @test size(uz) == size(w_rhs)

    hybrid_solver = VortexMethod.HybridSolver(fft_solver, VortexMethod.FFTSolver(), Inf)
    @test fieldtype(typeof(hybrid_solver), :primary) === typeof(hybrid_solver.primary)
    @test fieldtype(typeof(hybrid_solver), :fallback) === typeof(hybrid_solver.fallback)
    @test @inferred(VortexMethod.Poisson.solve_poisson!(
        hybrid_solver, u_rhs, v_rhs, w_rhs, domain,
    )) isa NTuple{3,Array{Float64,3}}
end

@testset "Curl RHS uses second-order periodic differences" begin
    domain = VortexMethod.default_domain()
    nx, ny, nz = 6, 5, 4
    dx, dy, dz = VortexMethod.grid_spacing(domain, VortexMethod.GridSpec(nx, ny, nz))
    ζx = Array{Float64}(undef, nz, ny, nx)
    ζy = similar(ζx)
    ζz = similar(ζx)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ζx[k,j,i] = sin(0.2i + 0.3j + 0.5k)
        ζy[k,j,i] = cos(0.4i - 0.2j + 0.3k)
        ζz[k,j,i] = sin(0.1i - 0.4j + 0.6k)
    end

    prev(i, n) = i == 1 ? n : i - 1
    next(i, n) = i == n ? 1 : i + 1

    u_exp = similar(ζx)
    v_exp = similar(ζx)
    w_exp = similar(ζx)
    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        dZdy = (ζz[k,next(j,ny),i] - ζz[k,prev(j,ny),i]) / (2dy)
        dYdz = (ζy[next(k,nz),j,i] - ζy[prev(k,nz),j,i]) / (2dz)
        dXdz = (ζx[next(k,nz),j,i] - ζx[prev(k,nz),j,i]) / (2dz)
        dZdx = (ζz[k,j,next(i,nx)] - ζz[k,j,prev(i,nx)]) / (2dx)
        dYdx = (ζy[k,j,next(i,nx)] - ζy[k,j,prev(i,nx)]) / (2dx)
        dXdy = (ζx[k,next(j,ny),i] - ζx[k,prev(j,ny),i]) / (2dy)
        u_exp[k,j,i] = -(dZdy - dYdz)
        v_exp[k,j,i] = -(dXdz - dZdx)
        w_exp[k,j,i] = -(dYdx - dXdy)
    end

    u_rhs, v_rhs, w_rhs = VortexMethod.curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)

    @test u_rhs ≈ u_exp
    @test v_rhs ≈ v_exp
    @test w_rhs ≈ w_exp
end

@testset "Curl RHS workspace does not retain old derivative buffers" begin
    @test fieldcount(VortexMethod.Poisson.PoissonWorkspace{Float64}) == 0
end
