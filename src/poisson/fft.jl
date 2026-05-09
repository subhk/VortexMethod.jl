# 3D FFT-based Poisson solver and curl RHS

# Compatibility token for the in-place curl RHS API. The second-order periodic
# stencil computes directly into the output arrays and needs no derivative buffers.
struct PoissonWorkspace{T<:AbstractFloat}
end

# Constructor for workspace
function PoissonWorkspace(::Type{T}, nz::Int, ny::Int, nx::Int) where T<:AbstractFloat
    return PoissonWorkspace{T}()
end

PoissonWorkspace(nz::Int, ny::Int, nx::Int) = PoissonWorkspace(Float64, nz, ny, nx)

@inline prev_periodic_index(i::Int, n::Int) = i == 1 ? n : i - 1
@inline next_periodic_index(i::Int, n::Int) = i == n ? 1 : i + 1

# In-place version; workspace is retained for API compatibility.
function curl_rhs_centered!(_workspace::PoissonWorkspace{T}, 
                           u_rhs::AbstractArray{T,3}, v_rhs::AbstractArray{T,3}, w_rhs::AbstractArray{T,3},
                           ζx::AbstractArray{T,3}, ζy::AbstractArray{T,3}, ζz::AbstractArray{T,3},
                           dx::T, dy::T, dz::T) where T<:AbstractFloat
    nz, ny, nx = size(ζx)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ip = next_periodic_index(i, nx)
        im = prev_periodic_index(i, nx)
        jp = next_periodic_index(j, ny)
        jm = prev_periodic_index(j, ny)
        kp = next_periodic_index(k, nz)
        km = prev_periodic_index(k, nz)

        dZ_dy = (ζz[k,jp,i] - ζz[k,jm,i]) / (2dy)
        dY_dz = (ζy[kp,j,i] - ζy[km,j,i]) / (2dz)
        dX_dz = (ζx[kp,j,i] - ζx[km,j,i]) / (2dz)
        dZ_dx = (ζz[k,j,ip] - ζz[k,j,im]) / (2dx)
        dY_dx = (ζy[k,j,ip] - ζy[k,j,im]) / (2dx)
        dX_dy = (ζx[k,jp,i] - ζx[k,jm,i]) / (2dy)

        u_rhs[k,j,i] = -(dZ_dy - dY_dz)
        v_rhs[k,j,i] = -(dX_dz - dZ_dx)
        w_rhs[k,j,i] = -(dY_dx - dX_dy)
    end
    
    return nothing
end

# Backward-compatible wrapper that allocates
function curl_rhs_centered(ζx::AbstractArray{Float64,3}, ζy::AbstractArray{Float64,3}, ζz::AbstractArray{Float64,3},
                           dx::Float64, dy::Float64, dz::Float64)
    nz, ny, nx = size(ζx)
    workspace = PoissonWorkspace(nz, ny, nx)
    u_rhs = similar(ζx)
    v_rhs = similar(ζx)
    w_rhs = similar(ζx)
    curl_rhs_centered!(workspace, u_rhs, v_rhs, w_rhs, ζx, ζy, ζz, dx, dy, dz)
    return u_rhs, v_rhs, w_rhs
end

function require_periodic_boundary(boundary_condition::Symbol)
    boundary_condition == :periodic && return nothing
    throw(ArgumentError("poisson_velocity_fft supports only periodic FFT boundaries; requested boundary_condition=$boundary_condition"))
end

@inline function _spectral_wavenumber(i::Int, n::Int, L::Float64, ::Type{T}) where T<:AbstractFloat
    m = i - 1
    n2 = fld(n, 2)
    return m <= n2 ? T(2π * m / L) : T(-2π * (n - m) / L)
end

function init_mpi!()
    MPI.Finalized() && throw(ErrorException("MPI has already been finalized"))
    MPI.Initialized() || MPI.Init()
    return nothing
end

struct PencilPoissonWorkspace{T<:AbstractFloat,P,In,Out,InView,OutView}
    dims::NTuple{3,Int}
    comm_size::Int
    fft_plan::P
    u_local::In
    v_local::In
    w_local::In
    Fu::Out
    Fv::Out
    Fw::Out
    ux_local::In
    uy_local::In
    uz_local::In
    velocity_local_global::Matrix{T}
    velocity_global::Matrix{T}
    u_view::InView
    v_view::InView
    w_view::InView
    Fu_view::OutView
    Fv_view::OutView
    Fw_view::OutView
    ux_view::InView
    uy_view::InView
    uz_view::InView
end

function PencilPoissonWorkspace(::Type{T}, dims::NTuple{3,Int}, comm::MPI.Comm) where T<:AbstractFloat
    pen = Pencil(dims, comm)
    fft_plan = PencilFFTPlan(pen, Transforms.FFT(), T)
    u_local = allocate_input(fft_plan)
    v_local = allocate_input(fft_plan)
    w_local = allocate_input(fft_plan)
    Fu = allocate_output(fft_plan)
    Fv = allocate_output(fft_plan)
    Fw = allocate_output(fft_plan)
    ux_local = allocate_input(fft_plan)
    uy_local = allocate_input(fft_plan)
    uz_local = allocate_input(fft_plan)
    u_view = global_view(u_local)
    v_view = global_view(v_local)
    w_view = global_view(w_local)
    Fu_view = global_view(Fu)
    Fv_view = global_view(Fv)
    Fw_view = global_view(Fw)
    ux_view = global_view(ux_local)
    uy_view = global_view(uy_local)
    uz_view = global_view(uz_local)
    nz, ny, nx = dims
    return PencilPoissonWorkspace{T,typeof(fft_plan),typeof(u_local),typeof(Fu),
                                  typeof(u_view),typeof(Fu_view)}(
        dims,
        MPI.Comm_size(comm),
        fft_plan,
        u_local, v_local, w_local,
        Fu, Fv, Fw,
        ux_local, uy_local, uz_local,
        zeros(T, nz * ny * nx, 3),
        zeros(T, nz * ny * nx, 3),
        u_view, v_view, w_view,
        Fu_view, Fv_view, Fw_view,
        ux_view, uy_view, uz_view,
    )
end

function _pencil_poisson_workspace!(cache::Base.RefValue{Any},
                                    ::Type{T},
                                    dims::NTuple{3,Int},
                                    comm::MPI.Comm) where T<:AbstractFloat
    cached = cache[]
    if cached isa PencilPoissonWorkspace{T} &&
       cached.dims == dims &&
       cached.comm_size == MPI.Comm_size(comm)
        return cached
    end
    workspace = PencilPoissonWorkspace(T, dims, comm)
    cache[] = workspace
    return workspace
end

# FFT-based Poisson solve (periodic): ∇^2 U = RHS -> Û = -RHŜ/k^2
function poisson_velocity_fft(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3},
                              domain::DomainSpec; mode::Symbol=:spectral,
                              boundary_condition::Symbol=:periodic)
    require_periodic_boundary(boundary_condition)
    nz, ny, nx = size(u_rhs)
    dx = domain.Lx/nx
    dy = domain.Ly/ny
    dz = (2*domain.Lz)/nz

    kx = kvec(nx, domain.Lx)
    ky = kvec(ny, domain.Ly)
    kz = kvec(nz, 2*domain.Lz)

    # Create grids of wavenumbers or FD symbol
    if mode == :spectral
        KX = reshape(kx, 1,1,nx)
        KY = reshape(ky, 1,ny,1)
        KZ = reshape(kz, nz,1,1)
        sym = KX.^2 .+ KY.^2 .+ KZ.^2
    elseif mode == :fd
        # Discrete Laplacian symbol: 2(cos(2πm/n) - 1) / h² for 3-point central difference
        mx = collect(0:nx-1); my = collect(0:ny-1); mz = collect(0:nz-1)
        CX = reshape(2.0 .* (cos.(2pi .* mx ./ nx) .- 1.0), 1,1,nx) ./ (dx^2)
        CY = reshape(2.0 .* (cos.(2pi .* my ./ ny) .- 1.0), 1,ny,1) ./ (dy^2)
        CZ = reshape(2.0 .* (cos.(2pi .* mz ./ nz) .- 1.0), nz,1,1) ./ (dz^2)
        sym = CX .+ CY .+ CZ
    else
        error("Unknown Poisson mode: $mode (use :spectral or :fd)")
    end

    # Use full n-dimensional FFT
    Fu = FFTW.fft(u_rhs)
    Fv = FFTW.fft(v_rhs)
    Fw = FFTW.fft(w_rhs)

    # Avoid division by zero at k=0
    sym[1,1,1] = 1.0
    if mode == :fd
        scale = 0.5/(domain.Lx*domain.Ly*domain.Lz)
        Û = scale .* Fu ./ sym
        V̂ = scale .* Fv ./ sym
        Ŵ = scale .* Fw ./ sym
    else
        Û = -Fu ./ sym
        V̂ = -Fv ./ sym
        Ŵ = -Fw ./ sym
    end
    Û[1,1,1] = 0.0 + 0.0im
    V̂[1,1,1] = 0.0 + 0.0im
    Ŵ[1,1,1] = 0.0 + 0.0im

    ux = real(FFTW.ifft(Û))
    uy = real(FFTW.ifft(V̂))
    uz = real(FFTW.ifft(Ŵ))

    # Note: Periodic boundary conditions are handled automatically by FFT

    return ux, uy, uz
end

function poisson_velocity_fft!(ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                               Fu::Array{Complex{T},3},
                               Fv::Array{Complex{T},3},
                               Fw::Array{Complex{T},3},
                               u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                               domain::DomainSpec; mode::Symbol=:spectral,
                               boundary_condition::Symbol=:periodic) where T<:AbstractFloat
    require_periodic_boundary(boundary_condition)
    mode == :spectral || mode == :fd ||
        throw(ArgumentError("Unknown Poisson mode: $mode (use :spectral or :fd)"))
    size(ux) == size(u_rhs) == size(Fu) ||
        throw(DimensionMismatch("ux, Fu, and u_rhs must have matching sizes"))
    size(uy) == size(v_rhs) == size(Fv) ||
        throw(DimensionMismatch("uy, Fv, and v_rhs must have matching sizes"))
    size(uz) == size(w_rhs) == size(Fw) ||
        throw(DimensionMismatch("uz, Fw, and w_rhs must have matching sizes"))

    nz, ny, nx = size(u_rhs)
    dx = T(domain.Lx / nx)
    dy = T(domain.Ly / ny)
    dz = T((2 * domain.Lz) / nz)
    fd_scale = T(0.5 / (domain.Lx * domain.Ly * domain.Lz))

    @inbounds for I in eachindex(u_rhs, v_rhs, w_rhs, Fu, Fv, Fw)
        Fu[I] = Complex{T}(u_rhs[I], zero(T))
        Fv[I] = Complex{T}(v_rhs[I], zero(T))
        Fw[I] = Complex{T}(w_rhs[I], zero(T))
    end

    FFTW.fft!(Fu)
    FFTW.fft!(Fv)
    FFTW.fft!(Fw)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        if mode == :spectral
            kx = _spectral_wavenumber(i, nx, domain.Lx, T)
            ky = _spectral_wavenumber(j, ny, domain.Ly, T)
            kz = _spectral_wavenumber(k, nz, 2 * domain.Lz, T)
            sym = kx*kx + ky*ky + kz*kz
            factor = iszero(sym) ? zero(T) : -one(T) / sym
        else
            mx = T(i - 1)
            my = T(j - 1)
            mz = T(k - 1)
            sym = T(2) * (cos(T(2π) * mx / T(nx)) - one(T)) / (dx * dx) +
                  T(2) * (cos(T(2π) * my / T(ny)) - one(T)) / (dy * dy) +
                  T(2) * (cos(T(2π) * mz / T(nz)) - one(T)) / (dz * dz)
            factor = iszero(sym) ? zero(T) : fd_scale / sym
        end
        Fu[k,j,i] *= factor
        Fv[k,j,i] *= factor
        Fw[k,j,i] *= factor
    end

    FFTW.ifft!(Fu)
    FFTW.ifft!(Fv)
    FFTW.ifft!(Fw)

    @inbounds for I in eachindex(ux, uy, uz, Fu, Fv, Fw)
        ux[I] = real(Fu[I])
        uy[I] = real(Fv[I])
        uz[I] = real(Fw[I])
    end

    return ux, uy, uz
end

"""
poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=:spectral)

Parallel FFT-based Poisson solve using PencilFFTs for true MPI parallelism.
Distributes FFT computation across all MPI ranks using pencil decomposition.

# Arguments
- `u_rhs, v_rhs, w_rhs::Array{Float64,3}`: Right-hand side curl terms
- `domain::DomainSpec`: Domain specification
- `mode::Symbol=:spectral`: FFT mode (`:spectral` or `:fd`)

# Returns
- `(ux, uy, uz)`: Velocity field arrays with periodic boundary conditions applied
"""
function poisson_velocity_pencil_fft(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3},
                                     domain::DomainSpec; mode::Symbol=:spectral,
                                     boundary_condition::Symbol=:periodic)
    require_periodic_boundary(boundary_condition)
    init_mpi!()
    comm = MPI.COMM_WORLD
    nz, ny, nx = size(u_rhs)

    dx = domain.Lx/nx
    dy = domain.Ly/ny
    dz = (2*domain.Lz)/nz

    kx = kvec(nx, domain.Lx)
    ky = kvec(ny, domain.Ly)
    kz = kvec(nz, 2*domain.Lz)

    if mode != :spectral && mode != :fd
        error("Unknown Poisson mode: $mode (use :spectral or :fd)")
    end

    pen = Pencil((nz, ny, nx), comm)
    fft_plan = PencilFFTPlan(pen, Transforms.FFT())

    u_local = allocate_input(fft_plan)
    v_local = allocate_input(fft_plan)
    w_local = allocate_input(fft_plan)

    u_view = global_view(u_local)
    v_view = global_view(v_local)
    w_view = global_view(w_local)
    @inbounds for I in CartesianIndices(u_view)
        u_view[I] = u_rhs[I]
        v_view[I] = v_rhs[I]
        w_view[I] = w_rhs[I]
    end

    Fu = allocate_output(fft_plan)
    Fv = allocate_output(fft_plan)
    Fw = allocate_output(fft_plan)

    mul!(Fu, fft_plan, u_local)
    mul!(Fv, fft_plan, v_local)
    mul!(Fw, fft_plan, w_local)

    Fu_view = global_view(Fu)
    Fv_view = global_view(Fv)
    Fw_view = global_view(Fw)
    fd_scale = 0.5 / (domain.Lx * domain.Ly * domain.Lz)
    @inbounds for I in CartesianIndices(Fu_view)
        k, j, i = Tuple(I)
        if mode == :spectral
            sym = kz[k]^2 + ky[j]^2 + kx[i]^2
            factor = sym == 0.0 ? 0.0 : -1.0 / sym
        else
            mx = i - 1
            my = j - 1
            mz = k - 1
            sym = 2.0 * (cos(2pi * mx / nx) - 1.0) / dx^2 +
                  2.0 * (cos(2pi * my / ny) - 1.0) / dy^2 +
                  2.0 * (cos(2pi * mz / nz) - 1.0) / dz^2
            factor = sym == 0.0 ? 0.0 : fd_scale / sym
        end
        Fu_view[I] *= factor
        Fv_view[I] *= factor
        Fw_view[I] *= factor
    end

    ux_local = allocate_input(fft_plan)
    uy_local = allocate_input(fft_plan)
    uz_local = allocate_input(fft_plan)

    ldiv!(ux_local, fft_plan, Fu)
    ldiv!(uy_local, fft_plan, Fv)
    ldiv!(uz_local, fft_plan, Fw)

    ux_local_global = zeros(Float64, nz, ny, nx)
    uy_local_global = zeros(Float64, nz, ny, nx)
    uz_local_global = zeros(Float64, nz, ny, nx)

    ux_view = global_view(ux_local)
    uy_view = global_view(uy_local)
    uz_view = global_view(uz_local)
    @inbounds for I in CartesianIndices(ux_view)
        ux_local_global[I] = real(ux_view[I])
        uy_local_global[I] = real(uy_view[I])
        uz_local_global[I] = real(uz_view[I])
    end

    ux_gathered = zeros(Float64, nz, ny, nx)
    uy_gathered = zeros(Float64, nz, ny, nx)
    uz_gathered = zeros(Float64, nz, ny, nx)

    MPI.Allreduce!(ux_local_global, ux_gathered, MPI.SUM, comm)
    MPI.Allreduce!(uy_local_global, uy_gathered, MPI.SUM, comm)
    MPI.Allreduce!(uz_local_global, uz_gathered, MPI.SUM, comm)

    return ux_gathered, uy_gathered, uz_gathered
end

function _poisson_velocity_pencil_fft!(workspace::PencilPoissonWorkspace{T},
                                       ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                                       u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                                       domain::DomainSpec, comm::MPI.Comm,
                                       mode::Symbol) where T<:AbstractFloat
    nz, ny, nx = workspace.dims
    dx = T(domain.Lx / nx)
    dy = T(domain.Ly / ny)
    dz = T((2 * domain.Lz) / nz)

    u_view = workspace.u_view
    v_view = workspace.v_view
    w_view = workspace.w_view
    @inbounds for I in CartesianIndices(u_view)
        u_view[I] = u_rhs[I]
        v_view[I] = v_rhs[I]
        w_view[I] = w_rhs[I]
    end

    mul!(workspace.Fu, workspace.fft_plan, workspace.u_local)
    mul!(workspace.Fv, workspace.fft_plan, workspace.v_local)
    mul!(workspace.Fw, workspace.fft_plan, workspace.w_local)

    Fu_view = workspace.Fu_view
    Fv_view = workspace.Fv_view
    Fw_view = workspace.Fw_view
    fd_scale = T(0.5 / (domain.Lx * domain.Ly * domain.Lz))
    @inbounds for I in CartesianIndices(Fu_view)
        k, j, i = Tuple(I)
        if mode == :spectral
            kx = _spectral_wavenumber(i, nx, domain.Lx, T)
            ky = _spectral_wavenumber(j, ny, domain.Ly, T)
            kz = _spectral_wavenumber(k, nz, 2 * domain.Lz, T)
            sym = kx*kx + ky*ky + kz*kz
            factor = iszero(sym) ? zero(T) : -one(T) / sym
        else
            mx = T(i - 1)
            my = T(j - 1)
            mz = T(k - 1)
            sym = T(2) * (cos(T(2π) * mx / T(nx)) - one(T)) / (dx * dx) +
                  T(2) * (cos(T(2π) * my / T(ny)) - one(T)) / (dy * dy) +
                  T(2) * (cos(T(2π) * mz / T(nz)) - one(T)) / (dz * dz)
            factor = iszero(sym) ? zero(T) : fd_scale / sym
        end
        Fu_view[I] *= factor
        Fv_view[I] *= factor
        Fw_view[I] *= factor
    end

    ldiv!(workspace.ux_local, workspace.fft_plan, workspace.Fu)
    ldiv!(workspace.uy_local, workspace.fft_plan, workspace.Fv)
    ldiv!(workspace.uz_local, workspace.fft_plan, workspace.Fw)

    fill!(workspace.velocity_local_global, zero(T))

    ux_view = workspace.ux_view
    uy_view = workspace.uy_view
    uz_view = workspace.uz_view
    @inbounds for I in CartesianIndices(ux_view)
        k, j, i = Tuple(I)
        idx = k + (j - 1) * nz + (i - 1) * nz * ny
        workspace.velocity_local_global[idx,1] = real(ux_view[I])
        workspace.velocity_local_global[idx,2] = real(uy_view[I])
        workspace.velocity_local_global[idx,3] = real(uz_view[I])
    end

    MPI.Allreduce!(workspace.velocity_local_global, workspace.velocity_global, MPI.SUM, comm)

    @inbounds for idx in eachindex(ux, uy, uz)
        ux[idx] = workspace.velocity_global[idx,1]
        uy[idx] = workspace.velocity_global[idx,2]
        uz[idx] = workspace.velocity_global[idx,3]
    end
    return ux, uy, uz
end

function poisson_velocity_pencil_fft!(cache::Base.RefValue{Any},
                                      ux::Array{T,3}, uy::Array{T,3}, uz::Array{T,3},
                                      u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                                      domain::DomainSpec; mode::Symbol=:spectral,
                                      boundary_condition::Symbol=:periodic) where T<:AbstractFloat
    require_periodic_boundary(boundary_condition)
    mode == :spectral || mode == :fd ||
        throw(ArgumentError("Unknown Poisson mode: $mode (use :spectral or :fd)"))
    size(ux) == size(u_rhs) ||
        throw(DimensionMismatch("ux and u_rhs must have matching sizes"))
    size(uy) == size(v_rhs) ||
        throw(DimensionMismatch("uy and v_rhs must have matching sizes"))
    size(uz) == size(w_rhs) ||
        throw(DimensionMismatch("uz and w_rhs must have matching sizes"))

    init_mpi!()
    comm = MPI.COMM_WORLD
    dims = size(u_rhs)
    workspace = _pencil_poisson_workspace!(cache, T, dims, comm)
    return _poisson_velocity_pencil_fft!(workspace, ux, uy, uz,
                                         u_rhs, v_rhs, w_rhs,
                                         domain, comm, mode)
end

# MPI wrapper: compute Poisson solve on rank 0 and broadcast to all ranks (original implementation)
function poisson_velocity_fft_mpi(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3},
                                  domain::DomainSpec; mode::Symbol=:spectral,
                                  boundary_condition::Symbol=:periodic)
    require_periodic_boundary(boundary_condition)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Ux = Array{Float64}(undef, size(u_rhs))
    Uy = Array{Float64}(undef, size(v_rhs))
    Uz = Array{Float64}(undef, size(w_rhs))
    if rank == 0
        Ux0, Uy0, Uz0 = poisson_velocity_fft(u_rhs, v_rhs, w_rhs, domain;
                                             mode=mode, boundary_condition=boundary_condition)
        Ux .= Ux0; Uy .= Uy0; Uz .= Uz0
    end
    MPI.Bcast!(Ux, 0, comm)
    MPI.Bcast!(Uy, 0, comm)
    MPI.Bcast!(Uz, 0, comm)
    return Ux, Uy, Uz
end

function poisson_velocity_fft_mpi!(Ux::Array{T,3}, Uy::Array{T,3}, Uz::Array{T,3},
                                   Fu::Array{Complex{T},3},
                                   Fv::Array{Complex{T},3},
                                   Fw::Array{Complex{T},3},
                                   u_rhs::Array{T,3}, v_rhs::Array{T,3}, w_rhs::Array{T,3},
                                   domain::DomainSpec; mode::Symbol=:spectral,
                                   boundary_condition::Symbol=:periodic) where T<:AbstractFloat
    require_periodic_boundary(boundary_condition)
    init_mpi!()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    if rank == 0
        poisson_velocity_fft!(Ux, Uy, Uz, Fu, Fv, Fw, u_rhs, v_rhs, w_rhs, domain;
                              mode=mode, boundary_condition=boundary_condition)
    end
    MPI.Bcast!(Ux, 0, comm)
    MPI.Bcast!(Uy, 0, comm)
    MPI.Bcast!(Uz, 0, comm)
    return Ux, Uy, Uz
end
