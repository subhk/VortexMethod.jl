# 3D FFT-based Poisson solver and curl RHS

module Poisson3D

using FFTW
using MPI
using PencilFFTs
using LinearAlgebra
using ..DomainImpl

export curl_rhs_centered, curl_rhs_centered!, PoissonWorkspace, 
       poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft

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

function init_mpi!()
    MPI.Finalized() && throw(ErrorException("MPI has already been finalized"))
    MPI.Initialized() || MPI.Init()
    return nothing
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

end # module

using .Poisson3D: curl_rhs_centered, curl_rhs_centered!, PoissonWorkspace, 
                   poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft
