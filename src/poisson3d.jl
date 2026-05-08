# 3D FFT-based Poisson solver and curl RHS

module Poisson3D

using FFTW
using MPI
using PencilFFTs
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
                           VorX::AbstractArray{T,3}, VorY::AbstractArray{T,3}, VorZ::AbstractArray{T,3},
                           dx::T, dy::T, dz::T) where T<:AbstractFloat
    nz, ny, nx = size(VorX)

    @inbounds for k in 1:nz, j in 1:ny, i in 1:nx
        ip = next_periodic_index(i, nx)
        im = prev_periodic_index(i, nx)
        jp = next_periodic_index(j, ny)
        jm = prev_periodic_index(j, ny)
        kp = next_periodic_index(k, nz)
        km = prev_periodic_index(k, nz)

        dZ_dy = (VorZ[k,jp,i] - VorZ[k,jm,i]) / (2dy)
        dY_dz = (VorY[kp,j,i] - VorY[km,j,i]) / (2dz)
        dX_dz = (VorX[kp,j,i] - VorX[km,j,i]) / (2dz)
        dZ_dx = (VorZ[k,j,ip] - VorZ[k,j,im]) / (2dx)
        dY_dx = (VorY[k,j,ip] - VorY[k,j,im]) / (2dx)
        dX_dy = (VorX[k,jp,i] - VorX[k,jm,i]) / (2dy)

        u_rhs[k,j,i] = -(dZ_dy - dY_dz)
        v_rhs[k,j,i] = -(dX_dz - dZ_dx)
        w_rhs[k,j,i] = -(dY_dx - dX_dy)
    end
    
    return nothing
end

# Backward-compatible wrapper that allocates
function curl_rhs_centered(VorX::AbstractArray{Float64,3}, VorY::AbstractArray{Float64,3}, VorZ::AbstractArray{Float64,3},
                           dx::Float64, dy::Float64, dz::Float64)
    nz, ny, nx = size(VorX)
    workspace = PoissonWorkspace(nz, ny, nx)
    u_rhs = similar(VorX)
    v_rhs = similar(VorX)
    w_rhs = similar(VorX)
    curl_rhs_centered!(workspace, u_rhs, v_rhs, w_rhs, VorX, VorY, VorZ, dx, dy, dz)
    return u_rhs, v_rhs, w_rhs
end

function require_periodic_boundary(boundary_condition::Symbol)
    boundary_condition == :periodic && return nothing
    throw(ArgumentError("poisson_velocity_fft supports only periodic FFT boundaries; requested boundary_condition=$boundary_condition"))
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
    comm = MPI.COMM_WORLD
    nz, ny, nx = size(u_rhs)
    
    # Create pencil decomposition for 3D FFTs
    # PencilFFTs typically uses (z,y,x) ordering for 3D arrays
    pen = Pencil((nz, ny, nx), comm; permute_dims=(1,2,3))
    
    # Create FFT plans
    fft_plan = PencilFFTPlans(pen, Float64, FFT!)
    
    dx = domain.Lx/nx
    dy = domain.Ly/ny
    dz = (2*domain.Lz)/nz
    
    kx = kvec(nx, domain.Lx)
    ky = kvec(ny, domain.Ly)
    kz = kvec(nz, 2*domain.Lz)
    
    # Get local array dimensions for this MPI rank
    local_dims = size_local(pen, LogicalOrder())
    
    # Create local wavenumber grids for this rank's subdomain
    local_range_z, local_range_y, local_range_x = range_local(pen, LogicalOrder())
    
    if mode == :spectral
        KX = reshape(kx[local_range_x], 1, 1, length(local_range_x))
        KY = reshape(ky[local_range_y], 1, length(local_range_y), 1) 
        KZ = reshape(kz[local_range_z], length(local_range_z), 1, 1)
        sym = KX.^2 .+ KY.^2 .+ KZ.^2
    elseif mode == :fd
        # Discrete Laplacian symbol: 2(cos(2πm/n) - 1) / h² for 3-point central difference
        mx = collect(local_range_x .- 1); my = collect(local_range_y .- 1); mz = collect(local_range_z .- 1)
        CX = reshape(2.0 .* (cos.(2pi .* mx ./ nx) .- 1.0), 1, 1, length(mx)) ./ (dx^2)
        CY = reshape(2.0 .* (cos.(2pi .* my ./ ny) .- 1.0), 1, length(my), 1) ./ (dy^2)
        CZ = reshape(2.0 .* (cos.(2pi .* mz ./ nz) .- 1.0), length(mz), 1, 1) ./ (dz^2)
        sym = CX .+ CY .+ CZ
    else
        error("Unknown Poisson mode: $mode (use :spectral or :fd)")
    end
    
    # Allocate local arrays for this rank
    u_local = allocate_input(fft_plan)
    v_local = allocate_input(fft_plan)
    w_local = allocate_input(fft_plan)
    
    # Copy input data to local arrays (assuming input is already distributed)
    # In practice, you may need to distribute the data from global arrays
    u_local .= u_rhs[local_range_z, local_range_y, local_range_x]
    v_local .= v_rhs[local_range_z, local_range_y, local_range_x]
    w_local .= w_rhs[local_range_z, local_range_y, local_range_x]
    
    # Perform forward FFTs
    Fu = allocate_output(fft_plan)
    Fv = allocate_output(fft_plan)
    Fw = allocate_output(fft_plan)
    
    mul!(Fu, fft_plan, u_local)
    mul!(Fv, fft_plan, v_local)
    mul!(Fw, fft_plan, w_local)
    
    # Apply Poisson operator in Fourier space
    # Handle k=0 mode to avoid division by zero
    if 1 in local_range_z && 1 in local_range_y && 1 in local_range_x
        local_i = findfirst(x -> x == 1, local_range_z)
        local_j = findfirst(x -> x == 1, local_range_y) 
        local_k = findfirst(x -> x == 1, local_range_x)
        sym[local_i, local_j, local_k] = 1.0
    end
    
    if mode == :fd
        scale = 0.5/(domain.Lx*domain.Ly*domain.Lz)
        Û = scale .* Fu ./ sym
        V̂ = scale .* Fv ./ sym
        Ŵ = scale .* Fw ./ sym
    else
        Û = -Fu ./ sym
        V̂ = -Fv ./ sym  
        Ŵ = -Fw ./ sym
    end
    
    # Set k=0 mode to zero
    if 1 in local_range_z && 1 in local_range_y && 1 in local_range_x
        local_i = findfirst(x -> x == 1, local_range_z)
        local_j = findfirst(x -> x == 1, local_range_y)
        local_k = findfirst(x -> x == 1, local_range_x)
        Û[local_i, local_j, local_k] = 0.0 + 0.0im
        V̂[local_i, local_j, local_k] = 0.0 + 0.0im
        Ŵ[local_i, local_j, local_k] = 0.0 + 0.0im
    end
    
    # Perform inverse FFTs
    ux_local = allocate_input(fft_plan)
    uy_local = allocate_input(fft_plan)
    uz_local = allocate_input(fft_plan)
    
    ldiv!(ux_local, fft_plan, Û)
    ldiv!(uy_local, fft_plan, V̂)
    ldiv!(uz_local, fft_plan, Ŵ)
    
    # Gather results to global arrays using MPI.Allreduce
    # Each rank writes its local portion to a global-sized array (zeros elsewhere),
    # then Allreduce with SUM combines all the non-overlapping pieces
    ux_local_global = zeros(Float64, nz, ny, nx)
    uy_local_global = zeros(Float64, nz, ny, nx)
    uz_local_global = zeros(Float64, nz, ny, nx)

    # Each rank writes its local data to the appropriate region
    ux_local_global[local_range_z, local_range_y, local_range_x] .= real.(ux_local)
    uy_local_global[local_range_z, local_range_y, local_range_x] .= real.(uy_local)
    uz_local_global[local_range_z, local_range_y, local_range_x] .= real.(uz_local)

    # Combine all ranks' contributions (non-overlapping regions, so SUM works)
    ux_gathered = zeros(Float64, nz, ny, nx)
    uy_gathered = zeros(Float64, nz, ny, nx)
    uz_gathered = zeros(Float64, nz, ny, nx)

    MPI.Allreduce!(ux_local_global, ux_gathered, MPI.SUM, comm)
    MPI.Allreduce!(uy_local_global, uy_gathered, MPI.SUM, comm)
    MPI.Allreduce!(uz_local_global, uz_gathered, MPI.SUM, comm)
    
    # Note: Periodic boundary conditions are handled automatically by FFT
    
    return ux_gathered, uy_gathered, uz_gathered
end

# MPI wrapper: compute Poisson solve on rank 0 and broadcast to all ranks (original implementation)
function poisson_velocity_fft_mpi(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3},
                                  domain::DomainSpec; mode::Symbol=:spectral,
                                  boundary_condition::Symbol=:periodic)
    require_periodic_boundary(boundary_condition)
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
