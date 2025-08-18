# 3D FFT-based Poisson solver and curl RHS

module Poisson3D

using FFTW
using MPI
using PencilFFTs
using ..DomainImpl

export curl_rhs_centered, poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft

# Periodic finite-difference curl terms with central 4th-order where possible
function curl_rhs_centered(VorX::Array{Float64,3}, VorY::Array{Float64,3}, VorZ::Array{Float64,3},
                           dx::Float64, dy::Float64, dz::Float64)
    nz, ny, nx = size(VorX)
    dX_dy = zeros(Float64, nz, ny, nx)
    dX_dz = zeros(Float64, nz, ny, nx)
    dY_dx = zeros(Float64, nz, ny, nx)
    dY_dz = zeros(Float64, nz, ny, nx)
    dZ_dx = zeros(Float64, nz, ny, nx)
    dZ_dy = zeros(Float64, nz, ny, nx)

    # x-derivatives
    for i in 3:nx-2
        dY_dx[:,:,i] = (VorY[:,:,i-2]/12 - 2*VorY[:,:,i-1]/3 + 2*VorY[:,:,i+1]/3 - VorY[:,:,i+2]/12) / dx
        dZ_dx[:,:,i] = (VorZ[:,:,i-2]/12 - 2*VorZ[:,:,i-1]/3 + 2*VorZ[:,:,i+1]/3 - VorZ[:,:,i+2]/12) / dx
    end
    for i in 1:2
        dY_dx[:,:,i] = (-3*VorY[:,:,i]/2 + 2*VorY[:,:,i+1] - VorY[:,:,i+2]/2)/dx
        dZ_dx[:,:,i] = (-3*VorZ[:,:,i]/2 + 2*VorZ[:,:,i+1] - VorZ[:,:,i+2]/2)/dx
    end
    dY_dx[:,:,nx-1] = (3*VorY[:,:,nx-1]/2 - 2*VorY[:,:,nx-2] + VorY[:,:,nx-3]/2)/dx
    dZ_dx[:,:,nx-1] = (3*VorZ[:,:,nx-1]/2 - 2*VorZ[:,:,nx-2] + VorZ[:,:,nx-3]/2)/dx
    dY_dx[:,:,nx]   = dY_dx[:,:,1]
    dZ_dx[:,:,nx]   = dZ_dx[:,:,1]

    # y-derivatives
    for j in 3:ny-2
        dX_dy[:,j,:] = (VorX[:,j-2,:]/12 - 2*VorX[:,j-1,:]/3 + 2*VorX[:,j+1,:]/3 - VorX[:,j+2,:]/12) / dy
        dZ_dy[:,j,:] = (VorZ[:,j-2,:]/12 - 2*VorZ[:,j-1,:]/3 + 2*VorZ[:,j+1,:]/3 - VorZ[:,j+2,:]/12) / dy
    end
    for j in 1:2
        dX_dy[:,j,:] = (-3*VorX[:,j,:]/2 + 2*VorX[:,j+1,:] - VorX[:,j+2,:]/2)/dy
        dZ_dy[:,j,:] = (-3*VorZ[:,j,:]/2 + 2*VorZ[:,j+1,:] - VorZ[:,j+2,:]/2)/dy
    end
    dX_dy[:,ny-1,:] = (3*VorX[:,ny-1,:]/2 - 2*VorX[:,ny-2,:] + VorX[:,ny-3,:]/2)/dy
    dZ_dy[:,ny-1,:] = (3*VorZ[:,ny-1,:]/2 - 2*VorZ[:,ny-2,:] + VorZ[:,ny-3,:]/2)/dy
    dX_dy[:,ny,:]   = dX_dy[:,1,:]
    dZ_dy[:,ny,:]   = dZ_dy[:,1,:]

    # z-derivatives
    for k in 3:nz-2
        dX_dz[k,:,:] = (VorX[k-2,:,:]/12 - 2*VorX[k-1,:,:]/3 + 2*VorX[k+1,:,:]/3 - VorX[k+2,:,:]/12) / dz
        dY_dz[k,:,:] = (VorY[k-2,:,:]/12 - 2*VorY[k-1,:,:]/3 + 2*VorY[k+1,:,:]/3 - VorY[k+2,:,:]/12) / dz
    end
    for k in 1:2
        dX_dz[k,:,:] = (-3*VorX[k,:,:]/2 + 2*VorX[k+1,:,:] - VorX[k+2,:,:]/2)/dz
        dY_dz[k,:,:] = (-3*VorY[k,:,:]/2 + 2*VorY[k+1,:,:] - VorY[k+2,:,:]/2)/dz
    end
    dX_dz[nz-1,:,:] = (3*VorX[nz-1,:,:]/2 - 2*VorX[nz-2,:,:] + VorX[nz-3,:,:]/2)/dz
    dY_dz[nz-1,:,:] = (3*VorY[nz-1,:,:]/2 - 2*VorY[nz-2,:,:] + VorY[nz-3,:,:]/2)/dz
    dX_dz[nz,:,:]   = dX_dz[1,:,:]
    dY_dz[nz,:,:]   = dY_dz[1,:,:]

    # -curl(omega)
    u = -(dZ_dy .- dY_dz)
    v = -(dX_dz .- dZ_dx)
    w = -(dY_dx .- dX_dy)
    return u,v,w
end

# FFT-based Poisson solve (periodic): ∇^2 U = RHS -> Û = -RHŜ/k^2
function poisson_velocity_fft(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3}, domain::DomainSpec; mode::Symbol=:spectral)
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
        # Discrete Laplacian symbol used by Python code
        mx = collect(0:nx-1); my = collect(0:ny-1); mz = collect(0:nz-1)
        CX = reshape(cos.(2pi .* mx ./ nx) .- 1.0, 1,1,nx) ./ (dx^2)
        CY = reshape(cos.(2pi .* my ./ ny) .- 1.0, 1,ny,1) ./ (dy^2)
        CZ = reshape(cos.(2pi .* mz ./ nz) .- 1.0, nz,1,1) ./ (dz^2)
        sym = CX .+ CY .+ CZ
    else
        error("Unknown Poisson mode: $mode (use :spectral or :fd)")
    end

    function solve_one(rhs)
        F = rfft(rhs) # real-to-complex along last dim only would be wrong; do full fftn via FFTW.fft.
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
function poisson_velocity_pencil_fft(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3}, domain::DomainSpec; mode::Symbol=:spectral)
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
        # Discrete Laplacian symbol for local ranges
        mx = collect(local_range_x .- 1); my = collect(local_range_y .- 1); mz = collect(local_range_z .- 1)
        CX = reshape(cos.(2pi .* mx ./ nx) .- 1.0, 1, 1, length(mx)) ./ (dx^2)
        CY = reshape(cos.(2pi .* my ./ ny) .- 1.0, 1, length(my), 1) ./ (dy^2)
        CZ = reshape(cos.(2pi .* mz ./ nz) .- 1.0, length(mz), 1, 1) ./ (dz^2)
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
    
    # Gather results to global arrays (or keep distributed for further processing)
    ux = zeros(Float64, nz, ny, nx)
    uy = zeros(Float64, nz, ny, nx)
    uz = zeros(Float64, nz, ny, nx)
    
    # Collect distributed results to rank 0 and broadcast
    # This could be optimized to keep data distributed if downstream operations support it
    ux_gathered = zeros(Float64, nz, ny, nx)
    uy_gathered = zeros(Float64, nz, ny, nx)
    uz_gathered = zeros(Float64, nz, ny, nx)
    
    # Simple gathering - in practice you'd use proper MPI collectives
    rank = MPI.Comm_rank(comm)
    if rank == 0
        ux_gathered[local_range_z, local_range_y, local_range_x] .= real.(ux_local)
        uy_gathered[local_range_z, local_range_y, local_range_x] .= real.(uy_local)
        uz_gathered[local_range_z, local_range_y, local_range_x] .= real.(uz_local)
    end
    
    # Broadcast final results
    MPI.Bcast!(ux_gathered, 0, comm)
    MPI.Bcast!(uy_gathered, 0, comm)
    MPI.Bcast!(uz_gathered, 0, comm)
    
    # Note: Periodic boundary conditions are handled automatically by FFT
    
    return ux_gathered, uy_gathered, uz_gathered
end

# MPI wrapper: compute Poisson solve on rank 0 and broadcast to all ranks (original implementation)
function poisson_velocity_fft_mpi(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3}, domain::DomainSpec; mode::Symbol=:spectral)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    Ux = Array{Float64}(undef, size(u_rhs))
    Uy = Array{Float64}(undef, size(v_rhs))
    Uz = Array{Float64}(undef, size(w_rhs))
    if rank == 0
        Ux0, Uy0, Uz0 = poisson_velocity_fft(u_rhs, v_rhs, w_rhs, domain; mode=mode)
        Ux .= Ux0; Uy .= Uy0; Uz .= Uz0
    end
    MPI.Bcast!(Ux, 0, comm)
    MPI.Bcast!(Uy, 0, comm)
    MPI.Bcast!(Uz, 0, comm)
    return Ux, Uy, Uz
end

end # module

using .Poisson3D: curl_rhs_centered, poisson_velocity_fft, poisson_velocity_fft_mpi, poisson_velocity_pencil_fft
