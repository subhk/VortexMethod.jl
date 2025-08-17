# 3D FFT-based Poisson solver and curl RHS

module Poisson3D

using FFTW
using ..DomainImpl

export curl_rhs_centered, poisson_velocity_fft

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
function poisson_velocity_fft(u_rhs::Array{Float64,3}, v_rhs::Array{Float64,3}, w_rhs::Array{Float64,3}, dom::DomainSpec)
    nz, ny, nx = size(u_rhs)
    dx = dom.Lx/(nx-1)
    dy = dom.Ly/(ny-1)
    dz = (2*dom.Lz)/(nz-1)

    kx = kvec(nx, dom.Lx)
    ky = kvec(ny, dom.Ly)
    kz = kvec(nz, 2*dom.Lz)

    # Create grids of wavenumbers
    KX = reshape(kx, 1,1,nx)
    KY = reshape(ky, 1,ny,1)
    KZ = reshape(kz, nz,1,1)
    k2 = KX.^2 .+ KY.^2 .+ KZ.^2

    function solve_one(rhs)
        F = rfft(rhs) # real-to-complex along last dim only would be wrong; do full fftn via FFTW.fft.
    end

    # Use full n-dimensional FFT
    Fu = FFTW.fft(u_rhs)
    Fv = FFTW.fft(v_rhs)
    Fw = FFTW.fft(w_rhs)

    # Avoid division by zero at k=0
    k2[1,1,1] = 1.0
    Û = -Fu ./ k2
    V̂ = -Fv ./ k2
    Ŵ = -Fw ./ k2
    Û[1,1,1] = 0.0 + 0.0im
    V̂[1,1,1] = 0.0 + 0.0im
    Ŵ[1,1,1] = 0.0 + 0.0im

    ux = real(FFTW.ifft(Û))
    uy = real(FFTW.ifft(V̂))
    uz = real(FFTW.ifft(Ŵ))

    # periodic wrap to mimic python behavior
    ux[end, :, :] .= ux[1, :, :]
    uy[end, :, :] .= uy[1, :, :]
    uz[end, :, :] .= uz[1, :, :]
    ux[:, end, :] .= ux[:, 1, :]
    uy[:, end, :] .= uy[:, 1, :]
    uz[:, end, :] .= uz[:, 1, :]
    ux[:, :, end] .= ux[:, :, 1]
    uy[:, :, end] .= uy[:, :, 1]
    uz[:, :, end] .= uz[:, :, 1]

    return ux, uy, uz
end

end # module

using .Poisson3D: curl_rhs_centered, poisson_velocity_fft

