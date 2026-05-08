module Energy

using ..DomainImpl
using ..Poisson3D
using ..Peskin3D

export grid_ke, gamma_ke

function grid_ke(Ux::Array{Float64,3}, Uy::Array{Float64,3}, Uz::Array{Float64,3}, domain::DomainSpec, gr::GridSpec)
    # Use grid_spacing for periodic domain (step = L/n, not L/(n-1))
    dx, dy, dz = grid_spacing(domain, gr)
    s = 0.0
    @inbounds @simd for i in eachindex(Ux, Uy, Uz)
        s += Ux[i] * Ux[i] + Uy[i] * Uy[i] + Uz[i] * Uz[i]
    end
    return 0.5 * s * dx * dy * dz
end

function gamma_ke(eleGma::AbstractMatrix,
                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                  domain::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    dx,dy,dz = grid_spacing(domain, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    end
    return grid_ke(Ux,Uy,Uz, domain, gr)
end

end # module

using .Energy: grid_ke, gamma_ke
