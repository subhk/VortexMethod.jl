module Energy

using ..DomainImpl
using ..Poisson3D
using ..Peskin3D

export grid_ke, gamma_ke

function grid_ke(Ux::Array{Float64,3}, Uy::Array{Float64,3}, Uz::Array{Float64,3}, dom::DomainSpec, gr::GridSpec)
    dx,dy,dz = (dom.Lx/(gr.nx-1), dom.Ly/(gr.ny-1), (2*dom.Lz)/(gr.nz-1))
    KE = 0.5 * sum(Ux.^2 .+ Uy.^2 .+ Uz.^2) * dx * dy * dz
    return KE
end

function gamma_ke(eleGma::AbstractMatrix,
                  triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                  dom::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
    dx,dy,dz = grid_spacing(dom, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    end
    return grid_ke(Ux,Uy,Uz, dom, gr)
end

end # module

