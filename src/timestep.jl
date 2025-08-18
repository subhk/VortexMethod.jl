module TimeStepper

using ..DomainImpl
using ..Poisson3D
using ..Peskin3D
using ..Circulation
using ..Dissipation

export node_velocities, rk2_step!, rk2_step_with_dissipation!, grid_velocity, make_velocity_sampler

"""
grid_velocity(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=:spectral)

Computes the grid velocity fields (Ux,Uy,Uz) from element vorticity without interpolating to nodes.
Performs spreading to grid, curl RHS, and Poisson solve.
"""
function grid_velocity(eleGma, triXC, triYC, triZC, dom::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
    dx,dy,dz = grid_spacing(dom, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    if parallel_fft
        return poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    else
        return poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    end
end

"""
make_velocity_sampler(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=:spectral)

Returns a closure (x,y,z) -> (u,v,w) that interpolates velocity from a precomputed grid
velocity field built from the provided element vorticity and geometry. Useful to avoid
recomputing spread/Poisson repeatedly within a timestep.
"""
function make_velocity_sampler(eleGma, triXC, triYC, triZC, dom::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    Ux, Uy, Uz = grid_velocity(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    return (x::Float64, y::Float64, z::Float64) -> begin
        u, v, w = interpolate_node_velocity_mpi(Ux, Uy, Uz, (Float64[x]), (Float64[y]), (Float64[z]), dom, gr)
        (u[1], v[1], w[1])
    end
end

function node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
    dx,dy,dz = grid_spacing(dom, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    end
    u, v, w = interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, dom, gr)
    return u, v, w
end

function max_grid_speed(eleGma, triXC, triYC, triZC, dom::DomainSpec, gr::GridSpec; poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    VorX, VorY, VorZ = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, dom, gr)
    dx,dy,dz = grid_spacing(dom, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
    end
    # compute max |U| over grid and reduce across ranks
    magmax_local = maximum(sqrt.(Ux.^2 .+ Uy.^2 .+ Uz.^2))
    magmax = MPI.Allreduce(magmax_local, MPI.MAX, MPI.COMM_WORLD)
    return magmax
end

function rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, dom::DomainSpec, gr::GridSpec, dt::Float64;
                   At::Float64=0.0, adaptive::Bool=false, CFL::Float64=0.5, poisson_mode::Symbol=:spectral, parallel_fft::Bool=false)
    # velocities at t^n
    triXC = similar(eleGma, size(tri,1), 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    # compute node circulation from current gamma
    nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
    # adaptive dt based on grid max speed if requested
    if adaptive
        dx,dy,_ = grid_spacing(dom, gr)
        umax = max_grid_speed(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
        dt = CFL * min(dx,dy) / max(umax, 1e-12)
    end
    u1, v1, w1 = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    # half-step positions
    xh = nodeX .+ 0.5 .* dt .* u1
    yh = nodeY .+ 0.5 .* dt .* v1
    zh = nodeZ .+ 0.5 .* dt .* w1
    # periodic wrap
    xh .= mod.(xh, dom.Lx)
    yh .= mod.(yh, dom.Ly)
    zh .= mod.(zh .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # velocities at half step
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = xh[v]
        triYC[t,k] = yh[v]
        triZC[t,k] = zh[v]
    end
    # baroclinic update at half step
    if At != 0.0
        dGmid = baroclinic_ele_gamma(At, 0.5*dt, triXC, triYC, triZC)
        dTau = node_circulation_from_ele_gamma(triXC, triYC, triZC, dGmid)
        nodeCirc .+= dTau
    end
    # recompute gamma at half-step geometry from updated node circulation
    eleGma_mid = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
    u2, v2, w2 = node_velocities(eleGma_mid, triXC, triYC, triZC, xh, yh, zh, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    # full-step update
    nodeX .+= dt .* u2
    nodeY .+= dt .* v2
    nodeZ .+= dt .* w2
    nodeX .= mod.(nodeX, dom.Lx)
    nodeY .= mod.(nodeY, dom.Ly)
    nodeZ .= mod.(nodeZ .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # rebuild triangle coords at t^{n+1}
    triXC_new = similar(triXC); triYC_new = similar(triYC); triZC_new = similar(triZC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC_new[t,k] = nodeX[v]
        triYC_new[t,k] = nodeY[v]
        triZC_new[t,k] = nodeZ[v]
    end
    # second baroclinic update at end step
    if At != 0.0
        dGend = baroclinic_ele_gamma(At, 0.5*dt, triXC_new, triYC_new, triZC_new)
        dTau2 = node_circulation_from_ele_gamma(triXC_new, triYC_new, triZC_new, dGend)
        nodeCirc .+= dTau2
    end
    # produce gamma at new geometry from node circulation
    eleGma_new = ele_gamma_from_node_circ(nodeCirc, triXC_new, triYC_new, triZC_new)
    eleGma .= eleGma_new

    return dt
end

# Enhanced RK2 time stepping with dissipation models
function rk2_step_with_dissipation!(nodeX, nodeY, nodeZ, tri, eleGma, dom::DomainSpec, gr::GridSpec, dt::Float64,
                                   dissipation_model::DissipationModel=NoDissipation();
                                   At::Float64=0.0, adaptive::Bool=false, CFL::Float64=0.5, 
                                   poisson_mode::Symbol=:spectral, parallel_fft::Bool=false, kernel::KernelType=PeskinStandard())
    # velocities at t^n
    triXC = similar(eleGma, size(tri,1), 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    
    # Apply dissipation at beginning of step
    eleGma = apply_dissipation!(dissipation_model, eleGma, triXC, triYC, triZC, dom, gr, 0.5*dt)
    
    # compute node circulation from current gamma
    nodeCirc = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma)
    
    # adaptive dt based on grid max speed if requested
    if adaptive
        dx,dy,_ = grid_spacing(dom, gr)
        umax = max_grid_speed(eleGma, triXC, triYC, triZC, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
        dt = CFL * min(dx,dy) / max(umax, 1e-12)
    end
    
    # Use kernel-based spreading if specified
    if kernel != PeskinStandard()
        VorX, VorY, VorZ = spread_vorticity_to_grid_kernel_mpi(eleGma, triXC, triYC, triZC, dom, gr, kernel)
        dx,dy,dz = grid_spacing(dom, gr)
        u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
        if parallel_fft
            Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
        else
            Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
        end
        u1, v1, w1 = interpolate_node_velocity_kernel_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, dom, gr, kernel)
    else
        u1, v1, w1 = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    end

    # half-step positions
    xh = nodeX .+ 0.5 .* dt .* u1
    yh = nodeY .+ 0.5 .* dt .* v1
    zh = nodeZ .+ 0.5 .* dt .* w1
    # periodic wrap
    xh .= mod.(xh, dom.Lx)
    yh .= mod.(yh, dom.Ly)
    zh .= mod.(zh .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # velocities at half step
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = xh[v]
        triYC[t,k] = yh[v]
        triZC[t,k] = zh[v]
    end
    
    # baroclinic update at half step
    if At != 0.0
        dGmid = baroclinic_ele_gamma(At, 0.5*dt, triXC, triYC, triZC)
        dTau = node_circulation_from_ele_gamma(triXC, triYC, triZC, dGmid)
        nodeCirc .+= dTau
    end
    
    # recompute gamma at half-step geometry from updated node circulation
    eleGma_mid = ele_gamma_from_node_circ(nodeCirc, triXC, triYC, triZC)
    
    # Apply dissipation at mid-step
    eleGma_mid = apply_dissipation!(dissipation_model, eleGma_mid, triXC, triYC, triZC, dom, gr, 0.5*dt)
    
    if kernel != PeskinStandard()
        VorX, VorY, VorZ = spread_vorticity_to_grid_kernel_mpi(eleGma_mid, triXC, triYC, triZC, dom, gr, kernel)
        dx,dy,dz = grid_spacing(dom, gr)
        u_rhs, v_rhs, w_rhs = curl_rhs_centered(VorX, VorY, VorZ, dx, dy, dz)
        if parallel_fft
            Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
        else
            Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, dom; mode=poisson_mode)
        end
        u2, v2, w2 = interpolate_node_velocity_kernel_mpi(Ux, Uy, Uz, xh, yh, zh, dom, gr, kernel)
    else
        u2, v2, w2 = node_velocities(eleGma_mid, triXC, triYC, triZC, xh, yh, zh, dom, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    end

    # full-step update
    nodeX .+= dt .* u2
    nodeY .+= dt .* v2
    nodeZ .+= dt .* w2
    nodeX .= mod.(nodeX, dom.Lx)
    nodeY .= mod.(nodeY, dom.Ly)
    nodeZ .= mod.(nodeZ .+ dom.Lz, 2*dom.Lz) .- dom.Lz

    # rebuild triangle coords at t^{n+1}
    triXC_new = similar(triXC); triYC_new = similar(triYC); triZC_new = similar(triZC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC_new[t,k] = nodeX[v]
        triYC_new[t,k] = nodeY[v]
        triZC_new[t,k] = nodeZ[v]
    end
    
    # second baroclinic update at end step
    if At != 0.0
        dGend = baroclinic_ele_gamma(At, 0.5*dt, triXC_new, triYC_new, triZC_new)
        dTau2 = node_circulation_from_ele_gamma(triXC_new, triYC_new, triZC_new, dGend)
        nodeCirc .+= dTau2
    end
    
    # produce gamma at new geometry from node circulation
    eleGma_new = ele_gamma_from_node_circ(nodeCirc, triXC_new, triYC_new, triZC_new)
    eleGma .= eleGma_new

    return dt
end

end # module

using .TimeStepper: node_velocities, rk2_step!, rk2_step_with_dissipation!
