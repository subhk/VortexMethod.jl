module TimeStepper

using MPI
using ..DomainImpl
using ..Poisson3D
using ..Peskin3D
using ..Circulation
using ..Dissipation
using ..Kernels
using ..Workspace: VortexWorkspace

export node_velocities, node_velocities!, rk2_step!, rk2_step_with_dissipation!,
       grid_velocity, grid_velocity!, make_velocity_sampler

function triangle_coords_from_nodes!(triXC, triYC, triZC, nodeX, nodeY, nodeZ, tri)
    @inbounds for k in 1:3, t in 1:size(tri, 1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    return nothing
end

function refresh_workspace_geometry!(ws::VortexWorkspace, triXC, triYC, triZC,
                                     domain::DomainSpec, gr::GridSpec)
    Peskin3D._recompute_geometry!(ws, triXC, triYC, triZC, domain, gr)
    ws.geom_dirty[] = false
    return nothing
end

"""
grid_velocity(eleGma, triXC, triYC, triZC, domain, gr; poisson_mode=:spectral, parallel_fft=false)

Computes the grid velocity fields (Ux,Uy,Uz) from element vorticity without interpolating to nodes.
Performs spreading to grid, curl RHS, and Poisson solve.

# Arguments
- `poisson_mode::Symbol=:spectral`: FFT mode (`:spectral` or `:fd`)
- `parallel_fft::Bool=false`: Use PencilFFTs for distributed parallel FFT instead of rank-0 broadcast
"""
function grid_velocity(eleGma, triXC, triYC, triZC, 
                    domain::DomainSpec, 
                    gr::GridSpec; 
                    poisson_mode::Symbol=:spectral, 
                    parallel_fft::Bool=false)

    ζx, ζy, ζz = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    dx,dy,dz = grid_spacing(domain, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)

    if parallel_fft
        return poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    else
        return poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    end
end

function grid_velocity!(ws::VortexWorkspace{T},
                        eleGma::AbstractMatrix{T},
                        triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                        domain::DomainSpec,
                        gr::GridSpec;
                        poisson_mode::Symbol=:spectral,
                        parallel_fft::Bool=false) where T<:AbstractFloat
    spread_vorticity_to_grid_mpi!(ws, eleGma, triXC, triYC, triZC, domain, gr)
    dx, dy, dz = grid_spacing(domain, gr)
    curl_rhs_centered!(PoissonWorkspace(T, gr.nz, gr.ny, gr.nx),
                       ws.rhs_x, ws.rhs_y, ws.rhs_z,
                       ws.ζx, ws.ζy, ws.ζz,
                       T(dx), T(dy), T(dz))
    if parallel_fft
        poisson_velocity_pencil_fft!(ws.pencil_poisson,
                                     ws.gridUx, ws.gridUy, ws.gridUz,
                                     ws.rhs_x, ws.rhs_y, ws.rhs_z,
                                     domain; mode=poisson_mode)
    else
        poisson_velocity_fft_mpi!(ws.gridUx, ws.gridUy, ws.gridUz,
                                  ws.fft_x, ws.fft_y, ws.fft_z,
                                  ws.rhs_x, ws.rhs_y, ws.rhs_z,
                                  domain; mode=poisson_mode)
    end
    return ws.gridUx, ws.gridUy, ws.gridUz
end

function node_velocities!(ws::VortexWorkspace{T},
                          out_u::AbstractVector{T}, out_v::AbstractVector{T}, out_w::AbstractVector{T},
                          eleGma::AbstractMatrix{T},
                          triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                          nodeX::AbstractVector{T}, nodeY::AbstractVector{T}, nodeZ::AbstractVector{T},
                          domain::DomainSpec,
                          gr::GridSpec;
                          poisson_mode::Symbol=:spectral,
                          parallel_fft::Bool=false) where T<:AbstractFloat
    grid_velocity!(ws, eleGma, triXC, triYC, triZC, domain, gr;
                   poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    interpolate_node_velocity_mpi!(out_u, out_v, out_w, ws,
                                   ws.gridUx, ws.gridUy, ws.gridUz,
                                   nodeX, nodeY, nodeZ, domain, gr)
    return out_u, out_v, out_w
end

function _interpolate_grid_point(gridUx::Array{T,3}, gridUy::Array{T,3}, gridUz::Array{T,3},
                                 x::Real, y::Real, z::Real,
                                 domain::DomainSpec, gr::GridSpec) where T<:AbstractFloat
    nx, ny, nz = gr.nx, gr.ny, gr.nz
    dx, dy, dz = grid_spacing(domain, gr)
    delr = T(4)
    epsx = delr * T(dx)
    epsy = delr * T(dy)
    epsz = delr * T(dz)
    scale = one(T) / (T(8) * delr^3)
    xc = T(x)
    yc = T(y)
    zc = T(z)
    sx_acc = zero(T)
    sy_acc = zero(T)
    sz_acc = zero(T)

    @inbounds for sx_tile in -1:1, sy_tile in -1:1, sz_tile in -1:1
        xq = xc - T(sx_tile) * T(domain.Lx)
        yq = yc - T(sy_tile) * T(domain.Ly)
        zq = zc - T(sz_tile) * T(2 * domain.Lz)
        for k in 1:nz, j in 1:ny, i in 1:nx
            dxv = xq - T(i - 1) * T(dx)
            dyv = yq - T(j - 1) * T(dy)
            dzv = zq - (-T(domain.Lz) + T(k - 1) * T(dz))
            if abs(dxv) <= epsx && abs(dyv) <= epsy && abs(dzv) <= epsz
                w = (one(T) + cos(T(π) * dxv / epsx)) *
                    (one(T) + cos(T(π) * dyv / epsy)) *
                    (one(T) + cos(T(π) * dzv / epsz)) * scale
                sx_acc += gridUx[k,j,i] * w
                sy_acc += gridUy[k,j,i] * w
                sz_acc += gridUz[k,j,i] * w
            end
        end
    end
    return sx_acc, sy_acc, sz_acc
end

"""
make_velocity_sampler(eleGma, triXC, triYC, triZC, domain::DomainSpec, gr::GridSpec; poisson_mode=:spectral, parallel_fft=false)

Returns a closure (x,y,z) -> (u,v,w) that interpolates velocity from a precomputed grid
velocity field built from the provided element vorticity and geometry. Useful to avoid
recomputing spread/Poisson repeatedly within a timestep.

# Arguments
- `poisson_mode::Symbol=:spectral`: FFT mode (`:spectral` or `:fd`)  
- `parallel_fft::Bool=false`: Use PencilFFTs for distributed parallel FFT instead of rank-0 broadcast
"""
function make_velocity_sampler(eleGma, triXC, triYC, triZC, 
                            domain::DomainSpec, 
                            gr::GridSpec; 
                            poisson_mode::Symbol=:spectral, 
                            parallel_fft::Bool=false)

    Ux, Uy, Uz = grid_velocity(eleGma, triXC, triYC, triZC, domain, gr; 
                            poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    return (x::Real, y::Real, z::Real) -> _interpolate_grid_point(Ux, Uy, Uz, x, y, z, domain, gr)
end

function node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, 
                    domain::DomainSpec, 
                    gr::GridSpec; 
                    poisson_mode::Symbol=:spectral, 
                    parallel_fft::Bool=false)
                    
    ζx, ζy, ζz = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    dx, dy, dz = grid_spacing(domain, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)

    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    end

    u, v, w = interpolate_node_velocity_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr)

    return u, v, w
end

function max_grid_speed(eleGma, triXC, triYC, triZC, 
                        domain::DomainSpec, gr::GridSpec; 
                        poisson_mode::Symbol=:spectral, 
                        parallel_fft::Bool=false)
                        
    ζx, ζy, ζz = spread_vorticity_to_grid_mpi(eleGma, triXC, triYC, triZC, domain, gr)
    dx, dy, dz = grid_spacing(domain, gr)
    u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)

    if parallel_fft
        Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    else
        Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
    end

    # compute max |U| over grid and reduce across ranks
    magmax2_local = 0.0
    @inbounds @simd for i in eachindex(Ux, Uy, Uz)
        mag2 = Ux[i]*Ux[i] + Uy[i]*Uy[i] + Uz[i]*Uz[i]
        magmax2_local = max(magmax2_local, mag2)
    end
    magmax_local = sqrt(magmax2_local)
    magmax = MPI.Allreduce(magmax_local, MPI.MAX, MPI.COMM_WORLD)

    return magmax
end

function max_grid_speed!(ws::VortexWorkspace{T},
                         eleGma::AbstractMatrix{T},
                         triXC::AbstractMatrix, triYC::AbstractMatrix, triZC::AbstractMatrix,
                         domain::DomainSpec, gr::GridSpec;
                         poisson_mode::Symbol=:spectral,
                         parallel_fft::Bool=false) where T<:AbstractFloat
    Ux, Uy, Uz = grid_velocity!(ws, eleGma, triXC, triYC, triZC, domain, gr;
                                poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    magmax2_local = zero(T)
    @inbounds @simd for i in eachindex(Ux, Uy, Uz)
        mag2 = Ux[i]*Ux[i] + Uy[i]*Uy[i] + Uz[i]*Uz[i]
        magmax2_local = max(magmax2_local, mag2)
    end
    magmax_local = sqrt(magmax2_local)
    return MPI.Allreduce(magmax_local, MPI.MAX, MPI.COMM_WORLD)
end

function rk2_step!(ws::VortexWorkspace{T},
                   nodeX::AbstractVector{T}, nodeY::AbstractVector{T}, nodeZ::AbstractVector{T},
                   tri, eleGma::AbstractMatrix{T},
                   domain::DomainSpec, gr::GridSpec, dt::Real;
                   At=zero(T), adaptive::Bool=false,
                   CFL::Real=T(0.5), poisson_mode::Symbol=:spectral,
                   parallel_fft::Bool=false) where T<:AbstractFloat
    dt_used = T(dt)

    triangle_coords_from_nodes!(ws.triXC, ws.triYC, ws.triZC, nodeX, nodeY, nodeZ, tri)
    ws.geom_dirty[] = true
    node_velocities!(ws, ws.u1, ws.v1, ws.w1, eleGma,
                     ws.triXC, ws.triYC, ws.triZC,
                     nodeX, nodeY, nodeZ, domain, gr;
                     poisson_mode=poisson_mode, parallel_fft=parallel_fft)
    node_circulation_from_ele_gamma!(ws.nodeΓ, ws.geom, eleGma)

    if adaptive
        dx, dy, _ = grid_spacing(domain, gr)
        magmax2_local = zero(T)
        @inbounds @simd for i in eachindex(ws.gridUx, ws.gridUy, ws.gridUz)
            mag2 = ws.gridUx[i]*ws.gridUx[i] + ws.gridUy[i]*ws.gridUy[i] + ws.gridUz[i]*ws.gridUz[i]
            magmax2_local = max(magmax2_local, mag2)
        end
        umax = MPI.Allreduce(sqrt(magmax2_local), MPI.MAX, MPI.COMM_WORLD)
        dt_used = T(CFL) * min(T(dx), T(dy)) / max(T(umax), T(1e-12))
    end

    @inbounds for i in eachindex(nodeX, nodeY, nodeZ, ws.u1, ws.v1, ws.w1)
        ws.xh[i] = mod(nodeX[i] + T(0.5) * dt_used * ws.u1[i], T(domain.Lx))
        ws.yh[i] = mod(nodeY[i] + T(0.5) * dt_used * ws.v1[i], T(domain.Ly))
        ws.zh[i] = mod(nodeZ[i] + T(domain.Lz) + T(0.5) * dt_used * ws.w1[i],
                       T(2 * domain.Lz)) - T(domain.Lz)
    end

    triangle_coords_from_nodes!(ws.triXC, ws.triYC, ws.triZC, ws.xh, ws.yh, ws.zh, tri)
    refresh_workspace_geometry!(ws, ws.triXC, ws.triYC, ws.triZC, domain, gr)
    if Circulation.has_baroclinicity(At)
        dGmid = baroclinic_ele_gamma(At, 0.5 * Float64(dt_used), ws.triXC, ws.triYC, ws.triZC; domain=domain)
        dTau = node_circulation_from_ele_gamma(ws.geom, dGmid)
        ws.nodeΓ .+= dTau
    end

    ele_gamma_from_node_circ!(ws.eleGma_mid, ws.geom, ws.nodeΓ)
    node_velocities!(ws, ws.u2, ws.v2, ws.w2, ws.eleGma_mid,
                     ws.triXC, ws.triYC, ws.triZC,
                     ws.xh, ws.yh, ws.zh, domain, gr;
                     poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    @inbounds for i in eachindex(nodeX, nodeY, nodeZ, ws.u2, ws.v2, ws.w2)
        nodeX[i] = mod(nodeX[i] + dt_used * ws.u2[i], T(domain.Lx))
        nodeY[i] = mod(nodeY[i] + dt_used * ws.v2[i], T(domain.Ly))
        nodeZ[i] = mod(nodeZ[i] + T(domain.Lz) + dt_used * ws.w2[i],
                       T(2 * domain.Lz)) - T(domain.Lz)
    end

    triangle_coords_from_nodes!(ws.triXC_new, ws.triYC_new, ws.triZC_new,
                                nodeX, nodeY, nodeZ, tri)
    refresh_workspace_geometry!(ws, ws.triXC_new, ws.triYC_new, ws.triZC_new, domain, gr)
    if Circulation.has_baroclinicity(At)
        dGend = baroclinic_ele_gamma(At, 0.5 * Float64(dt_used), ws.triXC_new, ws.triYC_new, ws.triZC_new; domain=domain)
        dTau2 = node_circulation_from_ele_gamma(ws.geom, dGend)
        ws.nodeΓ .+= dTau2
    end

    ele_gamma_from_node_circ!(ws.eleGma_new, ws.geom, ws.nodeΓ)
    eleGma .= ws.eleGma_new
    return Float64(dt_used)
end

function rk2_step!(nodeX, nodeY, nodeZ, tri, eleGma, 
                domain::DomainSpec, gr::GridSpec, dt::Float64;
                At=0.0, adaptive::Bool=false, 
                CFL::Float64=0.5, poisson_mode::Symbol=:spectral, 
                parallel_fft::Bool=false)

    # velocities at t^n
    triXC = similar(eleGma, size(tri,1), 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end

    # compute node Γ from current gamma
    nodeΓ = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma; domain=domain)

    # adaptive dt based on grid max speed if requested
    if adaptive
        dx,dy,_ = grid_spacing(domain, gr)
        umax = max_grid_speed(eleGma, triXC, triYC, triZC, domain, gr; poisson_mode=poisson_mode, parallel_fft=parallel_fft)
        dt = CFL * min(dx,dy) / max(umax, 1e-12)
    end
    u1, v1, w1 = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, domain, gr; 
                                poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    # half-step positions
    xh = nodeX .+ 0.5 .* dt .* u1
    yh = nodeY .+ 0.5 .* dt .* v1
    zh = nodeZ .+ 0.5 .* dt .* w1

    # periodic wrap
    xh .= mod.(xh, domain.Lx)
    yh .= mod.(yh, domain.Ly)
    zh .= mod.(zh .+ domain.Lz, 2*domain.Lz) .- domain.Lz

    # velocities at half step
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = xh[v]
        triYC[t,k] = yh[v]
        triZC[t,k] = zh[v]
    end

    # baroclinic update at half step
    if Circulation.has_baroclinicity(At)
        dGmid = baroclinic_ele_gamma(At, 0.5*dt, triXC, triYC, triZC; domain=domain)
        dTau = node_circulation_from_ele_gamma(triXC, triYC, triZC, dGmid; domain=domain)
        nodeΓ .+= dTau
    end

    # recompute gamma at half-step geometry from updated node Γ
    eleGma_mid = ele_gamma_from_node_circ(nodeΓ, triXC, triYC, triZC; domain=domain)
    u2, v2, w2 = node_velocities(eleGma_mid, triXC, triYC, triZC, xh, yh, zh, domain, gr; 
                                poisson_mode=poisson_mode, parallel_fft=parallel_fft)

    # full-step update
    nodeX .+= dt .* u2
    nodeY .+= dt .* v2
    nodeZ .+= dt .* w2
    nodeX .= mod.(nodeX, domain.Lx)
    nodeY .= mod.(nodeY, domain.Ly)
    nodeZ .= mod.(nodeZ .+ domain.Lz, 2*domain.Lz) .- domain.Lz

    # rebuild triangle coords at t^{n+1}
    triXC_new = similar(triXC); triYC_new = similar(triYC); triZC_new = similar(triZC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC_new[t,k] = nodeX[v]
        triYC_new[t,k] = nodeY[v]
        triZC_new[t,k] = nodeZ[v]
    end

    # second baroclinic update at end step
    if Circulation.has_baroclinicity(At)
        dGend = baroclinic_ele_gamma(At, 0.5*dt, triXC_new, triYC_new, triZC_new; domain=domain)
        dTau2 = node_circulation_from_ele_gamma(triXC_new, triYC_new, triZC_new, dGend; domain=domain)
        nodeΓ .+= dTau2
    end

    # produce gamma at new geometry from node Γ
    eleGma_new = ele_gamma_from_node_circ(nodeΓ, triXC_new, triYC_new, triZC_new; domain=domain)
    eleGma .= eleGma_new

    return dt
end

# Enhanced RK2 time stepping with dissipation models
function rk2_step_with_dissipation!(nodeX, nodeY, nodeZ, tri, eleGma, 
                                domain::DomainSpec, gr::GridSpec, dt::Float64,
                                dissipation_model::DissipationModel=NoDissipation();
                                At=0.0, 
                                adaptive::Bool=false, 
                                CFL::Float64=0.5, 
                                poisson_mode::Symbol=:spectral, 
                                parallel_fft::Bool=false, 
                                kernel::KernelType=PeskinStandard())

    # velocities at t^n
    triXC = similar(eleGma, size(tri,1), 3); triYC = similar(triXC); triZC = similar(triXC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = nodeX[v]
        triYC[t,k] = nodeY[v]
        triZC[t,k] = nodeZ[v]
    end
    
    # Apply dissipation at beginning of step
    eleGma = apply_dissipation!(dissipation_model, eleGma, triXC, triYC, triZC, domain, gr, 0.5*dt)

    # compute node Γ from current gamma
    nodeΓ = node_circulation_from_ele_gamma(triXC, triYC, triZC, eleGma; domain=domain)
    
    # adaptive dt based on grid max speed if requested
    if adaptive
        dx, dy, _ = grid_spacing(domain, gr)
        umax = max_grid_speed(eleGma, triXC, triYC, triZC, domain, gr; 
                            poisson_mode=poisson_mode, parallel_fft=parallel_fft)
        dt = CFL * min(dx, dy) / max(umax, 1e-12)
    end
    
    # Use kernel-based spreading if specified
    if kernel != PeskinStandard()
        ζx, ζy, ζz = spread_vorticity_to_grid_kernel_mpi(eleGma, triXC, triYC, triZC, domain, gr, kernel)
        dx, dy, dz = grid_spacing(domain, gr)
        u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)

        if parallel_fft
            Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
        else
            Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
        end
        u1, v1, w1 = interpolate_node_velocity_kernel_mpi(Ux, Uy, Uz, nodeX, nodeY, nodeZ, domain, gr, kernel)
    else
        u1, v1, w1 = node_velocities(eleGma, triXC, triYC, triZC, nodeX, nodeY, nodeZ, 
                                domain, gr; poisson_mode=poisson_mode, 
                                parallel_fft=parallel_fft)
    end

    # half-step positions
    xh = nodeX .+ 0.5 .* dt .* u1
    yh = nodeY .+ 0.5 .* dt .* v1
    zh = nodeZ .+ 0.5 .* dt .* w1

    # periodic wrap
    xh .= mod.(xh, domain.Lx)
    yh .= mod.(yh, domain.Ly)
    zh .= mod.(zh .+ domain.Lz, 2*domain.Lz) .- domain.Lz

    # velocities at half step
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC[t,k] = xh[v]
        triYC[t,k] = yh[v]
        triZC[t,k] = zh[v]
    end
    
    # baroclinic update at half step
    if Circulation.has_baroclinicity(At)
        dGmid = baroclinic_ele_gamma(At, 0.5*dt, triXC, triYC, triZC; domain=domain)
        dTau = node_circulation_from_ele_gamma(triXC, triYC, triZC, dGmid; domain=domain)
        nodeΓ .+= dTau
    end
    
    # recompute gamma at half-step geometry from updated node Γ
    eleGma_mid = ele_gamma_from_node_circ(nodeΓ, triXC, triYC, triZC; domain=domain)
    
    # Apply dissipation at mid-step
    eleGma_mid = apply_dissipation!(dissipation_model, eleGma_mid, triXC, triYC, triZC, domain, gr, 0.5*dt)
    
    if kernel != PeskinStandard()
        ζx, ζy, ζz = spread_vorticity_to_grid_kernel_mpi(eleGma_mid, triXC, triYC, triZC, domain, gr, kernel)
        dx,dy,dz = grid_spacing(domain, gr)
        u_rhs, v_rhs, w_rhs = curl_rhs_centered(ζx, ζy, ζz, dx, dy, dz)
        
        if parallel_fft
            Ux, Uy, Uz = poisson_velocity_pencil_fft(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
        else
            Ux, Uy, Uz = poisson_velocity_fft_mpi(u_rhs, v_rhs, w_rhs, domain; mode=poisson_mode)
        end
        u2, v2, w2 = interpolate_node_velocity_kernel_mpi(Ux, Uy, Uz, xh, yh, zh, domain, gr, kernel)
    else
        u2, v2, w2 = node_velocities(eleGma_mid, triXC, triYC, triZC, xh, yh, zh, 
                                    domain, gr; poisson_mode=poisson_mode, 
                                    parallel_fft=parallel_fft)
    end

    # full-step update
    nodeX .+= dt .* u2
    nodeY .+= dt .* v2
    nodeZ .+= dt .* w2
    nodeX .= mod.(nodeX, domain.Lx)
    nodeY .= mod.(nodeY, domain.Ly)
    nodeZ .= mod.(nodeZ .+ domain.Lz, 2*domain.Lz) .- domain.Lz

    # rebuild triangle coords at t^{n+1}
    triXC_new = similar(triXC); triYC_new = similar(triYC); triZC_new = similar(triZC)
    @inbounds for k in 1:3, t in 1:size(tri,1)
        v = tri[t,k]
        triXC_new[t,k] = nodeX[v]
        triYC_new[t,k] = nodeY[v]
        triZC_new[t,k] = nodeZ[v]
    end
    
    # second baroclinic update at end step
    if Circulation.has_baroclinicity(At)
        dGend = baroclinic_ele_gamma(At, 0.5*dt, triXC_new, triYC_new, triZC_new; domain=domain)
        dTau2 = node_circulation_from_ele_gamma(triXC_new, triYC_new, triZC_new, dGend; domain=domain)
        nodeΓ .+= dTau2
    end
    
    # produce gamma at new geometry from node Γ
    eleGma_new = ele_gamma_from_node_circ(nodeΓ, triXC_new, triYC_new, triZC_new; domain=domain)
    eleGma .= eleGma_new

    return dt
end

end # module

using .TimeStepper: node_velocities, node_velocities!, rk2_step!, rk2_step_with_dissipation!,
                    grid_velocity, grid_velocity!, make_velocity_sampler
